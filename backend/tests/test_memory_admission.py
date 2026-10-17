"""Memory-based admission for shared GPUs (#788).

Unit tests of ``core.memory_admission`` and ``WorkerScheduler``'s memory
rule, then the acceptance flow end to end over ``/ws/worker`` with a fake
agent (it speaks ``core.worker_protocol`` by hand, like
``test_target_any_sweeps_studies.py``): two 6 GiB runs share a 16 GiB card,
a third is refused with the message, and the budgets, the ledger and the
history default show where the Workers and Runs panels read them.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import pytest
from httpx import AsyncClient
from httpx_ws import aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport

from app.core import memory_admission as mem
from app.core import worker_protocol as wp
from app.core.auth import TOKEN_HEADER, TOKEN_QUERY_PARAM, session_token
from app.core.db import Database
from app.core.run_service import (
    QueueLimits,
    RunService,
    RunSubmitError,
    normalize_options,
)
from app.core.run_store import RunStore
from app.core.scheduling import Placement, WorkerScheduler
from app.core.version import get_version
from app.core.worker_store import WorkerStore
from app.core.workers import Worker, WorkerRegistry
from app.main import app

GIB = mem.GIB
BASE = "http://127.0.0.1:8000"
WS_PATH = f"{wp.WORKER_WS_PATH}?{TOKEN_QUERY_PARAM}={session_token()}"
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


# ── memory_admission ─────────────────────────────────────────────────────


def test_the_ledger_moves_a_budget_from_reserved_to_running():
    ledger = mem.MemoryLedger()
    slot = ("box", "cuda:0")
    ledger.reserve("a", [slot], 6 * GIB)
    ledger.reserve("b", [slot, ("box", "cuda:1")], 2 * GIB)
    assert ledger.started("a") is True
    assert ledger.started("nobody") is False
    account = ledger.account(slot)
    assert (account.reserved, account.running) == (2 * GIB, 6 * GIB)
    assert account.runs == ["a", "b"] and ledger.holders(slot) == 2
    assert ledger.budget_of("b") == 2 * GIB and ledger.holds("a")
    ledger.release("a")
    ledger.release("a")  # idempotent
    assert ledger.account(slot).running == 0 and ledger.budget_of("a") is None


def test_headroom_keeps_promises_within_the_card_and_free_memory():
    account = mem.SlotAccount(reserved=2 * GIB, running=6 * GIB)
    # The promises: 16 - 2 - 6 = 8; what is free now: 15 - 2 = 13.
    assert mem.headroom(16 * GIB, 15 * GIB, account) == 8 * GIB
    # Someone else's job holds most of the card: free decides.
    assert mem.headroom(16 * GIB, 3 * GIB, account) == 1 * GIB
    assert mem.headroom(16 * GIB, 1 * GIB, account) == 0
    assert mem.headroom(16 * GIB, None, account) == 8 * GIB
    assert mem.headroom(None, 15 * GIB, account) is None


def test_graph_hash_ignores_layout_and_settings_but_not_params():
    graph = {"nodes": [
        {"id": "a", "type": "T", "position": {"x": 1}, "data": {
            "params": {"n": 1}, "label": "A"}},
        {"id": "b", "type": "U", "data": {"params": {}}}],
        "edges": [{"source": "a", "target": "b", "sourceHandle": "o",
                   "targetHandle": "i", "id": "e1"}],
        "settings": {"seed": 1}}
    moved = {**graph, "settings": {"seed": 2}, "nodes": [
        {"id": "b", "type": "U", "data": {"params": {}}},
        {"id": "a", "type": "T", "position": {"x": 9}, "data": {
            "params": {"n": 1}}}],
        "edges": [{"source": "a", "target": "b", "sourceHandle": "o",
                   "targetHandle": "i", "id": "renamed"}]}
    assert mem.graph_hash(graph) == mem.graph_hash(moved)
    other = {**graph, "nodes": [{"id": "a", "type": "T", "data": {
        "params": {"n": 2}}}, graph["nodes"][1]]}
    assert mem.graph_hash(other) != mem.graph_hash(graph)


def test_peak_bytes_reads_the_largest_device_figure():
    assert mem.peak_bytes({
        "cuda:0": {"allocated_bytes": 1, "reserved_bytes": 3 * GIB},
        "cuda:1": {"allocated_bytes": 4 * GIB},
        "cuda:2": "junk"}) == 4 * GIB
    assert mem.peak_bytes({"mps": {"held_bytes": 5}}) == 5
    assert mem.peak_bytes({"cuda:0": {"reserved_bytes": True}}) is None
    assert mem.peak_bytes(None) is None


async def test_the_history_source_suggests_the_last_peak_times_1_2():
    seen: list[str] = []

    async def lookup(digest: str) -> Any:
        seen.append(digest)
        return ("run-1", {"cuda:0": {"reserved_bytes": 10 * GIB}})

    graph = {"nodes": [{"id": "a", "type": "T"}], "edges": []}
    suggestion = await mem.HistoryBudgetSource(lookup).suggest(graph)
    assert seen == [mem.graph_hash(graph)]
    assert suggestion.budget == mem.Budget(12 * GIB, mem.SOURCE_HISTORY,
                                           from_run="run-1")
    assert suggestion.peak_bytes == 10 * GIB
    assert suggestion.budget.describe() == {
        "bytes": 12 * GIB, "source": "history", "from_run": "run-1"}

    async def nothing(_digest: str) -> Any:
        return None

    async def no_peak(_digest: str) -> Any:
        return ("run-2", {"cuda:0": {}})

    async def broken(_digest: str) -> Any:
        raise OSError("disk")

    for lookup_ in (nothing, no_peak, broken):
        assert await mem.HistoryBudgetSource(lookup_).suggest(graph) is None


def test_messages_name_the_card_its_room_and_its_runs():
    assert mem.format_gib(6.3 * GIB) == "6.3 GiB"
    assert mem.format_gib(1) == "0.1 GiB"
    assert mem.format_gib(None) == "? GiB"
    assert mem.needs_message(12 * GIB, [mem.does_not_fit(
        "cuda:1", "gpu-box-1", int(6.3 * GIB),
        ["a1b2c3d4e5", "c3d4e5f6a7"])]) == (
        "needs 12.0 GiB, cuda:1 on gpu-box-1 has 6.3 GiB free "
        "(runs a1b2c3d4, c3d4e5f6)")
    assert "no memory total" in mem.does_not_fit("cuda:0", "b", None, [])


# ── target options ───────────────────────────────────────────────────────


def test_target_memory_gib_and_wait_are_validated_and_kept_only_when_given():
    plain = normalize_options({"target": {"worker": "box"}})
    assert plain["target"] == {"worker": "box", "devices": None}
    given = normalize_options({"target": {"worker": "box", "memory_gib": 6,
                                          "wait": True}})
    assert given["target"] == {"worker": "box", "devices": None,
                               "memory_gib": 6.0, "wait": True}
    for bad in (0, -1, True, "6", float("nan"), 5000):
        with pytest.raises(RunSubmitError, match="memory_gib must be"):
            normalize_options({"target": {"worker": "box",
                                          "memory_gib": bad}})
    with pytest.raises(RunSubmitError, match="wait must be"):
        normalize_options({"target": {"worker": "any", "wait": "no"}})
    with pytest.raises(RunSubmitError, match="needs a worker"):
        normalize_options({"target": {"memory_gib": 2}})
    with pytest.raises(RunSubmitError, match="'memory_gib', 'wait'"):
        normalize_options({"target": {"worker": "box", "budget": 1}})


# ── WorkerScheduler ──────────────────────────────────────────────────────


@dataclass
class _Entry:
    run_id: str
    queue_key: str
    options: dict[str, Any] = field(default_factory=dict)
    memory_budget: Any = None


async def _noop(*_: Any) -> None:
    return None


def _card(value: str, total: int | None = 16 * GIB,
          free: int | None = 15 * GIB) -> wp.DeviceReport:
    return wp.DeviceReport(value=value, kind=value.split(":")[0],
                           total_bytes=total, free_bytes=free, source="nvml")


def _worker(name: str, *devices: wp.DeviceReport,
            capable: bool = True) -> Worker:
    return Worker(
        id=name, name=name, version="t", fingerprint=None, platform={},
        mounts={}, devices=list(devices), send=_noop, close=_noop,
        heartbeat_s=10,
        capabilities=(wp.CAPABILITY_MEMORY_BUDGET,) if capable else ())


class _Registry:
    def __init__(self, *workers: Worker, shared: tuple = ()) -> None:
        self._workers = {w.id: w for w in workers}
        self.shared = set(shared)

    def get(self, worker_id: str) -> Worker | None:
        return self._workers.get(worker_id)

    def workers(self) -> list[Worker]:
        return sorted(self._workers.values(), key=lambda w: w.name)

    def device_policy(self, worker_id: str, device: str) -> str:
        return (mem.POLICY_SHARED if (worker_id, device) in self.shared
                else mem.POLICY_EXCLUSIVE)


def _options(worker: str = "box", device: str = "cuda:0",
             devices: list[str] | None = None, **target: Any) -> dict:
    return {"device": device,
            "target": {"worker": worker, "devices": devices, **target}}


def _entry(run_id: str, budget_gib: float | None = None, **kw: Any) -> _Entry:
    return _Entry(run_id, "box/cuda:0", _options(**kw),
                  None if budget_gib is None
                  else mem.Budget(mem.gib_to_bytes(budget_gib)))


def _sched(*workers: Worker, shared: tuple = (("box", "cuda:0"),),
           here: tuple = ("cpu",)) -> WorkerScheduler:
    return WorkerScheduler(QueueLimits(), _Registry(*workers, shared=shared),
                           lambda: list(here))


def test_two_budgets_share_a_card_and_a_third_does_not_fit():
    scheduler = _sched(_worker("box", _card("cuda:0")))
    taken: dict = {}
    for run_id in ("r1", "r2"):
        placement = scheduler.place(_entry(run_id, 6), taken)
        assert placement == Placement(
            "box", ("cuda:0",), (("box", "cuda:0"),),
            memory_budget=6 * GIB, budget_devices=("cuda:0",),
            budget_slots=(("box", "cuda:0"),))
        taken[("box", "cuda:0")] = taken.get(("box", "cuda:0"), 0) + 1
        scheduler.ledger.reserve(run_id, placement.budget_slots, 6 * GIB)
    # Reserved, not started: free (15) minus the promises (12) decides.
    assert scheduler.place(_entry("r3", 4), taken) is None
    scheduler.ledger.started("r1")
    scheduler.ledger.started("r2")
    assert scheduler.place(_entry("r3", 6), taken) is None
    assert scheduler.place(_entry("r3", 4), taken) is not None
    reason = scheduler.explain(_entry("r3", 6))
    assert reason == ("waiting for memory: needs 6.0 GiB, cuda:0 on box has "
                      "4.0 GiB free (runs r1, r2)")


def test_a_run_without_a_budget_never_takes_a_shared_card():
    scheduler = _sched(_worker("box", _card("cuda:0")))
    assert scheduler.place(_entry("r1"), {}) is None
    assert "declare target.memory_gib" in scheduler.explain(_entry("r1"))


def test_a_holder_without_a_budget_keeps_the_card_to_itself():
    """A run on the slot that holds no budget (a same-host card's in-process
    run, a run from before the card became shared): budgeted runs wait."""
    scheduler = _sched(_worker("box", _card("cuda:0")))
    assert scheduler.place(_entry("r1", 1), {("box", "cuda:0"): 1}) is None
    scheduler.ledger.reserve("r0", [("box", "cuda:0")], GIB)
    assert scheduler.place(_entry("r1", 1), {("box", "cuda:0"): 1}) is not None


def test_an_exclusive_card_or_an_agent_without_the_capability_counts():
    scheduler = _sched(_worker("box", _card("cuda:0")), shared=())
    placement = scheduler.place(_entry("r1", 6), {})
    assert placement == Placement("box", ("cuda:0",), (("box", "cuda:0"),))
    assert scheduler.place(_entry("r2", 6), {("box", "cuda:0"): 1}) is None
    old = _sched(_worker("box", _card("cuda:0"), capable=False))
    assert old.shared(old._registry.get("box"), "cuda:0") is False
    assert old.place(_entry("r1"), {}) is not None
    assert old.shared(old._registry.get("box"), "nonsense") is False


def test_a_card_without_a_total_cannot_admit_by_memory():
    scheduler = _sched(_worker("box", _card("cuda:0", total=None,
                                            free=None)))
    assert scheduler.place(_entry("r1", 1), {}) is None


def test_a_bare_kind_takes_the_first_card_that_fits():
    scheduler = _sched(
        _worker("box", _card("cuda:0", free=2 * GIB), _card("cuda:1")),
        shared=(("box", "cuda:0"), ("box", "cuda:1")))
    placement = scheduler.place(_entry("r1", 6, device="cuda"), {})
    assert placement.devices == ("cuda:1",)


def test_a_set_admits_each_shared_card_by_memory_and_the_rest_by_count():
    scheduler = _sched(_worker("box", _card("cuda:0"), _card("cuda:1")))
    entry = _entry("r1", 6, devices=["cuda:0", "cuda:1"])
    placement = scheduler.place(entry, {})
    assert placement.devices == ("cuda:0", "cuda:1")
    assert placement.budget_devices == ("cuda:0",)
    assert scheduler.place(entry, {("box", "cuda:1"): 1}) is None


def test_memory_refusal_at_submit():
    scheduler = _sched(_worker("box", _card("cuda:0")))
    refuse = scheduler.memory_refusal
    assert refuse(_options(), 6 * GIB, {}, wait=False) is None
    assert "declare target.memory_gib" in refuse(_options(), None, {},
                                                 wait=True)
    assert refuse(_options(), 20 * GIB, {}, wait=True) == (
        "needs 20.0 GiB, cuda:0 on box has 16.0 GiB in total")
    scheduler.ledger.reserve("aaaaaaaa11", [("box", "cuda:0")], 12 * GIB)
    scheduler.ledger.started("aaaaaaaa11")
    taken = {("box", "cuda:0"): 1}
    assert refuse(_options(), 6 * GIB, taken, wait=False) == (
        "needs 6.0 GiB, cuda:0 on box has 4.0 GiB free (runs aaaaaaaa)")
    assert refuse(_options(), 6 * GIB, taken, wait=True) is None
    # Local runs, and runs that may go somewhere admitted by count, are
    # never judged here.
    assert refuse({"device": "cpu"}, None, {}, wait=False) is None
    assert refuse(_options(worker="nobody"), None, {}, wait=False) is None
    assert refuse(_options(device="cpu"), None, {}, wait=False) is None


def test_memory_refusal_for_any():
    scheduler = _sched(_worker("box", _card("cuda:0")),
                       _worker("tiny", _card("cuda:0", total=4 * GIB)),
                       shared=(("box", "cuda:0"), ("tiny", "cuda:0")))
    refuse = scheduler.memory_refusal
    any_cuda = _options(worker="any", device="cuda")
    assert refuse(any_cuda, 6 * GIB, {}, wait=True) is None
    assert "every device this run may use" in refuse(any_cuda, None, {},
                                                     wait=True)
    assert refuse(any_cuda, 32 * GIB, {}, wait=True) == (
        "needs 32.0 GiB, cuda:0 on box has 16.0 GiB in total; cuda:0 on "
        "tiny has 4.0 GiB in total")
    scheduler.ledger.reserve("x", [("box", "cuda:0")], 12 * GIB)
    scheduler.ledger.started("x")
    taken = {("box", "cuda:0"): 1}
    assert refuse(any_cuda, 6 * GIB, taken, wait=True) is None
    assert refuse(any_cuda, 6 * GIB, taken, wait=False).startswith(
        "needs 6.0 GiB, cuda:0 on box has 4.0 GiB free (runs x)")
    # This process offers the device: it takes the run by count.
    here = _sched(_worker("box", _card("cuda:0")), here=("cpu", "cuda:0"))
    assert here.memory_refusal(any_cuda, None, {}, wait=False) is None
    # A device set on one worker is judged as one place.
    sets = _sched(_worker("box", _card("cuda:0"), _card("cuda:1")),
                  shared=(("box", "cuda:0"), ("box", "cuda:1")))
    assert sets.memory_refusal(
        _options(devices=["cuda:0", "cuda:1"]), 20 * GIB, {},
        wait=True).count("in total") == 2


def test_wants_budget_only_where_a_shared_card_could_take_the_run():
    scheduler = _sched(_worker("box", _card("cuda:0")),
                       _worker("other", _card("cuda:0")))
    assert scheduler.wants_budget(_options()) is True
    assert scheduler.wants_budget(_options(worker="other")) is False
    assert scheduler.wants_budget(_options(worker="any",
                                           device="cuda")) is True
    assert scheduler.wants_budget(_options(worker="nobody")) is False
    assert scheduler.wants_budget({"device": "cuda:0"}) is False
    no_policies = WorkerScheduler(QueueLimits(), _NoPolicies(), list)
    assert no_policies.shared(_worker("box", _card("cuda:0")),
                              "cuda:0") is False


class _NoPolicies:
    def get(self, _worker_id: str) -> None:
        return None

    def workers(self) -> list:
        return []


def test_the_registry_keeps_policies_by_name():
    registry = WorkerRegistry()
    calls: list[int] = []
    registry.add_listener(lambda: calls.append(1))
    registry.load_policies([("box", "CUDA:00", "shared"),
                            ("box", "tpu:0", "shared"),
                            ("box", "cuda:1", "bogus")])
    assert registry.device_policy("box", "cuda:0") == mem.POLICY_SHARED
    assert registry.device_policy("box", "cuda:1") == mem.POLICY_EXCLUSIVE
    assert registry.device_policy("box", "nonsense") == mem.POLICY_EXCLUSIVE
    registry.set_device_policy("box", "cuda:1", "shared")
    registry.set_device_policy("box", "cuda:0", "exclusive")
    assert registry.device_policy("box", "cuda:0") == mem.POLICY_EXCLUSIVE
    assert calls == [1, 1]
    registry.forget_policies("box")
    assert registry.device_policy("box", "cuda:1") == mem.POLICY_EXCLUSIVE


async def test_the_store_keeps_policies_and_revoking_drops_them(tmp_path):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    try:
        store = WorkerStore(database)
        await store.set_device_policy("box", "cuda:0", "shared")
        await store.set_device_policy("box", "cuda:0", "shared")
        await store.set_device_policy("box", "cuda:1", "shared")
        await store.set_device_policy("box", "cuda:1", "exclusive")
        await store.set_device_policy("tiny", "cuda:0", "shared")
        assert await store.list_device_policies() == [
            ("box", "cuda:0", "shared"), ("tiny", "cuda:0", "shared")]
        await store.delete_worker("box")
        assert await store.list_device_policies() == [
            ("tiny", "cuda:0", "shared")]
    finally:
        database.close()


# ── end to end over /ws/worker ───────────────────────────────────────────


def _graph(value: float = 0.0) -> dict[str, Any]:
    return {"nodes": [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "x", "type": "TensorCreate",
         "data": {"params": {"shape": "2", "fill": "full", "value": value}}}],
        "edges": [{"id": "t", "source": "start", "target": "x",
                   "sourceHandle": "trigger", "type": "trigger"}]}


@pytest.fixture
async def env(tmp_path):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    registry = WorkerRegistry(heartbeat_s=10)
    store = RunStore(database)
    service = RunService(
        store, workers=registry, shutdown_grace_s=5.0,
        limits=QueueLimits(cpu=2, gpu=1, interactive=2, overrides=()))
    app.state.db = database
    app.state.run_service = service
    app.state.worker_registry = registry
    app.state.worker_store = WorkerStore(database)
    try:
        yield service, registry
    finally:
        await service.shutdown()
        await registry.shutdown()
        database.close()
        for attribute in ("db", "run_service", "worker_registry",
                          "worker_store"):
            if hasattr(app.state, attribute):
                delattr(app.state, attribute)


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGIWebSocketTransport(app=app),
                       base_url=BASE, headers={TOKEN_HEADER: session_token()})


class FakeAgent:
    def __init__(self, ws: Any) -> None:
        self.ws = ws

    async def send(self, frame: Any) -> None:
        await self.ws.send_text(wp.encode_frame(frame))

    async def recv(self, timeout: float = 10.0) -> Any:
        while True:
            raw = await asyncio.wait_for(self.ws.receive_text(), timeout)
            frame = wp.decode_control_frame(raw)
            if not isinstance(frame, wp.Ack):
                return frame

    async def assignment(self, timeout: float = 10.0) -> wp.Assign:
        frame = await self.recv(timeout)
        assert isinstance(frame, wp.Assign), frame
        return frame

    async def nothing(self, timeout: float = 0.3) -> None:
        with pytest.raises(asyncio.TimeoutError):
            await self.recv(timeout)

    async def hello(self) -> Any:
        await self.send(wp.Hello(
            name="box", version=get_version(),
            capabilities=list(wp.CAPABILITIES),
            devices=[_card("cpu", None, None), _card("cuda:0")]))
        return await self.recv()

    async def finish(self, assign: wp.Assign, seq: int = 2,
                     peak: int | None = None) -> None:
        await self.send(wp.RunFinished(
            run_id=assign.run_id, seq=seq, status="succeeded",
            peak_memory=None if peak is None else {"cuda:0": {
                "allocated_bytes": peak, "reserved_bytes": peak,
                "exact": True}}))


async def _terminal(service: RunService, run_id: str) -> Any:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        record = await service.store.get_run(run_id)
        if record is not None and record.status in TERMINAL \
                and run_id not in service.active_run_ids():
            return record
        await asyncio.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish")


def _card_entry(listing: dict, value: str = "cuda:0") -> dict:
    worker = next(w for w in listing["workers"] if w["id"] == "box")
    return next(d for d in worker["devices"] if d["value"] == value)


async def test_two_6_gib_runs_share_a_16_gib_card_and_a_third_is_refused(
        env):
    service, registry = env
    async with _client() as client, aconnect_ws(WS_PATH, client) as ws:
        agent = FakeAgent(ws)
        assert isinstance(await agent.hello(), wp.Welcome)
        listing = (await client.get("/api/workers")).json()
        assert _card_entry(listing)["policy"] == "exclusive"
        assert _card_entry(listing)["admission"] is None
        assert listing["workers"][0]["memory_budget"] is True

        response = await client.put("/api/workers/box/devices/cuda:0/policy",
                                    json={"policy": "shared"})
        assert response.status_code == 200, response.text
        card = next(d for d in response.json()["devices"]
                    if d["value"] == "cuda:0")
        assert card["policy"] == "shared"
        assert card["admission"]["by_memory"] is True
        assert card["admission"]["headroom_bytes"] == 15 * GIB

        def submit(**target: Any):
            return client.post("/api/runs", json={
                "graph": _graph(), "options": {
                    "device": "cuda:0",
                    "target": {"worker": "box", **target}}})

        first = await submit(memory_gib=6)
        second = await submit(memory_gib=6)
        assert first.json()["status"] == second.json()["status"] == "running"
        assigns = [await agent.assignment(), await agent.assignment()]
        assert [a.memory_budgets for a in assigns] == [
            {"cuda:0": 6 * GIB}, {"cuda:0": 6 * GIB}]
        admission = _card_entry((await client.get("/api/workers")).json()
                                )["admission"]
        assert (admission["reserved_bytes"], admission["running_bytes"]) \
            == (12 * GIB, 0)
        for assign in assigns:
            await agent.send(wp.RunStarted(run_id=assign.run_id, seq=1,
                                           pid=1))

        async def running() -> dict:
            return _card_entry((await client.get("/api/workers")).json()
                               )["admission"]

        deadline = time.monotonic() + 10
        while (await running())["running_bytes"] != 12 * GIB:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.02)

        third = await submit(memory_gib=6)
        assert third.status_code == 400
        assert third.json()["detail"] == (
            f"needs 6.0 GiB, cuda:0 on box has 4.0 GiB free "
            f"(runs {assigns[0].run_id[:8]}, {assigns[1].run_id[:8]})")
        assert (await submit()).json()["detail"].startswith(
            "cuda:0 on box is shared by memory and this run has no memory "
            "budget")
        # ``any`` waits for room; ``wait: false`` refuses instead.
        waiting = await client.post("/api/runs", json={
            "graph": _graph(), "options": {"device": "cuda:0", "target": {
                "worker": "any", "memory_gib": 6}}})
        assert waiting.json()["status"] == "queued"
        reason = service.queue_reason(waiting.json()["run_id"])
        assert reason is None or "memory" in reason
        refused = await client.post("/api/runs", json={
            "graph": _graph(), "options": {"device": "cuda:0", "target": {
                "worker": "any", "memory_gib": 6, "wait": False}}})
        assert refused.status_code == 400
        assert refused.json()["detail"].startswith("needs 6.0 GiB")

        # One run ends with its peak: the waiting ``any`` run takes its room.
        await agent.finish(assigns[0], peak=2 * GIB)
        record = await _terminal(service, assigns[0].run_id)
        facts = record.execution_facts
        assert facts["memory_budget"] == {"bytes": 6 * GIB,
                                          "source": "declared",
                                          "devices": ["cuda:0"]}
        assert facts["graph_hash"] == mem.graph_hash(_graph())
        assert facts["peak_memory"]["cuda:0"]["reserved_bytes"] == 2 * GIB
        queued = await agent.assignment()
        assert queued.run_id == waiting.json()["run_id"]
        assert queued.memory_budgets == {"cuda:0": 6 * GIB}

        # The same graph again, no budget: the last peak times 1.2.
        estimate = (await client.post("/api/runs/estimate",
                                      json={"graph": _graph()})).json()
        assert estimate == {"suggested_gib": 2.4, "source": "history",
                            "peak_bytes": 2 * GIB,
                            "from_run": assigns[0].run_id}
        again = await submit()
        assert again.json()["status"] == "running"
        budgeted = await agent.assignment()
        assert budgeted.memory_budgets == {
            "cuda:0": mem.gib_to_bytes(2 * 1.2)}
        await agent.send(wp.RunStarted(run_id=budgeted.run_id, seq=1, pid=1))
        # Another graph has no history.
        assert (await client.post("/api/runs/estimate", json={
            "graph": _graph(1.0)})).json()["suggested_gib"] is None
        assert (await client.post("/api/runs/estimate", json={
            "graph": {"nodes": []}})).status_code == 400

        # Back to exclusive: stored, and the card counts again.
        response = await client.put("/api/workers/box/devices/cuda:0/policy",
                                    json={"policy": "exclusive"})
        assert _card_entry({"workers": [response.json()]})["policy"] \
            == "exclusive"
        assert await app.state.worker_store.list_device_policies() == []
        for assign in (assigns[1], queued, budgeted):
            await agent.finish(assign)
            await _terminal(service, assign.run_id)


async def test_a_named_run_may_wait_and_a_never_fitting_batch_is_refused(
        env):
    service, registry = env
    async with _client() as client, aconnect_ws(WS_PATH, client) as ws:
        agent = FakeAgent(ws)
        await agent.hello()
        await client.put("/api/workers/box/devices/cuda:0/policy",
                         json={"policy": "shared"})
        first = await client.post("/api/runs", json={
            "graph": _graph(), "options": {"device": "cuda:0", "target": {
                "worker": "box", "memory_gib": 12}}})
        assert first.json()["status"] == "running"
        assign = await agent.assignment()
        waits = await client.post("/api/runs", json={
            "graph": _graph(), "options": {"device": "cuda:0", "target": {
                "worker": "box", "memory_gib": 12, "wait": True}}})
        assert waits.json()["status"] == "queued"
        assert service.queue_reason(waits.json()["run_id"]).startswith(
            "waiting for memory: needs 12.0 GiB, cuda:0 on box has 3.0 GiB")
        await agent.nothing()
        with pytest.raises(RunSubmitError, match="16.0 GiB in total"):
            service.check_placeable_batch([_graph()], {
                "device": "cuda:0",
                "target": {"worker": "box", "memory_gib": 20}})
        assert service.check_placeable_batch([_graph()], {
            "device": "cuda:0",
            "target": {"worker": "box", "memory_gib": 2}}) == [
            service._needs_of(_graph())]
        await agent.send(wp.RunStarted(run_id=assign.run_id, seq=1, pid=1))
        await agent.send(wp.RunFinished(
            run_id=assign.run_id, seq=2, status="failed",
            error="Node py ran out of memory on cuda:0."))
        record = await _terminal(service, assign.run_id)
        assert record.error.endswith(
            "This run was capped to its memory budget of 12.0 GiB on cuda:0 "
            "(target.memory_gib): raise the budget, or make the run need "
            "less.")
        second = await agent.assignment()
        assert second.run_id == waits.json()["run_id"]
        await agent.send(wp.RunStarted(run_id=second.run_id, seq=1, pid=1))
        await agent.finish(second)
        await _terminal(service, second.run_id)


async def test_the_policy_route_refuses_what_it_cannot_share(env):
    async with _client() as client, aconnect_ws(WS_PATH, client) as ws:
        await FakeAgent(ws).hello()
        put = client.put
        assert (await put("/api/workers/nobody/devices/cuda:0/policy",
                          json={"policy": "shared"})).status_code == 404
        assert (await put("/api/workers/box/devices/cuda:0/policy",
                          json={"policy": "half"})).status_code == 422
        assert (await put("/api/workers/box/devices/tpu:0/policy",
                          json={"policy": "shared"})).status_code == 422
        assert (await put("/api/workers/box/devices/cuda:3/policy",
                          json={"policy": "shared"})).status_code == 404
        assert (await put("/api/workers/box/devices/cpu/policy",
                          json={"policy": "shared"})).status_code == 422


def test_the_budget_note_is_added_to_an_out_of_memory_failure_only():
    from app.core.executors import ExecutionOutcome, _with_budget_note

    budgeted = Placement("box", ("cuda:0",), (("box", "cuda:0"),),
                         memory_budget=GIB, budget_devices=("cuda:0",))
    plain = Placement("box", ("cuda:0",), (("box", "cuda:0"),))
    oom = ExecutionOutcome("failed", "CUDA out of memory")
    assert "capped to its memory budget of 1.0 GiB" in _with_budget_note(
        oom, budgeted).error
    for outcome, placement in (
            (oom, plain), (ExecutionOutcome("failed", "boom"), budgeted),
            (ExecutionOutcome("succeeded"), budgeted),
            (ExecutionOutcome("failed", None), budgeted)):
        assert _with_budget_note(outcome, placement) is outcome
