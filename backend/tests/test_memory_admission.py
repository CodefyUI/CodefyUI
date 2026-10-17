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
from app.core.sweep_store import SweepStore
from app.core.version import get_version
from app.core.worker_store import WorkerStore
from app.core.workers import Worker, WorkerRegistry
from app.main import app

GIB = mem.GIB
BASE = "http://127.0.0.1:8000"
WS_PATH = f"{wp.WORKER_WS_PATH}?{TOKEN_QUERY_PARAM}={session_token()}"
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


@pytest.fixture(autouse=True)
def _no_cuda_context(monkeypatch):
    """Budgets alone, so the arithmetic below is round. The CUDA context a
    real card adds is tested on its own (``context`` in the name)."""
    from app.config import settings

    monkeypatch.setattr(settings, "EXECUTION_CUDA_CONTEXT_MIB", 0)


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
    # MPS's held_bytes is what the driver held at the END, not a peak: a
    # budget from it could sit below what the run needs (review item 4).
    assert mem.peak_bytes({"mps": {"held_bytes": 5}}) is None
    assert mem.peak_bytes({"mps": {"held_bytes": 9 * GIB},
                           "cuda:0": {"reserved_bytes": GIB}}) == GIB
    assert mem.peak_bytes({"cuda:0": {"reserved_bytes": True}}) is None
    assert mem.peak_bytes(None) is None


async def test_the_history_source_suggests_the_last_peak_times_1_2():
    seen: list[str] = []

    async def lookup(digest: str) -> Any:
        seen.append(digest)
        return ("run-1", {"cuda:0": {"reserved_bytes": 10 * GIB}})

    graph = {"nodes": [{"id": "a", "type": "T"}], "edges": []}
    options = {"device": "cuda:0", "target": {"devices": None}}
    suggestion = await mem.HistoryBudgetSource(lookup).suggest(graph, options)
    assert seen == [mem.graph_hash(graph, options)]
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
        assert await mem.HistoryBudgetSource(lookup_).suggest(
            graph, options) is None


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
    with pytest.raises(RunSubmitError, match="'memory_gib', 'pool', 'wait'"):
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
    assert "queue it again with target.memory_gib" in scheduler.explain(
        _entry("r1"))


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


async def test_a_heartbeat_with_new_memory_figures_reschedules_a_shared_card():
    """Free memory is capacity on a shared card: a run waiting for it is
    offered again on the next heartbeat, not only when a run ends."""
    registry = WorkerRegistry()
    calls: list[int] = []
    worker, _ = registry.connect(
        wp.Hello(name="box", version="t", devices=[_card("cuda:0")]),
        send=_noop, close=_noop)
    registry.add_listener(lambda: calls.append(1))
    registry.heartbeat(worker, wp.Heartbeat(devices=[
        _card("cuda:0", free=14 * GIB)]))
    assert calls == []  # exclusive: memory figures are not capacity
    registry.set_device_policy("box", "cuda:0", "shared")
    registry.heartbeat(worker, wp.Heartbeat(devices=[
        _card("cuda:0", free=13 * GIB)]))
    registry.heartbeat(worker, wp.Heartbeat(devices=[
        _card("cuda:0", free=13 * GIB)]))  # unchanged: nothing
    assert calls == [1, 1]
    await registry.shutdown()


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
    app.state.sweep_store = SweepStore(database)
    try:
        yield service, registry
    finally:
        await service.shutdown()
        await registry.shutdown()
        database.close()
        for attribute in ("db", "run_service", "worker_registry",
                          "worker_store", "sweep_store"):
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
        assert facts["graph_hash"] == mem.graph_hash(
            _graph(), {"device": "cuda:0"})
        assert facts["peak_memory"]["cuda:0"]["reserved_bytes"] == 2 * GIB
        queued = await agent.assignment()
        assert queued.run_id == waiting.json()["run_id"]
        assert queued.memory_budgets == {"cuda:0": 6 * GIB}

        # The same graph again, no budget: the last peak times 1.2.
        on_one_card = {"device": "cuda:0",
                       "target": {"worker": "box", "devices": ["cuda:0"]}}
        estimate = (await client.post("/api/runs/estimate", json={
            "graph": _graph(), "options": on_one_card})).json()
        history = estimate.pop("history")
        assert estimate == {"suggested_gib": 2.4, "source": "history",
                            "peak_bytes": 2 * GIB,
                            "from_run": assigns[0].run_id,
                            # #794: a measured peak, and no model to count.
                            "confidence": "measured",
                            "static_estimate_bytes": None, "static": None}
        assert [row["run_id"] for row in history] == [assigns[0].run_id]
        assert history[0]["peak_bytes"] == 2 * GIB
        # Round 2, item 2: the peak is per card of a one-card run; the same
        # graph on two cards, or with no device at all, has no history.
        for options in ({"device": "cuda:0", "target": {
                "worker": "box", "devices": ["cuda:0", "cuda:1"]}}, None):
            assert (await client.post("/api/runs/estimate", json={
                "graph": _graph(), "options": options})).json()[
                "suggested_gib"] is None
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
            await service.check_batch_budgets([_graph()], {
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
    # #794: a run given its budget by default is told that one is what to
    # declare; a declared one keeps the #788 wording.
    defaulted = _with_budget_note(oom, budgeted, {
        "target": {"worker": "box"}}).error
    assert "capped to the memory budget it was given by default, 1.0 GiB " \
        "on cuda:0 (from its graph's last peak or a static estimate): " \
        "declare a larger target.memory_gib" in defaulted
    assert "(target.memory_gib): raise" in _with_budget_note(oom, budgeted, {
        "target": {"worker": "box", "memory_gib": 1}}).error


# ── #794: the static estimate as the default ─────────────────────────────


async def test_a_run_without_history_gets_the_static_estimate(env):
    """A run of a model graph that no run has measured, on a shared card
    and with no ``memory_gib``: its budget is the static estimate, recorded
    as ``source: "estimate"`` (never as a peak), and ``/api/runs/estimate``
    offers the same figure the run was given. Once a run of the graph
    reports its peak, the history wins."""
    from app.core import memory_estimate as me
    from tests.test_memory_estimate import _training_graph

    service, registry = env
    graph = _training_graph()
    static = me.estimate_static(graph)
    assert static is not None and static.confidence == "medium"
    async with _client() as client, aconnect_ws(WS_PATH, client) as ws:
        agent = FakeAgent(ws)
        await agent.hello()
        await client.put("/api/workers/box/devices/cuda:0/policy",
                         json={"policy": "shared"})
        options = {"device": "cuda:0", "target": {"worker": "box"}}
        estimate = (await client.post("/api/runs/estimate", json={
            "graph": graph, "options": {"device": "cuda:0"}})).json()
        assert estimate["source"] == "estimate"
        assert estimate["confidence"] == "medium"
        assert estimate["peak_bytes"] is None and estimate["from_run"] is None
        assert estimate["history"] == []
        assert estimate["static_estimate_bytes"] == static.bytes
        assert estimate["static"]["parameters"] == 212
        assert estimate["suggested_gib"] == round(static.bytes / GIB, 2)

        first = await client.post("/api/runs", json={
            "graph": graph, "options": options})
        assert first.json()["status"] == "running", first.text
        assign = await agent.assignment()
        assert assign.memory_budgets == {"cuda:0": static.bytes}
        await agent.send(wp.RunStarted(run_id=assign.run_id, seq=1, pid=1))
        await agent.finish(assign, peak=GIB // 4)
        record = await _terminal(service, assign.run_id)
        assert record.execution_facts["memory_budget"] == {
            "bytes": static.bytes, "source": "estimate",
            "devices": ["cuda:0"]}

        # The same graph, a new seed and more epochs: the same structural
        # hash, so its measured peak is the default now.
        again = _training_graph(seed=5, epochs=40)
        estimate = (await client.post("/api/runs/estimate", json={
            "graph": again, "options": {"device": "cuda:0"}})).json()
        assert (estimate["source"], estimate["confidence"],
                estimate["from_run"]) == ("history", "measured",
                                          assign.run_id)
        assert [row["peak_bytes"] for row in estimate["history"]] == [
            GIB // 4]
        # The static estimate is still reported beside it.
        assert estimate["static_estimate_bytes"] == static.bytes


async def test_a_low_estimate_is_never_a_default_budget(env):
    """A graph whose model cannot be counted: the run still needs a
    declared budget, as before #794."""
    from tests.test_memory_estimate import _node, _training_graph

    service, registry = env
    graph = _training_graph(extra=[_node("bt", "BradleyTerryTrain")])
    async with _client() as client, aconnect_ws(WS_PATH, client) as ws:
        agent = FakeAgent(ws)
        await agent.hello()
        await client.put("/api/workers/box/devices/cuda:0/policy",
                         json={"policy": "shared"})
        estimate = (await client.post("/api/runs/estimate", json={
            "graph": graph, "options": {"device": "cuda:0"}})).json()
        assert estimate["suggested_gib"] is None
        assert estimate["confidence"] == "low"
        assert estimate["static"]["uncounted"] == ["BradleyTerryTrain"]
        refused = await client.post("/api/runs", json={
            "graph": graph, "options": {"device": "cuda:0", "target": {
                "worker": "box"}}})
        assert refused.status_code == 400
        assert "has no memory budget" in refused.json()["detail"]


# ── code review of #837 ──────────────────────────────────────────────────


def test_a_card_held_by_a_run_without_a_budget_says_so():
    """Review item 3: the reason names the holder, not a fit that is not
    the problem (the card has room)."""
    scheduler = _sched(_worker("box", _card("cuda:0")))
    taken = {("box", "cuda:0"): 1}
    reason = scheduler.explain(_entry("r1", 6), taken)
    assert reason == (
        "waiting for memory: needs 6.0 GiB, cuda:0 on box is held by a run "
        "without a memory budget (a run on this server, or one started "
        "before the card was shared) until it ends")
    assert "free" not in reason
    assert scheduler.memory_refusal(_options(), 6 * GIB, taken,
                                    wait=False) == reason.removeprefix(
        "waiting for memory: ")
    # The holder has a budget: the room decides again.
    scheduler.ledger.reserve("r0", [("box", "cuda:0")], 12 * GIB)
    scheduler.ledger.started("r0")
    assert "4.0 GiB free" in scheduler.explain(_entry("r1", 6), taken)


def test_a_worker_that_is_away_is_never_refused_on_its_stale_figures():
    """Review item 6: an offline worker's last free figure is not now; a
    budget that fits its total waits, one larger than its total is still
    refused (it would wait forever)."""
    from app.core.workers import STATUS_OFFLINE

    worker = _worker("box", _card("cuda:0", free=GIB))
    scheduler = _sched(worker)
    refuse = scheduler.memory_refusal
    assert refuse(_options(), 6 * GIB, {}, wait=False).startswith(
        "needs 6.0 GiB, cuda:0 on box has 1.0 GiB free")
    worker.status = STATUS_OFFLINE
    assert refuse(_options(), 6 * GIB, {}, wait=False) is None
    assert refuse(_options(), 20 * GIB, {}, wait=False) == (
        "needs 20.0 GiB, cuda:0 on box has 16.0 GiB in total")


def test_places_is_the_one_enumeration_of_where_a_run_may_go():
    """Review item 10: one candidate list for wants_budget, the refusal and
    the waiting reason."""
    scheduler = _sched(_worker("box", _card("cuda:0"), _card("cuda:1")),
                       _worker("other", _card("cuda:0")),
                       shared=(("box", "cuda:0"), ("box", "cuda:1")))
    places, here = scheduler.places(_options(worker="any", device="cuda"))
    assert here is False
    assert [[(w.id, d) for w, d in place] for place in places] == [
        [("box", "cuda:0")], [("box", "cuda:1")], [("other", "cuda:0")]]
    places, _ = scheduler.places(_options(devices=["cuda:1", "cuda:0"]))
    assert [[(w.id, d) for w, d in place] for place in places] == [
        [("box", "cuda:1"), ("box", "cuda:0")]]
    assert scheduler.places({"device": "cpu"}) == ([], True)
    assert scheduler.places(_options(worker="nobody")) == ([], False)


def test_admission_adds_the_cuda_context_to_every_budget(monkeypatch):
    """The CUDA context sits outside the allocator cap: admission, the
    ledger and the messages count it (review, question 3)."""
    from app.config import settings

    monkeypatch.setattr(settings, "EXECUTION_CUDA_CONTEXT_MIB", 1024)
    assert mem.context_bytes("cuda:0") == GIB
    assert mem.context_bytes("mps") == 0 and mem.context_bytes("cpu") == 0
    assert mem.footprint(6 * GIB, "cuda:0") == 7 * GIB
    assert mem.footprint(6 * GIB, "mps") == 6 * GIB
    scheduler = _sched(_worker("box", _card("cuda:0")))
    taken: dict = {}
    for run_id in ("r1", "r2"):
        placement = scheduler.place(_entry(run_id, 6), taken)
        assert placement is not None
        taken[("box", "cuda:0")] = taken.get(("box", "cuda:0"), 0) + 1
        scheduler.ledger.reserve(run_id, placement.budget_slots,
                                 mem.footprint(6 * GIB, "cuda:0"))
        scheduler.ledger.started(run_id)
    # 16 - 2 x 7 = 2 GiB of room: a 1.5 GiB budget is 2.5 with its context.
    assert scheduler.place(_entry("r3", 1.5), taken) is None
    assert scheduler.place(_entry("r3", 1), taken) is not None
    assert scheduler.explain(_entry("r3", 1.5), taken) == (
        "waiting for memory: needs 1.5 GiB (+1.0 GiB CUDA context), cuda:0 "
        "on box has 2.0 GiB free (runs r1, r2)")
    assert scheduler.memory_refusal(_options(), 15.5 * GIB, {}, wait=True) \
        == ("needs 15.5 GiB (+1.0 GiB CUDA context), cuda:0 on box has "
            "16.0 GiB in total")
    admission = scheduler.admission(_worker("box", _card("cuda:0")),
                                    "cuda:0")
    assert admission["context_bytes"] == GIB


async def test_a_heartbeat_reschedules_only_when_a_budgeted_run_waits():
    """Review item 8: new memory figures on a shared card reschedule only
    when the run service says a run with a budget waits for that worker."""
    registry = WorkerRegistry()
    calls: list[int] = []
    waiting = {"box": False}
    registry.memory_waiting = lambda worker_id: waiting.get(worker_id, False)
    worker, _ = registry.connect(
        wp.Hello(name="box", version="t", devices=[_card("cuda:0")]),
        send=_noop, close=_noop)
    registry.set_device_policy("box", "cuda:0", "shared")
    registry.add_listener(lambda: calls.append(1))
    registry.heartbeat(worker, wp.Heartbeat(devices=[
        _card("cuda:0", free=14 * GIB)]))
    assert calls == []
    waiting["box"] = True
    registry.heartbeat(worker, wp.Heartbeat(devices=[
        _card("cuda:0", free=13 * GIB)]))
    assert calls == [1]
    await registry.shutdown()


async def test_the_run_service_answers_whether_a_budgeted_run_waits(env):
    service, registry = env
    assert registry.memory_waiting == service._memory_waiting
    assert service._memory_waiting("box") is False
    async with _client() as client, aconnect_ws(WS_PATH, client) as ws:
        agent = FakeAgent(ws)
        await agent.hello()
        await client.put("/api/workers/box/devices/cuda:0/policy",
                         json={"policy": "shared"})
        first = await client.post("/api/runs", json={
            "graph": _graph(), "options": {"device": "cuda:0", "target": {
                "worker": "box", "memory_gib": 12}}})
        assign = await agent.assignment()
        assert service._memory_waiting("box") is False
        waits = await client.post("/api/runs", json={
            "graph": _graph(), "options": {"device": "cuda:0", "target": {
                "worker": "box", "memory_gib": 12, "wait": True}}})
        assert waits.json()["status"] == "queued"
        assert service._memory_waiting("box") is True
        assert service._memory_waiting("other") is False
        await agent.send(wp.RunStarted(run_id=assign.run_id, seq=1, pid=1))
        await agent.finish(assign)
        await _terminal(service, first.json()["run_id"])
        second = await agent.assignment()
        await agent.send(wp.RunStarted(run_id=second.run_id, seq=1, pid=1))
        await agent.finish(second)
        await _terminal(service, second.run_id)


async def test_the_workers_listing_shows_the_schedulers_own_figures(env):
    """Review item 9: GET /api/workers reads WorkerScheduler.admission, so
    what the panel shows is what admission decides on."""
    service, registry = env
    async with _client() as client, aconnect_ws(WS_PATH, client) as ws:
        await FakeAgent(ws).hello()
        await client.put("/api/workers/box/devices/cuda:0/policy",
                         json={"policy": "shared"})
        listing = (await client.get("/api/workers")).json()
        assert _card_entry(listing)["admission"] == \
            service._scheduler.admission(registry.get("box"), "cuda:0")
        assert service.card_admission(registry.get("box"), "cuda:0")[
            "context_bytes"] == 0
    assert RunService(None).card_admission(None, "cuda:0") is None


class _LaterSource:
    """A budget source with nothing to say until *budget* is set."""

    def __init__(self) -> None:
        self.budget: int | None = None
        self.calls = 0

    async def suggest(self, _graph: Any,
                      _options: Any) -> mem.Suggestion | None:
        self.calls += 1
        if self.budget is None:
            return None
        return mem.Suggestion(mem.Budget(self.budget, "history", "x" * 32),
                              self.budget)


async def test_a_run_queued_while_its_card_was_exclusive_is_not_stranded(
        env):
    """Review item 2 (round 1): a run that waited for an exclusive card gets
    its graph's default when the card is set to shared, so it is placed once
    the card admits by memory. Round 2 item 7: no history lookup at submit
    while no shared card could take the run."""
    service, registry = env
    source = service._budget_source = _LaterSource()
    async with _client() as client, aconnect_ws(WS_PATH, client) as ws:
        agent = FakeAgent(ws)
        await agent.hello()

        def submit(value: float) -> Any:
            return client.post("/api/runs", json={
                "graph": _graph(value), "options": {
                    "device": "cuda:0", "target": {"worker": "box"}}})

        await submit(0.0)  # exclusive: by count, no budget
        holder_assign = await agent.assignment()
        assert holder_assign.memory_budgets == {}
        source.budget = 2 * GIB
        early = await submit(1.0)
        assert early.json()["status"] == "queued"
        assert source.calls == 0  # exclusive: not looked up
        assert service._pending_by_id[early.json()["run_id"]] \
            .memory_budget is None

        await client.put("/api/workers/box/devices/cuda:0/policy",
                         json={"policy": "shared"})
        assert service._pending_by_id[early.json()["run_id"]] \
            .memory_budget.bytes == 2 * GIB
        # The holder has no budget: the run waits for it, and says so.
        assert "held by a run without a memory budget" in \
            service.queue_reason(early.json()["run_id"])
        await agent.send(wp.RunStarted(run_id=holder_assign.run_id, seq=1,
                                       pid=1))
        await agent.finish(holder_assign)
        placed = await agent.assignment()
        assert placed.memory_budgets == {"cuda:0": 2 * GIB}
        await agent.send(wp.RunStarted(run_id=placed.run_id, seq=1, pid=1))
        await agent.finish(placed)
        await _terminal(service, placed.run_id)


async def test_a_waiting_run_takes_the_peak_a_run_of_its_graph_reports(env):
    """Review round 2, item 5: a run waiting on a shared card with no budget
    (its graph had no history) gets one the moment a run of the same graph on
    the same devices succeeds with a peak; until then its reason says how to
    get it going."""
    service, registry = env
    async with _client() as client, aconnect_ws(WS_PATH, client) as ws:
        agent = FakeAgent(ws)
        await agent.hello()

        def submit() -> Any:
            return client.post("/api/runs", json={
                "graph": _graph(), "options": {
                    "device": "cuda:0", "target": {"worker": "box"}}})

        first = await submit()  # exclusive: by count
        holder = await agent.assignment()
        waiting = (await submit()).json()["run_id"]
        await client.put("/api/workers/box/devices/cuda:0/policy",
                         json={"policy": "shared"})
        assert service._pending_by_id[waiting].memory_budget is None
        reason = service.queue_reason(waiting)
        assert "cancel it and queue it again with target.memory_gib" in reason
        await agent.send(wp.RunStarted(run_id=holder.run_id, seq=1, pid=1))
        await agent.finish(holder, peak=2 * GIB)
        assert (await _terminal(service, first.json()["run_id"])).status \
            == "succeeded"
        placed = await agent.assignment()
        assert placed.run_id == waiting
        assert placed.memory_budgets == {"cuda:0": mem.gib_to_bytes(2 * 1.2)}
        await agent.send(wp.RunStarted(run_id=placed.run_id, seq=1, pid=1))
        await agent.finish(placed)
        record = await _terminal(service, placed.run_id)
        assert record.execution_facts["memory_budget"]["source"] == "history"


async def test_refresh_budgets_skips_a_gone_run_and_a_local_one(env):
    service, _ = env
    source = service._budget_source = _LaterSource()
    source.budget = GIB

    @dataclass
    class _Pending:
        run_id: str
        options: dict
        needs: Any = field(default_factory=dict)
        memory_budget: Any = None
        graph_hash: str | None = None

    service._scheduler.wants_budget = lambda _options: True
    gone = _Pending("g", {"target": {"worker": "box"}})
    local = _Pending("l", {}, needs=None)
    service._pending_by_id.update(g=gone, l=local)
    try:
        await service.refresh_budgets()
        assert service._budgetless("other") == []
    finally:
        for run_id in ("g", "l"):
            service._pending_by_id.pop(run_id, None)
        del service._scheduler.wants_budget
    assert gone.memory_budget is None  # no row: nothing to read
    assert local.memory_budget is None
    assert RunService(None)._budgetless() == []


async def test_a_batch_without_a_budget_is_refused_before_any_child(env):
    """Review item 1: a sweep whose children declare no memory_gib and can
    only go to shared cards is refused whole, with no row, rather than part
    way through its children; each child's default is derived first, and
    handed to its submit (round 2, item 7)."""
    service, _ = env
    async with _client() as client, aconnect_ws(WS_PATH, client) as ws:
        agent = FakeAgent(ws)
        await agent.hello()
        await client.put("/api/workers/box/devices/cuda:0/policy",
                         json={"policy": "shared"})
        options = {"device": "cuda:0", "target": {"worker": "box"}}
        sweep = {
            "base_graph": _graph(),
            "sweep_spec": {"method": "grid", "params": [
                {"node_id": "x", "param": "value", "values": [1.0, 2.0]}]},
            "objective": {"metric": "val_loss", "direction": "minimize"},
            "options": options}
        response = await client.post("/api/sweeps", json=sweep)
        assert response.status_code == 400, response.text
        assert "has no memory budget" in response.json()["detail"]
        assert await service.store.list_runs() == []
        with pytest.raises(RunSubmitError, match="has no memory budget"):
            await service.check_batch_budgets([_graph()], options)
        # A derived default too large for the card is refused as well.
        source = service._budget_source = _LaterSource()
        source.budget = 20 * GIB
        with pytest.raises(RunSubmitError, match="16.0 GiB in total"):
            await service.check_batch_budgets([_graph(), _graph()], options)
        source.budget = 2 * GIB
        assert await service.check_batch_budgets(
            [_graph(), _graph(1.0)], options) == [
            mem.Budget(2 * GIB, "history", "x" * 32)] * 2
        # Declared, or a card that counts: nothing to derive.
        assert await service.check_batch_budgets([_graph()], {
            "device": "cuda:0",
            "target": {"worker": "box", "memory_gib": 2}}) == [None]
        assert await service.check_batch_budgets(
            [_graph()], {"device": "cpu"}) == [None]
        await agent.nothing()

        # The sweep goes through, its children take the derived budgets
        # without a second lookup each.
        response = await client.post("/api/sweeps", json=sweep)
        assert response.status_code == 201, response.text
        calls = source.calls
        first = await agent.assignment()
        assert first.memory_budgets == {"cuda:0": 2 * GIB}
        assert source.calls == calls
        await agent.send(wp.RunStarted(run_id=first.run_id, seq=1, pid=1))
        second = await agent.assignment()
        assert second.memory_budgets == {"cuda:0": 2 * GIB}
        for assign in (first, second):
            if assign is second:
                await agent.send(wp.RunStarted(run_id=assign.run_id, seq=1,
                                               pid=1))
            await agent.finish(assign)
            await _terminal(service, assign.run_id)
    assert await RunService(None).check_batch_budgets([_graph()], {}) \
        == [None]


async def test_a_batch_child_that_may_not_wait_is_refused_up_front(env):
    """Review round 2, item 1: the pre-check judges each child with its own
    ``target.wait``, so ``wait: false`` refuses the whole batch when a child
    does not fit now, instead of half of it at submit."""
    service, _ = env
    async with _client() as client, aconnect_ws(WS_PATH, client) as ws:
        agent = FakeAgent(ws)
        await agent.hello()
        await client.put("/api/workers/box/devices/cuda:0/policy",
                         json={"policy": "shared"})
        holder = await client.post("/api/runs", json={
            "graph": _graph(), "options": {"device": "cuda:0", "target": {
                "worker": "box", "memory_gib": 12}}})
        assign = await agent.assignment()
        body = {
            "base_graph": _graph(),
            "sweep_spec": {"method": "grid", "params": [
                {"node_id": "x", "param": "value", "values": [1.0, 2.0]}]},
            "objective": {"metric": "val_loss", "direction": "minimize"},
            "options": {"device": "cuda:0", "target": {
                "worker": "box", "memory_gib": 6, "wait": False}}}
        response = await client.post("/api/sweeps", json=body)
        assert response.status_code == 400, response.text
        assert response.json()["detail"].startswith("needs 6.0 GiB")
        assert [r.id for r in await service.store.list_runs()] == [
            holder.json()["run_id"]]
        await agent.send(wp.RunStarted(run_id=assign.run_id, seq=1, pid=1))
        await agent.finish(assign)
        await _terminal(service, assign.run_id)


async def test_the_last_peak_is_found_by_an_indexed_column(tmp_path):
    """Review item 7: ``exec_runs.graph_hash`` (MIGRATION_015) and its index
    answer last_peak_memory, newest first, without a json scan."""
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    try:
        store = RunStore(database)
        plan = database._conn.execute(
            "EXPLAIN QUERY PLAN SELECT id FROM exec_runs "
            "INDEXED BY idx_exec_runs_graph_hash WHERE graph_hash = ? "
            "AND status = ? ORDER BY finished_at DESC LIMIT 20",
            ("h", "succeeded")).fetchall()
        assert any("idx_exec_runs_graph_hash" in str(tuple(row))
                   for row in plan)
        assert await store.last_peak_memory("h") is None

        async def finished(peak: Any, graph_hash: str = "h") -> str:
            from app.core.run_store import RunProvenance

            record = await store.create_run(
                graph_snapshot=_graph(), provenance=RunProvenance(),
                worker_id="box", graph_hash=graph_hash)
            facts = {} if peak is None else {"peak_memory": peak}
            await store.mark_running(record.id, execution_facts=facts)
            await store.mark_finished(record.id, "succeeded")
            return record.id

        older = await finished({"cuda:0": {"reserved_bytes": GIB}})
        await finished({"cuda:0": {"reserved_bytes": 9 * GIB}}, "other")
        await finished(None)  # newest, but no peak: the older one answers
        assert await store.last_peak_memory("h") == (
            older, {"cuda:0": {"reserved_bytes": GIB}})
    finally:
        database.close()


def test_the_one_shot_pool_is_judged_on_its_one_shot_workers_only():
    """#790 with #788: a pool run's candidates are the one-shot workers
    (never this process, never a regular worker), and ``any``'s are every
    other worker, as ``place`` picks them."""
    one_shot = _worker("job-1", _card("cuda:0", total=8 * GIB))
    one_shot.one_shot = True
    scheduler = _sched(_worker("box", _card("cuda:0")), one_shot,
                       shared=(("box", "cuda:0"), ("job-1", "cuda:0")),
                       here=("cpu", "cuda:0"))
    pool = _options(worker="any", device="cuda", pool="one-shot")
    places, here = scheduler.places(pool)
    assert here is False
    assert [[(w.id, d) for w, d in p] for p in places] == [
        [("job-1", "cuda:0")]]
    places, here = scheduler.places(_options(worker="any", device="cuda"))
    assert here is True and [p[0][0].id for p in places] == ["box"]
    assert "has no memory budget" in scheduler.memory_refusal(
        pool, None, {}, wait=True)
    assert scheduler.memory_refusal(pool, 12 * GIB, {}, wait=True) == (
        "needs 12.0 GiB, cuda:0 on job-1 has 8.0 GiB in total")
    entry = _Entry("r1", "any/cuda", pool, mem.Budget(6 * GIB))
    placement = scheduler.place(entry, {})
    assert placement.worker_id == "job-1"
    assert placement.memory_budget == 6 * GIB


def test_the_history_key_counts_the_devices_a_run_uses():
    """Review round 2, item 2: a per-card peak of a run on two cards is not
    a budget for the same graph on one; MPS history is not CUDA's."""
    graph = {"nodes": [{"id": "a", "type": "T"}], "edges": []}
    assert mem.run_shape({"device": "cuda:1"}) == "cuda x1"
    assert mem.run_shape({"device": "cuda", "target": {
        "devices": ["cuda:2", "cuda:0"]}}) == "cuda x2"
    assert mem.run_shape({"device": "mps"}) == "mps x1"
    assert mem.run_shape(None) == "cpu x1"
    assert mem.run_shape({"device": None, "target": None}) == "cpu x1"
    one = mem.graph_hash(graph, {"device": "cuda:0"})
    # Which card does not matter, how many and of what kind does.
    assert one == mem.graph_hash(graph, {"device": "cuda:3", "target": {
        "devices": ["cuda:3"]}})
    assert one != mem.graph_hash(graph, {"device": "cuda:0", "target": {
        "devices": ["cuda:0", "cuda:1"]}})
    assert one != mem.graph_hash(graph, {"device": "mps"})


def test_a_card_reserved_for_a_waiting_set_says_so():
    """Review round 2, item 6: a slot the admission view marks reserved (an
    earlier run waiting for its whole device set) is named as such, not as
    a holder without a budget."""
    from app.core.scheduling import RESERVED_SLOT

    scheduler = _sched(_worker("box", _card("cuda:0")))
    reason = scheduler.explain(_entry("r1", 6),
                               {("box", "cuda:0"): RESERVED_SLOT})
    assert reason == (
        "waiting for memory: needs 6.0 GiB, cuda:0 on box is reserved for "
        "an earlier run that waits for its whole device set")
    assert scheduler.place(_entry("r1", 6),
                           {("box", "cuda:0"): RESERVED_SLOT}) is None


async def test_an_adopted_run_takes_its_budget_back_in_the_ledger(env):
    """#793 with #788: a worker run adopted after a restart still holds its
    card under its cap, so the restarted server's ledger counts its budget
    as running at once and does not admit onto the card what it uses."""
    from app.core.run_service import WORKER_RESUME_KEY
    from app.core.run_store import RunProvenance

    service, _ = env
    store = service.store
    record = await store.create_run(
        graph_snapshot=_graph(), provenance=RunProvenance(),
        options={"device": "cuda:0", "target": {
            "worker": "box", "devices": ["cuda:0"]}},
        worker_id="box", devices=["cuda:0"], graph_hash="h")
    await store.mark_running(record.id, execution_facts={
        "graph_hash": "h",
        "memory_budget": {"bytes": 6 * GIB, "source": "history",
                          "from_run": "r0", "devices": ["cuda:0"]},
        WORKER_RESUME_KEY: {"last_seq": 1, "slots": [["box", "cuda:0"]]}})
    [(row, placement, last_seq)] = await service._adoptable_worker_runs()
    assert placement.memory_budget == 6 * GIB
    assert placement.budget_devices == ("cuda:0",)
    assert placement.budget_slots == (("box", "cuda:0"),)
    service._adopt(row, placement, last_seq)
    account = service._ledger.account(("box", "cuda:0"))
    assert (account.reserved, account.running, account.runs) == (
        0, 6 * GIB, [record.id])
    active = service._runs[record.id]
    assert active.memory_budget == mem.Budget(6 * GIB, "history", "r0")
    assert active.graph_hash == "h"


def test_an_adopted_placement_without_a_budget_admits_by_count():
    from app.core.run_service import _adopted_budget, _adopted_placement

    slots = (("box", "cuda:0"),)
    assert _adopted_placement("box", ("cuda:0",), slots, {}) == Placement(
        "box", ("cuda:0",), slots)
    for broken in ({"bytes": 0}, {"bytes": True}, "x",
                   {"bytes": GIB, "devices": ["cuda:9"]}):
        assert _adopted_placement("box", ("cuda:0",), slots, {
            "memory_budget": broken}).memory_budget is None
    assert _adopted_budget({"memory_budget": {"bytes": GIB, "source": 3}}) \
        == mem.Budget(GIB)
