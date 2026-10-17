"""One physical card offered by several workers counts once (#842).

Two agents on one host (an SSH worker and an enrolled agent through
``ssh -R``, say) both offer the same card. Each is its own ``(worker,
device)`` slot, so before #842 two ``any`` runs at a limit of 1 ran on the
card at once. The registry now groups the slots of one card by the UUID its
offerers report (``WorkerRegistry.card_slots``), and the scheduler counts the
runs of every slot of a card, under the smallest cap, with one memory
ledger. Unit tests first, then the bug's flow end to end over
``/ws/worker`` with two fake agents.
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
from app.core.run_service import QueueLimits, RunService, parse_queue_overrides
from app.core.run_store import RunStore
from app.core.scheduling import CardView, Placement, WorkerScheduler
from app.core.sweep_store import SweepStore
from app.core.version import get_version
from app.core.worker_store import WorkerStore
from app.core.workers import WorkerRegistry
from app.main import app

GIB = mem.GIB
UUID = "11111111-2222-3333-4444-555555555555"
BASE = "http://127.0.0.1:8000"
WS_PATH = f"{wp.WORKER_WS_PATH}?{TOKEN_QUERY_PARAM}={session_token()}"
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


@pytest.fixture(autouse=True)
def _no_cuda_context(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "EXECUTION_CUDA_CONTEXT_MIB", 0)


async def _noop(*_: Any) -> None:
    return None


class _Fact:
    def __init__(self, value: str, uuid: str | None = None) -> None:
        self.value, self.uuid = value, uuid


def _card(value: str, uuid: str | None, total: int | None = 16 * GIB,
          free: int | None = 15 * GIB) -> wp.DeviceReport:
    return wp.DeviceReport(value=value, kind=value.split(":")[0], uuid=uuid,
                           total_bytes=total, free_bytes=free,
                           source="none" if total is None else "nvml")


def _hello(name: str, *devices: wp.DeviceReport) -> wp.Hello:
    return wp.Hello(name=name, version="t",
                    capabilities=list(wp.CAPABILITIES),
                    devices=[_card("cpu", None, None, None), *devices])


def _registry(*hellos: wp.Hello, local: list | None = None,
              same_host: tuple[str, ...] = ()) -> WorkerRegistry:
    registry = WorkerRegistry(local_devices=lambda: list(local or []))
    for hello in hellos:
        registry.connect(hello, send=_noop, close=_noop,
                         same_host=hello.name in same_host)
    return registry


@dataclass
class _Entry:
    run_id: str
    queue_key: str
    options: dict[str, Any] = field(default_factory=dict)
    memory_budget: Any = None


def _any(run_id: str, device: str = "cuda") -> _Entry:
    return _Entry(run_id, f"any/{device}", {
        "device": device, "target": {"worker": "any", "devices": None}})


def _on(worker: str, device: str = "cuda:0",
        budget_gib: float | None = None) -> _Entry:
    return _Entry(f"{worker}-{device}", f"{worker}/{device}", {
        "device": device, "target": {"worker": worker, "devices": None}},
        memory_budget=None if budget_gib is None
        else mem.Budget(int(budget_gib * GIB)))


def _hold(taken: dict, placement: Placement) -> None:
    for slot in placement.slots:
        taken[slot] = taken.get(slot, 0) + 1


# ── the registry's card index ────────────────────────────────────────────


def test_two_workers_offering_one_uuid_are_one_card_whatever_the_numbers():
    """Matched by UUID, so another ``CUDA_VISIBLE_DEVICES`` (or a SLURM
    allocation) on one agent, which numbers the card ``cuda:1`` there,
    still names the same card. A card without a UUID, or with a UUID
    nobody else offers, keeps its own slot."""
    registry = _registry(
        _hello("meow-e2e", _card("cuda:0", UUID), _card("cuda:1", "other")),
        _hello("meow-join", _card("cuda:1", UUID.upper()),
               _card("cuda:0", None)))
    assert registry.card_slots(("meow-e2e", "cuda:0")) == (
        ("meow-e2e", "cuda:0"), ("meow-join", "cuda:1"))
    assert registry.card_slots(("meow-join", "cuda:1")) == (
        ("meow-e2e", "cuda:0"), ("meow-join", "cuda:1"))
    for alone in (("meow-e2e", "cuda:1"), ("meow-join", "cuda:0"),
                  ("meow-e2e", "cpu"), ("nobody", "cuda:0")):
        assert registry.card_slots(alone) == (alone,)
    assert registry.card_peers("meow-e2e", "cuda:0") == (
        ("meow-join", "cuda:1"),)
    assert registry.card_peers("meow-join", "cuda:1") == (
        ("meow-e2e", "cuda:0"),)
    assert registry.card_peers("meow-e2e", "cuda:1") == ()
    assert registry.card_peers("meow-e2e", "not a device") == ()


def test_this_processs_card_joins_and_a_shared_alias_adds_no_slot():
    """This process's card is an offerer too. A same-host worker whose card
    is ALREADY this process's slot (``Worker.shared``, #773) adds a peer but
    no second slot; a worker on the same card that was not recognised as
    same-host (another machine id) adds its own slot to the card."""
    registry = _registry(
        _hello("near", _card("cuda:0", UUID)),
        _hello("boxed", _card("cuda:0", UUID)),
        local=[_Fact("cpu"), _Fact("cuda:0", UUID)], same_host=("near",))
    near = registry.get("near")
    assert near.shared["cuda:0"] == "cuda:0"
    assert registry.card_slots(("local", "cuda:0")) == (
        ("local", "cuda:0"), ("boxed", "cuda:0"))
    assert registry.card_peers("local", "cuda:0") == (
        ("boxed", "cuda:0"), ("near", "cuda:0"))
    assert registry.card_peers("near", "cuda:0") == (
        ("local", "cuda:0"), ("boxed", "cuda:0"))
    # Two same-host workers aliased onto this process's card: one slot.
    aliased = _registry(_hello("a", _card("cuda:0", UUID)),
                        _hello("b", _card("cuda:0", UUID)),
                        local=[_Fact("cuda:0", UUID)], same_host=("a", "b"))
    assert aliased.card_slots(("local", "cuda:0")) == (("local", "cuda:0"),)
    assert len(aliased.card_peers("a", "cuda:0")) == 2


async def test_the_index_follows_connects_heartbeats_and_removals(caplog):
    registry = _registry(_hello("a", _card("cuda:0", UUID)))
    try:
        assert registry.card_slots(("a", "cuda:0")) == (("a", "cuda:0"),)
        with caplog.at_level("INFO", logger="app.core.workers"):
            b, _ = registry.connect(_hello("b", _card("cuda:0", None)),
                                    send=_noop, close=_noop)
            assert registry.card_slots(("a", "cuda:0")) == (("a", "cuda:0"),)
            # Its inventory reads the UUID now: a change of identity.
            registry.heartbeat(b, wp.Heartbeat(devices=[
                _card("cpu", None, None, None), _card("cuda:0", UUID)]))
            assert registry.card_slots(("a", "cuda:0")) == (
                ("a", "cuda:0"), ("b", "cuda:0"))
            # A heartbeat with new memory figures only: logged once.
            registry.heartbeat(b, wp.Heartbeat(devices=[
                _card("cpu", None, None, None),
                _card("cuda:0", UUID, free=3 * GIB)]))
            assert registry.card_slots(("b", "cuda:0")) == (
                ("a", "cuda:0"), ("b", "cuda:0"))
        logged = [r.getMessage() for r in caplog.records
                  if "one physical card" in r.getMessage()]
        assert logged == [f"one physical card (uuid {UUID}) is offered as "
                          "a/cuda:0, b/cuda:0; runs on it share one slot"]
        # A disconnected worker keeps its devices (its runs keep their
        # slots while they wait, #783), so its card stays grouped.
        registry.disconnect(b, "test")
        assert registry.card_slots(("a", "cuda:0")) == (
            ("a", "cuda:0"), ("b", "cuda:0"))
        registry.forget("b")
        assert registry.card_slots(("a", "cuda:0")) == (("a", "cuda:0"),)
    finally:
        await registry.shutdown()


def test_this_processs_devices_are_not_probed_until_a_worker_names_a_card():
    """A server without workers (or whose workers report no UUID) never
    lists its own devices for this (#773 probed them lazily, on the first
    same-host worker; the card index keeps that)."""
    calls = []

    def local_devices():
        calls.append(1)
        return [_Fact("cuda:0", UUID)]

    registry = WorkerRegistry(local_devices=local_devices)
    assert registry.card_slots(("local", "cuda:0")) == (("local", "cuda:0"),)
    registry.connect(_hello("old", _card("cuda:0", None)), send=_noop,
                     close=_noop)
    assert registry.card_slots(("old", "cuda:0")) == (("old", "cuda:0"),)
    assert calls == []
    registry.connect(_hello("new", _card("cuda:0", UUID)), send=_noop,
                     close=_noop)
    assert registry.card_slots(("new", "cuda:0")) == (
        ("local", "cuda:0"), ("new", "cuda:0"))
    assert calls == [1]


# ── placement per card ───────────────────────────────────────────────────


def _two_on_one_card(limits: QueueLimits = QueueLimits(),
                     local: tuple = ("cpu",)) -> WorkerScheduler:
    registry = _registry(_hello("meow-e2e", _card("cuda:0", UUID)),
                         _hello("meow-join", _card("cuda:0", UUID)))
    return WorkerScheduler(limits, registry, lambda: list(local))


def test_two_any_runs_at_limit_1_never_share_the_card():
    """The bug: the second ``any`` run went to the other agent's slot of
    the same card. Now it waits, and starts once the first is done."""
    scheduler = _two_on_one_card()
    taken: dict = {}
    first = scheduler.place(_any("r1"), taken)
    assert first == Placement("meow-e2e", ("cuda:0",),
                              (("meow-e2e", "cuda:0"),))
    _hold(taken, first)
    assert scheduler.place(_any("r2"), taken) is None
    # Naming the other agent does not get round it either.
    assert scheduler.place(_on("meow-join"), taken) is None
    taken[("meow-e2e", "cuda:0")] = 0
    assert scheduler.place(_any("r2"), taken) is not None
    # The busy-card wait is just a turn, not a reason.
    assert scheduler.explain(_on("meow-join"),
                             {("meow-e2e", "cuda:0"): 1}) is None


def test_a_card_takes_the_smallest_cap_among_its_offerers():
    raised = QueueLimits(overrides=parse_queue_overrides("meow-e2e/cuda:0=2"))
    scheduler = _two_on_one_card(raised)
    taken = {("meow-join", "cuda:0"): 1}
    assert scheduler.place(_on("meow-e2e"), taken) is None
    both = QueueLimits(overrides=parse_queue_overrides(
        "meow-e2e/cuda:0=2,meow-join/cuda:0=2"))
    assert _two_on_one_card(both).place(_on("meow-e2e"), taken) is not None


def test_a_local_run_counts_a_worker_run_on_the_same_card():
    """This process's card and a worker's card with its UUID that was not
    aliased (not recognised as same-host): one card for the in-process lane
    too."""
    registry = _registry(_hello("boxed", _card("cuda:0", UUID)),
                         local=[_Fact("cuda:0", UUID)])
    scheduler = WorkerScheduler(QueueLimits(), registry,
                                lambda: ["cpu", "cuda:0"])
    local = _Entry("l", "cuda:0", {"device": "cuda:0"})
    assert scheduler.place(local, {("boxed", "cuda:0"): 1}) is None
    assert scheduler.place(_on("boxed"), {("local", "cuda:0"): 1}) is None
    assert scheduler.place(local, {}) == Placement(
        "local", ("cuda:0",), (("local", "cuda:0"),))


def test_without_grouping_taken_and_caps_read_as_before():
    """A registry without ``card_slots`` (or one whose grouping raises)
    leaves every slot its own card."""

    class _Plain:
        def __init__(self, registry: WorkerRegistry) -> None:
            self.get, self.workers = registry.get, registry.workers

    registry = _registry(_hello("meow-e2e", _card("cuda:0", UUID)),
                         _hello("meow-join", _card("cuda:0", UUID)))
    plain = WorkerScheduler(QueueLimits(), _Plain(registry), lambda: [])
    assert plain.place(_on("meow-join"), {("meow-e2e", "cuda:0"): 1}) \
        is not None

    def broken(_slot):
        raise RuntimeError("boom")

    registry.card_slots = broken
    failing = WorkerScheduler(QueueLimits(), registry, lambda: [])
    assert failing.card_slots(("x", "cuda:0")) == (("x", "cuda:0"),)


def test_the_card_view_sums_a_cards_slots_and_is_not_wrapped_twice():
    cards = {("a", "cuda:0"): (("a", "cuda:0"), ("b", "cuda:0")),
             ("b", "cuda:0"): (("a", "cuda:0"), ("b", "cuda:0"))}
    view = CardView({("a", "cuda:0"): 1, ("b", "cuda:0"): 2, ("c", "x"): 5},
                    lambda slot: cards.get(slot, (slot,)))
    assert view[("a", "cuda:0")] == view.get(("b", "cuda:0")) == 3
    assert view.get(("c", "x")) == 5 and view.get(("d", "x"), 0) == 0
    assert len(view) == 3 and set(view) == {("a", "cuda:0"), ("b", "cuda:0"),
                                            ("c", "x")}
    scheduler = _two_on_one_card()
    assert scheduler._on_cards(view) is view


# ── memory admission per card (#788) ─────────────────────────────────────


class _Policies:
    def __init__(self, registry: WorkerRegistry, *shared: str) -> None:
        self._registry, self._shared = registry, set(shared)

    def get(self, worker_id):
        return self._registry.get(worker_id)

    def workers(self):
        return self._registry.workers()

    def card_slots(self, slot):
        return self._registry.card_slots(slot)

    def device_policy(self, worker_id, device):
        return (mem.POLICY_SHARED if worker_id in self._shared
                else mem.POLICY_EXCLUSIVE)


def _shared_card(*shared: str) -> WorkerScheduler:
    registry = _registry(_hello("meow-e2e", _card("cuda:0", UUID)),
                         _hello("meow-join", _card("cuda:0", UUID)))
    return WorkerScheduler(QueueLimits(), _Policies(registry, *shared),
                           lambda: [])


def test_one_ledger_per_card_across_its_offerers():
    """Shared by memory on both agents: a budget promised through one
    agent's slot leaves less room through the other's."""
    scheduler = _shared_card("meow-e2e", "meow-join")
    taken: dict = {}
    first = scheduler.place(_on("meow-e2e", budget_gib=8), taken)
    assert first is not None and first.budget_slots == (
        ("meow-e2e", "cuda:0"),)
    scheduler.ledger.reserve("r1", first.budget_slots, 8 * GIB)
    _hold(taken, first)
    # 16 total - 8 promised = 8 left on the card, through either agent.
    assert scheduler.room(scheduler._registry.get("meow-join"),
                          "cuda:0")[1] == 7 * GIB  # free 15 - 8 reserved
    assert scheduler.place(_on("meow-join", budget_gib=8), taken) is None
    assert scheduler.place(_on("meow-join", budget_gib=6), taken) is not None
    admission = scheduler.admission(scheduler._registry.get("meow-join"),
                                    "cuda:0")
    assert admission["reserved_bytes"] == 8 * GIB
    assert admission["runs"] == ["r1"]


def test_a_count_run_on_one_offerer_holds_the_card_from_budgeted_runs():
    """Shared through one agent, exclusive through the other: a run without
    a budget there has the card to itself, and the other way round."""
    scheduler = _shared_card("meow-e2e")
    held = {("meow-join", "cuda:0"): 1}
    assert scheduler.place(_on("meow-e2e", budget_gib=1), held) is None
    _, now, why = scheduler.card_state(scheduler._registry.get("meow-e2e"),
                                       "cuda:0", held, GIB)
    assert not now and why == mem.held_without_budget("cuda:0", "meow-e2e")
    budgeted = scheduler.place(_on("meow-e2e", budget_gib=1), {})
    scheduler.ledger.reserve("r1", budgeted.budget_slots, GIB)
    taken: dict = {}
    _hold(taken, budgeted)
    assert scheduler.place(_on("meow-join"), taken) is None


# ── end to end over /ws/worker ───────────────────────────────────────────


def _graph() -> dict[str, Any]:
    return {"nodes": [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "x", "type": "TensorCreate",
         "data": {"params": {"shape": "2", "fill": "full", "value": 0.0}}}],
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


class _Agent:
    def __init__(self, ws: Any, name: str) -> None:
        self.ws, self.name = ws, name

    async def send(self, frame: Any) -> None:
        await self.ws.send_text(wp.encode_frame(frame))

    async def recv(self, timeout: float = 10.0) -> Any:
        while True:
            raw = await asyncio.wait_for(self.ws.receive_text(), timeout)
            frame = wp.decode_control_frame(raw)
            if not isinstance(frame, wp.Ack):
                return frame

    async def hello(self) -> Any:
        await self.send(wp.Hello(
            name=self.name, version=get_version(),
            capabilities=list(wp.CAPABILITIES),
            devices=[_card("cpu", None, None, None), _card("cuda:0", UUID)]))
        return await self.recv()


async def _assigned(*agents: _Agent, timeout: float = 10.0
                    ) -> tuple[_Agent, wp.Assign]:
    """The one agent that gets an ``assign`` within *timeout*."""
    tasks = {asyncio.ensure_future(a.recv(timeout)): a for a in agents}
    done, pending = await asyncio.wait(tasks, timeout=timeout,
                                       return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    assert done, "no agent got an assignment"
    task = done.pop()
    frame = task.result()
    assert isinstance(frame, wp.Assign), frame
    return tasks[task], frame


async def _terminal(service: RunService, run_id: str) -> Any:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        record = await service.store.get_run(run_id)
        if record is not None and record.status in TERMINAL \
                and run_id not in service.active_run_ids():
            return record
        await asyncio.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish")


def _device(listing: dict, worker: str, value: str = "cuda:0") -> dict:
    entry = next(w for w in listing["workers"] if w["id"] == worker)
    return next(d for d in entry["devices"] if d["value"] == value)


async def test_two_agents_on_one_card_run_any_cuda_runs_one_at_a_time(env):
    service, _ = env
    async with _client() as client, \
            aconnect_ws(WS_PATH, client) as ws_a, \
            aconnect_ws(WS_PATH, client) as ws_b:
        agents = (_Agent(ws_a, "meow-e2e"), _Agent(ws_b, "meow-join"))
        for agent in agents:
            assert isinstance(await agent.hello(), wp.Welcome)
        listing = (await client.get("/api/workers")).json()
        assert _device(listing, "meow-e2e")["same_card"] == [
            {"worker": "meow-join", "device": "cuda:0"}]
        assert _device(listing, "meow-join")["same_card"] == [
            {"worker": "meow-e2e", "device": "cuda:0"}]
        assert _device(listing, "meow-e2e", "cpu")["same_card"] == []

        def submit():
            return client.post("/api/runs", json={
                "graph": _graph(), "options": {
                    "device": "cuda", "target": {"worker": "any"}}})

        first, second = (await submit()).json(), (await submit()).json()
        assert first["status"] == "running"
        assert second["status"] == "queued"
        owner, assign = await _assigned(*agents)
        assert assign.run_id == first["run_id"]
        await owner.send(wp.RunStarted(run_id=assign.run_id, seq=1, pid=1))
        listing = (await client.get("/api/workers")).json()
        for name in ("meow-e2e", "meow-join"):
            assert _device(listing, name)["slots"] == {"limit": 1, "taken": 1}
        # Neither agent gets the second run while the card is busy.
        with pytest.raises(AssertionError):
            await _assigned(*agents, timeout=0.5)
        assert service.is_queued(second["run_id"])

        await owner.send(wp.RunFinished(run_id=assign.run_id, seq=2,
                                        status="succeeded"))
        await _terminal(service, assign.run_id)
        owner, assign = await _assigned(*agents)
        assert assign.run_id == second["run_id"]
        await owner.send(wp.RunFinished(run_id=assign.run_id, seq=1,
                                        status="succeeded"))
        await _terminal(service, assign.run_id)


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGIWebSocketTransport(app=app),
                       base_url=BASE, headers={TOKEN_HEADER: session_token()})
