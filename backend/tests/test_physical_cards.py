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
from app.core.scheduling import (
    RESERVED_SLOT,
    CardBudget,
    Placement,
    SlotBudget,
    WorkerScheduler,
)
from app.core.sweep_store import SweepStore
from app.core.version import get_version
from app.core.worker_store import WorkerStore
from app.core.workers import WorkerRegistry
from app.main import app

from .test_worker_adoption import db  # noqa: F401

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


#: The registries a test made, shut down after it (``_shut_down``).
_MADE: list[WorkerRegistry] = []


@pytest.fixture(autouse=True)
async def _shut_down():
    """Every test here is async: a registry's watchdog needs a running
    loop, which a sync test does not have once another module closed it."""
    yield
    while _MADE:
        await _MADE.pop().shutdown()


def _registry(*hellos: wp.Hello, local: list | None = None,
              same_host: tuple[str, ...] = ()) -> WorkerRegistry:
    registry = WorkerRegistry(local_devices=lambda: list(local or []))
    _MADE.append(registry)
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


async def test_two_workers_offering_one_uuid_are_one_card_whatever_the_numbers():
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
    assert registry.card_peers("meow-e2e", "cuda:1") == ()
    assert registry.card_peers("meow-e2e", "not a device") == ()
    assert registry.card_identity(("meow-e2e", "cuda:0")) == f"gpu:{UUID}"
    assert registry.card_identity(("meow-join", "cuda:0")) is None


async def test_a_same_host_workers_devices_are_this_processs_too():
    """One mechanism for #773 and #842: a same-host worker's card, CPU and
    Apple GPU are grouped with this process's (no slot is aliased), and a
    worker on the same card that was not recognised as same-host (another
    machine id) joins by UUID; its CPU stays its own."""
    registry = _registry(
        _hello("near", _card("cuda:0", UUID), _card("mps:0", None)),
        _hello("boxed", _card("cuda:0", UUID)),
        local=[_Fact("cpu"), _Fact("cuda:0", UUID), _Fact("mps:0")],
        same_host=("near",))
    near = registry.get("near")
    assert near.shared == {"cpu": "cpu", "cuda:0": "cuda:0",
                           "mps:0": "mps:0"}
    assert registry.card_slots(("local", "cuda:0")) == (
        ("local", "cuda:0"), ("boxed", "cuda:0"), ("near", "cuda:0"))
    assert registry.card_slots(("near", "cpu")) == (
        ("local", "cpu"), ("near", "cpu"))
    assert registry.card_slots(("near", "mps:0")) == (
        ("local", "mps:0"), ("near", "mps:0"))
    assert registry.card_slots(("boxed", "cpu")) == (("boxed", "cpu"),)
    assert registry.card_peers("near", "cuda:0") == (
        ("local", "cuda:0"), ("boxed", "cuda:0"))


async def test_the_index_is_rebuilt_only_when_devices_change(caplog):
    registry = _registry(_hello("a", _card("cuda:0", UUID)))
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
        index = registry._cards
        # New memory figures, a drain, a policy: the index stays.
        registry.heartbeat(b, wp.Heartbeat(devices=[
            _card("cpu", None, None, None),
            _card("cuda:0", UUID, free=3 * GIB)]))
        registry.drain("b", True)
        registry.set_device_policy("b", "cuda:0", mem.POLICY_SHARED)
        assert registry.card_slots(("b", "cuda:0")) == (
            ("a", "cuda:0"), ("b", "cuda:0"))
        assert registry._cards is index
    logged = [r.getMessage() for r in caplog.records
              if "one physical device" in r.getMessage()]
    assert logged == [f"one physical device (gpu:{UUID}) is offered as "
                      "a/cuda:0, b/cuda:0; runs on it count against one "
                      "budget"]
    # A disconnected worker keeps its devices (its runs keep their slots
    # while they wait, #783), so its card stays grouped, but it is not
    # online.
    registry.disconnect(b, "test")
    assert registry.card_slots(("a", "cuda:0")) == (
        ("a", "cuda:0"), ("b", "cuda:0"))
    assert not registry.slot_online(("b", "cuda:0"))
    assert registry.slot_online(("local", "cuda:0"))
    assert not registry.slot_online(("nobody", "cuda:0"))
    registry.forget("b")
    assert registry.card_slots(("a", "cuda:0")) == (("a", "cuda:0"),)


async def test_this_processs_devices_are_not_probed_until_a_worker_names_one():
    """A server whose workers name no device it could share never lists
    its own devices for this."""
    calls = []

    def local_devices():
        calls.append(1)
        return [_Fact("cuda:0", UUID)]

    registry = WorkerRegistry(local_devices=local_devices)
    _MADE.append(registry)
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


async def test_the_server_probes_its_devices_in_a_thread_never_on_the_loop():
    """``probe_off_loop`` (the server's registry): the probe runs in a
    thread, once; until it lands this process shares nothing, and when it
    lands the same-host workers are matched again and the queue is told."""
    import threading

    threads, changed = [], []

    def local_devices():
        threads.append(threading.current_thread())
        return [_Fact("cpu"), _Fact("cuda:0", UUID)]

    registry = WorkerRegistry(local_devices=local_devices,
                              probe_off_loop=True)
    _MADE.append(registry)
    registry.add_listener(lambda: changed.append(1))
    near, _ = registry.connect(_hello("near", _card("cuda:0", UUID)),
                               send=_noop, close=_noop, same_host=True)
    # Not probed yet: nothing of this process's is matched, and the
    # lookup started the probe instead of running it here.
    assert near.shared == {}
    assert registry.card_slots(("near", "cuda:0")) == (("near", "cuda:0"),)
    await registry.probe_local_devices()
    await registry.probe_local_devices()  # once
    assert threads and threads[0] is not threading.main_thread()
    assert len(threads) == 1
    assert near.shared == {"cpu": "cpu", "cuda:0": "cuda:0"}
    assert registry.card_slots(("near", "cuda:0")) == (
        ("local", "cuda:0"), ("near", "cuda:0"))
    assert changed

    def broken():
        raise RuntimeError("no NVML")

    failing = WorkerRegistry(local_devices=broken, probe_off_loop=True)
    _MADE.append(failing)
    await failing.probe_local_devices()
    assert failing._local_identities() == {}
    # Without a running loop the lookup cannot start a probe: nothing.
    idle = WorkerRegistry(local_devices=local_devices, probe_off_loop=True)
    _MADE.append(idle)
    assert await asyncio.to_thread(idle._local_devices) == []


async def test_an_adopted_runs_slots_are_grouped_before_its_worker_is_back():
    """Review item 1: a run adopted after a restart (#793) whose worker is
    not known yet holds its slot through a placeholder that lists no
    devices; the card identity its ``worker_resume`` recorded groups it
    with the card's other offerers meanwhile."""
    registry = _registry(_hello("meow-join", _card("cuda:0", UUID)))
    assert registry.card_slots(("meow-join", "cuda:0")) == (
        ("meow-join", "cuda:0"),)
    registry.remember_cards({("meow-e2e", "cuda:0"): f"gpu:{UUID}"})
    registry.remember_cards({})
    assert registry.card_slots(("meow-join", "cuda:0")) == (
        ("meow-join", "cuda:0"), ("meow-e2e", "cuda:0"))
    scheduler = WorkerScheduler(QueueLimits(), registry, lambda: [])
    assert scheduler.place(_on("meow-join"),
                           {("meow-e2e", "cuda:0"): 1}) is None
    registry.forget("meow-e2e")
    assert registry.card_slots(("meow-join", "cuda:0")) == (
        ("meow-join", "cuda:0"),)


# ── placement per card ───────────────────────────────────────────────────


def _two_on_one_card(limits: QueueLimits = QueueLimits(),
                     local: tuple = ("cpu",)) -> WorkerScheduler:
    registry = _registry(_hello("meow-e2e", _card("cuda:0", UUID)),
                         _hello("meow-join", _card("cuda:0", UUID)))
    return WorkerScheduler(limits, registry, lambda: list(local))


async def test_two_any_runs_at_limit_1_never_share_the_card():
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
    assert scheduler.card_held(("meow-join", "cuda:0"), taken) == 1
    taken[("meow-e2e", "cuda:0")] = 0
    assert scheduler.place(_any("r2"), taken) is not None
    # The busy-card wait is just a turn, not a reason.
    assert scheduler.explain(_on("meow-join"),
                             {("meow-e2e", "cuda:0"): 1}) is None


async def test_a_card_takes_the_smallest_cap_among_its_online_offerers():
    """Review item 2: an offerer that went away does not keep the card's
    cap down, unless it still holds a run there."""
    raised = QueueLimits(overrides=parse_queue_overrides("meow-e2e/cuda:0=2"))
    scheduler = _two_on_one_card(raised)
    taken = {("meow-join", "cuda:0"): 1}
    assert scheduler.place(_on("meow-e2e"), taken) is None
    assert scheduler.card_limit(("meow-e2e", "cuda:0"), {}) == (
        1, ("meow-join", "cuda:0"))
    both = QueueLimits(overrides=parse_queue_overrides(
        "meow-e2e/cuda:0=2,meow-join/cuda:0=2"))
    assert _two_on_one_card(both).place(_on("meow-e2e"), taken) is not None
    # meow-join goes away: its cap no longer counts ...
    registry = scheduler._registry
    registry.disconnect(registry.get("meow-join"), "test")
    assert scheduler.card_limit(("meow-e2e", "cuda:0"), {}) == (
        2, ("meow-e2e", "cuda:0"))
    assert scheduler.place(_on("meow-e2e"),
                           {("meow-e2e", "cuda:0"): 1}) is not None
    # ... unless a run of it still holds the card (it may resume, #783).
    assert scheduler.card_limit(("meow-e2e", "cuda:0"), taken) == (
        1, ("meow-join", "cuda:0"))
    assert scheduler.place(_on("meow-e2e"), taken) is None


async def test_a_local_run_counts_a_worker_run_on_the_same_card():
    """This process's card and a worker's card with its UUID that was not
    recognised as same-host: one card for the in-process lane too."""
    registry = _registry(_hello("boxed", _card("cuda:0", UUID)),
                         local=[_Fact("cuda:0", UUID)])
    scheduler = WorkerScheduler(QueueLimits(), registry,
                                lambda: ["cpu", "cuda:0"])
    local = _Entry("l", "cuda:0", {"device": "cuda:0"})
    assert scheduler.place(local, {("boxed", "cuda:0"): 1}) is None
    assert scheduler.place(_on("boxed"), {("local", "cuda:0"): 1}) is None
    assert scheduler.place(local, {}) == Placement(
        "local", ("cuda:0",), (("local", "cuda:0"),))


async def test_without_grouping_taken_and_caps_read_as_before():
    """A registry without ``card_slots`` (or one whose grouping raises)
    leaves every slot its own card; :class:`SlotBudget` is the 3.0 rule."""

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
    budget = SlotBudget(QueueLimits())
    assert budget.held(("a", "cuda:0"), {("a", "cuda:0"): 1}) == 1
    assert budget.limit(("a", "cuda:0"), {}) == (1, ("a", "cuda:0"))
    assert not budget.free(("a", "cuda:0"), {("a", "cuda:0"): 1})


async def test_the_card_budget_sums_held_runs_and_keeps_reservations():
    registry = _registry(_hello("a", _card("cuda:0", UUID)),
                         _hello("b", _card("cuda:0", UUID)))
    budget = CardBudget(QueueLimits(), registry)
    taken = {("a", "cuda:0"): 1, ("b", "cuda:0"): 2, ("c", "x"): 5}
    assert budget.held(("a", "cuda:0"), taken) == 3
    assert budget.held(("c", "x"), taken) == 5
    assert budget.held(("d", "x"), taken) == 0
    reserved = {("b", "cuda:0"): RESERVED_SLOT}
    assert budget.held(("a", "cuda:0"), reserved) >= RESERVED_SLOT
    assert not budget.free(("a", "cuda:0"), reserved)


# ── memory admission per card (#788) ─────────────────────────────────────


def _shared_card(*shared: str) -> WorkerScheduler:
    registry = _registry(_hello("meow-e2e", _card("cuda:0", UUID)),
                         _hello("meow-join", _card("cuda:0", UUID)))
    for name in shared:
        registry.set_device_policy(name, "cuda:0", mem.POLICY_SHARED)
    return WorkerScheduler(QueueLimits(), registry, lambda: [])


async def test_one_ledger_per_card_across_its_offerers():
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


async def test_the_stricter_policy_wins_on_a_card_with_several_offerers():
    """Coordinator decision: shared through one agent and exclusive through
    the other (online), the card admits by count through both; once the
    exclusive offerer is away, by memory again."""
    scheduler = _shared_card("meow-e2e")
    e2e = scheduler._registry.get("meow-e2e")
    assert not scheduler.shared(e2e, "cuda:0")
    assert scheduler.admission(e2e, "cuda:0")["by_memory"] is False
    held = {("meow-join", "cuda:0"): 1}
    assert scheduler.place(_on("meow-e2e", budget_gib=1), held) is None
    by_count = scheduler.place(_on("meow-e2e", budget_gib=1), {})
    assert by_count is not None and by_count.memory_budget is None
    assert scheduler.place(_on("meow-join"),
                           {("meow-e2e", "cuda:0"): 1}) is None
    scheduler._registry.disconnect(scheduler._registry.get("meow-join"),
                                   "test")
    assert scheduler.shared(e2e, "cuda:0")
    assert scheduler.shared(e2e, "not a device") is False


async def test_an_in_process_run_holds_a_shared_card_from_budgeted_runs():
    """A same-host worker's card shared by memory: this process's run on
    it is a holder without a budget, so a budgeted run waits (#788)."""
    registry = _registry(_hello("near", _card("cuda:0", UUID)),
                         local=[_Fact("cuda:0", UUID)], same_host=("near",))
    registry.set_device_policy("near", "cuda:0", mem.POLICY_SHARED)
    scheduler = WorkerScheduler(QueueLimits(), registry, lambda: ["cuda:0"])
    near = registry.get("near")
    assert scheduler.shared(near, "cuda:0")
    held = {("local", "cuda:0"): 1}
    _, now, why = scheduler.card_state(near, "cuda:0", held, GIB)
    assert not now and why == mem.held_without_budget("cuda:0", "near")
    assert scheduler.place(_on("near", budget_gib=1), held) is None
    assert scheduler.place(_on("near", budget_gib=1), {}) is not None


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


async def _terminal(service: RunService, run_id: str) -> Any:
    deadline = time.monotonic() + 30
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
            {"worker": "meow-join", "device": "cuda:0",
             "policy": "exclusive", "online": True}]
        assert _device(listing, "meow-join")["same_card"] == [
            {"worker": "meow-e2e", "device": "cuda:0",
             "policy": "exclusive", "online": True}]
        assert _device(listing, "meow-e2e", "cpu")["same_card"] == []

        def submit():
            return client.post("/api/runs", json={
                "graph": _graph(), "options": {
                    "device": "cuda", "target": {"worker": "any"}}})

        first, second = (await submit()).json(), (await submit()).json()
        assert first["status"] == "running"
        assert second["status"] == "queued"
        # ``any`` tries the workers in name order: meow-e2e gets the run.
        # (Only that socket is read: cancelling a pending receive on the
        # other one can break the test transport on a slow runner.)
        owner = agents[0]
        assign = await owner.recv(timeout=30)
        assert isinstance(assign, wp.Assign)
        assert assign.run_id == first["run_id"]
        await owner.send(wp.RunStarted(run_id=assign.run_id, seq=1, pid=1))
        # A restarted server groups the run's slot before its worker is
        # back (review item 1): the card identity is stored with it.
        facts = (await service.store.get_run(assign.run_id)).execution_facts
        assert facts["worker_resume"]["cards"] == [f"gpu:{UUID}"]
        listing = (await client.get("/api/workers")).json()
        for name in ("meow-e2e", "meow-join"):
            assert _device(listing, name)["slots"] == {"limit": 1, "taken": 1}
        # Neither agent gets the second run while the card is busy.
        assert service.is_queued(second["run_id"])
        assert all(not w["running"] for w in listing["workers"]
                   if w["id"] == "meow-join")
        assert service.queue_reason(second["run_id"]) is None  # its turn

        await owner.send(wp.RunFinished(run_id=assign.run_id, seq=2,
                                        status="succeeded"))
        await _terminal(service, assign.run_id)
        assign = await owner.recv(timeout=30)
        assert isinstance(assign, wp.Assign)
        assert assign.run_id == second["run_id"]
        await owner.send(wp.RunFinished(run_id=assign.run_id, seq=1,
                                        status="succeeded"))
        await _terminal(service, assign.run_id)


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGIWebSocketTransport(app=app),
                       base_url=BASE, headers={TOKEN_HEADER: session_token()})


async def test_a_restart_groups_an_adopted_runs_card_with_its_offerers(
        db):  # noqa: F811
    """Review item 1, through ``RunService``: the card identity stored in
    ``worker_resume`` reaches the registry when the run is adopted, so the
    adopted run's card is full for the other agent offering it, before
    its own worker has reconnected."""
    from app.core.run_service import WORKER_RESUME_KEY

    first = db()
    store = first.service.store
    await store.create_run(run_id="r1", graph_snapshot=_graph(), options={},
                           worker_id="meow-e2e")
    await store.mark_running(
        "r1", queue_key="meow-e2e/cuda:0", worker_id="meow-e2e",
        devices=["cuda:0"], execution_facts={WORKER_RESUME_KEY: {
            "last_seq": 0, "slots": [["meow-e2e", "cuda:0"]],
            "cards": [f"gpu:{UUID}"]}})
    await first.close()
    second = db()
    await second.service.recover_interrupted()
    second.registry.connect(_hello("meow-join", _card("cuda:0", UUID)),
                            send=_noop, close=_noop)
    assert second.service.slot_usage(("meow-join", "cuda:0")) == {
        "limit": 1, "taken": 1}
