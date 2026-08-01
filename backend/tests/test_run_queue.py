"""Tests for the per-device run queue — RunService scheduling (#123).

Five concerns, in order:

1. **FIFO.** Runs on one device start in submit order, one (or ``cpu``'s
   two) at a time, and the next starts when a slot frees.
2. **Isolation.** Each resolved device has its own queue: a saturated
   ``cuda:0`` must not delay a ``cpu`` run, and vice versa.
3. **Bypass.** The interactive lane skips the FIFO and is bounded by its own
   cap plus one-run-per-session, which is the structural close of the shared
   cache/weights hazard #121 disclosed.
4. **Cancel.** A run cancelled while WAITING leaves no trace: it never
   executed, never took a slot, and never delays the queue behind it.
5. **Lifecycle.** A waiting run is observable (position, long poll, live
   subscription) and is retired honestly when the server stops.

Device strings are FAKED (see ``fake_devices``) rather than probed, because
CI has no GPU and the whole point of the module is that ``cuda:0`` and
``cpu`` are different keys.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from app.core import run_service as run_service_module
from app.core.cache import ExecutionCache
from app.core.db import Database
from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from app.core.node_registry import registry
from app.core.run_service import (
    EVENT_RUN_COMPLETED,
    EVENT_RUN_STARTED,
    EVENT_RUN_STOPPED,
    LANE_INTERACTIVE,
    STOP_REASON_CANCELLED,
    STOP_REASON_INTERRUPTED,
    InteractiveSession,
    QueueLimits,
    RunService,
    RunServiceUnavailable,
    canonical_queue_key,
    parse_queue_overrides,
)
from app.core.run_store import (
    STATUS_CANCELLED,
    STATUS_INTERRUPTED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
    RunProvenance,
    RunStore,
)

# ── the probe node ────────────────────────────────────────────────────────


class _Probe:
    """What the graphs report back: start order and observed concurrency.

    A plain object rather than module globals so the fixture that resets it
    cannot miss a field. Mutated from ENGINE WORKER THREADS, hence the lock —
    ``append`` alone would be safe under the GIL, but the peak-concurrency
    read-modify-write is not.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.started: list[str] = []
        self.inflight = 0
        self.peak: dict[str, int] = {}
        #: Peak across ALL keys. The per-key counter is labelled by the
        #: probe's own param, so it cannot see two ALIASES of one device
        #: overlapping; this can.
        self.peak_total = 0

    def enter(self, label: str, device: str) -> None:
        with self.lock:
            self.started.append(label)
            self.inflight += 1
            self.peak[device] = max(self.peak.get(device, 0), self.inflight)
            self.peak_total = max(self.peak_total, self.inflight)

    def leave(self) -> None:
        with self.lock:
            self.inflight -= 1


_probe = _Probe()


class _QueueProbeNode(BaseNode):
    """Blocks a worker thread and records that it did.

    ``device`` is a PARAM, not read off the context: the peak counter is
    per-key, and a test that fakes device resolution wants the key it asked
    for rather than whatever the machine actually has.
    """

    NODE_NAME = "_QueueProbe"
    CATEGORY = "Test"
    DESCRIPTION = "Records its start and sleeps"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="label", param_type=ParamType.STRING,
                            default=""),
            ParamDefinition(name="device", param_type=ParamType.STRING,
                            default="cpu"),
            ParamDefinition(name="seconds", param_type=ParamType.FLOAT,
                            default=0.05),
        ]

    def execute(self, inputs: dict[str, Any],
                params: dict[str, Any]) -> dict[str, Any]:
        _probe.enter(str(params.get("label", "")), str(params.get("device")))
        try:
            time.sleep(float(params.get("seconds", 0.05)))
        finally:
            _probe.leave()
        return {"value": inputs.get("value")}


_TEST_NODES = {"_QueueProbe": _QueueProbeNode}


@pytest.fixture(autouse=True)
def _register_test_nodes():
    registry._nodes.update(_TEST_NODES)
    yield
    for name in _TEST_NODES:
        registry._nodes.pop(name, None)


@pytest.fixture(autouse=True)
def probe():
    """A clean probe per test. Returned so a test can read it directly."""
    global _probe
    _probe = _Probe()
    return _probe


# ── fixtures / helpers ────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def fake_devices(monkeypatch):
    """Every device string resolves to itself.

    ``resolve_device`` degrades ``cuda``/``mps`` to ``cpu`` on a machine
    without them, which on CI would collapse every queue in this module into
    one and make the per-device tests silently assert nothing. Patching it is
    honest here: the scheduler only ever handles the RESULT as an opaque
    string, and nothing in this file asks torch for anything.
    """
    monkeypatch.setattr(
        run_service_module, "resolve_device",
        lambda requested: (requested or "cpu").strip().lower() or "cpu")
    # Bare ``cuda`` canonicalises to the process's current device; pin it
    # so these tests mean the same thing on a CI box with no GPU and on a
    # workstation whose current device is not 0.
    monkeypatch.setattr(run_service_module, "_current_cuda_index",
                        lambda: 0)


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
def store(db):
    return RunStore(db)


@pytest.fixture
async def make_service(store):
    """Build services with explicit limits; drain them all afterwards.

    Limits are passed in rather than read from ``settings`` so a test states
    the policy it is testing (``gpu=1`` and a five-deep queue is the whole
    acceptance criterion) instead of depending on a default that config is
    free to change.
    """
    built: list[RunService] = []

    def _make(*, cpu: int = 2, gpu: int = 1, interactive: int = 2,
              overrides: tuple[tuple[str, int], ...] = (),
              **kwargs: Any) -> RunService:
        service = RunService(
            store, shutdown_grace_s=5.0,
            limits=QueueLimits(cpu=cpu, gpu=gpu, interactive=interactive,
                               overrides=overrides),
            **kwargs)
        built.append(service)
        return service

    try:
        yield _make
    finally:
        for service in built:
            await service.shutdown()


@pytest.fixture
async def service(make_service):
    """The common policy: one run per GPU, two per CPU, two interactive."""
    return make_service()


def _graph(label: str = "run", *, device: str = "cpu",
           seconds: float = 0.05) -> dict[str, Any]:
    """Start -> _TestSource -> _QueueProbe. The probe is what reports back."""
    return {
        "nodes": [
            {"id": "start", "type": "Start", "data": {"params": {}}},
            {"id": "src", "type": "_TestSource",
             "data": {"params": {"val": label}}},
            {"id": "probe", "type": "_QueueProbe", "data": {"params": {
                "label": label, "device": device, "seconds": seconds}}},
        ],
        "edges": [
            {"id": "et", "source": "start", "target": "src",
             "sourceHandle": "trigger", "type": "trigger"},
            {"id": "e1", "source": "src", "target": "probe",
             "sourceHandle": "value", "targetHandle": "value"},
        ],
    }


async def _submit(service: RunService, label: str, *, device: str = "cpu",
                  seconds: float = 0.05, lane: str | None = None,
                  session: InteractiveSession | None = None):
    """Submit one probe graph on *device*, returning the ``SubmitResult``."""
    options: dict[str, Any] = {"device": device}
    if lane is not None:
        options["lane"] = lane
    return await service.submit(
        _graph(label, device=device, seconds=seconds), options=options,
        name=label, session=session)


async def _await_terminal(store: RunStore, run_id: str, *,
                          timeout: float = 15.0):
    """Poll the STORE (never the in-process registry) until the row is done."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = await store.get_run(run_id)
        assert record is not None, f"run {run_id} vanished"
        if record.finished_at is not None:
            return record
        await asyncio.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish within {timeout}s")


async def _wait_until(predicate, *, timeout: float = 15.0,
                      what: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"{what} did not happen within {timeout}s")


# ── limits ────────────────────────────────────────────────────────────────


def test_limits_default_by_device_class():
    limits = QueueLimits(cpu=2, gpu=1)
    assert limits.for_key("cpu") == 2
    # Everything that is not cpu is treated as an accelerator, including a
    # backend nobody has added yet. (``cuda`` / ``mps`` never reach here as
    # keys — see canonical_queue_key — but the classification is by prefix,
    # not by an allowlist, so it holds for them too.)
    for key in ("cuda:0", "cuda:7", "mps:0", "xpu"):
        assert limits.for_key(key) == 1


def test_device_aliases_canonicalise_onto_one_key():
    """``cuda`` and ``cuda:0`` are one card, so they must be one queue."""
    assert canonical_queue_key("cuda") == "cuda:0"
    assert canonical_queue_key("mps") == "mps:0"
    # Already canonical, and a genuinely different card stays different.
    assert canonical_queue_key("cuda:0") == "cuda:0"
    assert canonical_queue_key("cuda:1") == "cuda:1"
    assert canonical_queue_key("mps:0") == "mps:0"
    assert canonical_queue_key("cpu") == "cpu"


def test_a_bare_cuda_follows_the_process_current_device(monkeypatch):
    """Not hardcoded 0: a bare ``cuda`` means whatever torch means by it.

    Pinning 0 would recreate the very collision this exists to close on any
    process whose current device is not 0 — ``cuda`` would be filed under
    ``cuda:0`` while actually running on card 1.
    """
    monkeypatch.setattr(run_service_module, "_current_cuda_index", lambda: 3)
    assert canonical_queue_key("cuda") == "cuda:3"
    assert canonical_queue_key("cuda:0") == "cuda:0"


def test_limits_overrides_beat_the_class_default():
    limits = QueueLimits(cpu=2, gpu=1, overrides=(("cuda:0", 4),))
    assert limits.for_key("cuda:0") == 4
    assert limits.for_key("cuda:1") == 1     # untouched
    assert limits.for_key("cpu") == 2


def test_limits_never_admit_zero():
    """A cap of zero is a queue that cannot drain — a hang, not a policy."""
    assert QueueLimits(cpu=0, gpu=0).for_key("cpu") == 1
    assert QueueLimits(overrides=(("cpu", -3),)).for_key("cpu") == 1


def test_override_parsing_splits_on_the_last_equals():
    """A key legitimately contains a colon; it must not confuse the split."""
    assert parse_queue_overrides("cuda:0=2,cpu=8") == (("cuda:0", 2),
                                                       ("cpu", 8))
    assert parse_queue_overrides(" CUDA:1 = 3 ") == (("cuda:1", 3),)
    assert parse_queue_overrides("") == ()


def test_override_parsing_skips_junk_instead_of_raising():
    """One typo in an env var must not stop the server from booting."""
    assert parse_queue_overrides("nope,cpu=x,cpu=0,=4,cuda:0=2") == (
        ("cuda:0", 2),)


# ── FIFO ──────────────────────────────────────────────────────────────────


async def test_five_gpu_runs_execute_serially_in_submit_order(service, store,
                                                              probe):
    """The acceptance criterion: cap 1 means one at a time, oldest first."""
    labels = [f"gpu-{i}" for i in range(5)]
    results = [await _submit(service, label, device="cuda:0", seconds=0.05)
               for label in labels]

    assert results[0].status == STATUS_RUNNING
    assert [r.status for r in results[1:]] == [STATUS_QUEUED] * 4
    assert service.queue_snapshot() == {
        "cuda:0": [r.run_id for r in results[1:]]}

    for result in results:
        record = await _await_terminal(store, result.run_id)
        assert record.status == STATUS_SUCCEEDED
        assert record.queue_key == "cuda:0"

    assert probe.started == labels          # FIFO, not just "all of them"
    assert probe.peak["cuda:0"] == 1        # and never two at once


async def test_queue_position_counts_from_the_front(service):
    results = [await _submit(service, f"gpu-{i}", device="cuda:0",
                             seconds=0.4) for i in range(4)]
    running, waiting = results[0], results[1:]

    # The running run holds a slot, not a place in line.
    assert service.queue_position(running.run_id) is None
    assert service.is_queued(running.run_id) is False
    assert [service.queue_position(r.run_id) for r in waiting] == [1, 2, 3]
    assert all(service.is_queued(r.run_id) for r in waiting)
    assert service.queue_position("nope") is None


async def test_cpu_admits_two_at_once_and_queues_the_third(service, store,
                                                           probe):
    """The cap is per key, and ``cpu``'s is two — contention, not death."""
    results = [await _submit(service, f"cpu-{i}", seconds=0.25)
               for i in range(3)]
    assert [r.status for r in results] == [STATUS_RUNNING, STATUS_RUNNING,
                                           STATUS_QUEUED]
    for result in results:
        await _await_terminal(store, result.run_id)
    assert probe.peak["cpu"] == 2


async def test_an_override_raises_one_devices_cap(make_service, store, probe):
    service = make_service(gpu=1, overrides=(("cuda:0", 2),))
    results = [await _submit(service, f"gpu-{i}", device="cuda:0",
                             seconds=0.2) for i in range(3)]
    assert [r.status for r in results] == [STATUS_RUNNING, STATUS_RUNNING,
                                           STATUS_QUEUED]
    for result in results:
        await _await_terminal(store, result.run_id)
    assert probe.peak["cuda:0"] == 2


async def test_a_freed_slot_promotes_the_next_run(service, store, probe):
    """The pump runs off a run ENDING, not off a timer."""
    first = await _submit(service, "first", device="cuda:0", seconds=0.05)
    second = await _submit(service, "second", device="cuda:0", seconds=0.05)
    assert second.status == STATUS_QUEUED

    await _await_terminal(store, first.run_id)
    await _wait_until(lambda: service.is_active(second.run_id),
                      what="the queued run was promoted")
    assert service.queue_snapshot() == {}
    await _await_terminal(store, second.run_id)
    assert probe.started == ["first", "second"]


async def test_a_failed_run_still_frees_its_slot(service, store):
    """Even a run whose graph the engine refuses must not wedge the queue."""
    broken = await service.submit({"nodes": [{"id": "x", "type": "NoSuchNode",
                                              "data": {"params": {}}}]},
                                  options={"device": "cuda:0"})
    queued = await _submit(service, "after", device="cuda:0", seconds=0.05)

    assert (await _await_terminal(store, broken.run_id)).status == "failed"
    assert (await _await_terminal(store, queued.run_id)).status == (
        STATUS_SUCCEEDED)


# ── per-key isolation ─────────────────────────────────────────────────────


async def test_a_saturated_gpu_does_not_delay_a_cpu_run(service, store, probe):
    """Two queues, two keys. The CPU run must not wait behind CUDA work."""
    gpu = [await _submit(service, f"gpu-{i}", device="cuda:0", seconds=0.4)
           for i in range(3)]
    cpu = await _submit(service, "cpu-now", seconds=0.05)

    assert cpu.status == STATUS_RUNNING
    record = await _await_terminal(store, cpu.run_id, timeout=5)
    assert record.status == STATUS_SUCCEEDED
    # ...while the GPU queue is demonstrably still backed up behind it.
    assert service.queue_snapshot()["cuda:0"]
    assert "cpu-now" in probe.started

    for result in gpu:
        await _await_terminal(store, result.run_id)


async def test_bare_cuda_and_cuda0_share_one_queue(service, store, probe):
    """The regression: two aliases of one card must not run concurrently.

    ``resolve_device`` returns both strings verbatim, so without
    canonicalisation these become two independent FIFOs over one GPU, each
    admitting its own run — and the two fight over the same VRAM, which is
    precisely what a cap of 1 exists to prevent.
    """
    bare = await _submit(service, "bare", device="cuda", seconds=0.2)
    indexed = await _submit(service, "indexed", device="cuda:0", seconds=0.2)

    assert bare.status == STATUS_RUNNING
    assert indexed.status == STATUS_QUEUED, \
        "cuda and cuda:0 were scheduled as two independent queues"
    assert list(service.queue_snapshot()) == ["cuda:0"]

    for result in (bare, indexed):
        record = await _await_terminal(store, result.run_id)
        # Both rows record the SAME canonical key, so a later reader cannot
        # tell them apart by device either.
        assert record.queue_key == "cuda:0"
    assert probe.peak_total == 1
    assert probe.started == ["bare", "indexed"]


async def test_bare_mps_and_mps0_share_one_queue(service):
    first = await _submit(service, "a", device="mps", seconds=0.3)
    second = await _submit(service, "b", device="mps:0", seconds=0.3)
    assert (first.status, second.status) == (STATUS_RUNNING, STATUS_QUEUED)
    assert list(service.queue_snapshot()) == ["mps:0"]


async def test_each_device_keeps_its_own_line(service):
    first_gpu = await _submit(service, "a", device="cuda:0", seconds=0.4)
    second_gpu = await _submit(service, "b", device="cuda:0", seconds=0.4)
    other_gpu = await _submit(service, "c", device="cuda:1", seconds=0.4)
    mps = await _submit(service, "d", device="mps", seconds=0.4)

    assert first_gpu.status == other_gpu.status == mps.status == STATUS_RUNNING
    assert second_gpu.status == STATUS_QUEUED
    assert service.queue_snapshot() == {"cuda:0": [second_gpu.run_id]}
    # Position is within the key, so the one waiting run is first in line and
    # not "fourth of everything submitted".
    assert service.queue_position(second_gpu.run_id) == 1


# ── the interactive lane ──────────────────────────────────────────────────


async def test_an_interactive_run_starts_alongside_a_full_gpu_queue(
        service, store, probe):
    """The acceptance criterion: a classroom demo never waits for training."""
    training = [await _submit(service, f"train-{i}", device="cuda:0",
                              seconds=0.4) for i in range(5)]
    assert len(service.queue_snapshot()["cuda:0"]) == 4

    demo = await _submit(service, "demo", device="cuda:0", seconds=0.05,
                         lane=LANE_INTERACTIVE,
                         session=InteractiveSession())
    assert demo.status == STATUS_RUNNING
    assert service.is_active(demo.run_id)
    assert service.queue_position(demo.run_id) is None

    record = await _await_terminal(store, demo.run_id, timeout=5)
    assert record.status == STATUS_SUCCEEDED
    assert record.options["lane"] == LANE_INTERACTIVE
    # It really did overtake: four training runs are still waiting.
    assert len(service.queue_snapshot()["cuda:0"]) == 4

    for result in training:
        await _await_terminal(store, result.run_id)


async def test_the_interactive_cap_refuses_rather_than_queues(make_service,
                                                              store):
    """A hard cap is hard: a live user gets an answer, not a silent wait."""
    service = make_service(interactive=1)
    first = await _submit(service, "one", lane=LANE_INTERACTIVE,
                          seconds=0.4, session=InteractiveSession())

    with pytest.raises(RunServiceUnavailable, match="interactive run"):
        await _submit(service, "two", lane=LANE_INTERACTIVE,
                      session=InteractiveSession())

    # Refused BEFORE the row was written: no orphan for recovery to clean up.
    assert [r.name for r in await store.list_runs()] == ["one"]

    await _await_terminal(store, first.run_id)
    # The slot comes back, so the canvas is usable again immediately.
    again = await _submit(service, "three", lane=LANE_INTERACTIVE,
                          session=InteractiveSession())
    assert again.status == STATUS_RUNNING
    await _await_terminal(store, again.run_id)


async def test_one_run_per_session_is_enforced_by_the_service(service, store):
    """#121's disclosed gap, closed structurally rather than by a button.

    Two concurrent runs behind one ExecutionCache would serve each other
    half-built tensors. The frontend prevents it by disabling Run; that is a
    convention, and this is the guarantee.
    """
    cache = ExecutionCache()
    first = await _submit(service, "one", lane=LANE_INTERACTIVE, seconds=0.4,
                          session=InteractiveSession(cache=cache))

    with pytest.raises(RunServiceUnavailable, match="session"):
        await _submit(service, "two", lane=LANE_INTERACTIVE,
                      session=InteractiveSession(cache=cache))

    # A DIFFERENT socket's cache is unaffected — the cap bounds those.
    other = await _submit(service, "other", lane=LANE_INTERACTIVE,
                          seconds=0.05,
                          session=InteractiveSession(cache=ExecutionCache()))
    assert other.status == STATUS_RUNNING

    await _await_terminal(store, first.run_id)
    await _await_terminal(store, other.run_id)
    # Freed on completion, so the same socket's next click works.
    reused = await _submit(service, "again", lane=LANE_INTERACTIVE,
                           session=InteractiveSession(cache=cache))
    assert reused.status == STATUS_RUNNING
    await _await_terminal(store, reused.run_id)


async def test_the_rule_keys_on_the_cache_not_the_session_wrapper(service,
                                                                  store):
    """The shape the WebSocket actually produces — and a live-caught bug.

    ``ws_execution.handle_execute`` builds a NEW InteractiveSession on every
    execute message (``changed_nodes`` differs per click) around the ONE
    cache it made when the socket opened. Keying on the wrapper compares two
    objects that are never the same one, so the rule silently never fires.
    """
    cache = ExecutionCache()
    first_wrapper = InteractiveSession(cache=cache, changed_nodes=("a",))
    second_wrapper = InteractiveSession(cache=cache, changed_nodes=("b",))
    assert first_wrapper is not second_wrapper

    first = await _submit(service, "one", lane=LANE_INTERACTIVE, seconds=0.4,
                          session=first_wrapper)
    assert first.status == STATUS_RUNNING
    with pytest.raises(RunServiceUnavailable, match="session"):
        await _submit(service, "two", lane=LANE_INTERACTIVE,
                      session=second_wrapper)
    await _await_terminal(store, first.run_id)


async def test_the_session_is_free_the_moment_the_run_reports_it_is_over(
        service, store):
    """A canvas re-enables **Run** on ``execution_complete``; the next click
    must work.

    Releasing the session in the task's cleanup instead — after the terminal
    event rather than before it — leaves a window in which the client has
    been told the run ended and the very next submit is still refused. A
    live canvas found it; this pins the ordering.
    """
    cache = ExecutionCache()
    first = await _submit(service, "one", lane=LANE_INTERACTIVE, seconds=0.05,
                          session=InteractiveSession(cache=cache))

    async with service.subscribe(first.run_id) as subscription:
        while True:
            record = await asyncio.wait_for(subscription.queue.get(),
                                            timeout=10)
            if record is None or record.type == EVENT_RUN_COMPLETED:
                break
        # The instant the client hears it is over, with no yield in between.
        second = await _submit(service, "two", lane=LANE_INTERACTIVE,
                               seconds=0.05,
                               session=InteractiveSession(cache=cache))
    assert second.status == STATUS_RUNNING
    await _await_terminal(store, second.run_id)


async def test_a_freed_device_slot_is_returned_before_the_bookkeeping(
        service, store):
    """Same ordering for the queue: the next run starts when the GPU is idle,
    not when the previous run has finished writing rows."""
    first = await _submit(service, "first", device="cuda:0", seconds=0.05)
    second = await _submit(service, "second", device="cuda:0", seconds=0.05)

    async with service.subscribe(first.run_id) as subscription:
        while True:
            record = await asyncio.wait_for(subscription.queue.get(),
                                            timeout=10)
            if record is None or record.type == EVENT_RUN_COMPLETED:
                break
        assert service.is_active(second.run_id), "the queue did not move"
    await _await_terminal(store, second.run_id)


async def test_two_sockets_do_not_serialise_each_other(service, store):
    """Each canvas connection has its own cache, so each may have a run."""
    first = await _submit(service, "one", lane=LANE_INTERACTIVE, seconds=0.2,
                          session=InteractiveSession(cache=ExecutionCache()))
    second = await _submit(service, "two", lane=LANE_INTERACTIVE, seconds=0.2,
                           session=InteractiveSession(cache=ExecutionCache()))
    assert first.status == second.status == STATUS_RUNNING
    for result in (first, second):
        await _await_terminal(store, result.run_id)


async def test_a_sessionless_interactive_run_still_counts_against_the_cap(
        make_service, store):
    service = make_service(interactive=1)
    first = await _submit(service, "one", lane=LANE_INTERACTIVE, seconds=0.3)
    assert first.status == STATUS_RUNNING
    with pytest.raises(RunServiceUnavailable, match="interactive run"):
        await _submit(service, "two", lane=LANE_INTERACTIVE)
    await _await_terminal(store, first.run_id)


async def test_an_unknown_lane_queues_rather_than_bypassing(service):
    """Bypass is a capability; a typo'd lane must not grant one."""
    first = await _submit(service, "a", device="cuda:0", seconds=0.3)
    typo = await service.submit(_graph("b", device="cuda:0", seconds=0.3),
                                options={"device": "cuda:0",
                                         "lane": "interactve"})
    assert first.status == STATUS_RUNNING
    assert typo.status == STATUS_QUEUED


# ── cancelling a waiting run ──────────────────────────────────────────────


async def test_cancelling_a_queued_run_dequeues_it_with_no_side_effects(
        service, store, probe):
    running = await _submit(service, "running", device="cuda:0", seconds=0.3)
    waiting = await _submit(service, "waiting", device="cuda:0", seconds=0.3)
    assert waiting.status == STATUS_QUEUED

    outcome = await service.cancel(waiting.run_id)
    assert outcome is not None
    assert (outcome.cancelled, outcome.status) == (True, STATUS_CANCELLED)

    row = await store.get_run(waiting.run_id)
    assert row.status == STATUS_CANCELLED
    assert row.started_at is None            # it never ran
    assert row.finished_at is not None
    assert service.queue_snapshot() == {}
    assert service.queue_position(waiting.run_id) is None

    # The run in front is untouched, and nothing was promoted in its place.
    assert (await _await_terminal(store, running.run_id)).status == (
        STATUS_SUCCEEDED)
    await asyncio.sleep(0.05)
    assert probe.started == ["running"]
    assert service.is_active(waiting.run_id) is False


async def test_a_cancelled_waiting_run_gets_a_terminal_event(service, store):
    """A follower needs a frame that says it ENDED, not silence."""
    await _submit(service, "running", device="cuda:0", seconds=0.3)
    waiting = await _submit(service, "waiting", device="cuda:0", seconds=0.3)
    await service.cancel(waiting.run_id)

    events = await store.get_events(waiting.run_id)
    assert [event.type for event in events] == [EVENT_RUN_STOPPED]
    assert events[0].payload == {"reason": STOP_REASON_CANCELLED}


async def test_cancelling_the_middle_of_a_queue_keeps_the_order(service,
                                                                store, probe):
    running = await _submit(service, "a", device="cuda:0", seconds=0.05)
    middle = await _submit(service, "b", device="cuda:0", seconds=0.05)
    last = await _submit(service, "c", device="cuda:0", seconds=0.05)

    await service.cancel(middle.run_id)
    assert service.queue_snapshot() == {"cuda:0": [last.run_id]}
    assert service.queue_position(last.run_id) == 1

    for result in (running, last):
        await _await_terminal(store, result.run_id)
    assert probe.started == ["a", "c"]


async def test_cancel_still_answers_for_runs_it_never_queued(service, store):
    """The store fallback (an orphan row, a finished run) is unchanged."""
    assert await service.cancel("nope") is None
    orphan = await store.create_run(graph_snapshot=_graph(), options={},
                                    provenance=RunProvenance())
    outcome = await service.cancel(orphan.id)
    assert (outcome.cancelled, outcome.status) == (True, STATUS_CANCELLED)


# ── observing a waiting run ───────────────────────────────────────────────


async def test_a_long_poll_on_a_queued_run_parks_and_wakes_on_its_start(
        service, store):
    """Without the pending waiter this is a busy loop against the database."""
    running = await _submit(service, "running", device="cuda:0", seconds=0.05)
    waiting = await _submit(service, "waiting", device="cuda:0", seconds=0.05)

    poll = asyncio.create_task(
        service.wait_for_events(waiting.run_id, wait=10.0))
    await asyncio.sleep(0.05)
    assert not poll.done(), "the poll returned instantly instead of parking"

    await _await_terminal(store, running.run_id)
    events = await asyncio.wait_for(poll, timeout=10)
    # Whatever else had landed by the time the read ran, the poll woke on the
    # promotion and the log starts where a fresh follower expects it to.
    assert events and events[0].type == EVENT_RUN_STARTED
    assert events[0].cursor == 1
    await _await_terminal(store, waiting.run_id)


async def test_a_long_poll_on_a_queued_run_wakes_when_it_is_cancelled(
        service, store):
    await _submit(service, "running", device="cuda:0", seconds=0.3)
    waiting = await _submit(service, "waiting", device="cuda:0", seconds=0.3)

    poll = asyncio.create_task(
        service.wait_for_events(waiting.run_id, wait=10.0))
    await asyncio.sleep(0.05)
    await service.cancel(waiting.run_id)
    events = await asyncio.wait_for(poll, timeout=10)
    assert [event.type for event in events] == [EVENT_RUN_STOPPED]


async def test_a_subscription_survives_the_promotion(service, store):
    """Attach to a run that has not started, and see it start.

    The subscriber set is handed to the started run, so there is no
    re-subscribe and no window in which an event could be missed.
    """
    running = await _submit(service, "running", device="cuda:0", seconds=0.05)
    waiting = await _submit(service, "waiting", device="cuda:0", seconds=0.05)

    async with service.subscribe(waiting.run_id) as subscription:
        assert subscription.closed is False
        await _await_terminal(store, running.run_id)
        record = await asyncio.wait_for(subscription.queue.get(), timeout=10)
        assert record is not None and record.type == EVENT_RUN_STARTED
    await _await_terminal(store, waiting.run_id)


async def test_subscribing_to_a_cancelled_waiting_run_closes_the_feed(
        service, store):
    await _submit(service, "running", device="cuda:0", seconds=0.3)
    waiting = await _submit(service, "waiting", device="cuda:0", seconds=0.3)

    async with service.subscribe(waiting.run_id) as subscription:
        await service.cancel(waiting.run_id)
        assert subscription.closed is True
        assert await asyncio.wait_for(subscription.queue.get(),
                                      timeout=5) is None


# ── the queued run's graph lives on the row ───────────────────────────────


async def test_a_promoted_run_reads_its_graph_back_from_the_row(service,
                                                                store, probe):
    """The queue holds ids, not graphs — so a deep queue costs no memory."""
    running = await _submit(service, "running", device="cuda:0", seconds=0.05)
    waiting = await _submit(service, "waiting", device="cuda:0", seconds=0.05)

    entry = service._pending_by_id[waiting.run_id]
    assert not hasattr(entry, "graph")
    assert await store.get_graph_snapshot(waiting.run_id) is not None

    for result in (running, waiting):
        await _await_terminal(store, result.run_id)
    assert probe.started == ["running", "waiting"]


async def test_a_queued_run_deleted_before_it_starts_does_not_wedge_the_queue(
        service, store, probe):
    """The row is the input; if it is gone the promotion gives up quietly."""
    running = await _submit(service, "running", device="cuda:0", seconds=0.05)
    doomed = await _submit(service, "doomed", device="cuda:0", seconds=0.05)
    after = await _submit(service, "after", device="cuda:0", seconds=0.05)

    assert await store.delete_run(doomed.run_id) is True
    await _await_terminal(store, running.run_id)
    await _await_terminal(store, after.run_id)
    assert probe.started == ["running", "after"]
    assert service.queue_snapshot() == {}


# ── shutdown ──────────────────────────────────────────────────────────────


async def test_shutdown_retires_the_whole_queue_as_interrupted(make_service,
                                                               store, probe):
    """Nothing resumes a queue across a restart, so the rows say so NOW.

    Identical to what the next boot's ``recover_interrupted`` would write for
    the same rows — the two paths converge, this one just does not wait.
    """
    service = make_service()
    running = await _submit(service, "running", device="cuda:0", seconds=0.05)
    waiting = [await _submit(service, f"waiting-{i}", device="cuda:0",
                             seconds=0.05) for i in range(3)]

    await service.shutdown()

    for result in waiting:
        row = await store.get_run(result.run_id)
        assert row.status == STATUS_INTERRUPTED
        assert row.started_at is None
        assert row.finished_at is not None
        events = await store.get_events(result.run_id)
        assert [event.type for event in events] == [EVENT_RUN_STOPPED]
        assert events[0].payload == {"reason": STOP_REASON_INTERRUPTED}

    assert service.queue_snapshot() == {}
    assert (await store.get_run(running.run_id)).finished_at is not None
    assert probe.started == ["running"]


async def test_a_submit_that_lands_during_shutdown_is_refused(make_service,
                                                              store,
                                                              monkeypatch):
    """A run persisted after the queue was drained must not join it.

    Otherwise it waits forever on a pump that will never run again, with the
    lifespan closing the database underneath it.
    """
    service = make_service()
    reached_db = asyncio.Event()
    release = asyncio.Event()
    original = RunStore.create_run

    async def stalling_create(self, **kwargs):
        record = await original(self, **kwargs)
        reached_db.set()
        await release.wait()
        return record

    monkeypatch.setattr(RunStore, "create_run", stalling_create)
    task = asyncio.create_task(_submit(service, "late", device="cuda:0"))
    await asyncio.wait_for(reached_db.wait(), timeout=5)

    await service.shutdown()
    release.set()
    with pytest.raises(RunServiceUnavailable):
        await task
    assert service.queue_snapshot() == {}


async def test_recovery_refuses_to_run_while_the_queue_is_populated(service):
    """Retiring rows the scheduler still holds would strand the queue."""
    await _submit(service, "running", device="cuda:0", seconds=0.3)
    await _submit(service, "waiting", device="cuda:0", seconds=0.3)
    with pytest.raises(RuntimeError, match="queued run"):
        await service.recover_interrupted()


async def test_a_restart_retires_rows_a_hard_kill_left_queued(make_service,
                                                              store):
    """The other half of the shutdown story: no graceful stop at all."""
    orphan = await store.create_run(graph_snapshot=_graph(), options={},
                                    queue_key="cuda:0",
                                    provenance=RunProvenance())
    assert orphan.status == STATUS_QUEUED

    fresh = make_service()
    assert await fresh.recover_interrupted() == 1
    row = await store.get_run(orphan.id)
    assert row.status == STATUS_INTERRUPTED
    assert row.started_at is None
