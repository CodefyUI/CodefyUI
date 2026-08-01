"""Tests for app.core.run_service — the server-owned run lifecycle (#120).

Five concerns, in order:

1. The three acceptance criteria of the issue: a run outlives whatever
   submitted it; two concurrent runs are independent; a restart retires
   ``running`` rows instead of leaving zombies.
2. Options normalization (the submit contract's only strict surface).
3. Event persistence + fan-out: every engine callback reaches
   ``exec_run_events`` AND any live subscriber, drop-tolerantly.
4. Metrics: scalars off progress payloads, batched.
5. Lifecycle plumbing: cancel semantics, retention, shutdown drain.

The graphs here are deliberately tiny — Start -> source -> Print — because
none of this is testing the engine.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from app.core.db import Database
from app.core.node_base import (
    MEDIA_IMAGE,
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from app.core.node_registry import registry
from app.core.run_service import (
    EVENT_NODE_STATUS,
    EVENT_RUN_COMPLETED,
    EVENT_RUN_FAILED,
    EVENT_RUN_STARTED,
    EVENT_RUN_STOPPED,
    STOP_REASON_CANCELLED,
    STOP_REASON_INTERRUPTED,
    RunService,
    RunServiceUnavailable,
    RunSubmitError,
    cap_event_payload,
    json_size,
    normalize_name,
    normalize_options,
)
from app.core.run_store import (
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
    RunProvenance,
    RunStore,
)

# ── test nodes ────────────────────────────────────────────────────────────


class _RunSlowNode(BaseNode):
    """Blocks the worker thread for ``seconds`` — the cancel/shutdown probe."""

    NODE_NAME = "_RunSlow"
    CATEGORY = "Test"
    DESCRIPTION = "Sleeps, then passes through"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [ParamDefinition(name="seconds", param_type=ParamType.FLOAT,
                                default=0.3)]

    def execute(self, inputs: dict[str, Any],
                params: dict[str, Any]) -> dict[str, Any]:
        time.sleep(float(params.get("seconds", 0.3)))
        return {"value": inputs.get("value")}


class _RunMetricsNode(BaseNode):
    """Emits progress payloads shaped like the training loop's epochs."""

    NODE_NAME = "_RunMetrics"
    CATEGORY = "Test"
    DESCRIPTION = "Emits epoch progress events"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [ParamDefinition(name="epochs", param_type=ParamType.INT,
                                default=3)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any],
                progress_callback=None) -> dict[str, Any]:
        epochs = int(params.get("epochs", 3))
        for epoch in range(1, epochs + 1):
            if progress_callback:
                progress_callback({
                    "event": "epoch",
                    "epoch": epoch,
                    "total_epochs": epochs,
                    "loss": 1.0 / epoch,
                    "accuracy": 0.5 + epoch / 100,
                    "label": "not-a-metric",
                    "converged": False,
                })
        return {"value": inputs.get("value")}


class _RunNanNode(BaseNode):
    """Emits a diverged (NaN) scalar — the JSON-token hazard."""

    NODE_NAME = "_RunNan"
    CATEGORY = "Test"
    DESCRIPTION = "Emits a NaN progress scalar"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any],
                progress_callback=None) -> dict[str, Any]:
        if progress_callback:
            progress_callback({"event": "epoch", "epoch": 1,
                               "loss": float("nan")})
        return {"value": inputs.get("value")}


class _RunImageNode(BaseNode):
    """DECLARES an image port and returns a base64 blob of a given size."""

    NODE_NAME = "_RunImage"
    CATEGORY = "Test"
    DESCRIPTION = "Emits a declared base64 image"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="value", data_type=DataType.ANY),
            PortDefinition(name="image", data_type=DataType.ANY,
                           media=MEDIA_IMAGE),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [ParamDefinition(name="size", param_type=ParamType.INT,
                                default=64)]

    def execute(self, inputs: dict[str, Any],
                params: dict[str, Any]) -> dict[str, Any]:
        return {"value": inputs.get("value"),
                "image": "A" * int(params.get("size", 64))}


class _RunBoomNode(BaseNode):
    """Raises — drives the failed-run path."""

    NODE_NAME = "_RunBoom"
    CATEGORY = "Test"
    DESCRIPTION = "Raises RuntimeError"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs: dict[str, Any],
                params: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom: intentional run-service test failure")


_TEST_NODES = {
    "_RunSlow": _RunSlowNode,
    "_RunMetrics": _RunMetricsNode,
    "_RunNan": _RunNanNode,
    "_RunImage": _RunImageNode,
    "_RunBoom": _RunBoomNode,
}


@pytest.fixture(autouse=True)
def _register_test_nodes():
    registry._nodes.update(_TEST_NODES)
    yield
    for name in _TEST_NODES:
        registry._nodes.pop(name, None)


# ── fixtures / helpers ────────────────────────────────────────────────────


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
async def service(store):
    svc = RunService(store, shutdown_grace_s=2.0)
    try:
        yield svc
    finally:
        await svc.shutdown()


def _graph(middle_type: str = "_TestSource", *,
           params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Start -> <middle> -> Print, with a passthrough source in front.

    ``_TestSource`` has no inputs, so it doubles as the value producer for
    middles that take one.
    """
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "src", "type": "_TestSource", "data": {"params": {"val": "hi"}}},
        {"id": "print", "type": "Print", "data": {"params": {"label": "out"}}},
    ]
    edges = [
        {"id": "et", "source": "start", "target": "src",
         "sourceHandle": "trigger", "type": "trigger"},
    ]
    if middle_type == "_TestSource":
        edges.append({"id": "e1", "source": "src", "target": "print",
                      "sourceHandle": "value", "targetHandle": "value"})
    else:
        nodes.insert(2, {"id": "mid", "type": middle_type,
                         "data": {"params": params or {}}})
        edges.append({"id": "e1", "source": "src", "target": "mid",
                      "sourceHandle": "value", "targetHandle": "value"})
        edges.append({"id": "e2", "source": "mid", "target": "print",
                      "sourceHandle": "value", "targetHandle": "value"})
    return {"nodes": nodes, "edges": edges}


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


# ── acceptance criterion 1: the run outlives its submitter ────────────────


async def test_run_survives_the_submitter_going_away(store, service):
    """Submit, drop every reference the caller held, run still completes.

    The REST-level version of this (kill the HTTP client) lives in
    test_api_runs.py; this is the service-level guarantee it rests on:
    nothing about the run's lifetime is tied to the submitting coroutine.
    """
    submitted = await service.submit(_graph())
    run_id = submitted.run_id
    assert submitted.status == STATUS_RUNNING

    # Simulate the submitter disappearing: the task that called submit()
    # ends here and holds nothing. Only the service owns the run.
    del submitted

    record = await _await_terminal(store, run_id)
    assert record.status == STATUS_SUCCEEDED
    assert record.error is None
    assert record.started_at is not None

    events = await store.get_events(run_id)
    assert [e.type for e in events][0] == EVENT_RUN_STARTED
    assert [e.type for e in events][-1] == EVENT_RUN_COMPLETED
    assert [e.cursor for e in events] == list(range(1, len(events) + 1))


async def test_submitting_task_cancellation_does_not_kill_the_run(store, service):
    """Cancelling the CALLER after submit() must not touch the run.

    This is the exact failure mode of the WS handler (a disconnect cancels
    the graph task); the run service owns the task itself.
    """
    holder: dict[str, str] = {}

    async def submitter() -> None:
        result = await service.submit(_graph("_RunSlow", params={"seconds": 0.2}))
        holder["run_id"] = result.run_id
        await asyncio.sleep(30)  # still "connected" when we cancel it

    task = asyncio.create_task(submitter())
    while "run_id" not in holder:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    record = await _await_terminal(store, holder["run_id"])
    assert record.status == STATUS_SUCCEEDED


# ── acceptance criterion 2: concurrent runs are independent ───────────────


async def test_two_concurrent_runs_do_not_cancel_each_other(store, service):
    """Regression vs ws_execution.py:171-175 (a second execute killed the first)."""
    first, second = await asyncio.gather(
        service.submit(_graph("_RunSlow", params={"seconds": 0.25})),
        service.submit(_graph("_RunSlow", params={"seconds": 0.05})),
    )
    assert first.run_id != second.run_id

    records = await asyncio.gather(
        _await_terminal(store, first.run_id),
        _await_terminal(store, second.run_id),
    )
    assert [r.status for r in records] == [STATUS_SUCCEEDED, STATUS_SUCCEEDED]

    # Event logs are per run and neither is empty or cross-contaminated.
    for record in records:
        events = await store.get_events(record.id)
        assert {e.run_id for e in events} == {record.id}
        assert events[-1].type == EVENT_RUN_COMPLETED


async def test_cancelling_one_run_leaves_the_other_alone(store, service):
    victim = await service.submit(_graph("_RunSlow", params={"seconds": 0.4}))
    survivor = await service.submit(_graph("_RunSlow", params={"seconds": 0.4}))

    outcome = await service.cancel(victim.run_id)
    assert outcome is not None and outcome.cancelled is True

    dead = await _await_terminal(store, victim.run_id)
    alive = await _await_terminal(store, survivor.run_id)
    assert dead.status == STATUS_CANCELLED
    assert alive.status == STATUS_SUCCEEDED


# ── acceptance criterion 3: restart honesty ───────────────────────────────


async def test_restart_marks_running_rows_interrupted(store):
    """A row left ``running`` by a crashed process is retired at startup."""
    orphan = await store.create_run(graph_snapshot=_graph(), options={},
                                    provenance=RunProvenance())
    await store.mark_running(orphan.id)
    assert (await store.get_run(orphan.id)).status == STATUS_RUNNING

    # A brand-new process comes up over the same database.
    fresh = RunService(store)
    interrupted = await fresh.recover_interrupted()
    assert interrupted == 1

    row = await store.get_run(orphan.id)
    assert row.status == STATUS_INTERRUPTED
    assert row.finished_at is not None
    # Idempotent: a second boot finds nothing left to retire.
    assert await fresh.recover_interrupted() == 0


async def test_recovery_refuses_to_run_with_live_runs(store, service):
    """Recovery is a STARTUP call; running it mid-flight would falsify rows."""
    await service.submit(_graph("_RunSlow", params={"seconds": 0.2}))
    with pytest.raises(RuntimeError, match="active run"):
        await service.recover_interrupted()


async def test_recovery_runs_before_retention(store, tmp_path):
    """Interrupted rows become prunable — the ordering that bounds metrics.

    An orphaned ``running`` row is never eligible for retention (active runs
    are live state). Recovery first, prune second, is what lets an
    abandoned run's metrics/events actually go away.
    """
    orphan = await store.create_run(graph_snapshot=_graph(), options={},
                                    provenance=RunProvenance())
    await store.mark_running(orphan.id)
    await store.log_metric(orphan.id, "loss", 1.0, 1)

    svc = RunService(store, retention_keep_last=0)
    # Prune alone cannot touch it while it still claims to be running.
    assert await svc.prune_retention() == 0
    assert await svc.recover_interrupted() == 1
    assert await svc.prune_retention() == 1
    assert await store.get_run(orphan.id) is None
    assert await store.get_metrics(orphan.id) == []


# ── options normalization ─────────────────────────────────────────────────


def test_options_defaults():
    assert normalize_options(None) == {
        "device": "cpu", "seed": None, "record_outputs": False,
        "lane": "queued",
    }
    assert normalize_options({}) == normalize_options(None)


def test_options_accept_every_documented_key():
    assert normalize_options({
        "device": " CUDA ", "seed": 7, "record_outputs": True, "lane": "gpu",
    }) == {"device": "cuda", "seed": 7, "record_outputs": True, "lane": "gpu"}


@pytest.mark.parametrize("device", ["cpu", "auto", "cuda", "cuda:1", "mps",
                                    "mps:0", " CUDA:0 "])
def test_options_accept_the_known_device_vocabulary(device):
    assert normalize_options({"device": device})["device"] == device.strip().lower()


@pytest.mark.parametrize("bad", [
    {"devcie": "cpu"},              # key typo -> loud, not silently ignored
    {"device": "cudda"},            # VALUE typo -> equally loud
    {"device": "gpu"},
    {"device": "cuda:"},
    {"device": "cuda:x"},
    {"device": 3},
    {"device": ""},
    {"seed": -1},
    {"seed": 2 ** 32},
    {"seed": "abc"},
    {"seed": True},                 # a bool is not a seed
    {"lane": ""},
    {"lane": "x" * 65},
    {"record_outputs": "yes"},
    ["not", "a", "dict"],
])
def test_options_reject_bad_input(bad):
    with pytest.raises(RunSubmitError):
        normalize_options(bad)


def test_name_is_normalized_and_bounded():
    assert normalize_name(None) is None
    assert normalize_name("   ") is None          # blank means unnamed
    assert normalize_name("  sweep 3 ") == "sweep 3"
    assert normalize_name("x" * 64) == "x" * 64
    for bad in ("x" * 65, 7):
        with pytest.raises(RunSubmitError):
            normalize_name(bad)


async def test_submit_rejects_a_malformed_graph(service):
    for bad in (None, [], {"edges": []}, {"nodes": {}}, {"nodes": []},
                {"nodes": [{"id": "a"}], "edges": {}}):
        with pytest.raises(RunSubmitError):
            await service.submit(bad)


async def test_submit_persists_normalized_options_and_snapshot(store, service):
    graph = _graph()
    submitted = await service.submit(graph, options={"seed": 11}, name="demo")
    await _await_terminal(store, submitted.run_id)

    record = await store.get_run(submitted.run_id)
    assert record.name == "demo"
    assert record.options == {"device": "cpu", "seed": 11,
                              "record_outputs": False, "lane": "queued"}
    # queue_key carries the RESOLVED device (mark_running's contract).
    assert record.queue_key == "cpu"
    snapshot = await store.get_graph_snapshot(submitted.run_id)
    assert [n["id"] for n in snapshot["nodes"]] == [n["id"] for n in graph["nodes"]]
    assert snapshot["presets"] == []


async def test_run_id_is_also_the_execution_id(store, service):
    """One id format: RunStore's uuid4().hex, reused as the execution id."""
    submitted = await service.submit(_graph("_RunSlow", params={"seconds": 0.2}))
    active = service.active_run_ids()
    assert active == [submitted.run_id]
    assert service.execution_id(submitted.run_id) == submitted.run_id
    assert len(submitted.run_id) == 32 and "-" not in submitted.run_id
    await _await_terminal(store, submitted.run_id)


# ── events: persistence + fan-out ─────────────────────────────────────────


async def test_every_callback_lands_in_the_event_log(store, service):
    submitted = await service.submit(_graph())
    await _await_terminal(store, submitted.run_id)

    events = await store.get_events(submitted.run_id)
    types = [e.type for e in events]
    assert types[0] == EVENT_RUN_STARTED
    assert types[-1] == EVENT_RUN_COMPLETED
    assert types.count(EVENT_NODE_STATUS) >= 4  # running+completed per node

    statuses = {(e.payload["node_id"], e.payload["status"])
                for e in events if e.type == EVENT_NODE_STATUS}
    assert ("print", "running") in statuses
    assert ("print", "completed") in statuses

    # #117 structured outputs ride along, not a re-sniffed heuristic.
    completed = [e for e in events if e.type == EVENT_NODE_STATUS
                 and e.payload["status"] == "completed"
                 and e.payload["node_id"] == "print"][0]
    kinds = {entry["output_kind"] for entry in completed.payload["outputs"]}
    assert kinds == {"text", "tensor_summary"}


async def test_failure_is_recorded_on_the_row_and_in_the_log(store, service):
    submitted = await service.submit(_graph("_RunBoom"))
    record = await _await_terminal(store, submitted.run_id)

    assert record.status == STATUS_FAILED
    assert "boom" in record.error
    events = await store.get_events(submitted.run_id)
    assert events[-1].type == EVENT_RUN_FAILED
    assert "boom" in events[-1].payload["error"]


async def test_cancel_records_a_stop_reason(store, service):
    submitted = await service.submit(_graph("_RunSlow", params={"seconds": 0.3}))
    await service.cancel(submitted.run_id)
    record = await _await_terminal(store, submitted.run_id)

    assert record.status == STATUS_CANCELLED
    events = await store.get_events(submitted.run_id)
    assert events[-1].type == EVENT_RUN_STOPPED
    assert events[-1].payload == {"reason": STOP_REASON_CANCELLED}


async def test_cancel_is_honest_about_unknown_and_finished_runs(store, service):
    assert await service.cancel("no-such-run") is None

    submitted = await service.submit(_graph())
    await _await_terminal(store, submitted.run_id)
    outcome = await service.cancel(submitted.run_id)
    assert outcome is not None
    assert outcome.cancelled is False
    assert outcome.status == STATUS_SUCCEEDED


async def test_cancel_of_a_never_started_run_uses_the_guarded_write(store, service):
    """A queued row (the #123 lane) is retired directly, race-guarded."""
    queued = await store.create_run(graph_snapshot=_graph(), options={},
                                    provenance=RunProvenance())
    outcome = await service.cancel(queued.id)
    assert outcome.cancelled is True
    assert outcome.status == STATUS_CANCELLED
    assert (await store.get_run(queued.id)).status == STATUS_CANCELLED
    # Second cancel is a no-op, not a status rewrite.
    again = await service.cancel(queued.id)
    assert again.cancelled is False
    assert (await store.get_run(queued.id)).status == STATUS_CANCELLED


async def test_subscribers_see_events_live(store, service):
    submitted = await service.submit(_graph("_RunSlow", params={"seconds": 0.2}))
    seen: list[str] = []
    async with service.subscribe(submitted.run_id) as subscription:
        while True:
            event = await asyncio.wait_for(subscription.queue.get(), timeout=10)
            if event is None:          # sentinel: the run is over
                break
            seen.append(event.type)
    assert EVENT_NODE_STATUS in seen
    assert seen[-1] == EVENT_RUN_COMPLETED
    assert subscription.closed is True


async def test_subscriber_backpressure_drops_instead_of_stalling(store, service):
    """A slow subscriber must never block the run; it re-reads the tail."""
    submitted = await service.submit(_graph())
    async with service.subscribe(submitted.run_id, maxsize=1) as subscription:
        record = await _await_terminal(store, submitted.run_id)
        assert record.status == STATUS_SUCCEEDED
        assert subscription.take_dropped() > 0
        assert subscription.take_dropped() == 0     # counter resets
        # Drop-tolerance contract: the store still holds every event.
        events = await store.get_events(submitted.run_id)
        assert events[-1].type == EVENT_RUN_COMPLETED


async def test_subscribe_to_a_finished_run_yields_a_closed_subscription(store, service):
    submitted = await service.submit(_graph())
    await _await_terminal(store, submitted.run_id)
    async with service.subscribe(submitted.run_id) as subscription:
        assert subscription.closed is True


async def test_wait_for_events_paginates_by_cursor(store, service):
    submitted = await service.submit(_graph())
    await _await_terminal(store, submitted.run_id)

    everything = await service.wait_for_events(submitted.run_id)
    assert len(everything) >= 5
    tail = await service.wait_for_events(submitted.run_id,
                                         after_cursor=everything[1].cursor)
    assert [e.cursor for e in tail] == [e.cursor for e in everything[2:]]
    assert await service.wait_for_events(
        submitted.run_id, after_cursor=everything[-1].cursor) == []


async def test_long_poll_returns_immediately_for_a_finished_run(store, service):
    submitted = await service.submit(_graph())
    events = await _await_terminal(store, submitted.run_id) and \
        await store.get_events(submitted.run_id)
    started = time.monotonic()
    assert await service.wait_for_events(
        submitted.run_id, after_cursor=events[-1].cursor, wait=5.0) == []
    assert time.monotonic() - started < 2.0, "long poll hung on a dead run"


async def test_long_poll_wakes_on_the_next_event(store, service):
    submitted = await service.submit(_graph("_RunSlow", params={"seconds": 0.3}))
    first = await service.wait_for_events(submitted.run_id, wait=5.0)
    assert first, "no events at all"

    started = time.monotonic()
    later = await service.wait_for_events(
        submitted.run_id, after_cursor=first[-1].cursor, wait=10.0)
    elapsed = time.monotonic() - started
    assert later, "long poll timed out instead of waking"
    assert elapsed < 9.0, "woke on the timeout, not on the event"
    await _await_terminal(store, submitted.run_id)


# ── event payload bounds ──────────────────────────────────────────────────


def test_cap_leaves_ordinary_payloads_untouched():
    payload = {"node_id": "n", "status": "completed", "outputs": [
        {"output_kind": "text", "text": "hello"},
        {"output_kind": "tensor_summary", "tensor_summary": {"v": [1, 2, 3]}},
    ]}
    assert cap_event_payload(payload, cap_bytes=64 * 1024) is payload
    assert cap_event_payload(payload, cap_bytes=0) is payload      # disabled


def test_cap_elides_an_oversized_entry_but_keeps_its_identity():
    big = {"output_kind": "image", "port": "image",
           "image": {"format": "png", "encoding": "base64", "data": "A" * 5000}}
    small = {"output_kind": "text", "text": "kept"}
    capped = cap_event_payload(
        {"node_id": "n", "status": "completed", "outputs": [big, small]},
        cap_bytes=2000)

    assert json_size(capped) <= 2000
    elided, intact = capped["outputs"]
    # Enough identity survives for a UI to draw a placeholder in the right
    # slot, and nothing pretends to still be an image payload.
    assert elided["output_kind"] == "image" and elided["port"] == "image"
    assert elided["elided"] is True and elided["bytes"] > 5000
    assert "image" not in elided
    assert intact == small                       # the small one is untouched
    assert capped["node_id"] == "n"              # envelope survives


def test_cap_falls_back_to_a_marker_when_there_is_nothing_to_trim():
    capped = cap_event_payload({"error": "x" * 5000, "run_id": "r"},
                               cap_bytes=500)
    assert capped["elided"] is True
    assert capped["run_id"] == "r"               # identity kept
    assert "error" not in capped
    assert json_size(capped) <= 500


async def test_an_oversized_image_event_is_stored_elided(store):
    svc = RunService(store, event_payload_cap_bytes=2000)
    try:
        submitted = await svc.submit(_graph("_RunImage",
                                            params={"size": 20000}))
        await _await_terminal(store, submitted.run_id)
        events = await store.get_events(submitted.run_id)
        images = [entry
                  for event in events if event.type == EVENT_NODE_STATUS
                  for entry in (event.payload.get("outputs") or [])
                  if entry.get("output_kind") == "image"]
        assert images, "the declared image port produced no entry at all"
        assert all(entry.get("elided") is True for entry in images)
        assert all(json_size(event.payload) <= 2000 for event in events)
    finally:
        await svc.shutdown()


async def test_a_small_image_event_is_stored_intact(store, service):
    submitted = await service.submit(_graph("_RunImage", params={"size": 64}))
    await _await_terminal(store, submitted.run_id)
    events = await store.get_events(submitted.run_id)
    images = [entry
              for event in events if event.type == EVENT_NODE_STATUS
              for entry in (event.payload.get("outputs") or [])
              if entry.get("output_kind") == "image"]
    assert len(images) == 1
    assert images[0]["image"]["data"] == "A" * 64
    assert "elided" not in images[0]


async def test_non_finite_scalars_never_reach_a_subscriber(store, service):
    """A NaN on the wire is a token JSON.parse rejects — it must not fan out."""
    submitted = await service.submit(_graph("_RunNan"))
    received: list[Any] = []
    async with service.subscribe(submitted.run_id) as subscription:
        while True:
            event = await asyncio.wait_for(subscription.queue.get(), timeout=10)
            if event is None:
                break
            received.append(event)

    progress = [entry
                for event in received if event.type == EVENT_NODE_STATUS
                for entry in (event.payload.get("outputs") or [])
                if entry.get("output_kind") == "progress"]
    assert progress, "no progress event was fanned out"
    assert progress[0]["progress"]["loss"] is None

    # And the stored copy says exactly the same thing.
    stored = [entry
              for event in await store.get_events(submitted.run_id)
              for entry in ((event.payload or {}).get("outputs") or [])
              if entry.get("output_kind") == "progress"]
    assert stored[0]["progress"]["loss"] is None
    # The METRIC still records the point at its step, as a NULL value, so a
    # chart shows the break where the training diverged.
    points = await store.get_metrics(submitted.run_id, name="loss")
    assert [(p.step, p.value) for p in points] == [(1, None)]


# ── metrics ───────────────────────────────────────────────────────────────


async def test_progress_scalars_land_in_exec_run_metrics(store, service):
    submitted = await service.submit(_graph("_RunMetrics",
                                            params={"epochs": 3}))
    await _await_terminal(store, submitted.run_id)

    names = await store.list_metric_names(submitted.run_id)
    assert names == ["accuracy", "loss"]     # label/converged are not metrics
    loss = await store.get_metrics(submitted.run_id, name="loss")
    assert [p.step for p in loss] == [1, 2, 3]
    assert loss[0].value == pytest.approx(1.0)
    assert loss[2].value == pytest.approx(1 / 3)
    assert {p.node_id for p in loss} == {"mid"}
    # Structural keys never become series.
    assert "epoch" not in names and "total_epochs" not in names


async def test_metrics_are_written_in_batches(store, service, monkeypatch):
    """One transaction per flush, never one per point (#119 carry-forward)."""
    calls: list[int] = []
    original = RunStore.log_metrics

    async def counting(self, run_id, points):
        batch = list(points)
        calls.append(len(batch))
        return await original(self, run_id, batch)

    monkeypatch.setattr(RunStore, "log_metrics", counting)
    monkeypatch.setattr(RunStore, "log_metric", None, raising=False)

    submitted = await service.submit(_graph("_RunMetrics",
                                            params={"epochs": 6}))
    await _await_terminal(store, submitted.run_id)

    assert calls, "no metric flush happened at all"
    assert sum(calls) == 12          # 6 epochs x {loss, accuracy}
    assert max(calls) > 1, "points were flushed one at a time"


# ── retention + shutdown ──────────────────────────────────────────────────


async def test_retention_prunes_after_each_terminal_run(store):
    svc = RunService(store, retention_keep_last=2)
    try:
        ids = []
        for _ in range(4):
            submitted = await svc.submit(_graph())
            await _await_terminal(store, submitted.run_id)
            ids.append(submitted.run_id)
        # The prune is the last step of the finishing task, which can still
        # be a tick behind the row write _await_terminal watches for.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            remaining = {r.id for r in await store.list_runs()}
            if len(remaining) <= 2:
                break
            await asyncio.sleep(0.02)
        assert remaining == set(ids[-2:])
    finally:
        await svc.shutdown()


async def test_retention_is_off_when_keep_last_is_none(store):
    svc = RunService(store, retention_keep_last=None)
    try:
        for _ in range(3):
            submitted = await svc.submit(_graph())
            await _await_terminal(store, submitted.run_id)
        assert len(await store.list_runs()) == 3
        assert await svc.prune_retention() == 0
    finally:
        await svc.shutdown()


async def test_shutdown_drains_runs_before_the_database_closes(db, store):
    """The #119 carry-forward: no DB work may be in flight at db.close().

    A run stopped by shutdown is ``interrupted`` (the server went away), not
    ``cancelled`` (a user asked) — the two are different facts.
    """
    svc = RunService(store, shutdown_grace_s=10.0)
    submitted = await svc.submit(_graph("_RunSlow", params={"seconds": 0.3}))
    await asyncio.sleep(0.05)

    await svc.shutdown()
    assert svc.active_run_ids() == []

    # Everything settled BEFORE this point: closing now must not orphan a
    # write, and the row must already be terminal.
    record = await store.get_run(submitted.run_id)
    assert record.status == STATUS_INTERRUPTED
    assert record.finished_at is not None
    events = await store.get_events(submitted.run_id)
    assert events[-1].type == EVENT_RUN_STOPPED
    assert events[-1].payload == {"reason": STOP_REASON_INTERRUPTED}
    db.close()


async def test_shutdown_keeps_an_earlier_cancel_reason(db, store):
    """Cancel then shut down: the user's intent came first and stands."""
    svc = RunService(store, shutdown_grace_s=10.0)
    submitted = await svc.submit(_graph("_RunSlow", params={"seconds": 0.3}))
    await asyncio.sleep(0.05)
    await svc.cancel(submitted.run_id)
    await svc.shutdown()

    record = await store.get_run(submitted.run_id)
    assert record.status == STATUS_CANCELLED
    events = await store.get_events(submitted.run_id)
    assert events[-1].payload == {"reason": STOP_REASON_CANCELLED}
    db.close()


async def test_shutdown_hard_cancels_after_the_grace_period(db, store):
    """A run that ignores the flag is force-cancelled; the row stays honest.

    With no terminal write, the row is still ``running`` — which the NEXT
    startup's recovery converts to ``interrupted``. That is the design: a
    lie is never written, and the truth arrives one boot later.
    """
    svc = RunService(store, shutdown_grace_s=0.05)
    submitted = await svc.submit(_graph("_RunSlow", params={"seconds": 1.5}))
    await asyncio.sleep(0.05)
    await svc.shutdown()

    assert svc.active_run_ids() == []
    record = await store.get_run(submitted.run_id)
    assert record.status in (STATUS_RUNNING, STATUS_INTERRUPTED)
    if record.status == STATUS_RUNNING:
        assert await svc.recover_interrupted() == 1
        assert (await store.get_run(submitted.run_id)).status == STATUS_INTERRUPTED
    db.close()


async def test_submit_is_refused_after_shutdown(store):
    svc = RunService(store)
    await svc.shutdown()
    with pytest.raises(RunServiceUnavailable, match="shutting down"):
        await svc.submit(_graph())


async def test_submit_racing_shutdown_cannot_escape_the_drain(
    store, monkeypatch,
):
    """A submit parked in create_run when shutdown starts must be refused.

    Otherwise it resumes AFTER shutdown snapshotted the registry, registers
    a task nobody drains, and the lifespan closes the database underneath it
    — precisely the race this whole issue exists to close.
    """
    svc = RunService(store)
    reached_db = asyncio.Event()
    release = asyncio.Event()
    original = RunStore.create_run

    async def stalling_create(self, **kwargs):
        record = await original(self, **kwargs)
        reached_db.set()
        await release.wait()
        return record

    monkeypatch.setattr(RunStore, "create_run", stalling_create)
    task = asyncio.create_task(svc.submit(_graph()))
    await asyncio.wait_for(reached_db.wait(), timeout=5)

    await svc.shutdown()                 # snapshots a registry without it
    release.set()
    with pytest.raises(RunServiceUnavailable):
        await task

    assert svc.active_run_ids() == []
    rows = await store.list_runs()
    assert [row.status for row in rows] == ["queued"]
    assert await svc.recover_interrupted() == 1


async def test_cancel_during_finalize_does_not_claim_a_cancel(
    store, service, monkeypatch,
):
    """Once the outcome is decided, a cancel must not report that it won."""
    in_finalize = asyncio.Event()
    release = asyncio.Event()
    original = RunService._flush_metrics

    async def stalling_flush(self, active, *, force=False):
        if force:                        # the terminal flush, once per run
            in_finalize.set()
            await release.wait()
        return await original(self, active, force=force)

    monkeypatch.setattr(RunService, "_flush_metrics", stalling_flush)
    submitted = await service.submit(_graph())
    await asyncio.wait_for(in_finalize.wait(), timeout=10)

    outcome = await service.cancel(submitted.run_id)
    assert outcome is not None and outcome.cancelled is False

    release.set()
    record = await _await_terminal(store, submitted.run_id)
    assert record.status == STATUS_SUCCEEDED


async def test_a_cancelled_submit_never_strands_a_registry_entry(
    store, service, monkeypatch,
):
    """Cancel the submitter mid-``create_run``: nothing may be half-owned.

    The other half of this invariant needs no test because it is structural:
    ``_start`` is synchronous, so there is no await between registering a run
    and creating the task that owns it, and no cancellation can land there.
    This covers the window that DOES exist — the persist — where the honest
    outcome is a durable ``queued`` row that startup recovery retires.
    """
    reached_db = asyncio.Event()
    original = RunStore.create_run

    async def stalling_create(self, **kwargs):
        record = await original(self, **kwargs)
        reached_db.set()
        await asyncio.sleep(30)          # the submitter is parked here
        return record

    monkeypatch.setattr(RunStore, "create_run", stalling_create)
    task = asyncio.create_task(service.submit(_graph()))
    await asyncio.wait_for(reached_db.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert service.active_run_ids() == []
    rows = await store.list_runs()
    assert [row.status for row in rows] == ["queued"]
    assert await service.recover_interrupted() == 1


async def test_shutdown_is_idempotent(store, service):
    await service.shutdown()
    await service.shutdown()
    assert service.active_run_ids() == []
