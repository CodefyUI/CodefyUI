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

from app.core.cache import ExecutionCache
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
from app.core.seeding import derive_seed
from app.core.run_service import (
    EVENT_ARTIFACT,
    EVENT_METRIC,
    EVENT_NODE_STATUS,
    EVENT_RUN_COMPLETED,
    EVENT_RUN_FAILED,
    EVENT_RUN_STARTED,
    EVENT_RUN_STOPPED,
    EVENT_WARNING,
    LANE_INTERACTIVE,
    OPTION_KEYS,
    STOP_REASON_CANCELLED,
    STOP_REASON_INTERRUPTED,
    TRUNCATION_MARKER,
    InteractiveSession,
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


class _RunLogMetricsNode(BaseNode):
    """Uses the #122 node-facing API: log_metric, log_artifact, batch frames.

    Shaped exactly like ``TrainingLoopNode``: a config frame first, then per
    epoch an explicit ``train_loss``/``val_loss`` followed by an epoch
    progress payload that ALSO carries ``loss``/``val_loss`` for the live
    chart. Whether those two paths double-file the same point is the thing
    the tests below exist to pin.
    """

    NODE_NAME = "_RunLogMetrics"
    CATEGORY = "Test"
    DESCRIPTION = "Logs explicit metrics and an artifact"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="epochs", param_type=ParamType.INT, default=3),
            ParamDefinition(name="artifact", param_type=ParamType.BOOL,
                            default=False),
            ParamDefinition(name="batches", param_type=ParamType.INT, default=0),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any],
                progress_callback=None, *, context=None) -> dict[str, Any]:
        epochs = int(params.get("epochs", 3))
        if progress_callback:
            progress_callback({"event": "config", "config": {"epochs": epochs}})
        for epoch in range(1, epochs + 1):
            for batch in range(1, int(params.get("batches", 0)) + 1):
                if progress_callback:
                    progress_callback({
                        "event": "batch", "epoch": epoch, "batch": batch,
                        "total_batches": int(params.get("batches", 0)),
                        "loss": 1.0 / (epoch * batch),
                    })
            if context is not None:
                context.log_metric("train_loss", 1.0 / epoch, epoch)
                context.log_metric("val_loss", 2.0 / epoch, epoch)
            if progress_callback:
                progress_callback({"event": "epoch", "epoch": epoch,
                                   "total_epochs": epochs,
                                   "loss": 1.0 / epoch, "val_loss": 2.0 / epoch})
        if params.get("artifact") and context is not None:
            context.log_artifact("checkpoint", "interrupted/fake.pt",
                                 {"reason": "interrupted", "epoch": epochs})
        return {"value": inputs.get("value")}


class _RunFloodNode(BaseNode):
    """Logs far more points than a tiny outbox can hold — the overflow probe."""

    NODE_NAME = "_RunFlood"
    CATEGORY = "Test"
    DESCRIPTION = "Overruns the outbox"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [ParamDefinition(name="points", param_type=ParamType.INT,
                                default=64)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any],
                *, context=None) -> dict[str, Any]:
        for step in range(1, int(params.get("points", 64)) + 1):
            context.log_metric("loss", 1.0 / step, step)
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

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        # Pads the message, for the "an enormous error is still readable"
        # path in cap_event_payload.
        return [ParamDefinition(name="size", param_type=ParamType.INT,
                                default=0)]

    def execute(self, inputs: dict[str, Any],
                params: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom: intentional run-service test failure"
                           + "x" * int(params.get("size", 0)))


_TEST_NODES = {
    "_RunSlow": _RunSlowNode,
    "_RunMetrics": _RunMetricsNode,
    "_RunLogMetrics": _RunLogMetricsNode,
    "_RunFlood": _RunFloodNode,
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


#: What ``normalize_options`` fills in when the caller says nothing. Note
#: ``weights_persistent`` is False here while ``ExecutionContext``'s own
#: default is True: a server-owned run is isolated unless someone asks.
_DEFAULT_OPTIONS = {
    "device": "cpu", "seed": None, "lane": "queued", "graph_id": "",
    "error_mode": "fail_fast", "max_retries": 0, "record_outputs": False,
    "verbose": False, "weights_persistent": False, "backward_mode": False,
    "auto_backward": False, "deterministic": False,
}


def test_options_defaults():
    assert normalize_options(None) == _DEFAULT_OPTIONS
    assert normalize_options({}) == normalize_options(None)


def test_options_accept_every_documented_key():
    """Every key of the vocabulary round-trips; none is silently dropped."""
    asked = {
        "device": " CUDA ", "seed": 7, "record_outputs": True, "lane": "gpu",
        "verbose": True, "graph_id": "canvas-1", "weights_persistent": True,
        "backward_mode": True, "auto_backward": True,
        "error_mode": "retry", "max_retries": 3, "deterministic": True,
    }
    assert set(asked) == OPTION_KEYS, "the vocabulary grew without a test"
    assert normalize_options(asked) == {**asked, "device": "cuda"}


@pytest.mark.parametrize("lane", [LANE_INTERACTIVE, "queued", "gpu"])
def test_lane_is_carried_through_verbatim(lane):
    assert normalize_options({"lane": lane})["lane"] == lane


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
    {"verbose": "yes"},
    {"weights_persistent": 1},      # an int is not a bool
    {"backward_mode": None},
    {"auto_backward": "true"},
    {"graph_id": 7},
    {"graph_id": "x" * 129},
    {"error_mode": "fail-fast"},    # VALUE typo -> loud, not "continue"
    {"error_mode": None},
    {"max_retries": -1},
    {"max_retries": 11},
    {"max_retries": True},          # a bool is not a count
    {"max_retries": "3"},
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
    assert record.options == {**_DEFAULT_OPTIONS, "seed": 11}
    # queue_key carries the RESOLVED device (mark_running's contract).
    assert record.queue_key == "cpu"
    snapshot = await store.get_graph_snapshot(submitted.run_id)
    assert [n["id"] for n in snapshot["nodes"]] == [n["id"] for n in graph["nodes"]]
    assert snapshot["presets"] == []


# ── the interactive lane (#121) ───────────────────────────────────────────


async def test_a_session_is_refused_on_any_other_lane(service):
    """Statefulness must stay answerable from the stored row.

    A session on a ``queued`` run would mean a row that says "isolated"
    driving an execution that shares an nn.Module with a browser tab.
    """
    with pytest.raises(RunSubmitError) as excinfo:
        await service.submit(_graph(), options={"lane": "queued"},
                             session=InteractiveSession())
    assert LANE_INTERACTIVE in str(excinfo.value)


async def test_a_server_owned_run_is_isolated_even_when_it_asks_not_to_be(
    store, service,
):
    """``weights_persistent`` alone cannot make a queued run stateful.

    Isolation is a capability question, not a flag question: without a
    session there is no module store, so ``StatefulModuleMixin`` rebuilds
    every module regardless of what the options say.
    """
    submitted = await service.submit(
        _graph(), options={"weights_persistent": True})
    assert service._runs[submitted.run_id].context.node_state_store is None
    await _await_terminal(store, submitted.run_id)


async def test_the_interactive_lane_lends_its_cache_to_the_run(store, service):
    """Two submits sharing one cache: the second is served from it.

    This is the canvas's editing loop — hit Run, tweak one node, hit Run
    again and only the changed subtree re-executes — expressed at the level
    the service actually implements it.
    """
    cache = ExecutionCache()
    options = {"lane": LANE_INTERACTIVE}

    first = await service.submit(_graph(), options=options,
                                 session=InteractiveSession(cache=cache))
    await _await_terminal(store, first.run_id)
    second = await service.submit(_graph(), options=options,
                                  session=InteractiveSession(cache=cache))
    await _await_terminal(store, second.run_id)

    def statuses(run_events):
        return {(e.payload["node_id"], e.payload["status"])
                for e in run_events if e.type == EVENT_NODE_STATUS}

    assert ("src", "completed") in statuses(await store.get_events(first.run_id))
    assert ("src", "cached") in statuses(await store.get_events(second.run_id))


async def test_changed_nodes_forces_a_re_run_past_the_cache(store, service):
    cache = ExecutionCache()
    options = {"lane": LANE_INTERACTIVE}
    first = await service.submit(_graph(), options=options,
                                 session=InteractiveSession(cache=cache))
    await _await_terminal(store, first.run_id)
    second = await service.submit(
        _graph(), options=options,
        session=InteractiveSession(cache=cache, changed_nodes=("src",)))
    await _await_terminal(store, second.run_id)

    statuses = {e.payload["status"]
                for e in await store.get_events(second.run_id)
                if e.type == EVENT_NODE_STATUS and e.payload["node_id"] == "src"}
    assert "cached" not in statuses
    assert "completed" in statuses


async def test_error_mode_reaches_the_engine(store, service):
    """``continue`` finishes the run instead of failing it at the first node."""
    submitted = await service.submit(_graph("_RunBoom"),
                                     options={"error_mode": "continue"})
    record = await _await_terminal(store, submitted.run_id)
    assert record.status == STATUS_SUCCEEDED
    statuses = {(e.payload["node_id"], e.payload["status"])
                for e in await store.get_events(submitted.run_id)
                if e.type == EVENT_NODE_STATUS}
    assert ("mid", "error") in statuses     # the node still reported failure


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


def test_the_payload_cap_comes_from_the_setting(store):
    """ONE source for the ceiling — no module copy to drift out of step.

    #120 shipped an independent ``128 * 1024`` in this module alongside the
    setting, so raising ``CODEFYUI_RUN_EVENT_PAYLOAD_CAP_BYTES`` moved only
    the service the lifespan built. A default ARGUMENT would not have fixed
    it either: it is evaluated at import time.
    """
    from app.config import settings

    original = settings.RUN_EVENT_PAYLOAD_CAP_BYTES
    settings.RUN_EVENT_PAYLOAD_CAP_BYTES = 777
    try:
        assert RunService(store)._event_payload_cap_bytes == 777
        # An explicit argument still wins, for tests that need a tiny cap.
        assert RunService(store, event_payload_cap_bytes=5) \
            ._event_payload_cap_bytes == 5
    finally:
        settings.RUN_EVENT_PAYLOAD_CAP_BYTES = original


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


def test_cap_keeps_the_head_of_an_oversized_error_string():
    """#121: the marker fallback used to swallow the error text entirely.

    An ``execution_error`` carrying a 200 KB traceback has no ``outputs`` to
    elide, so it collapsed straight to the marker and the user got a red
    banner with NO message — on the one payload where the text IS the point.
    """
    capped = cap_event_payload(
        {"error": "RuntimeError: boom\n" + "x" * 5000, "run_id": "r"},
        cap_bytes=500)
    assert json_size(capped) <= 500
    assert capped["run_id"] == "r"
    assert "elided" not in capped, "the marker fallback should not be reached"
    assert capped["error"].startswith("RuntimeError: boom\n")
    assert capped["error"].endswith(TRUNCATION_MARKER)


def test_cap_spends_its_string_budget_where_it_is_needed():
    """Short strings survive intact; the giant one pays for the whole cap."""
    capped = cap_event_payload(
        {"status": "error", "node_id": "trainer", "error": "E" * 5000},
        cap_bytes=400)
    assert json_size(capped) <= 400
    assert capped["status"] == "error"
    assert capped["node_id"] == "trainer"
    assert capped["error"].endswith(TRUNCATION_MARKER)
    assert len(capped["error"]) > 100          # a useful head, not a stub


def test_cap_truncates_non_ascii_strings_by_BYTES_not_characters():
    """``ensure_ascii`` makes one CJK char cost six bytes on the wire.

    Slicing by character count would leave the payload over cap and drop it
    into the marker fallback — the exact failure this replaces, just harder
    to notice because it only fires for non-English error messages.
    """
    capped = cap_event_payload({"error": "錯誤" * 2000}, cap_bytes=500)
    assert json_size(capped) <= 500
    assert capped["error"].startswith("錯誤")
    assert capped["error"].endswith(TRUNCATION_MARKER)


def test_cap_falls_back_to_a_marker_when_there_is_nothing_to_trim():
    """No entries to elide, no strings to shorten: identity only."""
    capped = cap_event_payload(
        {"weights": [1.25] * 5000, "run_id": "r", "status": "completed"},
        cap_bytes=500)
    assert capped["elided"] is True
    assert capped["run_id"] == "r"               # identity kept
    assert capped["status"] == "completed"
    assert "weights" not in capped
    assert json_size(capped) <= 500


async def test_an_oversized_failure_keeps_its_message_in_the_log(store):
    """End to end: a run that fails enormously is still diagnosable."""
    svc = RunService(store, event_payload_cap_bytes=1000)
    try:
        submitted = await svc.submit(_graph("_RunBoom", params={"size": 9000}))
        await _await_terminal(store, submitted.run_id)
        events = await store.get_events(submitted.run_id)
        failure = next(e for e in events if e.type == EVENT_RUN_FAILED)
        assert failure.payload["error"].startswith("boom: intentional")
        assert failure.payload["error"].endswith(TRUNCATION_MARKER)
        assert json_size(failure.payload) <= 1000
    finally:
        await svc.shutdown()


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


# ── #122: what a node logs for itself ─────────────────────────────────────


async def test_explicit_metrics_land_and_are_not_double_filed(store, service):
    """``context.log_metric`` writes the series; inference stands down.

    The node reports ``val_loss`` twice over — once explicitly, once inside
    its epoch progress payload. Only the explicit one may become rows, or
    every validation loss in the product is stored twice.
    """
    submitted = await service.submit(_graph("_RunLogMetrics",
                                            params={"epochs": 3}))
    await _await_terminal(store, submitted.run_id)

    assert await store.list_metric_names(submitted.run_id) == [
        "train_loss", "val_loss"]
    val = await store.get_metrics(submitted.run_id, name="val_loss")
    assert [(p.step, round(p.value, 6)) for p in val] == [
        (1, 2.0), (2, 1.0), (3, round(2 / 3, 6))]
    assert {p.node_id for p in val} == {"mid"}


async def test_batch_progress_frames_are_not_mined_for_metrics(store, service):
    """A wall-clock-throttled frame is liveness, not a measurement."""
    submitted = await service.submit(
        _graph("_RunLogMetrics", params={"epochs": 2, "batches": 5}))
    await _await_terminal(store, submitted.run_id)

    names = await store.list_metric_names(submitted.run_id)
    assert names == ["train_loss", "val_loss"]
    # ...but the frames themselves are still in the durable event log, which
    # is what a live chart and a re-attaching client read.
    events = await store.get_events(submitted.run_id)
    frames = [p for e in events
              for entry in ((e.payload or {}).get("outputs") or [])
              if (p := entry.get("progress"))
              and p.get("event") == "batch"]
    assert [f["batch"] for f in frames] == [1, 2, 3, 4, 5] * 2


async def test_a_metric_event_is_emitted_per_flush(store, service):
    """One ``metric`` event per FLUSH, carrying the points that landed."""
    submitted = await service.submit(_graph("_RunLogMetrics",
                                            params={"epochs": 3}))
    await _await_terminal(store, submitted.run_id)

    events = [e for e in await store.get_events(submitted.run_id)
              if e.type == EVENT_METRIC]
    assert events, "no metric event was emitted"
    points = [p for e in events for p in e.payload["points"]]
    assert len(points) == 6                       # 3 epochs x 2 series
    assert {p["name"] for p in points} == {"train_loss", "val_loss"}
    assert all(p["node_id"] == "mid" for p in points)
    # Never one event per point on a batched path.
    assert len(events) < len(points)
    # And the terminal frame is still last.
    assert (await store.get_events(submitted.run_id))[-1].type == \
        EVENT_RUN_COMPLETED


async def test_log_artifact_registers_a_row_and_announces_it(store, service):
    submitted = await service.submit(
        _graph("_RunLogMetrics", params={"epochs": 1, "artifact": True}))
    await _await_terminal(store, submitted.run_id)

    artifacts = await store.list_artifacts(submitted.run_id, kind="checkpoint")
    assert len(artifacts) == 1
    assert artifacts[0].path == "interrupted/fake.pt"
    assert artifacts[0].meta == {"reason": "interrupted", "epoch": 1,
                                 "node_id": "mid"}

    announced = [e for e in await store.get_events(submitted.run_id)
                 if e.type == EVENT_ARTIFACT]
    assert len(announced) == 1
    assert announced[0].payload["artifact_id"] == artifacts[0].id
    assert announced[0].payload["path"] == "interrupted/fake.pt"


async def test_dropped_signals_are_reported_as_a_warning(store, monkeypatch):
    """Overflow sheds load and SAYS so; the run still succeeds.

    The outbox is shrunk by patching the context the service builds, rather
    than by adding a knob nobody would ever turn in production — capacity is
    a property of the queue, not a run option.
    """
    import app.core.run_service as run_service_module
    from app.core.execution_context import EventOutbox, ExecutionContext

    monkeypatch.setattr(
        run_service_module, "ExecutionContext",
        lambda **kwargs: ExecutionContext(outbox=EventOutbox(capacity=4),
                                          **kwargs),
    )
    svc = RunService(store, shutdown_grace_s=2.0)
    try:
        submitted = await svc.submit(_graph("_RunFlood",
                                            params={"points": 200}))
        await _await_terminal(store, submitted.run_id)
    finally:
        await svc.shutdown()

    record = await store.get_run(submitted.run_id)
    assert record is not None and record.status == STATUS_SUCCEEDED

    warnings = [e for e in await store.get_events(submitted.run_id)
                if e.type == EVENT_WARNING]
    assert warnings, "load was shed without telling anyone"
    dropped = sum(w.payload["count"] for w in warnings)
    kept = len(await store.get_metrics(submitted.run_id, name="loss"))
    assert dropped > 0
    assert dropped + kept == 200, "every point is delivered or accounted for"
    assert all(w.payload["kind"] == "dropped_signals" for w in warnings)


async def test_metrics_are_queryable_after_a_simulated_restart(
    store, service, tmp_path,
):
    """Persistence proof: a new process reads a finished run's series.

    The restart is real as far as this layer can make it — a second
    ``Database`` over the same file, a fresh ``RunStore`` and a fresh
    ``RunService`` whose startup recovery runs first, exactly as
    ``main.py``'s lifespan does.
    """
    submitted = await service.submit(
        _graph("_RunLogMetrics", params={"epochs": 4, "artifact": True}))
    await _await_terminal(store, submitted.run_id)

    reopened = Database(tmp_path / "codefyui.db")
    reopened.connect()
    try:
        restarted_store = RunStore(reopened)
        restarted = RunService(restarted_store)
        assert await restarted.recover_interrupted() == 0, (
            "a finished run must not be retired by startup recovery"
        )

        record = await restarted_store.get_run(submitted.run_id)
        assert record is not None and record.status == STATUS_SUCCEEDED
        assert await restarted_store.list_metric_names(submitted.run_id) == [
            "train_loss", "val_loss"]
        train = await restarted_store.get_metrics(submitted.run_id,
                                                  name="train_loss")
        assert [p.step for p in train] == [1, 2, 3, 4]
        assert train[0].value == pytest.approx(1.0)
        assert train[-1].value == pytest.approx(0.25)
        assert len(await restarted_store.list_artifacts(
            submitted.run_id, kind="checkpoint")) == 1
    finally:
        reopened.close()


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


# ── the seed reaches the CONTEXT, not just the row (#188 review, I2) ──────
#
# The single line in ``_start`` that passes `seed=` / `deterministic=` into
# ``ExecutionContext`` is what makes reproducibility real for every canvas
# and CLI run, and the review severed it at runtime with the whole suite
# staying green: every test either built an ``ExecutionContext`` by hand or
# asserted only on ``record.options``, which is the STORED value.
#
# These tests go through ``RunService.submit`` and observe what the executing
# node actually received, so cutting that wire fails them.

_SEED_PROBE = "_SeedProbe"


class _SeedProbeNode(BaseNode):
    """Reports the run-scoped seed state the ENGINE handed it."""

    NODE_NAME = _SEED_PROBE
    CATEGORY = "Utility"
    DESCRIPTION = "Test node: echoes the seed derived from its context."

    @classmethod
    def define_inputs(cls):
        # ``_graph`` wires src.value -> mid.value, so a middle needs the port.
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls):
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs, params, progress_callback=None, *, context=None):
        derived = context.derive_seed("probe") if context is not None else None
        if context is not None:
            # Logged as metrics because that is what survives to the store;
            # a server-owned run does not hand its outputs back.
            context.log_metric("probe_has_seed", 1.0 if derived else 0.0, 0)
            if derived is not None:
                context.log_metric("probe_seed", float(derived), 0)
            context.log_metric(
                "probe_deterministic",
                1.0 if getattr(context, "deterministic", False) else 0.0, 0)
        return {"value": inputs.get("value")}


@pytest.fixture
def _probe_node():
    registry._nodes[_SEED_PROBE] = _SeedProbeNode
    yield
    registry._nodes.pop(_SEED_PROBE, None)


async def _probe_metrics(service, store, options):
    result = await service.submit(_graph(_SEED_PROBE), options=options)
    await _await_terminal(store, result.run_id)
    points = await store.get_metrics(result.run_id)
    return {p.name: p.value for p in points}


@pytest.mark.asyncio
async def test_a_submitted_seed_reaches_the_executing_node(
    service, store, _probe_node,
):
    """Submit -> queue -> context -> node. Cut the wire and this fails."""
    metrics = await _probe_metrics(service, store, {"seed": 4242})

    assert metrics["probe_has_seed"] == 1.0
    assert metrics["probe_seed"] == float(derive_seed(4242, "probe"))


@pytest.mark.asyncio
async def test_two_submissions_with_one_seed_derive_the_same_node_seed(
    service, store, _probe_node,
):
    """The property a user cares about, observed through the real service."""
    first = await _probe_metrics(service, store, {"seed": 7})
    again = await _probe_metrics(service, store, {"seed": 7})
    other = await _probe_metrics(service, store, {"seed": 8})

    assert first["probe_seed"] == again["probe_seed"]
    assert first["probe_seed"] != other["probe_seed"]


@pytest.mark.asyncio
async def test_an_unseeded_submission_leaves_the_node_unseeded(
    service, store, _probe_node,
):
    """No seed must mean NO seed, not "seeded with 0"."""
    metrics = await _probe_metrics(service, store, {})

    assert metrics["probe_has_seed"] == 0.0
    assert "probe_seed" not in metrics


@pytest.mark.asyncio
async def test_the_deterministic_option_reaches_the_executing_node(
    service, store, _probe_node,
):
    on = await _probe_metrics(service, store, {"deterministic": True})
    off = await _probe_metrics(service, store, {})

    assert on["probe_deterministic"] == 1.0
    assert off["probe_deterministic"] == 0.0


@pytest.mark.asyncio
async def test_a_seeded_run_does_not_overlap_another_run(service, store):
    """The bitwise promise, made true rather than narrowed.

    Per-node seeding writes PROCESS-GLOBAL RNGs, so a second run drawing
    from them moves the first run's numbers — measured at 8/8 values
    diverged before ``_RunExclusion`` existed, at the shipped cpu queue depth
    of 2. A seeded run now waits for the runs in flight and runs alone.

    Asserted on the OVERLAP itself rather than on floats: the numbers are
    the symptom, the overlap is the cause, and a test on the cause cannot
    pass by luck.
    """
    inflight: list[str] = []
    overlaps: list[tuple[str, ...]] = []
    original = service._execute

    async def _watched(active, graph, options, session=None):
        label = "seeded" if options.get("seed") is not None else "plain"
        inflight.append(label)
        if len(inflight) > 1:
            overlaps.append(tuple(inflight))
        try:
            return await original(active, graph, options, session)
        finally:
            inflight.remove(label)

    service._execute = _watched

    await asyncio.gather(
        service.submit(_graph(), options={"seed": 11}),
        service.submit(_graph(), options={}),
        service.submit(_graph(), options={"seed": 12}),
    )
    for _ in range(600):
        if not service._runs and not any(service._pending.values()):
            break
        await asyncio.sleep(0.02)

    assert not any("seeded" in pair for pair in overlaps), (
        f"a seeded run overlapped another run: {overlaps}")


@pytest.mark.asyncio
async def test_unseeded_runs_still_overlap_each_other(service, store):
    """The gate must not become a global serialisation of the server."""
    entered = asyncio.Event()
    release = asyncio.Event()
    concurrent = 0
    peak = 0
    original = service._execute

    async def _watched(active, graph, options, session=None):
        nonlocal concurrent, peak
        concurrent += 1
        peak = max(peak, concurrent)
        entered.set()
        try:
            await asyncio.wait_for(release.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        try:
            return await original(active, graph, options, session)
        finally:
            concurrent -= 1

    service._execute = _watched

    await service.submit(_graph(), options={})
    await service.submit(_graph(), options={})
    await asyncio.wait_for(entered.wait(), timeout=2.0)
    for _ in range(100):
        if peak >= 2:
            break
        await asyncio.sleep(0.02)
    release.set()

    assert peak >= 2, "the cpu queue admits 2; unseeded runs must still overlap"
