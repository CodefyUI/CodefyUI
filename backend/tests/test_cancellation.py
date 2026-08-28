"""Tests for execution cancellation.

Two eras, deliberately kept in one file:

* **Node-boundary cancellation** (Phase 2) — the engine refuses to start the
  next node once the context is cancelled. Still true, still tested first.
* **Per-batch cooperative cancellation** (#122) — the long-loop nodes poll
  ``context.should_stop()`` inside their own loops, so Stop lands in about
  one batch instead of at the end of a multi-hour run. This is the half the
  suite was missing: before #122 nothing here exercised a node that keeps
  running after the flag is set.

The #122 tests all run on the CPU with fixed seeds. Where they need a run
to be "long", they use a dataset that sleeps in ``__getitem__`` rather than
a big model, so the wall clock is a property of the test rather than of the
machine it runs on.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from app.config import settings
from app.core.cache import ExecutionCache
from app.core.execution_context import (
    INTERRUPTED_KEY,
    CancellationError,
    ExecutionContext,
)
from app.core.graph_engine import execute_graph
from app.core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from app.core.node_registry import registry
from app.nodes.io.checkpoint_node import CheckpointLoaderNode
from app.nodes.training.evaluate_model_node import EvaluateModelNode
from app.nodes.training.training_loop_node import TrainingLoopNode


def _start_node(nid="start"):
    return {"id": nid, "type": "Start", "data": {"params": {}}}


def _trigger(eid, src, tgt):
    return {"id": eid, "source": src, "target": tgt, "sourceHandle": "trigger", "type": "trigger"}


# ── node-boundary cancellation (Phase 2) ──────────────────────────────────


def test_execution_context_cancel():
    ctx = ExecutionContext()
    assert not ctx.cancelled
    ctx.cancel()
    assert ctx.cancelled


@pytest.mark.asyncio
async def test_cancel_before_execution():
    """Cancelling context before execution raises CancellationError."""
    ctx = ExecutionContext()
    ctx.cancel()

    nodes = [
        _start_node(),
        {"id": "1", "type": "_TestSource", "data": {"params": {}}},
    ]
    edges = [_trigger("et", "start", "1")]

    with pytest.raises(CancellationError):
        await execute_graph(nodes, edges, context=ctx)


@pytest.mark.asyncio
async def test_cancel_during_execution():
    """Cancelling mid-execution stops before later nodes run."""
    ctx = ExecutionContext()
    executed_nodes = []

    async def on_progress(node_id, status, data):
        if status == "running":
            executed_nodes.append(node_id)
            if node_id == "1":
                ctx.cancel()

    nodes = [
        _start_node(),
        {"id": "1", "type": "_TestSource", "data": {"params": {}}},
        {"id": "2", "type": "Print", "data": {"params": {}}},
    ]
    edges = [
        _trigger("et", "start", "1"),
        {"source": "1", "target": "2", "sourceHandle": "value", "targetHandle": "value"},
    ]

    with pytest.raises(CancellationError):
        await execute_graph(nodes, edges, on_progress=on_progress, context=ctx)

    # Node 1 started, but node 2 should not have started
    assert "1" in executed_nodes


@pytest.mark.asyncio
async def test_cancel_during_the_last_level_still_raises():
    """#122: a stop has to be observed even with no next node to check it.

    Before this, a graph whose final node was the training loop returned a
    normal result after a Stop, and the run service filed it ``succeeded``.
    """
    ctx = ExecutionContext()

    async def on_progress(node_id, status, data):
        if node_id == "1" and status == "running":
            ctx.cancel()

    nodes = [_start_node(), {"id": "1", "type": "_TestSource", "data": {"params": {}}}]
    edges = [_trigger("et", "start", "1")]

    with pytest.raises(CancellationError):
        await execute_graph(nodes, edges, on_progress=on_progress, context=ctx)


# ── should_stop() is readable from a worker thread ────────────────────────


def test_should_stop_matches_cancelled_and_is_thread_safe():
    ctx = ExecutionContext()
    seen: list[bool] = []

    def reader():
        seen.append(ctx.should_stop())

    thread = threading.Thread(target=reader)
    thread.start()
    thread.join()
    assert seen == [False]

    ctx.cancel()
    seen.clear()
    thread = threading.Thread(target=reader)
    thread.start()
    thread.join()
    assert seen == [True]
    assert ctx.should_stop() is ctx.cancelled is True


# ── #122 fixtures: a training graph that would run for a long time ────────

IN_FEATURES = 4
OUT_CLASSES = 3


class _SleepyDataset(torch.utils.data.Dataset):
    """A dataset whose every read costs *delay* seconds.

    Makes a run long WITHOUT making it heavy, so the cancellation-latency
    assertion is about the engine rather than about how fast this machine
    multiplies matrices.
    """

    def __init__(self, n: int, delay: float) -> None:
        gen = torch.Generator().manual_seed(1234)
        self.x = torch.randn(n, IN_FEATURES, generator=gen)
        self.y = torch.randint(0, OUT_CLASSES, (n,), generator=gen)
        self.delay = delay

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        if self.delay:
            time.sleep(self.delay)
        return self.x[index], self.y[index]


class _TrainingInputsNode(BaseNode):
    """Model + dataloader + optimizer + loss for a one-node training graph.

    A test node rather than five real ones: this file is testing the loop's
    stopping behaviour, not Dataset/DataLoader/Optimizer wiring (which
    test_training_resume.py and the example tests already cover).
    """

    NODE_NAME = "_TrainingInputs"
    CATEGORY = "Test"
    DESCRIPTION = "Emits a tiny training setup over a deliberately slow dataset"
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL),
            PortDefinition(name="dataloader", data_type=DataType.DATALOADER),
            PortDefinition(name="optimizer", data_type=DataType.OPTIMIZER),
            PortDefinition(name="loss_fn", data_type=DataType.LOSS_FN),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="samples", param_type=ParamType.INT, default=4000),
            ParamDefinition(name="batch_size", param_type=ParamType.INT, default=4),
            ParamDefinition(name="delay", param_type=ParamType.FLOAT, default=0.02),
        ]

    def execute(self, inputs, params):
        # The engine hands over the graph's raw params, without the defaults
        # declared above, so every read needs its own fallback.
        torch.manual_seed(7)
        model = nn.Linear(IN_FEATURES, OUT_CLASSES)
        dataset = _SleepyDataset(int(params.get("samples", 4000)),
                                 float(params.get("delay", 0.02)))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=int(params.get("batch_size", 4)), shuffle=False)
        return {
            "model": model,
            "dataloader": loader,
            "optimizer": torch.optim.SGD(model.parameters(), lr=0.05),
            "loss_fn": nn.CrossEntropyLoss(),
        }


_TEST_NODES = {"_TrainingInputs": _TrainingInputsNode}


@pytest.fixture(autouse=True)
def _register_test_nodes():
    registry._nodes.update(_TEST_NODES)
    yield
    for name in _TEST_NODES:
        registry._nodes.pop(name, None)


@pytest.fixture(autouse=True)
def interrupt_dir(tmp_path, monkeypatch):
    """Point MODELS_DIR at a per-test directory; yield its interrupt folder.

    Redirecting the whole setting rather than diffing the shared
    ``MODELS_DIR`` afterwards: a set-difference over a directory two workers
    are writing to is a race under ``pytest-xdist``, and it can only clean up
    files it can attribute. An isolated root makes "no file was written" an
    assertion this file can actually make (see the contract-runner test) and
    leaves nothing behind either way.
    """
    from app.core.checkpoints import INTERRUPT_DIRNAME

    models = tmp_path / "data" / "models"
    models.mkdir(parents=True)
    monkeypatch.setattr(settings, "MODELS_DIR", models)
    return models / INTERRUPT_DIRNAME


def _recording_context(**kwargs) -> ExecutionContext:
    """A context that records artifacts, as a real run's does.

    ``signals_recorded`` is set by ``execute_graph`` from its ``on_signal``,
    so a test calling ``execute()`` directly has to say so — and a node only
    writes an interrupt checkpoint when something will record the row.
    """
    return ExecutionContext(signals_recorded=True, **kwargs)


def _training_graph(*, epochs: int, source_params: dict | None = None,
                    train_params: dict | None = None):
    nodes = [
        _start_node(),
        {"id": "inputs", "type": "_TrainingInputs",
         "data": {"params": source_params or {}}},
        {"id": "train", "type": "TrainingLoop",
         "data": {"params": {"epochs": epochs, "device": "cpu",
                             **(train_params or {})}}},
    ]
    edges = [_trigger("et", "start", "inputs")]
    for port in ("model", "dataloader", "optimizer", "loss_fn"):
        edges.append({"id": f"e-{port}", "source": "inputs", "target": "train",
                      "sourceHandle": port, "targetHandle": port})
    return nodes, edges


# ── acceptance: a cancel lands within 2s of a multi-minute training ───────

#: The acceptance criterion, verbatim from the issue.
CANCEL_DEADLINE_S = 2.0


@pytest.mark.asyncio
async def test_cancel_lands_within_two_seconds_of_a_long_training():
    """The headline #122 criterion.

    4000 samples at 20 ms a read, batch size 4, 50 epochs -- a nominal
    66-minute run. The Stop is issued once training is genuinely under way
    (a batch progress frame has been seen) and the clock starts there, so
    what is measured is cancellation latency and not start-up.

    Before #122 this test does not merely fail, it hangs: nothing inside
    ``TrainingLoopNode`` ever looked at the flag.
    """
    ctx = ExecutionContext()
    running = asyncio.Event()

    async def on_progress(node_id, status, data):
        if status == "progress" and (data or {}).get("event") == "batch":
            running.set()

    nodes, edges = _training_graph(epochs=50)
    task = asyncio.create_task(
        execute_graph(nodes, edges, on_progress=on_progress, context=ctx))

    await asyncio.wait_for(running.wait(), timeout=30.0)
    cancelled_at = time.monotonic()
    ctx.cancel()

    with pytest.raises(CancellationError):
        await asyncio.wait_for(task, timeout=30.0)
    elapsed = time.monotonic() - cancelled_at

    assert elapsed <= CANCEL_DEADLINE_S, (
        f"cancel took {elapsed:.2f}s to land, budget is {CANCEL_DEADLINE_S}s"
    )


@pytest.mark.asyncio
async def test_an_interrupted_node_reports_interrupted_and_partial_outputs():
    """A stopped node returns cleanly; the engine says ``interrupted``."""
    ctx = ExecutionContext()
    statuses: list[tuple[str, str]] = []
    running = asyncio.Event()

    async def on_progress(node_id, status, data):
        if status == "progress":
            if (data or {}).get("event") == "epoch":
                running.set()
            return
        statuses.append((node_id, status))

    # Two fast epochs' worth of data, ten epochs asked for: the stop lands
    # while there is plenty left to do.
    nodes, edges = _training_graph(
        epochs=10, source_params={"samples": 64, "batch_size": 8,
                                  "delay": 0.01})
    task = asyncio.create_task(
        execute_graph(nodes, edges, on_progress=on_progress, context=ctx))
    await asyncio.wait_for(running.wait(), timeout=30.0)
    ctx.cancel()
    with pytest.raises(CancellationError):
        await asyncio.wait_for(task, timeout=30.0)

    assert ("train", "interrupted") in statuses
    assert ("train", "completed") not in statuses
    assert ("train", "error") not in statuses, (
        "a cooperative stop must not look like a failure"
    )


def test_interrupt_marker_carries_where_it_stopped():
    """The result marker, straight off ``execute`` with a pre-cancelled context."""
    ctx = _recording_context()
    ctx.cancel()
    model = nn.Linear(IN_FEATURES, OUT_CLASSES)
    result = TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": torch.utils.data.DataLoader(
                _SleepyDataset(16, 0.0), batch_size=4, shuffle=False),
            "optimizer": torch.optim.SGD(model.parameters(), lr=0.05),
            "loss_fn": nn.CrossEntropyLoss(),
        },
        {"epochs": 3, "device": "cpu"},
        context=ctx,
    )

    marker = result[INTERRUPTED_KEY]
    assert marker["epoch"] == 0 and marker["batch"] == 0
    assert marker["phase"] == "train"
    assert result["metrics"]["interrupted"] is True
    assert result["losses"].numel() == 0, "no epoch completed"


# ── acceptance: interrupt -> resume keeps the loss curve continuous ───────

TOL = dict(rel=1e-5, abs=1e-7)


def _loader(n: int = 24, batch_size: int = 6):
    gen = torch.Generator().manual_seed(1234)
    x = torch.randn(n, IN_FEATURES, generator=gen)
    y = torch.randint(0, OUT_CLASSES, (n,), generator=gen)
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=batch_size,
        shuffle=False)


def _fresh_model() -> nn.Module:
    torch.manual_seed(7)
    return nn.Linear(IN_FEATURES, OUT_CLASSES)


class _StopAfterEpochs(ExecutionContext):
    """Cancels itself once ``should_stop`` has seen *epochs* epoch events.

    Driving the stop off the node's own progress stream keeps the test
    deterministic: no sleeps, no races, and the interruption always lands at
    the same batch.
    """

    def arm(self, epochs: int) -> None:
        self._target = epochs
        self._epochs_seen = 0

    def saw_epoch(self) -> None:
        self._epochs_seen += 1
        if self._epochs_seen >= self._target:
            self.cancel()


def _scheduler(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)


def test_interrupt_checkpoint_resumes_with_loss_continuity(interrupt_dir):
    """4 epochs straight == 2 epochs, Stop, resume from the interrupt file.

    The resume goes through the #118 path exactly as a user would wire it:
    ``CheckpointLoader`` reads the file the interruption wrote and its
    ``epoch`` output feeds ``TrainingLoop.start_epoch``. The remaining
    epochs' losses have to match the straight run's tail — that is what
    "loss continuity" means, and it is only true if the optimizer state and
    the epoch counter both survived.

    An ``StepLR`` rides along on both legs, so the LR SCHEDULE has to survive
    the interruption too: the interrupt checkpoint stores the scheduler state
    (#118's key), ``CheckpointLoader`` restores it, and the resumed leg's
    ``lr_history`` must equal the straight run's tail. Without that the
    resumed run trains the remaining epochs at the wrong learning rate and
    the loss comparison below would be the only thing to notice — quietly,
    and only for schedules that decay fast enough.
    """
    loader = _loader()
    loss_fn = nn.CrossEntropyLoss()

    straight_model = _fresh_model()
    straight_opt = torch.optim.SGD(straight_model.parameters(), lr=0.1,
                                   momentum=0.9)
    straight = TrainingLoopNode().execute(
        {"model": straight_model, "dataloader": loader, "loss_fn": loss_fn,
         "optimizer": straight_opt, "lr_scheduler": _scheduler(straight_opt)},
        {"epochs": 4, "device": "cpu"},
    )

    # ── leg 1: stop after two epochs ─────────────────────────────────────
    ctx = _StopAfterEpochs(signals_recorded=True)
    ctx.arm(2)
    model = _fresh_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = _scheduler(optimizer)

    def on_progress(payload):
        if payload.get("event") == "epoch":
            ctx.saw_epoch()

    first = TrainingLoopNode().execute(
        {"model": model, "dataloader": loader, "loss_fn": loss_fn,
         "optimizer": optimizer, "lr_scheduler": scheduler},
        {"epochs": 4, "device": "cpu"},
        progress_callback=on_progress,
        context=ctx,
    )

    assert first["metrics"]["interrupted"] is True
    assert first["metrics"]["total_epochs_run"] == 2
    checkpoint_path = first["metrics"]["interrupt_checkpoint"]
    assert checkpoint_path, "no interrupt checkpoint was written"
    # Staged through a temp file and renamed: one .pt, no debris.
    assert [p.name for p in interrupt_dir.iterdir()] == [
        Path(checkpoint_path).name]

    # The artifact row is queued on the context for the run service.
    artifacts = [s for s in ctx.outbox.drain()[0]
                 if getattr(s, "kind", None) == "checkpoint"]
    assert len(artifacts) == 1
    assert artifacts[0].meta == {"reason": "interrupted", "epoch": 2,
                                 "batch": 0}

    # ── leg 2: resume through CheckpointLoader -> start_epoch ────────────
    resumed_model = _fresh_model()
    resumed_opt = torch.optim.SGD(resumed_model.parameters(), lr=0.1,
                                  momentum=0.9)
    resumed_sched = _scheduler(resumed_opt)
    restored = CheckpointLoaderNode().execute(
        {"model": resumed_model, "optimizer": resumed_opt,
         "lr_scheduler": resumed_sched},
        {"path": checkpoint_path, "device": "cpu"},
    )
    assert restored["epoch"] == 2
    assert resumed_sched.last_epoch == 2, "the schedule did not survive"

    second = TrainingLoopNode().execute(
        {"model": restored["model"], "dataloader": loader, "loss_fn": loss_fn,
         "optimizer": restored["optimizer"],
         "lr_scheduler": restored["lr_scheduler"],
         "start_epoch": restored["epoch"]},
        {"epochs": 4, "device": "cpu"},
    )

    assert second["metrics"]["total_epochs_run"] == 2
    assert second["metrics"]["start_epoch"] == 2
    assert second["metrics"]["lr_history"] == pytest.approx(
        straight["metrics"]["lr_history"][2:], rel=1e-9)
    assert second["losses"].tolist() == pytest.approx(
        straight["losses"][2:].tolist(), **TOL)
    for key, value in straight["model"].state_dict().items():
        assert torch.allclose(value, second["model"].state_dict()[key],
                              rtol=1e-5, atol=1e-7), key


def _training_preset(name: str = "_InterruptiblePipeline") -> dict:
    """A two-node preset: TrainingLoop feeding a Print.

    Two nodes rather than one so "never rolls up to completed" has something
    to be true about — the preset only reports ``completed`` once EVERY
    internal node has finished, and a single-node preset would satisfy that
    vacuously on any status that increments the done count.
    """
    return {
        "preset_name": name,
        "category": "Test",
        "description": "",
        "tags": [],
        "nodes": [
            {"id": "train", "type": "TrainingLoop",
             "params": {"epochs": 10, "device": "cpu"}},
            {"id": "show", "type": "Print", "params": {"label": "metrics"}},
        ],
        "edges": [
            {"source": "train", "target": "show",
             "sourceHandle": "metrics", "targetHandle": "value"},
        ],
        "exposed_inputs": [
            {"name": port, "internal_node": "train", "internal_port": port,
             "data_type": "ANY", "description": ""}
            for port in ("model", "dataloader", "optimizer", "loss_fn")
        ],
        "exposed_outputs": [],
        "exposed_params": [],
    }


@pytest.mark.asyncio
async def test_a_preset_containing_a_stopped_node_settles_interrupted():
    """A preset whose training node stopped must not report ``completed``.

    ``_emit_preset_aware`` rolls internal node statuses up into one preset
    status, and ``completed`` is emitted once the done count reaches the
    internal node count. ``interrupted`` has to settle the preset the way
    ``error`` does — otherwise a stopped training pipeline shows a green
    preset, which is the exact opposite of what happened.
    """
    from app.core.graph_engine import build_preset_fallback

    ctx = ExecutionContext()
    statuses: list[tuple[str, str]] = []
    running = asyncio.Event()

    async def on_progress(node_id, status, data):
        if status == "progress":
            if (data or {}).get("event") == "epoch":
                running.set()
            return
        statuses.append((node_id, status))

    preset = _training_preset()
    nodes = [
        _start_node(),
        {"id": "inputs", "type": "_TrainingInputs",
         "data": {"params": {"samples": 64, "batch_size": 8, "delay": 0.01}}},
        {"id": "pipeline", "type": f"preset:{preset['preset_name']}",
         "position": {"x": 0, "y": 0}, "data": {"params": {}}},
    ]
    edges = [_trigger("et", "start", "inputs")] + [
        {"id": f"e-{port}", "source": "inputs", "target": "pipeline",
         "sourceHandle": port, "targetHandle": port}
        for port in ("model", "dataloader", "optimizer", "loss_fn")
    ]

    task = asyncio.create_task(execute_graph(
        nodes, edges, on_progress=on_progress, context=ctx,
        preset_fallback=build_preset_fallback([preset]),
    ))
    await asyncio.wait_for(running.wait(), timeout=30.0)
    ctx.cancel()
    with pytest.raises(CancellationError):
        await asyncio.wait_for(task, timeout=30.0)

    preset_statuses = [s for nid, s in statuses if nid == "pipeline"]
    assert "interrupted" in preset_statuses
    assert "completed" not in preset_statuses
    # The internal node ids never surface; the roll-up is the whole point.
    assert not any(nid.startswith("pipeline__") for nid, _ in statuses)


@pytest.mark.asyncio
async def test_no_checkpoint_is_written_when_the_run_records_no_artifacts(
    interrupt_dir,
):
    """The REST contract runner's shape: cancel, and leave NO orphan file.

    ``execute_contract_run`` cancels its context on a timeout and passes no
    ``on_signal``, so an ``ArtifactSignal`` would be discarded — and the
    ``.pt`` beside it would have no row, no index and no sweeper, growing a
    model-sized hole on every timeout. This drives the engine the same way
    (``on_signal=None``) because that argument IS the mechanism; the route
    adds nothing to it.
    """
    ctx = ExecutionContext()
    running = asyncio.Event()

    async def on_progress(node_id, status, data):
        if status == "progress" and (data or {}).get("event") == "epoch":
            running.set()

    # 0.01 like every sibling of this await-a-frame-then-cancel shape, and
    # for the same reason: with a zero delay the 10 epochs can finish before
    # the loop gets around to running the cancel, and the assertion below
    # fails with DID-NOT-RAISE for reasons having nothing to do with
    # checkpoints. 8 batches x 10 ms leaves ~700 ms of headroom.
    nodes, edges = _training_graph(
        epochs=10, source_params={"samples": 64, "batch_size": 8,
                                  "delay": 0.01})
    task = asyncio.create_task(
        execute_graph(nodes, edges, on_progress=on_progress, context=ctx))
    await asyncio.wait_for(running.wait(), timeout=30.0)
    # The engine stamped the capability from its (absent) on_signal.
    assert ctx.can_record_artifacts() is False
    ctx.cancel()
    with pytest.raises(CancellationError):
        await asyncio.wait_for(task, timeout=30.0)

    assert not interrupt_dir.exists() or list(interrupt_dir.iterdir()) == [], (
        "an interrupted run with nowhere to record the row still wrote a file"
    )


def test_mid_epoch_interrupt_replays_the_partial_epoch(monkeypatch):
    """Stopping inside epoch 1 checkpoints epoch 1 as INCOMPLETE.

    The partial epoch's weight updates are already in the model and its loss
    average is not, so the honest resume point is the start of that epoch —
    ``epoch=1`` here, not 2. Getting this wrong is how a resumed run skips
    data it never trained on.

    The throttle is disabled so the stop can be aimed at an exact batch;
    with the real 0.5 s floor a loop this fast emits one frame in total.
    """
    monkeypatch.setattr("app.core.loop_control.PROGRESS_MIN_INTERVAL_S", 0.0)
    ctx = _recording_context()
    model = _fresh_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches_seen = {"n": 0}

    def on_progress(payload):
        if payload.get("event") == "batch":
            batches_seen["n"] += 1
            if payload["epoch"] == 2 and payload["batch"] == 2:
                ctx.cancel()

    result = TrainingLoopNode().execute(
        {"model": model, "dataloader": _loader(), "loss_fn": nn.CrossEntropyLoss(),
         "optimizer": optimizer},
        {"epochs": 5, "device": "cpu"},
        progress_callback=on_progress,
        context=ctx,
    )

    assert batches_seen["n"] > 0, "per-batch progress never fired"
    marker = result[INTERRUPTED_KEY]
    assert marker["epoch"] == 1, "one epoch is complete, so resume at 1"
    assert marker["batch"] == 2
    assert result["metrics"]["total_epochs_run"] == 1

    checkpoint = torch.load(marker["checkpoint_path"], map_location="cpu",
                            weights_only=True)
    assert checkpoint["epoch"] == 1
    assert "optimizer_state_dict" in checkpoint


# ── acceptance: a 1-epoch, many-batch run shows progress movement ─────────


def test_one_epoch_over_many_batches_reports_every_batch(monkeypatch):
    """The regression vs per-epoch-only progress.

    The throttle is turned off for this test so the assertion is about
    "reports each batch" rather than about the wall clock; the throttle
    itself is pinned separately below.
    """
    monkeypatch.setattr("app.core.loop_control.PROGRESS_MIN_INTERVAL_S", 0.0)

    events: list[dict] = []
    model = _fresh_model()
    TrainingLoopNode().execute(
        {"model": model, "dataloader": _loader(n=40, batch_size=4),
         "loss_fn": nn.CrossEntropyLoss(),
         "optimizer": torch.optim.SGD(model.parameters(), lr=0.05)},
        {"epochs": 1, "device": "cpu"},
        progress_callback=events.append,
    )

    batches = [e for e in events if e.get("event") == "batch"]
    assert [e["batch"] for e in batches] == list(range(1, 11))
    assert all(e["total_batches"] == 10 for e in batches)
    assert all(e["epoch"] == 1 for e in batches)
    assert all(isinstance(e["loss"], float) for e in batches)
    # The per-epoch frame still lands, unchanged, after the batch frames.
    assert [e["event"] for e in events[-1:]] == ["epoch"]


def test_batch_progress_is_throttled_by_default():
    """At most one frame per PROGRESS_MIN_INTERVAL_S, however fast the loop."""
    from app.core.loop_control import PROGRESS_MIN_INTERVAL_S

    events: list[dict] = []
    model = _fresh_model()
    started = time.monotonic()
    TrainingLoopNode().execute(
        {"model": model, "dataloader": _loader(n=400, batch_size=2),
         "loss_fn": nn.CrossEntropyLoss(),
         "optimizer": torch.optim.SGD(model.parameters(), lr=0.05)},
        {"epochs": 1, "device": "cpu"},
        progress_callback=events.append,
    )
    elapsed = time.monotonic() - started

    batches = [e for e in events if e.get("event") == "batch"]
    ceiling = 1 + int(elapsed / PROGRESS_MIN_INTERVAL_S) + 1
    assert 1 <= len(batches) <= ceiling, (
        f"{len(batches)} batch frames over {elapsed:.2f}s of 200 batches"
    )


def test_batch_metrics_are_opt_in():
    """``train_loss_batch`` only exists when the option is on."""
    def run(batch_metrics: bool) -> list[str]:
        ctx = ExecutionContext()
        model = _fresh_model()
        TrainingLoopNode().execute(
            {"model": model, "dataloader": _loader(n=24, batch_size=6),
             "loss_fn": nn.CrossEntropyLoss(),
             "optimizer": torch.optim.SGD(model.parameters(), lr=0.05)},
            {"epochs": 2, "device": "cpu", "batch_metrics": batch_metrics},
            context=ctx,
        )
        return [s.name for s in ctx.outbox.drain()[0]]

    off = run(False)
    assert off == ["train_loss", "lr", "train_loss", "lr"]

    on = run(True)
    assert on.count("train_loss_batch") == 8, "4 batches x 2 epochs"
    assert on.count("train_loss") == 2
    assert on.count("lr") == 2


def test_validation_loss_is_logged_when_present():
    """#202: val_accuracy joins val_loss in this list -- this scenario is a
    classifier (CrossEntropyLoss) with a val_dataloader wired, exactly
    val_accuracy's gate, so it is expected here too."""
    ctx = ExecutionContext()
    model = _fresh_model()
    TrainingLoopNode().execute(
        {"model": model, "dataloader": _loader(n=24, batch_size=6),
         "val_dataloader": _loader(n=12, batch_size=6),
         "loss_fn": nn.CrossEntropyLoss(),
         "optimizer": torch.optim.SGD(model.parameters(), lr=0.05)},
        {"epochs": 2, "device": "cpu"},
        context=ctx,
    )
    points = ctx.outbox.drain()[0]
    assert [(p.name, p.step) for p in points] == [
        ("train_loss", 1), ("val_loss", 1), ("val_accuracy", 1), ("lr", 1),
        ("train_loss", 2), ("val_loss", 2), ("val_accuracy", 2), ("lr", 2),
    ]


def test_every_previously_inferred_series_is_still_logged():
    """Explicit logging must not silently drop a series inference gave us.

    ``_collect_metrics`` stands down per NODE, so the moment TrainingLoop
    logs anything explicitly, NOTHING in its epoch progress payload is mined
    any more. Everything ``scalar_metrics`` used to take from that payload —
    ``lr`` and, under early stopping, ``patience_counter`` and
    ``best_epoch`` — therefore has to be logged by hand. ``loss`` is the one
    deliberate change: it is ``train_loss`` now. #202 adds a second:
    ``val_accuracy`` is now in the epoch payload too (this scenario is a
    classifier with a val_dataloader, its gate), and is likewise logged
    explicitly.
    """
    model = _fresh_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ctx = ExecutionContext()
    payloads: list[dict] = []

    TrainingLoopNode().execute(
        {"model": model, "dataloader": _loader(n=24, batch_size=6),
         "val_dataloader": _loader(n=12, batch_size=6),
         "loss_fn": nn.CrossEntropyLoss(), "optimizer": optimizer,
         "lr_scheduler": _scheduler(optimizer)},
        {"epochs": 3, "device": "cpu", "early_stopping_patience": 5},
        progress_callback=payloads.append,
        context=ctx,
    )

    logged = {p.name for p in ctx.outbox.drain()[0]}
    # What inference WOULD have produced from the epoch payloads.
    from app.core.run_service import scalar_metrics

    inferable = {
        point.name
        for payload in payloads if payload.get("event") == "epoch"
        for point in scalar_metrics(payload, node_id=None, step=1)
    }
    assert inferable == {"loss", "val_loss", "val_accuracy", "lr",
                         "patience_counter", "best_epoch"}
    assert logged == (inferable - {"loss"}) | {"train_loss"}


def test_lr_series_tracks_the_schedule():
    """The `lr` series exists for a TrainingLoop run and follows the LR."""
    model = _fresh_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ctx = ExecutionContext()

    result = TrainingLoopNode().execute(
        {"model": model, "dataloader": _loader(n=24, batch_size=6),
         "loss_fn": nn.CrossEntropyLoss(), "optimizer": optimizer,
         "lr_scheduler": _scheduler(optimizer)},
        {"epochs": 4, "device": "cpu"},
        context=ctx,
    )

    lr_points = [p for p in ctx.outbox.drain()[0] if p.name == "lr"]
    assert [p.step for p in lr_points] == [1, 2, 3, 4]
    assert [p.value for p in lr_points] == pytest.approx(
        result["metrics"]["lr_history"], rel=1e-9)
    assert lr_points[0].value != lr_points[-1].value, (
        "StepLR should have decayed; a constant series proves nothing"
    )


# ── the other long-loop nodes ─────────────────────────────────────────────


def test_evaluate_model_stops_mid_pass_and_returns_partial_counts(monkeypatch):
    monkeypatch.setattr("app.core.loop_control.PROGRESS_MIN_INTERVAL_S", 0.0)
    ctx = ExecutionContext()
    model = _fresh_model()
    dataset = _SleepyDataset(64, 0.0)

    def on_progress(payload):
        if payload.get("batch") == 2:
            ctx.cancel()

    result = EvaluateModelNode().execute(
        {"model": model, "dataset": dataset},
        {"batch_size": 8, "device": "cpu"},
        progress_callback=on_progress,
        context=ctx,
    )

    assert INTERRUPTED_KEY in result
    assert 0 < result["total"] < 64, "a partial pass, honestly labelled"
    assert 0.0 <= result["accuracy"] <= 1.0
    # An incomplete evaluation must not be filed as an accuracy measurement.
    assert [s.name for s in ctx.outbox.drain()[0]] == []


def test_evaluate_model_logs_accuracy_when_it_finishes():
    ctx = ExecutionContext()
    result = EvaluateModelNode().execute(
        {"model": _fresh_model(), "dataset": _SleepyDataset(32, 0.0)},
        {"batch_size": 8, "device": "cpu"},
        context=ctx,
    )
    assert INTERRUPTED_KEY not in result
    points = ctx.outbox.drain()[0]
    assert [(p.name, p.value) for p in points] == [
        ("eval_accuracy", result["accuracy"])]


def test_ddpm_sampler_stops_between_timesteps():
    from app.nodes.diffusion.ddpm_sampler_node import DDPMSamplerNode

    ctx = ExecutionContext()
    calls = {"n": 0}

    def model(x, t):
        calls["n"] += 1
        if calls["n"] == 3:
            ctx.cancel()
        return torch.zeros_like(x)

    result = DDPMSamplerNode().execute(
        {"model": model, "noise": torch.zeros(1, 1, 4, 4)},
        {"num_steps": 40, "seed": 0},
        context=ctx,
    )

    assert result[INTERRUPTED_KEY]["batch"] == 3
    assert calls["n"] == 3, "the loop stopped instead of running 40 steps"
    assert result["image"].shape == (1, 1, 4, 4)


def test_diffusion_training_loop_stops_between_batches():
    from app.nodes.training.diffusion_training_loop_node import (
        DiffusionTrainingLoopNode,
    )

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x, t):
            flat = x.reshape(x.shape[0], -1)
            return self.lin(flat).reshape(x.shape)

    ctx = _recording_context()
    images = torch.randn(16, 1, 2, 2, generator=torch.Generator().manual_seed(0))
    dataset = torch.utils.data.TensorDataset(images, torch.zeros(16))

    def on_progress(payload):
        if payload.get("event") == "epoch" and payload["epoch"] == 2:
            ctx.cancel()

    result = DiffusionTrainingLoopNode().execute(
        {"model": _Tiny(), "dataset": dataset},
        {"epochs": 500, "batch_size": 8, "num_timesteps": 4, "device": "cpu"},
        progress_callback=on_progress,
        context=ctx,
    )

    marker = result[INTERRUPTED_KEY]
    assert marker["epoch"] == 2
    # The phase tells a stop BETWEEN epochs apart from one on the first
    # batch of an epoch; both report batch=0.
    assert marker["phase"] == "epoch" and marker["batch"] == 0
    assert result["losses"].numel() == 2, "500 epochs were asked for"
    assert marker["checkpoint_path"]
    names = [s.name for s in ctx.outbox.drain()[0] if hasattr(s, "name")]
    assert names == ["train_loss", "train_loss"]


def test_map_node_stops_between_items():
    from app.core.preset_registry import preset_registry
    from app.nodes.dataflow.map_node import MapNode

    preset_name = next(
        (name for name, preset in preset_registry.presets.items()
         if preset.exposed_inputs and preset.exposed_outputs),
        None,
    )
    if preset_name is None:  # pragma: no cover - the repo ships presets
        pytest.skip("no preset with exposed ports is registered")

    ctx = ExecutionContext()
    ctx.cancel()
    result = MapNode().execute(
        {"items": [1, 2, 3]}, {"subgraph": preset_name}, context=ctx)

    assert result[INTERRUPTED_KEY]["batch"] == 0
    assert result["results"] == []


# ── a stopped node is never cached ────────────────────────────────────────


class _InterruptedOnceNode(BaseNode):
    """Interrupted on its first execute, whole on every one after.

    Cacheable, like the real nodes this is about: ``TextEmbedding`` stopped
    before its first batch returns a ``(0, 0)`` tensor and the marker saying
    so. What is under test is what the ENGINE does with that, not how the
    node produced it.
    """

    NODE_NAME = "_InterruptedOnce"
    CATEGORY = "Test"
    DESCRIPTION = "Returns a partial, interrupted result the first time only"
    cacheable = True

    #: Class-level: the engine builds a fresh instance per node execution,
    #: so an instance attribute could not tell the second run from the first.
    calls = 0

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [ParamDefinition(name="val", param_type=ParamType.STRING,
                                default="whole")]

    def execute(self, inputs, params):
        type(self).calls += 1
        if type(self).calls == 1:
            # The shape ``interrupted_result`` produces: what was paid for,
            # plus the marker saying it is not all of it.
            return {"value": "partial", INTERRUPTED_KEY: {"batch": 0}}
        return {"value": params.get("val", "whole")}


@pytest.fixture
def interrupted_once():
    _InterruptedOnceNode.calls = 0
    registry._nodes["_InterruptedOnce"] = _InterruptedOnceNode
    yield _InterruptedOnceNode
    registry._nodes.pop("_InterruptedOnce", None)
    _InterruptedOnceNode.calls = 0


@pytest.mark.asyncio
async def test_interrupted_result_is_not_cached(interrupted_once):
    """A stopped node re-runs; it does not come back as ``cached``.

    The cache stores what a node returned and nothing about how much of the
    work it represents, so a partial result put there is served on the next
    Run as the finished answer -- silently, under a green ``cached`` badge,
    to somebody who pressed Stop once and has no way to tell.
    """
    cache = ExecutionCache()
    nodes = [
        _start_node(),
        {"id": "1", "type": "_InterruptedOnce",
         "data": {"params": {"val": "whole"}}},
    ]
    edges = [_trigger("et", "start", "1")]

    first: list[tuple[str, str]] = []
    second: list[tuple[str, str]] = []

    def _recorder(into):
        async def on_progress(node_id, status, data):
            into.append((node_id, status))
        return on_progress

    await execute_graph(nodes, edges, on_progress=_recorder(first), cache=cache)
    results = await execute_graph(
        nodes, edges, on_progress=_recorder(second), cache=cache)

    assert interrupted_once.calls == 2, (
        "the second run was served from the cache, so the partial result of "
        "the first is now the answer to every run after it")
    assert ("1", "interrupted") in first
    assert ("1", "completed") in second
    assert ("1", "cached") not in second
    assert results["1"]["value"] == "whole", (
        "the second run returned the partial value, not the whole one")
