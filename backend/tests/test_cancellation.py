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

import pytest
import torch
import torch.nn as nn

from app.config import settings
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
def _clean_interrupt_dir():
    """Remove the interrupt checkpoints these tests write."""
    from app.core.checkpoints import INTERRUPT_DIRNAME

    directory = settings.MODELS_DIR / INTERRUPT_DIRNAME
    before = set(directory.glob("*.pt")) if directory.exists() else set()
    yield
    if directory.exists():
        for path in set(directory.glob("*.pt")) - before:
            path.unlink(missing_ok=True)


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
    ctx = ExecutionContext()
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


def test_interrupt_checkpoint_resumes_with_loss_continuity():
    """4 epochs straight == 2 epochs, Stop, resume from the interrupt file.

    The resume goes through the #118 path exactly as a user would wire it:
    ``CheckpointLoader`` reads the file the interruption wrote and its
    ``epoch`` output feeds ``TrainingLoop.start_epoch``. The remaining
    epochs' losses have to match the straight run's tail — that is what
    "loss continuity" means, and it is only true if the optimizer state and
    the epoch counter both survived.
    """
    loader = _loader()
    loss_fn = nn.CrossEntropyLoss()

    straight_model = _fresh_model()
    straight = TrainingLoopNode().execute(
        {"model": straight_model, "dataloader": loader, "loss_fn": loss_fn,
         "optimizer": torch.optim.SGD(straight_model.parameters(), lr=0.1,
                                      momentum=0.9)},
        {"epochs": 4, "device": "cpu"},
    )

    # ── leg 1: stop after two epochs ─────────────────────────────────────
    ctx = _StopAfterEpochs()
    ctx.arm(2)
    model = _fresh_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    def on_progress(payload):
        if payload.get("event") == "epoch":
            ctx.saw_epoch()

    first = TrainingLoopNode().execute(
        {"model": model, "dataloader": loader, "loss_fn": loss_fn,
         "optimizer": optimizer},
        {"epochs": 4, "device": "cpu"},
        progress_callback=on_progress,
        context=ctx,
    )

    assert first["metrics"]["interrupted"] is True
    assert first["metrics"]["total_epochs_run"] == 2
    checkpoint_path = first["metrics"]["interrupt_checkpoint"]
    assert checkpoint_path, "no interrupt checkpoint was written"

    # The artifact row is queued on the context for the run service.
    artifacts = [s for s in ctx.outbox.drain()[0]
                 if getattr(s, "kind", None) == "checkpoint"]
    assert len(artifacts) == 1
    assert artifacts[0].meta == {"reason": "interrupted", "epoch": 2,
                                 "batch": 0}

    # ── leg 2: resume through CheckpointLoader -> start_epoch ────────────
    resumed_model = _fresh_model()
    restored = CheckpointLoaderNode().execute(
        {"model": resumed_model,
         "optimizer": torch.optim.SGD(resumed_model.parameters(), lr=0.1,
                                      momentum=0.9)},
        {"path": checkpoint_path, "device": "cpu"},
    )
    assert restored["epoch"] == 2

    second = TrainingLoopNode().execute(
        {"model": restored["model"], "dataloader": loader, "loss_fn": loss_fn,
         "optimizer": restored["optimizer"], "start_epoch": restored["epoch"]},
        {"epochs": 4, "device": "cpu"},
    )

    assert second["metrics"]["total_epochs_run"] == 2
    assert second["metrics"]["start_epoch"] == 2
    assert second["losses"].tolist() == pytest.approx(
        straight["losses"][2:].tolist(), **TOL)
    for key, value in straight["model"].state_dict().items():
        assert torch.allclose(value, second["model"].state_dict()[key],
                              rtol=1e-5, atol=1e-7), key


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
    ctx = ExecutionContext()
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
    assert off == ["train_loss", "train_loss"]

    on = run(True)
    assert on.count("train_loss_batch") == 8, "4 batches x 2 epochs"
    assert on.count("train_loss") == 2


def test_validation_loss_is_logged_when_present():
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
        ("train_loss", 1), ("val_loss", 1), ("train_loss", 2), ("val_loss", 2),
    ]


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

    ctx = ExecutionContext()
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

    assert result[INTERRUPTED_KEY]["epoch"] == 2
    assert result["losses"].numel() == 2, "500 epochs were asked for"
    assert result[INTERRUPTED_KEY]["checkpoint_path"]
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
