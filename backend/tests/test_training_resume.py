"""Resume-from-checkpoint tests for #118.

Before #118 a "resume" only restored *weights*: ``TrainingLoopNode`` threw the
incoming optimizer away and built a fresh one, so Adam moments / SGD momentum
buffers / step counts were lost, the epoch counter restarted at 0, and the LR
schedule was reset. The tests here pin the three properties a real resume has
to have:

* training 4 epochs straight == training 2, checkpointing, restoring and
  training 2 more (Adam **and** SGD+momentum),
* the learning rate at absolute epoch N is the same on both paths,
* a checkpoint written by the pre-#118 format (no scheduler key) still loads.

Everything runs on CPU with fixed seeds and ``shuffle=False`` loaders so the
two paths are numerically comparable.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from app.config import settings
from app.nodes.io.checkpoint_node import CheckpointLoaderNode, CheckpointSaverNode
from app.nodes.training.training_loop_node import (
    TrainingLoopNode,
    _sync_optimizer_state_device,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

IN_FEATURES = 4
OUT_CLASSES = 3
TOTAL_EPOCHS = 4
SPLIT_EPOCH = 2

# The two paths perform the same float ops in the same order, so they agree to
# well within float32 noise; the tolerance only absorbs platform-level FMA
# differences, not algorithmic drift.
TOL = dict(rtol=1e-5, atol=1e-7)


@pytest.fixture
def ckpt_path(request):
    """A unique checkpoint filename under MODELS_DIR, removed afterwards."""
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"_resume_{abs(hash(request.node.nodeid)) % (10**10)}.pt"
    yield name
    (settings.MODELS_DIR / name).unlink(missing_ok=True)


def _loader(n: int = 24, batch_size: int = 6):
    """A fixed dataset in a fixed order -- no shuffling, no per-run RNG draw."""
    gen = torch.Generator().manual_seed(1234)
    x = torch.randn(n, IN_FEATURES, generator=gen)
    y = torch.randint(0, OUT_CLASSES, (n,), generator=gen)
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _model() -> nn.Module:
    torch.manual_seed(7)
    return nn.Linear(IN_FEATURES, OUT_CLASSES)


def _optimizer(kind: str, model: nn.Module):
    if kind == "adam":
        return torch.optim.Adam(model.parameters(), lr=0.05)
    if kind == "sgd_momentum":
        return torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    raise AssertionError(f"unknown optimizer kind {kind!r}")


def _train(model, optimizer, loader, params, **inputs):
    return TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": loader,
            "optimizer": optimizer,
            "loss_fn": nn.CrossEntropyLoss(),
            **inputs,
        },
        {"device": "cpu", **params},
    )


def _assert_same_weights(left: nn.Module, right: nn.Module) -> None:
    left_sd, right_sd = left.state_dict(), right.state_dict()
    assert left_sd.keys() == right_sd.keys()
    for key in left_sd:
        assert torch.allclose(left_sd[key], right_sd[key], **TOL), (
            f"parameter {key!r} diverged between the straight and resumed runs"
        )


@pytest.mark.parametrize("optimizer_kind", ["adam", "sgd_momentum"])
def test_resume_matches_straight_run(optimizer_kind: str, ckpt_path: str) -> None:
    """4 epochs straight == 2 epochs + checkpoint + restore + 2 epochs.

    This is the headline #118 regression. It fails before the fix for both
    optimizers because the resumed leg starts from empty optimizer state:
    Adam restarts its bias correction and SGD restarts its momentum buffer.
    """
    loader = _loader()

    straight_model = _model()
    straight_opt = _optimizer(optimizer_kind, straight_model)
    straight = _train(straight_model, straight_opt, loader, {"epochs": TOTAL_EPOCHS})

    # ---- leg 1: train SPLIT_EPOCH epochs, then checkpoint --------------
    model = _model()
    optimizer = _optimizer(optimizer_kind, model)
    first = _train(model, optimizer, loader, {"epochs": SPLIT_EPOCH})
    CheckpointSaverNode().execute(
        {"model": first["model"], "optimizer": optimizer, "losses": first["losses"]},
        {"path": ckpt_path, "epoch": SPLIT_EPOCH},
    )

    # ---- leg 2: fresh objects, restore, train the remaining epochs -----
    resumed_model = _model()
    resumed_opt = _optimizer(optimizer_kind, resumed_model)
    restored = CheckpointLoaderNode().execute(
        {"model": resumed_model, "optimizer": resumed_opt},
        {"path": ckpt_path, "device": "cpu"},
    )
    assert restored["epoch"] == SPLIT_EPOCH
    second = _train(
        restored["model"], restored["optimizer"], loader,
        {"epochs": TOTAL_EPOCHS - SPLIT_EPOCH},
    )

    straight_tail = straight["losses"][SPLIT_EPOCH:].tolist()
    assert second["losses"].tolist() == pytest.approx(straight_tail, rel=1e-5, abs=1e-7)
    _assert_same_weights(straight["model"], second["model"])


@pytest.mark.parametrize("optimizer_kind", ["adam", "sgd_momentum"])
def test_resume_through_start_epoch_port(optimizer_kind: str, ckpt_path: str) -> None:
    """The same equivalence through the wiring a user actually builds.

    ``CheckpointLoader.epoch`` goes into ``TrainingLoop.start_epoch`` and the
    ``epochs`` param keeps its absolute meaning, so the resumed node runs only
    the epochs that are left instead of another full ``epochs`` of them.
    """
    loader = _loader()

    straight_model = _model()
    straight = _train(
        straight_model, _optimizer(optimizer_kind, straight_model), loader,
        {"epochs": TOTAL_EPOCHS},
    )

    model = _model()
    optimizer = _optimizer(optimizer_kind, model)
    first = _train(model, optimizer, loader, {"epochs": SPLIT_EPOCH})
    CheckpointSaverNode().execute(
        {"model": first["model"], "optimizer": optimizer},
        {"path": ckpt_path, "epoch": SPLIT_EPOCH},
    )

    resumed_model = _model()
    restored = CheckpointLoaderNode().execute(
        {"model": resumed_model, "optimizer": _optimizer(optimizer_kind, resumed_model)},
        {"path": ckpt_path, "device": "cpu"},
    )
    second = _train(
        restored["model"], restored["optimizer"], loader,
        {"epochs": TOTAL_EPOCHS},
        start_epoch=restored["epoch"],
    )

    assert second["metrics"]["total_epochs_run"] == TOTAL_EPOCHS - SPLIT_EPOCH
    assert second["metrics"]["start_epoch"] == SPLIT_EPOCH
    assert second["metrics"]["last_epoch"] == TOTAL_EPOCHS
    assert second["losses"].tolist() == pytest.approx(
        straight["losses"][SPLIT_EPOCH:].tolist(), rel=1e-5, abs=1e-7
    )
    _assert_same_weights(straight["model"], second["model"])


def test_optimizer_state_survives_the_training_loop() -> None:
    """The optimizer handed to the node is the one that gets trained.

    Before #118 the node built its own optimizer internally, so the object a
    downstream ``CheckpointSaver`` sees still had empty ``state`` -- the save
    side of a resume was broken as well as the load side.
    """
    model = _model()
    optimizer = _optimizer("sgd_momentum", model)
    assert optimizer.state == {}

    result = _train(model, optimizer, _loader(), {"epochs": 1})

    assert result["model"] is model
    assert optimizer.state, "the incoming optimizer accumulated no state"
    buffers = [s["momentum_buffer"] for s in optimizer.state.values()]
    assert all(b is not None and b.abs().sum() > 0 for b in buffers)
    saved = optimizer.state_dict()
    assert saved["state"], "optimizer.state_dict() would checkpoint nothing"


def test_start_epoch_makes_progress_events_absolute() -> None:
    """Progress events, logs and metrics count epochs from start_epoch."""
    model = _model()
    events: list[dict] = []
    TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": _loader(),
            "optimizer": _optimizer("adam", model),
            "loss_fn": nn.CrossEntropyLoss(),
            "start_epoch": 3,
        },
        {"epochs": 5, "device": "cpu"},
        progress_callback=events.append,
    )

    epochs = [e for e in events if e.get("event") == "epoch"]
    assert [e["epoch"] for e in epochs] == [4, 5]
    assert all(e["total_epochs"] == 5 for e in epochs)
    assert all(e["start_epoch"] == 3 for e in epochs)

    config = next(e for e in events if e.get("event") == "config")
    assert config["config"]["start_epoch"] == 3
    assert config["config"]["epochs"] == 5


def test_progress_payload_unchanged_without_start_epoch() -> None:
    """A non-resumed run emits exactly the keys it emitted before #118."""
    model = _model()
    events: list[dict] = []
    TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": _loader(),
            "optimizer": _optimizer("adam", model),
            "loss_fn": nn.CrossEntropyLoss(),
        },
        {"epochs": 2, "device": "cpu"},
        progress_callback=events.append,
    )
    epochs = [e for e in events if e.get("event") == "epoch"]
    assert [e["epoch"] for e in epochs] == [1, 2]
    assert all("start_epoch" not in e for e in epochs)


def test_start_epoch_at_or_past_total_runs_nothing() -> None:
    """Resuming a finished run is a no-op, not a crash."""
    model = _model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = _train(
        model, _optimizer("adam", model), _loader(), {"epochs": 3}, start_epoch=3,
    )
    assert result["losses"].numel() == 0
    assert result["metrics"]["total_epochs_run"] == 0
    for key, value in before.items():
        assert torch.equal(value, result["model"].state_dict()[key])


def test_start_epoch_accepts_tensor_and_rejects_garbage() -> None:
    """The SCALAR port takes whatever a checkpoint stored; junk means 0."""
    for value, expected in [
        (torch.tensor(2), 2),
        (2.0, 2),
        (-5, 0),
        (None, 0),
        ("not-a-number", 0),
    ]:
        model = _model()
        result = _train(
            model, _optimizer("adam", model), _loader(), {"epochs": 3},
            start_epoch=value,
        )
        assert result["metrics"]["start_epoch"] == expected, f"start_epoch={value!r}"


def test_optimizer_is_rebound_when_parameters_are_replaced() -> None:
    """A structurally identical model gets the optimizer's state re-keyed.

    This is the ``rebound`` branch of the resume rule: different parameter
    *objects*, same shapes. The momentum buffers have to survive and to end up
    attached to the new parameters.
    """
    model = _model()
    optimizer = _optimizer("sgd_momentum", model)
    _train(model, optimizer, _loader(), {"epochs": 1})
    buffers_before = [
        s["momentum_buffer"].clone() for s in optimizer.state.values()
    ]

    replacement = _model()
    replacement.load_state_dict(model.state_dict())
    assert all(a is not b for a, b in zip(model.parameters(), replacement.parameters()))

    _train(replacement, optimizer, _loader(), {"epochs": 1})

    assert set(optimizer.state) == set(replacement.parameters())
    assert len(optimizer.state) == len(buffers_before)
    assert all(
        s["momentum_buffer"] is not None for s in optimizer.state.values()
    )
    # State carried over rather than restarting from zero.
    assert any(
        not torch.equal(before, s["momentum_buffer"])
        for before, s in zip(buffers_before, optimizer.state.values())
    )


def test_optimizer_is_rebuilt_for_a_different_model() -> None:
    """The ``rebuilt`` branch: an optimizer from another architecture."""
    other = nn.Linear(IN_FEATURES, OUT_CLASSES + 5)
    optimizer = torch.optim.SGD(other.parameters(), lr=0.1, momentum=0.9)
    optimizer.state[next(other.parameters())] = {
        "momentum_buffer": torch.ones(OUT_CLASSES + 5, IN_FEATURES)
    }

    model = _model()
    result = _train(model, optimizer, _loader(), {"epochs": 1})

    assert result["losses"].shape == (1,)
    # The stale state did not leak into this run's optimizer.
    assert all(p.shape[0] == OUT_CLASSES for p in model.parameters() if p.dim() == 2)


# ---------------------------------------------------------------------------
# LR schedule continuation
# ---------------------------------------------------------------------------

SCHED_EPOCHS = 6
SCHED_SPLIT = 3


def _scheduler(kind: str, optimizer):
    import torch.optim.lr_scheduler as sched_module

    if kind == "StepLR":
        return sched_module.StepLR(optimizer, step_size=2, gamma=0.5)
    if kind == "CosineAnnealingLR":
        return sched_module.CosineAnnealingLR(optimizer, T_max=SCHED_EPOCHS)
    raise AssertionError(f"unknown scheduler kind {kind!r}")


def _straight_lr_run(kind: str, loader):
    model = _model()
    optimizer = _optimizer("sgd_momentum", model)
    return _train(
        model, optimizer, loader, {"epochs": SCHED_EPOCHS},
        lr_scheduler=_scheduler(kind, optimizer),
    )


@pytest.mark.parametrize("scheduler_kind", ["StepLR", "CosineAnnealingLR"])
def test_scheduler_continues_from_checkpointed_state(
    scheduler_kind: str, ckpt_path: str
) -> None:
    """LR at absolute epoch N is the same straight-through and after a resume.

    The scheduler travels through the checkpoint, which is the exact path:
    ``last_epoch``/``_step_count``/``base_lrs`` are all restored verbatim.
    """
    loader = _loader()
    straight = _straight_lr_run(scheduler_kind, loader)

    model = _model()
    optimizer = _optimizer("sgd_momentum", model)
    scheduler = _scheduler(scheduler_kind, optimizer)
    first = _train(
        model, optimizer, loader, {"epochs": SCHED_SPLIT}, lr_scheduler=scheduler,
    )
    CheckpointSaverNode().execute(
        {"model": first["model"], "optimizer": optimizer, "lr_scheduler": scheduler},
        {"path": ckpt_path, "epoch": SCHED_SPLIT},
    )

    resumed_model = _model()
    resumed_opt = _optimizer("sgd_momentum", resumed_model)
    resumed_sched = _scheduler(scheduler_kind, resumed_opt)
    restored = CheckpointLoaderNode().execute(
        {"model": resumed_model, "optimizer": resumed_opt, "lr_scheduler": resumed_sched},
        {"path": ckpt_path, "device": "cpu"},
    )
    assert restored["lr_scheduler"] is resumed_sched
    assert resumed_sched.last_epoch == SCHED_SPLIT

    second = _train(
        restored["model"], restored["optimizer"], loader, {"epochs": SCHED_EPOCHS},
        lr_scheduler=restored["lr_scheduler"], start_epoch=restored["epoch"],
    )

    assert second["metrics"]["lr_history"] == pytest.approx(
        straight["metrics"]["lr_history"][SCHED_SPLIT:], rel=1e-9
    )
    assert second["losses"].tolist() == pytest.approx(
        straight["losses"][SCHED_SPLIT:].tolist(), rel=1e-5, abs=1e-7
    )
    _assert_same_weights(straight["model"], second["model"])


@pytest.mark.parametrize("scheduler_kind", ["StepLR", "CosineAnnealingLR"])
def test_scheduler_fast_forwards_when_checkpoint_has_no_scheduler_state(
    scheduler_kind: str, ckpt_path: str
) -> None:
    """Fallback path: only ``start_epoch`` is known, no saved scheduler state.

    The scheduler is rebuilt by the graph (as ``LRScheduler`` would) over the
    optimizer at its original LR and then fast-forwarded, which has to land on
    the same LR the straight run is using at that epoch.
    """
    loader = _loader()
    straight = _straight_lr_run(scheduler_kind, loader)

    model = _model()
    optimizer = _optimizer("sgd_momentum", model)
    first = _train(
        model, optimizer, loader, {"epochs": SCHED_SPLIT},
        lr_scheduler=_scheduler(scheduler_kind, optimizer),
    )
    # Deliberately NOT passing lr_scheduler -- a pre-#118 style checkpoint.
    CheckpointSaverNode().execute(
        {"model": first["model"], "optimizer": optimizer},
        {"path": ckpt_path, "epoch": SCHED_SPLIT},
    )

    resumed_model = _model()
    resumed_opt = _optimizer("sgd_momentum", resumed_model)
    # Built before the checkpoint is loaded, so base_lrs are the run's original
    # LRs -- that is the ordering the graph produces (LRScheduler consumes the
    # Optimizer node's output) and what the fast-forward documents it needs.
    resumed_sched = _scheduler(scheduler_kind, resumed_opt)
    restored = CheckpointLoaderNode().execute(
        {"model": resumed_model, "optimizer": resumed_opt, "lr_scheduler": resumed_sched},
        {"path": ckpt_path, "device": "cpu"},
    )
    assert resumed_sched.last_epoch == 0, "nothing should have restored the schedule"

    second = _train(
        restored["model"], restored["optimizer"], loader, {"epochs": SCHED_EPOCHS},
        lr_scheduler=resumed_sched, start_epoch=restored["epoch"],
    )

    assert resumed_sched._step_count == 1 + SCHED_EPOCHS, (
        "_step_count must keep counting through the resume, not restart"
    )
    assert second["metrics"]["lr_history"] == pytest.approx(
        straight["metrics"]["lr_history"][SCHED_SPLIT:], rel=1e-9
    )
    assert second["losses"].tolist() == pytest.approx(
        straight["losses"][SCHED_SPLIT:].tolist(), rel=1e-5, abs=1e-7
    )


def test_reduce_lr_on_plateau_is_left_alone_by_the_fast_forward() -> None:
    """A metric-driven schedule cannot be replayed; it must not be mangled."""
    import torch.optim.lr_scheduler as sched_module

    model = _model()
    optimizer = _optimizer("sgd_momentum", model)
    scheduler = sched_module.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)

    result = _train(
        model, optimizer, _loader(), {"epochs": 4},
        lr_scheduler=scheduler, start_epoch=2,
    )

    assert result["metrics"]["total_epochs_run"] == 2
    assert optimizer.param_groups[0]["lr"] > 0


def test_old_format_checkpoint_without_scheduler_key_still_loads(ckpt_path: str) -> None:
    """A checkpoint written before #118 has no scheduler key at all."""
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = _model()
    optimizer = _optimizer("sgd_momentum", model)
    legacy = {
        "epoch": SPLIT_EPOCH,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(legacy, str(settings.MODELS_DIR / ckpt_path))

    target_model = _model()
    target_opt = _optimizer("sgd_momentum", target_model)
    scheduler = _scheduler("StepLR", target_opt)

    restored = CheckpointLoaderNode().execute(
        {"model": target_model, "optimizer": target_opt, "lr_scheduler": scheduler},
        {"path": ckpt_path, "device": "cpu"},
    )

    assert restored["epoch"] == SPLIT_EPOCH
    assert restored["lr_scheduler"] is scheduler
    assert scheduler.last_epoch == 0, "an absent key must not disturb the scheduler"
    assert restored["losses"].numel() == 0

    # ...and the same file still loads for a graph with no scheduler wired in.
    plain_model = _model()
    plain = CheckpointLoaderNode().execute(
        {"model": plain_model, "optimizer": _optimizer("sgd_momentum", plain_model)},
        {"path": ckpt_path, "device": "cpu"},
    )
    assert plain["epoch"] == SPLIT_EPOCH
    assert plain["lr_scheduler"] is None


# ---------------------------------------------------------------------------
# Keeping the optimizer means keeping its state on the right device
# ---------------------------------------------------------------------------


def test_state_device_sync_is_a_no_op_when_nothing_moved() -> None:
    """Aligned state must not be rebuilt -- the sync only repairs mismatches."""
    model = _model()
    optimizer = _optimizer("sgd_momentum", model)
    _train(model, optimizer, _loader(), {"epochs": 1})
    buffers = {p: s["momentum_buffer"] for p, s in optimizer.state.items()}

    _sync_optimizer_state_device(optimizer)

    assert {p: s["momentum_buffer"] for p, s in optimizer.state.items()} == buffers


def test_resume_graph_is_wireable() -> None:
    """The new ports type-check through the real graph validator.

    Calling ``execute()`` directly cannot catch a port that no user could
    actually connect, so build the resume graph the editor would produce --
    ``CheckpointLoader.epoch -> TrainingLoop.start_epoch`` (SCALAR) and the
    scheduler round trip (ANY) -- and run it past ``validate_graph``.
    Structural only: graphs containing ``TrainingLoop``/``DataLoader`` are not
    executed in the suite (see test_builtin_examples.py).
    """
    from app.core.graph_engine import validate_graph

    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "data", "type": "SyntheticShapes", "data": {"params": {}}},
        {"id": "loader", "type": "DataLoader", "data": {"params": {"batch_size": 8}}},
        {"id": "model", "type": "SequentialModel", "data": {"params": {}}},
        {"id": "opt", "type": "Optimizer", "data": {"params": {"type": "Adam"}}},
        {"id": "sched", "type": "LRScheduler", "data": {"params": {"type": "StepLR"}}},
        {"id": "loss", "type": "Loss", "data": {"params": {}}},
        {"id": "load", "type": "CheckpointLoader", "data": {"params": {"path": "run.pt"}}},
        {"id": "train", "type": "TrainingLoop", "data": {"params": {"epochs": 4}}},
        {"id": "save", "type": "CheckpointSaver", "data": {"params": {"path": "run.pt", "epoch": 4}}},
    ]

    def edge(source, source_handle, target, target_handle):
        return {
            "id": f"{source}.{source_handle}->{target}.{target_handle}",
            "source": source, "sourceHandle": source_handle,
            "target": target, "targetHandle": target_handle,
        }

    edges = [
        {"id": "t1", "source": "start", "target": "data", "sourceHandle": "trigger", "type": "trigger"},
        {"id": "t2", "source": "start", "target": "model", "sourceHandle": "trigger", "type": "trigger"},
        {"id": "t3", "source": "start", "target": "loss", "sourceHandle": "trigger", "type": "trigger"},
        edge("data", "dataset", "loader", "dataset"),
        edge("model", "model", "opt", "model"),
        edge("model", "model", "load", "model"),
        edge("opt", "optimizer", "sched", "optimizer"),
        edge("opt", "optimizer", "load", "optimizer"),
        edge("sched", "scheduler", "load", "lr_scheduler"),
        edge("load", "model", "train", "model"),
        edge("load", "optimizer", "train", "optimizer"),
        edge("load", "lr_scheduler", "train", "lr_scheduler"),
        edge("load", "epoch", "train", "start_epoch"),
        edge("loader", "dataloader", "train", "dataloader"),
        edge("loss", "loss_fn", "train", "loss_fn"),
        edge("train", "model", "save", "model"),
        edge("load", "optimizer", "save", "optimizer"),
        edge("load", "lr_scheduler", "save", "lr_scheduler"),
    ]

    assert validate_graph(nodes, edges) == []


@requires_cuda
def test_training_on_cuda_moves_reused_optimizer_state() -> None:
    """Reusing the optimizer must not strand its buffers on the old device.

    ``model.to("cuda")`` swaps each parameter's storage but leaves the
    optimizer's momentum buffers on the CPU, so a naive "keep the optimizer"
    would blow up on the first step with a device mismatch.
    """
    loader = _loader()
    model = _model()
    optimizer = _optimizer("sgd_momentum", model)
    _train(model.cpu(), optimizer, loader, {"epochs": 1, "device": "cpu"})
    assert all(
        s["momentum_buffer"].device.type == "cpu" for s in optimizer.state.values()
    )

    result = _train(model, optimizer, loader, {"epochs": 1, "device": "cuda"})

    assert result["losses"].shape == (1,)
    assert all(p.device.type == "cuda" for p in result["model"].parameters())
    assert all(
        s["momentum_buffer"].device.type == "cuda" for s in optimizer.state.values()
    )
