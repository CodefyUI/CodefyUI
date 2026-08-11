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

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from app.config import settings
from app.nodes.io.checkpoint_node import CheckpointLoaderNode, CheckpointSaverNode
from app.nodes.training.training_loop_node import (
    TrainingLoopNode,
    _completed_optimizer_steps,
    _fast_forward_scheduler,
    _prepare_optimizer,
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
    """A per-test checkpoint filename under MODELS_DIR, removed afterwards.

    Derived from the test id rather than a random suffix so parametrized cases
    never collide and a file left behind by a hard crash is traceable.
    """
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    stem = "".join(c if c.isalnum() else "_" for c in request.node.name)
    name = f"_resume_{stem}.pt"
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


def _train(model, optimizer, loader, params, *, context=None, **inputs):
    return TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": loader,
            "optimizer": optimizer,
            "loss_fn": nn.CrossEntropyLoss(),
            **inputs,
        },
        {"device": "cpu", **params},
        context=context,
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


def test_prepare_optimizer_picks_the_documented_branch() -> None:
    """The resume rule itself, asserted on the mode it reports.

    The end-to-end tests below can look similar whichever branch ran, so pin
    the decision directly: same parameter objects -> ``reused``, structurally
    identical replacements -> ``rebound``, anything else -> ``rebuilt``.
    """
    model = _model()
    optimizer = _optimizer("sgd_momentum", model)

    assert _prepare_optimizer(optimizer, model) == (optimizer, "reused")

    replacement = _model()
    rebound, mode = _prepare_optimizer(optimizer, replacement)
    assert mode == "rebound"
    assert rebound is optimizer, "rebinding must not swap the optimizer object"
    tracked = [p for g in rebound.param_groups for p in g["params"]]
    assert all(a is b for a, b in zip(tracked, replacement.parameters()))

    other = nn.Linear(IN_FEATURES, OUT_CLASSES + 5)
    stale = _optimizer("sgd_momentum", other)
    fresh, mode = _prepare_optimizer(stale, model)
    assert mode == "rebuilt"
    assert fresh is not stale
    assert type(fresh) is type(stale)
    tracked = [p for g in fresh.param_groups for p in g["params"]]
    assert all(a is b for a, b in zip(tracked, model.parameters()))


# Which per-parameter state key proves the optimizer continued rather than
# restarted, for each optimizer under test.
_STATE_KEY = {"adam": "exp_avg", "sgd_momentum": "momentum_buffer"}


@pytest.mark.parametrize("optimizer_kind", ["adam", "sgd_momentum"])
def test_optimizer_is_rebound_when_parameters_are_replaced(optimizer_kind: str) -> None:
    """A structurally identical model gets the optimizer's state re-keyed.

    The ``rebound`` branch end to end: different parameter *objects*, same
    shapes. Adam's moments and SGD's momentum buffer both have to survive and
    end up attached to the new parameters.
    """
    key = _STATE_KEY[optimizer_kind]
    model = _model()
    optimizer = _optimizer(optimizer_kind, model)
    _train(model, optimizer, _loader(), {"epochs": 1})
    before = [s[key].clone() for s in optimizer.state.values()]
    assert before and all(b.abs().sum() > 0 for b in before)

    replacement = _model()
    replacement.load_state_dict(model.state_dict())
    assert all(a is not b for a, b in zip(model.parameters(), replacement.parameters()))

    result = _train(replacement, optimizer, _loader(), {"epochs": 1})

    assert result["model"] is replacement
    # Keyed on the NEW parameter objects: the groups were re-bound. A rebuild
    # would have left this optimizer's state sitting on the old parameters.
    assert set(optimizer.state) == set(replacement.parameters())
    after = [s[key] for s in optimizer.state.values()]
    assert len(after) == len(before)
    assert all(a is not None for a in after)
    # State continued rather than restarting from zero.
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_rebind_preserves_per_group_hyperparameters() -> None:
    """Multi-group optimizers keep their per-group overrides and split."""
    model = _model()
    weight, bias = list(model.parameters())
    optimizer = torch.optim.SGD(
        [
            {"params": [weight], "lr": 0.05},
            {"params": [bias], "lr": 0.2, "momentum": 0.5},
        ],
        lr=0.1,
        momentum=0.9,
    )
    model(torch.zeros(2, IN_FEATURES)).sum().backward()
    optimizer.step()
    assert len(optimizer.state) == 2

    replacement = _model()
    rebound, mode = _prepare_optimizer(optimizer, replacement)

    assert mode == "rebound"
    assert [g["lr"] for g in rebound.param_groups] == [0.05, 0.2]
    assert [g["momentum"] for g in rebound.param_groups] == [0.9, 0.5]
    assert [len(g["params"]) for g in rebound.param_groups] == [1, 1]
    new_weight, new_bias = list(replacement.parameters())
    assert rebound.param_groups[0]["params"][0] is new_weight
    assert rebound.param_groups[1]["params"][0] is new_bias
    assert set(rebound.state) == {new_weight, new_bias}


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
# The optimizer passthrough output (#148)
#
# Before this port existed, CheckpointSaver could only be fed from the
# Optimizer node, and the saved state was right only because
# _prepare_optimizer usually mutates that same object in place and because
# the train.model -> save.model edge happens to force train to run first.
# Two assumptions, neither visible on the canvas, and the `rebuilt` branch
# breaks the first one outright.
# ---------------------------------------------------------------------------


def test_the_optimizer_output_carries_the_object_that_was_reused() -> None:
    model = _model()
    optimizer = _optimizer("sgd_momentum", model)

    result = _train(model, optimizer, _loader(), {"epochs": 1})

    assert result["optimizer"] is optimizer
    assert optimizer.state, "the run should have left state on it"


def test_the_optimizer_output_carries_the_object_that_was_rebound() -> None:
    """Rebinding mutates in place, so the output is the same object -- but
    now that is an assertion instead of an assumption."""
    model = _model()
    optimizer = _optimizer("adam", model)
    _train(model, optimizer, _loader(), {"epochs": 1})

    replacement = _model()
    replacement.load_state_dict(model.state_dict())
    result = _train(replacement, optimizer, _loader(), {"epochs": 1})

    assert result["optimizer"] is optimizer
    assert set(optimizer.state) == set(replacement.parameters())


def test_the_optimizer_output_carries_the_REBUILT_object_not_the_input() -> None:
    """#148's actual correctness hole.

    In the ``rebuilt`` branch the optimizer that trains is a NEW object, so
    a CheckpointSaver wired from the Optimizer node stores an optimizer that
    never saw this model -- silently, with a valid-looking checkpoint. The
    passthrough is the only way to reach the real one.
    """
    other = nn.Linear(IN_FEATURES, OUT_CLASSES + 5)
    stale = torch.optim.SGD(other.parameters(), lr=0.1, momentum=0.9)

    model = _model()
    result = _train(model, stale, _loader(), {"epochs": 1})
    trained = result["optimizer"]

    assert trained is not stale
    assert type(trained) is type(stale)
    tracked = [p for g in trained.param_groups for p in g["params"]]
    assert all(a is b for a, b in zip(tracked, model.parameters()))
    # The one that trained has momentum for THIS model; the input has none.
    assert trained.state and set(trained.state) <= set(model.parameters())
    assert not stale.state


def test_the_optimizer_output_round_trips_a_checkpoint(ckpt_path: str) -> None:
    """The wiring #148 asks for, end to end: train -> save -> load.

    Uses the ``rebuilt`` branch on purpose, because that is the case where
    saving the graph's Optimizer node instead would store the wrong state.
    """
    from app.nodes.io.checkpoint_node import CheckpointLoaderNode, CheckpointSaverNode

    other = nn.Linear(IN_FEATURES, OUT_CLASSES + 5)
    stale = torch.optim.SGD(other.parameters(), lr=0.1, momentum=0.9)
    model = _model()
    result = _train(model, stale, _loader(), {"epochs": 2})
    trained = result["optimizer"]
    saved = [s["momentum_buffer"].clone() for s in trained.state.values()]
    assert saved and all(b.abs().sum() > 0 for b in saved)

    CheckpointSaverNode().execute(
        {
            "model": result["model"],
            "optimizer": trained,
            "losses": result["losses"],
        },
        {"path": ckpt_path, "epoch": 2},
    )

    fresh_model = _model()
    fresh_optimizer = torch.optim.SGD(
        fresh_model.parameters(), lr=0.1, momentum=0.9)
    restored = CheckpointLoaderNode().execute(
        {"model": fresh_model, "optimizer": fresh_optimizer},
        {"path": ckpt_path, "device": "cpu"},
    )

    assert restored["epoch"] == 2
    reloaded = [s["momentum_buffer"]
                for s in restored["optimizer"].state.values()]
    assert len(reloaded) == len(saved)
    for before, after in zip(saved, reloaded):
        assert torch.allclose(before, after, **TOL)


def test_the_optimizer_output_wires_to_checkpoint_saver() -> None:
    """The port type-checks through the real validator, which is the only
    thing that proves a user could actually draw this edge (#148)."""
    from app.core.graph_engine import validate_graph

    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "data", "type": "SyntheticShapes", "data": {"params": {}}},
        {"id": "loader", "type": "DataLoader", "data": {"params": {"batch_size": 8}}},
        {"id": "model", "type": "SequentialModel", "data": {"params": {}}},
        {"id": "opt", "type": "Optimizer", "data": {"params": {"type": "Adam"}}},
        {"id": "loss", "type": "Loss", "data": {"params": {}}},
        {"id": "train", "type": "TrainingLoop", "data": {"params": {"epochs": 4}}},
        {"id": "save", "type": "CheckpointSaver", "data": {"params": {"path": "run.pt", "epoch": 4}}},
    ]
    edges = [
        {"id": "t1", "source": "start", "target": "data", "sourceHandle": "trigger", "type": "trigger"},
        {"id": "t2", "source": "start", "target": "model", "sourceHandle": "trigger", "type": "trigger"},
        {"id": "t3", "source": "start", "target": "loss", "sourceHandle": "trigger", "type": "trigger"},
        {"id": "e1", "source": "data", "sourceHandle": "dataset", "target": "loader", "targetHandle": "dataset"},
        {"id": "e2", "source": "model", "sourceHandle": "model", "target": "opt", "targetHandle": "model"},
        {"id": "e3", "source": "model", "sourceHandle": "model", "target": "train", "targetHandle": "model"},
        {"id": "e4", "source": "opt", "sourceHandle": "optimizer", "target": "train", "targetHandle": "optimizer"},
        {"id": "e5", "source": "loader", "sourceHandle": "dataloader", "target": "train", "targetHandle": "dataloader"},
        {"id": "e6", "source": "loss", "sourceHandle": "loss_fn", "target": "train", "targetHandle": "loss_fn"},
        {"id": "e7", "source": "train", "sourceHandle": "model", "target": "save", "targetHandle": "model"},
        # The point of the test: the save path's optimizer now comes from the
        # node that trained it, not from the node that constructed it.
        {"id": "e8", "source": "train", "sourceHandle": "optimizer", "target": "save", "targetHandle": "optimizer"},
    ]

    assert validate_graph(nodes, edges) == []


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


def test_scheduler_ahead_of_start_epoch_warns_and_keeps_its_own_position(caplog) -> None:
    """A schedule/epoch desync is exactly what this issue removes -- say so."""
    import logging

    model = _model()
    optimizer = _optimizer("sgd_momentum", model)
    scheduler = _scheduler("StepLR", optimizer)
    for _ in range(5):
        optimizer.step()
        scheduler.step()
    assert scheduler.last_epoch == 5

    with caplog.at_level(logging.WARNING, logger="app.nodes.training.training_loop_node"):
        _train(
            model, optimizer, _loader(), {"epochs": 4},
            lr_scheduler=scheduler, start_epoch=2,
        )

    warnings_seen = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("last_epoch=5" in m and "epoch 2" in m for m in warnings_seen), warnings_seen
    # The scheduler kept its own position rather than being rewound.
    assert scheduler.last_epoch == 5 + 2


def test_replay_failure_degrades_instead_of_failing_the_run(caplog) -> None:
    """A schedule that cannot be replayed must not take the training run down.

    The failure is injected rather than borrowed from a real scheduler
    (``OneCycleLR`` raises once stepped past ``total_steps``) because a real
    one keeps raising from inside the epoch loop too. That in-loop step is
    unguarded on purpose -- a schedule failing during actual training is a
    genuine misconfiguration and predates #118 -- so borrowing it would test
    something other than the replay.
    """
    import logging

    class _BrittleStepLR(torch.optim.lr_scheduler.StepLR):
        """Raises for its next ``_pending_failures`` steps, then behaves.

        Defaults to 0 so the ``_initial_step()`` inside ``__init__`` is not
        the one that trips; the test arms it afterwards.
        """

        _pending_failures = 0

        def step(self, *args, **kwargs):
            if self._pending_failures > 0:
                self._pending_failures -= 1
                raise RuntimeError("this schedule cannot be replayed")
            return super().step(*args, **kwargs)

    model = _model()
    optimizer = _optimizer("sgd_momentum", model)
    scheduler = _BrittleStepLR(optimizer, step_size=2, gamma=0.5)
    scheduler._pending_failures = 1

    with caplog.at_level(logging.WARNING, logger="app.nodes.training.training_loop_node"):
        result = _train(
            model, optimizer, _loader(), {"epochs": 5},
            lr_scheduler=scheduler, start_epoch=3,
        )

    assert result["metrics"]["total_epochs_run"] == 2, "training carried on regardless"
    assert result["metrics"]["start_epoch"] == 3
    messages = [r.getMessage() for r in caplog.records]
    assert any("replaying epoch 1 of 3" in m for m in messages), messages


@pytest.mark.filterwarnings("ignore:custom schedule notice")
def test_replay_only_suppresses_the_step_order_warnings() -> None:
    """The suppression is by message, so a real schedule warning still lands.

    The marker only silences the copy this scheduler emits while being
    *constructed*, which is outside the block under test; the recorder below
    still sees the one raised during the replay.
    """
    import warnings as warnings_mod

    class _NoisyStepLR(torch.optim.lr_scheduler.StepLR):
        def get_lr(self):
            warnings_mod.warn("custom schedule notice", UserWarning)
            return super().get_lr()

    model = _model()
    optimizer = _optimizer("sgd_momentum", model)
    scheduler = _NoisyStepLR(optimizer, step_size=2, gamma=0.5)

    with warnings_mod.catch_warnings(record=True) as seen:
        warnings_mod.simplefilter("always")
        assert _fast_forward_scheduler(scheduler, optimizer, 2) is True

    messages = [str(w.message) for w in seen]
    assert any("custom schedule notice" in m for m in messages), messages
    assert not any("lr_scheduler.step()" in m for m in messages), messages


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


# ---------------------------------------------------------------------------
# #149: honest base_lrs -- a DEFENSIVE invariant, not a fix for a reachable
# bug. Verified directly (see task-2-report.md): build_checkpoint derives
# ``initial_lrs`` from ``g.get("initial_lr", g["lr"]) for g in
# optimizer.param_groups`` -- the SAME live param_groups that
# ``optimizer.state_dict()`` (called in the same dict literal, nothing
# mutating in between) also reads. For any checkpoint this module can
# actually produce, the two are therefore mathematically identical, and
# restoring ``initial_lrs`` in CheckpointLoaderNode is a provable no-op
# against the installed torch 2.11.0+cu128 -- ``Optimizer.load_state_dict``'s
# ``update_group`` already carries a saved ``initial_lr`` through by
# accident whenever the checkpoint's own dict has it, and when it does not,
# a freshly built scheduler's own ``setdefault("initial_lr", lr)`` lands on
# the identical number this key would have restored.
#
# What the key buys instead: CodefyUI's OWN contract stops depending on
# ``optimizer.state_dict()``/``load_state_dict()`` choosing to carry
# ``initial_lr`` through -- it is read from ``optimizer.param_groups``
# directly and restored explicitly. The test below proves that decoupling
# is real by constructing the one situation where it would matter: a
# ``state_dict()`` that does NOT serialise ``initial_lr`` while the live
# object still carries it. No real torch behaviour produces that today
# (confirmed empirically -- see the report); it stands in for "a future
# torch change, or a plugin's custom Optimizer subclass, alters what gets
# serialised."


def test_checkpoint_captures_initial_lr_explicitly(ckpt_path: str) -> None:
    """The SAVE side: ``initial_lrs`` prefers the scheduler's true origin.

    Just that the key exists and, once a scheduler has decayed the live lr,
    holds the ORIGINAL rather than the current value -- using ordinary
    torch behaviour throughout, so (as the section note above explains)
    this checkpoint's ``optimizer_state_dict`` also happens to carry
    ``initial_lr`` by the same accident. That is expected, not a gap: this
    test pins the CAPTURE logic's own correctness, independent of whether
    anything downstream needed it.
    """
    model = _model()
    optimizer = _optimizer("sgd_momentum", model)
    scheduler = _scheduler("StepLR", optimizer)  # step_size=2, gamma=0.5
    for _ in range(2):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05), (
        "the decay must actually have happened, or this test proves nothing"
    )

    CheckpointSaverNode().execute(
        {"model": model, "optimizer": optimizer},
        {"path": ckpt_path, "epoch": 1},
    )
    raw = torch.load(settings.MODELS_DIR / ckpt_path, map_location="cpu",
                     weights_only=True)
    assert raw["initial_lrs"] == pytest.approx([0.1])


def test_initial_lr_is_captured_from_the_live_optimizer_not_its_state_dict(
    ckpt_path: str,
) -> None:
    """The one real divergence the defensive framing protects against.

    ``optimizer.state_dict`` is patched to drop ``initial_lr`` from what it
    serialises -- standing in for a torch (or custom-optimizer) behaviour
    change -- while leaving it on the LIVE ``param_groups`` untouched.
    ``build_checkpoint`` must still capture the truth (it reads
    ``param_groups`` directly, not through the crippled ``state_dict()``),
    and ``CheckpointLoaderNode`` must still restore it from ``initial_lrs``
    even though ``optimizer_state_dict`` -- the ordinary carrier -- lacks
    it. This is the shape of test the brief asked for ("fails on the
    history-dependent path specifically"); this codebase does not have a
    real history that produces it, so the test manufactures the one
    dependency the code takes on torch's serialisation choices, rather than
    hand-editing a value no writer would ever produce.
    """
    from unittest import mock

    from app.core import checkpoints

    model = _model()
    optimizer = _optimizer("sgd_momentum", model)
    scheduler = _scheduler("StepLR", optimizer)  # step_size=2, gamma=0.5
    for _ in range(2):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["initial_lr"] == pytest.approx(0.1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)

    real_state_dict = optimizer.state_dict

    def _stripped_state_dict():
        state = real_state_dict()
        for group in state["param_groups"]:
            group.pop("initial_lr", None)
        return state

    with mock.patch.object(optimizer, "state_dict",
                           side_effect=_stripped_state_dict):
        target = checkpoints.write_checkpoint(
            ckpt_path, model, optimizer, epoch=4)

    raw = torch.load(target, map_location="cpu", weights_only=True)
    assert "initial_lr" not in raw["optimizer_state_dict"]["param_groups"][0], (
        "the patched state_dict() must actually have dropped the key, or "
        "this test is not exercising the simulated divergence"
    )
    assert raw["initial_lrs"] == pytest.approx([0.1]), (
        "captured from the LIVE param_groups, independent of what the "
        "(crippled) state_dict() serialised"
    )

    resumed_model = _model()
    resumed_opt = _optimizer("sgd_momentum", resumed_model)
    restored = CheckpointLoaderNode().execute(
        {"model": resumed_model, "optimizer": resumed_opt},
        {"path": ckpt_path, "device": "cpu"},
    )
    restored_opt = restored["optimizer"]
    assert restored_opt.param_groups[0]["lr"] == pytest.approx(0.05), (
        "the resumed lr itself must still be the checkpoint's current value"
    )
    assert restored_opt.param_groups[0]["initial_lr"] == pytest.approx(0.1)

    fresh_scheduler = torch.optim.lr_scheduler.StepLR(restored_opt, step_size=1)
    assert fresh_scheduler.base_lrs == pytest.approx([0.1])


# ---------------------------------------------------------------------------
# #203: periodic checkpointing -- TrainingLoop.checkpoint_every
# ---------------------------------------------------------------------------
#
# A server crash mid-run survives nothing today: CheckpointSaver only fires
# after TrainingLoop returns (never, if the process dies first) and the
# interrupt checkpoint only fires on a COOPERATIVE stop (never, on a
# SIGKILL/OOM-kill/server-restart -- NOT power loss either; see
# save_periodic_checkpoint's docstring). checkpoint_every makes a HEALTHY
# run write an ordinary checkpoint -- same core.checkpoints machinery, same
# exec_run_artifacts row discipline as the interrupt path -- every N
# completed epochs, so at most N epochs are ever at risk.
#
# The tested boundary below is the QUEUE (ctx.outbox.drain()), not an
# actual exec_run_artifacts ROW -- the same boundary
# test_cancellation.py's own interrupt-checkpoint tests use, for the same
# reason: proving the queue -> row mapping itself (ArtifactSignal ->
# RunService._record_artifact -> RunStore.add_artifact) is generic and
# already covered once, for every producer, by
# test_run_service.py::test_log_artifact_registers_a_row_and_announces_it.
# save_periodic_checkpoint calls the identical context.log_artifact(...)
# API that test exercises, so re-proving the row lands here would just
# duplicate that coverage under a different producer.


def _recording_context(**kwargs):
    """A context that records artifacts, as a real interactive/queued run
    does. Mirrors test_cancellation.py's helper of the same shape."""
    from app.core.execution_context import ExecutionContext

    return ExecutionContext(signals_recorded=True, **kwargs)


@pytest.fixture
def periodic_dir(tmp_path, monkeypatch):
    """Point MODELS_DIR at a per-test directory; yield its periodic folder.

    Isolated rather than the shared real MODELS_DIR (unlike ``ckpt_path``
    above) because these tests assert exact file COUNTS and NAMES, which a
    directory other tests also write into cannot support.
    """
    from app.core.checkpoints import PERIODIC_DIRNAME

    models = tmp_path / "data" / "models"
    models.mkdir(parents=True)
    monkeypatch.setattr(settings, "MODELS_DIR", models)
    return models / PERIODIC_DIRNAME


def _checkpoint_signals(ctx):
    """Drain and filter to ``checkpoint`` signals -- and assert none of them
    were EVICTED first. The outbox is drop-oldest under load (see
    ``save_periodic_checkpoint``'s "always-present" orphan-window note); an
    evicted checkpoint signal is a file with no row, which every test using
    this helper is otherwise implicitly assuming did not happen. One line,
    so that assumption is checked rather than merely made."""
    signals, dropped = ctx.outbox.drain()
    assert dropped == 0, f"{dropped} signal(s) evicted before delivery"
    return [s for s in signals if getattr(s, "kind", None) == "checkpoint"]


def test_checkpoint_every_defaults_to_off(periodic_dir):
    """0 = disabled, and 0 is the default -- matching every sibling knob on
    this node (early_stopping_patience, grad_clip_norm, max_steps)."""
    ctx = _recording_context()
    model = _model()
    TrainingLoopNode().execute(
        {"model": model, "dataloader": _loader(),
         "optimizer": _optimizer("sgd_momentum", model),
         "loss_fn": nn.CrossEntropyLoss()},
        {"epochs": 4, "device": "cpu"},  # checkpoint_every omitted
        context=ctx,
    )
    assert not periodic_dir.exists(), "nothing should be written by default"
    assert _checkpoint_signals(ctx) == []


def test_checkpoint_every_writes_a_file_and_row_at_each_multiple(periodic_dir):
    """Fires on the ABSOLUTE epoch number, at every multiple -- one distinct
    file and one exec_run_artifacts row per event, exactly like the
    interrupt path's own artifact discipline."""
    ctx = _recording_context()
    model = _model()
    TrainingLoopNode().execute(
        {"model": model, "dataloader": _loader(),
         "optimizer": _optimizer("sgd_momentum", model),
         "loss_fn": nn.CrossEntropyLoss()},
        {"epochs": 5, "checkpoint_every": 2, "device": "cpu"},
        context=ctx,
    )
    # epochs=5 at every-2 fires at absolute epoch 2 and 4 (not 5: 5 % 2 != 0)
    files = sorted(p.name for p in periodic_dir.iterdir())
    assert len(files) == 2, files

    artifacts = _checkpoint_signals(ctx)
    assert [a.meta.get("epoch") for a in artifacts] == [2, 4]
    assert all(a.meta.get("reason") == "periodic" for a in artifacts)
    assert all(a.kind == "checkpoint" for a in artifacts)
    # The row IS the file: every artifact path this run logged exists.
    assert {Path(a.path).name for a in artifacts} == set(files)


def test_checkpoint_every_uses_absolute_epochs_on_a_resumed_run(periodic_dir):
    """The discriminating case a fully-successful, never-resumed run cannot
    exercise: ``start_epoch`` combined with ``checkpoint_every`` must
    produce ABSOLUTE epoch numbers, not epochs-completed-THIS-CALL.

    epochs=6, checkpoint_every=2, start_epoch=2 runs absolute epochs
    3,4,5,6 (call-relative 1,2,3,4). The correct, absolute-indexed
    implementation fires at absolute 4 and 6. A regression that counted
    epochs completed in THIS call instead (an easy mistake given this
    file's own "arrays are per call" / "epochs are absolute" distinction)
    would report [2, 4] -- call-relative counts mistaken for epoch
    numbers -- which also happens to be the wrong SET of firing points,
    not just a mislabelling: this pins both where periodic checkpoints
    land AND what they call themselves, in the realistic crash -> resume
    -> crash again scenario #203 exists for.
    """
    ctx = _recording_context()
    model = _model()
    TrainingLoopNode().execute(
        {"model": model, "dataloader": _loader(),
         "optimizer": _optimizer("sgd_momentum", model),
         "loss_fn": nn.CrossEntropyLoss(), "start_epoch": 2},
        {"epochs": 6, "checkpoint_every": 2, "device": "cpu"},
        context=ctx,
    )
    artifacts = _checkpoint_signals(ctx)
    assert [a.meta.get("epoch") for a in artifacts] == [4, 6]


def test_checkpoint_every_declines_when_the_run_records_no_artifacts(periodic_dir):
    """The same "no row, no file" rule as the interrupt path (#122): a run
    with nowhere durable to put the row must not write the file either."""
    from app.core.execution_context import ExecutionContext

    ctx = ExecutionContext()  # signals_recorded defaults to False
    assert ctx.can_record_artifacts() is False
    model = _model()
    TrainingLoopNode().execute(
        {"model": model, "dataloader": _loader(),
         "optimizer": _optimizer("sgd_momentum", model),
         "loss_fn": nn.CrossEntropyLoss()},
        {"epochs": 3, "checkpoint_every": 1, "device": "cpu"},
        context=ctx,
    )
    assert not periodic_dir.exists() or list(periodic_dir.iterdir()) == [], (
        "a run that cannot record artifacts still wrote a periodic checkpoint"
    )


def test_resume_from_a_periodic_checkpoint_matches_the_straight_run(periodic_dir):
    """The #203 acceptance criterion: kill-and-resume through the ordinary
    CheckpointLoader -> start_epoch wiring, with no new concepts -- a
    periodic checkpoint resumes exactly like a hand-placed one."""
    loader = _loader()
    straight_model = _model()
    straight = _train(
        straight_model, _optimizer("sgd_momentum", straight_model), loader,
        {"epochs": TOTAL_EPOCHS},
    )

    ctx = _recording_context()
    model = _model()
    optimizer = _optimizer("sgd_momentum", model)
    TrainingLoopNode().execute(
        {"model": model, "dataloader": loader, "optimizer": optimizer,
         "loss_fn": nn.CrossEntropyLoss()},
        {"epochs": SPLIT_EPOCH, "checkpoint_every": SPLIT_EPOCH, "device": "cpu"},
        context=ctx,
    )
    artifacts = _checkpoint_signals(ctx)
    assert len(artifacts) == 1
    checkpoint_path = artifacts[0].path

    resumed_model = _model()
    restored = CheckpointLoaderNode().execute(
        {"model": resumed_model, "optimizer": _optimizer("sgd_momentum", resumed_model)},
        {"path": checkpoint_path, "device": "cpu"},
    )
    assert restored["epoch"] == SPLIT_EPOCH
    second = _train(
        restored["model"], restored["optimizer"], loader,
        {"epochs": TOTAL_EPOCHS}, start_epoch=restored["epoch"],
    )

    assert second["metrics"]["total_epochs_run"] == TOTAL_EPOCHS - SPLIT_EPOCH
    assert second["losses"].tolist() == pytest.approx(
        straight["losses"][SPLIT_EPOCH:].tolist(), rel=1e-5, abs=1e-7
    )
    _assert_same_weights(straight["model"], second["model"])


# ---------------------------------------------------------------------------
# Which clock the fast-forward replays (#316)
#
# The replay above steps the scheduler once per completed EPOCH, which is
# exactly right while the loop also steps it once per epoch. #303 added
# ``scheduler_step=optimizer_step``, and the replay went on counting epochs:
# a per-step schedule resumed after 1000 optimizer steps was fast-forwarded
# by a handful of steps and then trained on a learning rate from the wrong
# end of the schedule, with nothing said anywhere.
# ---------------------------------------------------------------------------

#: 24 samples at batch_size 6 (``_loader``) is 4 batches, so a per-step run
#: takes 4 optimizer steps per epoch and a 3-epoch resume owes the schedule
#: 12 steps rather than 3.
STEPS_PER_EPOCH = 4
PER_STEP = {"scheduler_step": "optimizer_step"}


def _per_step_cosine(optimizer):
    """A cosine denominated in this run's optimizer steps, not its epochs."""
    import torch.optim.lr_scheduler as sched_module

    return sched_module.CosineAnnealingLR(
        optimizer, T_max=SCHED_EPOCHS * STEPS_PER_EPOCH)


def _resume_warnings(context):
    from app.core.execution_context import WarningSignal

    signals, _dropped = context.outbox.drain()
    return [s for s in signals if isinstance(s, WarningSignal)]


def _unmeasurable_loader():
    """A loader with no ``len()`` -- an ``IterableDataset``, 2 batches/epoch.

    The case where the completed step count is NOT recoverable: epochs
    cannot be converted into optimizer steps without a batch count.
    """
    class _Stream(torch.utils.data.IterableDataset):
        def __iter__(self):
            gen = torch.Generator().manual_seed(4321)
            for _ in range(4):
                yield (torch.randn(IN_FEATURES, generator=gen),
                       torch.randint(0, OUT_CLASSES, (1,), generator=gen).squeeze())

    loader = torch.utils.data.DataLoader(_Stream(), batch_size=2)
    with pytest.raises(TypeError):
        len(loader)
    return loader


def test_a_per_step_resume_replays_optimizer_steps_not_epochs() -> None:
    """The headline #316 regression, end to end through the node.

    A per-step run of SCHED_EPOCHS epochs takes SCHED_EPOCHS x
    STEPS_PER_EPOCH scheduler steps, so the resumed leg's learning-rate
    trajectory can only match the straight run's if the fast-forward
    replayed SCHED_SPLIT x STEPS_PER_EPOCH of them. Before the fix it
    replayed SCHED_SPLIT, leaving the cosine near the top of its curve
    while the straight run was already half way down it.
    """
    loader = _loader()

    straight_model = _model()
    straight_opt = _optimizer("sgd_momentum", straight_model)
    straight = _train(
        straight_model, straight_opt, loader,
        {"epochs": SCHED_EPOCHS, **PER_STEP},
        lr_scheduler=_per_step_cosine(straight_opt),
    )

    resumed_model = _model()
    resumed_opt = _optimizer("sgd_momentum", resumed_model)
    resumed_sched = _per_step_cosine(resumed_opt)
    second = _train(
        resumed_model, resumed_opt, loader,
        {"epochs": SCHED_EPOCHS, **PER_STEP},
        lr_scheduler=resumed_sched, start_epoch=SCHED_SPLIT,
    )

    assert resumed_sched.last_epoch == SCHED_EPOCHS * STEPS_PER_EPOCH, (
        "the schedule should end where a straight per-step run ends"
    )
    assert second["metrics"]["lr_history"] == pytest.approx(
        straight["metrics"]["lr_history"][SCHED_SPLIT:], rel=1e-9
    )


def test_the_per_step_fast_forward_lands_on_a_freshly_stepped_scheduler() -> None:
    """The replay is measured against the only reference there is: the same
    scheduler stepped the same number of times from scratch."""
    reference_model = _model()
    reference_opt = _optimizer("sgd_momentum", reference_model)
    reference = _per_step_cosine(reference_opt)
    for _ in range(SCHED_SPLIT * STEPS_PER_EPOCH):
        reference_opt.step()
        reference.step()

    model = _model()
    optimizer = _optimizer("sgd_momentum", model)
    scheduler = _per_step_cosine(optimizer)
    assert _fast_forward_scheduler(
        scheduler, optimizer, SCHED_SPLIT,
        replay_steps=SCHED_SPLIT * STEPS_PER_EPOCH) is True

    assert scheduler.last_epoch == reference.last_epoch
    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        reference_opt.param_groups[0]["lr"], rel=1e-12)


def test_the_issues_own_example_lands_deep_in_the_anneal() -> None:
    """#316's numbers: warmup_cosine(total_steps=1500) resumed after 1000
    optimizer steps. Replaying epochs instead leaves it in the WARMUP, at a
    rising learning rate, for a run that is two thirds finished.
    """
    from app.nodes.training.lr_scheduler_node import LRSchedulerNode
    from app.nodes.training.training_loop_node import _prepare_scheduler

    warmup_steps, total_steps = 100, 1500
    completed = _completed_optimizer_steps(
        start_epoch=10, batches_per_epoch=100, accumulate_steps=1)
    assert completed == 1000, "10 epochs of 100 batches is 1000 steps"

    def build():
        model = _model()
        optimizer = _optimizer("sgd_momentum", model)
        scheduler = LRSchedulerNode().execute(
            {"optimizer": optimizer},
            {"type": "warmup_cosine", "warmup_steps": warmup_steps,
             "total_steps": total_steps},
        )["scheduler"]
        return optimizer, scheduler

    optimizer, scheduler = build()
    # The LR the schedule ramps UP to and then anneals away from -- not the
    # live ``lr``, which the warmup's first step has already pulled to
    # nearly zero by the time the scheduler exists.
    peak_lr = optimizer.param_groups[0]["initial_lr"]
    _prepare_scheduler(scheduler, optimizer, 10, "reused", None,
                       per_step=True, completed_steps=completed)
    per_step_lr = optimizer.param_groups[0]["lr"]

    epoch_optimizer, epoch_scheduler = build()
    _prepare_scheduler(epoch_scheduler, epoch_optimizer, 10, "reused", None)
    epoch_clock_lr = epoch_optimizer.param_groups[0]["lr"]

    assert scheduler.last_epoch == completed
    assert epoch_scheduler.last_epoch == 10

    # Two thirds down the cosine: past the ramp, and DESCENDING.
    assert 0.1 * peak_lr < per_step_lr < 0.5 * peak_lr, per_step_lr
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] < per_step_lr

    # The epoch clock leaves the same resume 10 steps into a 100-step warmup
    # ramp: barely off the floor, and RISING. Same checkpoint, same schedule,
    # opposite end of the curve -- and nothing about it looked wrong.
    assert epoch_clock_lr < 0.15 * peak_lr, epoch_clock_lr
    epoch_optimizer.step()
    epoch_scheduler.step()
    assert epoch_optimizer.param_groups[0]["lr"] > epoch_clock_lr


def test_a_per_step_resume_with_no_countable_batches_refuses_and_says_so(
) -> None:
    """(b): when the step count is NOT recoverable, replaying epochs anyway
    is the silently-wrong behaviour this issue is about. Refuse, and name
    the resume route that always works."""
    from app.core.execution_context import ExecutionContext
    from app.nodes.training.training_loop_node import (
        SCHEDULE_NOTE_PREFIX,
        SCHEDULE_RESUME_WARNING_KIND,
    )

    loader = _unmeasurable_loader()
    model = _model()
    optimizer = _optimizer("sgd_momentum", model)
    scheduler = _per_step_cosine(optimizer)
    context = ExecutionContext()

    result = TrainingLoopNode().execute(
        {"model": model, "dataloader": loader, "optimizer": optimizer,
         "loss_fn": nn.CrossEntropyLoss(), "lr_scheduler": scheduler,
         "start_epoch": 1},
        {"epochs": 3, "device": "cpu", **PER_STEP},
        context=context,
    )

    # 2 epochs of 2 batches: the only steps the schedule ever took are the
    # ones this leg actually ran. Nothing was replayed -- least of all the
    # single epoch-step that would have looked like progress.
    assert scheduler.last_epoch == 4
    assert result["metrics"]["total_epochs_run"] == 2

    warnings = _resume_warnings(context)
    assert len(warnings) == 1, warnings
    assert warnings[0].kind == SCHEDULE_RESUME_WARNING_KIND
    assert "optimizer step" in warnings[0].detail
    assert "CheckpointSaver.lr_scheduler" in warnings[0].detail
    assert result["__log__"].startswith(SCHEDULE_NOTE_PREFIX)


def test_an_epoch_mode_resume_still_replays_one_step_per_epoch() -> None:
    """The historical clock, unchanged: one step per completed epoch, and
    nothing to advise about."""
    from app.core.execution_context import ExecutionContext

    loader = _loader()
    model = _model()
    optimizer = _optimizer("sgd_momentum", model)
    scheduler = _scheduler("CosineAnnealingLR", optimizer)
    context = ExecutionContext()

    result = _train(
        model, optimizer, loader, {"epochs": SCHED_EPOCHS},
        lr_scheduler=scheduler, start_epoch=SCHED_SPLIT, context=context,
    )

    assert scheduler.last_epoch == SCHED_EPOCHS
    assert _resume_warnings(context) == []
    assert "__log__" not in result


def test_a_plateau_resume_keeps_its_metric_driven_advisory_in_per_step_mode(
) -> None:
    """ReduceLROnPlateau stays per-epoch in BOTH modes (#297), so the clock
    question never arises for it -- the advisory it gets must still be the
    one about its plateau history, not the per-step one."""
    import torch.optim.lr_scheduler as sched_module

    from app.core.execution_context import ExecutionContext
    from app.nodes.training.training_loop_node import (
        SCHEDULE_RESUME_WARNING_KIND,
    )

    model = _model()
    optimizer = _optimizer("sgd_momentum", model)
    scheduler = sched_module.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)
    context = ExecutionContext()

    _train(
        model, optimizer, _loader(), {"epochs": 4, **PER_STEP},
        lr_scheduler=scheduler, start_epoch=2, context=context,
    )

    warnings = _resume_warnings(context)
    assert len(warnings) == 1, warnings
    assert warnings[0].kind == SCHEDULE_RESUME_WARNING_KIND
    assert "ReduceLROnPlateau" in warnings[0].detail
    assert "plateau history" in warnings[0].detail


def test_restored_per_step_state_is_checked_against_the_step_count(caplog) -> None:
    """The state-restore path reads the same clock (#316).

    A per-step schedule saved through ``CheckpointSaver.lr_scheduler`` comes
    back at ``last_epoch=<steps>``, which is never the epoch index -- so
    comparing the two accused the one resume route that IS exact of holding a
    checkpoint written at two different times, on every per-step run.
    """
    import logging

    loader = _loader()

    def resume_with(last_epoch: int, **kwargs):
        model = _model()
        optimizer = _optimizer("sgd_momentum", model)
        scheduler = _per_step_cosine(optimizer)
        for _ in range(last_epoch):
            optimizer.step()
            scheduler.step()
        with caplog.at_level(
                logging.WARNING,
                logger="app.nodes.training.training_loop_node"):
            caplog.clear()
            _train(model, optimizer, loader,
                   {"epochs": SCHED_EPOCHS, **kwargs},
                   lr_scheduler=scheduler, start_epoch=SCHED_SPLIT)
        return [r.getMessage() for r in caplog.records
                if r.levelno >= logging.WARNING]

    # Exactly where SCHED_SPLIT epochs of this loader leave a per-step run:
    # nothing to report.
    assert resume_with(SCHED_SPLIT * STEPS_PER_EPOCH, **PER_STEP) == []

    # A genuine desync, still reported -- in the unit the schedule counts in.
    desynced = resume_with(SCHED_SPLIT * STEPS_PER_EPOCH + 3, **PER_STEP)
    assert any("optimizer step" in m for m in desynced), desynced

    # And the epoch clock keeps its own reading: 12 steps is not epoch 3.
    epoch_mode = resume_with(SCHED_SPLIT * STEPS_PER_EPOCH)
    assert any("last_epoch=12" in m and "epoch 3" in m for m in epoch_mode), (
        epoch_mode)


@pytest.mark.parametrize("start_epoch, batches, accumulate, expected", [
    # The plain case: 3 epochs of 4 batches is 12 optimizer steps.
    (3, 4, 1, 12),
    # Accumulation divides by CEILING, because the loop applies an epoch's
    # short tail window as a step of its own: 5 batches at accumulate 2 is
    # 3 steps per epoch, not 2.
    (2, 5, 2, 6),
    (1, 4, 4, 1),
    # Nothing completed means nothing to replay.
    (0, 4, 1, 0),
    # Not recoverable: no len() on the loader, an empty loader, a length
    # that is not a number, a nonsense epoch index.
    (3, None, 1, None),
    (3, 0, 1, None),
    (3, "four", 1, None),
    (-1, 4, 1, None),
])
def test_completed_optimizer_steps_only_answers_when_it_can(
    start_epoch, batches, accumulate, expected
) -> None:
    assert _completed_optimizer_steps(
        start_epoch=start_epoch, batches_per_epoch=batches,
        accumulate_steps=accumulate) == expected
