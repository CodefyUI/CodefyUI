"""Tests for TrainingLoopNode (end-to-end mini training)."""

from __future__ import annotations

import torch
import torch.nn as nn

from app.nodes.training.training_loop_node import TrainingLoopNode


def _make_dataset(n=8, in_features=4, out_classes=2):
    """Tiny supervised dataset (n samples)."""
    X = torch.randn(n, in_features)
    y = torch.randint(0, out_classes, (n,))
    return torch.utils.data.TensorDataset(X, y)


def _make_loader(dataset, batch_size=4):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


def test_node_metadata():
    assert TrainingLoopNode.NODE_NAME == "TrainingLoop"
    assert TrainingLoopNode.CATEGORY == "Training"
    out_names = [p.name for p in TrainingLoopNode.define_outputs()]
    assert "model" in out_names
    assert "losses" in out_names
    assert "metrics" in out_names


def test_basic_training_returns_losses_per_epoch():
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    loader = _make_loader(_make_dataset())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    res = TrainingLoopNode().execute(
        {"model": model, "dataloader": loader, "optimizer": optimizer, "loss_fn": loss_fn},
        {"epochs": 3, "device": "cpu"},
    )
    assert res["losses"].shape == (3,)
    assert res["metrics"]["total_epochs_run"] == 3


def test_training_decreases_loss_when_overfitting():
    torch.manual_seed(42)
    # Set up something trivially learnable
    X = torch.randn(32, 4)
    W = torch.randn(4, 2)
    y = (X @ W).argmax(dim=-1)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = _make_loader(dataset, batch_size=8)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    res = TrainingLoopNode().execute(
        {"model": model, "dataloader": loader, "optimizer": optimizer, "loss_fn": loss_fn},
        {"epochs": 20, "device": "cpu"},
    )
    losses = res["losses"]
    assert losses[-1] < losses[0]


def test_validation_loop_produces_val_losses():
    model = nn.Linear(4, 2)
    train_loader = _make_loader(_make_dataset())
    val_loader = _make_loader(_make_dataset(n=4))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    res = TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": train_loader,
            "optimizer": optimizer,
            "loss_fn": loss_fn,
            "val_dataloader": val_loader,
        },
        {"epochs": 2, "device": "cpu"},
    )
    assert res["val_losses"].shape == (2,)
    assert res["metrics"]["final_val_loss"] is not None


def test_progress_callback_invoked_per_epoch():
    model = nn.Linear(4, 2)
    loader = _make_loader(_make_dataset())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    seen_events = []
    def cb(payload):
        seen_events.append(payload.get("event"))
    TrainingLoopNode().execute(
        {"model": model, "dataloader": loader, "optimizer": optimizer, "loss_fn": loss_fn},
        {"epochs": 2, "device": "cpu"},
        progress_callback=cb,
    )
    assert "config" in seen_events
    assert seen_events.count("epoch") == 2


def test_grad_clip_does_not_error():
    model = nn.Linear(4, 2)
    loader = _make_loader(_make_dataset())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    res = TrainingLoopNode().execute(
        {"model": model, "dataloader": loader, "optimizer": optimizer, "loss_fn": loss_fn},
        {"epochs": 1, "device": "cpu", "grad_clip_norm": 1.0},
    )
    assert res["losses"].shape == (1,)


# ── max_steps / log_interval / deterministic (core#134) ───────────────────


class _RecordingContext:
    """Collects ``log_metric`` calls; everything else is a no-op."""

    def __init__(self, deterministic=False):
        self.deterministic = deterministic
        self.current_node_id = "train"
        self.metrics = []
        self.seed = None

    def should_stop(self):
        return False

    def log_metric(self, name, value, step, node_id=None):
        self.metrics.append((name, value, step))

    def can_record_artifacts(self):
        return False


def _train(params, *, context=None, n=8, batch_size=2):
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    loader = _make_loader(_make_dataset(n=n), batch_size=batch_size)
    return TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": loader,
            "optimizer": torch.optim.SGD(model.parameters(), lr=0.01),
            "loss_fn": nn.CrossEntropyLoss(),
        },
        {"device": "cpu", **params},
        context=context,
    )


def test_max_steps_stops_the_run_mid_epoch():
    """8 samples at batch 2 is 4 steps/epoch; a budget of 5 spans two epochs."""
    result = _train({"epochs": 10, "max_steps": 5})

    assert result["metrics"]["total_steps"] == 5
    assert result["metrics"]["stopped_at_max_steps"] is True
    assert result["metrics"]["total_epochs_run"] == 2
    assert result["losses"].shape == (2,)


def test_max_steps_zero_means_no_limit():
    result = _train({"epochs": 2, "max_steps": 0})

    assert result["metrics"]["total_steps"] == 8
    assert result["metrics"]["stopped_at_max_steps"] is False
    assert result["losses"].shape == (2,)


def test_max_steps_is_not_reported_as_an_interruption():
    """A spent budget is a normal finish, not a Stop the user can resume."""
    result = _train({"epochs": 10, "max_steps": 3})

    assert result["metrics"]["interrupted"] is False
    assert "__interrupted__" not in result


def test_log_interval_thins_the_batch_series_without_moving_the_steps():
    every = _RecordingContext()
    _train({"epochs": 1, "batch_metrics": True, "log_interval": 1},
           context=every)
    thinned = _RecordingContext()
    _train({"epochs": 1, "batch_metrics": True, "log_interval": 2},
           context=thinned)

    def _steps(ctx):
        return [step for name, _, step in ctx.metrics
                if name == "train_loss_batch"]

    assert _steps(every) == [1, 2, 3, 4]
    # Same x-axis, fewer points -- a chart at interval 2 overlays one at
    # interval 1 rather than being squashed onto its own scale.
    assert _steps(thinned) == [2, 4]


def test_log_interval_does_nothing_without_batch_metrics():
    ctx = _RecordingContext()
    _train({"epochs": 1, "log_interval": 1}, context=ctx)

    assert not [m for m in ctx.metrics if m[0] == "train_loss_batch"]


def test_deterministic_param_turns_torch_determinism_on():
    previously = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        _train({"epochs": 1, "deterministic": True})
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(previously)


def test_deterministic_run_option_applies_without_the_node_param():
    """The run-level switch reaches the node through the context."""
    previously = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        _train({"epochs": 1}, context=_RecordingContext(deterministic=True))
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(previously)


def test_defaults_leave_the_loop_unchanged():
    """No new key set: same epochs, same step count, no early exit."""
    result = _train({"epochs": 3})

    assert result["losses"].shape == (3,)
    assert result["metrics"]["total_steps"] == 12
    assert result["metrics"]["stopped_at_max_steps"] is False


# ── val_accuracy per epoch (#202, 1a) ───────────────────────────────────
#
# Gated exactly like val_loss: present only when a val_dataloader is
# wired. The classification-only gate (1b) is proven separately, below.


def _classification_scenario(n_train=8, n_val=4, in_features=4, out_classes=2):
    torch.manual_seed(0)
    model = nn.Linear(in_features, out_classes)
    train_loader = _make_loader(
        _make_dataset(n=n_train, in_features=in_features, out_classes=out_classes),
        batch_size=2,
    )
    val_loader = _make_loader(
        _make_dataset(n=n_val, in_features=in_features, out_classes=out_classes),
        batch_size=n_val,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    return model, train_loader, val_loader, optimizer


def test_classification_run_with_val_dataloader_records_val_accuracy_per_epoch():
    """Acceptance: a classifier run with a val_dataloader plots
    val_accuracy against epoch -- one point per epoch, in [0, 1]."""
    model, train_loader, val_loader, optimizer = _classification_scenario()
    ctx = _RecordingContext()
    TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": train_loader,
            "optimizer": optimizer,
            "loss_fn": nn.CrossEntropyLoss(),
            "val_dataloader": val_loader,
        },
        {"epochs": 3, "device": "cpu"},
        context=ctx,
    )
    acc_points = [(step, value) for name, value, step in ctx.metrics
                  if name == "val_accuracy"]
    assert [step for step, _ in acc_points] == [1, 2, 3]
    assert all(0.0 <= value <= 1.0 for _, value in acc_points)


def test_no_val_dataloader_records_no_val_accuracy_and_does_not_crash():
    """Acceptance: a run with no val_dataloader emits no accuracy series
    and does not crash."""
    model = nn.Linear(4, 2)
    ctx = _RecordingContext()
    result = TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": _make_loader(_make_dataset()),
            "optimizer": torch.optim.SGD(model.parameters(), lr=0.01),
            "loss_fn": nn.CrossEntropyLoss(),
        },
        {"epochs": 2, "device": "cpu"},
        context=ctx,
    )
    assert result["metrics"]["total_epochs_run"] == 2
    assert not [m for m in ctx.metrics if m[0] == "val_accuracy"]
