"""Tests for TrainingLoopNode (end-to-end mini training)."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from app.nodes.training.training_loop_node import TrainingLoopNode


def _make_dataset(n=8, in_features=4, out_classes=2):
    """Tiny supervised dataset (n samples)."""
    X = torch.randn(n, in_features)
    y = torch.randint(0, out_classes, (n,))
    return torch.utils.data.TensorDataset(X, y)


def _make_regression_dataset(n=8, in_features=4, out_features=1):
    """Tiny regression dataset: float targets shaped like the model's
    output, so MSELoss compares like-shaped tensors rather than silently
    broadcasting a [B] target against a [B, out_features] prediction."""
    X = torch.randn(n, in_features)
    y = torch.randn(n, out_features)
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


# ── classification-only gate (#202, 1b) ─────────────────────────────────
#
# Getting this wrong in the permissive direction (emitting an "accuracy"
# for a regression run) is worse than not shipping the feature: it is a
# meaningless number presented with authority.


def test_regression_loss_with_val_dataloader_records_no_val_accuracy():
    """Acceptance: a regression run emits no accuracy series at all --
    even though its val_dataloader means the validation loop (and
    val_loss) still runs normally."""
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    train_loader = _make_loader(_make_regression_dataset(n=8))
    val_loader = _make_loader(_make_regression_dataset(n=4), batch_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    ctx = _RecordingContext()
    TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": train_loader,
            "optimizer": optimizer,
            "loss_fn": nn.MSELoss(),
            "val_dataloader": val_loader,
        },
        {"epochs": 2, "device": "cpu"},
        context=ctx,
    )
    names = {name for name, _, _ in ctx.metrics}
    assert "val_loss" in names
    assert "val_accuracy" not in names


def test_bce_loss_with_val_dataloader_records_no_val_accuracy():
    """BCEWithLogitsLoss is binary/multi-label, not argmax-shaped over
    dim=1 the way CrossEntropyLoss/NLLLoss are -- also excluded."""
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    train_loader = _make_loader(_make_regression_dataset(n=8, out_features=3))
    val_loader = _make_loader(_make_regression_dataset(n=4, out_features=3), batch_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    ctx = _RecordingContext()
    TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": train_loader,
            "optimizer": optimizer,
            "loss_fn": nn.BCEWithLogitsLoss(),
            "val_dataloader": val_loader,
        },
        {"epochs": 1, "device": "cpu"},
        context=ctx,
    )
    names = {name for name, _, _ in ctx.metrics}
    assert "val_accuracy" not in names


def test_nll_loss_with_val_dataloader_records_val_accuracy():
    """NLLLoss is the other classification loss the gate must admit --
    log-probabilities against integer class indices, same argmax(dim=1)
    shape as CrossEntropyLoss."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 2), nn.LogSoftmax(dim=1))
    train_loader = _make_loader(_make_dataset(n=8))
    val_loader = _make_loader(_make_dataset(n=4), batch_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    ctx = _RecordingContext()
    TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": train_loader,
            "optimizer": optimizer,
            "loss_fn": nn.NLLLoss(),
            "val_dataloader": val_loader,
        },
        {"epochs": 1, "device": "cpu"},
        context=ctx,
    )
    names = {name for name, _, _ in ctx.metrics}
    assert "val_accuracy" in names


# ── monitor: early stopping can watch val_accuracy (#202, 1c) ──────────


def test_monitor_param_defaults_to_val_loss_with_both_options():
    """default val_loss to preserve current behaviour."""
    p = next(p for p in TrainingLoopNode.define_params() if p.name == "monitor")
    assert p.default == "val_loss"
    assert p.options == ["val_loss", "val_accuracy"]


def test_monitor_omitted_behaves_exactly_like_val_loss():
    """Regression guard for "default val_loss to preserve current
    behaviour": leaving the new param unset must produce bit-identical
    results to spelling it out."""
    def _run(with_monitor_key):
        torch.manual_seed(0)
        model = nn.Linear(4, 2)
        loader = _make_loader(_make_dataset(n=8), batch_size=2)
        val_loader = _make_loader(_make_dataset(n=4), batch_size=4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        params = {"epochs": 6, "device": "cpu", "early_stopping_patience": 2}
        if with_monitor_key:
            params["monitor"] = "val_loss"
        return TrainingLoopNode().execute(
            {
                "model": model, "dataloader": loader, "optimizer": optimizer,
                "loss_fn": nn.CrossEntropyLoss(), "val_dataloader": val_loader,
            },
            params,
        )

    omitted = _run(False)
    explicit = _run(True)
    assert omitted["metrics"]["total_epochs_run"] == explicit["metrics"]["total_epochs_run"]
    assert omitted["metrics"]["best_epoch"] == explicit["metrics"]["best_epoch"]
    assert torch.equal(omitted["losses"], explicit["losses"])


class _ScriptedEvalModel(nn.Module):
    """A model whose EVAL-mode forward pass returns pre-scripted logits --
    one script entry consumed per validation call -- so a test can pin an
    exact, independently-controlled accuracy trajectory across epochs.
    TRAIN-mode forward passes are a real, differentiable Linear op (so the
    training phase runs and backward() has something to do) and never
    touch the script. Keyed off ``self.training``, which model.train() /
    model.eval() genuinely toggle for a model (unlike for a loss module --
    see _ScriptedValLoss below)."""

    def __init__(self, eval_logits: list[torch.Tensor], in_features=4, out_classes=2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_classes)
        self._eval_logits = eval_logits
        self._eval_index = 0

    def forward(self, x):
        if self.training:
            return self.linear(x)
        logits = self._eval_logits[self._eval_index]
        self._eval_index += 1
        return logits


class _ScriptedValLoss(nn.CrossEntropyLoss):
    """A CrossEntropyLoss whose VALUE is scripted per validation call,
    decoupled from the actual logits/targets, so a test can pin an exact
    loss trajectory independent of the accuracy trajectory (itself driven
    by _ScriptedEvalModel's real, independently-chosen logits).
    isinstance(this, nn.CrossEntropyLoss) is True, so the 1b classification
    gate still admits it.

    ``loss_fn.train()``/``.eval()`` are never called by TrainingLoop --
    only ``model.train()``/``.eval()`` are -- so ``self.training`` cannot
    be used to tell a train call from a val call here. Instead this counts
    calls: with exactly one train batch and one val batch per epoch (the
    scenario below), calls alternate train, val, train, val, ... starting
    with train, so val calls are the odd-indexed ones.
    """

    def __init__(self, eval_losses: list[float]):
        super().__init__()
        self._eval_losses = eval_losses
        self._call = 0
        self._eval_index = 0

    def forward(self, outputs, targets):
        real = super().forward(outputs, targets)
        is_val_call = self._call % 2 == 1
        self._call += 1
        if is_val_call:
            value = self._eval_losses[self._eval_index]
            self._eval_index += 1
            return real * 0 + value
        return real


def test_monitor_val_accuracy_stops_at_a_different_epoch_than_val_loss():
    """Acceptance: monitor=val_accuracy stops on the accuracy plateau
    rather than the loss turn.

    Loss and accuracy are scripted independently (see the two helper
    classes above) so the two curves deliberately DISAGREE about which
    epoch is best: loss improves through epoch 2 then turns worse at
    epoch 3; accuracy ties at epoch 2 (no improvement under a strict
    threshold) then improves at epoch 3. With early_stopping_patience=1,
    each monitor stops at ITS OWN plateau, not the other's -- proving
    monitor genuinely changes which epoch ends the run, not just which
    number gets logged.
    """
    val_targets = torch.tensor([0, 1, 0, 1])
    val_X = torch.randn(4, 4)
    # 2/4 correct (rows 0, 2); tied between epoch 1 and 2.
    acc_50 = torch.tensor([[1., -1.], [1., -1.], [1., -1.], [1., -1.]])
    # 4/4 correct; only at epoch 3.
    acc_100 = torch.tensor([[1., -1.], [-1., 1.], [1., -1.], [-1., 1.]])

    def _run(monitor):
        torch.manual_seed(0)
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(val_X, val_targets), batch_size=4)
        model = _ScriptedEvalModel([acc_50, acc_50, acc_100])
        loss_fn = _ScriptedValLoss([2.0, 1.0, 1.5])  # improves, improves, turns
        train_loader = _make_loader(_make_dataset(n=2), batch_size=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        return TrainingLoopNode().execute(
            {
                "model": model, "dataloader": train_loader, "optimizer": optimizer,
                "loss_fn": loss_fn, "val_dataloader": val_loader,
            },
            {"epochs": 5, "device": "cpu", "early_stopping_patience": 1,
             "monitor": monitor},
        )

    by_loss = _run("val_loss")
    by_accuracy = _run("val_accuracy")

    # Loss: ep1=2.0 (best, first), ep2=1.0 (improves, new best),
    # ep3=1.5 (worse -> patience_counter=1 >= patience=1 -> stop after ep3).
    assert by_loss["metrics"]["total_epochs_run"] == 3
    assert by_loss["metrics"]["best_epoch"] == 2

    # Accuracy: ep1=0.5 (best, first), ep2=0.5 (TIE, not a strict
    # improvement -> patience_counter=1 >= patience=1 -> stop after ep2).
    assert by_accuracy["metrics"]["total_epochs_run"] == 2
    assert by_accuracy["metrics"]["best_epoch"] == 1


def test_monitor_val_accuracy_falls_back_to_val_loss_with_no_val_dataloader(caplog):
    """Degenerate case 1: no val_dataloader means there is no val_accuracy
    series (1a's gate) to monitor. Falls back to val_loss -- which, with
    no val_dataloader either, is itself the pre-existing avg_train_loss
    fallback -- so the run completes normally rather than crashing or
    silently comparing against a value that was never computed."""
    with caplog.at_level(logging.WARNING, logger="app.nodes.training.training_loop_node"):
        result = _train({
            "epochs": 3, "early_stopping_patience": 5, "monitor": "val_accuracy",
        })

    assert result["metrics"]["total_epochs_run"] == 3
    assert result["metrics"]["monitor"] == "val_loss"
    assert result["metrics"]["monitor_requested"] == "val_accuracy"
    messages = [r.getMessage() for r in caplog.records]
    assert any("val_accuracy" in m and "val_dataloader" in m for m in messages), messages


def test_monitor_val_accuracy_falls_back_to_val_loss_for_a_regression_loss(caplog):
    """Degenerate case 2: a regression loss never produces val_accuracy
    (1b's gate), so monitor=val_accuracy falls back to val_loss here too,
    even though a val_dataloader IS wired."""
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    train_loader = _make_loader(_make_regression_dataset(n=8))
    val_loader = _make_loader(_make_regression_dataset(n=4), batch_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with caplog.at_level(logging.WARNING, logger="app.nodes.training.training_loop_node"):
        res = TrainingLoopNode().execute(
            {
                "model": model, "dataloader": train_loader, "optimizer": optimizer,
                "loss_fn": nn.MSELoss(), "val_dataloader": val_loader,
            },
            {"epochs": 3, "device": "cpu", "early_stopping_patience": 5,
             "monitor": "val_accuracy"},
        )

    assert res["metrics"]["total_epochs_run"] == 3
    assert res["metrics"]["monitor"] == "val_loss"
    messages = [r.getMessage() for r in caplog.records]
    assert any("classification" in m.lower() for m in messages), messages


def test_monitor_val_loss_reports_no_requested_key_when_not_falling_back():
    """metrics["monitor_requested"] only appears when a fallback actually
    happened -- an ordinary val_loss run has nothing to report a
    discrepancy about."""
    result = _train({"epochs": 2, "early_stopping_patience": 1})
    assert result["metrics"]["monitor"] == "val_loss"
    assert "monitor_requested" not in result["metrics"]


def test_monitor_key_absent_from_metrics_when_early_stopping_is_off():
    """monitor is meaningless when patience=0 (early stopping disabled);
    metrics should not claim a value was monitored when none was."""
    result = _train({"epochs": 2, "monitor": "val_accuracy"})
    assert "monitor" not in result["metrics"]
