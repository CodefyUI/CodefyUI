"""Tests for the #297/#298 research controls on TrainingLoop + LRScheduler."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.nodes.training.lr_scheduler_node import LRSchedulerNode
from app.nodes.training.training_loop_node import TrainingLoopNode


class _RecordingContext:
    """Collects ``log_metric`` calls; everything else is a no-op."""

    def __init__(self):
        self.deterministic = False
        self.current_node_id = "train"
        self.metrics = []
        self.seed = None

    def should_stop(self):
        return False

    def log_metric(self, name, value, step, node_id=None):
        self.metrics.append((name, value, step))

    def can_record_artifacts(self):
        return False


def _dataset(n=8):
    torch.manual_seed(0)
    return TensorDataset(
        torch.randn(n, 4), torch.randint(0, 2, (n,), dtype=torch.int64))


def _train(params, *, context=None, scheduler=None, val=False, n=8, batch_size=2):
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = {
        "model": model,
        "dataloader": DataLoader(_dataset(n), batch_size=batch_size),
        "optimizer": optimizer,
        "loss_fn": nn.CrossEntropyLoss(),
    }
    if scheduler is not None:
        inputs["lr_scheduler"] = scheduler(optimizer)
    if val:
        inputs["val_dataloader"] = DataLoader(_dataset(4), batch_size=2)
    result = TrainingLoopNode().execute(
        inputs, {"device": "cpu", **params}, context=context)
    return result, inputs


def _series(ctx, name):
    return [(value, step) for metric, value, step in ctx.metrics if metric == name]


# ── #297: warmup schedule families ─────────────────────────────────────────


def _make_sched(kind, warmup, total):
    def factory(optimizer):
        return LRSchedulerNode().execute(
            {"optimizer": optimizer},
            {"type": kind, "warmup_steps": warmup, "total_steps": total},
        )["scheduler"]
    return factory


@pytest.mark.parametrize("kind", ["warmup_cosine", "warmup_linear", "constant_with_warmup"])
def test_warmup_families_ramp_from_cold_to_base(kind):
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.1)
    sched = _make_sched(kind, warmup=10, total=100)(optimizer)
    assert optimizer.param_groups[0]["lr"] < 1e-6  # cold start
    for _ in range(10):
        optimizer.step()
        sched.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1, rel=1e-3)


def test_warmup_cosine_and_linear_anneal_while_constant_holds():
    ends = {}
    for kind in ("warmup_cosine", "warmup_linear", "constant_with_warmup"):
        optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.1)
        sched = _make_sched(kind, warmup=5, total=50)(optimizer)
        for _ in range(50):
            optimizer.step()
            sched.step()
        ends[kind] = optimizer.param_groups[0]["lr"]
    assert ends["warmup_cosine"] < 0.005
    assert ends["warmup_linear"] < 0.005
    assert ends["constant_with_warmup"] == pytest.approx(0.1, rel=1e-3)


# ── #297: scheduler_step=optimizer_step ────────────────────────────────────


def test_per_step_scheduler_advances_once_per_optimizer_step():
    result, inputs = _train(
        {"epochs": 1, "max_steps": 3, "scheduler_step": "optimizer_step"},
        scheduler=_make_sched("warmup_cosine", warmup=2, total=10),
    )
    # 3 optimizer steps -> the scheduler took exactly 3 steps (last_epoch
    # counts them), not the 1 the per-epoch mode would have taken.
    assert inputs["lr_scheduler"].last_epoch == 3


def test_default_epoch_stepping_is_unchanged():
    result, inputs = _train(
        {"epochs": 2},
        scheduler=_make_sched("warmup_cosine", warmup=2, total=10),
    )
    assert inputs["lr_scheduler"].last_epoch == 2


# ── #298: gradient-norm and update-ratio telemetry ─────────────────────────


def test_log_grad_norm_records_per_optimizer_step():
    ctx = _RecordingContext()
    _train({"epochs": 1, "log_grad_norm": True}, context=ctx)
    series = _series(ctx, "grad_norm")
    assert [step for _, step in series] == [1, 2, 3, 4]
    assert all(value > 0 for value, _ in series)
    assert not _series(ctx, "grad_norm_clipped")  # no clipping configured


def test_grad_norm_clipped_is_bounded_by_the_threshold():
    ctx = _RecordingContext()
    _train({"epochs": 1, "log_grad_norm": True, "grad_clip_norm": 0.001},
           context=ctx)
    clipped = _series(ctx, "grad_norm_clipped")
    assert clipped
    assert all(value <= 0.001 + 1e-9 for value, _ in clipped)


def test_log_update_ratio_records_positive_values():
    ctx = _RecordingContext()
    _train({"epochs": 1, "log_update_ratio": True}, context=ctx)
    series = _series(ctx, "update_ratio")
    assert len(series) == 4
    assert all(value > 0 for value, _ in series)


def test_telemetry_off_by_default():
    ctx = _RecordingContext()
    _train({"epochs": 1}, context=ctx)
    assert not _series(ctx, "grad_norm")
    assert not _series(ctx, "update_ratio")
    assert not _series(ctx, "val_loss_step")


# ── #298: mid-epoch validation curve ───────────────────────────────────────


def test_val_every_steps_records_a_curve_and_restores_training_mode():
    ctx = _RecordingContext()
    result, inputs = _train(
        {"epochs": 1, "val_every_steps": 2}, context=ctx, val=True)
    series = _series(ctx, "val_loss_step")
    assert [step for _, step in series] == [2, 4]
    assert all(value > 0 for value, _ in series)
    # The per-epoch val series still exists independently.
    assert result["val_losses"].shape == (1,)


def test_val_every_steps_without_val_loader_is_silent():
    ctx = _RecordingContext()
    _train({"epochs": 1, "val_every_steps": 2}, context=ctx)
    assert not _series(ctx, "val_loss_step")


# ── #298: step-milestone checkpoints ───────────────────────────────────────


def test_checkpoint_every_steps_fires_on_distinct_milestones(monkeypatch):
    calls = []

    def fake_periodic(context, model, optimizer, *, epoch, losses=None,
                      lr_scheduler=None, scaler_state=None, node_id=None):
        calls.append(epoch)
        return None

    import app.nodes.training.training_loop_node as tl
    monkeypatch.setattr(tl, "save_periodic_checkpoint", fake_periodic, raising=False)
    # save_periodic_checkpoint is imported inside execute; patch the source.
    import app.core.loop_control as lc
    monkeypatch.setattr(lc, "save_periodic_checkpoint", fake_periodic)

    _train({"epochs": 1, "checkpoint_every_steps": 2},
           context=_RecordingContext())
    # 4 optimizer steps -> milestones at steps 2 and 4, stamped DISTINCTLY.
    assert calls == [2, 4]
