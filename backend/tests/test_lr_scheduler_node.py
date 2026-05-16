"""Tests for LRSchedulerNode."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from app.nodes.training.lr_scheduler_node import LRSchedulerNode


def _opt():
    model = nn.Linear(4, 2)
    return torch.optim.Adam(model.parameters(), lr=0.1)


def test_node_metadata():
    assert LRSchedulerNode.NODE_NAME == "LRScheduler"


def test_step_lr_decays_lr():
    opt = _opt()
    res = LRSchedulerNode().execute(
        {"optimizer": opt},
        {"type": "StepLR", "step_size": 1, "gamma": 0.5},
    )
    sched = res["scheduler"]
    # Take an optimizer step (required before scheduler step in newer torch)
    opt.step()
    sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.05)


def test_cosine_annealing_lr():
    opt = _opt()
    res = LRSchedulerNode().execute({"optimizer": opt}, {"type": "CosineAnnealingLR", "T_max": 10})
    assert res["scheduler"].__class__.__name__ == "CosineAnnealingLR"


def test_exponential_lr():
    opt = _opt()
    res = LRSchedulerNode().execute({"optimizer": opt}, {"type": "ExponentialLR", "gamma": 0.9})
    sched = res["scheduler"]
    opt.step()
    sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.09)


def test_reduce_lr_on_plateau():
    opt = _opt()
    res = LRSchedulerNode().execute({"optimizer": opt}, {"type": "ReduceLROnPlateau", "gamma": 0.5})
    assert res["scheduler"].__class__.__name__ == "ReduceLROnPlateau"


def test_one_cycle_lr():
    opt = _opt()
    res = LRSchedulerNode().execute(
        {"optimizer": opt},
        {"type": "OneCycleLR", "max_lr": 0.1, "total_steps": 100},
    )
    assert res["scheduler"].__class__.__name__ == "OneCycleLR"


def test_unsupported_scheduler_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        LRSchedulerNode().execute({"optimizer": _opt()}, {"type": "Bogus"})
