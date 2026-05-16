"""Tests for OptimizerNode (creating optimizers from a model)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from app.nodes.training.optimizer_node import OptimizerNode


def _model():
    return nn.Linear(4, 2)


def test_node_metadata():
    assert OptimizerNode.NODE_NAME == "Optimizer"
    assert OptimizerNode.CATEGORY == "Training"


def test_default_creates_adam():
    res = OptimizerNode().execute({"model": _model()}, {"type": "Adam", "lr": 0.001, "weight_decay": 0.0})
    assert res["optimizer"].__class__.__name__ == "Adam"


def test_sgd_optimizer():
    res = OptimizerNode().execute({"model": _model()}, {"type": "SGD", "lr": 0.01})
    assert res["optimizer"].__class__.__name__ == "SGD"
    assert res["optimizer"].param_groups[0]["lr"] == 0.01


def test_adamw_includes_weight_decay():
    res = OptimizerNode().execute({"model": _model()}, {"type": "AdamW", "lr": 0.001, "weight_decay": 0.05})
    assert res["optimizer"].param_groups[0]["weight_decay"] == 0.05


def test_rprop_drops_default_weight_decay_zero():
    # Rprop doesn't accept weight_decay, but zero should silently drop
    res = OptimizerNode().execute({"model": _model()}, {"type": "Rprop", "lr": 0.01, "weight_decay": 0.0})
    assert res["optimizer"].__class__.__name__ == "Rprop"


def test_rprop_rejects_nonzero_weight_decay():
    with pytest.raises(ValueError, match="weight_decay"):
        OptimizerNode().execute(
            {"model": _model()},
            {"type": "Rprop", "lr": 0.01, "weight_decay": 0.01},
        )


def test_unsupported_optimizer_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        OptimizerNode().execute({"model": _model()}, {"type": "Bogus", "lr": 0.001})


def test_lr_is_set_correctly():
    res = OptimizerNode().execute({"model": _model()}, {"type": "Adam", "lr": 0.5})
    assert res["optimizer"].param_groups[0]["lr"] == 0.5


@pytest.mark.parametrize("opt_type", ["Adam", "SGD", "AdamW", "RMSprop", "Adagrad", "RAdam", "NAdam", "ASGD"])
def test_all_supported_optimizers_create(opt_type):
    """Each listed optimizer should construct without error."""
    res = OptimizerNode().execute({"model": _model()}, {"type": opt_type, "lr": 0.001, "weight_decay": 0.0})
    assert res["optimizer"] is not None
