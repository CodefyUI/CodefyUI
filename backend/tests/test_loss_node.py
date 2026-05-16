"""Tests for LossNode."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from app.nodes.training.loss_node import LossNode


def test_node_metadata():
    assert LossNode.NODE_NAME == "Loss"
    assert LossNode.CATEGORY == "Training"
    assert LossNode.define_inputs() == []


def test_default_is_cross_entropy():
    res = LossNode().execute({}, {})
    assert isinstance(res["loss_fn"], nn.CrossEntropyLoss)


def test_mse_loss():
    res = LossNode().execute({}, {"type": "MSELoss"})
    fn = res["loss_fn"]
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.0, 4.0])
    loss = fn(a, b)
    assert torch.isclose(loss, torch.tensor(1.0 / 3))


def test_l1_loss():
    res = LossNode().execute({}, {"type": "L1Loss"})
    fn = res["loss_fn"]
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([2.0, 4.0, 6.0])
    loss = fn(a, b)
    # |1-2| + |2-4| + |3-6| = 1 + 2 + 3 = 6 / 3 = 2.0
    assert torch.isclose(loss, torch.tensor(2.0))


def test_unsupported_loss_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        LossNode().execute({}, {"type": "Bogus"})


@pytest.mark.parametrize(
    "loss_type",
    ["CrossEntropyLoss", "MSELoss", "BCEWithLogitsLoss", "L1Loss", "SmoothL1Loss",
     "NLLLoss", "KLDivLoss", "HuberLoss", "BCELoss", "MarginRankingLoss", "CosineEmbeddingLoss"],
)
def test_all_supported_losses_create(loss_type):
    res = LossNode().execute({}, {"type": loss_type})
    assert res["loss_fn"] is not None
