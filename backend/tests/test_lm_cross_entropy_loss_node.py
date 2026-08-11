"""Tests for LMCrossEntropyLossNode."""

from __future__ import annotations

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.nodes.llm.lm_cross_entropy_loss_node import LMCrossEntropyLossNode


def make(params=None):
    return LMCrossEntropyLossNode().execute({}, params or {})["loss_fn"]


def test_node_metadata():
    assert LMCrossEntropyLossNode.NODE_NAME == "LMCrossEntropyLoss"
    assert LMCrossEntropyLossNode.CATEGORY == "LLM"
    assert LMCrossEntropyLossNode.define_inputs() == []


def test_matches_manually_reshaped_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 11)
    targets = torch.randint(0, 11, (2, 5))
    loss = make()(logits, targets)
    reference = F.cross_entropy(logits.reshape(-1, 11), targets.reshape(-1))
    assert torch.isclose(loss, reference)


def test_accepts_already_flat_shapes():
    torch.manual_seed(0)
    logits = torch.randn(10, 7)
    targets = torch.randint(0, 7, (10,))
    assert torch.isclose(make()(logits, targets), F.cross_entropy(logits, targets))


def test_ignore_index_masks_positions():
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 5)
    targets = torch.tensor([[1, 2, -100, -100]])
    loss = make()(logits, targets)
    reference = F.cross_entropy(logits[0, :2], targets[0, :2])
    assert torch.isclose(loss, reference)


def test_label_smoothing_changes_the_loss():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 9)
    targets = torch.randint(0, 9, (2, 3))
    plain = make()(logits, targets)
    smoothed = make({"label_smoothing": 0.1})(logits, targets)
    assert not torch.isclose(plain, smoothed)


def test_not_an_nn_cross_entropy_instance():
    # TrainingLoop opens its val-accuracy argmax path for
    # isinstance(nn.CrossEntropyLoss); (B,T,V) logits would argmax the
    # time axis there, so this loss must NOT be a subclass.
    fn = make()
    assert not isinstance(fn, nn.CrossEntropyLoss)
    assert isinstance(fn, nn.Module)


def test_gradient_flows():
    logits = torch.randn(2, 4, 6, requires_grad=True)
    targets = torch.randint(0, 6, (2, 4))
    make()(logits, targets).backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0


def test_full_model_pickle_roundtrip():
    fn = pickle.loads(pickle.dumps(make({"ignore_index": -1})))
    logits = torch.randn(1, 3, 4)
    targets = torch.tensor([[0, 3, -1]])
    reference = F.cross_entropy(
        logits.reshape(-1, 4), targets.reshape(-1), ignore_index=-1,
    )
    assert torch.isclose(fn(logits, targets), reference)
