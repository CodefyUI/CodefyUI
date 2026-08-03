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


# ── full parameterization (core#134) ──────────────────────────────────────
#
# Every test compares against a raw ``torch.nn`` reference built with the
# same kwarg. That is what catches the failure mode the issue names: a
# schema that accepts a value the node then throws away.


def _loss(params: dict):
    return LossNode().execute({}, params)["loss_fn"]


def test_label_smoothing_measurably_changes_cross_entropy():
    """The headline acceptance criterion for parameter application.

    A confident, CORRECT prediction costs more under label smoothing,
    because part of the target mass now sits on the other classes. Compared
    to torch's own smoothed loss so this cannot pass by accident.
    """
    logits = torch.tensor([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    targets = torch.tensor([0, 1])

    plain = _loss({"type": "CrossEntropyLoss"})(logits, targets)
    smoothed = _loss({"type": "CrossEntropyLoss",
                      "label_smoothing": 0.1})(logits, targets)
    reference = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, targets)

    assert torch.isclose(smoothed, reference)
    assert smoothed.item() > plain.item()


def test_label_smoothing_is_rejected_on_a_loss_that_has_none():
    with pytest.raises(ValueError, match="does not accept label_smoothing"):
        _loss({"type": "MSELoss", "label_smoothing": 0.1})


@pytest.mark.parametrize("loss_type", [
    "CrossEntropyLoss", "MSELoss", "BCEWithLogitsLoss", "L1Loss",
    "SmoothL1Loss", "NLLLoss", "KLDivLoss", "HuberLoss", "BCELoss",
    "MarginRankingLoss", "CosineEmbeddingLoss",
])
def test_reduction_reaches_every_loss(loss_type):
    assert _loss({"type": loss_type, "reduction": "sum"}).reduction == "sum"


def test_reduction_none_keeps_the_loss_per_sample():
    """Not just stored on the module -- it changes the output SHAPE."""
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    targets = torch.tensor([0, 1, 0])

    per_sample = _loss({"type": "CrossEntropyLoss",
                        "reduction": "none"})(logits, targets)
    summed = _loss({"type": "CrossEntropyLoss",
                    "reduction": "sum"})(logits, targets)

    assert per_sample.shape == (3,)
    assert torch.isclose(per_sample.sum(), summed)
    assert torch.isclose(
        summed, nn.CrossEntropyLoss(reduction="sum")(logits, targets))


def test_unknown_reduction_is_rejected():
    with pytest.raises(ValueError, match="Unsupported reduction"):
        _loss({"type": "MSELoss", "reduction": "average"})


def test_class_weight_is_parsed_and_changes_the_loss():
    # ASYMMETRIC on purpose: with equal per-sample losses the weighted mean
    # normalises by the weight sum and lands back on the unweighted one, so
    # a symmetric fixture would "prove" weighting works while it did nothing.
    logits = torch.tensor([[2.0, 0.0], [0.0, 0.5]])
    targets = torch.tensor([0, 1])

    weighted = _loss({"type": "CrossEntropyLoss", "weight": "1, 5"})
    reference = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]))

    assert torch.equal(weighted.weight, torch.tensor([1.0, 5.0]))
    assert torch.isclose(weighted(logits, targets), reference(logits, targets))
    assert not torch.isclose(
        weighted(logits, targets),
        _loss({"type": "CrossEntropyLoss"})(logits, targets))


def test_blank_weight_means_unweighted():
    assert _loss({"type": "CrossEntropyLoss", "weight": ""}).weight is None
    assert _loss({"type": "CrossEntropyLoss", "weight": "   "}).weight is None


def test_weight_is_rejected_on_a_loss_that_has_none():
    with pytest.raises(ValueError, match="does not accept weight"):
        _loss({"type": "MSELoss", "weight": "1, 5"})


def test_ignore_index_drops_the_marked_target():
    """A sample labelled with ignore_index must contribute nothing."""
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [5.0, -5.0]])
    with_padding = torch.tensor([0, 1, 7])
    without_padding = torch.tensor([0, 1])

    node_loss = _loss({"type": "CrossEntropyLoss", "ignore_index": 7})
    assert torch.isclose(
        node_loss(logits, with_padding),
        _loss({"type": "CrossEntropyLoss"})(logits[:2], without_padding))
    assert torch.isclose(
        node_loss(logits, with_padding),
        nn.CrossEntropyLoss(ignore_index=7)(logits, with_padding))


def test_ignore_index_is_rejected_on_a_loss_that_has_none():
    with pytest.raises(ValueError, match="does not accept ignore_index"):
        _loss({"type": "MSELoss", "ignore_index": 0})


def test_pos_weight_rebalances_binary_logits():
    logits = torch.tensor([[0.5], [-0.5]])
    targets = torch.tensor([[1.0], [0.0]])

    node_loss = _loss({"type": "BCEWithLogitsLoss", "pos_weight": "3"})
    reference = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))

    assert torch.isclose(node_loss(logits, targets), reference(logits, targets))
    assert node_loss(logits, targets).item() > \
        _loss({"type": "BCEWithLogitsLoss"})(logits, targets).item()


def test_pos_weight_is_rejected_on_a_loss_that_has_none():
    with pytest.raises(ValueError, match="does not accept pos_weight"):
        _loss({"type": "MSELoss", "pos_weight": "3"})


@pytest.mark.parametrize("loss_type", [
    "CrossEntropyLoss", "MSELoss", "BCEWithLogitsLoss", "L1Loss",
    "SmoothL1Loss", "NLLLoss", "KLDivLoss", "HuberLoss", "BCELoss",
    "MarginRankingLoss", "CosineEmbeddingLoss",
])
def test_defaults_reproduce_the_pre_change_loss_exactly(loss_type):
    """An existing graph carries only ``type`` and must be unaffected."""
    built = _loss({"type": loss_type})
    reference = getattr(nn, loss_type)()

    assert built.reduction == reference.reduction
    for attribute in ("weight", "pos_weight", "ignore_index",
                      "label_smoothing"):
        if not hasattr(reference, attribute):
            continue
        expected = getattr(reference, attribute)
        actual = getattr(built, attribute)
        if isinstance(expected, torch.Tensor) or isinstance(actual, torch.Tensor):
            assert torch.equal(actual, expected), f"{loss_type}.{attribute}"
        else:
            assert actual == expected, f"{loss_type}.{attribute}"
