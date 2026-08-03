"""Tests for LossNode."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from app.nodes.training import loss_node
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


@pytest.mark.parametrize("param,value", [
    ("label_smoothing", 0.1),
    ("weight", "1, 5"),
    ("ignore_index", 0),
    ("pos_weight", "3"),
])
def test_a_param_the_type_hides_is_ignored_rather_than_fatal(param, value):
    """Tune it on CrossEntropy, switch to MSE, and the run must still work.

    ``visible_when`` hides the editor but never clears the value, so an
    error here names a field the user cannot see. Regression for the #188
    review's I5 — all four raised before.
    """
    built = _loss({"type": "MSELoss", param: value})
    assert isinstance(built, nn.MSELoss)
    assert built.reduction == "mean"


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


def test_a_hidden_leftover_comes_back_when_it_applies_again():
    tuned = {"type": "CrossEntropyLoss", "label_smoothing": 0.1}
    _loss({**tuned, "type": "MSELoss"})              # ignored, no raise
    assert _loss(tuned).label_smoothing == 0.1      # honoured again


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





def test_pos_weight_rebalances_binary_logits():
    logits = torch.tensor([[0.5], [-0.5]])
    targets = torch.tensor([[1.0], [0.0]])

    node_loss = _loss({"type": "BCEWithLogitsLoss", "pos_weight": "3"})
    reference = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))

    assert torch.isclose(node_loss(logits, targets), reference(logits, targets))
    assert node_loss(logits, targets).item() > \
        _loss({"type": "BCEWithLogitsLoss"})(logits, targets).item()





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


# ── the invariant that replaces the dead rejection (#188 re-review, D5) ───


def test_every_conditional_param_hides_exactly_where_it_does_not_apply():
    """Visibility and applicability are the SAME set, per param.

    That is what makes "a leftover on a hidden field means not set" a
    complete answer: an inapplicable value cannot be a visible one, so there
    is nothing left to reject. The node used to carry a guard that raised
    for a visible inapplicable param, and the re-review measured it at 0
    raising combinations out of 55 — dead code reading as a live guard, so
    it is gone and this stands in its place.

    Add a param that is visible where it does not apply and this fails,
    which is the moment to decide what should happen — as
    ``Optimizer.weight_decay``, the one param of that shape in either node,
    decided by raising.
    """
    definitions = {p.name: p for p in LossNode.define_params()}

    for param, applies_to in {
        "label_smoothing": loss_node._LABEL_SMOOTHING_TYPES,
        "weight": loss_node._WEIGHT_TYPES,
        "ignore_index": loss_node._IGNORE_INDEX_TYPES,
        "pos_weight": loss_node._POS_WEIGHT_TYPES,
    }.items():
        assert definitions[param].visible_when == {"type": sorted(applies_to)}, (
            f"{param} is shown for types that cannot take it")

    always_visible = {name for name, d in definitions.items()
                      if not d.visible_when}
    assert always_visible == {"type", "reduction"}, (
        "a new always-visible param needs an explicit decision: every loss "
        "must accept it, or execute() must reject it where it does not")
