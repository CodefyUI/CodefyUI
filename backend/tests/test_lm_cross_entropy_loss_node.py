"""Tests for LMCrossEntropyLossNode (#289).

The node is three lines of construction, so almost everything here is about
the loss module it hands out: that it equals hand-reshaped
``F.cross_entropy``, that its two params reach torch, that it says something
useful when the shapes are wrong -- and that it is NOT a subclass of
``nn.CrossEntropyLoss``, which is the trap the whole design avoids.
"""

from __future__ import annotations

import io

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.node_base import DataType
from app.nodes.llm.lm_cross_entropy_loss_node import (
    DEFAULT_IGNORE_INDEX,
    LMCrossEntropyLoss,
    LMCrossEntropyLossNode,
)

_BATCH, _SEQ, _VOCAB = 3, 5, 7


def _loss_fn(**params):
    return LMCrossEntropyLossNode().execute({}, params)["loss_fn"]


def _logits_and_targets(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(_BATCH, _SEQ, _VOCAB, generator=generator)
    targets = torch.randint(
        0, _VOCAB, (_BATCH, _SEQ), generator=generator, dtype=torch.int64)
    return logits, targets


# ── the node's contract ───────────────────────────────────────────────────

def test_node_metadata():
    assert LMCrossEntropyLossNode.NODE_NAME == "LMCrossEntropyLoss"
    assert LMCrossEntropyLossNode.CATEGORY == "LLM"
    assert LMCrossEntropyLossNode.define_inputs() == []
    outputs = LMCrossEntropyLossNode.define_outputs()
    assert [p.name for p in outputs] == ["loss_fn"]
    assert outputs[0].data_type is DataType.LOSS_FN


def test_the_node_stays_cacheable():
    """It hands out a small immutable function object built from two numbers.
    Nothing downstream mutates it, so the recorded handle describes it
    exactly -- the same reasoning that leaves ``Loss`` cacheable (#254)."""
    assert LMCrossEntropyLossNode.cacheable is True


def test_param_definitions_match_the_issue_spec():
    params = {p.name: p for p in LMCrossEntropyLossNode.define_params()}
    assert set(params) == {"ignore_index", "label_smoothing"}
    assert params["ignore_index"].default == -100
    assert params["label_smoothing"].default == 0.0
    assert params["label_smoothing"].min_value == 0.0
    assert params["label_smoothing"].max_value == 0.3


def test_the_ignore_index_default_matches_the_generic_loss_node():
    """A padding label written for one loss has to work in the other."""
    from app.nodes.training.loss_node import DEFAULT_IGNORE_INDEX as generic

    assert DEFAULT_IGNORE_INDEX == generic == -100


# ── the regression pin this design exists for ─────────────────────────────

def test_the_loss_is_not_a_cross_entropy_or_nll_subclass():
    """``TrainingLoop`` gates ``val_accuracy`` on
    ``isinstance(loss_fn, (nn.CrossEntropyLoss, nn.NLLLoss))`` and then runs
    ``outputs.argmax(dim=1)``. On [B, T, V] logits ``dim=1`` is the TIME axis,
    so subclassing either one would have produced a silent, meaningless
    accuracy number on every language-model run -- and early stopping can be
    pointed at it. Composing ``F.cross_entropy`` inside a plain module is
    what keeps that branch shut.
    """
    loss_fn = _loss_fn()
    assert isinstance(loss_fn, nn.Module)
    assert not isinstance(loss_fn, nn.CrossEntropyLoss)
    assert not isinstance(loss_fn, nn.NLLLoss)
    assert not isinstance(loss_fn, nn.modules.loss._WeightedLoss)


def test_the_training_loop_still_reads_the_flag_off_those_two_classes():
    """If ``TrainingLoop`` ever widens its gate -- to ``_Loss``, or to a duck
    type -- the test above stops protecting anything, so the assumption is
    pinned against the real source rather than trusted."""
    import inspect

    from app.nodes.training import training_loop_node

    source = inspect.getsource(training_loop_node)
    assert ("is_classification_loss = isinstance("
            "loss_fn, (nn.CrossEntropyLoss, nn.NLLLoss))" in source), (
        "TrainingLoop no longer decides val_accuracy by that isinstance "
        "check; re-check whether LMCrossEntropyLoss still avoids the "
        "argmax(dim=1) path it was designed to avoid (#289).")


# ── numerical agreement ───────────────────────────────────────────────────

def test_it_equals_hand_reshaped_cross_entropy():
    logits, targets = _logits_and_targets()
    expected = F.cross_entropy(
        logits.reshape(-1, _VOCAB), targets.reshape(-1))
    assert _loss_fn()(logits, targets).item() == pytest.approx(
        expected.item(), rel=1e-6)


def test_the_result_is_a_scalar_that_can_be_backpropagated():
    logits, targets = _logits_and_targets()
    logits.requires_grad_(True)
    loss = _loss_fn()(logits, targets)
    assert loss.dim() == 0
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_it_is_the_mean_over_positions_not_over_sequences():
    """A per-sequence mean would differ whenever the batch and the sequence
    length differ, which is every real run. Flattening both axes together is
    the definition every LM implementation uses."""
    logits, targets = _logits_and_targets()
    flat = _loss_fn()(logits.reshape(1, _BATCH * _SEQ, _VOCAB),
                      targets.reshape(1, _BATCH * _SEQ))
    assert _loss_fn()(logits, targets).item() == pytest.approx(
        flat.item(), rel=1e-6)


def test_non_contiguous_logits_are_accepted():
    """A transposed or sliced tensor out of an attention stack is not
    contiguous, and ``view`` would fail on exactly the inputs this loss
    exists to consume."""
    logits, targets = _logits_and_targets()
    # Same values, same shape, different memory layout: laid out as
    # (seq, batch, vocab) and viewed back as (batch, seq, vocab).
    transposed = logits.transpose(0, 1).contiguous().transpose(0, 1)
    assert transposed.shape == logits.shape
    assert torch.equal(transposed, logits)
    assert not transposed.is_contiguous()
    assert _loss_fn()(transposed, targets).item() == pytest.approx(
        _loss_fn()(logits, targets).item(), rel=1e-6)


def test_a_perfect_prediction_costs_almost_nothing():
    """The sanity check that a wrong axis would fail: put all the mass on the
    true token at every position and the loss has to collapse."""
    _, targets = _logits_and_targets()
    logits = torch.zeros(_BATCH, _SEQ, _VOCAB)
    logits.scatter_(2, targets.unsqueeze(-1), 30.0)
    assert _loss_fn()(logits, targets).item() < 1e-6


def test_the_loss_stays_in_float32_under_bf16_autocast():
    """``TrainingLoop`` runs the loss inside ``policy.autocast()`` precisely
    so that cross-entropy's softmax reduction happens in float32 rather than
    in bf16. Composing ``F.cross_entropy`` keeps that; a hand-rolled
    log-softmax would not."""
    logits, targets = _logits_and_targets()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = _loss_fn()(logits, targets)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss).all()


# ── ignore_index ──────────────────────────────────────────────────────────

def test_ignore_index_removes_those_positions_from_the_mean():
    logits, targets = _logits_and_targets()
    targets[0, 0] = DEFAULT_IGNORE_INDEX
    targets[2, 4] = DEFAULT_IGNORE_INDEX
    kept = targets != DEFAULT_IGNORE_INDEX
    expected = F.cross_entropy(logits[kept], targets[kept])

    assert _loss_fn()(logits, targets).item() == pytest.approx(
        expected.item(), rel=1e-6)


def test_ignored_positions_receive_no_gradient():
    logits, targets = _logits_and_targets()
    targets[1, 2] = DEFAULT_IGNORE_INDEX
    logits.requires_grad_(True)
    _loss_fn()(logits, targets).backward()
    assert torch.equal(logits.grad[1, 2], torch.zeros(_VOCAB))
    assert logits.grad[1, 3].abs().sum() > 0


def test_a_custom_ignore_index_is_honoured():
    logits, targets = _logits_and_targets()
    targets[0, 1] = 0
    without = _loss_fn(ignore_index=-1)(logits, targets)
    ignoring_zero = _loss_fn(ignore_index=0)(logits, targets)
    kept = targets != 0
    expected = F.cross_entropy(logits[kept], targets[kept])
    assert ignoring_zero.item() == pytest.approx(expected.item(), rel=1e-6)
    assert ignoring_zero.item() != pytest.approx(without.item(), rel=1e-6)


# ── label_smoothing ───────────────────────────────────────────────────────

def test_label_smoothing_reaches_torch():
    logits, targets = _logits_and_targets()
    expected = F.cross_entropy(
        logits.reshape(-1, _VOCAB), targets.reshape(-1), label_smoothing=0.1)
    smoothed = _loss_fn(label_smoothing=0.1)(logits, targets)
    assert smoothed.item() == pytest.approx(expected.item(), rel=1e-6)
    # And it is not silently ignored -- smoothing a confident prediction has
    # to cost something.
    assert smoothed.item() != pytest.approx(
        _loss_fn()(logits, targets).item(), rel=1e-6)


def test_label_smoothing_is_clamped_to_the_domain_torch_accepts():
    """Above 1.0 torch raises outright; below 0 there is nothing to smooth.
    Only reachable through hand-built graph JSON -- the FLOAT widget caps at
    the declared 0.3."""
    assert _loss_fn(label_smoothing=5.0).label_smoothing == 1.0
    assert _loss_fn(label_smoothing=-2.0).label_smoothing == 0.0


def test_the_params_survive_hand_written_json():
    assert _loss_fn(ignore_index=None).ignore_index == DEFAULT_IGNORE_INDEX
    assert _loss_fn(ignore_index="-1").ignore_index == -1
    assert _loss_fn(label_smoothing=None).label_smoothing == 0.0
    assert _loss_fn(label_smoothing="0.1").label_smoothing == pytest.approx(0.1)
    with pytest.raises(ValueError, match="whole number"):
        _loss_fn(ignore_index="pad")
    with pytest.raises(ValueError, match="label_smoothing must be a number"):
        _loss_fn(label_smoothing="a lot")


# ── shape errors say what to change ───────────────────────────────────────

def test_two_dimensional_logits_are_refused():
    """The mistake a learner makes by reaching for this node from a plain
    classifier graph."""
    with pytest.raises(ValueError, match="batch, seq_len, vocab_size"):
        _loss_fn()(torch.randn(4, _VOCAB), torch.zeros(4, dtype=torch.int64))


def test_one_hot_targets_are_refused():
    logits, targets = _logits_and_targets()
    one_hot = F.one_hot(targets, _VOCAB)
    with pytest.raises(ValueError, match="one token id per position"):
        _loss_fn()(logits, one_hot)


def test_a_target_length_mismatch_names_both_shapes():
    logits, _ = _logits_and_targets()
    short = torch.zeros(_BATCH, _SEQ - 1, dtype=torch.int64)
    with pytest.raises(ValueError) as excinfo:
        _loss_fn()(logits, short)
    message = str(excinfo.value)
    assert str((_BATCH, _SEQ)) in message and str((_BATCH, _SEQ - 1)) in message


def test_float_targets_are_refused_rather_than_read_as_probabilities():
    logits, targets = _logits_and_targets()
    with pytest.raises(ValueError, match="integer target token ids"):
        _loss_fn()(logits, targets.float())


# ── it behaves like the training loop expects a loss to behave ────────────

def test_it_can_be_moved_to_a_device():
    """``TrainingLoop`` calls ``to_device(loss_fn, device)`` on whatever it is
    handed, so the loss has to be a real module even though it owns nothing."""
    from app.core.device_utils import to_device

    loss_fn = to_device(_loss_fn(), "cpu")
    assert list(loss_fn.parameters()) == []
    assert list(loss_fn.buffers()) == []
    logits, targets = _logits_and_targets()
    assert torch.isfinite(loss_fn(logits, targets))


def test_it_survives_a_torch_save_round_trip():
    """Module scope, not a closure inside ``execute`` -- #283 was exactly this
    defect one package over: a class pickle cannot name is a checkpoint that
    cannot be written."""
    buffer = io.BytesIO()
    torch.save(_loss_fn(ignore_index=-1, label_smoothing=0.1), buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    assert isinstance(restored, LMCrossEntropyLoss)
    assert restored.ignore_index == -1
    assert restored.label_smoothing == pytest.approx(0.1)


def test_the_repr_names_the_settings():
    """What a learner sees when they print the loss, and what the node card's
    summary is built from."""
    text = repr(_loss_fn(ignore_index=-1, label_smoothing=0.1))
    assert "ignore_index=-1" in text
    assert "label_smoothing=0.1" in text
