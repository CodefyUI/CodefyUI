"""Tests for MaskedFillNode.

The node exists so a graph built from primitive tensor ops can apply the mask
``AttentionMask`` emits. Its whole contract is therefore about interoperating
with that node and with ``Softmax`` — which is what most of these assert:
``True`` means blocked, the fill lands *before* the softmax, and a row of
scores that survives still normalises to 1.
"""

from __future__ import annotations

import math

import pytest
import torch

from app.nodes.llm.attention_mask_node import AttentionMaskNode
from app.nodes.tensor_ops.masked_fill_node import MaskedFillNode


def _run(tensor, mask, **params):
    p = {"value": "-inf", "custom_value": 0.0}
    p.update(params)
    return MaskedFillNode().execute({"tensor": tensor, "mask": mask}, p)


def test_node_metadata():
    assert MaskedFillNode.NODE_NAME == "MaskedFill"
    assert MaskedFillNode.CATEGORY == "Tensor Operations"
    assert [p.name for p in MaskedFillNode.define_inputs()] == ["tensor", "mask"]
    assert [p.name for p in MaskedFillNode.define_outputs()] == ["tensor"]


def test_true_means_blocked():
    x = torch.ones(2, 2)
    mask = torch.tensor([[False, True], [True, False]])
    out = _run(x, mask)["tensor"]
    assert out[0, 0] == 1.0 and out[1, 1] == 1.0
    assert out[0, 1] == -math.inf and out[1, 0] == -math.inf


def test_shape_is_preserved():
    x = torch.randn(3, 5, 5)
    mask = torch.zeros(5, 5, dtype=torch.bool)
    assert _run(x, mask)["tensor"].shape == (3, 5, 5)


def test_mask_broadcasts_over_leading_dims():
    """A [seq, seq] causal mask must apply to [batch, seq, seq] scores."""
    x = torch.zeros(4, 3, 3)
    mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    out = _run(x, mask)["tensor"]
    for b in range(4):
        assert torch.isinf(out[b, 0, 1]) and torch.isinf(out[b, 0, 2])
        assert out[b, 2, 0] == 0.0


def test_non_boolean_mask_is_read_as_nonzero_blocked():
    x = torch.ones(2, 2)
    out = _run(x, torch.tensor([[0.0, 1.0], [0.0, 0.0]]))["tensor"]
    assert out[0, 1] == -math.inf
    assert out[0, 0] == 1.0 and out[1, 0] == 1.0


def test_zero_and_custom_fill_modes():
    x = torch.ones(2, 2)
    mask = torch.tensor([[False, True], [False, False]])
    assert _run(x, mask, value="zero")["tensor"][0, 1] == 0.0
    assert _run(x, mask, value="custom", custom_value=-7.5)["tensor"][0, 1] == -7.5


def test_integer_tensor_is_refused_with_a_readable_message():
    with pytest.raises(ValueError, match="floating-point"):
        _run(torch.ones(2, 2, dtype=torch.long), torch.zeros(2, 2, dtype=torch.bool))


def test_non_broadcastable_mask_names_both_shapes():
    with pytest.raises(ValueError, match=r"\(4, 4\).*\(3, 3\)|does not broadcast"):
        _run(torch.zeros(3, 3), torch.zeros(4, 4, dtype=torch.bool))


def test_unknown_value_mode_is_refused():
    with pytest.raises(ValueError, match="unknown value mode"):
        _run(torch.zeros(2, 2), torch.zeros(2, 2, dtype=torch.bool), value="nope")


# ── The reason the node exists: causal attention out of primitives ──────────


def test_softmax_after_minus_inf_fill_gives_a_clean_lower_triangle():
    """AttentionMask -> MaskedFill -> Softmax is the causal-attention chain.

    Blocked positions must come out at exactly zero, and every row -- even
    the first, which keeps a single position -- must still sum to 1. That is
    the property `-inf` buys and post-softmax zeroing destroys.
    """
    seq = 5
    scores = torch.randn(seq, seq)
    mask = AttentionMaskNode().execute(
        {"tensor": torch.zeros(seq, 8)}, {"mode": "causal", "pad_token": ""}
    )["mask"]

    filled = _run(scores, mask)["tensor"]
    weights = torch.softmax(filled, dim=-1)

    upper = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    assert torch.all(weights[upper] == 0.0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(seq), atol=1e-6)
    assert weights[0, 0] == pytest.approx(1.0)


def test_zero_fill_does_not_give_valid_attention_weights():
    """The counter-example the `value` param's description warns about."""
    seq = 4
    scores = torch.randn(seq, seq)
    mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)

    weights = torch.softmax(_run(scores, mask, value="zero")["tensor"], dim=-1)
    # Filling with 0 leaves exp(0)=1 in every blocked slot, so the blocked
    # positions keep a share of the probability mass.
    assert not torch.all(weights[mask] == 0.0)
