"""Tests for EduCrossAttentionNode."""

from __future__ import annotations

import pytest
import torch

from cdui_plugins.deep.nodes.edu_cross_attention_node import EduCrossAttentionNode


def _run(query, context, *, mask=None, q_labels=None, k_labels=None, **params):
    p = {"embed_dim": 8, "num_heads": 2, "seed": 42}
    p.update(params)
    inputs: dict = {"query": query, "context": context}
    if mask is not None:
        inputs["mask"] = mask
    if q_labels is not None:
        inputs["q_labels"] = q_labels
    if k_labels is not None:
        inputs["k_labels"] = k_labels
    return EduCrossAttentionNode().execute(inputs, p)


def test_node_metadata():
    assert EduCrossAttentionNode.NODE_NAME == "Edu-CrossAttention"
    assert EduCrossAttentionNode.CATEGORY == "Diffusion"
    out_names = [p.name for p in EduCrossAttentionNode.define_outputs()]
    assert out_names == ["output", "weights", "q_labels", "k_labels"]


def test_output_shape_2d_inputs():
    """Q [Q_seq, D], K/V [K_seq, D] → output [Q_seq, D], weights [H, Q_seq, K_seq]."""
    q = torch.randn(5, 8, generator=torch.Generator().manual_seed(0))
    ctx = torch.randn(7, 8, generator=torch.Generator().manual_seed(1))
    res = _run(q, ctx, num_heads=2)
    assert res["output"].shape == (5, 8)
    assert res["weights"].shape == (2, 5, 7)


def test_q_seq_can_differ_from_k_seq():
    """Critical for cross-attention: Q from latent, K/V from text — different lengths."""
    q = torch.randn(10, 8)  # 10 image patches
    ctx = torch.randn(4, 8)  # 4 text tokens
    res = _run(q, ctx, num_heads=2)
    assert res["output"].shape == (10, 8)
    assert res["weights"].shape == (2, 10, 4)


def test_softmax_rows_sum_to_one():
    """Each Q's attention over K must sum to 1."""
    q = torch.randn(6, 8)
    ctx = torch.randn(9, 8)
    res = _run(q, ctx)
    row_sums = res["weights"].sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_3d_inputs_with_batch():
    """[Q_seq, batch, D] and [K_seq, batch, D] → output [Q_seq, batch, D]."""
    q = torch.randn(5, 3, 8)
    ctx = torch.randn(7, 3, 8)
    res = _run(q, ctx, num_heads=2)
    assert res["output"].shape == (5, 3, 8)
    # weights: [batch, H, Q_seq, K_seq]
    assert res["weights"].shape == (3, 2, 5, 7)


def test_explicit_mask_blocks_columns():
    """Mask [Q_seq, K_seq] (True = blocked) — masked columns get zero weight."""
    q = torch.randn(4, 8)
    ctx = torch.randn(5, 8)
    mask = torch.zeros(4, 5, dtype=torch.bool)
    mask[:, 2] = True  # block context position 2 entirely
    res = _run(q, ctx, mask=mask)
    # All heads should respect — column 2 zeros across rows.
    assert torch.all(res["weights"][:, :, 2] == 0.0)


def test_deterministic_given_seed():
    q = torch.randn(4, 8, generator=torch.Generator().manual_seed(0))
    ctx = torch.randn(5, 8, generator=torch.Generator().manual_seed(1))
    a = _run(q, ctx, seed=42)
    b = _run(q, ctx, seed=42)
    assert torch.allclose(a["output"], b["output"])
    assert torch.allclose(a["weights"], b["weights"])


def test_labels_pass_through():
    q = torch.randn(3, 8)
    ctx = torch.randn(4, 8)
    res = _run(q, ctx, q_labels=["a", "b", "c"], k_labels=["w", "x", "y", "z"])
    assert res["q_labels"] == ["a", "b", "c"]
    assert res["k_labels"] == ["w", "x", "y", "z"]


def test_labels_default_empty():
    q = torch.randn(3, 8)
    ctx = torch.randn(4, 8)
    res = _run(q, ctx)
    assert res["q_labels"] == []
    assert res["k_labels"] == []


def test_embed_dim_must_divide_num_heads():
    with pytest.raises(ValueError, match="num_heads"):
        _run(torch.zeros(2, 8), torch.zeros(3, 8), embed_dim=8, num_heads=3)


def test_query_embed_dim_must_match_context():
    """Q and context must share the same last-dim (embed_dim)."""
    with pytest.raises(ValueError, match="embed_dim"):
        _run(torch.zeros(2, 8), torch.zeros(3, 16))


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        EduCrossAttentionNode().execute(
            {"query": torch.zeros(2, 8)},
            {"embed_dim": 8, "num_heads": 2, "seed": 42},
        )
