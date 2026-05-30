"""Tests for EduMultiHeadAttentionNode."""

from __future__ import annotations

import pytest
import torch

from cdui_plugins.deep.nodes.edu_multi_head_attention_node import EduMultiHeadAttentionNode


def _run(tensor, *, mask=None, labels=None, **params):
    p = {"embed_dim": 8, "num_heads": 2, "causal": False, "seed": 42}
    p.update(params)
    inputs: dict = {"tensor": tensor}
    if mask is not None:
        inputs["mask"] = mask
    if labels is not None:
        inputs["labels"] = labels
    return EduMultiHeadAttentionNode().execute(inputs, p)


def test_node_metadata():
    assert EduMultiHeadAttentionNode.NODE_NAME == "Edu-MultiHeadAttention"
    assert EduMultiHeadAttentionNode.CATEGORY == "LLM"
    out_names = [p.name for p in EduMultiHeadAttentionNode.define_outputs()]
    assert out_names == ["output", "weights", "labels"]


def test_output_shape_2d_input():
    res = _run(torch.zeros(5, 8), num_heads=2)
    assert res["output"].shape == (5, 8)
    assert res["weights"].shape == (2, 5, 5)  # [H, seq, seq]


def test_output_shape_3d_input():
    res = _run(torch.zeros(5, 3, 8), num_heads=4, embed_dim=8)
    assert res["output"].shape == (5, 3, 8)
    assert res["weights"].shape == (3, 4, 5, 5)  # [batch, H, seq, seq]


def test_each_head_softmax_rows_sum_to_one():
    res = _run(torch.randn(6, 8), num_heads=2)
    # weights [H, seq, seq] — sum along last dim must be 1 for each (head, query).
    row_sums = res["weights"].sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_causal_mask_blocks_future_per_head():
    res = _run(torch.randn(5, 8), num_heads=2, causal=True)
    w = res["weights"]  # [H, seq, seq]
    for h in range(2):
        for i in range(5):
            for j in range(i + 1, 5):
                assert w[h, i, j].item() == 0.0


def test_heads_produce_different_attention_patterns():
    """Different head projections → different attention patterns (most of the time)."""
    res = _run(torch.randn(6, 8, generator=torch.Generator().manual_seed(7)), num_heads=4)
    w = res["weights"]  # [4, 6, 6]
    # At least some pair of heads should differ.
    h0, h1 = w[0], w[1]
    assert not torch.allclose(h0, h1, atol=1e-3)


def test_embed_dim_must_divide_evenly_by_num_heads():
    with pytest.raises(ValueError, match="num_heads"):
        _run(torch.zeros(4, 8), embed_dim=8, num_heads=3)  # 8 % 3 != 0


def test_deterministic_given_seed():
    x = torch.randn(4, 8, generator=torch.Generator().manual_seed(11))
    a = _run(x, seed=42)
    b = _run(x, seed=42)
    assert torch.allclose(a["output"], b["output"])
    assert torch.allclose(a["weights"], b["weights"])


def test_explicit_mask_blocks_columns():
    seq = 4
    mask = torch.zeros(seq, seq, dtype=torch.bool)
    mask[:, 2] = True
    res = _run(torch.randn(seq, 8), mask=mask, num_heads=2)
    # All heads should respect the mask.
    assert torch.all(res["weights"][:, :, 2] == 0.0)


def test_embed_dim_mismatch_raises():
    with pytest.raises(ValueError, match="embed_dim"):
        _run(torch.zeros(4, 4), embed_dim=8, num_heads=2)


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        EduMultiHeadAttentionNode().execute(
            {}, {"embed_dim": 8, "num_heads": 2, "causal": False, "seed": 42}
        )
