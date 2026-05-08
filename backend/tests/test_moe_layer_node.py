"""Tests for MoELayerNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.transformer.moe_layer_node import MoELayerNode


def _run(x, **params):
    p = {
        "num_experts": 4,
        "top_k": 2,
        "hidden_dim": 8,
        "expert_hidden_dim": 16,
        "seed": 42,
    }
    p.update(params)
    return MoELayerNode().execute({"x": x}, p)


def test_node_metadata():
    assert MoELayerNode.NODE_NAME == "MoELayer"
    assert MoELayerNode.CATEGORY == "Transformer"
    out_names = [p.name for p in MoELayerNode.define_outputs()]
    assert "output" in out_names
    assert "routing_weights" in out_names
    assert "expert_indices" in out_names


def test_output_preserves_shape():
    """[B, T, H] in → [B, T, H] out."""
    x = torch.randn(2, 5, 8, generator=torch.Generator().manual_seed(0))
    res = _run(x)
    assert res["output"].shape == (2, 5, 8)


def test_routing_weights_top_k():
    """routing_weights should have shape [B, T, top_k] and sum to 1 per token."""
    x = torch.randn(2, 5, 8, generator=torch.Generator().manual_seed(0))
    res = _run(x, top_k=2)
    rw = res["routing_weights"]
    assert rw.shape == (2, 5, 2)
    sums = rw.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_expert_indices_in_range():
    x = torch.randn(2, 3, 8, generator=torch.Generator().manual_seed(0))
    res = _run(x, num_experts=4, top_k=2)
    idx = res["expert_indices"]
    assert idx.shape == (2, 3, 2)
    assert int(idx.min()) >= 0
    assert int(idx.max()) < 4


def test_top_k_one_uses_single_expert():
    x = torch.randn(1, 4, 8, generator=torch.Generator().manual_seed(0))
    res = _run(x, top_k=1)
    assert res["routing_weights"].shape == (1, 4, 1)
    # Each token should be assigned exactly one expert; weights all == 1.
    assert torch.allclose(res["routing_weights"], torch.ones(1, 4, 1), atol=1e-5)


def test_seed_reproducible():
    x = torch.randn(2, 3, 8, generator=torch.Generator().manual_seed(0))
    a = _run(x, seed=42)
    b = _run(x, seed=42)
    assert torch.allclose(a["output"], b["output"])
    c = _run(x, seed=99)
    assert not torch.allclose(a["output"], c["output"])


def test_top_k_clamped_to_num_experts():
    x = torch.randn(1, 2, 8, generator=torch.Generator().manual_seed(0))
    # top_k > num_experts should be clamped — no crash
    res = _run(x, num_experts=3, top_k=10)
    assert res["routing_weights"].shape[-1] == 3


def test_dim_mismatch_raises():
    x = torch.randn(1, 2, 16)  # H=16 but hidden_dim=8
    with pytest.raises(RuntimeError):
        _run(x, hidden_dim=8)


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        MoELayerNode().execute(
            {},
            {"num_experts": 4, "top_k": 2, "hidden_dim": 8, "expert_hidden_dim": 16, "seed": 42},
        )
