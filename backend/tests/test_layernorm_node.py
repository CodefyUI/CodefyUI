"""Tests for LayerNormNode."""

from __future__ import annotations

import torch

from app.core.execution_context import ExecutionContext
from app.core.node_state_store import NodeStateStore
from app.nodes.normalization.layernorm_node import LayerNormNode


def _ctx(verbose=False):
    return ExecutionContext(
        graph_id="g",
        weights_persistent=True,
        node_state_store=NodeStateStore(),
        current_node_id="ln",
        verbose=verbose,
    )


def test_node_metadata():
    assert LayerNormNode.NODE_NAME == "LayerNorm"
    assert LayerNormNode.CATEGORY == "Normalization"


def test_parse_shape_handles_single_int():
    assert LayerNormNode._parse_shape("512") == [512]


def test_parse_shape_handles_multiple_dims():
    assert LayerNormNode._parse_shape("64,32") == [64, 32]


def test_parse_shape_ignores_whitespace():
    assert LayerNormNode._parse_shape(" 64 , 32 ") == [64, 32]


def test_normalizes_last_dim_to_zero_mean():
    x = torch.randn(2, 4, 16)
    res = LayerNormNode().execute(
        {"tensor": x},
        {"normalized_shape": "16", "eps": 1e-5},
        context=_ctx(),
    )
    # Mean over last dim should be ~0 per sample
    mean = res["tensor"].mean(dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)


def test_normalizes_last_dim_to_unit_var():
    x = torch.randn(2, 4, 16)
    res = LayerNormNode().execute(
        {"tensor": x},
        {"normalized_shape": "16", "eps": 1e-5},
        context=_ctx(),
    )
    var = res["tensor"].var(dim=-1, unbiased=False)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-3)


def test_verbose_mode_records_steps():
    x = torch.randn(2, 4, 16)
    res = LayerNormNode().execute(
        {"tensor": x},
        {"normalized_shape": "16", "eps": 1e-5},
        context=_ctx(verbose=True),
    )
    assert "__steps__" in res
    step_names = [s.name for s in res["__steps__"]]
    assert "compute_mean" in step_names
    assert "normalize" in step_names


def test_multi_dim_normalized_shape():
    x = torch.randn(2, 4, 8)
    res = LayerNormNode().execute(
        {"tensor": x},
        {"normalized_shape": "4,8", "eps": 1e-5},
        context=_ctx(),
    )
    assert res["tensor"].shape == x.shape
