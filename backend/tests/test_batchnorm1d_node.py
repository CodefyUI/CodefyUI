"""Tests for BatchNorm1dNode."""

from __future__ import annotations

import torch

from app.core.execution_context import ExecutionContext
from app.core.node_state_store import NodeStateStore
from app.nodes.normalization.batchnorm1d_node import BatchNorm1dNode


def _ctx():
    return ExecutionContext(
        graph_id="g",
        weights_persistent=True,
        node_state_store=NodeStateStore(),
        current_node_id="bn1d",
    )


def test_node_metadata():
    assert BatchNorm1dNode.NODE_NAME == "BatchNorm1d"
    assert BatchNorm1dNode.CATEGORY == "Normalization"


def test_2d_input_normalizes_features():
    torch.manual_seed(0)
    x = torch.randn(16, 8)
    res = BatchNorm1dNode().execute({"tensor": x}, {"num_features": 8}, context=_ctx())
    assert res["tensor"].shape == x.shape
    # In training mode, per-feature mean should be ~0
    assert torch.allclose(res["tensor"].mean(dim=0), torch.zeros(8), atol=1e-5)


def test_3d_input_supported():
    x = torch.randn(2, 8, 16)
    res = BatchNorm1dNode().execute({"tensor": x}, {"num_features": 8}, context=_ctx())
    assert res["tensor"].shape == x.shape


def test_normalized_variance_near_one():
    torch.manual_seed(0)
    x = torch.randn(32, 4)
    res = BatchNorm1dNode().execute({"tensor": x}, {"num_features": 4}, context=_ctx())
    var = res["tensor"].var(dim=0, unbiased=False)
    assert torch.allclose(var, torch.ones(4), atol=1e-3)
