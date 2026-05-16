"""Tests for InstanceNorm2dNode."""

from __future__ import annotations

import torch

from app.core.execution_context import ExecutionContext
from app.core.node_state_store import NodeStateStore
from app.nodes.normalization.instancenorm2d_node import InstanceNorm2dNode


def _ctx():
    return ExecutionContext(
        graph_id="g",
        weights_persistent=True,
        node_state_store=NodeStateStore(),
        current_node_id="in",
    )


def test_node_metadata():
    assert InstanceNorm2dNode.NODE_NAME == "InstanceNorm2d"
    assert InstanceNorm2dNode.CATEGORY == "Normalization"


def test_preserves_shape():
    x = torch.randn(2, 4, 8, 8)
    res = InstanceNorm2dNode().execute(
        {"tensor": x},
        {"num_features": 4, "affine": False},
        context=_ctx(),
    )
    assert res["tensor"].shape == x.shape


def test_normalizes_per_instance_per_channel():
    """Mean over spatial dims per (N, C) should be ~0."""
    torch.manual_seed(0)
    x = torch.randn(4, 8, 16, 16)
    res = InstanceNorm2dNode().execute(
        {"tensor": x},
        {"num_features": 8, "affine": False},
        context=_ctx(),
    )
    per_nc_mean = res["tensor"].mean(dim=(2, 3))
    assert torch.allclose(per_nc_mean, torch.zeros_like(per_nc_mean), atol=1e-5)


def test_affine_true_includes_learnable_params():
    node = InstanceNorm2dNode()
    ctx = _ctx()
    mod = node.get_or_build_module(ctx, {"num_features": 4, "affine": True})
    assert mod.weight is not None
    assert mod.bias is not None


def test_affine_false_has_no_params():
    node = InstanceNorm2dNode()
    ctx = _ctx()
    mod = node.get_or_build_module(ctx, {"num_features": 4, "affine": False})
    assert mod.weight is None
    assert mod.bias is None
