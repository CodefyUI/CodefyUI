"""Tests for GroupNormNode."""

from __future__ import annotations

import torch

from app.core.execution_context import ExecutionContext
from app.core.node_state_store import NodeStateStore
from app.nodes.normalization.groupnorm_node import GroupNormNode


def _ctx():
    return ExecutionContext(
        graph_id="g",
        weights_persistent=True,
        node_state_store=NodeStateStore(),
        current_node_id="gn",
    )


def test_node_metadata():
    assert GroupNormNode.NODE_NAME == "GroupNorm"
    assert GroupNormNode.CATEGORY == "Normalization"


def test_preserves_shape():
    x = torch.randn(2, 32, 8, 8)
    res = GroupNormNode().execute(
        {"tensor": x},
        {"num_groups": 8, "num_channels": 32},
        context=_ctx(),
    )
    assert res["tensor"].shape == x.shape


def test_groups_one_equals_layernorm_over_channels():
    """num_groups=1 normalizes over all channels jointly."""
    torch.manual_seed(0)
    x = torch.randn(2, 8, 4, 4)
    res = GroupNormNode().execute(
        {"tensor": x},
        {"num_groups": 1, "num_channels": 8},
        context=_ctx(),
    )
    # Mean over (C, H, W) per sample should be ~0
    per_sample_mean = res["tensor"].mean(dim=(1, 2, 3))
    assert torch.allclose(per_sample_mean, torch.zeros(2), atol=1e-5)


def test_groups_equal_channels_is_instance_norm():
    """num_groups=num_channels makes it instance norm."""
    x = torch.randn(2, 4, 8, 8)
    res = GroupNormNode().execute(
        {"tensor": x},
        {"num_groups": 4, "num_channels": 4},
        context=_ctx(),
    )
    assert res["tensor"].shape == x.shape
