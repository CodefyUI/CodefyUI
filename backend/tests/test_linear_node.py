"""Tests for LinearNode."""

from __future__ import annotations

import torch

from app.core.execution_context import ExecutionContext
from app.core.node_state_store import NodeStateStore
from app.nodes.utility.linear_node import LinearNode


def _ctx(verbose=False):
    return ExecutionContext(
        graph_id="g",
        weights_persistent=True,
        node_state_store=NodeStateStore(),
        current_node_id="lin",
        verbose=verbose,
    )


def test_node_metadata():
    assert LinearNode.NODE_NAME == "Linear"
    assert LinearNode.CATEGORY == "Utility"
    assert "in_features" in LinearNode.structural_params
    assert "out_features" in LinearNode.structural_params


def test_output_features_match_param():
    x = torch.randn(8, 32)
    res = LinearNode().execute(
        {"tensor": x},
        {"in_features": 32, "out_features": 10},
        context=_ctx(),
    )
    assert res["tensor"].shape == (8, 10)


def test_preserves_batch_dims():
    x = torch.randn(4, 16, 32)
    res = LinearNode().execute(
        {"tensor": x},
        {"in_features": 32, "out_features": 64},
        context=_ctx(),
    )
    assert res["tensor"].shape == (4, 16, 64)


def test_verbose_mode_records_steps():
    x = torch.randn(2, 32)
    res = LinearNode().execute(
        {"tensor": x},
        {"in_features": 32, "out_features": 10},
        context=_ctx(verbose=True),
    )
    assert "__steps__" in res
    step_names = [s.name for s in res["__steps__"]]
    assert "input" in step_names
    assert "weight_matrix" in step_names
    assert "linear_output" in step_names


def test_zero_input_gives_bias_only():
    x = torch.zeros(1, 8)
    node = LinearNode()
    ctx = _ctx()
    res = node.execute({"tensor": x}, {"in_features": 8, "out_features": 4}, context=ctx)
    # Output should equal the bias broadcast
    mod = node.get_or_build_module(ctx, {"in_features": 8, "out_features": 4})
    assert torch.allclose(res["tensor"][0], mod.bias)
