"""Tests for BatchNormNode (BatchNorm2d)."""

from __future__ import annotations

import torch

from app.nodes.cnn.batchnorm_node import BatchNormNode


def _run(tensor, context=None, **params):
    return BatchNormNode().execute({"tensor": tensor}, params, context=context)


def test_node_metadata():
    assert BatchNormNode.NODE_NAME == "BatchNorm2d"
    assert BatchNormNode.CATEGORY == "CNN"
    assert "num_features" in BatchNormNode.structural_params


def test_output_shape_matches_input_shape():
    x = torch.randn(4, 16, 8, 8)
    res = _run(x, num_features=16)
    assert res["tensor"].shape == x.shape


def test_normalized_channel_has_near_zero_mean_in_training_mode():
    """In training mode (no running stats updated yet), per-channel mean should be ~0."""
    torch.manual_seed(0)
    x = torch.randn(8, 4, 16, 16)
    res = _run(x, num_features=4)
    # Mean across (N, H, W) per channel should be ~0
    per_channel_mean = res["tensor"].mean(dim=(0, 2, 3))
    assert torch.allclose(per_channel_mean, torch.zeros(4), atol=1e-5)


def test_normalized_channel_has_near_unit_var_in_training_mode():
    torch.manual_seed(0)
    x = torch.randn(8, 4, 16, 16)
    res = _run(x, num_features=4)
    per_channel_var = res["tensor"].var(dim=(0, 2, 3), unbiased=False)
    assert torch.allclose(per_channel_var, torch.ones(4), atol=1e-3)


def test_verbose_mode_records_steps():
    class _Ctx:
        verbose = True
        graph_id = "g"
        weights_persistent = False
        node_state_store = None
        current_node_id = "bn"

    x = torch.randn(2, 4, 4, 4)
    # Verbose mode requires get_or_build_module which needs a non-None context;
    # build_module falls back when state store is missing.
    from app.core.execution_context import ExecutionContext
    from app.core.node_state_store import NodeStateStore
    ctx = ExecutionContext(
        graph_id="g",
        weights_persistent=False,
        node_state_store=NodeStateStore(),
        current_node_id="bn",
        verbose=True,
    )
    res = _run(x, num_features=4, context=ctx)
    assert "__steps__" in res
    step_names = [s.name for s in res["__steps__"]]
    assert "per_channel_mean" in step_names
    assert "scale_shift" in step_names
