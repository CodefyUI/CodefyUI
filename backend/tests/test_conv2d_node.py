"""Tests for Conv2dNode."""

from __future__ import annotations

import torch

from app.core.execution_context import ExecutionContext
from app.core.node_state_store import NodeStateStore
from app.nodes.cnn.conv2d_node import Conv2dNode


def _ctx(**kw):
    return ExecutionContext(
        graph_id="g",
        weights_persistent=True,
        node_state_store=NodeStateStore(),
        current_node_id="conv",
        **kw,
    )


def test_node_metadata():
    assert Conv2dNode.NODE_NAME == "Conv2d"
    assert Conv2dNode.CATEGORY == "CNN"
    assert "in_channels" in Conv2dNode.structural_params
    assert "out_channels" in Conv2dNode.structural_params


def test_output_channels_match_param():
    x = torch.randn(1, 3, 16, 16)
    res = Conv2dNode().execute(
        {"tensor": x},
        {"in_channels": 3, "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1},
        context=_ctx(),
    )
    assert res["tensor"].shape[1] == 8


def test_same_padding_preserves_spatial_dims():
    x = torch.randn(2, 1, 8, 8)
    res = Conv2dNode().execute(
        {"tensor": x},
        {"in_channels": 1, "out_channels": 4, "kernel_size": 3, "stride": 1, "padding": 1},
        context=_ctx(),
    )
    assert res["tensor"].shape[2] == 8
    assert res["tensor"].shape[3] == 8


def test_stride_two_halves_spatial_dims():
    x = torch.randn(1, 1, 16, 16)
    res = Conv2dNode().execute(
        {"tensor": x},
        {"in_channels": 1, "out_channels": 4, "kernel_size": 3, "stride": 2, "padding": 1},
        context=_ctx(),
    )
    assert res["tensor"].shape[2] == 8
    assert res["tensor"].shape[3] == 8


def test_no_padding_shrinks():
    x = torch.randn(1, 1, 16, 16)
    res = Conv2dNode().execute(
        {"tensor": x},
        {"in_channels": 1, "out_channels": 4, "kernel_size": 3, "stride": 1, "padding": 0},
        context=_ctx(),
    )
    assert res["tensor"].shape[2] == 14


def test_verbose_mode_records_steps():
    x = torch.randn(1, 1, 8, 8)
    ctx = _ctx(verbose=True)
    res = Conv2dNode().execute(
        {"tensor": x},
        {"in_channels": 1, "out_channels": 4, "kernel_size": 3, "stride": 1, "padding": 1},
        context=ctx,
    )
    assert "__steps__" in res
    step_names = [s.name for s in res["__steps__"]]
    assert "input_tensor" in step_names
    assert "kernel_weights" in step_names
    assert "convolved_output" in step_names


def test_batch_preserved():
    x = torch.randn(8, 1, 4, 4)
    res = Conv2dNode().execute(
        {"tensor": x},
        {"in_channels": 1, "out_channels": 2, "kernel_size": 3, "stride": 1, "padding": 1},
        context=_ctx(),
    )
    assert res["tensor"].shape[0] == 8


# ── #703: groups, dilation and bias ──────────────────────────────────────


def test_canvas_conv2d_builds_a_depthwise_conv():
    x = torch.randn(2, 6, 9, 9)
    node = Conv2dNode()
    params = {"in_channels": 6, "out_channels": 6, "kernel_size": 3, "stride": 1,
              "padding": 2, "dilation": 2, "groups": 6, "bias": False}
    conv = node.build_module(params)
    assert conv.weight.shape == (6, 1, 3, 3)
    assert conv.bias is None
    out = node.execute({"tensor": x}, params, context=_ctx())["tensor"]
    assert out.shape == (2, 6, 9, 9)


def test_canvas_conv2d_defaults_build_the_old_module():
    defaults = {p.name: p.default for p in Conv2dNode.define_params()}
    conv = Conv2dNode().build_module(defaults)
    assert conv.groups == 1 and conv.dilation == (1, 1) and conv.bias is not None


def test_changing_groups_rebuilds_the_persisted_module():
    # groups changes the weight shape, so it has to be part of the
    # structure hash; otherwise a persisted (6, 6, 3, 3) weight would be
    # handed to a node that now asks for a depthwise conv.
    ctx = _ctx()
    base = {"in_channels": 6, "out_channels": 6, "kernel_size": 3, "stride": 1, "padding": 1}
    x = torch.randn(1, 6, 8, 8)
    Conv2dNode().execute({"tensor": x}, {**base, "groups": 1}, context=ctx)
    node = Conv2dNode()
    node.execute({"tensor": x}, {**base, "groups": 6}, context=ctx)
    assert node.get_or_build_module(ctx, {**base, "groups": 6}).groups == 6
