"""Tests for ConvTranspose2dNode."""

from __future__ import annotations

import torch

from app.core.execution_context import ExecutionContext
from app.core.node_state_store import NodeStateStore
from app.nodes.cnn.conv_transpose2d_node import ConvTranspose2dNode


def _ctx():
    return ExecutionContext(
        graph_id="g",
        weights_persistent=True,
        node_state_store=NodeStateStore(),
        current_node_id="ct",
    )


def test_node_metadata():
    assert ConvTranspose2dNode.NODE_NAME == "ConvTranspose2d"
    assert ConvTranspose2dNode.CATEGORY == "CNN"


def test_default_doubles_spatial_dims():
    x = torch.randn(1, 64, 8, 8)
    # default: kernel 2, stride 2, padding 0
    res = ConvTranspose2dNode().execute({"tensor": x}, {}, context=_ctx())
    assert res["tensor"].shape == (1, 32, 16, 16)


def test_output_channels_match_param():
    x = torch.randn(1, 32, 4, 4)
    res = ConvTranspose2dNode().execute(
        {"tensor": x},
        {"in_channels": 32, "out_channels": 16, "kernel_size": 2, "stride": 2, "padding": 0},
        context=_ctx(),
    )
    assert res["tensor"].shape[1] == 16


def test_stride_two_doubles_spatial():
    x = torch.randn(1, 4, 8, 8)
    res = ConvTranspose2dNode().execute(
        {"tensor": x},
        {"in_channels": 4, "out_channels": 4, "kernel_size": 2, "stride": 2, "padding": 0},
        context=_ctx(),
    )
    assert res["tensor"].shape == (1, 4, 16, 16)


def test_kernel_size_3_stride_2_with_padding_1():
    x = torch.randn(1, 4, 8, 8)
    res = ConvTranspose2dNode().execute(
        {"tensor": x},
        {"in_channels": 4, "out_channels": 4, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
        context=_ctx(),
    )
    # (in - 1) * stride - 2*padding + kernel + output_padding = 7*2 - 2 + 3 + 1 = 16
    assert res["tensor"].shape == (1, 4, 16, 16)
