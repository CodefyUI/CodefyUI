"""Tests for Conv1dNode."""

from __future__ import annotations

import torch

from app.core.execution_context import ExecutionContext
from app.core.node_state_store import NodeStateStore
from app.nodes.cnn.conv1d_node import Conv1dNode


def _ctx():
    return ExecutionContext(
        graph_id="g",
        weights_persistent=True,
        node_state_store=NodeStateStore(),
        current_node_id="conv1d",
    )


def test_node_metadata():
    assert Conv1dNode.NODE_NAME == "Conv1d"
    assert Conv1dNode.CATEGORY == "CNN"


def test_output_channels_match_param():
    x = torch.randn(2, 3, 32)
    res = Conv1dNode().execute(
        {"tensor": x},
        {"in_channels": 3, "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1},
        context=_ctx(),
    )
    assert res["tensor"].shape[1] == 8


def test_same_padding_preserves_length():
    x = torch.randn(1, 1, 32)
    res = Conv1dNode().execute(
        {"tensor": x},
        {"in_channels": 1, "out_channels": 4, "kernel_size": 3, "stride": 1, "padding": 1},
        context=_ctx(),
    )
    assert res["tensor"].shape[2] == 32


def test_stride_two_halves_length():
    x = torch.randn(1, 1, 32)
    res = Conv1dNode().execute(
        {"tensor": x},
        {"in_channels": 1, "out_channels": 4, "kernel_size": 3, "stride": 2, "padding": 1},
        context=_ctx(),
    )
    assert res["tensor"].shape[2] == 16


def test_batch_preserved():
    x = torch.randn(4, 1, 16)
    res = Conv1dNode().execute(
        {"tensor": x},
        {"in_channels": 1, "out_channels": 2, "kernel_size": 3, "stride": 1, "padding": 1},
        context=_ctx(),
    )
    assert res["tensor"].shape[0] == 4
