"""Tests for GRUNode."""

from __future__ import annotations

import torch

from app.core.execution_context import ExecutionContext
from app.core.node_state_store import NodeStateStore
from app.nodes.rnn.gru_node import GRUNode


def _ctx():
    return ExecutionContext(
        graph_id="g",
        weights_persistent=True,
        node_state_store=NodeStateStore(),
        current_node_id="gru",
    )


def test_node_metadata():
    assert GRUNode.NODE_NAME == "GRU"
    assert GRUNode.CATEGORY == "RNN"
    out_names = [p.name for p in GRUNode.define_outputs()]
    assert out_names == ["output", "hidden"]


def test_output_shape_batch_first():
    x = torch.randn(2, 10, 16)
    res = GRUNode().execute(
        {"tensor": x},
        {"input_size": 16, "hidden_size": 32, "num_layers": 1, "batch_first": True, "bidirectional": False},
        context=_ctx(),
    )
    assert res["output"].shape == (2, 10, 32)
    assert res["hidden"].shape == (1, 2, 32)


def test_bidirectional_doubles_output():
    x = torch.randn(2, 10, 16)
    res = GRUNode().execute(
        {"tensor": x},
        {"input_size": 16, "hidden_size": 32, "num_layers": 1, "batch_first": True, "bidirectional": True},
        context=_ctx(),
    )
    assert res["output"].shape == (2, 10, 64)


def test_multi_layer_stacked():
    x = torch.randn(1, 5, 8)
    res = GRUNode().execute(
        {"tensor": x},
        {"input_size": 8, "hidden_size": 16, "num_layers": 2, "batch_first": True, "bidirectional": False},
        context=_ctx(),
    )
    assert res["hidden"].shape == (2, 1, 16)


def test_batch_first_false_swaps_axes():
    x = torch.randn(10, 2, 16)
    res = GRUNode().execute(
        {"tensor": x},
        {"input_size": 16, "hidden_size": 32, "num_layers": 1, "batch_first": False, "bidirectional": False},
        context=_ctx(),
    )
    assert res["output"].shape == (10, 2, 32)
