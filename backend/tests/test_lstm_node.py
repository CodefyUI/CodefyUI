"""Tests for LSTMNode."""

from __future__ import annotations

import torch

from app.core.execution_context import ExecutionContext
from app.core.node_state_store import NodeStateStore
from app.nodes.rnn.lstm_node import LSTMNode


def _ctx():
    return ExecutionContext(
        graph_id="g",
        weights_persistent=True,
        node_state_store=NodeStateStore(),
        current_node_id="lstm",
    )


def test_node_metadata():
    assert LSTMNode.NODE_NAME == "LSTM"
    assert LSTMNode.CATEGORY == "RNN"
    out_names = [p.name for p in LSTMNode.define_outputs()]
    assert out_names == ["output", "hidden"]


def test_output_shape_batch_first():
    x = torch.randn(2, 10, 16)  # (batch, seq, input)
    res = LSTMNode().execute(
        {"tensor": x},
        {"input_size": 16, "hidden_size": 32, "num_layers": 1, "batch_first": True, "bidirectional": False},
        context=_ctx(),
    )
    # output: (batch, seq, hidden)
    assert res["output"].shape == (2, 10, 32)
    # hidden: (num_layers, batch, hidden)
    assert res["hidden"].shape == (1, 2, 32)


def test_bidirectional_doubles_output_features():
    x = torch.randn(2, 10, 16)
    res = LSTMNode().execute(
        {"tensor": x},
        {"input_size": 16, "hidden_size": 32, "num_layers": 1, "batch_first": True, "bidirectional": True},
        context=_ctx(),
    )
    assert res["output"].shape == (2, 10, 64)


def test_multi_layer():
    x = torch.randn(1, 5, 8)
    res = LSTMNode().execute(
        {"tensor": x},
        {"input_size": 8, "hidden_size": 16, "num_layers": 3, "batch_first": True, "bidirectional": False},
        context=_ctx(),
    )
    assert res["hidden"].shape == (3, 1, 16)


def test_batch_first_false():
    x = torch.randn(10, 2, 16)  # (seq, batch, input)
    res = LSTMNode().execute(
        {"tensor": x},
        {"input_size": 16, "hidden_size": 32, "num_layers": 1, "batch_first": False, "bidirectional": False},
        context=_ctx(),
    )
    # output: (seq, batch, hidden)
    assert res["output"].shape == (10, 2, 32)


def test_structural_param_change_rebuilds():
    node = LSTMNode()
    ctx = _ctx()
    x = torch.randn(1, 5, 8)
    res_a = node.execute(
        {"tensor": x},
        {"input_size": 8, "hidden_size": 16, "num_layers": 1, "batch_first": True, "bidirectional": False},
        context=ctx,
    )
    res_b = node.execute(
        {"tensor": x},
        {"input_size": 8, "hidden_size": 32, "num_layers": 1, "batch_first": True, "bidirectional": False},
        context=ctx,
    )
    assert res_a["output"].shape[-1] == 16
    assert res_b["output"].shape[-1] == 32
