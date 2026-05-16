"""Tests for TransformerEncoderNode."""

from __future__ import annotations

import torch

from app.core.execution_context import ExecutionContext
from app.core.node_state_store import NodeStateStore
from app.nodes.transformer.encoder_node import TransformerEncoderNode


def _ctx():
    return ExecutionContext(
        graph_id="g",
        weights_persistent=True,
        node_state_store=NodeStateStore(),
        current_node_id="enc",
    )


def test_node_metadata():
    assert TransformerEncoderNode.NODE_NAME == "TransformerEncoder"
    assert TransformerEncoderNode.CATEGORY == "Transformer"


def test_preserves_input_shape():
    # (seq_len, batch, d_model)
    x = torch.randn(8, 2, 16)
    res = TransformerEncoderNode().execute(
        {"tensor": x},
        {"d_model": 16, "nhead": 4, "num_layers": 2, "dim_feedforward": 32},
        context=_ctx(),
    )
    assert res["tensor"].shape == x.shape


def test_different_seq_lengths_work():
    x1 = torch.randn(4, 1, 16)
    x2 = torch.randn(16, 1, 16)
    enc = TransformerEncoderNode()
    ctx = _ctx()
    p = {"d_model": 16, "nhead": 4, "num_layers": 1, "dim_feedforward": 32}
    res1 = enc.execute({"tensor": x1}, p, context=ctx)
    res2 = enc.execute({"tensor": x2}, p, context=ctx)
    assert res1["tensor"].shape == (4, 1, 16)
    assert res2["tensor"].shape == (16, 1, 16)


def test_num_layers_param_changes_depth():
    """Different num_layers triggers a new module via structural_params."""
    enc = TransformerEncoderNode()
    ctx = _ctx()
    a = enc.get_or_build_module(ctx, {"d_model": 16, "nhead": 4, "num_layers": 2, "dim_feedforward": 32})
    b = enc.get_or_build_module(ctx, {"d_model": 16, "nhead": 4, "num_layers": 4, "dim_feedforward": 32})
    assert a is not b
    assert len(b.layers) == 4
