"""Tests for TransformerDecoderNode."""

from __future__ import annotations

import torch

from app.core.execution_context import ExecutionContext
from app.core.node_state_store import NodeStateStore
from app.nodes.transformer.decoder_node import TransformerDecoderNode


def _ctx():
    return ExecutionContext(
        graph_id="g",
        weights_persistent=True,
        node_state_store=NodeStateStore(),
        current_node_id="dec",
    )


def test_node_metadata():
    assert TransformerDecoderNode.NODE_NAME == "TransformerDecoder"
    assert TransformerDecoderNode.CATEGORY == "Transformer"
    in_names = [p.name for p in TransformerDecoderNode.define_inputs()]
    assert "tensor" in in_names
    assert "memory" in in_names


def test_preserves_target_shape():
    tgt = torch.randn(5, 2, 16)
    mem = torch.randn(8, 2, 16)
    res = TransformerDecoderNode().execute(
        {"tensor": tgt, "memory": mem},
        {"d_model": 16, "nhead": 4, "num_layers": 1, "dim_feedforward": 32},
        context=_ctx(),
    )
    assert res["tensor"].shape == (5, 2, 16)


def test_different_target_and_memory_lengths():
    tgt = torch.randn(3, 1, 16)
    mem = torch.randn(10, 1, 16)
    res = TransformerDecoderNode().execute(
        {"tensor": tgt, "memory": mem},
        {"d_model": 16, "nhead": 4, "num_layers": 1, "dim_feedforward": 32},
        context=_ctx(),
    )
    assert res["tensor"].shape == (3, 1, 16)


def test_multiple_layers_run_without_error():
    tgt = torch.randn(4, 1, 16)
    mem = torch.randn(4, 1, 16)
    res = TransformerDecoderNode().execute(
        {"tensor": tgt, "memory": mem},
        {"d_model": 16, "nhead": 4, "num_layers": 3, "dim_feedforward": 32},
        context=_ctx(),
    )
    assert res["tensor"].shape == (4, 1, 16)


def test_two_runs_on_the_same_input_give_the_same_output():
    """#690: the node is never trained, so dropout must not be active."""
    torch.manual_seed(0)
    tgt = torch.randn(5, 2, 16)
    mem = torch.randn(7, 2, 16)
    dec = TransformerDecoderNode()
    ctx = _ctx()
    p = {"d_model": 16, "nhead": 4, "num_layers": 2, "dim_feedforward": 32}
    first = dec.execute({"tensor": tgt, "memory": mem}, p, context=ctx)["tensor"]
    second = dec.execute({"tensor": tgt, "memory": mem}, p, context=ctx)["tensor"]
    assert torch.equal(first, second)
