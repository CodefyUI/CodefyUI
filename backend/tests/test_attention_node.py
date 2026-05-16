"""Tests for MultiHeadAttentionNode."""

from __future__ import annotations

import torch

from app.core.execution_context import ExecutionContext
from app.core.node_state_store import NodeStateStore
from app.nodes.transformer.attention_node import MultiHeadAttentionNode


def _ctx(verbose=False):
    return ExecutionContext(
        graph_id="g",
        weights_persistent=True,
        node_state_store=NodeStateStore(),
        current_node_id="mha",
        verbose=verbose,
    )


def test_node_metadata():
    assert MultiHeadAttentionNode.NODE_NAME == "MultiHeadAttention"
    assert MultiHeadAttentionNode.CATEGORY == "Transformer"
    in_names = [p.name for p in MultiHeadAttentionNode.define_inputs()]
    out_names = [p.name for p in MultiHeadAttentionNode.define_outputs()]
    assert in_names == ["query", "key", "value"]
    assert out_names == ["output", "weights"]


def test_self_attention_output_shape():
    # nn.MultiheadAttention default uses (seq, batch, embed)
    q = torch.randn(5, 2, 32)
    k = torch.randn(5, 2, 32)
    v = torch.randn(5, 2, 32)
    res = MultiHeadAttentionNode().execute(
        {"query": q, "key": k, "value": v},
        {"embed_dim": 32, "num_heads": 4},
        context=_ctx(),
    )
    assert res["output"].shape == (5, 2, 32)


def test_attention_weights_sum_to_one():
    q = torch.randn(5, 1, 16)
    k = torch.randn(5, 1, 16)
    v = torch.randn(5, 1, 16)
    res = MultiHeadAttentionNode().execute(
        {"query": q, "key": k, "value": v},
        {"embed_dim": 16, "num_heads": 4},
        context=_ctx(),
    )
    # nn.MultiheadAttention returns avg_attn_weights of shape (batch, target_seq, source_seq)
    weights = res["weights"]
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-5)


def test_verbose_mode_records_steps():
    q = torch.randn(4, 1, 16)
    k = torch.randn(4, 1, 16)
    v = torch.randn(4, 1, 16)
    res = MultiHeadAttentionNode().execute(
        {"query": q, "key": k, "value": v},
        {"embed_dim": 16, "num_heads": 4},
        context=_ctx(verbose=True),
    )
    assert "__steps__" in res
    step_names = [s.name for s in res["__steps__"]]
    assert "inputs_qkv" in step_names
    assert "scaled_scores" in step_names
    assert "softmax_weights" in step_names


def test_cross_attention_with_different_sequences():
    q = torch.randn(3, 1, 16)
    k = torch.randn(7, 1, 16)
    v = torch.randn(7, 1, 16)
    res = MultiHeadAttentionNode().execute(
        {"query": q, "key": k, "value": v},
        {"embed_dim": 16, "num_heads": 4},
        context=_ctx(),
    )
    # Output sequence length matches query sequence length
    assert res["output"].shape == (3, 1, 16)
