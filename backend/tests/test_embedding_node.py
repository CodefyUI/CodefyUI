"""Tests for EmbeddingNode."""

from __future__ import annotations

import torch

from app.core.execution_context import ExecutionContext
from app.core.node_state_store import NodeStateStore
from app.nodes.utility.embedding_node import EmbeddingNode


def _ctx(verbose=False):
    return ExecutionContext(
        graph_id="g",
        weights_persistent=True,
        node_state_store=NodeStateStore(),
        current_node_id="emb",
        verbose=verbose,
    )


def test_node_metadata():
    assert EmbeddingNode.NODE_NAME == "Embedding"
    assert EmbeddingNode.CATEGORY == "Utility"


def test_lookup_returns_correct_shape():
    indices = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    res = EmbeddingNode().execute(
        {"tensor": indices},
        {"num_embeddings": 100, "embedding_dim": 16, "padding_idx": -1},
        context=_ctx(),
    )
    assert res["tensor"].shape == (2, 3, 16)


def test_padding_idx_zeroed_in_table():
    """When padding_idx is set, that row of the embedding table is zero."""
    indices = torch.tensor([[0, 1, 2]], dtype=torch.long)
    node = EmbeddingNode()
    ctx = _ctx()
    mod = node.get_or_build_module(ctx, {"num_embeddings": 10, "embedding_dim": 4, "padding_idx": 0})
    assert torch.all(mod.weight[0] == 0)


def test_negative_padding_idx_means_no_padding():
    node = EmbeddingNode()
    ctx = _ctx()
    mod = node.get_or_build_module(ctx, {"num_embeddings": 10, "embedding_dim": 4, "padding_idx": -1})
    assert mod.padding_idx is None


def test_verbose_records_steps():
    indices = torch.tensor([1, 2, 3], dtype=torch.long)
    res = EmbeddingNode().execute(
        {"tensor": indices},
        {"num_embeddings": 10, "embedding_dim": 4, "padding_idx": -1},
        context=_ctx(verbose=True),
    )
    assert "__steps__" in res
    step_names = [s.name for s in res["__steps__"]]
    assert "indices" in step_names
    assert "lookup" in step_names


def test_normalise_for_hash_collapses_negative_padding():
    node = EmbeddingNode()
    norm_a = node._normalise_for_hash({"padding_idx": -1, "x": 1})
    norm_b = node._normalise_for_hash({"padding_idx": -5, "x": 1})
    assert norm_a["padding_idx"] is None
    assert norm_b["padding_idx"] is None
    assert norm_a == norm_b


def test_normalise_for_hash_preserves_positive_padding():
    node = EmbeddingNode()
    norm = node._normalise_for_hash({"padding_idx": 0, "x": 1})
    assert norm["padding_idx"] == 0
