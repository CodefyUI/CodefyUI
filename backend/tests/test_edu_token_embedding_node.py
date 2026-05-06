"""Tests for EduTokenEmbeddingNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.llm.edu_token_embedding_node import EduTokenEmbeddingNode


def _run(*, tokens=None, token_ids=None, **params):
    p = {"embed_dim": 8, "vocab_size": 32, "seed": 42, "mode": "hash"}
    p.update(params)
    inputs: dict = {}
    if tokens is not None:
        inputs["tokens"] = tokens
    if token_ids is not None:
        inputs["token_ids"] = token_ids
    return EduTokenEmbeddingNode().execute(inputs, p)


def test_node_metadata():
    assert EduTokenEmbeddingNode.NODE_NAME == "EduTokenEmbedding"
    assert EduTokenEmbeddingNode.CATEGORY == "LLM"
    out_names = [p.name for p in EduTokenEmbeddingNode.define_outputs()]
    assert out_names == ["embeddings", "vocab"]


def test_hash_mode_output_shape():
    res = _run(tokens=["the", "cat", "sat"], embed_dim=8, mode="hash")
    assert res["embeddings"].shape == (3, 8)
    assert res["embeddings"].dtype == torch.float32


def test_hash_mode_is_deterministic_per_token():
    """Same token maps to same vector across runs (stable hash)."""
    a = _run(tokens=["cat", "dog", "cat"], embed_dim=8, seed=42, mode="hash")
    # First and third positions are both "cat" — should be identical rows.
    assert torch.allclose(a["embeddings"][0], a["embeddings"][2])


def test_hash_mode_changes_with_seed():
    a = _run(tokens=["cat"], seed=1, mode="hash")
    b = _run(tokens=["cat"], seed=2, mode="hash")
    assert not torch.allclose(a["embeddings"], b["embeddings"])


def test_ordinal_mode_assigns_ids_by_first_appearance():
    """First unique token = id 0, second new = 1, etc."""
    res = _run(tokens=["cat", "dog", "cat", "fish"], mode="ordinal")
    # Same tokens at positions 0 and 2 should have identical embeddings.
    assert torch.allclose(res["embeddings"][0], res["embeddings"][2])
    # Different tokens differ.
    assert not torch.allclose(res["embeddings"][0], res["embeddings"][1])
    assert not torch.allclose(res["embeddings"][0], res["embeddings"][3])


def test_ordinal_mode_vocab_lists_unique_tokens():
    res = _run(tokens=["cat", "dog", "cat", "fish"], mode="ordinal")
    assert res["vocab"] == ["cat", "dog", "fish"]


def test_token_ids_input_works():
    """Numeric IDs path — useful when chained from Tokenizer.token_ids."""
    res = _run(token_ids=[5, 7, 5, 9], embed_dim=8, mode="hash")
    assert res["embeddings"].shape == (4, 8)
    # Same id at positions 0 and 2 → identical embeddings.
    assert torch.allclose(res["embeddings"][0], res["embeddings"][2])


def test_token_ids_modulo_vocab_size():
    """IDs >= vocab_size wrap around (we don't crash on big tiktoken IDs)."""
    res = _run(token_ids=[100000, 100032], vocab_size=32, embed_dim=4, seed=42, mode="hash")
    # 100000 % 32 == 0, 100032 % 32 == 0 → identical rows.
    assert torch.allclose(res["embeddings"][0], res["embeddings"][1])


def test_empty_input_returns_empty_tensor():
    res = _run(tokens=[], mode="hash")
    assert res["embeddings"].shape == (0, 8)
    assert res["vocab"] == []


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown"):
        _run(tokens=["a"], mode="not-a-mode")


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        EduTokenEmbeddingNode().execute({}, {"embed_dim": 8, "vocab_size": 32, "seed": 42, "mode": "hash"})
