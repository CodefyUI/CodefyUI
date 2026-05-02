"""Tests for WordVectorNode."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.nodes.llm.word_vector_node import WordVectorNode, _load_backend
from app.nodes.llm._demo_vectors import DIM as DEMO_DIM


def _run(words=None, **params):
    p = {
        "backend": "demo-16d",
        "words": "king queen man woman",
        "normalize": False,
        "keep_oov": False,
    }
    p.update(params)
    inputs = {"tokens": words} if words is not None else {}
    return WordVectorNode().execute(inputs, p)


def test_node_metadata():
    assert WordVectorNode.NODE_NAME == "WordVector"
    assert WordVectorNode.CATEGORY == "LLM"
    assert [p.name for p in WordVectorNode.define_outputs()] == ["embeddings", "labels"]


def test_demo_backend_returns_correct_shape():
    res = _run(["king", "queen", "man", "woman"])
    assert res["embeddings"].shape == (4, DEMO_DIM)
    assert res["labels"] == ["king", "queen", "man", "woman"]
    assert res["embeddings"].dtype == torch.float32


def test_oov_words_are_dropped_by_default():
    res = _run(["king", "asdfqwerty", "queen"])
    assert res["labels"] == ["king", "queen"]
    assert res["embeddings"].shape == (2, DEMO_DIM)


def test_keep_oov_emits_zero_rows():
    res = _run(["king", "asdfqwerty", "queen"], keep_oov=True)
    assert res["labels"] == ["king", "asdfqwerty", "queen"]
    assert res["embeddings"].shape == (3, DEMO_DIM)
    assert torch.all(res["embeddings"][1] == 0)


def test_words_param_used_when_no_input_connected():
    res = WordVectorNode().execute(
        {},
        {
            "backend": "demo-16d",
            "words": "cat, dog, fish",
            "normalize": False,
            "keep_oov": False,
        },
    )
    assert res["labels"] == ["cat", "dog", "fish"]


def test_case_insensitive_lookup():
    res = _run(["KING", "Queen", "MAN"])
    assert res["labels"] == ["king", "queen", "man"]


def test_empty_input_returns_empty_tensor():
    res = _run([])
    assert res["embeddings"].shape == (0, DEMO_DIM)
    assert res["labels"] == []


def test_king_minus_man_plus_woman_is_close_to_queen():
    """Canonical demo: in the demo-16d basis, the analogy holds exactly."""
    res = _run(["king", "queen", "man", "woman"])
    e = res["embeddings"]
    king, queen, man, woman = e[0], e[1], e[2], e[3]
    analogy = king - man + woman
    diff = torch.norm(analogy - queen).item()
    # Vectors are sparse hand-built tuples; the analogy is exact (zero diff).
    assert diff < 1e-5


def test_normalize_makes_unit_rows():
    res = _run(["king", "queen", "cat"], normalize=True)
    norms = torch.norm(res["embeddings"], dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_normalize_leaves_zero_rows_alone():
    res = _run(["king", "asdfqwerty"], keep_oov=True, normalize=True)
    # Row 0 (king) is unit; row 1 (zero vector) stays zero.
    norms = torch.norm(res["embeddings"], dim=1)
    assert abs(norms[0].item() - 1.0) < 1e-6
    assert norms[1].item() == 0.0


def test_glove_backend_raises_friendly_error():
    _load_backend.cache_clear()
    with pytest.raises(NotImplementedError, match="not yet published"):
        _run(["king"], backend="glove-50d")


def test_unknown_backend_raises():
    _load_backend.cache_clear()
    with pytest.raises(ValueError, match="Unknown WordVector backend"):
        _run(["king"], backend="not-a-real-backend")
