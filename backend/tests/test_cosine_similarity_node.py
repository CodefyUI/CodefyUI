"""Tests for CosineSimilarityNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.llm.cosine_similarity_node import CosineSimilarityNode
from app.nodes.llm.word_vector_node import WordVectorNode


def _run(queries, keys, key_labels=None, **params):
    p = {"top_k": 3, "exclude_self_words": ""}
    p.update(params)
    inputs = {"queries": queries, "keys": keys}
    if key_labels is not None:
        inputs["key_labels"] = key_labels
    return CosineSimilarityNode().execute(inputs, p)


def test_node_metadata():
    assert CosineSimilarityNode.NODE_NAME == "CosineSimilarity"
    assert CosineSimilarityNode.CATEGORY == "LLM"


def test_identical_vectors_score_one():
    v = torch.tensor([[1.0, 2.0, 3.0]])
    res = _run(v, v)
    assert pytest.approx(res["similarity"][0, 0].item(), abs=1e-6) == 1.0


def test_orthogonal_vectors_score_zero():
    q = torch.tensor([[1.0, 0.0]])
    k = torch.tensor([[0.0, 1.0]])
    res = _run(q, k)
    assert abs(res["similarity"][0, 0].item()) < 1e-6


def test_antiparallel_vectors_score_minus_one():
    q = torch.tensor([[1.0, 0.0]])
    k = torch.tensor([[-1.0, 0.0]])
    res = _run(q, k)
    assert pytest.approx(res["similarity"][0, 0].item(), abs=1e-6) == -1.0


def test_normalisation_is_handled_internally():
    q = torch.tensor([[3.0, 4.0]])  # not unit length
    k = torch.tensor([[6.0, 8.0]])  # parallel to q, also not unit
    res = _run(q, k)
    assert pytest.approx(res["similarity"][0, 0].item(), abs=1e-6) == 1.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="dimension mismatch"):
        _run(torch.zeros(1, 3), torch.zeros(1, 4))


def test_top_k_returns_correct_indices_and_labels():
    keys = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    labels = ["a", "b", "ab", "c"]
    queries = torch.tensor([[1.0, 0.0, 0.0]])
    res = _run(queries, keys, key_labels=labels, top_k=2)
    # Closest to "a" should be a (cos=1.0), then ab (cos≈0.707).
    assert res["top_k_indices"] == [[0, 2]]
    assert res["top_k_labels"] == [["a", "ab"]]


def test_top_k_empty_labels_when_not_provided():
    res = _run(torch.zeros(1, 2), torch.eye(3, 2), top_k=2)
    assert res["top_k_labels"] == [[]]


def test_top_k_clamped_to_K():
    res = _run(torch.zeros(1, 2), torch.eye(2, 2), top_k=10)
    assert len(res["top_k_indices"][0]) == 2


def test_exclude_self_words_filters_topk():
    keys = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    labels = ["self", "near", "far"]
    res = _run(torch.tensor([[1.0, 0.0]]), keys, key_labels=labels, top_k=1, exclude_self_words="self")
    assert res["top_k_indices"] == [[1]]
    assert res["top_k_labels"] == [["near"]]


def test_one_d_query_is_treated_as_single_row():
    res = _run(torch.tensor([1.0, 0.0]), torch.eye(2, 2))
    assert res["similarity"].shape == (1, 2)


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires both"):
        CosineSimilarityNode().execute({"queries": torch.zeros(1, 2)}, {"top_k": 5})


def test_king_minus_man_plus_woman_finds_queen_in_demo():
    """Canonical analogy end-to-end against the demo-16d vocabulary."""
    wv = WordVectorNode()
    # Build the analogy vector via the existing WordVector + simple arithmetic.
    triple = wv.execute(
        {"tokens": ["king", "man", "woman"]},
        {"backend": "demo-16d", "words": "", "normalize": False, "keep_oov": False},
    )["embeddings"]
    analogy = (triple[0] - triple[1] + triple[2]).unsqueeze(0)

    # Project against the full demo vocab.
    from app.nodes.llm._demo_vectors import DEMO_VECTORS

    vocab = sorted(DEMO_VECTORS.keys())
    full = wv.execute(
        {"tokens": vocab},
        {"backend": "demo-16d", "words": "", "normalize": False, "keep_oov": False},
    )
    res = CosineSimilarityNode().execute(
        {
            "queries": analogy,
            "keys": full["embeddings"],
            "key_labels": full["labels"],
        },
        {"top_k": 1, "exclude_self_words": "king,man,woman"},
    )
    assert res["top_k_labels"][0] == ["queen"]
