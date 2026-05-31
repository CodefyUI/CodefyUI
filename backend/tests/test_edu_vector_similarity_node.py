"""Tests for EduVectorSimilarityNode (chapter pack I1-3)."""

from __future__ import annotations

import pytest
import torch

from cdui_plugins.foundations.nodes.edu_vector_similarity_node import (
    EduVectorSimilarityNode,
)


class _Ctx:
    """Minimal stand-in for ExecutionContext exposing only `verbose`."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose


def _run(query, keys, *, context=None, **params):
    p = {"metric": "cosine"}
    p.update(params)
    return EduVectorSimilarityNode().execute(
        {"query": query, "keys": keys}, p, context=context
    )


def test_node_metadata():
    assert EduVectorSimilarityNode.NODE_NAME == "Edu-VectorSimilarity"
    assert EduVectorSimilarityNode.CATEGORY == "Data"
    out_names = [p.name for p in EduVectorSimilarityNode.define_outputs()]
    assert out_names == ["similarity", "query_norms", "key_norms"]


def test_cosine_identical_unit_vectors_is_one():
    # A unit vector compared with itself has cosine similarity 1.
    q = torch.tensor([1.0, 0.0, 0.0])
    keys = torch.tensor([[1.0, 0.0, 0.0]])
    res = _run(q, keys, metric="cosine")
    assert res["similarity"][0, 0].item() == pytest.approx(1.0, abs=1e-6)


def test_cosine_orthogonal_vectors_is_zero():
    # Orthogonal vectors have zero cosine similarity.
    q = torch.tensor([1.0, 0.0])
    keys = torch.tensor([[0.0, 1.0]])
    res = _run(q, keys, metric="cosine")
    assert res["similarity"][0, 0].item() == pytest.approx(0.0, abs=1e-6)


def test_cosine_is_scale_invariant():
    # Cosine ignores magnitude: scaling either vector leaves it unchanged.
    q = torch.tensor([[3.0, 4.0]])  # ||q|| = 5
    keys = torch.tensor([[3.0, 4.0]])
    res = _run(q, keys, metric="cosine")
    assert res["similarity"][0, 0].item() == pytest.approx(1.0, abs=1e-6)


def test_dot_metric_returns_raw_inner_products():
    # Hand-computed small case:
    #   q0·k0 = 1*1 + 2*0 = 1 ; q0·k1 = 1*3 + 2*4 = 11
    #   q1·k0 = 0*1 + 1*0 = 0 ; q1·k1 = 0*3 + 1*4 = 4
    q = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
    keys = torch.tensor([[1.0, 0.0], [3.0, 4.0]])
    res = _run(q, keys, metric="dot")
    expected = torch.tensor([[1.0, 11.0], [0.0, 4.0]])
    assert torch.allclose(res["similarity"], expected, atol=1e-6)


def test_one_d_query_produces_1_by_m():
    q = torch.tensor([1.0, 0.0, 0.0])            # [D]
    keys = torch.randn(4, 3)                       # [M=4, D=3]
    res = _run(q, keys)
    assert tuple(res["similarity"].shape) == (1, 4)
    assert tuple(res["query_norms"].shape) == (1,)
    assert tuple(res["key_norms"].shape) == (4,)


def test_n_by_d_query_produces_n_by_m():
    q = torch.randn(2, 3)                           # [N=2, D=3]
    keys = torch.randn(5, 3)                        # [M=5, D=3]
    res = _run(q, keys)
    assert tuple(res["similarity"].shape) == (2, 5)
    assert tuple(res["query_norms"].shape) == (2,)
    assert tuple(res["key_norms"].shape) == (5,)


def test_norms_returned_for_dot_metric():
    # Even with the dot metric, the norms are surfaced for inspection.
    q = torch.tensor([[3.0, 4.0]])
    keys = torch.tensor([[0.0, 5.0]])
    res = _run(q, keys, metric="dot")
    assert res["query_norms"].item() == pytest.approx(5.0, abs=1e-6)
    assert res["key_norms"].item() == pytest.approx(5.0, abs=1e-6)


def test_dimension_mismatch_raises():
    q = torch.randn(2, 4)        # D=4
    keys = torch.randn(3, 5)     # D=5
    with pytest.raises(ValueError, match="match"):
        _run(q, keys)


def test_keys_must_be_2d():
    q = torch.randn(3)
    with pytest.raises(ValueError, match="keys"):
        _run(q, torch.randn(3))  # 1-D keys are invalid


def test_query_must_be_1d_or_2d():
    q = torch.randn(2, 2, 3)     # 3-D query is invalid
    keys = torch.randn(4, 3)
    with pytest.raises(ValueError, match="query"):
        _run(q, keys)


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        EduVectorSimilarityNode().execute({"query": torch.randn(3)}, {})


def test_verbose_records_steps():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    res = _run(q, keys, context=_Ctx(verbose=True))
    steps = res.get("__steps__")
    assert steps is not None and len(steps) >= 1
    names = [s.name for s in steps]
    assert names == ["dot_products", "norms", "similarity"]


def test_non_verbose_has_no_steps():
    q = torch.tensor([[1.0, 0.0]])
    keys = torch.tensor([[1.0, 0.0]])
    # No context → not verbose.
    res = _run(q, keys)
    assert "__steps__" not in res
    # Explicit context with verbose=False → still no steps.
    res2 = _run(q, keys, context=_Ctx(verbose=False))
    assert "__steps__" not in res2
