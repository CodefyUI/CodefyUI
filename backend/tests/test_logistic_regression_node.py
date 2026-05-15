"""Tests for LogisticRegressionNode (sklearn wrapper)."""

from __future__ import annotations

import pytest
import torch

from app.nodes.classical.logistic_regression_node import LogisticRegressionNode


def _run(x_train, y_train, x_query, **params):
    p = {"C": 1.0, "max_iter": 200, "penalty": "l2"}
    p.update(params)
    return LogisticRegressionNode().execute(
        {"x_train": x_train, "y_train": y_train, "x_query": x_query}, p
    )


def test_node_metadata():
    assert LogisticRegressionNode.NODE_NAME == "LogisticRegression"
    assert LogisticRegressionNode.CATEGORY == "Classical"
    out_names = [p.name for p in LogisticRegressionNode.define_outputs()]
    assert set(out_names) >= {"predictions", "probabilities", "classes", "coef"}


def test_separates_two_clusters():
    """Two well-separated clusters → near-perfect classification."""
    torch.manual_seed(0)
    a = torch.randn(50, 2) + torch.tensor([5.0, 5.0])
    b = torch.randn(50, 2) + torch.tensor([-5.0, -5.0])
    x = torch.cat([a, b], dim=0)
    y = ["a"] * 50 + ["b"] * 50
    res = _run(x, y, x)
    correct = sum(1 for i, p in enumerate(res["predictions"]) if p == y[i])
    assert correct >= 95


def test_probabilities_sum_to_one():
    torch.manual_seed(1)
    x = torch.randn(40, 3)
    y = ["a"] * 20 + ["b"] * 20
    res = _run(x, y, x)
    sums = res["probabilities"].sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_classes_match_columns():
    torch.manual_seed(2)
    x = torch.randn(30, 2)
    y = ["a"] * 10 + ["b"] * 10 + ["c"] * 10
    res = _run(x, y, x)
    assert res["classes"] == ["a", "b", "c"]
    assert res["probabilities"].shape == (30, 3)


def test_works_with_integer_labels():
    x = torch.randn(20, 2, generator=torch.Generator().manual_seed(0))
    y = torch.tensor([0] * 10 + [1] * 10)
    res = _run(x, y, x)
    # Predictions get stringified for consistency
    assert all(isinstance(p, str) for p in res["predictions"])


def test_label_count_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        _run(torch.zeros(5, 2), ["a"] * 3, torch.zeros(1, 2))


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        LogisticRegressionNode().execute(
            {"x_train": torch.zeros(5, 2)},
            {"C": 1.0, "max_iter": 200, "penalty": "l2"},
        )


def test_single_class_raises():
    with pytest.raises(ValueError, match="class"):
        _run(torch.randn(10, 2), ["a"] * 10, torch.randn(2, 2))
