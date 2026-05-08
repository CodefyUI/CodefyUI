"""Tests for DecisionTreeClassifierNode (sklearn wrapper)."""

from __future__ import annotations

import pytest
import torch

from app.nodes.classical.decision_tree_classifier_node import (
    DecisionTreeClassifierNode,
)


def _run(x_train, y_train, x_query, **params):
    p = {"max_depth": 5, "criterion": "gini", "random_state": 42}
    p.update(params)
    return DecisionTreeClassifierNode().execute(
        {"x_train": x_train, "y_train": y_train, "x_query": x_query}, p
    )


def test_node_metadata():
    assert DecisionTreeClassifierNode.NODE_NAME == "DecisionTreeClassifier"
    assert DecisionTreeClassifierNode.CATEGORY == "Classical"
    out_names = [p.name for p in DecisionTreeClassifierNode.define_outputs()]
    assert set(out_names) >= {
        "predictions",
        "feature_importances",
        "tree_text",
        "classes",
    }


def test_recovers_decision_rule():
    """A tree should perfectly fit a small dataset with a clear split."""
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = ["a", "a", "a", "b", "b", "b"]
    res = _run(x, y, x, max_depth=3)
    correct = sum(1 for i, p in enumerate(res["predictions"]) if p == y[i])
    assert correct == 6


def test_feature_importances_shape():
    x = torch.randn(40, 5, generator=torch.Generator().manual_seed(0))
    y = ["a"] * 20 + ["b"] * 20
    res = _run(x, y, x)
    assert res["feature_importances"].shape == (5,)
    # Importances should sum to 1 (or be all-zero for an empty tree).
    s = float(res["feature_importances"].sum())
    assert abs(s - 1.0) < 1e-5 or s == 0.0


def test_dominant_feature_dominates_importance():
    """A tree should put nearly all importance on the only useful feature."""
    torch.manual_seed(0)
    n = 60
    # Feature 0 perfectly determines the label; feature 1 is noise.
    x0 = torch.cat([torch.full((n // 2,), -1.0), torch.full((n // 2,), 1.0)]).unsqueeze(1)
    x1 = torch.randn(n, 1)
    x = torch.cat([x0, x1], dim=1)
    y = ["a"] * (n // 2) + ["b"] * (n // 2)
    res = _run(x, y, x, max_depth=3)
    importances = res["feature_importances"].tolist()
    assert importances[0] > importances[1]


def test_tree_text_is_string():
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    y = ["a", "a", "b", "b"]
    res = _run(x, y, x)
    assert isinstance(res["tree_text"], str)
    # sklearn's export_text always emits a header
    assert "feature" in res["tree_text"].lower() or "class" in res["tree_text"].lower()


def test_label_count_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        _run(torch.zeros(5, 2), ["a"] * 3, torch.zeros(1, 2))


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        DecisionTreeClassifierNode().execute(
            {"x_train": torch.zeros(5, 2)},
            {"max_depth": 5, "criterion": "gini", "random_state": 42},
        )


def test_unlimited_depth_via_zero():
    """max_depth=0 should map to None (no limit)."""
    x = torch.randn(20, 2, generator=torch.Generator().manual_seed(0))
    y = ["a"] * 10 + ["b"] * 10
    # No crash and produces the right shape
    res = _run(x, y, x, max_depth=0)
    assert len(res["predictions"]) == 20
