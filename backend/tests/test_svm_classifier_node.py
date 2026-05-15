"""Tests for SVMClassifierNode (sklearn wrapper)."""

from __future__ import annotations

import pytest
import torch

from app.nodes.classical.svm_classifier_node import SVMClassifierNode


def _run(x_train, y_train, x_query, **params):
    p = {"C": 1.0, "kernel": "rbf", "gamma": "scale"}
    p.update(params)
    return SVMClassifierNode().execute(
        {"x_train": x_train, "y_train": y_train, "x_query": x_query}, p
    )


def test_node_metadata():
    assert SVMClassifierNode.NODE_NAME == "SVMClassifier"
    assert SVMClassifierNode.CATEGORY == "Classical"
    out_names = [p.name for p in SVMClassifierNode.define_outputs()]
    assert set(out_names) >= {"predictions", "support_vectors", "classes"}


def test_separates_clean_clusters():
    torch.manual_seed(0)
    a = torch.randn(40, 2) + torch.tensor([3.0, 3.0])
    b = torch.randn(40, 2) + torch.tensor([-3.0, -3.0])
    x = torch.cat([a, b], dim=0)
    y = ["a"] * 40 + ["b"] * 40
    res = _run(x, y, x, kernel="linear")
    correct = sum(1 for i, p in enumerate(res["predictions"]) if p == y[i])
    assert correct >= 75  # near-perfect on a clearly-linearly-separable problem


def test_rbf_handles_nonlinear():
    """RBF should handle xor-like data better than linear."""
    torch.manual_seed(0)
    # Mixed quadrants → not linearly separable
    pts = []
    labels = []
    for sign_a in [-1, 1]:
        for sign_b in [-1, 1]:
            cluster = torch.randn(20, 2) * 0.3 + torch.tensor([sign_a * 2.0, sign_b * 2.0])
            pts.append(cluster)
            labels += ["a" if sign_a == sign_b else "b"] * 20
    x = torch.cat(pts, dim=0)
    res = _run(x, labels, x, kernel="rbf")
    correct = sum(1 for i, p in enumerate(res["predictions"]) if p == labels[i])
    assert correct >= 70  # RBF should crush this


def test_classes_returned():
    x = torch.randn(20, 2, generator=torch.Generator().manual_seed(0))
    y = ["a"] * 10 + ["b"] * 10
    res = _run(x, y, x)
    assert res["classes"] == ["a", "b"]


def test_support_vectors_subset_of_train():
    """SVs must be points from the training set."""
    torch.manual_seed(0)
    x = torch.randn(30, 2)
    y = ["a"] * 15 + ["b"] * 15
    res = _run(x, y, x, kernel="linear")
    sv = res["support_vectors"]
    # Each SV must equal a row in x_train
    for sv_row in sv:
        diffs = (x - sv_row).abs().sum(dim=1)
        assert diffs.min().item() < 1e-4


def test_label_count_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        _run(torch.zeros(5, 2), ["a"] * 3, torch.zeros(1, 2))


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        SVMClassifierNode().execute(
            {"x_train": torch.zeros(5, 2)},
            {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
        )
