"""Tests for MLPClassifierNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.classical.mlp_classifier_node import MLPClassifierNode


def _toy_two_class():
    """Two well-separated 2D clusters — trivially classifiable."""
    g = torch.Generator().manual_seed(0)
    n = 40
    a = torch.randn(n, 2, generator=g) + torch.tensor([2.0, 2.0])
    b = torch.randn(n, 2, generator=g) + torch.tensor([-2.0, -2.0])
    x = torch.cat([a, b], dim=0)
    y = ["0"] * n + ["1"] * n
    return x, y


def test_node_metadata():
    assert MLPClassifierNode.NODE_NAME == "MLPClassifier"
    assert MLPClassifierNode.CATEGORY == "Classical"
    out_names = [p.name for p in MLPClassifierNode.define_outputs()]
    assert out_names == ["predictions", "probabilities", "classes", "train_loss"]


def test_easy_two_class_high_accuracy():
    x, y = _toy_two_class()
    out = MLPClassifierNode().execute(
        {"x_train": x, "y_train": y, "x_query": x},
        {"hidden_sizes": "8", "max_iter": 200, "seed": 0},
    )
    correct = sum(1 for p, t in zip(out["predictions"], y) if p == t)
    assert correct / len(y) > 0.9


def test_identity_activation_collapses_to_linear():
    """activation='identity' should still train but is a single linear model — included as a sanity-check that the param surface lets the textbook show this."""
    x, y = _toy_two_class()
    out = MLPClassifierNode().execute(
        {"x_train": x, "y_train": y, "x_query": x},
        {"hidden_sizes": "8,8", "activation": "identity", "max_iter": 100, "seed": 0},
    )
    # Should still classify two linearly-separable clusters correctly.
    correct = sum(1 for p, t in zip(out["predictions"], y) if p == t)
    assert correct / len(y) > 0.85


def test_invalid_hidden_sizes_raises():
    x, y = _toy_two_class()
    with pytest.raises(ValueError, match="at least one layer"):
        MLPClassifierNode().execute(
            {"x_train": x, "y_train": y, "x_query": x},
            {"hidden_sizes": "", "max_iter": 10},
        )
