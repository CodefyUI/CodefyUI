"""Tests for EduLogisticRegressionNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.classical.edu_logistic_regression_node import EduLogisticRegressionNode


def _run(x_train, y_train, x_query, **params):
    p = {"lr": 0.1, "epochs": 500, "regularization": 0.0}
    p.update(params)
    return EduLogisticRegressionNode().execute(
        {"x_train": x_train, "y_train": y_train, "x_query": x_query}, p
    )


def test_node_metadata():
    assert EduLogisticRegressionNode.NODE_NAME == "EduLogisticRegression"
    assert EduLogisticRegressionNode.CATEGORY == "Classical"
    out_names = [p.name for p in EduLogisticRegressionNode.define_outputs()]
    assert out_names == ["predictions", "probabilities", "weights", "bias", "classes"]


def test_binary_separable_data():
    """Two well-separated clusters should be perfectly classified."""
    torch.manual_seed(0)
    cluster_a = torch.randn(50, 2) + torch.tensor([5.0, 5.0])
    cluster_b = torch.randn(50, 2) + torch.tensor([-5.0, -5.0])
    x = torch.cat([cluster_a, cluster_b], dim=0)
    y = ["a"] * 50 + ["b"] * 50
    res = _run(x, y, x, epochs=300)
    correct = sum(1 for i in range(100) if res["predictions"][i] == y[i])
    assert correct >= 95  # at least 95% accurate on training


def test_multiclass_separable_data():
    """Three Gaussian clusters → 3-class classification."""
    torch.manual_seed(0)
    centers = torch.tensor([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
    x_list = []
    y_list = []
    for cls, c in enumerate(centers):
        x_list.append(torch.randn(40, 2) * 0.3 + c)
        y_list += [str(cls)] * 40
    x = torch.cat(x_list)
    res = _run(x, y_list, x, epochs=500, lr=0.05)
    correct = sum(1 for i in range(120) if res["predictions"][i] == y_list[i])
    assert correct >= 110  # ≥ ~92%


def test_probabilities_sum_to_one():
    torch.manual_seed(0)
    x = torch.randn(20, 4)
    y = ["a"] * 10 + ["b"] * 10
    res = _run(x, y, x, epochs=50)
    sums = res["probabilities"].sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_classes_output_tracks_unique_labels():
    torch.manual_seed(0)
    x = torch.randn(30, 3)
    y = ["cat", "dog", "fish"] * 10
    res = _run(x, y, x[:5], epochs=20)
    assert sorted(res["classes"]) == ["cat", "dog", "fish"]
    # Predictions are from the same set of classes.
    for p in res["predictions"]:
        assert p in {"cat", "dog", "fish"}


def test_query_can_differ_from_training():
    torch.manual_seed(0)
    x_train = torch.randn(50, 2)
    y_train = ["a"] * 25 + ["b"] * 25
    x_query = torch.randn(7, 2)
    res = _run(x_train, y_train, x_query, epochs=20)
    assert len(res["predictions"]) == 7
    assert res["probabilities"].shape == (7, 2)


def test_regularization_shrinks_weights():
    """L2 reg with small-enough lr to stay numerically stable should shrink weights."""
    torch.manual_seed(0)
    cluster_a = torch.randn(30, 2) + torch.tensor([5.0, 5.0])
    cluster_b = torch.randn(30, 2) + torch.tensor([-5.0, -5.0])
    x = torch.cat([cluster_a, cluster_b])
    y = ["a"] * 30 + ["b"] * 30
    # Use small lr so high reg doesn't explode the GD update step.
    no_reg = _run(x, y, x, epochs=200, lr=0.05, regularization=0.0)
    mid_reg = _run(x, y, x, epochs=200, lr=0.05, regularization=0.5)
    assert mid_reg["weights"].norm().item() < no_reg["weights"].norm().item()


def test_label_count_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        _run(torch.zeros(5, 2), ["a"] * 3, torch.zeros(1, 2))


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        EduLogisticRegressionNode().execute(
            {"x_train": torch.zeros(5, 2)},
            {"lr": 0.1, "epochs": 10, "regularization": 0.0},
        )


def test_single_class_raises():
    """All same label → can't fit a classifier."""
    with pytest.raises(ValueError, match="class"):
        _run(torch.zeros(5, 2), ["a"] * 5, torch.zeros(1, 2))
