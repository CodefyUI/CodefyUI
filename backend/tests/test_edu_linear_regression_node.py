"""Tests for EduLinearRegressionNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.classical.edu_linear_regression_node import EduLinearRegressionNode


def _run(x_train, y_train, x_query, **params):
    p = {"method": "closed_form", "lr": 0.01, "epochs": 100, "regularization": 0.0}
    p.update(params)
    return EduLinearRegressionNode().execute(
        {"x_train": x_train, "y_train": y_train, "x_query": x_query}, p
    )


def test_node_metadata():
    assert EduLinearRegressionNode.NODE_NAME == "EduLinearRegression"
    assert EduLinearRegressionNode.CATEGORY == "Classical"
    out_names = [p.name for p in EduLinearRegressionNode.define_outputs()]
    assert out_names == ["predictions", "weights", "bias"]


def test_closed_form_recovers_perfect_line():
    """y = 2x + 1 exactly — closed form should recover w=2, b=1."""
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
    res = _run(x, y, x, method="closed_form")
    assert torch.allclose(res["weights"].squeeze(), torch.tensor([2.0]), atol=1e-4)
    assert abs(res["bias"].item() - 1.0) < 1e-4
    # Predictions on training data should match perfectly.
    assert torch.allclose(res["predictions"].squeeze(), y, atol=1e-3)


def test_closed_form_multivariate():
    """y = 1*x1 + 2*x2 + 3 — recover w = [1, 2], b = 3."""
    torch.manual_seed(0)
    x = torch.randn(100, 2)
    y = x[:, 0] + 2 * x[:, 1] + 3
    res = _run(x, y, x, method="closed_form")
    assert torch.allclose(res["weights"], torch.tensor([1.0, 2.0]), atol=1e-3)
    assert abs(res["bias"].item() - 3.0) < 1e-3


def test_gradient_descent_converges_to_closed_form():
    """GD with enough epochs should approximate closed-form solution."""
    torch.manual_seed(0)
    x = torch.randn(200, 3)
    y = x @ torch.tensor([1.0, -1.0, 0.5]) + 2.0
    cf = _run(x, y, x, method="closed_form")
    gd = _run(x, y, x, method="gradient_descent", lr=0.05, epochs=2000)
    # Should agree to within ~5% — GD won't be exact but should be close.
    assert torch.allclose(gd["weights"], cf["weights"], atol=0.05)
    assert abs(gd["bias"].item() - cf["bias"].item()) < 0.05


def test_predictions_on_unseen_query():
    """y = x → query at x=10 should predict ~10."""
    x = torch.tensor([[0.0], [1.0], [2.0]])
    y = torch.tensor([0.0, 1.0, 2.0])
    res = _run(x, y, torch.tensor([[10.0]]), method="closed_form")
    assert abs(res["predictions"].item() - 10.0) < 1e-3


def test_regularization_shrinks_weights():
    """Strong L2 regularization should pull weights toward zero."""
    torch.manual_seed(0)
    x = torch.randn(50, 4)
    y = x @ torch.tensor([10.0, -10.0, 5.0, 5.0]) + 0.0
    no_reg = _run(x, y, x, method="closed_form", regularization=0.0)
    high_reg = _run(x, y, x, method="closed_form", regularization=100.0)
    # Regularised weights should have smaller L2 norm.
    assert high_reg["weights"].norm().item() < no_reg["weights"].norm().item()


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="method"):
        _run(torch.zeros(5, 2), torch.zeros(5), torch.zeros(1, 2), method="unknown")


def test_label_count_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        _run(torch.zeros(5, 2), torch.zeros(3), torch.zeros(1, 2))


def test_accepts_label_list():
    """Labels as a Python list of floats."""
    res = _run(
        torch.tensor([[0.0], [1.0], [2.0]]),
        [0.0, 1.0, 2.0],
        torch.tensor([[5.0]]),
    )
    assert abs(res["predictions"].item() - 5.0) < 1e-3


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        EduLinearRegressionNode().execute(
            {"x_train": torch.zeros(5, 2)},
            {"method": "closed_form", "lr": 0.01, "epochs": 100, "regularization": 0.0},
        )
