"""Tests for LinearRegressionNode (sklearn wrapper)."""

from __future__ import annotations

import pytest
import torch

from app.nodes.classical.linear_regression_node import LinearRegressionNode


def _run(x_train, y_train, x_query, **params):
    p = {"fit_intercept": True}
    p.update(params)
    return LinearRegressionNode().execute(
        {"x_train": x_train, "y_train": y_train, "x_query": x_query}, p
    )


def test_node_metadata():
    assert LinearRegressionNode.NODE_NAME == "LinearRegression"
    assert LinearRegressionNode.CATEGORY == "Classical"
    out_names = [p.name for p in LinearRegressionNode.define_outputs()]
    assert "predictions" in out_names
    assert "coef" in out_names
    assert "intercept" in out_names


def test_recovers_known_line():
    """y = 2 x + 5 should be recovered exactly (closed-form OLS, no noise)."""
    x = torch.arange(10, dtype=torch.float32).view(-1, 1)
    y = 2 * x.squeeze() + 5
    res = _run(x, y, x)
    assert torch.allclose(res["coef"].view(-1), torch.tensor([2.0]), atol=1e-4)
    assert abs(float(res["intercept"]) - 5.0) < 1e-4
    assert torch.allclose(res["predictions"], y, atol=1e-4)


def test_no_intercept_passes_through_origin():
    x = torch.arange(10, dtype=torch.float32).view(-1, 1)
    y = 3 * x.squeeze()
    res = _run(x, y, x, fit_intercept=False)
    assert abs(float(res["intercept"])) < 1e-6
    assert torch.allclose(res["coef"].view(-1), torch.tensor([3.0]), atol=1e-4)


def test_multivariate():
    """y = 1*x1 + 2*x2 + 3."""
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
    y = x[:, 0] + 2.0 * x[:, 1] + 3.0
    res = _run(x, y, x)
    assert torch.allclose(res["coef"].view(-1), torch.tensor([1.0, 2.0]), atol=1e-4)
    assert abs(float(res["intercept"]) - 3.0) < 1e-4


def test_prediction_shape_matches_query():
    x_train = torch.arange(10, dtype=torch.float32).view(-1, 1)
    y_train = 2 * x_train.squeeze()
    x_query = torch.tensor([[100.0], [200.0], [300.0]])
    res = _run(x_train, y_train, x_query)
    assert res["predictions"].shape == (3,)
    assert torch.allclose(res["predictions"], torch.tensor([200.0, 400.0, 600.0]), atol=1e-3)


def test_label_count_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        _run(torch.zeros(5, 2), torch.zeros(3), torch.zeros(1, 2))


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        LinearRegressionNode().execute(
            {"x_train": torch.zeros(5, 2)},
            {"fit_intercept": True},
        )
