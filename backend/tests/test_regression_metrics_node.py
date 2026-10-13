"""RegressionMetrics: R² over several outputs, and mismatched shapes.

The 2026-10-13 audit found that two targets on different scales (about 300
and about 3) with features carrying no signal scored R² = 0.94: both inputs
were flattened, so SS_tot was taken around the mean of all values together
and the gap between the two columns counted as explained variance.
sklearn's ``r2_score`` gives -0.0123 (per column [0.0238, -0.0485]).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.nodes.classical.regression_metrics_node import RegressionMetricsNode


def _run(predictions, targets):
    return RegressionMetricsNode().execute(
        {"predictions": predictions, "targets": targets}, {})


def _two_scale_problem():
    rng = np.random.default_rng(0)
    targets = np.c_[300 + 50 * rng.normal(size=50), 3 + rng.normal(size=50)]
    # Predicting each column's mean plus noise: no real signal.
    predictions = np.c_[300 + 5 * rng.normal(size=50), 3 + 0.1 * rng.normal(size=50)]
    return predictions, targets


def test_multi_output_r2_is_the_mean_of_the_per_column_r2():
    predictions, targets = _two_scale_problem()
    out = _run(torch.from_numpy(predictions), torch.from_numpy(targets))
    assert out["r2"] == pytest.approx(r2_score(targets, predictions))
    assert out["r2"] < 0.1  # pooled, it was close to 1
    assert out["mse"] == pytest.approx(mean_squared_error(targets, predictions))
    assert out["mae"] == pytest.approx(mean_absolute_error(targets, predictions))


def test_a_nested_list_of_targets_is_read_as_rows():
    predictions, targets = _two_scale_problem()
    out = _run(torch.from_numpy(predictions), targets.tolist())
    assert out["r2"] == pytest.approx(r2_score(targets, predictions))


def test_same_count_different_shape_is_refused():
    predictions, targets = _two_scale_problem()
    with pytest.raises(ValueError, match=r"RegressionMetrics: `predictions` has shape \[50, 2\]"):
        _run(torch.from_numpy(predictions), list(targets.T.reshape(-1)))


@pytest.mark.parametrize("pred_shape,target_kind", [
    ("flat", "list"), ("column", "list"), ("flat", "column"), ("column", "flat"),
])
def test_single_output_is_unchanged(pred_shape, target_kind):
    rng = np.random.default_rng(1)
    y = rng.normal(size=40)
    y_hat = y + 0.3 * rng.normal(size=40)
    predictions = torch.from_numpy(y_hat)
    if pred_shape == "column":
        predictions = predictions.view(-1, 1)
    if target_kind == "list":
        targets = list(y)
    elif target_kind == "column":
        targets = torch.from_numpy(y).view(-1, 1)
    else:
        targets = torch.from_numpy(y)
    out = _run(predictions, targets)
    assert out["r2"] == pytest.approx(r2_score(y, y_hat))
    assert out["mse"] == pytest.approx(mean_squared_error(y, y_hat))


def test_a_constant_target_column_gives_nan():
    targets = torch.tensor([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
    out = _run(targets + 0.1, targets)
    assert np.isnan(out["r2"])


def test_length_mismatch_still_names_both_counts():
    with pytest.raises(ValueError, match="3 predictions vs 2 targets"):
        _run(torch.tensor([1.0, 2.0, 3.0]), [1.0, 2.0])
