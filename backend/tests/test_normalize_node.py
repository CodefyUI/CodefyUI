"""Tests for NormalizeNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.data.normalize_node import NormalizeNode


def _run(tensor, **params):
    p = {"mode": "zscore", "axis": 0}
    p.update(params)
    return NormalizeNode().execute({"tensor": tensor}, p)


def test_node_metadata():
    assert NormalizeNode.NODE_NAME == "Normalize"
    assert NormalizeNode.CATEGORY == "Data"
    out_names = [p.name for p in NormalizeNode.define_outputs()]
    assert out_names == ["tensor", "stats"]


def test_zscore_per_column():
    """Each column should have mean 0, std 1 after zscore (axis=0)."""
    x = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    res = _run(x, mode="zscore", axis=0)
    means = res["tensor"].mean(dim=0)
    stds = res["tensor"].std(dim=0, unbiased=False)
    assert torch.allclose(means, torch.zeros(2), atol=1e-6)
    assert torch.allclose(stds, torch.ones(2), atol=1e-5)


def test_minmax_per_column_range_zero_one():
    x = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    res = _run(x, mode="minmax", axis=0)
    assert res["tensor"].min(dim=0).values.allclose(torch.zeros(2))
    assert res["tensor"].max(dim=0).values.allclose(torch.ones(2))


def test_unit_norm_each_row_has_norm_one():
    x = torch.tensor([[3.0, 4.0], [1.0, 0.0], [0.0, 5.0]])
    res = _run(x, mode="unit_norm", axis=1)
    norms = res["tensor"].norm(dim=1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-6)


def test_zscore_axis_1():
    """axis=1 means normalize across each row (per-sample)."""
    x = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    res = _run(x, mode="zscore", axis=1)
    means = res["tensor"].mean(dim=1)
    assert torch.allclose(means, torch.zeros(2), atol=1e-6)


def test_stats_returned_for_zscore():
    x = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    res = _run(x, mode="zscore", axis=0)
    assert "mean" in res["stats"]
    assert "std" in res["stats"]


def test_stats_returned_for_minmax():
    x = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
    res = _run(x, mode="minmax", axis=0)
    assert "min" in res["stats"]
    assert "max" in res["stats"]


def test_zero_variance_column_handled():
    """A constant column produces std=0; should not divide by zero."""
    x = torch.tensor([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    res = _run(x, mode="zscore", axis=0)
    # The constant column should become zero (mean subtracted, no divide).
    assert torch.allclose(res["tensor"][:, 0], torch.zeros(3), atol=1e-6)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        _run(torch.zeros(3, 2), mode="not-a-mode")


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        NormalizeNode().execute({}, {"mode": "zscore", "axis": 0})


def test_dtype_is_float32():
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    res = _run(x)
    assert res["tensor"].dtype == torch.float32
