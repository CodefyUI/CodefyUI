"""Tests for ColumnSelectorNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.data.column_selector_node import ColumnSelectorNode


def _run(tensor, *, columns=None, **params):
    p = {"indices": "0", "names": ""}
    p.update(params)
    inputs: dict = {"tensor": tensor}
    if columns is not None:
        inputs["columns"] = columns
    return ColumnSelectorNode().execute(inputs, p)


def test_node_metadata():
    assert ColumnSelectorNode.NODE_NAME == "ColumnSelector"
    assert ColumnSelectorNode.CATEGORY == "Data"


def test_select_by_indices():
    x = torch.arange(12, dtype=torch.float32).view(3, 4)
    res = _run(x, indices="0,2")
    expected = torch.tensor([[0.0, 2.0], [4.0, 6.0], [8.0, 10.0]])
    assert torch.equal(res["tensor"], expected)


def test_single_index():
    x = torch.arange(12, dtype=torch.float32).view(3, 4)
    res = _run(x, indices="3")
    assert res["tensor"].shape == (3, 1)
    assert torch.equal(res["tensor"][:, 0], torch.tensor([3.0, 7.0, 11.0]))


def test_select_by_names_with_columns_input():
    x = torch.arange(12, dtype=torch.float32).view(3, 4)
    res = _run(x, columns=["a", "b", "c", "d"], names="a,c")
    expected = torch.tensor([[0.0, 2.0], [4.0, 6.0], [8.0, 10.0]])
    assert torch.equal(res["tensor"], expected)


def test_names_wins_over_indices_when_both_set():
    x = torch.arange(12, dtype=torch.float32).view(3, 4)
    # indices says "select col 0", names says "select col c (=2)" — names wins.
    res = _run(x, columns=["a", "b", "c", "d"], indices="0", names="c")
    assert torch.equal(res["tensor"][:, 0], torch.tensor([2.0, 6.0, 10.0]))


def test_unknown_index_raises():
    with pytest.raises(ValueError, match="indices"):
        _run(torch.zeros(3, 4), indices="0,99")


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="names"):
        _run(torch.zeros(3, 4), columns=["a", "b"], names="nonexistent")


def test_names_without_columns_input_raises():
    with pytest.raises(ValueError, match="columns"):
        _run(torch.zeros(3, 4), names="a")


def test_empty_indices_returns_zero_cols():
    x = torch.zeros(3, 4)
    res = _run(x, indices="")
    assert res["tensor"].shape == (3, 0)


def test_preserves_dtype():
    x = torch.zeros(2, 3, dtype=torch.float64)
    res = _run(x, indices="0,1")
    assert res["tensor"].dtype == torch.float64


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        ColumnSelectorNode().execute({}, {"indices": "0", "names": ""})
