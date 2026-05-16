"""Tests for ReduceNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.dataflow.reduce_node import ReduceNode


def _run(items, **params):
    return ReduceNode().execute({"items": items}, params)


def test_node_metadata():
    assert ReduceNode.NODE_NAME == "Reduce"
    assert ReduceNode.CATEGORY == "Data Flow"


def test_sum_scalars():
    res = _run([1.0, 2.0, 3.0], operation="sum")
    assert torch.isclose(res["result"], torch.tensor(6.0))
    assert res["count"] == 3.0


def test_mean_scalars():
    res = _run([2.0, 4.0, 6.0], operation="mean")
    assert torch.isclose(res["result"], torch.tensor(4.0))


def test_min_max():
    res_min = _run([5.0, 2.0, 8.0], operation="min")
    res_max = _run([5.0, 2.0, 8.0], operation="max")
    assert torch.isclose(res_min["result"], torch.tensor(2.0))
    assert torch.isclose(res_max["result"], torch.tensor(8.0))


def test_first_returns_first_element():
    res = _run(["a", "b", "c"], operation="first")
    assert res["result"] == "a"


def test_last_returns_last_element():
    res = _run(["a", "b", "c"], operation="last")
    assert res["result"] == "c"


def test_concat_tensors():
    a = torch.zeros(2, 3)
    b = torch.ones(3, 3)
    res = _run([a, b], operation="concat", dim=0)
    assert res["result"].shape == (5, 3)


def test_stack_tensors():
    items = [torch.zeros(3), torch.ones(3)]
    res = _run(items, operation="stack", dim=0)
    assert res["result"].shape == (2, 3)


def test_empty_list_raises():
    with pytest.raises(ValueError, match="empty"):
        _run([], operation="sum")


def test_non_list_input_raises():
    with pytest.raises(ValueError, match="list"):
        ReduceNode().execute({"items": "not a list"}, {"operation": "sum"})


def test_mixed_string_for_numeric_op_raises():
    with pytest.raises(ValueError, match="numeric"):
        _run(["not", "numbers"], operation="sum")
