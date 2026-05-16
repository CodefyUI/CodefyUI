"""Tests for ConcatNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.utility.concat_node import ConcatNode


def _run(a, b, **params):
    return ConcatNode().execute({"tensor_a": a, "tensor_b": b}, params)


def test_node_metadata():
    assert ConcatNode.NODE_NAME == "Concat"
    assert ConcatNode.CATEGORY == "Utility"


def test_concat_along_dim_zero():
    a = torch.zeros(2, 3)
    b = torch.ones(3, 3)
    res = _run(a, b, dim=0)
    assert res["tensor"].shape == (5, 3)
    assert torch.all(res["tensor"][:2] == 0)
    assert torch.all(res["tensor"][2:] == 1)


def test_concat_along_dim_one():
    a = torch.zeros(3, 2)
    b = torch.ones(3, 5)
    res = _run(a, b, dim=1)
    assert res["tensor"].shape == (3, 7)


def test_default_dim_is_zero():
    a = torch.zeros(2, 3)
    b = torch.zeros(2, 3)
    res = ConcatNode().execute({"tensor_a": a, "tensor_b": b}, {})
    assert res["tensor"].shape == (4, 3)


def test_concat_4d_tensors():
    a = torch.zeros(1, 3, 4, 4)
    b = torch.zeros(2, 3, 4, 4)
    res = _run(a, b, dim=0)
    assert res["tensor"].shape == (3, 3, 4, 4)


def test_incompatible_other_dims_raises():
    a = torch.zeros(2, 3)
    b = torch.zeros(2, 4)
    with pytest.raises(RuntimeError):
        _run(a, b, dim=0)
