"""Tests for StackNode."""

from __future__ import annotations

import torch

from app.nodes.tensor_ops.stack_node import StackNode


def _run(a, b, **params):
    return StackNode().execute({"tensor_a": a, "tensor_b": b}, params)


def test_node_metadata():
    assert StackNode.NODE_NAME == "Stack"


def test_stack_creates_new_dim_zero():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    res = _run(a, b, dim=0)
    assert res["tensor"].shape == (2, 3)
    assert torch.equal(res["tensor"][0], a)
    assert torch.equal(res["tensor"][1], b)


def test_stack_dim_one():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    res = _run(a, b, dim=1)
    assert res["tensor"].shape == (2, 2)
    assert res["tensor"][0, 0] == 1.0
    assert res["tensor"][0, 1] == 3.0


def test_stack_2d_tensors():
    a = torch.zeros(3, 4)
    b = torch.ones(3, 4)
    res = _run(a, b, dim=0)
    assert res["tensor"].shape == (2, 3, 4)


def test_stack_default_dim_is_zero():
    a = torch.zeros(2)
    b = torch.ones(2)
    res = StackNode().execute({"tensor_a": a, "tensor_b": b}, {})
    assert res["tensor"].shape == (2, 2)
