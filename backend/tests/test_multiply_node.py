"""Tests for MultiplyNode."""

from __future__ import annotations

import torch

from app.nodes.tensor_ops.multiply_node import MultiplyNode


def _run(a, b):
    return MultiplyNode().execute({"tensor_a": a, "tensor_b": b}, {})


def test_node_metadata():
    assert MultiplyNode.NODE_NAME == "Multiply"
    assert MultiplyNode.CATEGORY == "Tensor Operations"
    in_names = [p.name for p in MultiplyNode.define_inputs()]
    out_names = [p.name for p in MultiplyNode.define_outputs()]
    assert in_names == ["tensor_a", "tensor_b"]
    assert out_names == ["tensor"]


def test_elementwise_product():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    res = _run(a, b)
    assert torch.allclose(res["tensor"], torch.tensor([4.0, 10.0, 18.0]))


def test_multiply_by_zero():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.zeros(3)
    res = _run(a, b)
    assert torch.allclose(res["tensor"], torch.zeros(3))


def test_multiply_by_one_is_identity():
    a = torch.tensor([1.5, 2.5, 3.5])
    res = _run(a, torch.ones(3))
    assert torch.allclose(res["tensor"], a)


def test_broadcasting_with_scalar():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor(2.0)
    res = _run(a, b)
    assert torch.allclose(res["tensor"], torch.tensor([[2.0, 4.0], [6.0, 8.0]]))


def test_broadcasting_row_vector_and_matrix():
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = torch.tensor([10.0, 20.0, 30.0])
    res = _run(a, b)
    expected = torch.tensor([[10.0, 40.0, 90.0], [40.0, 100.0, 180.0]])
    assert torch.allclose(res["tensor"], expected)
