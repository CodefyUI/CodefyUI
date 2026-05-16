"""Tests for AddNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.tensor_ops.add_node import AddNode


def _run(a, b, **params):
    return AddNode().execute({"tensor_a": a, "tensor_b": b}, params)


def test_node_metadata():
    assert AddNode.NODE_NAME == "Add"
    assert AddNode.CATEGORY == "Tensor Operations"
    in_names = [p.name for p in AddNode.define_inputs()]
    out_names = [p.name for p in AddNode.define_outputs()]
    param_names = [p.name for p in AddNode.define_params()]
    assert in_names == ["tensor_a", "tensor_b"]
    assert out_names == ["tensor"]
    assert "alpha" in param_names


def test_elementwise_sum():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    res = _run(a, b)
    assert torch.allclose(res["tensor"], torch.tensor([5.0, 7.0, 9.0]))


def test_alpha_multiplier_scales_b():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([10.0, 20.0])
    res = _run(a, b, alpha=0.5)
    assert torch.allclose(res["tensor"], torch.tensor([6.0, 12.0]))


def test_alpha_default_is_one():
    a = torch.zeros(3)
    b = torch.tensor([1.0, 2.0, 3.0])
    res = _run(a, b)
    assert torch.allclose(res["tensor"], b)


def test_broadcasting_supported():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([10.0, 20.0])
    res = _run(a, b)
    assert torch.allclose(res["tensor"], torch.tensor([[11.0, 22.0], [13.0, 24.0]]))


def test_negative_alpha_subtracts():
    a = torch.tensor([5.0])
    b = torch.tensor([3.0])
    res = _run(a, b, alpha=-1.0)
    assert torch.allclose(res["tensor"], torch.tensor([2.0]))


def test_preserves_float_dtype():
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    b = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)
    res = _run(a, b)
    assert res["tensor"].dtype == torch.float64


def test_integer_tensors_with_integer_alpha():
    a = torch.tensor([1, 2, 3], dtype=torch.int32)
    b = torch.tensor([4, 5, 6], dtype=torch.int32)
    res = _run(a, b, alpha=1)
    assert res["tensor"].dtype == torch.int32
    assert torch.equal(res["tensor"], torch.tensor([5, 7, 9], dtype=torch.int32))
