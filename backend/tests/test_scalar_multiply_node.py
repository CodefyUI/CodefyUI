"""Tests for ScalarMultiplyNode."""

from __future__ import annotations

import torch

from app.nodes.tensor_ops.scalar_multiply_node import ScalarMultiplyNode


def _run(tensor, **params):
    return ScalarMultiplyNode().execute({"tensor": tensor}, params)


def test_node_metadata():
    assert ScalarMultiplyNode.NODE_NAME == "ScalarMultiply"
    assert ScalarMultiplyNode.CATEGORY == "Tensor Operations"
    in_names = [p.name for p in ScalarMultiplyNode.define_inputs()]
    out_names = [p.name for p in ScalarMultiplyNode.define_outputs()]
    param_names = [p.name for p in ScalarMultiplyNode.define_params()]
    assert in_names == ["tensor"]
    assert out_names == ["tensor"]
    assert param_names == ["scalar"]


def test_scalar_param_default_is_one():
    params = {p.name: p for p in ScalarMultiplyNode.define_params()}
    assert params["scalar"].default == 1.0


def test_multiply_by_two():
    x = torch.tensor([1.0, 2.0, 3.0])
    res = _run(x, scalar=2.0)
    assert torch.equal(res["tensor"], torch.tensor([2.0, 4.0, 6.0]))


def test_multiply_by_zero_zeroes_tensor():
    x = torch.tensor([1.0, 2.0, 3.0])
    res = _run(x, scalar=0.0)
    assert torch.equal(res["tensor"], torch.zeros(3))


def test_multiply_by_negative():
    x = torch.tensor([1.0, -2.0, 3.0])
    res = _run(x, scalar=-1.0)
    assert torch.equal(res["tensor"], torch.tensor([-1.0, 2.0, -3.0]))


def test_multiply_by_fractional_scalar():
    x = torch.ones(3)
    res = _run(x, scalar=0.299)
    assert torch.allclose(res["tensor"], torch.full((3,), 0.299))


def test_preserves_shape_for_multidim_tensor():
    x = torch.randn(1, 3, 4, 5)
    res = _run(x, scalar=2.0)
    assert res["tensor"].shape == x.shape
    assert torch.allclose(res["tensor"], x * 2.0)


def test_default_scalar_is_identity():
    x = torch.tensor([1.0, 2.0, 3.0])
    res = ScalarMultiplyNode().execute({"tensor": x}, {})
    assert torch.equal(res["tensor"], x)


def test_string_scalar_param_gets_coerced():
    x = torch.tensor([1.0, 2.0])
    res = _run(x, scalar="2.5")
    assert torch.allclose(res["tensor"], torch.tensor([2.5, 5.0]))


def test_preserves_dtype():
    x = torch.tensor([1.0, 2.0], dtype=torch.float64)
    res = _run(x, scalar=2.0)
    assert res["tensor"].dtype == torch.float64
