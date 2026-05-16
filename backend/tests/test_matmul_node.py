"""Tests for MatMulNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.tensor_ops.matmul_node import MatMulNode


def _run(a, b):
    return MatMulNode().execute({"tensor_a": a, "tensor_b": b}, {})


def test_node_metadata():
    assert MatMulNode.NODE_NAME == "MatMul"
    assert MatMulNode.CATEGORY == "Tensor Operations"


def test_2d_matrix_product():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    res = _run(a, b)
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
    assert torch.allclose(res["tensor"], expected)


def test_matrix_vector_product():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([5.0, 6.0])
    res = _run(a, b)
    assert torch.allclose(res["tensor"], torch.tensor([17.0, 39.0]))


def test_identity_matrix_product():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    eye = torch.eye(2)
    res = _run(a, eye)
    assert torch.allclose(res["tensor"], a)


def test_batched_matmul():
    a = torch.randn(4, 3, 5)
    b = torch.randn(4, 5, 2)
    res = _run(a, b)
    assert res["tensor"].shape == (4, 3, 2)


def test_incompatible_shapes_raise():
    a = torch.randn(3, 4)
    b = torch.randn(5, 6)
    with pytest.raises(RuntimeError):
        _run(a, b)


def test_output_shape_for_attention_style():
    # (B, H, S, D) @ (B, H, D, S) -> (B, H, S, S)
    q = torch.randn(2, 4, 8, 16)
    k_t = torch.randn(2, 4, 16, 8)
    res = _run(q, k_t)
    assert res["tensor"].shape == (2, 4, 8, 8)
