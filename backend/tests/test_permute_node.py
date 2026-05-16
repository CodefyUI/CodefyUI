"""Tests for PermuteNode."""

from __future__ import annotations

import torch

from app.nodes.tensor_ops.permute_node import PermuteNode


def _run(tensor, **params):
    return PermuteNode().execute({"tensor": tensor}, params)


def test_node_metadata():
    assert PermuteNode.NODE_NAME == "Permute"


def test_swap_last_two_dims():
    x = torch.arange(24).reshape(2, 3, 4)
    res = _run(x, dims="0,2,1")
    assert res["tensor"].shape == (2, 4, 3)


def test_transpose_2d():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    res = _run(x, dims="1,0")
    assert res["tensor"].shape == (3, 2)
    assert res["tensor"][0, 0] == 1.0
    assert res["tensor"][2, 1] == 6.0


def test_identity_permutation():
    x = torch.randn(2, 3, 4)
    res = _run(x, dims="0,1,2")
    assert torch.equal(res["tensor"], x)


def test_handles_whitespace_in_dims():
    x = torch.randn(2, 3)
    res = _run(x, dims=" 1 , 0 ")
    assert res["tensor"].shape == (3, 2)


def test_full_reverse_4d():
    x = torch.randn(2, 3, 4, 5)
    res = _run(x, dims="3,2,1,0")
    assert res["tensor"].shape == (5, 4, 3, 2)
