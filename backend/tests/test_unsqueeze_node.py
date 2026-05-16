"""Tests for UnsqueezeNode."""

from __future__ import annotations

import torch

from app.nodes.tensor_ops.unsqueeze_node import UnsqueezeNode


def _run(tensor, **params):
    return UnsqueezeNode().execute({"tensor": tensor}, params)


def test_node_metadata():
    assert UnsqueezeNode.NODE_NAME == "Unsqueeze"


def test_add_dim_at_position_zero():
    x = torch.zeros(3, 4)
    res = _run(x, dim=0)
    assert res["tensor"].shape == (1, 3, 4)


def test_add_dim_at_position_one():
    x = torch.zeros(3, 4)
    res = _run(x, dim=1)
    assert res["tensor"].shape == (3, 1, 4)


def test_add_dim_at_negative_position():
    x = torch.zeros(3, 4)
    res = _run(x, dim=-1)
    assert res["tensor"].shape == (3, 4, 1)


def test_preserves_values():
    x = torch.tensor([1.0, 2.0, 3.0])
    res = _run(x, dim=0)
    assert torch.equal(res["tensor"].squeeze(), x)


def test_unsqueeze_scalar():
    x = torch.tensor(5.0)
    res = _run(x, dim=0)
    assert res["tensor"].shape == (1,)
