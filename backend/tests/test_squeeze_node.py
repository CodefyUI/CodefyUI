"""Tests for SqueezeNode."""

from __future__ import annotations

import torch

from app.nodes.tensor_ops.squeeze_node import SqueezeNode


def _run(tensor, **params):
    return SqueezeNode().execute({"tensor": tensor}, params)


def test_node_metadata():
    assert SqueezeNode.NODE_NAME == "Squeeze"


def test_squeeze_all_size_one_dims():
    x = torch.zeros(1, 3, 1, 4, 1)
    res = _run(x, dim=-1)
    assert res["tensor"].shape == (3, 4)


def test_squeeze_specific_dim():
    x = torch.zeros(1, 3, 1, 4)
    res = _run(x, dim=0)
    assert res["tensor"].shape == (3, 1, 4)


def test_no_size_one_dims_no_change():
    x = torch.zeros(2, 3, 4)
    res = _run(x, dim=-1)
    assert res["tensor"].shape == (2, 3, 4)


def test_squeeze_dim_not_size_one_leaves_unchanged():
    x = torch.zeros(2, 3, 4)
    res = _run(x, dim=1)
    # Squeezing a dim that isn't 1 is a no-op
    assert res["tensor"].shape == (2, 3, 4)
