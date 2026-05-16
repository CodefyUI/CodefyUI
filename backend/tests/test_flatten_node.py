"""Tests for FlattenNode."""

from __future__ import annotations

import torch

from app.nodes.utility.flatten_node import FlattenNode


def _run(tensor, **params):
    return FlattenNode().execute({"tensor": tensor}, params)


def test_node_metadata():
    assert FlattenNode.NODE_NAME == "Flatten"


def test_default_keeps_batch_dim():
    # Default start_dim=1 keeps batch dim, flattens the rest
    x = torch.randn(4, 3, 8, 8)
    res = _run(x)
    assert res["tensor"].shape == (4, 192)


def test_flatten_all_with_start_dim_zero():
    x = torch.randn(2, 3, 4)
    res = _run(x, start_dim=0)
    assert res["tensor"].shape == (24,)


def test_flatten_preserves_values_in_order():
    x = torch.arange(24).reshape(2, 3, 4)
    res = _run(x, start_dim=1)
    assert res["tensor"].shape == (2, 12)
    # First batch element should be [0..11]
    assert torch.equal(res["tensor"][0], torch.arange(12))


def test_flatten_2d_unchanged():
    x = torch.randn(4, 16)
    res = _run(x)
    assert res["tensor"].shape == (4, 16)
