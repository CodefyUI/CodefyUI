"""Tests for SplitNode."""

from __future__ import annotations

import torch

from app.nodes.tensor_ops.split_node import SplitNode


def _run(tensor, **params):
    return SplitNode().execute({"tensor": tensor}, params)


def test_node_metadata():
    assert SplitNode.NODE_NAME == "Split"
    out_names = [p.name for p in SplitNode.define_outputs()]
    assert out_names == ["chunk_0", "chunk_1"]


def test_split_into_two_along_dim_zero():
    x = torch.arange(8).reshape(4, 2)
    res = _run(x, chunks=2, dim=0)
    assert "chunk_0" in res
    assert "chunk_1" in res
    assert res["chunk_0"].shape == (2, 2)
    assert res["chunk_1"].shape == (2, 2)
    assert torch.equal(torch.cat([res["chunk_0"], res["chunk_1"]], dim=0), x)


def test_split_along_dim_one():
    x = torch.arange(8).reshape(2, 4)
    res = _run(x, chunks=2, dim=1)
    assert res["chunk_0"].shape == (2, 2)
    assert res["chunk_1"].shape == (2, 2)


def test_split_into_four_chunks_returns_all():
    x = torch.arange(8)
    res = _run(x, chunks=4, dim=0)
    # Output is whatever torch.chunk returns; only first two are declared outputs
    # but all chunks are returned in the dict
    assert "chunk_0" in res
    assert "chunk_1" in res
    assert "chunk_2" in res
    assert "chunk_3" in res
    assert res["chunk_0"].shape == (2,)


def test_default_two_chunks_along_dim_zero():
    x = torch.zeros(6)
    res = SplitNode().execute({"tensor": x}, {})
    assert res["chunk_0"].shape == (3,)
    assert res["chunk_1"].shape == (3,)
