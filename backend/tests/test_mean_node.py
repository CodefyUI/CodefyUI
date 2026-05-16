"""Tests for MeanNode."""

from __future__ import annotations

import torch

from app.nodes.tensor_ops.mean_node import MeanNode


def _run(tensor, **params):
    return MeanNode().execute({"tensor": tensor}, params)


def test_node_metadata():
    assert MeanNode.NODE_NAME == "Mean"
    assert MeanNode.CATEGORY == "Tensor Operations"
    param_names = [p.name for p in MeanNode.define_params()]
    assert "dim" in param_names
    assert "keepdim" in param_names


def test_mean_default_last_dim():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    res = _run(x)
    assert torch.allclose(res["tensor"], torch.tensor([2.0, 5.0]))


def test_mean_first_dim():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    res = _run(x, dim="0")
    assert torch.allclose(res["tensor"], torch.tensor([2.0, 3.0]))


def test_mean_multiple_dims():
    x = torch.ones(2, 3, 4)
    res = _run(x, dim="1,2")
    # Mean of all ones is 1.0; reduce over dims 1 and 2 leaves shape (2,)
    assert res["tensor"].shape == (2,)
    assert torch.allclose(res["tensor"], torch.ones(2))


def test_keepdim_true_preserves_rank():
    x = torch.randn(2, 3, 4)
    res = _run(x, dim="-1", keepdim=True)
    assert res["tensor"].shape == (2, 3, 1)


def test_keepdim_false_squeezes():
    x = torch.randn(2, 3, 4)
    res = _run(x, dim="-1", keepdim=False)
    assert res["tensor"].shape == (2, 3)


def test_mean_handles_whitespace_in_dim():
    x = torch.ones(2, 3, 4)
    res = _run(x, dim=" 1 , 2 ")
    assert res["tensor"].shape == (2,)
