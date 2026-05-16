"""Tests for ReshapeNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.utility.reshape_node import ReshapeNode


def _run(tensor, **params):
    return ReshapeNode().execute({"tensor": tensor}, params)


def test_node_metadata():
    assert ReshapeNode.NODE_NAME == "Reshape"
    assert ReshapeNode.CATEGORY == "Utility"


def test_basic_reshape():
    x = torch.arange(12)
    res = _run(x, shape="3,4")
    assert res["tensor"].shape == (3, 4)


def test_negative_one_infers_dim():
    x = torch.arange(20)
    res = _run(x, shape="-1,4")
    assert res["tensor"].shape == (5, 4)


def test_default_flatten_minus_one_784():
    # Default shape "-1,784" works on 784-divisible tensors
    x = torch.zeros(2, 1, 28, 28)  # 2*1*28*28 = 1568 = 2*784
    res = _run(x, shape="-1,784")
    assert res["tensor"].shape == (2, 784)


def test_handles_whitespace():
    x = torch.arange(6)
    res = _run(x, shape=" 2 , 3 ")
    assert res["tensor"].shape == (2, 3)


def test_4d_reshape_for_image():
    x = torch.arange(48)
    res = _run(x, shape="1,3,4,4")
    assert res["tensor"].shape == (1, 3, 4, 4)


def test_missing_tensor_input_raises():
    with pytest.raises(ValueError, match="requires"):
        ReshapeNode().execute({}, {"shape": "1"})


def test_preserves_data():
    x = torch.tensor([1, 2, 3, 4, 5, 6])
    res = _run(x, shape="2,3")
    assert torch.equal(res["tensor"], torch.tensor([[1, 2, 3], [4, 5, 6]]))
