"""Tests for MaxPool2dNode."""

from __future__ import annotations

import torch

from app.nodes.cnn.maxpool2d_node import MaxPool2dNode


def _run(tensor, **params):
    return MaxPool2dNode().execute({"tensor": tensor}, params)


def test_node_metadata():
    assert MaxPool2dNode.NODE_NAME == "MaxPool2d"
    assert MaxPool2dNode.CATEGORY == "CNN"


def test_default_2x2_halves_spatial_dims():
    x = torch.randn(1, 3, 8, 8)
    res = _run(x)
    assert res["tensor"].shape == (1, 3, 4, 4)


def test_picks_max_value_in_window():
    # Single-channel 2x2: max of [[1,2],[3,4]] is 4
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    res = _run(x, kernel_size=2, stride=2)
    assert res["tensor"].shape == (1, 1, 1, 1)
    assert res["tensor"].item() == 4.0


def test_stride_one_preserves_more_size():
    x = torch.randn(1, 1, 4, 4)
    res = _run(x, kernel_size=2, stride=1)
    assert res["tensor"].shape == (1, 1, 3, 3)


def test_3x3_kernel():
    x = torch.randn(2, 4, 6, 6)
    res = _run(x, kernel_size=3, stride=3)
    assert res["tensor"].shape == (2, 4, 2, 2)


def test_channels_preserved():
    x = torch.randn(2, 8, 4, 4)
    res = _run(x)
    assert res["tensor"].shape[1] == 8
