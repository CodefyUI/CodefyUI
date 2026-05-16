"""Tests for AvgPool2dNode."""

from __future__ import annotations

import torch

from app.nodes.cnn.avgpool2d_node import AvgPool2dNode


def _run(tensor, **params):
    return AvgPool2dNode().execute({"tensor": tensor}, params)


def test_node_metadata():
    assert AvgPool2dNode.NODE_NAME == "AvgPool2d"


def test_default_halves_spatial_dims():
    x = torch.randn(1, 3, 8, 8)
    res = _run(x)
    assert res["tensor"].shape == (1, 3, 4, 4)


def test_averages_values_in_window():
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    res = _run(x, kernel_size=2, stride=2)
    # Average of [1,2,3,4] = 2.5
    assert res["tensor"].item() == 2.5


def test_constant_input_yields_constant_output():
    x = torch.full((1, 1, 4, 4), 5.0)
    res = _run(x)
    assert torch.all(res["tensor"] == 5.0)


def test_padding_param_increases_output_size():
    x = torch.randn(1, 1, 4, 4)
    res = _run(x, kernel_size=2, stride=2, padding=1)
    # With padding=1, output size = (4 + 2*1 - 2)/2 + 1 = 3
    assert res["tensor"].shape == (1, 1, 3, 3)
