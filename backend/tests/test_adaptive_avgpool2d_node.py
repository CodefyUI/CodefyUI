"""Tests for AdaptiveAvgPool2dNode."""

from __future__ import annotations

import torch

from app.nodes.cnn.adaptive_avgpool2d_node import AdaptiveAvgPool2dNode


def _run(tensor, **params):
    return AdaptiveAvgPool2dNode().execute({"tensor": tensor}, params)


def test_node_metadata():
    assert AdaptiveAvgPool2dNode.NODE_NAME == "AdaptiveAvgPool2d"


def test_default_output_1x1():
    x = torch.randn(1, 3, 8, 8)
    res = _run(x)
    assert res["tensor"].shape == (1, 3, 1, 1)


def test_custom_output_size():
    x = torch.randn(2, 4, 16, 16)
    res = _run(x, output_height=4, output_width=4)
    assert res["tensor"].shape == (2, 4, 4, 4)


def test_global_pooling_is_mean():
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    res = _run(x, output_height=1, output_width=1)
    assert torch.isclose(res["tensor"], torch.tensor(2.5))


def test_works_for_different_input_sizes():
    """Adaptive: output size is fixed regardless of input size."""
    res1 = _run(torch.randn(1, 3, 100, 100), output_height=7, output_width=7)
    res2 = _run(torch.randn(1, 3, 20, 20), output_height=7, output_width=7)
    assert res1["tensor"].shape == res2["tensor"].shape == (1, 3, 7, 7)


def test_non_square_output():
    x = torch.randn(1, 3, 8, 16)
    res = _run(x, output_height=2, output_width=4)
    assert res["tensor"].shape == (1, 3, 2, 4)
