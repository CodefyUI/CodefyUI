"""Tests for VisualizeNode."""

from __future__ import annotations

import base64

import pytest
import torch

from app.nodes.utility.visualize_node import VisualizeNode


def _run(data, **params):
    return VisualizeNode().execute({"data": data}, params)


def test_node_metadata():
    assert VisualizeNode.NODE_NAME == "Visualize"
    assert VisualizeNode.CATEGORY == "Utility"


def test_line_plot_returns_base64_png():
    data = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0])
    res = _run(data, plot_type="line")
    assert "image" in res
    # Should be valid base64
    decoded = base64.b64decode(res["image"])
    # PNG magic header is 89 50 4E 47
    assert decoded[:4] == b"\x89PNG"


def test_histogram_returns_image():
    data = torch.randn(1000)
    res = _run(data, plot_type="histogram", title="Test")
    decoded = base64.b64decode(res["image"])
    assert decoded[:4] == b"\x89PNG"


def test_heatmap_2d():
    data = torch.randn(10, 10)
    res = _run(data, plot_type="heatmap")
    decoded = base64.b64decode(res["image"])
    assert decoded[:4] == b"\x89PNG"


def test_heatmap_pads_1d_to_square():
    data = torch.arange(50.0)
    res = _run(data, plot_type="heatmap")
    decoded = base64.b64decode(res["image"])
    assert decoded[:4] == b"\x89PNG"


def test_image_chw():
    data = torch.rand(3, 16, 16)
    res = _run(data, plot_type="image")
    decoded = base64.b64decode(res["image"])
    assert decoded[:4] == b"\x89PNG"


def test_image_grayscale_2d():
    data = torch.rand(8, 8)
    res = _run(data, plot_type="image")
    decoded = base64.b64decode(res["image"])
    assert decoded[:4] == b"\x89PNG"


def test_unknown_plot_type_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        _run(torch.zeros(3), plot_type="bogus")


def test_handles_numpy_array():
    import numpy as np
    data = np.array([1.0, 2.0, 3.0])
    res = _run(data, plot_type="line")
    assert "image" in res


def test_handles_python_list():
    res = _run([1, 2, 3, 4, 5], plot_type="line")
    assert "image" in res
