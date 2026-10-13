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


# ── line: a [K, N] tensor draws K series ──────────────────────────────────


@pytest.fixture
def saved_figures(monkeypatch):
    """The line count and legend text of every figure the node saves."""
    from matplotlib.figure import Figure

    seen: list[dict] = []
    original = Figure.savefig

    def spy(self, *args, **kwargs):
        ax = self.axes[0]
        legend = ax.get_legend()
        seen.append({
            "lines": [line.get_ydata().tolist() for line in ax.get_lines()],
            "legend": [t.get_text() for t in legend.get_texts()] if legend else [],
        })
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", spy)
    return seen


def test_line_draws_one_series_per_row_with_a_legend(saved_figures):
    data = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
    res = _run(data, plot_type="line")
    assert base64.b64decode(res["image"])[:4] == b"\x89PNG"
    assert "__log__" not in res
    (fig,) = saved_figures
    assert fig["legend"] == ["series 0", "series 1", "series 2"]
    assert fig["lines"] == data.tolist()


def test_line_of_a_1d_tensor_is_unchanged(saved_figures):
    _run(torch.tensor([1.0, 2.0, 3.0]), plot_type="line")
    (fig,) = saved_figures
    assert fig["lines"] == [[1.0, 2.0, 3.0]] and fig["legend"] == []


def test_line_of_an_n_by_1_tensor_draws_one_series(saved_figures):
    _run(torch.tensor([[1.0], [2.0], [3.0], [4.0]]), plot_type="line")
    (fig,) = saved_figures
    assert fig["lines"] == [[1.0, 2.0, 3.0, 4.0]] and fig["legend"] == []


def test_line_with_too_many_rows_flattens_and_says_so(saved_figures):
    data = torch.arange(13 * 2, dtype=torch.float32).reshape(13, 2)
    res = _run(data, plot_type="line")
    (fig,) = saved_figures
    assert fig["lines"] == [data.flatten().tolist()] and fig["legend"] == []
    assert "13" in res["__log__"] and "flattened" in res["__log__"]
