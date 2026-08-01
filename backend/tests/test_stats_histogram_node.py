"""Stats-Histogram: numpy parity, range modes, density, and the chart payload."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from cdui_plugins.stats.nodes.stats_histogram_node import StatsHistogramNode


def _run(tensor, params=None):
    return StatsHistogramNode().execute({"tensor": tensor}, params or {})


def test_node_metadata():
    assert StatsHistogramNode.NODE_NAME == "Stats-Histogram"
    outputs = StatsHistogramNode.define_outputs()
    assert [p.name for p in outputs] == [
        "table", "columns", "row_labels", "chart", "dropped",
    ]
    chart = next(p for p in outputs if p.name == "chart")
    assert chart.media == "chart", "the chart port must declare its media kind"


def test_counts_and_edges_match_numpy_histogram():
    data = np.random.default_rng(7).normal(size=500)
    result = _run(torch.tensor(data), {"bins": 12})

    counts, edges = np.histogram(data, bins=12)
    table = result["table"].numpy()
    assert table[:, 0] == pytest.approx(edges[:-1], rel=1e-5)
    assert table[:, 1] == pytest.approx(edges[1:], rel=1e-5)
    assert table[:, 2].tolist() == pytest.approx(counts.astype(float))
    assert result["columns"] == ["bin_start", "bin_end", "count"]


def test_density_matches_numpy_and_renames_the_column():
    data = np.random.default_rng(3).normal(size=400)
    result = _run(torch.tensor(data), {"bins": 8, "density": True})

    density, _edges = np.histogram(data, bins=8, density=True)
    assert result["table"].numpy()[:, 2] == pytest.approx(density, rel=1e-5)
    assert result["columns"][2] == "density"


def test_manual_range_pins_the_edges_regardless_of_the_data():
    result = _run(
        torch.tensor([0.4, 0.5, 0.6]),
        {"bins": 4, "range_mode": "manual", "range_min": 0.0, "range_max": 1.0},
    )
    table = result["table"].numpy()
    assert table[0, 0] == pytest.approx(0.0)
    assert table[-1, 1] == pytest.approx(1.0)
    assert table[:, 2].tolist() == pytest.approx([0.0, 1.0, 2.0, 0.0])


def test_non_finite_values_are_dropped_and_counted():
    data = torch.tensor([1.0, 2.0, float("nan"), float("inf"), -float("inf"), 3.0])
    result = _run(data, {"bins": 2})
    assert result["dropped"] == 3
    assert result["table"].numpy()[:, 2].sum() == pytest.approx(3.0)
    assert "non-finite" in result["chart"]["note"]


def test_chart_payload_is_a_bar_chart_with_one_bar_per_bin():
    result = _run(torch.tensor([1.0, 2.0, 3.0, 4.0]), {"bins": 4, "title": "Widths"})
    chart = result["chart"]
    assert chart["kind"] == "bar"
    assert chart["title"] == "Widths"
    assert len(chart["bars"]) == 4
    assert chart["bars"][0]["label"] == result["row_labels"][0]


def test_chart_payload_is_json_safe():
    """The run store serialises events with allow_nan=False."""
    chart = _run(torch.tensor([1.0, float("nan"), 3.0]), {"bins": 3})["chart"]
    encoded = json.dumps(chart, allow_nan=False)
    assert "NaN" not in encoded and "Infinity" not in encoded


def test_row_labels_read_as_half_open_intervals():
    labels = _run(torch.tensor([0.0, 1.0]), {"bins": 2})["row_labels"]
    assert labels == ["[0, 0.5)", "[0.5, 1)"]


# ── boundaries ───────────────────────────────────────────────────────────────

def test_a_single_bin_holds_everything():
    result = _run(torch.tensor([1.0, 2.0, 3.0]), {"bins": 1})
    assert result["table"].numpy()[:, 2].tolist() == pytest.approx([3.0])


def test_constant_data_still_produces_a_usable_range():
    """numpy widens a zero-width auto range to +-0.5; the node inherits that."""
    result = _run(torch.tensor([5.0, 5.0, 5.0]), {"bins": 2})
    table = result["table"].numpy()
    assert table[0, 0] < table[-1, 1]
    assert table[:, 2].sum() == pytest.approx(3.0)


def test_an_empty_tensor_produces_empty_bins_not_an_exception():
    """numpy cannot autodetect a range from nothing — the node substitutes 0..1."""
    result = _run(torch.zeros(0), {"bins": 4})
    assert result["dropped"] == 0
    assert result["table"].numpy()[:, 2].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert result["chart"]["bars"][0]["value"] == 0.0


def test_an_all_nan_tensor_behaves_like_an_empty_one():
    result = _run(torch.tensor([float("nan")] * 3), {"bins": 2})
    assert result["dropped"] == 3
    assert result["table"].numpy()[:, 2].sum() == 0.0


@pytest.mark.parametrize(
    "params, message",
    [
        ({"bins": 0}, "at least 1"),
        ({"range_mode": "sideways"}, "unknown range_mode"),
        ({"range_mode": "manual", "range_min": 1.0, "range_max": 1.0}, "range_min < range_max"),
        ({"range_mode": "manual", "range_min": 2.0, "range_max": 1.0}, "range_min < range_max"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(ValueError, match=message):
        _run(torch.tensor([1.0, 2.0]), params)


def test_a_2d_tensor_is_flattened():
    result = _run(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), {"bins": 2})
    assert result["table"].numpy()[:, 2].sum() == pytest.approx(4.0)
