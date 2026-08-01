"""Stats-Percentile: numpy parity for each axis, and NaN-skipping semantics."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from cdui_plugins.stats.nodes.stats_percentile_node import StatsPercentileNode


def _run(tensor, params=None, columns=None):
    inputs = {"tensor": tensor}
    if columns is not None:
        inputs["columns"] = columns
    return StatsPercentileNode().execute(inputs, params or {})


def test_node_metadata():
    assert StatsPercentileNode.NODE_NAME == "Stats-Percentile"
    assert [p.name for p in StatsPercentileNode.define_outputs()] == [
        "percentiles",
        "columns",
        "row_labels",
    ]


def test_axis_all_matches_numpy_percentile():
    data = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    result = _run(torch.tensor(data), {"q": "10,50,90"})

    assert result["row_labels"] == ["10%", "50%", "90%"]
    assert result["columns"] == ["all"]
    assert result["percentiles"].shape == (3,)
    expected = np.percentile(data, [10, 50, 90])
    assert result["percentiles"].tolist() == pytest.approx(expected, rel=1e-6)


def test_axis_columns_matches_numpy_per_column():
    data = np.arange(24, dtype=np.float64).reshape(8, 3)
    result = _run(
        torch.tensor(data), {"q": "25,75", "axis": "columns"}, columns=["a", "b", "c"]
    )

    assert result["columns"] == ["a", "b", "c"]
    assert result["percentiles"].shape == (2, 3)
    expected = np.percentile(data, [25, 75], axis=0)
    assert result["percentiles"].numpy() == pytest.approx(expected, rel=1e-6)


def test_axis_rows_matches_numpy_per_row():
    data = np.arange(12, dtype=np.float64).reshape(3, 4)
    result = _run(torch.tensor(data), {"q": "50", "axis": "rows"})

    assert result["columns"] == ["r0", "r1", "r2"]
    expected = np.percentile(data, [50], axis=1)
    assert result["percentiles"].numpy() == pytest.approx(expected, rel=1e-6)


def test_interpolation_is_linear_not_nearest():
    # Median of [1, 2, 3, 4] is 2.5 under linear interpolation, 2 or 3 under
    # a nearest-rank rule — the single clearest way to tell them apart.
    result = _run(torch.tensor([1.0, 2.0, 3.0, 4.0]), {"q": "50"})
    assert result["percentiles"].tolist() == pytest.approx([2.5])


def test_nan_is_skipped_matching_nanpercentile_not_percentile():
    data = np.array([1.0, float("nan"), 3.0, 5.0])
    result = _run(torch.tensor(data), {"q": "50"})

    assert result["percentiles"].tolist() == pytest.approx(
        [float(np.nanpercentile(data, 50))]
    )
    # np.percentile itself would propagate the NaN; the node deliberately does not.
    assert math.isnan(float(np.percentile(data, 50)))


def test_percentiles_are_sorted_and_deduplicated():
    result = _run(torch.arange(10, dtype=torch.float32), {"q": "90,10,10"})
    assert result["row_labels"] == ["10%", "90%"]


def test_fractional_percentiles_keep_their_label():
    result = _run(torch.arange(1000, dtype=torch.float32), {"q": "2.5,99.9"})
    assert result["row_labels"] == ["2.5%", "99.9%"]


# ── boundaries ───────────────────────────────────────────────────────────────

def test_an_all_nan_series_reports_nan_without_warning():
    result = _run(torch.tensor([float("nan"), float("nan")]), {"q": "50"})
    assert math.isnan(result["percentiles"].tolist()[0])


def test_a_single_value_is_its_own_every_percentile():
    result = _run(torch.tensor([7.0]), {"q": "0,50,100"})
    assert result["percentiles"].tolist() == pytest.approx([7.0, 7.0, 7.0])


def test_empty_q_is_rejected_rather_than_returning_an_empty_tensor():
    with pytest.raises(ValueError, match="`q` is empty"):
        _run(torch.zeros(4), {"q": " "})


def test_unknown_axis_is_rejected():
    with pytest.raises(ValueError, match="unknown axis"):
        _run(torch.zeros(4), {"axis": "sideways"})


def test_missing_input_is_a_clear_error():
    with pytest.raises(ValueError, match="requires a `tensor` input"):
        StatsPercentileNode().execute({}, {})


def test_output_is_float32():
    assert _run(torch.zeros(4))["percentiles"].dtype == torch.float32
