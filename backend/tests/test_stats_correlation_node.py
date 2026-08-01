"""Stats-Correlation: pandas parity for pearson and spearman, and NaN policy.

pandas is the reference here for the same reason as in the describe tests: it
implements pairwise-complete correlation and average-rank Spearman
independently, so agreeing with it is evidence, whereas re-deriving the same
formula next to the node would not be.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from cdui_plugins.stats.nodes.stats_correlation_node import (
    StatsCorrelationNode,
    average_ranks,
)


def _run(table, params=None, columns=None):
    inputs = {"table": table}
    if columns is not None:
        inputs["columns"] = columns
    return StatsCorrelationNode().execute(inputs, params or {})


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    a = rng.normal(size=60)
    return pd.DataFrame({
        "a": a,
        "b": 2.0 * a + rng.normal(scale=0.3, size=60),   # strongly linear
        "c": rng.normal(size=60),                         # unrelated
        "d": np.exp(a),                                   # monotonic, not linear
    })


def test_node_metadata():
    assert StatsCorrelationNode.NODE_NAME == "Stats-Correlation"
    outputs = StatsCorrelationNode.define_outputs()
    assert [p.name for p in outputs] == ["matrix", "columns", "row_labels", "chart"]
    assert next(p for p in outputs if p.name == "chart").media == "chart"


def test_pearson_matches_pandas(frame):
    result = _run(torch.tensor(frame.to_numpy()), columns=list(frame.columns))
    expected = frame.corr(method="pearson").to_numpy()
    assert result["matrix"].numpy() == pytest.approx(expected, rel=1e-5, abs=1e-6)
    assert result["columns"] == list(frame.columns)
    assert result["row_labels"] == result["columns"]


def test_spearman_matches_pandas(frame):
    result = _run(torch.tensor(frame.to_numpy()), {"method": "spearman"},
                  columns=list(frame.columns))
    expected = frame.corr(method="spearman").to_numpy()
    assert result["matrix"].numpy() == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_spearman_sees_a_monotonic_relationship_pearson_understates(frame):
    """exp(a) vs a is a perfect rank correlation but an imperfect linear one."""
    pearson = _run(torch.tensor(frame.to_numpy()))["matrix"].numpy()
    spearman = _run(torch.tensor(frame.to_numpy()), {"method": "spearman"})["matrix"].numpy()
    assert spearman[0, 3] == pytest.approx(1.0, abs=1e-6)
    assert pearson[0, 3] < 0.95


def test_the_diagonal_is_one_and_the_matrix_is_symmetric(frame):
    matrix = _run(torch.tensor(frame.to_numpy()))["matrix"].numpy()
    assert np.diag(matrix) == pytest.approx(np.ones(4))
    assert matrix == pytest.approx(matrix.T)


def test_nan_is_dropped_pairwise_by_default_like_pandas():
    frame = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [2.0, 4.0, 6.0, 8.0, 10.0],
        "c": [1.0, float("nan"), 2.0, 9.0, 4.0],
    })
    matrix = _run(torch.tensor(frame.to_numpy()))["matrix"].numpy()
    expected = frame.corr().to_numpy()

    # The hole in `c` must not touch the a-b cell.
    assert matrix[0, 1] == pytest.approx(1.0)
    assert matrix == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_drop_nan_off_lets_nan_propagate_like_numpy():
    table = torch.tensor([[1.0, 1.0], [2.0, float("nan")], [3.0, 5.0]])
    matrix = _run(table, {"drop_nan": False})["matrix"].numpy()
    assert matrix[0, 0] == pytest.approx(1.0)
    assert math.isnan(matrix[0, 1])
    assert math.isnan(matrix[1, 1])


def test_drop_nan_off_also_propagates_through_spearman():
    """Ranking a NaN would give it a finite rank and invent a correlation."""
    table = torch.tensor([[1.0, 1.0], [2.0, float("nan")], [3.0, 5.0]])
    matrix = _run(table, {"method": "spearman", "drop_nan": False})["matrix"].numpy()
    assert math.isnan(matrix[0, 1])


def test_chart_is_a_heatmap_pinned_to_the_diverging_range():
    result = _run(torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.1]]))
    chart = result["chart"]
    assert chart["kind"] == "heatmap"
    assert chart["colormap"] == "RdBu"
    # Never the data's own extent: a weak matrix must not look like a strong one.
    assert (chart["vmin"], chart["vmax"]) == (-1.0, 1.0)
    assert chart["row_labels"] == chart["col_labels"] == result["columns"]


def test_chart_payload_is_json_safe_even_with_undefined_cells():
    table = torch.tensor([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])  # constant column
    chart = _run(table)["chart"]
    json.dumps(chart, allow_nan=False)


# ── average_ranks ────────────────────────────────────────────────────────────

def test_average_ranks_assigns_shared_ranks_to_ties():
    assert average_ranks(np.array([10.0, 20.0, 20.0, 40.0])).tolist() == [
        1.0, 2.5, 2.5, 4.0,
    ]


def test_average_ranks_handles_an_all_equal_vector():
    assert average_ranks(np.array([5.0, 5.0, 5.0])).tolist() == [2.0, 2.0, 2.0]


def test_average_ranks_of_an_empty_vector_is_empty():
    assert average_ranks(np.array([])).tolist() == []


# ── boundaries ───────────────────────────────────────────────────────────────

def test_a_constant_column_has_no_defined_correlation():
    matrix = _run(torch.tensor([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]]))["matrix"].numpy()
    assert matrix[0, 0] == pytest.approx(1.0)
    assert math.isnan(matrix[0, 1])
    assert math.isnan(matrix[1, 1])


def test_a_single_row_has_too_few_pairs():
    matrix = _run(torch.tensor([[1.0, 2.0]]))["matrix"].numpy()
    assert np.isnan(matrix).all()


def test_a_single_column_is_a_one_by_one_matrix():
    result = _run(torch.tensor([[1.0], [2.0], [3.0]]))
    assert result["matrix"].shape == (1, 1)
    assert result["matrix"].tolist() == [[pytest.approx(1.0)]]


def test_column_names_default_when_none_are_connected():
    assert _run(torch.zeros((3, 2)))["columns"] == ["c0", "c1"]


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        _run(torch.zeros((3, 2)), {"method": "kendall"})


def test_missing_table_input_is_a_clear_error():
    with pytest.raises(ValueError, match="requires a `table` input"):
        StatsCorrelationNode().execute({}, {})


def test_output_is_float32():
    assert _run(torch.zeros((3, 2)))["matrix"].dtype == torch.float32
