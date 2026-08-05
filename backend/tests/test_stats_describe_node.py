"""Stats-Describe: pandas `describe()` parity plus the axis / percentile knobs.

pandas is a backend dependency already (CSVReader reads through it), but the
stats pack itself never imports it — the pack is numpy + torch only. pandas
appears here purely as the reference implementation, which is the point: a
test that recomputed the statistics with the same numpy calls the node uses
would pass no matter how wrong both were.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cdui_plugins.stats.nodes.stats_describe_node import StatsDescribeNode

IRIS_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


def _run(table, params=None, columns=None):
    inputs = {"table": table}
    if columns is not None:
        inputs["columns"] = columns
    return StatsDescribeNode().execute(inputs, params or {})


def _as_frame(result) -> pd.DataFrame:
    """The node's output as a DataFrame indexed the way pandas indexes describe()."""
    return pd.DataFrame(
        result["table"].numpy(),
        index=result["row_labels"],
        columns=result["columns"],
    )


@pytest.fixture(scope="module")
def iris() -> pd.DataFrame:
    # Resolved relative to this file, not the pytest cwd (#185) -- a bare
    # "data/samples/iris.csv" only worked when pytest ran from backend/.
    return pd.read_csv(
        Path(__file__).resolve().parents[1] / "data/samples/iris.csv"
    )[IRIS_COLUMNS]


def test_node_metadata():
    assert StatsDescribeNode.NODE_NAME == "Stats-Describe"
    assert StatsDescribeNode.CATEGORY == "Data"
    assert [p.name for p in StatsDescribeNode.define_outputs()] == [
        "table",
        "columns",
        "row_labels",
    ]
    assert StatsDescribeNode.cacheable is True


def test_reproduces_pandas_describe_on_the_iris_fixture(iris):
    """The acceptance criterion: every cell matches pandas on a real CSV."""
    result = _run(torch.tensor(iris.to_numpy(), dtype=torch.float32),
                  columns=IRIS_COLUMNS)
    expected = iris.describe()

    assert result["row_labels"] == list(expected.index)
    assert result["columns"] == list(expected.columns)

    got = _as_frame(result)
    for stat in expected.index:
        for column in expected.columns:
            assert got.loc[stat, column] == pytest.approx(
                expected.loc[stat, column], rel=1e-5
            ), f"{stat} of {column!r}"


def test_row_labels_and_order_match_pandas():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    result = _run(torch.tensor(frame.to_numpy()), columns=["a"])
    assert result["row_labels"] == ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    assert result["row_labels"] == list(frame.describe().index)


def test_custom_percentiles_report_exactly_what_was_asked_for():
    """No implicit median.

    pandas <=2.x appended 0.5 to any percentile list and 3.0 stopped; the CI
    matrix spans both (3.10 gets pandas 2.x, 3.11+ gets 3.x). The node follows
    the surviving rule, so this asserts the labels itself and only borrows
    pandas for the VALUES of the percentiles both versions agree on.
    """
    frame = pd.DataFrame({"a": [float(v) for v in range(1, 21)]})
    result = _run(torch.tensor(frame.to_numpy()), {"percentiles": "10,90"}, columns=["a"])

    assert result["row_labels"] == ["count", "mean", "std", "min", "10%", "90%", "max"]

    expected = frame.describe(percentiles=[0.1, 0.9])
    got = _as_frame(result)
    for stat in result["row_labels"]:
        assert got.loc[stat, "a"] == pytest.approx(expected.loc[stat, "a"], rel=1e-6), stat


def test_percentiles_are_sorted_and_deduplicated():
    result = _run(torch.arange(10, dtype=torch.float32).reshape(-1, 1),
                  {"percentiles": "75,25,25,50"})
    assert result["row_labels"] == ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]


def test_std_is_the_sample_std_not_the_population_one():
    result = _run(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
    got = _as_frame(result)
    # ddof=1 => sqrt(5/3) = 1.29099; the population std would be 1.11803.
    assert got.loc["std", "c0"] == pytest.approx(math.sqrt(5.0 / 3.0), rel=1e-6)


# Arithmetic on an inf legitimately raises "invalid value encountered" inside
# BOTH numpy and pandas (inf - inf); that is the behaviour under test, not a
# defect, and the warning would otherwise be the suite's only noise.
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_nan_is_missing_and_inf_is_a_value_like_pandas():
    frame = pd.DataFrame({"holes": [1.0, float("nan"), 3.0], "big": [1.0, 2.0, float("inf")]})
    result = _run(torch.tensor(frame.to_numpy()), columns=list(frame.columns))
    expected = frame.describe()
    got = _as_frame(result)

    assert got.loc["count", "holes"] == 2.0 == expected.loc["count", "holes"]
    assert got.loc["mean", "holes"] == pytest.approx(2.0)
    # An inf is data, so it counts and it poisons the mean — in both libraries.
    assert got.loc["count", "big"] == 3.0 == expected.loc["count", "big"]
    assert math.isinf(got.loc["mean", "big"]) and math.isinf(expected.loc["mean", "big"])


def test_axis_rows_describes_each_row():
    result = _run(torch.tensor([[1.0, 3.0], [10.0, 20.0]]), {"axis": "rows"})
    got = _as_frame(result)
    assert result["columns"] == ["r0", "r1"]
    assert got.loc["mean", "r0"] == pytest.approx(2.0)
    assert got.loc["mean", "r1"] == pytest.approx(15.0)


def test_axis_all_describes_the_whole_table_as_one_series():
    result = _run(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), {"axis": "all"})
    got = _as_frame(result)
    assert result["columns"] == ["all"]
    assert got.loc["count", "all"] == 4.0
    assert got.loc["mean", "all"] == pytest.approx(2.5)


def test_a_1d_vector_is_treated_as_one_column():
    result = _run(torch.tensor([1.0, 2.0, 3.0]))
    assert result["columns"] == ["c0"]
    assert _as_frame(result).loc["mean", "c0"] == pytest.approx(2.0)


def test_missing_column_names_are_generated_and_extra_ones_trimmed():
    table = torch.zeros((2, 3))
    assert _run(table)["columns"] == ["c0", "c1", "c2"]
    assert _run(table, columns=["a"])["columns"] == ["a", "c1", "c2"]
    assert _run(table, columns=["a", "b", "c", "d"])["columns"] == ["a", "b", "c"]


def test_duplicate_column_names_are_disambiguated():
    assert _run(torch.zeros((2, 3)), columns=["a", "a", "a"])["columns"] == [
        "a", "a#1", "a#2",
    ]


# ── boundaries ───────────────────────────────────────────────────────────────

def test_an_all_nan_column_reports_count_zero_and_nan_statistics():
    result = _run(torch.tensor([[float("nan")], [float("nan")]]))
    got = _as_frame(result)
    assert got.loc["count", "c0"] == 0.0
    for stat in ("mean", "std", "min", "50%", "max"):
        assert math.isnan(got.loc[stat, "c0"]), stat


def test_a_single_row_has_no_sample_std():
    result = _run(torch.tensor([[7.0]]))
    got = _as_frame(result)
    assert got.loc["count", "c0"] == 1.0
    assert got.loc["mean", "c0"] == pytest.approx(7.0)
    # ddof=1 with one observation is undefined — pandas says NaN, so do we.
    assert math.isnan(got.loc["std", "c0"])
    assert math.isnan(pd.DataFrame({"a": [7.0]}).describe().loc["std", "a"])


def test_zero_rows_produce_an_empty_but_well_formed_table():
    result = _run(torch.zeros((0, 2)))
    assert result["columns"] == ["c0", "c1"]
    assert result["table"].shape == (8, 2)
    assert result["table"][0].tolist() == [0.0, 0.0]  # count


def test_a_list_of_lists_is_accepted_like_a_tensor():
    result = _run([[1.0, 2.0], [3.0, 4.0]])
    assert _as_frame(result).loc["mean", "c0"] == pytest.approx(2.0)


def test_missing_table_input_is_a_clear_error():
    with pytest.raises(ValueError, match="requires a `table` input"):
        StatsDescribeNode().execute({}, {})


def test_a_3d_tensor_is_rejected_by_shape_not_silently_flattened():
    with pytest.raises(ValueError, match="2D"):
        _run(torch.zeros((2, 3, 4)))


def test_unknown_axis_is_rejected():
    with pytest.raises(ValueError, match="unknown axis"):
        _run(torch.zeros((2, 2)), {"axis": "diagonal"})


@pytest.mark.parametrize("bad", ["abc", "25,abc", "-1", "101"])
def test_invalid_percentiles_are_rejected(bad):
    with pytest.raises(ValueError, match="percentiles"):
        _run(torch.zeros((3, 1)), {"percentiles": bad})


def test_the_output_table_is_float32_like_every_other_table_in_the_repo():
    assert _run(torch.zeros((3, 1)))["table"].dtype == torch.float32


def test_output_is_a_valid_input_for_a_second_describe():
    """The table contract round-trips: describe(describe(x)) is well-formed."""
    first = _run(torch.tensor(np.arange(12, dtype=np.float32).reshape(4, 3)))
    second = _run(first["table"], columns=first["columns"])
    assert second["columns"] == first["columns"]
    assert second["table"].shape[1] == 3
