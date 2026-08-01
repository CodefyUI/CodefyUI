"""Stats-GroupByAggregate: pandas groupby parity, both grouping sources."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from cdui_plugins.stats.nodes.stats_group_by_aggregate_node import (
    StatsGroupByAggregateNode,
)

IRIS_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


def _run(table, params=None, columns=None, keys=None):
    inputs = {"table": table}
    if columns is not None:
        inputs["columns"] = columns
    if keys is not None:
        inputs["keys"] = keys
    return StatsGroupByAggregateNode().execute(inputs, params or {})


@pytest.fixture(scope="module")
def iris() -> pd.DataFrame:
    return pd.read_csv("data/samples/iris.csv")


def test_node_metadata():
    assert StatsGroupByAggregateNode.NODE_NAME == "Stats-GroupByAggregate"
    assert [p.name for p in StatsGroupByAggregateNode.define_outputs()] == [
        "table", "columns", "row_labels", "counts",
    ]


@pytest.mark.parametrize("agg", ["mean", "sum", "count", "min", "max", "std"])
def test_every_aggregate_matches_pandas_groupby_on_iris(iris, agg):
    """The CSV -> GroupBy demo shape, checked against pandas for each aggregate."""
    result = _run(
        torch.tensor(iris[IRIS_COLUMNS].to_numpy(), dtype=torch.float32),
        {"agg": agg},
        columns=IRIS_COLUMNS,
        keys=[str(v) for v in iris["species"].tolist()],
    )
    expected = iris.groupby("species")[IRIS_COLUMNS].agg(agg)

    assert result["row_labels"] == list(expected.index)
    assert result["columns"] == [f"{c} [{agg}]" for c in IRIS_COLUMNS]
    assert result["table"].numpy() == pytest.approx(expected.to_numpy(), rel=1e-5)


def test_group_counts_are_the_row_counts(iris):
    result = _run(
        torch.tensor(iris[IRIS_COLUMNS].to_numpy(), dtype=torch.float32),
        columns=IRIS_COLUMNS,
        keys=[str(v) for v in iris["species"].tolist()],
    )
    expected = iris.groupby("species").size()
    assert result["counts"].tolist() == pytest.approx(expected.to_numpy())


def test_groups_come_back_in_sorted_order_like_pandas():
    result = _run(
        torch.tensor([[1.0], [2.0], [3.0]]),
        keys=["zebra", "apple", "mango"],
    )
    assert result["row_labels"] == ["apple", "mango", "zebra"]
    assert result["table"].tolist() == [[2.0], [3.0], [1.0]]


def test_group_by_a_table_column_by_name():
    table = torch.tensor([[1.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
    result = _run(table, {"group_by": "key"}, columns=["key", "value"])
    assert result["row_labels"] == ["1", "2"]
    assert result["columns"] == ["value [mean]"]
    assert result["table"].tolist() == [[15.0], [30.0]]


def test_group_by_a_table_column_by_index():
    table = torch.tensor([[1.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
    result = _run(table, {"group_by": "0"})
    assert result["table"].tolist() == [[15.0], [30.0]]


def test_group_by_several_columns_makes_a_compound_key():
    table = torch.tensor([
        [1.0, 1.0, 5.0],
        [1.0, 2.0, 7.0],
        [1.0, 1.0, 9.0],
    ])
    result = _run(table, {"group_by": "0,1"}, columns=["a", "b", "v"])
    assert result["row_labels"] == ["1|1", "1|2"]
    assert result["table"].tolist() == [[7.0], [7.0]]


def test_numeric_group_keys_sort_numerically_not_as_text():
    """Sorting the rendered labels would put 10 before 2."""
    table = torch.tensor([[10.0, 1.0], [2.0, 2.0]])
    result = _run(table, {"group_by": "0"})
    assert result["row_labels"] == ["2", "10"]


def test_group_by_param_wins_over_a_connected_keys_input():
    table = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
    result = _run(table, {"group_by": "0"}, keys=["same", "same"])
    assert result["row_labels"] == ["1", "2"]


def test_agg_overrides_apply_per_column_and_show_in_the_names():
    table = torch.tensor([[1.0, 10.0], [3.0, 20.0]])
    result = _run(
        table, {"agg": "mean", "agg_overrides": "b=max"}, columns=["a", "b"],
        keys=["g", "g"],
    )
    assert result["columns"] == ["a [mean]", "b [max]"]
    assert result["table"].tolist() == [[2.0, 20.0]]


def test_std_is_the_sample_std_like_pandas():
    result = _run(torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
                  {"agg": "std"}, keys=["g"] * 4)
    expected = pd.DataFrame({"g": ["g"] * 4, "v": [1.0, 2.0, 3.0, 4.0]}).groupby("g").std()
    assert result["table"].tolist()[0][0] == pytest.approx(expected.iloc[0, 0], rel=1e-6)


def test_nan_is_skipped_per_column_like_pandas():
    frame = pd.DataFrame({"g": ["a", "a", "a"], "v": [1.0, float("nan"), 3.0]})
    result = _run(torch.tensor([[1.0], [float("nan")], [3.0]]), keys=list(frame["g"]))
    assert result["table"].tolist() == [[pytest.approx(2.0)]]
    assert result["table"].tolist()[0][0] == pytest.approx(
        frame.groupby("g").mean().iloc[0, 0]
    )


def test_count_counts_present_values_while_counts_counts_rows():
    result = _run(
        torch.tensor([[1.0], [float("nan")], [3.0]]), {"agg": "count"}, keys=["g"] * 3
    )
    assert result["table"].tolist() == [[2.0]]   # two present values
    assert result["counts"].tolist() == [3.0]    # three rows in the group


# ── boundaries ───────────────────────────────────────────────────────────────

def test_an_all_nan_group_sums_to_zero_and_averages_to_nan_like_pandas():
    frame = pd.DataFrame({"g": ["a", "a"], "v": [float("nan")] * 2})
    table = torch.tensor([[float("nan")], [float("nan")]])

    summed = _run(table, {"agg": "sum"}, keys=["a", "a"])["table"].tolist()[0][0]
    assert summed == 0.0 == frame.groupby("g").sum().iloc[0, 0]

    averaged = _run(table, {"agg": "mean"}, keys=["a", "a"])["table"].tolist()[0][0]
    assert math.isnan(averaged)
    assert math.isnan(frame.groupby("g").mean().iloc[0, 0])


def test_a_one_row_group_has_no_sample_std():
    result = _run(torch.tensor([[5.0]]), {"agg": "std"}, keys=["only"])
    assert math.isnan(result["table"].tolist()[0][0])


def test_every_row_in_its_own_group_is_the_identity():
    table = torch.tensor([[1.0], [2.0], [3.0]])
    result = _run(table, keys=["a", "b", "c"])
    assert result["table"].tolist() == [[1.0], [2.0], [3.0]]
    assert result["counts"].tolist() == [1.0, 1.0, 1.0]


def test_no_keys_and_no_group_by_is_a_clear_error():
    with pytest.raises(ValueError, match="nothing to group by"):
        _run(torch.zeros((2, 2)))


def test_misaligned_keys_are_rejected():
    with pytest.raises(ValueError, match="must be aligned"):
        _run(torch.zeros((3, 1)), keys=["a", "b"])


def test_grouping_by_every_column_leaves_nothing_to_aggregate():
    with pytest.raises(ValueError, match="nothing left to aggregate"):
        _run(torch.zeros((2, 1)), {"group_by": "0"})


@pytest.mark.parametrize(
    "params, message",
    [
        ({"agg": "median"}, "unknown agg"),
        ({"agg_overrides": "a"}, "not a 'column=agg' pair"),
        ({"agg_overrides": "a=median"}, "unknown aggregate"),
        ({"agg_overrides": "nope=max"}, "unknown column"),
        ({"group_by": "nope"}, "unknown column"),
        ({"group_by": "9"}, "out of range"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(ValueError, match=message):
        _run(torch.zeros((2, 2)), params, columns=["a", "b"], keys=["g", "g"])


def test_output_is_float32():
    result = _run(torch.zeros((2, 1)), keys=["g", "g"])
    assert result["table"].dtype == torch.float32
    assert result["counts"].dtype == torch.float32


def test_output_feeds_straight_into_a_table_view_contract():
    """[groups, columns] + names + row labels — the pack's table contract."""
    result = _run(torch.tensor(np.arange(6.0).reshape(3, 2)), keys=["a", "b", "a"])
    assert result["table"].shape == (2, 2)
    assert len(result["columns"]) == result["table"].shape[1]
    assert len(result["row_labels"]) == result["table"].shape[0]
