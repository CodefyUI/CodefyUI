"""Stats-TableView and Stats-ChartView — the pack's two display nodes."""

from __future__ import annotations

import json

import pytest
import torch

from cdui_plugins.stats.nodes.stats_chart_view_node import (
    StatsChartViewNode,
    convert_chart,
)
from cdui_plugins.stats.nodes.stats_table_view_node import StatsTableViewNode


def _table_view(table, params=None, **inputs):
    return StatsTableViewNode().execute({"table": table, **inputs}, params or {})


def _chart_view(params=None, **inputs):
    return StatsChartViewNode().execute(dict(inputs), params or {})


# ── Stats-TableView ──────────────────────────────────────────────────────────

def test_table_view_metadata():
    assert StatsTableViewNode.NODE_NAME == "Stats-TableView"
    assert [p.name for p in StatsTableViewNode.define_outputs()] == ["text"]


def test_table_view_emits_the_log_key_so_it_renders_as_a_text_entry():
    """`__log__` is what the #117 output_kind channel turns into a text entry."""
    result = _table_view(torch.tensor([[1.0]]))
    assert result["__log__"] == result["text"]


def test_table_view_renders_headers_row_labels_and_aligned_values():
    result = _table_view(
        torch.tensor([[1.0, 22.5], [333.0, 4.0]]),
        columns=["a", "bb"],
        row_labels=["first", "second"],
    )
    lines = result["text"].splitlines()
    assert lines[0].split() == ["a", "bb"]
    assert lines[1].split() == ["first", "1", "22.5000"]
    assert lines[2].split() == ["second", "333", "4"]
    # Right-aligned on the widest cell, so the numbers form a column.
    assert len(lines[1]) == len(lines[2])


def test_table_view_prints_integers_without_a_decimal_tail():
    assert "150" in _table_view(torch.tensor([[150.0]]))["text"]
    assert "150.0000" not in _table_view(torch.tensor([[150.0]]))["text"]


def test_table_view_honours_precision():
    text = _table_view(torch.tensor([[1.23456]]), {"precision": 2})["text"]
    assert "1.23" in text and "1.2346" not in text


def test_table_view_truncates_and_says_how_many_rows_it_left_out():
    text = _table_view(torch.zeros((10, 1)), {"max_rows": 3})["text"]
    assert "... 7 more row(s) of 10" in text
    assert len(text.splitlines()) == 5  # header + 3 rows + the notice


def test_max_rows_zero_shows_every_row():
    text = _table_view(torch.zeros((10, 1)), {"max_rows": 0})["text"]
    assert "more row(s)" not in text
    assert len(text.splitlines()) == 11


def test_table_view_title_is_printed_above_the_table():
    text = _table_view(torch.tensor([[1.0]]), {"title": "Summary"})["text"]
    assert text.splitlines()[0] == "Summary"


def test_table_view_generates_missing_labels():
    text = _table_view(torch.zeros((2, 2)))["text"]
    assert "c0" in text and "c1" in text
    assert text.splitlines()[1].startswith("0")


def test_table_view_renders_nan_and_inf_readably():
    text = _table_view(torch.tensor([[float("nan"), float("inf"), -float("inf")]]))["text"]
    assert "NaN" in text and "inf" in text and "-inf" in text


def test_table_view_rejects_a_missing_table():
    with pytest.raises(ValueError, match="requires a `table` input"):
        StatsTableViewNode().execute({}, {})


# ── Stats-ChartView ──────────────────────────────────────────────────────────

def test_chart_view_metadata():
    assert StatsChartViewNode.NODE_NAME == "Stats-ChartView"
    outputs = StatsChartViewNode.define_outputs()
    assert [p.name for p in outputs] == ["chart"]
    assert outputs[0].media == "chart", "the output must declare the chart media kind"


def test_chart_view_passes_a_payload_through_unchanged_on_auto():
    payload = {"kind": "heatmap", "matrix": [[1.0]], "title": "T"}
    assert _chart_view(chart=payload)["chart"] == payload


def test_chart_view_retitles_without_touching_the_data():
    payload = {"kind": "bar", "bars": [{"label": "a", "value": 1.0}], "title": "old"}
    result = _chart_view({"title": "new"}, chart=payload)["chart"]
    assert result["title"] == "new"
    assert result["bars"] == payload["bars"]
    assert payload["title"] == "old", "the incoming payload must not be mutated"


def test_chart_view_charts_a_table_with_row_labels_as_bars():
    """The CSV -> GroupBy -> Chart demo shape."""
    result = _chart_view(
        table=torch.tensor([[1.0], [2.0], [3.0]]),
        columns=["petal"],
        row_labels=["setosa", "versicolor", "virginica"],
    )["chart"]
    assert result["kind"] == "bar"
    assert [b["label"] for b in result["bars"]] == ["setosa", "versicolor", "virginica"]
    assert [b["value"] for b in result["bars"]] == [1.0, 2.0, 3.0]


def test_chart_view_charts_a_table_without_row_labels_as_lines():
    result = _chart_view(table=torch.tensor([[1.0, 4.0], [2.0, 5.0]]))["chart"]
    assert result["kind"] == "line"
    assert [s["name"] for s in result["series"]] == ["c0", "c1"]
    assert result["series"][0]["points"] == [[0.0, 1.0], [1.0, 2.0]]


def test_a_multi_column_bar_chart_says_which_column_it_drew():
    result = _chart_view(
        {"kind": "bar"},
        table=torch.tensor([[1.0, 2.0]]),
        columns=["a", "b"],
        row_labels=["g"],
    )["chart"]
    assert result["bars"][0]["value"] == 1.0
    assert "'a'" in result["note"]


def test_columns_filter_picks_the_charted_column():
    result = _chart_view(
        {"kind": "bar", "columns_filter": "b"},
        table=torch.tensor([[1.0, 2.0]]),
        columns=["a", "b"],
        row_labels=["g"],
    )["chart"]
    assert result["bars"][0]["value"] == 2.0
    assert "note" not in result


def test_scatter_from_a_table_uses_the_first_two_columns():
    result = _chart_view(
        {"kind": "scatter"},
        table=torch.tensor([[1.0, 10.0], [2.0, 20.0]]),
        columns=["x", "y"],
    )["chart"]
    assert result["kind"] == "scatter"
    assert result["points"] == [
        {"x": 1.0, "y": 10.0, "label": "0"},
        {"x": 2.0, "y": 20.0, "label": "1"},
    ]
    assert (result["x_label"], result["y_label"]) == ("x", "y")


def test_heatmap_from_a_table_keeps_both_axes_labelled():
    result = _chart_view(
        {"kind": "heatmap"},
        table=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        columns=["p", "q"],
        row_labels=["u", "v"],
    )["chart"]
    assert result["matrix"] == [[1.0, 2.0], [3.0, 4.0]]
    assert result["row_labels"] == ["u", "v"]
    assert result["col_labels"] == ["p", "q"]


def test_chart_view_rejects_a_scatter_of_one_column():
    with pytest.raises(ValueError, match="needs two columns"):
        _chart_view({"kind": "scatter"}, table=torch.tensor([[1.0], [2.0]]))


def test_chart_view_needs_one_of_its_two_inputs():
    with pytest.raises(ValueError, match="connect either a `chart` payload or a `table`"):
        _chart_view()


def test_a_dict_that_is_not_a_chart_payload_is_rejected():
    with pytest.raises(ValueError, match="carries no `kind`"):
        _chart_view(chart={"bars": []})


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown kind"):
        _chart_view({"kind": "pie"}, table=torch.zeros((2, 2)))


def test_unknown_columns_filter_is_rejected():
    with pytest.raises(ValueError, match="unknown column"):
        _chart_view({"columns_filter": "nope"}, table=torch.zeros((2, 1)))


# ── convert_chart ────────────────────────────────────────────────────────────

BARS = {"kind": "bar", "title": "T", "bars": [
    {"label": "a", "value": 1.0}, {"label": "b", "value": 2.0},
]}


def test_converting_to_the_same_kind_is_a_no_op():
    assert convert_chart(BARS, "bar") is BARS


def test_bars_become_a_single_line_series_indexed_by_position():
    line = convert_chart(BARS, "line")
    assert line["kind"] == "line"
    assert line["series"][0]["points"] == [[0.0, 1.0], [1.0, 2.0]]
    assert line["title"] == "T"


def test_bars_become_scatter_points_keeping_their_labels():
    scatter = convert_chart(BARS, "scatter")
    assert scatter["kind"] == "scatter"
    assert [p["label"] for p in scatter["points"]] == ["a", "b"]


def test_a_line_becomes_bars_labelled_by_x():
    line = {"kind": "line", "series": [{"name": "s", "points": [[3.0, 9.0]]}]}
    bars = convert_chart(line, "bar")
    assert bars["bars"] == [{"label": "s", "value": 9.0}]


def test_a_heatmap_refuses_to_become_a_bar_chart_and_says_so():
    heatmap = {"kind": "heatmap", "matrix": [[1.0]]}
    result = convert_chart(heatmap, "bar")
    assert result["kind"] == "heatmap"
    assert "cannot redraw" in result["note"]


def test_nothing_can_become_a_heatmap():
    result = convert_chart(BARS, "heatmap")
    assert result["kind"] == "bar"
    assert "cannot redraw" in result["note"]


def test_an_existing_note_survives_a_conversion():
    noted = dict(BARS, note="sampled")
    assert convert_chart(noted, "line")["note"].startswith("sampled")


def test_every_produced_spec_is_json_safe():
    for kind in ("bar", "line", "scatter"):
        json.dumps(convert_chart(BARS, kind), allow_nan=False)
