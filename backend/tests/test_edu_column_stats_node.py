"""Tests for EduColumnStatsNode (chapter pack C1)."""

from __future__ import annotations

import pytest
import torch

from cdui_plugins.foundations.nodes.edu_column_stats_node import EduColumnStatsNode


def _run(table, **params):
    return EduColumnStatsNode().execute({"table": table}, params)


def test_node_metadata():
    assert EduColumnStatsNode.NODE_NAME == "Edu-ColumnStats"
    assert EduColumnStatsNode.CATEGORY == "Classical"
    out_names = [p.name for p in EduColumnStatsNode.define_outputs()]
    assert out_names == ["means", "stds", "mins", "maxs", "counts"]


def test_simple_column_means():
    table = torch.tensor([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    res = _run(table)
    # column 0: mean(1,3,5)=3; column 1: mean(10,20,30)=20
    assert res["means"].tolist() == pytest.approx([3.0, 20.0])


def test_population_std_default():
    # values 1,3,5 → variance = mean((-2)², 0², 2²) = (4+0+4)/3 = 8/3
    table = torch.tensor([[1.0], [3.0], [5.0]])
    res = _run(table)
    assert res["stds"].item() == pytest.approx((8.0 / 3.0) ** 0.5, rel=1e-6)


def test_sample_std_when_unbiased():
    # same data, divide by N-1=2 → variance = 8/2 = 4 → std = 2
    table = torch.tensor([[1.0], [3.0], [5.0]])
    res = _run(table, unbiased=True)
    assert res["stds"].item() == pytest.approx(2.0, rel=1e-6)


def test_min_and_max():
    table = torch.tensor([[1.0, 9.0], [-1.0, 7.0], [5.0, 0.0]])
    res = _run(table)
    assert res["mins"].tolist() == pytest.approx([-1.0, 0.0])
    assert res["maxs"].tolist() == pytest.approx([5.0, 9.0])


def test_counts_match_row_count():
    table = torch.zeros(7, 4)
    res = _run(table)
    assert res["counts"].tolist() == [7.0, 7.0, 7.0, 7.0]


def test_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2D"):
        EduColumnStatsNode().execute({"table": torch.zeros(5)}, {})


def test_rejects_empty_table():
    with pytest.raises(ValueError, match="zero rows"):
        EduColumnStatsNode().execute({"table": torch.zeros(0, 3)}, {})


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        EduColumnStatsNode().execute({}, {})
