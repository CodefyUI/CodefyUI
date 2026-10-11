"""Tests for CSVReaderNode."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from app.nodes.data.csv_reader_node import CSVReaderNode


@pytest.fixture
def tiny_csv(tmp_path: Path) -> Path:
    """A 4-row CSV with three numeric columns + one string label column."""
    path = tmp_path / "tiny.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "c", "label"])
        w.writerow([1.0, 2.0, 3.0, "x"])
        w.writerow([4.0, 5.0, 6.0, "y"])
        w.writerow([7.0, 8.0, 9.0, "x"])
        w.writerow([10.0, 11.0, 12.0, "y"])
    return path


def _run(path, **params):
    p = {"path": str(path), "target_column": "", "include_columns": "", "skip_header": True}
    p.update(params)
    return CSVReaderNode().execute({}, p)


def test_node_metadata():
    assert CSVReaderNode.NODE_NAME == "CSVReader"
    assert CSVReaderNode.CATEGORY == "Data"
    out_names = [p.name for p in CSVReaderNode.define_outputs()]
    assert out_names == ["tensor", "labels", "columns"]


def test_loads_numeric_columns_only(tiny_csv):
    """No target_column → numeric cols become tensor, string col dropped."""
    res = _run(tiny_csv)
    assert res["tensor"].shape == (4, 3)
    # Strings can't go into the tensor; they're dropped here.
    assert res["columns"] == ["a", "b", "c"]
    # Labels list should be empty because no target_column was set.
    assert res["labels"] == []


def test_target_column_extracted_to_labels(tiny_csv):
    res = _run(tiny_csv, target_column="label")
    assert res["tensor"].shape == (4, 3)
    assert res["columns"] == ["a", "b", "c"]
    assert res["labels"] == ["x", "y", "x", "y"]


def test_include_columns_filters_features(tiny_csv):
    res = _run(tiny_csv, include_columns="a,c", target_column="label")
    assert res["tensor"].shape == (4, 2)
    assert res["columns"] == ["a", "c"]


def test_iris_sample_loads():
    """Built-in iris sample shipped with the repo."""
    iris_path = Path(__file__).resolve().parent.parent / "data" / "samples" / "iris.csv"
    if not iris_path.exists():
        pytest.skip("iris.csv sample not generated yet")
    res = _run(iris_path, target_column="species")
    assert res["tensor"].shape == (150, 4)
    assert len(res["labels"]) == 150
    assert "setosa" in res["labels"]


def test_dtype_is_float32(tiny_csv):
    res = _run(tiny_csv)
    assert res["tensor"].dtype == torch.float32


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _run(tmp_path / "does-not-exist.csv")


def test_unknown_target_column_raises(tiny_csv):
    with pytest.raises(ValueError, match="target_column"):
        _run(tiny_csv, target_column="not-a-column")


def test_unknown_include_column_raises(tiny_csv):
    with pytest.raises(ValueError, match="include_columns"):
        _run(tiny_csv, include_columns="a,zzzz")


def test_empty_csv_returns_empty_tensor(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("a,b,c\n", encoding="utf-8")
    res = _run(path)
    assert res["tensor"].shape == (0, 3)
    assert res["labels"] == []


def test_target_with_numeric_values_stringified(tmp_path):
    """Target column always emerges as string labels (for compatibility with text classifier nodes)."""
    path = tmp_path / "num.csv"
    path.write_text("x,y,target\n1,2,0\n3,4,1\n5,6,0\n", encoding="utf-8")
    res = _run(path, target_column="target")
    assert res["labels"] == ["0", "1", "0"]


# ── #709: missing target values, integral labels, NaN features ──


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "data.csv"
    path.write_text(text)
    return path


_GAPPY = (
    "x1,x2,label\n"
    "0.1,0.2,0\n"
    "0.3,0.1,1\n"
    "0.2,0.4,0\n"
    "0.5,0.7,\n"
    "0.6,0.9,1\n"
    "0.4,0.3,\n"
)


def test_missing_target_is_refused_naming_the_rows(tmp_path):
    with pytest.raises(ValueError, match=r"label.*2 row\(s\): data rows 4, 6"):
        _run(_write(tmp_path, _GAPPY), target_column="label")


def test_drop_missing_target_drops_rows_keeps_integer_labels_and_logs(tmp_path, caplog):
    warnings: list[tuple[str, str]] = []

    class _Ctx:
        def log_warning(self, kind, detail):
            warnings.append((kind, detail))

    p = {"path": str(_write(tmp_path, _GAPPY)), "target_column": "label",
         "include_columns": "", "skip_header": True, "drop_missing_target": True}
    with caplog.at_level("WARNING"):
        res = CSVReaderNode().execute({}, p, context=_Ctx())
    assert res["labels"] == ["0", "1", "0", "1"]
    assert torch.allclose(
        res["tensor"], torch.tensor([[0.1, 0.2], [0.3, 0.1], [0.2, 0.4], [0.6, 0.9]])
    )
    assert "data rows 4, 6" in caplog.text
    assert len(warnings) == 1 and "data rows 4, 6" in warnings[0][1]


def test_float_written_integer_labels_render_as_ints(tmp_path):
    res = _run(_write(tmp_path, "x,label\n1,0.0\n2,1.0\n3,2.0\n"), target_column="label")
    assert res["labels"] == ["0", "1", "2"]


def test_non_integral_float_labels_keep_their_decimals(tmp_path):
    res = _run(_write(tmp_path, "x,label\n1,0.5\n2,1.0\n"), target_column="label")
    assert res["labels"] == ["0.5", "1.0"]


def test_nan_in_a_feature_column_is_warned_by_name(tmp_path, caplog):
    path = _write(tmp_path, "x1,x2,label\n1,2,a\n,4,b\n5,6,a\n")
    with caplog.at_level("WARNING"):
        res = _run(path, target_column="label")
    assert torch.isnan(res["tensor"][1, 0])
    assert "x1 (1)" in caplog.text
    assert "x2" not in caplog.text
