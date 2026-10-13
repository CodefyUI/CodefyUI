"""Tests for FileReaderNode."""

from __future__ import annotations

import pytest

from app.config import settings
from app.nodes.io.file_reader_node import FileReaderNode


def test_node_metadata():
    assert FileReaderNode.NODE_NAME == "FileReader"
    assert FileReaderNode.CATEGORY == "IO"


def test_empty_path_raises():
    with pytest.raises(ValueError, match="path"):
        FileReaderNode().execute({}, {"path": ""})


def test_path_outside_allowed_dirs_raises(tmp_path):
    bad = tmp_path / "any.txt"
    bad.write_text("hi")
    with pytest.raises(ValueError, match="within the project data directories"):
        FileReaderNode().execute({}, {"path": str(bad), "mode": "text"})


def test_read_text_from_graphs_dir():
    f = settings.GRAPHS_DIR / "_test_read.txt"
    settings.GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    f.write_text("hello world")
    try:
        res = FileReaderNode().execute({}, {"path": str(f), "mode": "text"})
        assert res["text"] == "hello world"
    finally:
        f.unlink(missing_ok=True)


def test_read_csv_to_tensor():
    f = settings.GRAPHS_DIR / "_test_read.csv"
    settings.GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    f.write_text("a,b,c\n1,2,3\n4,5,6\n")
    try:
        res = FileReaderNode().execute(
            {},
            {"path": str(f), "mode": "csv", "csv_header": True},
        )
        assert res["tensor"].shape == (2, 3)
        assert res["tensor"][0, 0].item() == 1.0
        assert res["tensor"][1, 2].item() == 6.0
    finally:
        f.unlink(missing_ok=True)


def test_nonexistent_file_raises():
    p = settings.GRAPHS_DIR / "_definitely_missing.txt"
    with pytest.raises(FileNotFoundError):
        FileReaderNode().execute({}, {"path": str(p), "mode": "text"})


# -- 2026-10-13 audit: why a CSV gave no tensor ------------------------------


@pytest.fixture
def graphs_dir(tmp_path, monkeypatch):
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    monkeypatch.setattr(settings, "GRAPHS_DIR", graphs)
    return graphs


def test_ragged_csv_names_the_row_and_the_column_counts(graphs_dir):
    """A ragged CSV used to be reported as "non-numeric"."""
    f = graphs_dir / "ragged.csv"
    f.write_text("a,b,c\n1,2,3\n4,5\n")
    res = FileReaderNode().execute({}, {"path": str(f), "mode": "csv", "csv_header": True})
    assert res["tensor"].numel() == 0
    assert res["__log__"].startswith(
        "[WARN] FileReader: ragged.csv line 3 has 2 column(s), but line 2 has 3")


def test_non_numeric_csv_names_the_cell(graphs_dir):
    f = graphs_dir / "words.csv"
    f.write_text("a,b\n1,2\n3,x\n")
    res = FileReaderNode().execute({}, {"path": str(f), "mode": "csv", "csv_header": True})
    assert res["tensor"].numel() == 0
    assert "words.csv line 3, column 2 is 'x'" in res["__log__"]


def test_numeric_csv_has_no_log_line(graphs_dir):
    f = graphs_dir / "ok.csv"
    f.write_text("a,b\n1,2\n3,4\n")
    res = FileReaderNode().execute({}, {"path": str(f), "mode": "csv", "csv_header": True})
    assert res["tensor"].tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert "__log__" not in res
