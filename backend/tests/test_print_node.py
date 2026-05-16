"""Tests for PrintNode."""

from __future__ import annotations

from app.nodes.utility.print_node import PrintNode


def test_node_metadata():
    assert PrintNode.NODE_NAME == "Print"
    assert PrintNode.CATEGORY == "Utility"


def test_passes_value_through():
    res = PrintNode().execute({"value": 42}, {})
    assert res["value"] == 42


def test_log_includes_value():
    res = PrintNode().execute({"value": "hello"}, {})
    assert "__log__" in res
    assert "hello" in res["__log__"]


def test_label_prefix_in_log(capsys):
    res = PrintNode().execute({"value": "world"}, {"label": "greeting"})
    assert res["__log__"] == "[greeting] world"
    captured = capsys.readouterr()
    assert "[greeting] world" in captured.out


def test_empty_label_omits_prefix():
    res = PrintNode().execute({"value": 1}, {"label": ""})
    assert res["__log__"] == "1"


def test_passes_through_complex_objects():
    obj = {"key": [1, 2, 3]}
    res = PrintNode().execute({"value": obj}, {})
    assert res["value"] is obj


def test_handles_missing_input():
    res = PrintNode().execute({}, {})
    # value is None when not provided
    assert res["value"] is None
