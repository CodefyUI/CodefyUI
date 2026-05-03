"""Tests for TextInputNode."""

from __future__ import annotations

from app.nodes.data.text_input_node import TextInputNode


def test_node_metadata():
    assert TextInputNode.NODE_NAME == "TextInput"
    assert TextInputNode.CATEGORY == "Data"
    assert TextInputNode.define_inputs() == []
    outputs = TextInputNode.define_outputs()
    assert len(outputs) == 1 and outputs[0].name == "text"


def test_default_value_passes_through():
    res = TextInputNode().execute({}, {"value": "hello"})
    assert res == {"text": "hello"}


def test_multiline_preserved():
    raw = "line one\nline two\n\tindented"
    res = TextInputNode().execute({}, {"value": raw})
    assert res["text"] == raw


def test_empty_value_returns_empty_string():
    res = TextInputNode().execute({}, {"value": ""})
    assert res == {"text": ""}


def test_missing_value_falls_back_to_empty():
    res = TextInputNode().execute({}, {})
    assert res == {"text": ""}


def test_non_string_value_is_coerced():
    res = TextInputNode().execute({}, {"value": 42})
    assert res == {"text": "42"}
