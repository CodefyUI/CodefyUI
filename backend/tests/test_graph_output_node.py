"""Tests for the GraphOutput node — declares a named output of the graph."""

from __future__ import annotations

from app.core.node_base import DataType
from app.nodes.io.graph_output_node import GraphOutputNode


def test_metadata():
    assert GraphOutputNode.NODE_NAME == "GraphOutput"
    assert GraphOutputNode.CATEGORY == "IO"
    assert "API" in GraphOutputNode.DESCRIPTION  # palette search finds it


def test_ports():
    inputs = GraphOutputNode.define_inputs()
    assert len(inputs) == 1
    assert inputs[0].name == "value"
    assert inputs[0].data_type == DataType.ANY
    assert inputs[0].optional is False  # missing connection -> required-input check
    assert GraphOutputNode.define_outputs() == []


def test_params():
    params = GraphOutputNode.define_params()
    names = [p.name for p in params]
    assert names == ["name", "description"]
    by_name = {p.name: p for p in params}
    assert by_name["name"].default == "output"
    assert by_name["description"].default == ""


def test_execute_passes_value_through():
    res = GraphOutputNode().execute({"value": 42}, {"name": "y"})
    assert res == {"value": 42}


def test_execute_missing_input_yields_none():
    assert GraphOutputNode().execute({}, {}) == {"value": None}


def test_registry_discovers_graph_output():
    from app.core.node_registry import registry

    assert registry.get("GraphOutput") is GraphOutputNode
