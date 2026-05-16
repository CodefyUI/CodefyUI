"""Tests for MapNode (preset-driven mapping over a list)."""

from __future__ import annotations

import pytest

from app.nodes.dataflow.map_node import MapNode


def test_node_metadata():
    assert MapNode.NODE_NAME == "Map"
    assert MapNode.CATEGORY == "Data Flow"


def test_non_list_input_raises():
    with pytest.raises(ValueError, match="list"):
        MapNode().execute({"items": "not a list"}, {"subgraph": "x"})


def test_empty_subgraph_name_raises():
    with pytest.raises(ValueError, match="subgraph parameter"):
        MapNode().execute({"items": [1, 2]}, {"subgraph": ""})


def test_unknown_subgraph_raises():
    with pytest.raises(ValueError, match="not found"):
        MapNode().execute({"items": [1, 2]}, {"subgraph": "definitely_not_a_real_preset"})
