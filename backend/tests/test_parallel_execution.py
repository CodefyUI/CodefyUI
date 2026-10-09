"""Tests for parallel node execution (Phase 4)."""

import time

import pytest

from app.core.graph_engine import execute_graph, topological_levels
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.node_registry import registry


def _start_node(nid="start"):
    return {"id": nid, "type": "Start", "data": {"params": {}}}


def _trigger(eid, src, tgt):
    return {"id": eid, "source": src, "target": tgt, "sourceHandle": "trigger", "type": "trigger"}


def test_topological_levels_diamond():
    """Diamond graph: A -> B, A -> C, B -> D, C -> D produces 3 levels."""
    nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}]
    edges = [
        {"source": "a", "target": "b"},
        {"source": "a", "target": "c"},
        {"source": "b", "target": "d"},
        {"source": "c", "target": "d"},
    ]
    levels = topological_levels(nodes, edges)
    assert levels[0] == ["a"]
    assert set(levels[1]) == {"b", "c"}
    assert levels[2] == ["d"]


def test_topological_levels_independent():
    """All independent nodes should be in a single level."""
    nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    edges = []
    levels = topological_levels(nodes, edges)
    assert len(levels) == 1
    assert set(levels[0]) == {"a", "b", "c"}


@pytest.mark.asyncio
async def test_parallel_execution_diamond():
    """B and C should run concurrently in a diamond graph."""

    class SlowNode(BaseNode):
        NODE_NAME = "_TestSlow"
        CATEGORY = "Test"
        DESCRIPTION = "Sleeps briefly"

        @classmethod
        def define_inputs(cls):
            # Two inputs, so D joins B and C without two wires on one input,
            # which a graph may not have (#658).
            return [
                PortDefinition(name="input", data_type=DataType.ANY, optional=True),
                PortDefinition(name="other", data_type=DataType.ANY, optional=True),
            ]

        @classmethod
        def define_outputs(cls):
            return [PortDefinition(name="output", data_type=DataType.ANY)]

        def execute(self, inputs, params):
            time.sleep(0.1)
            return {"output": params.get("id", "?")}

    registry._nodes["_TestSlow"] = SlowNode
    try:
        nodes = [
            _start_node(),
            {"id": "a", "type": "_TestSlow", "data": {"params": {"id": "a"}}},
            {"id": "b", "type": "_TestSlow", "data": {"params": {"id": "b"}}},
            {"id": "c", "type": "_TestSlow", "data": {"params": {"id": "c"}}},
            {"id": "d", "type": "_TestSlow", "data": {"params": {"id": "d"}}},
        ]
        edges = [
            _trigger("et", "start", "a"),
            {"source": "a", "target": "b", "sourceHandle": "output", "targetHandle": "input"},
            {"source": "a", "target": "c", "sourceHandle": "output", "targetHandle": "input"},
            {"source": "b", "target": "d", "sourceHandle": "output", "targetHandle": "input"},
            {"source": "c", "target": "d", "sourceHandle": "output", "targetHandle": "other"},
        ]

        t0 = time.time()
        results = await execute_graph(nodes, edges)
        elapsed = time.time() - t0

        # Sequential would be ~0.4s (4 * 0.1s). Parallel B+C should bring it to ~0.3s.
        assert elapsed < 0.38, f"Expected parallel execution, took {elapsed:.2f}s"
        assert results["d"]["output"] == "d"
    finally:
        registry._nodes.pop("_TestSlow", None)
