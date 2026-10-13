"""#559 item 2 -- what a node inside a block or preset card produced reaches the run.

A card's ``node_status`` frames roll its nodes up, and until this change the
terminal one carried ``None``: a picture or chart drawn inside a block never
reached the run's event log, and the card's own output had no summary for its
edge to show. Now:

- each node inside a container reports what it produced on its own
  ``inner_node_status`` event, under its run id (``blk/plot``): its media by
  the ports it declares, and its port summary, never its text (the card
  relays every line already, #601). Event media needs no Record outputs;
- the card's terminal frame carries the values on its drawn output ports, so
  it is summarised like any node.

Covered: a real Visualize image inside a block with recording off, a chart in
two copies of one block and inside a preset card, the card's and a nested
card's result, a card that ran nothing, an interrupted card, a cache hit, and
``on_expanded``.
"""

from __future__ import annotations

import asyncio
import base64
import time
from typing import Any

import pytest

from app.core.cache import ExecutionCache
from app.core.db import Database
from app.core.graph_engine import execute_graph
from app.core.loop_control import interrupted_result
from app.core.node_base import MEDIA_CHART, BaseNode, DataType, PortDefinition
from app.core.node_registry import registry
from app.core.run_service import (
    EVENT_INNER_NODE_STATUS,
    EVENT_NODE_STATUS,
    RunService,
)
from app.core.run_store import STATUS_SUCCEEDED, RunStore

# -- test nodes -----------------------------------------------------------------


class _ChartNode(BaseNode):
    """Draws its input as a one-point bar chart and passes it through."""

    NODE_NAME = "_ContainerMediaChart"
    CATEGORY = "Test"
    DESCRIPTION = "A chart of its input"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY, optional=True)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="chart", data_type=DataType.ANY, media=MEDIA_CHART),
            PortDefinition(name="value", data_type=DataType.ANY),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        value = inputs.get("value", params.get("value", 0))
        return {
            "chart": {"kind": "bar", "title": f"v={value}",
                      "labels": ["v"], "values": [1.0]},
            "value": value,
            "__log__": f"charted {value}",
        }


class _StopsEarlyNode(BaseNode):
    """Returns a partial result marked interrupted, as a stopped loop does."""

    NODE_NAME = "_ContainerMediaStopsEarly"
    CATEGORY = "Test"
    DESCRIPTION = "Reports that it stopped early"
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        return {"value": 7, "__log__": "stopped at epoch 1",
                **interrupted_result(epoch=1)}


class _BoomNode(BaseNode):
    """Always fails."""

    NODE_NAME = "_ContainerMediaBoom"
    CATEGORY = "Test"
    DESCRIPTION = "Raises on every execution"
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("failed on purpose")


_TEST_NODES = {
    "_ContainerMediaChart": _ChartNode,
    "_ContainerMediaStopsEarly": _StopsEarlyNode,
    "_ContainerMediaBoom": _BoomNode,
}


@pytest.fixture(autouse=True)
def _register_test_nodes():
    registry._nodes.update(_TEST_NODES)
    yield
    for name in _TEST_NODES:
        registry._nodes.pop(name, None)


# -- graph building -------------------------------------------------------------


def _node(node_id: str, node_type: str, **params) -> dict:
    return {"id": node_id, "type": node_type, "position": {"x": 0, "y": 0},
            "data": {"params": params}}


def _trigger(source: str, target: str) -> dict:
    return {"id": f"{source}->{target}", "source": source,
            "sourceHandle": "trigger", "target": target,
            "targetHandle": "__trigger", "type": "trigger"}


def _wire(source: str, source_handle: str, target: str, target_handle: str) -> dict:
    return {"id": f"{source}.{source_handle}->{target}.{target_handle}",
            "source": source, "sourceHandle": source_handle,
            "target": target, "targetHandle": target_handle, "type": "data"}


def _block(sid: str, nodes: list[dict], edges: list[dict], *,
           inputs: list[tuple[str, str, str]] = (),
           outputs: list[tuple[str, str, str]] = ()) -> dict:
    """A block definition; ports are ``(port, innerNode, innerPort)``."""
    return {
        "id": sid, "name": sid, "nodes": nodes, "edges": edges,
        "interface": {
            "inputs": [{"port": p, "innerNode": n, "innerPort": h}
                       for p, n, h in inputs],
            "outputs": [{"port": p, "innerNode": n, "innerPort": h}
                        for p, n, h in outputs],
            "triggerTargets": [],
        },
    }


#: One chart node, fed through the block's input, exposed as `out`.
CHART = _block(
    "chart",
    [_node("c", "_ContainerMediaChart")],
    [],
    inputs=[("in", "c", "value")],
    outputs=[("out", "c", "value")],
)

#: CHART one level further in, exposed again.
HOLDER = _block(
    "holder",
    [_node("nest", "subgraph:chart")],
    [],
    inputs=[("in", "nest", "in")],
    outputs=[("out", "nest", "out")],
)


def _fed(*boxes: tuple[str, str, Any]) -> tuple[list[dict], list[dict]]:
    """Start -> TextInput(value) -> each ``(id, type, value)`` box's `in`."""
    nodes = [_node("start", "Start")]
    edges = []
    for box_id, box_type, value in boxes:
        source = f"src_{box_id}"
        nodes += [_node(source, "TextInput", value=str(value)),
                  _node(box_id, box_type)]
        edges += [_trigger("start", source), _wire(source, "text", box_id, "in")]
    return nodes, edges


async def _run(nodes, edges, *, subgraphs=(), **kwargs):
    """Every (node, status, data) both callbacks hear, in order."""
    frames: list[tuple[str, str, Any]] = []

    async def on_progress(node_id, status, data):
        frames.append((node_id, status, data))

    async def on_inner_status(node_id, container_id, status, data):
        frames.append((node_id, status, data))

    await execute_graph(nodes, edges, on_progress=on_progress,
                        on_inner_status=on_inner_status,
                        subgraphs=list(subgraphs), **kwargs)
    return frames


def _data(frames, node_id: str, status: str) -> Any:
    matches = [d for n, s, d in frames if n == node_id and s == status]
    assert len(matches) == 1, (node_id, status, frames)
    return matches[0]


# -- the engine -------------------------------------------------------------------


async def test_a_card_terminal_frame_carries_its_drawn_output_ports():
    nodes, edges = _fed(("blk", "subgraph:chart", "a"))
    frames = await _run(nodes, edges, subgraphs=[CHART])
    assert _data(frames, "blk", "completed") == {"out": "a"}
    # The inner node's own frame carries its whole result, media included.
    inner = _data(frames, "blk/c", "completed")
    assert inner["chart"]["title"] == "v=a"


async def test_a_nested_card_reports_its_own_ports_to_the_inner_listener():
    nodes, edges = _fed(("blk", "subgraph:holder", "b"))
    frames = await _run(nodes, edges, subgraphs=[CHART, HOLDER])
    assert _data(frames, "blk/nest", "completed") == {"out": "b"}
    assert _data(frames, "blk", "completed") == {"out": "b"}
    assert _data(frames, "blk/nest/c", "completed")["chart"]["title"] == "v=b"


async def test_a_card_that_ran_nothing_reports_no_result():
    nodes = [_node("start", "Start"), _node("bad", "_ContainerMediaBoom"),
             _node("blk", "subgraph:chart")]
    edges = [_trigger("start", "bad"), _wire("bad", "value", "blk", "in")]
    frames = await _run(nodes, edges, subgraphs=[CHART], error_mode="continue")
    assert _data(frames, "blk/c", "skipped") is None
    assert _data(frames, "blk", "skipped") is None


async def test_an_error_frame_still_says_what_failed():
    boom = _block("boom", [_node("x", "_ContainerMediaBoom")], [],
                  outputs=[("out", "x", "value")])
    nodes = [_node("start", "Start"), _node("blk", "subgraph:boom")]
    edges = [_trigger("start", "blk")]
    frames = await _run(nodes, edges, subgraphs=[boom], error_mode="continue")
    assert _data(frames, "blk", "error")["error_type"] == "RuntimeError"


async def test_an_interrupted_card_reports_its_ports_not_the_inner_nodes():
    early = _block("early", [_node("loop", "_ContainerMediaStopsEarly")], [],
                   outputs=[("result", "loop", "value")])
    nodes = [_node("start", "Start"), _node("blk", "subgraph:early")]
    edges = [_trigger("start", "blk")]
    frames = await _run(nodes, edges, subgraphs=[early])
    # Its own port, by its own name, and the line the node wrote -- which
    # goes out on this frame and no other.
    assert _data(frames, "blk", "interrupted") == {
        "result": 7, "__log__": "stopped at epoch 1"}


async def test_a_cache_hit_inside_a_block_carries_the_cached_result():
    nodes, edges = _fed(("blk", "subgraph:chart", "c"))
    cache = ExecutionCache()
    await _run(nodes, edges, subgraphs=[CHART], cache=cache)
    frames = await _run(nodes, edges, subgraphs=[CHART], cache=cache)
    assert _data(frames, "blk/c", "cached")["chart"]["title"] == "v=c"
    assert _data(frames, "blk", "cached") == {"out": "c"}


async def test_on_expanded_names_every_node_the_run_executes():
    nodes, edges = _fed(("blk", "subgraph:holder", "d"))
    seen: list[list[str]] = []
    await execute_graph(nodes, edges, subgraphs=[CHART, HOLDER],
                        on_expanded=lambda expanded: seen.append(
                            sorted(n["id"] for n in expanded)))
    assert seen == [["blk/nest/c", "src_blk", "start"]]


# -- what a run stores ------------------------------------------------------------


@pytest.fixture
def run_store(tmp_path):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    try:
        yield RunStore(database)
    finally:
        database.close()


@pytest.fixture
async def service(run_store):
    svc = RunService(run_store, shutdown_grace_s=2.0)
    try:
        yield svc
    finally:
        await svc.shutdown()


async def _events(run_store, service, graph: dict, **options):
    submitted = await service.submit(graph, options=options or None)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        record = await run_store.get_run(submitted.run_id)
        if record is not None and record.finished_at is not None:
            break
        await asyncio.sleep(0.02)
    else:
        raise AssertionError("the run did not finish")
    assert record.status == STATUS_SUCCEEDED, record.error
    return await run_store.get_events(submitted.run_id)


def _kinds(payload: dict) -> list[str]:
    return [entry["output_kind"] for entry in payload.get("outputs") or ()]


async def test_an_image_drawn_inside_a_block_reaches_the_run_log_without_recording(
    run_store, service,
):
    """A real Visualize, as a learner would put one inside a block."""
    viz = _block(
        "viz",
        [_node("make", "TensorCreate", shape="8", fill="full", value=1.5),
         _node("plot", "Visualize", plot_type="line", title="inside")],
        [_wire("make", "tensor", "plot", "data")],
    )
    nodes = [_node("start", "Start"), _node("blk", "subgraph:viz")]
    edges = [_trigger("start", "blk")]
    events = await _events(run_store, service, {
        "nodes": nodes, "edges": edges, "subgraphs": [viz]},
        record_outputs=False)

    inner = [e.payload for e in events if e.type == EVENT_INNER_NODE_STATUS]
    plot = [p for p in inner if p["node_id"] == "blk/plot"
            and p["status"] == "completed"]
    assert len(plot) == 1
    images = [e for e in plot[0]["outputs"] if e["output_kind"] == "image"]
    assert len(images) == 1
    assert images[0]["port"] == "image"
    assert images[0]["image"]["format"] == "png"
    assert base64.b64decode(images[0]["image"]["data"])[:8] == b"\x89PNG\r\n\x1a\n"
    assert plot[0]["container_id"] == "blk"
    # The tensor the block made is summarised on its own node.
    make = next(p for p in inner if p["node_id"] == "blk/make"
                and p["status"] == "completed")
    assert _kinds(make) == ["tensor_summary"]
    assert make["outputs"][0]["tensor_summary"]["tensor"]["mean"] == 1.5
    # The card's frames stay the card's: no picture filed under it twice.
    cards = [e.payload for e in events if e.type == EVENT_NODE_STATUS]
    assert not [p for p in cards if "image" in _kinds(p)]


async def test_two_copies_of_one_block_each_report_their_own_chart(
    run_store, service,
):
    nodes, edges = _fed(("one", "subgraph:chart", "1"),
                        ("two", "subgraph:chart", "2"))
    events = await _events(run_store, service, {
        "nodes": nodes, "edges": edges, "subgraphs": [CHART]})

    charts = {
        e.payload["node_id"]: [entry["chart"]["title"]
                               for entry in e.payload.get("outputs") or ()
                               if entry["output_kind"] == "chart"]
        for e in events
        if e.type == EVENT_INNER_NODE_STATUS and e.payload["status"] == "completed"
    }
    assert charts == {"one/c": ["v=1"], "two/c": ["v=2"]}
    # Each card's terminal frame summarises its own port.
    terminal = {
        e.payload["node_id"]: e.payload["outputs"]
        for e in events
        if e.type == EVENT_NODE_STATUS and e.payload["status"] == "completed"
        and e.payload["node_id"] in ("one", "two")
    }
    assert terminal["one"] == [{"output_kind": "tensor_summary", "tensor_summary": {
        "out": {"type": "string", "value": "1"}}}]
    assert terminal["two"][0]["tensor_summary"]["out"]["value"] == "2"


async def test_a_chart_inside_a_preset_card_reaches_the_run_log(run_store, service):
    preset = {
        "preset_name": "_ContainerMediaPreset",
        "nodes": [{"id": "c", "type": "_ContainerMediaChart", "params": {"value": 3}}],
        "edges": [],
        "exposed_inputs": [],
        "exposed_outputs": [
            {"name": "value", "internal_node": "c", "internal_port": "value"}],
    }
    nodes = [_node("start", "Start"),
             _node("CARD", "preset:_ContainerMediaPreset")]
    edges = [_trigger("start", "CARD")]
    events = await _events(run_store, service, {
        "nodes": nodes, "edges": edges, "presets": [preset]})

    inner = [e.payload for e in events if e.type == EVENT_INNER_NODE_STATUS
             and e.payload["status"] == "completed"]
    assert [(p["node_id"], p["container_id"], _kinds(p)) for p in inner] == [
        ("CARD__c", "CARD", ["chart", "tensor_summary"])]
    # The card still relays the node's line, once, on its own frames.
    lines = [entry["text"] for e in events if e.type == EVENT_NODE_STATUS
             for entry in e.payload.get("outputs") or ()
             if entry["output_kind"] == "text"]
    assert lines == ["charted 3"]
