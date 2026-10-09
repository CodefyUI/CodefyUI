"""#559 -- a node inside a block or preset card reports its own status.

``node_status`` rolls every node inside a container up into the outermost
card the top level draws, and that stays exactly as it was. An open block
shows the nodes inside it, though, and those had nothing to show: no running
border, no error, nothing for a node the run bypassed. ``execute_graph`` now
also reports each of them to ``on_inner_status`` under the id the run gives
it (``blk/nest/mul``), and every container nested inside another rolled up
the way the card is; ``RunService`` stores those as ``inner_node_status``
events. A bypassed node is reported once, up front, as ``bypassed``.

Covered: the frames for one and two levels of nesting and for two copies of
one block, an error (fail-fast and continue), a stop, an early stop, a cache
hit, a bypass at the top level and inside a block, progress, and the events
a run stores.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from app.core.cache import ExecutionCache
from app.core.db import Database
from app.core.execution_context import CancellationError, ExecutionContext
from app.core.graph_engine import containers_between, execute_graph
from app.core.loop_control import interrupted_result
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.node_registry import registry
from app.core.run_service import (
    EVENT_INNER_NODE_STATUS,
    EVENT_NODE_STATUS,
    RunService,
)
from app.core.run_store import STATUS_SUCCEEDED, RunStore

# -- test nodes -----------------------------------------------------------------


class _InnerBoomNode(BaseNode):
    """Always fails."""

    NODE_NAME = "_InnerStatusBoom"
    CATEGORY = "Test"
    DESCRIPTION = "Raises on every execution"
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY, optional=True)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        raise KeyError("inner node failed on purpose")


class _InnerStopsEarlyNode(BaseNode):
    """Returns a partial result marked interrupted, as a stopped loop does."""

    NODE_NAME = "_InnerStatusStopsEarly"
    CATEGORY = "Test"
    DESCRIPTION = "Reports that it stopped early"
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY, optional=True)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        return {"value": None, **interrupted_result(epoch=1)}


class _InnerProgressNode(BaseNode):
    """Reports one epoch of progress, then passes its input through."""

    NODE_NAME = "_InnerStatusProgress"
    CATEGORY = "Test"
    DESCRIPTION = "Reports progress"
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY, optional=True)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs, params, progress_callback=None):
        progress_callback({"event": "epoch", "epoch": 1, "total_epochs": 1})
        return {"value": inputs.get("value")}


class _InnerStopsRunNode(BaseNode):
    """Presses Stop on the run it is part of, then completes."""

    NODE_NAME = "_InnerStatusStopsRun"
    CATEGORY = "Test"
    DESCRIPTION = "Cancels its own run"
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY, optional=True)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs, params, context=None):
        context.cancel()
        return {"value": inputs.get("value")}


_TEST_NODES = {
    "_InnerStatusBoom": _InnerBoomNode,
    "_InnerStatusStopsEarly": _InnerStopsEarlyNode,
    "_InnerStatusProgress": _InnerProgressNode,
    "_InnerStatusStopsRun": _InnerStopsRunNode,
}


@pytest.fixture(autouse=True)
def _register_test_nodes():
    registry._nodes.update(_TEST_NODES)
    yield
    for name in _TEST_NODES:
        registry._nodes.pop(name, None)


# -- graph building -------------------------------------------------------------


def _node(node_id: str, node_type: str, *, bypassed: bool = False, **params) -> dict:
    data: dict[str, Any] = {"params": params}
    if bypassed:
        data["bypassed"] = True
    return {"id": node_id, "type": node_type, "position": {"x": 0, "y": 0},
            "data": data}


def _trigger(source: str, target: str) -> dict:
    return {"id": f"{source}->{target}", "source": source,
            "sourceHandle": "trigger", "target": target,
            "targetHandle": "__trigger", "type": "trigger"}


def _wire(source: str, source_handle: str, target: str, target_handle: str) -> dict:
    return {"id": f"{source}.{source_handle}->{target}.{target_handle}",
            "source": source, "sourceHandle": source_handle,
            "target": target, "targetHandle": target_handle, "type": "data"}


def _block(sid: str, nodes: list[dict], edges: list[dict], *,
           outputs: list[tuple[str, str, str]] = ()) -> dict:
    """A block definition; output ports are ``(port, innerNode, innerPort)``."""
    return {
        "id": sid, "name": sid, "nodes": nodes, "edges": edges,
        "interface": {
            "inputs": [],
            "outputs": [{"port": p, "innerNode": n, "innerPort": h}
                        for p, n, h in outputs],
            "triggerTargets": [],
        },
    }


#: TextInput -> Print: two inner nodes, so the box settles only after both.
PAIR = _block(
    "pair",
    [_node("a", "TextInput", value="hi"), _node("b", "Print", label="p")],
    [_wire("a", "text", "b", "value")],
    outputs=[("out", "b", "value")],
)

#: A block holding one instance of PAIR and a TextInput of its own.
OUTER = _block(
    "outer",
    [_node("nest", "subgraph:pair"), _node("t", "TextInput", value="o")],
    [],
    outputs=[("out", "nest", "out")],
)


def _start_then(*boxes: tuple[str, str]) -> tuple[list[dict], list[dict]]:
    """Start, triggering each ``(id, type)`` box."""
    nodes = [_node("start", "Start")]
    edges = []
    for box_id, box_type in boxes:
        nodes.append(_node(box_id, box_type))
        edges.append(_trigger("start", box_id))
    return nodes, edges


async def _run(nodes, edges, *, subgraphs=(), inner=True, **kwargs):
    """Every (node, status) both callbacks hear, in order, and the outputs."""
    outer: list[tuple[str, str]] = []
    inner_frames: list[tuple[str, str, str, Any]] = []
    order: list[tuple[str, str]] = []

    async def on_progress(node_id, status, data):
        outer.append((node_id, status))
        order.append((node_id, status))

    async def on_inner_status(node_id, container_id, status, data):
        inner_frames.append((node_id, container_id, status, data))
        order.append((node_id, status))

    outputs = await execute_graph(
        nodes, edges, on_progress=on_progress,
        on_inner_status=on_inner_status if inner else None,
        subgraphs=list(subgraphs), **kwargs,
    )
    return outer, inner_frames, order, outputs


def _statuses(frames, node_id: str) -> list[str]:
    return [status for nid, _, status, _ in frames if nid == node_id]


# -- the chain helper -------------------------------------------------------------


def test_containers_between_lists_the_nested_ones_innermost_first():
    mapping = {"blk/nest/mul": "blk/nest", "blk/nest": "blk", "blk/a": "blk"}
    assert containers_between("blk/nest/mul", mapping) == ["blk/nest"]
    assert containers_between("blk/a", mapping) == []
    assert containers_between("start", mapping) == []
    # A ring stops at the first id seen twice, as outermost_container does.
    assert containers_between("x", {"x": "y", "y": "x"}) == []


# -- one level -------------------------------------------------------------------


async def test_each_node_inside_a_block_reports_under_its_run_id():
    nodes, edges = _start_then(("blk", "subgraph:pair"))
    outer, inner, _, _ = await _run(nodes, edges, subgraphs=[PAIR])

    assert _statuses(inner, "blk/a") == ["running", "completed"]
    assert _statuses(inner, "blk/b") == ["running", "completed"]
    assert {container for _, container, _, _ in inner} == {"blk"}
    # The card itself is reported exactly as before.
    assert [s for n, s in outer if n == "blk"] == [
        "running", "running", "completed"]  # the second carries Print's line


async def test_the_card_frames_are_the_same_with_or_without_the_inner_listener():
    nodes, edges = _start_then(("blk", "subgraph:outer"))
    with_inner, _, _, _ = await _run(nodes, edges, subgraphs=[PAIR, OUTER])
    without, _, _, _ = await _run(nodes, edges, subgraphs=[PAIR, OUTER], inner=False)
    assert with_inner == without


async def test_an_inner_frame_comes_before_the_card_frame_it_settles():
    nodes, edges = _start_then(("blk", "subgraph:pair"))
    _, _, order, _ = await _run(nodes, edges, subgraphs=[PAIR])
    assert order.index(("blk/b", "completed")) < order.index(("blk", "completed"))
    assert order.index(("blk/a", "running")) < order.index(("blk", "running"))


# -- nesting and copies ------------------------------------------------------------


async def test_a_nested_block_rolls_up_and_its_nodes_report_at_full_depth():
    nodes, edges = _start_then(("blk", "subgraph:outer"))
    outer, inner, order, _ = await _run(nodes, edges, subgraphs=[PAIR, OUTER])

    assert _statuses(inner, "blk/nest/a") == ["running", "completed"]
    assert _statuses(inner, "blk/nest/b") == ["running", "completed"]
    assert _statuses(inner, "blk/t") == ["running", "completed"]
    # The nested card: one running, one terminal, once both its nodes are done.
    assert _statuses(inner, "blk/nest") == ["running", "completed"]
    assert order.index(("blk/nest", "completed")) > order.index(
        ("blk/nest/b", "completed"))
    assert {container for _, container, _, _ in inner} == {"blk"}
    assert not any("/" in node_id for node_id, _ in outer)


async def test_two_copies_of_one_block_never_share_a_status():
    nodes, edges = _start_then(("one", "subgraph:pair"), ("two", "subgraph:pair"))
    _, inner, _, _ = await _run(nodes, edges, subgraphs=[PAIR])

    by_container = {}
    for node_id, container, _, _ in inner:
        by_container.setdefault(container, set()).add(node_id)
    assert by_container == {"one": {"one/a", "one/b"}, "two": {"two/a", "two/b"}}


# -- failure, stop, early stop, cache ---------------------------------------------


BOOM = _block(
    "boom",
    [_node("ok", "TextInput", value="x"), _node("bad", "_InnerStatusBoom"),
     _node("after", "Print", label="z")],
    [_wire("ok", "text", "bad", "value"), _wire("bad", "value", "after", "value")],
)


async def test_a_failing_inner_node_carries_its_own_error():
    nodes, edges = _start_then(("blk", "subgraph:boom"))
    with pytest.raises(KeyError):
        await _run(nodes, edges, subgraphs=[BOOM])

    seen: list[tuple[str, str, Any]] = []

    async def on_inner_status(node_id, container_id, status, data):
        seen.append((node_id, status, data))

    with pytest.raises(KeyError):
        await execute_graph(nodes, edges, on_inner_status=on_inner_status,
                            subgraphs=[BOOM])
    errors = [(n, d) for n, s, d in seen if s == "error"]
    assert [n for n, _ in errors] == ["blk/bad"]
    assert errors[0][1]["error_type"] == "KeyError"
    # Fail-fast: nothing after the failure reports anything.
    assert not [n for n, _, _ in seen if n == "blk/after"]


async def test_continue_mode_reports_the_node_after_a_failure_as_skipped():
    nodes, edges = _start_then(("blk", "subgraph:boom"))
    outer, inner, _, _ = await _run(nodes, edges, subgraphs=[BOOM],
                                    error_mode="continue")
    assert _statuses(inner, "blk/ok") == ["running", "completed"]
    assert _statuses(inner, "blk/bad") == ["running", "error"]
    assert _statuses(inner, "blk/after") == ["skipped"]
    assert [s for n, s in outer if n == "blk"] == ["running", "error"]


async def test_a_stop_leaves_the_rest_of_the_block_unreported():
    stops = _block(
        "stops",
        [_node("first", "_InnerStatusStopsRun"), _node("second", "Print", label="s")],
        [_wire("first", "value", "second", "value")],
    )
    nodes, edges = _start_then(("blk", "subgraph:stops"))
    context = ExecutionContext()
    outer: list[tuple[str, str]] = []
    inner: list[tuple[str, str]] = []

    async def on_progress(node_id, status, data):
        outer.append((node_id, status))

    async def on_inner_status(node_id, container_id, status, data):
        inner.append((node_id, status))

    with pytest.raises(CancellationError):
        await execute_graph(nodes, edges, on_progress=on_progress,
                            on_inner_status=on_inner_status,
                            context=context, subgraphs=[stops])
    assert inner == [("blk/first", "running"), ("blk/first", "completed")]
    # The card never settles: the client settles what a stop left running.
    assert [s for n, s in outer if n == "blk"] == ["running"]


async def test_an_inner_node_that_stopped_early_is_interrupted_at_every_level():
    early = _block("early", [_node("loop", "_InnerStatusStopsEarly")], [])
    holder = _block("holder", [_node("nest", "subgraph:early")], [])
    nodes, edges = _start_then(("blk", "subgraph:holder"))
    outer, inner, _, _ = await _run(nodes, edges, subgraphs=[early, holder])
    assert _statuses(inner, "blk/nest/loop") == ["running", "interrupted"]
    assert _statuses(inner, "blk/nest") == ["running", "interrupted"]
    assert [s for n, s in outer if n == "blk"] == ["running", "interrupted"]


async def test_a_cache_hit_inside_a_nested_block_reports_cached_at_every_level():
    cached_pair = _block(
        "cpair",
        [_node("a", "TextInput", value="hi"), _node("b", "TextInput", value="yo")],
        [],
    )
    holder = _block("cholder", [_node("nest", "subgraph:cpair")], [])
    nodes, edges = _start_then(("blk", "subgraph:cholder"))
    cache = ExecutionCache()
    await _run(nodes, edges, subgraphs=[cached_pair, holder], cache=cache)
    outer, inner, _, _ = await _run(nodes, edges, subgraphs=[cached_pair, holder],
                                    cache=cache)
    assert _statuses(inner, "blk/nest/a") == ["cached"]
    assert _statuses(inner, "blk/nest/b") == ["cached"]
    assert _statuses(inner, "blk/nest") == ["running", "cached"]
    assert [s for n, s in outer if n == "blk"] == ["running", "cached"]


# -- bypass ---------------------------------------------------------------------------


async def test_a_bypassed_node_is_reported_before_anything_runs():
    nodes = [
        _node("start", "Start"),
        _node("text", "TextInput", value="hi"),
        _node("muted", "Print", bypassed=True, label="m"),
        _node("show", "Print", label="s"),
    ]
    edges = [_trigger("start", "text"), _wire("text", "text", "muted", "value"),
             _wire("muted", "value", "show", "value")]
    outer, _, _, _ = await _run(nodes, edges)
    assert outer[0] == ("muted", "bypassed")
    assert [s for n, s in outer if n == "muted"] == ["bypassed"]
    assert ("show", "completed") in outer


async def test_a_bypassed_node_inside_a_block_is_reported_to_the_inner_listener():
    muted = _block(
        "muted",
        [_node("a", "TextInput", value="hi"),
         _node("m", "Print", bypassed=True, label="m"),
         _node("b", "Print", label="b")],
        [_wire("a", "text", "m", "value"), _wire("m", "value", "b", "value")],
    )
    nodes, edges = _start_then(("blk", "subgraph:muted"))
    outer, inner, order, _ = await _run(nodes, edges, subgraphs=[muted])
    assert ("blk/m", "blk", "bypassed", None) in inner
    assert order[0] == ("blk/m", "bypassed")
    # Never counted toward the card: it settles once the two that ran are done.
    assert [s for n, s in outer if n == "blk"][-1] == "completed"
    assert not any(n == "blk/m" for n, _ in outer)


async def test_without_the_inner_listener_an_inner_bypass_says_nothing():
    muted = _block(
        "muted2",
        [_node("a", "TextInput", value="hi"),
         _node("m", "Print", bypassed=True, label="m")],
        [_wire("a", "text", "m", "value")],
    )
    nodes, edges = _start_then(("blk", "subgraph:muted2"))
    outer, _, _, _ = await _run(nodes, edges, subgraphs=[muted], inner=False)
    assert not any("/" in n for n, _ in outer)


# -- progress ---------------------------------------------------------------------------


PROGRESS = _block("prog", [_node("train", "_InnerStatusProgress")], [])


async def test_inner_progress_goes_to_the_inner_listener_with_the_inner_id():
    nodes, edges = _start_then(("blk", "subgraph:prog"))
    outer, inner, _, _ = await _run(nodes, edges, subgraphs=[PROGRESS])
    progress = [(n, c, d) for n, c, s, d in inner if s == "progress"]
    assert progress == [("blk/train", "blk",
                         {"event": "epoch", "epoch": 1, "total_epochs": 1})]
    assert not any(s == "progress" for _, s in outer)


async def test_without_the_inner_listener_progress_goes_to_the_card():
    nodes, edges = _start_then(("blk", "subgraph:prog"))
    outer, _, _, _ = await _run(nodes, edges, subgraphs=[PROGRESS], inner=False)
    assert ("blk", "progress") in outer


# -- what a run stores ------------------------------------------------------------------


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


async def _await_terminal(store: RunStore, run_id: str, *, timeout: float = 15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = await store.get_run(run_id)
        assert record is not None, f"run {run_id} vanished"
        if record.finished_at is not None:
            return record
        await asyncio.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish within {timeout}s")


async def test_a_run_stores_inner_events_beside_the_unchanged_card_events(
    run_store, service,
):
    nodes, edges = _start_then(("blk", "subgraph:outer"), ("p", "subgraph:prog"))
    submitted = await service.submit(
        {"nodes": nodes, "edges": edges, "subgraphs": [PAIR, OUTER, PROGRESS]})
    record = await _await_terminal(run_store, submitted.run_id)
    assert record.status == STATUS_SUCCEEDED

    events = await run_store.get_events(submitted.run_id)
    inner = [e.payload for e in events if e.type == EVENT_INNER_NODE_STATUS]
    cards = [e.payload for e in events if e.type == EVENT_NODE_STATUS]

    assert {"node_id": "blk/nest/b", "container_id": "blk",
            "status": "completed"} in inner
    assert {"node_id": "blk/nest", "container_id": "blk",
            "status": "completed"} in inner
    # No outputs on an inner event: the card's own frames carry them.
    assert all(set(p) <= {"node_id", "container_id", "status", "error",
                          "error_type"} for p in inner)
    assert not any("/" in p["node_id"] for p in cards)
    # Progress is ONE event: the card's, naming the node inside it.
    progress = [p for p in cards if p["status"] == "progress"]
    assert len(progress) == 1
    assert progress[0]["node_id"] == "p"
    assert progress[0]["inner_node_id"] == "p/train"
    assert progress[0]["outputs"][0]["progress"]["epoch"] == 1


async def test_a_stored_inner_error_names_the_failing_node(run_store, service):
    nodes, edges = _start_then(("blk", "subgraph:boom"))
    submitted = await service.submit(
        {"nodes": nodes, "edges": edges, "subgraphs": [BOOM]})
    await _await_terminal(run_store, submitted.run_id)
    events = await run_store.get_events(submitted.run_id)
    errors = [e.payload for e in events
              if e.type == EVENT_INNER_NODE_STATUS and e.payload["status"] == "error"]
    assert len(errors) == 1
    assert errors[0]["node_id"] == "blk/bad"
    assert errors[0]["error_type"] == "KeyError"
    assert "failed on purpose" in errors[0]["error"]
