"""#601 -- text a node inside a preset card or a block writes reaches the log.

A preset card and a block instance never run: expansion replaces each with
the nodes inside it, and ``_emit_preset_aware`` rolls their statuses up into
one status for the box the canvas draws. It dropped the inner nodes' results
on the way, and with them ``__log__``, the one result key the Execution Log
renders -- so a ``Print`` inside a box ran (its line reached the console) and
the log never showed it. The box now relays the line on a ``running`` frame
of its own, the moment the inner node finishes.

Covered here: the frames ``execute_graph`` sends, the ``node_status`` events
a run stores (what the editor shows, and what a reload replays), and the two
readers of a ``running`` frame -- the typed output builder and the headless
route's per-node timer. Every test also checks that the inner node really
ran, so none of them can pass on a build where nothing executed.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from app.api.routes_graph_run import _RunRequest, execute_contract_run
from app.core.cache import ExecutionCache
from app.core.db import Database
from app.core.graph_engine import build_preset_fallback, execute_graph
from app.core.loop_control import interrupted_result
from app.core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from app.core.node_registry import registry
from app.core.output_entries import build_node_output_entries
from app.core.run_service import EVENT_NODE_STATUS, RunService
from app.core.run_store import STATUS_SUCCEEDED, RunStore

# -- test nodes ------------------------------------------------------------------


class _InnerLogBoomNode(BaseNode):
    """Always fails: the internal that settles its box as error.

    Its one input is optional, so it can sit inside a box as a root or read
    another inner node's output (which is how a test orders it).
    """

    NODE_NAME = "_InnerLogBoom"
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
        raise RuntimeError("inner node failed on purpose")


class _InnerLogStopsEarlyNode(BaseNode):
    """Returns a partial result marked interrupted, as a stopped loop does."""

    NODE_NAME = "_InnerLogStopsEarly"
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
        return {"value": None, **interrupted_result(epoch=1)}


class _InnerLogSlowNode(BaseNode):
    """Sleeps, then passes its input through. Writes no log line."""

    NODE_NAME = "_InnerLogSlow"
    CATEGORY = "Test"
    DESCRIPTION = "Sleeps, then passes through"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [ParamDefinition(name="seconds", param_type=ParamType.FLOAT, default=0.3)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        time.sleep(float(params.get("seconds", 0.3)))
        return {"value": inputs.get("value")}


_TEST_NODES = {
    "_InnerLogBoom": _InnerLogBoomNode,
    "_InnerLogStopsEarly": _InnerLogStopsEarlyNode,
    "_InnerLogSlow": _InnerLogSlowNode,
}


@pytest.fixture(autouse=True)
def _register_test_nodes():
    registry._nodes.update(_TEST_NODES)
    yield
    for name in _TEST_NODES:
        registry._nodes.pop(name, None)


# -- graph building ----------------------------------------------------------------


def _node(node_id: str, node_type: str, **params) -> dict:
    return {
        "id": node_id,
        "type": node_type,
        "position": {"x": 0, "y": 0},
        "data": {"params": params},
    }


def _trigger(source: str, target: str) -> dict:
    return {
        "id": f"{source}->{target}",
        "source": source,
        "sourceHandle": "trigger",
        "target": target,
        "targetHandle": "__trigger",
        "type": "trigger",
    }


def _wire(source: str, source_handle: str, target: str, target_handle: str) -> dict:
    return {
        "id": f"{source}.{source_handle}->{target}.{target_handle}",
        "source": source,
        "sourceHandle": source_handle,
        "target": target,
        "targetHandle": target_handle,
        "type": "data",
    }


def _preset(name: str, nodes: list[dict], edges: list[dict], *,
            inputs: list[tuple[str, str, str]] = (),
            outputs: list[tuple[str, str, str]] = ()) -> dict:
    """A graph-owned preset; ports are ``(name, internal node, internal port)``."""
    return {
        "preset_name": name,
        "nodes": nodes,
        "edges": edges,
        "exposed_inputs": [
            {"name": n, "internal_node": node, "internal_port": port}
            for n, node, port in inputs
        ],
        "exposed_outputs": [
            {"name": n, "internal_node": node, "internal_port": port}
            for n, node, port in outputs
        ],
    }


def _inner(node_id: str, node_type: str, **params) -> dict:
    """A node inside a preset definition (presets store params flat)."""
    return {"id": node_id, "type": node_type, "params": params}


def _block(sid: str, nodes: list[dict], edges: list[dict], *,
           outputs: list[tuple[str, str, str]]) -> dict:
    """A block definition; ports are ``(port, innerNode, innerPort)``."""
    return {
        "id": sid,
        "name": sid,
        "nodes": nodes,
        "edges": edges,
        "interface": {
            "inputs": [],
            "outputs": [
                {"port": p, "innerNode": n, "innerPort": h} for p, n, h in outputs
            ],
            "triggerTargets": [],
        },
    }


#: TextInput -> Print, the shape Export as Subgraph makes of a two-node canvas.
SAY = _preset(
    "_InnerLogSay",
    [_inner("text", "TextInput", value="from-card"),
     _inner("print", "Print", label="card")],
    [{"source": "text", "sourceHandle": "text",
      "target": "print", "targetHandle": "value"}],
    outputs=[("value", "print", "value")],
)

#: The same pair as a block, the shape Collapse to subgraph makes.
SAY_BLOCK = _block(
    "_inner_log_say",
    [_node("a", "TextInput", value="from-block"),
     _node("b", "Print", label="block")],
    [_wire("a", "text", "b", "value")],
    outputs=[("out", "b", "value")],
)


def _start_then(box_id: str, box_type: str) -> tuple[list[dict], list[dict]]:
    """Start, triggering one box."""
    return (
        [_node("start", "Start"), _node(box_id, box_type)],
        [_trigger("start", box_id)],
    )


async def _frames(nodes, edges, *, presets=(), subgraphs=(), **kwargs):
    """Run once: every (node, status, data) the canvas would get, and the outputs."""
    seen: list[tuple[str, str, Any]] = []

    async def on_progress(node_id, status, data):
        if status != "progress":
            seen.append((node_id, status, data))

    outputs = await execute_graph(
        nodes, edges, on_progress=on_progress,
        preset_fallback=build_preset_fallback(list(presets)),
        subgraphs=list(subgraphs), **kwargs,
    )
    return seen, outputs


def _box(seen, box_id: str) -> list[tuple[str, Any]]:
    """The box's frames as (status, the text it carries)."""
    return [
        (status, (data or {}).get("__log__"))
        for node_id, status, data in seen
        if node_id == box_id
    ]


# -- the bug ---------------------------------------------------------------------


async def test_a_print_inside_a_preset_card_reaches_the_log_on_the_card():
    nodes, edges = _start_then("CARD", "preset:_InnerLogSay")

    seen, outputs = await _frames(nodes, edges, presets=[SAY])

    # The Print really ran: its result is in the run's outputs.
    assert outputs["CARD__print"]["__log__"] == "[card] from-card"
    # Its line goes out on the card, as text and nothing else, between the
    # card starting and the card finishing.
    assert ("CARD", "running", {"__log__": "[card] from-card"}) in seen
    assert _box(seen, "CARD") == [
        ("running", None),
        ("running", "[card] from-card"),
        ("completed", None),
    ]
    # Still a roll-up: the inner ids never reach the canvas.
    assert not any(node_id.startswith("CARD__") for node_id, _, _ in seen), seen


async def test_a_print_inside_a_block_reaches_the_log_on_the_block():
    nodes, edges = _start_then("BLK", "subgraph:_inner_log_say")

    seen, outputs = await _frames(nodes, edges, subgraphs=[SAY_BLOCK])

    assert outputs["BLK/b"]["__log__"] == "[block] from-block"
    assert _box(seen, "BLK") == [
        ("running", None),
        ("running", "[block] from-block"),
        ("completed", None),
    ]
    assert not any(node_id.startswith("BLK/") for node_id, _, _ in seen), seen


async def test_a_print_in_a_block_inside_a_block_is_logged_on_the_outer_instance():
    """The canvas draws only the outer instance; ``OUT/nest`` is no node of it."""
    inner = _block(
        "_inner_log_inner",
        [_node("a", "TextInput", value="from-nested"),
         _node("b", "Print", label="nested")],
        [_wire("a", "text", "b", "value")],
        outputs=[("out", "b", "value")],
    )
    outer = _block(
        "_inner_log_outer",
        [_node("nest", "subgraph:_inner_log_inner")],
        [],
        outputs=[("out", "nest", "out")],
    )
    nodes, edges = _start_then("OUT", "subgraph:_inner_log_outer")

    seen, outputs = await _frames(nodes, edges, subgraphs=[inner, outer])

    assert outputs["OUT/nest/b"]["__log__"] == "[nested] from-nested"
    assert ("OUT", "running", {"__log__": "[nested] from-nested"}) in seen
    assert _box(seen, "OUT")[-1] == ("completed", None)
    assert not any(node_id.startswith("OUT/") for node_id, _, _ in seen), seen


async def test_a_card_served_from_cache_still_shows_its_line():
    """A cached top-level Print re-shows its line; a cached card does too."""
    nodes, edges = _start_then("CARD", "preset:_InnerLogSay")
    cache = ExecutionCache()

    await _frames(nodes, edges, presets=[SAY], cache=cache)
    seen, _ = await _frames(nodes, edges, presets=[SAY], cache=cache)

    # Nothing inside ran this time, and the card says so...
    assert _box(seen, "CARD")[-1] == ("cached", None)
    # ...and still shows what the Print wrote, before it settles.
    assert _box(seen, "CARD") == [
        ("running", None),
        ("running", "[card] from-card"),
        ("cached", None),
    ]


#: Two Prints in the same level, so both report out of one gather.
PAIR = _preset(
    "_InnerLogPair",
    [_inner("t1", "TextInput", value="one"),
     _inner("p1", "Print", label="p1"),
     _inner("t2", "TextInput", value="two"),
     _inner("p2", "Print", label="p2")],
    [{"source": "t1", "sourceHandle": "text",
      "target": "p1", "targetHandle": "value"},
     {"source": "t2", "sourceHandle": "text",
      "target": "p2", "targetHandle": "value"}],
)


async def test_two_inner_nodes_reporting_together_settle_the_box_once():
    """One terminal frame, with every line before it, on a first and a cached run.

    A relay is an await between an inner node's count and the box's terminal
    check. Without holding the two together, the first Print counts, waits on
    its line, and finds the count full after its sibling has already settled
    the box -- 'completed' twice, or 'cached' twice on the second run.

    ``on_progress`` yields on every frame, as RunService's does (it writes each
    event under a lock, in a thread). The cached run is the deterministic half:
    no thread is involved, so the two Prints' reports always interleave there.
    """
    nodes, edges = _start_then("CARD", "preset:_InnerLogPair")
    cache = ExecutionCache()

    for settled in ("completed", "cached"):
        seen: list[tuple[str, str, Any]] = []

        async def on_progress(node_id, status, data):
            if status == "progress":
                return
            seen.append((node_id, status, (data or {}).get("__log__")))
            await asyncio.sleep(0)

        await execute_graph(nodes, edges, on_progress=on_progress, cache=cache,
                            preset_fallback=build_preset_fallback([PAIR]))

        box = [(status, line) for node_id, status, line in seen
               if node_id == "CARD"]
        assert [status for status, _ in box if status != "running"] == [settled], box
        assert box[-1] == (settled, None), box
        assert sorted(line for _, line in box if line) == ["[p1] one", "[p2] two"], box


async def test_a_line_written_before_the_box_fails_is_kept():
    """Sent live, not held for the box's last frame, so a later failure inside
    the box cannot take it down with it."""
    preset = _preset(
        "_InnerLogPrintsThenFails",
        [_inner("text", "TextInput", value="first"),
         _inner("print", "Print", label="early"),
         _inner("boom", "_InnerLogBoom")],
        [{"source": "text", "sourceHandle": "text",
          "target": "print", "targetHandle": "value"},
         # Reading the Print's output puts the failure one level after it.
         {"source": "print", "sourceHandle": "value",
          "target": "boom", "targetHandle": "value"}],
    )
    nodes, edges = _start_then("CARD", "preset:_InnerLogPrintsThenFails")

    seen, _ = await _frames(nodes, edges, presets=[preset], error_mode="continue")

    assert _box(seen, "CARD") == [
        ("running", None),
        ("running", "[early] first"),
        ("error", None),
    ]


@pytest.mark.parametrize(("settler", "settled"), [
    ("_InnerLogBoom", "error"),
    ("_InnerLogStopsEarly", "interrupted"),
])
async def test_a_box_an_internal_already_settled_is_not_repainted_running(
    settler, settled,
):
    """After 'error' or 'interrupted' the box is settled; a later line must
    not undo that.

    The editor shows a node's LATEST status, so a 'running' frame after the
    box's 'error' would put the failed box back to running -- and a failed
    run settles nothing it left running. The line is not shown in this one
    case, as before #601; the box's status is the more important fact.
    """
    preset = _preset(
        f"_InnerLogSettled{settled.title()}",
        [_inner("settler", settler),
         _inner("text", "TextInput", value="late"),
         _inner("print", "Print", label="late")],
        [{"source": "text", "sourceHandle": "text",
          "target": "print", "targetHandle": "value"}],
    )
    nodes, edges = _start_then("CARD", f"preset:{preset['preset_name']}")

    # settler and text are both roots (level 0); print reads text (level 1),
    # so the box always settles first.
    seen, outputs = await _frames(nodes, edges, presets=[preset],
                                  error_mode="continue")

    assert outputs["CARD__print"]["__log__"] == "[late] late"  # it did run
    box = [status for status, _ in _box(seen, "CARD")]
    assert box[-1] == settled, box
    assert "running" not in box[box.index(settled):], box


# -- what reads a running frame ---------------------------------------------------


def test_a_running_frame_carrying_a_line_yields_that_line_and_nothing_else():
    assert build_node_output_entries("running", {"__log__": "x"}) == [
        {"output_kind": "text", "text": "x"},
    ]
    # No media entry and no port summary: the frame is the BOX's, and a
    # summary of one inner node would overwrite the box's own.
    assert build_node_output_entries(
        "running", {"__log__": "x", "plot": "aGk=", "value": 1},
        {"image": ["plot"]},
    ) == [{"output_kind": "text", "text": "x"}]


def test_a_running_frame_without_a_line_yields_nothing():
    assert build_node_output_entries("running", None) == []
    assert build_node_output_entries("running", {"value": 1}) == []


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


async def test_a_run_stores_the_cards_line_for_the_editor_and_a_replay(
    run_store, service,
):
    """The stored ``node_status`` events are what the editor draws and replays."""
    nodes, edges = _start_then("CARD", "preset:_InnerLogSay")

    submitted = await service.submit(
        {"nodes": nodes, "edges": edges, "presets": [SAY]})
    record = await _await_terminal(run_store, submitted.run_id)

    assert record.status == STATUS_SUCCEEDED
    frames = [
        event.payload for event in await run_store.get_events(submitted.run_id)
        if event.type == EVENT_NODE_STATUS
    ]
    assert [(f["status"], f.get("outputs"))
            for f in frames if f["node_id"] == "CARD"] == [
        ("running", None),
        ("running", [{"output_kind": "text", "text": "[card] from-card"}]),
        ("completed", None),
    ]


async def test_a_run_stores_one_terminal_event_for_a_card_with_two_lines(
    run_store, service,
):
    """The race above through RunService, whose ``on_progress`` waits on a lock
    and a thread to write each event: one terminal event, both lines first."""
    nodes, edges = _start_then("CARD", "preset:_InnerLogPair")

    submitted = await service.submit(
        {"nodes": nodes, "edges": edges, "presets": [PAIR]})
    record = await _await_terminal(run_store, submitted.run_id)

    assert record.status == STATUS_SUCCEEDED
    card = [
        event.payload for event in await run_store.get_events(submitted.run_id)
        if event.type == EVENT_NODE_STATUS and event.payload["node_id"] == "CARD"
    ]
    statuses = [frame["status"] for frame in card]
    assert [s for s in statuses if s != "running"] == ["completed"], statuses
    assert statuses[-1] == "completed", statuses
    assert sorted(
        entry["text"] for frame in card for entry in frame.get("outputs", [])
    ) == ["[p1] one", "[p2] two"]


async def test_a_headless_run_times_a_card_from_its_first_running_frame():
    """A box's later 'running' frames carry text; they do not restart its clock.

    The card sleeps 0.3 s in its first node, then its Print relays a line and
    the card completes. Timed from the LAST 'running' frame the card would
    record about 0 s, which is how long the Print took, not the card.
    """
    timed = _preset(
        "_InnerLogTimed",
        [_inner("slow", "_InnerLogSlow", seconds=0.3),
         _inner("print", "Print", label="timed")],
        [{"source": "slow", "sourceHandle": "value",
          "target": "print", "targetHandle": "value"}],
        inputs=[("value", "slow", "value")],
        outputs=[("value", "print", "value")],
    )
    nodes = [
        _node("start", "Start"),
        _node("gi", "GraphInput", name="x", type="string", required=True,
              default="", description=""),
        _node("card", "preset:_InnerLogTimed"),
        _node("out", "GraphOutput", name="y", description=""),
    ]
    edges = [
        {"id": "t1", "source": "start", "target": "gi",
         "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
        _wire("gi", "value", "card", "value"),
        _wire("card", "value", "out", "value"),
    ]

    status, envelope, node_timings = await execute_contract_run(
        "inner-log-timing", nodes, edges, _RunRequest(inputs={"x": "hi"}),
        run_id="r-inner-log-timing", output_store=None,
        preset_fallback=build_preset_fallback([timed]),
    )

    assert status == 200, envelope
    assert envelope["outputs"] == {"y": "hi"}
    assert node_timings["card"] >= 0.25, node_timings
