"""Export Python when a trigger reaches a preset card (#560, #561).

Collapsing a selection records every node in it that Start triggered as one
of the block's ``interface.triggerTargets``, a preset card included, and
block expansion fans Start's trigger out to ``<block>/<card>``. Preset
expansion used to replace the card with its inner nodes and leave that edge
naming the card, and Export Python, looking the card up when splitting
flows, answered HTTP 500 ``Export failed: 'ab/pq'`` (#560). Since #561 preset
expansion fans a trigger into a card out to the card's inner roots, whether
the trigger comes from a block's fan-out or straight from Start, and a
trigger edge from a node the graph does not have is refused.

Each accepted graph is checked against the engine, not against values
written down here: the exported script has to run exactly the nodes
``execute_graph`` runs and produce the same outputs.

Graphs and data are synthetic and written by the tests.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from app.config import settings

ROWS_CSV = "t10-rows.csv"
CARD_CSV = "t10-card.csv"
#: The card's Print, so a run of the card's inside shows on stderr.
CARD_LABEL = "card-ran"
OUTER_LABEL = "outer-card-ran"

CARD = "T10 Read And Show"
OUTER = "T10 Card In A Card"
ONLY_CARD = "T10 Only A Card"
BOXED = "T10 Block In A Card"

#: Torch alone takes seconds to import, and a cold Windows runner is slow.
RUN_TIMEOUT_S = 120


# -- Graphs -----------------------------------------------------------------


def _node(node_id: str, node_type: str, params: dict | None = None, **data: Any) -> dict:
    return {
        "id": node_id,
        "type": node_type,
        "position": {"x": 0, "y": 0},
        "data": {"params": dict(params or {}), **data},
    }


def _card(node_id: str, preset_name: str) -> dict:
    """A preset card the way the palette drops one."""
    return _node(node_id, f"preset:{preset_name}", internalParams={})


def _output(node_id: str, name: str) -> dict:
    return _node(node_id, "GraphOutput", {"name": name, "description": ""})


def _trigger(source: str, target: str) -> dict:
    """Start's trigger the way the canvas saves it."""
    return {
        "id": f"t-{source}-{target}",
        "source": source,
        "target": target,
        "sourceHandle": "trigger",
        "targetHandle": "__trigger",
        "type": "trigger",
    }


def _wire(source: str, source_port: str, target: str, target_port: str) -> dict:
    return {
        "id": f"{source}.{source_port}-{target}.{target_port}",
        "source": source,
        "target": target,
        "sourceHandle": source_port,
        "targetHandle": target_port,
        "type": "data",
    }


def _block(
    block_id: str,
    nodes: list[dict],
    trigger_targets: list[str],
    outputs: tuple[tuple[str, str, str], ...] = (),
) -> dict:
    """A block definition as collapse writes it: ``outputs`` are
    ``(port, inner node, inner port)``."""
    return {
        "id": block_id,
        "name": block_id,
        "description": "",
        "nodes": nodes,
        "edges": [],
        "interface": {
            "inputs": [],
            "outputs": [
                {"port": port, "innerNode": inner, "innerPort": inner_port,
                 "data_type": "TENSOR"}
                for port, inner, inner_port in outputs
            ],
            "triggerTargets": trigger_targets,
        },
    }


def _preset(name: str, nodes: list[dict], edges: list[dict], outputs: list[dict]) -> dict:
    return {
        "preset_name": name,
        "category": "Custom",
        "description": "",
        "tags": [],
        "nodes": nodes,
        "edges": edges,
        "exposed_inputs": [],
        "exposed_outputs": outputs,
        "exposed_params": [],
    }


def _card_preset() -> dict:
    """A CSVReader whose table a Print shows and hands on."""
    return _preset(
        CARD,
        nodes=[
            {"id": "csv", "type": "CSVReader", "params": {"path": CARD_CSV}},
            {"id": "peek", "type": "Print", "params": {"label": CARD_LABEL}},
        ],
        edges=[{"source": "csv", "sourceHandle": "tensor",
                "target": "peek", "targetHandle": "value"}],
        outputs=[{"name": "value", "internal_node": "peek",
                  "internal_port": "value", "data_type": "ANY",
                  "description": ""}],
    )


def _outer_preset() -> dict:
    """A preset holding a card of :func:`_card_preset` and a Print it feeds."""
    return _preset(
        OUTER,
        nodes=[
            {"id": "ic", "type": f"preset:{CARD}", "params": {}},
            {"id": "tail", "type": "Print", "params": {"label": OUTER_LABEL}},
        ],
        edges=[{"source": "ic", "sourceHandle": "value",
                "target": "tail", "targetHandle": "value"}],
        outputs=[],
    )


def _only_card_preset() -> dict:
    """A preset whose one node is a card of :func:`_card_preset`."""
    return _preset(
        ONLY_CARD,
        nodes=[{"id": "ic", "type": f"preset:{CARD}", "params": {}}],
        edges=[],
        outputs=[],
    )


def _reader_and_card_block(block_id: str = "blk") -> dict:
    """The issue's block: Start fed both nodes before they were collapsed."""
    return _block(
        block_id,
        [_node("x", "CSVReader", {"path": ROWS_CSV}), _card("pq", CARD)],
        ["x", "pq"],
        outputs=(("rows", "x", "tensor"),),
    )


def _card_in_block() -> dict:
    """The issue as the editor builds it, with the table on a GraphOutput."""
    return {
        "name": "card-in-block",
        "nodes": [_node("start", "Start"), _node("ab", "subgraph:blk"),
                  _output("out", "rows")],
        "edges": [_trigger("start", "ab"), _wire("ab", "rows", "out", "value")],
        "presets": [_card_preset()],
        "subgraphs": [_reader_and_card_block()],
    }


def _nested_blocks() -> dict:
    """The issue's block collapsed again, with a reader Start also fed."""
    inner = _block(
        "inner",
        [_node("y", "CSVReader", {"path": ROWS_CSV}), _card("pq", CARD)],
        ["y", "pq"],
    )
    outer = _block(
        "outer",
        [_node("x", "CSVReader", {"path": ROWS_CSV}),
         _node("in1", "subgraph:inner")],
        ["x", "in1"],
        outputs=(("rows", "x", "tensor"),),
    )
    return {
        "name": "nested-blocks",
        "nodes": [_node("start", "Start"), _node("o", "subgraph:outer"),
                  _output("out", "rows")],
        "edges": [_trigger("start", "o"), _wire("o", "rows", "out", "value")],
        "presets": [_card_preset()],
        "subgraphs": [outer, inner],
    }


def _card_in_card() -> dict:
    """A block whose trigger reaches a card that holds another card."""
    box = _block(
        "box",
        [_node("x", "CSVReader", {"path": ROWS_CSV}), _card("oc", OUTER)],
        ["x", "oc"],
        outputs=(("rows", "x", "tensor"),),
    )
    return {
        "name": "card-in-card",
        "nodes": [_node("start", "Start"), _node("ab", "subgraph:box"),
                  _output("out", "rows")],
        "edges": [_trigger("start", "ab"), _wire("ab", "rows", "out", "value")],
        "presets": [_card_preset(), _outer_preset()],
        "subgraphs": [box],
    }


def _card_holding_only_a_card() -> dict:
    """A block whose trigger reaches a card that holds nothing but a card.

    No node sits directly in the outer card, so the trigger naming it
    reaches the inner card's nodes only when they count toward every
    container above them, not just the nearest one.
    """
    box = _block(
        "box",
        [_node("x", "CSVReader", {"path": ROWS_CSV}), _card("oc", ONLY_CARD)],
        ["x", "oc"],
        outputs=(("rows", "x", "tensor"),),
    )
    return {
        "name": "card-holding-only-a-card",
        "nodes": [_node("start", "Start"), _node("ab", "subgraph:box"),
                  _output("out", "rows")],
        "edges": [_trigger("start", "ab"), _wire("ab", "rows", "out", "value")],
        "presets": [_card_preset(), _only_card_preset()],
        "subgraphs": [box],
    }


def _two_instances() -> dict:
    """Two instances of the issue's block, each wired from Start."""
    return {
        "name": "two-instances",
        "nodes": [_node("start", "Start"), _node("a1", "subgraph:blk"),
                  _node("a2", "subgraph:blk"), _output("out1", "first"),
                  _output("out2", "second")],
        "edges": [_trigger("start", "a1"), _trigger("start", "a2"),
                  _wire("a1", "rows", "out1", "value"),
                  _wire("a2", "rows", "out2", "value")],
        "presets": [_card_preset()],
        "subgraphs": [_reader_and_card_block()],
    }


def _start_into_a_card() -> dict:
    """Start wired straight into a card, whose table a GraphOutput reads (#561)."""
    return {
        "name": "start-into-a-card",
        "nodes": [_node("start", "Start"), _card("c", CARD),
                  _output("out", "rows")],
        "edges": [_trigger("start", "c"), _wire("c", "value", "out", "value")],
        "presets": [_card_preset()],
        "subgraphs": [],
    }


def _start_into_a_card_in_a_card() -> dict:
    """Start wired straight into a card that holds another card (#561)."""
    return {
        "name": "start-into-a-card-in-a-card",
        "nodes": [_node("start", "Start"), _card("oc", OUTER)],
        "edges": [_trigger("start", "oc")],
        "presets": [_card_preset(), _outer_preset()],
        "subgraphs": [],
    }


def _trigger_from_a_missing_node() -> dict:
    """A hand-edited file: the node a trigger came from is gone, its edge is not."""
    return {
        "name": "missing-trigger-source",
        "nodes": [_node("x", "CSVReader", {"path": ROWS_CSV}),
                  _output("out", "rows")],
        "edges": [_trigger("gone", "x"), _wire("x", "tensor", "out", "value")],
        "presets": [],
        "subgraphs": [],
    }


#: Each accepted shape, and inner nodes its run has to include, so a parity
#: check cannot pass by both sides leaving the card out.
SHAPES = {
    "card in a block": (_card_in_block, {"ab/pq__csv", "ab/pq__peek"}),
    "nested blocks": (_nested_blocks, {"o/in1/pq__csv", "o/in1/pq__peek"}),
    "card in a card": (
        _card_in_card, {"ab/oc__ic__csv", "ab/oc__ic__peek", "ab/oc__tail"}),
    "card holding only a card": (
        _card_holding_only_a_card, {"ab/oc__ic__csv", "ab/oc__ic__peek"}),
    "two instances": (
        _two_instances, {"a1/pq__peek", "a2/pq__peek"}),
    "Start into a card": (
        _start_into_a_card, {"c__csv", "c__peek", "out"}),
    "Start into a card in a card": (
        _start_into_a_card_in_a_card,
        {"oc__ic__csv", "oc__ic__peek", "oc__tail"}),
}


def _write_data_files(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ROWS_CSV).write_bytes(b"a,b\n1.5,-2.25\n3,4\n")
    (folder / CARD_CSV).write_bytes(b"c\n7\n8\n")


# -- Export, and the two runners --------------------------------------------


async def _export(client, graph: dict) -> str:
    """The script the editor's Export Python button would download."""
    response = await client.post("/api/graph/export", json=graph)
    assert response.status_code == 200, response.text
    return response.json()["script"]


def _function_node_ids(script: str) -> list[str]:
    """The node id of every function returning ``_call(...)``, by ast alone."""
    found: list[str] = []
    for statement in ast.parse(script).body:
        if not isinstance(statement, ast.FunctionDef) or not statement.body:
            continue
        last = statement.body[-1]
        if (
            isinstance(last, ast.Return)
            and isinstance(last.value, ast.Call)
            and isinstance(last.value.func, ast.Name)
            and last.value.func.id == "_call"
        ):
            found.append(ast.literal_eval(last.value.args[1]))
    return found


def _context(graph_id: str):
    from app.core.execution_context import ExecutionContext

    return ExecutionContext(device="cpu", weights_persistent=False, graph_id=graph_id)


async def _run_in_engine(graph: dict) -> dict[str, dict]:
    from app.core.graph_engine import build_preset_fallback, execute_graph

    return await execute_graph(
        graph["nodes"],
        graph["edges"],
        context=_context("t10-engine"),
        preset_fallback=build_preset_fallback(graph["presets"]),
        subgraphs=graph["subgraphs"],
    )


def _run_script_here(script: str) -> dict[str, dict]:
    """The exported file's ``run_graph``, in this process.

    The session's registry is already discovered, so the file's own
    ``initialize_runtime()`` -- a full reset and rediscovery -- is skipped.
    The grader test below runs the whole file in a fresh interpreter.
    """
    module: dict[str, Any] = {"__name__": "t10_exported_graph"}
    exec(compile(script, "<exported graph>", "exec"), module)  # noqa: S102
    module["_RT"] = module["_load_runtime"](None)
    return module["run_graph"](_context("t10-script"), {})


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right))
    return left == right


# -- The issue as filed ------------------------------------------------------


async def test_the_issue_repro_exports(test_client):
    """#560 as filed: HTTP 500 ``Export failed: 'ab/pq'``."""
    from app.core.graph_engine import build_preset_fallback, prepare_executable_graph

    trigger_input = {"name": "trigger", "internal_node": "csv",
                     "internal_port": "", "data_type": "TRIGGER",
                     "description": ""}
    preset = _preset(
        "GL",
        nodes=[{"id": "csv", "type": "CSVReader", "params": {"path": "a.csv"}}],
        edges=[],
        outputs=[],
    )
    preset["exposed_inputs"] = [trigger_input]
    graph = {
        "name": "g",
        "nodes": [_node("start", "Start"), _node("ab", "subgraph:blk")],
        "edges": [{"id": "t", "source": "start", "target": "ab",
                   "sourceHandle": "trigger", "targetHandle": "",
                   "type": "trigger"}],
        "presets": [preset],
        "subgraphs": [_block(
            "blk",
            [_node("x", "CSVReader", {"path": "c.csv"}), _card("pq", "GL")],
            ["x", "pq"],
        )],
    }

    script = await _export(test_client, graph)

    executable, _, _ = prepare_executable_graph(
        graph["nodes"], graph["edges"],
        preset_fallback=build_preset_fallback(graph["presets"]),
        subgraphs=graph["subgraphs"],
    )
    assert sorted(_function_node_ids(script)) == sorted(
        node["id"] for node in executable)


# -- Every container shape ---------------------------------------------------


@pytest.mark.parametrize("shape", list(SHAPES))
async def test_the_script_runs_exactly_what_the_engine_runs(
    shape, test_client, tmp_path: Path, monkeypatch,
):
    build, inside = SHAPES[shape]
    graph = build()
    data = tmp_path / "data"
    _write_data_files(data)
    # A bare file name resolves here first, in this process.
    monkeypatch.setattr(settings, "DATA_FILES_DIR", data)

    script = await _export(test_client, graph)
    engine = await _run_in_engine(graph)
    exported = _run_script_here(script)

    assert inside <= set(engine), sorted(engine)
    # One function per node the engine ran, and no other.
    assert sorted(_function_node_ids(script)) == sorted(engine)
    assert sorted(exported) == sorted(engine)
    differing = [
        node_id for node_id in engine
        if not _same(exported[node_id], engine[node_id])
    ]
    assert not differing, differing


@pytest.mark.parametrize(
    ("shape", "instances"),
    [
        ("card in a block", ["ab"]),
        ("nested blocks", ["o"]),
        ("card in a card", ["ab"]),
        ("card holding only a card", ["ab"]),
        ("two instances", ["a1", "a2"]),
    ],
)
def test_a_block_holding_the_card_keeps_its_function(shape, instances):
    """Start's trigger into the card ties the card to the block's flow.

    Dropping that edge instead would still run every node, but it would
    put the card's inner nodes in a flow of their own, and a block split
    across flows is emitted inline, without the function the canvas box
    stands for.
    """
    from app.core.codegen import generate_python

    graph = SHAPES[shape][0]()
    script = generate_python(
        graph["nodes"], graph["edges"], name=graph["name"],
        presets=graph["presets"], subgraphs=graph["subgraphs"],
    )

    assert "inlined" not in script
    docstrings = [
        ast.get_docstring(statement) or ""
        for statement in ast.parse(script).body
        if isinstance(statement, ast.FunctionDef)
        and statement.name.startswith("subgraph_")
    ]
    assert len(docstrings) == len(instances), docstrings
    for instance in instances:
        assert any(f"instance {instance!r}" in text for text in docstrings), (
            instance, docstrings)


# -- The downloaded file, run the way a grader runs it -----------------------


def _grader_env(home: Path) -> dict[str, str]:
    """No CODEFYUI_* variable but CodefyUI's own state, pointed into *home*."""
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.upper().startswith("CODEFYUI_")
    }
    env["CODEFYUI_USER_DATA_DIR"] = str(home / "user-data")
    env["CODEFYUI_DATA_FILES_DIR"] = str(home / "data" / "files")
    env["CODEFYUI_IMAGES_DIR"] = str(home / "data" / "images")
    env["CODEFYUI_MODELS_DIR"] = str(home / "data" / "models")
    env["CODEFYUI_MEDIA_DIR"] = str(home / "data" / "media")
    return env


async def test_the_downloaded_file_answers_what_the_engine_answers(
    test_client, tmp_path: Path, monkeypatch,
):
    """``python <file>.py`` in a folder holding the data, no arguments."""
    from app.core import api_contract

    graph = _card_in_block()
    script = await _export(test_client, graph)
    folder = tmp_path / "grader"
    _write_data_files(folder)
    monkeypatch.setattr(settings, "DATA_FILES_DIR", folder)
    engine = await _run_in_engine(graph)
    collected, missing = api_contract.collect_outputs(
        api_contract.derive_contract(graph["nodes"]), engine)
    assert not missing
    expected = json.dumps(
        {name: api_contract.serialize_output(value)
         for name, value in collected.items()},
        ensure_ascii=False,
        sort_keys=True,
    )

    (folder / "t10_graph.py").write_bytes(script.encode("utf-8"))
    completed = subprocess.run(
        [sys.executable, "-I", "t10_graph.py"],
        cwd=folder,
        env=_grader_env(tmp_path / "home"),
        capture_output=True,
        timeout=RUN_TIMEOUT_S,
        check=False,
    )
    stdout = completed.stdout.decode("utf-8")
    stderr = completed.stderr.decode("utf-8", errors="replace")

    assert completed.returncode == 0, stderr
    lines = [line for line in stdout.splitlines() if line.strip()]
    assert lines and lines[-1] == expected, (stdout, stderr)
    # The card's inside ran in the script as it did in the engine.
    assert f"[{CARD_LABEL}]" in stderr


# -- A shape the engine refuses ----------------------------------------------


async def test_a_block_inside_a_card_is_refused_with_the_engines_reason(test_client):
    """A preset cannot carry a graph-local block; the export says so, as a 400."""
    from app.core.graph_engine import (
        GraphValidationError,
        build_preset_fallback,
        prepare_executable_graph,
    )

    boxed = _preset(
        BOXED,
        nodes=[{"id": "s", "type": "subgraph:blk", "params": {}}],
        edges=[],
        outputs=[],
    )
    graph = {
        "name": "block-in-card",
        "nodes": [_node("start", "Start"),
                  _node("x", "CSVReader", {"path": ROWS_CSV}),
                  _card("bc", BOXED)],
        "edges": [_trigger("start", "x"), _trigger("start", "bc")],
        "presets": [boxed, _card_preset()],
        "subgraphs": [_reader_and_card_block()],
    }
    with pytest.raises(GraphValidationError) as refused:
        prepare_executable_graph(
            graph["nodes"], graph["edges"],
            preset_fallback=build_preset_fallback(graph["presets"]),
            subgraphs=graph["subgraphs"],
        )

    response = await test_client.post("/api/graph/export", json=graph)

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == str(refused.value)
    assert f"Preset '{BOXED}' contains subgraph instance" in str(refused.value)


async def test_a_trigger_from_a_missing_node_is_refused_with_one_sentence(test_client):
    """The edge used to make ``x`` an entry point, so the engine ran the
    graph and the export followed. Validation, the engine and the export
    now refuse it in the same words (#561)."""
    from app.core.graph_engine import GraphValidationError, validate_graph

    graph = _trigger_from_a_missing_node()
    errors = validate_graph(graph["nodes"], graph["edges"])
    # Only the missing node is pinned; the advice after " -- " may change.
    assert len(errors) == 1 and errors[0].startswith(
        "A trigger edge comes from node 'gone', which is not in the graph -- "
    ), errors
    sentence = str(errors[0])
    with pytest.raises(GraphValidationError) as refused:
        await _run_in_engine(graph)
    assert str(refused.value) == sentence
    response = await test_client.post("/api/graph/export", json=graph)
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == sentence
