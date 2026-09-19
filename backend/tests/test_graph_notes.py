"""A note is a canvas annotation, not a node the graph runs.

The editor serializes a note into the same ``nodes`` list every executable
node lives in -- ``{"id": ..., "type": "note", "position": ..., "data":
{"noteContent": ...}}``, with no ``params`` key at all -- because that is
the one list a graph file has. Every backend reader therefore has to know
the difference, and two of them already did: ``POST /api/graph/export``
drops notes and the edges incident to them, and ``app.core.project`` splits
a note's geometry into the layout file. ``validate_graph`` did not: it
answered ``Unknown node type: note`` for a graph the editor had just drawn,
so a shipped example carrying an explanation would fail its own smoke test.

Shipped examples are about to carry notes, so the rule is pinned here at
the three places such a graph is read: the validator, the executor, and
``run_graph.py``. ``graph_nodes`` -- the helper the example suites read
their graphs through -- is pinned in the same file, because it is the same
rule applied to a test fixture.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import run_graph
from app.core.graph_engine import execute_graph, is_note_node, validate_graph

from tests._example_graphs import graph_nodes


def _note(node_id: str = "n1", **data) -> dict:
    """A note exactly as the canvas serializes one: no ``params`` anywhere."""
    return {
        "id": node_id,
        "type": "note",
        "position": {"x": 40, "y": -120},
        "data": {"noteKind": "text", "noteContent": "why this step exists",
                 **data},
    }


# ── the predicate ─────────────────────────────────────────────────────────

def test_is_note_node_matches_the_type_the_canvas_writes():
    assert is_note_node(_note()) is True
    assert is_note_node({"id": "a", "type": "Print", "data": {"params": {}}}) is False
    # No type at all is what a malformed hand-edited file looks like; it is
    # not a note, and the unknown-type branch is the right place for it.
    assert is_note_node({"id": "a"}) is False
    # Exact, like the two readers that already existed (routes_graph.export
    # and app.core.project): ``noteNode`` is the react-flow type, which the
    # serializer never writes to disk.
    assert is_note_node({"id": "a", "type": "noteNode"}) is False
    assert is_note_node({"id": "a", "type": "Note"}) is False


def test_graph_nodes_drops_notes_and_keeps_the_rest_in_order():
    graph = {
        "nodes": [
            {"id": "start", "type": "Start", "data": {"params": {}}},
            _note("overview"),
            {"id": "print", "type": "Print", "data": {"params": {}}},
        ],
        "edges": [],
    }
    assert [n["id"] for n in graph_nodes(graph)] == ["start", "print"]
    # A graph with no nodes key at all is a file this helper is asked about
    # before anything has validated it.
    assert graph_nodes({}) == []


# ── validate_graph ────────────────────────────────────────────────────────

def test_validate_graph_accepts_a_note_beside_a_valid_graph(sample_graph):
    nodes = [*sample_graph["nodes"], _note()]
    assert validate_graph(nodes, sample_graph["edges"]) == []


def test_validate_graph_still_reports_a_genuinely_unknown_type(sample_graph):
    """The lenient branch is for notes only, not for every type it misses."""
    nodes = [
        *sample_graph["nodes"],
        {"id": "typo", "type": "Prnit", "position": {"x": 0, "y": 0},
         "data": {"params": {}}},
    ]
    errors = validate_graph(nodes, sample_graph["edges"])
    assert any("Unknown node type: Prnit" in e for e in errors), errors


def test_a_note_trips_none_of_the_other_checks(sample_graph):
    """Missing params, an absent ``data``, a binding: still nothing to say.

    A note carries no ``params`` and no ports, so every check after the
    type lookup -- required inputs, parameter ranges, reachability -- has to
    pass over it rather than read through it. These are the three shapes a
    real file holds: a bound note, a note whose optional geometry was never
    written, and (from an older editor) one with no ``data`` at all.
    """
    nodes = [
        *sample_graph["nodes"],
        _note("bound", boundToNodeId="1", boundOffset={"x": 0, "y": -90}),
        _note("bare"),
        {"id": "dataless", "type": "note", "position": {"x": 0, "y": 0}},
    ]
    assert validate_graph(nodes, sample_graph["edges"]) == []


def test_an_edge_touching_a_note_is_not_a_dangling_edge(sample_graph):
    """Parity with ``POST /api/graph/export``, which drops the same edges.

    The canvas cannot draw one -- a note renders no handles -- but a
    hand-edited or machine-written file can carry one, and the export route
    has tolerated it since notes existed. A graph that exports has to be a
    graph that validates, or the two answers disagree about the same file.
    """
    nodes = [*sample_graph["nodes"], _note()]
    edges = [
        *sample_graph["edges"],
        {"id": "e_note", "source": "1", "target": "n1",
         "sourceHandle": "value", "targetHandle": "value", "type": "data"},
    ]
    assert validate_graph(nodes, edges) == []


def test_a_graph_of_nothing_but_notes_still_has_no_entry_point():
    """The leniency stops at the type check; a note is not something to run."""
    errors = validate_graph([_note()], [])
    assert any("no entry points" in e for e in errors), errors


# ── execution ─────────────────────────────────────────────────────────────

def test_a_graph_carrying_a_note_executes_and_the_note_produces_nothing(
    sample_graph,
):
    """Expected to have worked already -- a note is unreachable -- but pinned.

    Reachability is what keeps the note out of the executable set today. It
    is not a property of notes, it is a property of this note having no
    edges, so it is asserted rather than assumed.
    """
    nodes = [*sample_graph["nodes"], _note()]
    results = asyncio.run(
        execute_graph(nodes, sample_graph["edges"], error_mode="fail_fast"))
    assert "n1" not in results
    assert results["2"]["value"] == "test"


def test_a_graph_whose_note_is_wired_runs_the_same_way(sample_graph):
    """The execution twin of the dangling-edge test above.

    An edge into a note makes the note REACHABLE, which is what put it in
    the executable set and got it looked up in the registry mid-run -- after
    the upstream nodes had already printed. Validation answering "clean" for
    a graph the run then refuses is the disagreement the validator's own
    comment rules out, so the run drops the same notes and the same edges.
    """
    nodes = [*sample_graph["nodes"], _note()]
    edges = [
        *sample_graph["edges"],
        {"id": "e_note", "source": "1", "target": "n1",
         "sourceHandle": "value", "targetHandle": "value", "type": "data"},
    ]
    results = asyncio.run(execute_graph(nodes, edges, error_mode="fail_fast"))
    assert "n1" not in results
    assert results["2"]["value"] == "test"


# ── run_graph.py ──────────────────────────────────────────────────────────

async def test_run_graph_validates_a_file_that_carries_a_note(
    tmp_path, sample_graph,
):
    """``cdui run graph.json`` on an annotated example must not exit(1).

    ``run_graph.run`` turns a validation error into ``sys.exit(1)``, so the
    assertion is that no ``SystemExit`` is raised -- the CLI is the surface
    a shipped example is checked from outside the editor.
    """
    path = tmp_path / "annotated.json"
    graph = {
        "name": "annotated",
        "nodes": [*sample_graph["nodes"], _note()],
        "edges": sample_graph["edges"],
    }
    path.write_text(json.dumps(graph), encoding="utf-8")

    await run_graph.run(str(path), validate_only=True)


async def test_run_graph_still_exits_on_a_graph_it_cannot_validate(
    tmp_path, sample_graph,
):
    """The test above would pass just as well against a CLI that validates
    nothing, so the failing side is pinned beside it."""
    path = tmp_path / "broken.json"
    graph = {
        "name": "broken",
        "nodes": [
            *sample_graph["nodes"],
            {"id": "typo", "type": "Prnit", "position": {"x": 0, "y": 0},
             "data": {"params": {}}},
        ],
        "edges": sample_graph["edges"],
    }
    path.write_text(json.dumps(graph), encoding="utf-8")

    with pytest.raises(SystemExit):
        await run_graph.run(str(path), validate_only=True)
