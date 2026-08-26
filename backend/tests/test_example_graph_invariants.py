"""Structural invariants every shipped example graph must hold.

``test_builtin_examples.py`` and ``test_chapter_examples.py`` each execute
their own half of the shipped graphs and assert per-example facts. This file
is the other axis: one rule, checked across every graph in both roots, for
the failure modes that are silent everywhere else.

Silent is the operative word. ``validate_graph`` accepts two edges into one
input port without a word, and the engine resolves it by last-writer-wins --
so a graph with a duplicate can validate, execute, and print a plausible
number that came from the wrong upstream node. The 2026-08-21 example audit
found exactly that in ``DQN-Atari-RL`` and ``PPO-Robotics-RL``: an
undocumented ``SequentialModel`` sitting off to one side, wired into
``optimizer-1.model`` alongside the agent's own ``model`` output, so the
optimizer in a Deep-Q-Network example may well have been handed a detached
feed-forward stack instead of the DQN. Nothing failed. Nothing warned.

These are graph-shape rules only. Whether a graph *runs* is the other two
files' job, and deliberately not repeated here.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROOTS = (_REPO_ROOT / "examples", _REPO_ROOT / "plugins")


def _discover_all_shipped_graphs() -> list[Path]:
    """Every example graph an install can put in front of a user."""
    found: list[Path] = []
    found.extend((_REPO_ROOT / "examples").rglob("graph.json"))
    found.extend((_REPO_ROOT / "plugins").glob("*/examples/**/graph.json"))
    return sorted(found)


_GRAPHS = _discover_all_shipped_graphs()
assert _GRAPHS, "example invariant suite discovered no graphs"

_IDS = [p.relative_to(_REPO_ROOT).as_posix() for p in _GRAPHS]


def _data_edges(payload: dict) -> list[dict]:
    """Edges that carry a value, so trigger wiring is out of scope.

    A trigger port legitimately takes several inbound edges -- that is how a
    Start node fans out to more than one entry point -- so counting them here
    would flag correct graphs.
    """
    return [e for e in payload.get("edges", []) if e.get("type") != "trigger"]


@pytest.mark.parametrize("graph_path", _GRAPHS, ids=_IDS)
def test_no_input_port_is_fed_by_two_edges(graph_path: Path):
    payload = json.loads(graph_path.read_text(encoding="utf-8"))

    inbound = collections.Counter(
        (e.get("target"), e.get("targetHandle")) for e in _data_edges(payload)
    )
    doubled = {port: n for port, n in inbound.items() if n > 1}

    assert not doubled, (
        f"{graph_path.name} feeds an input port from more than one source: "
        f"{sorted(doubled)}. The engine takes the last edge and drops the "
        f"rest without warning, so the graph runs and shows a number from an "
        f"upstream node the reader cannot identify. Delete the edge that does "
        f"not belong -- and the node behind it, if that leaves it orphaned."
    )


@pytest.mark.parametrize("graph_path", _GRAPHS, ids=_IDS)
def test_every_edge_endpoint_is_a_node_in_the_graph(graph_path: Path):
    """An edge to a deleted node id. Cheap to check, easy to hand-edit in.

    ``validate_graph`` reports this one, but only when someone runs the
    graph; a dangling edge in a file nobody has opened since the edit sits
    there until a student hits it.
    """
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    node_ids = {n["id"] for n in payload.get("nodes", [])}

    dangling = [
        f"{e.get('id')}.{side}={e.get(side)!r}"
        for e in payload.get("edges", [])
        for side in ("source", "target")
        if e.get(side) not in node_ids
    ]

    assert not dangling, (
        f"{graph_path.name} has edges pointing at ids that are not nodes in "
        f"the file: {dangling}"
    )
