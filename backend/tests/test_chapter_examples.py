"""Smoke-test every chapter plugin example by actually executing its graph.

Each chapter plugin has at least one ``plugins/<plugin>/examples/**/graph.json``
that the textbook references. A broken graph wastes the student's first
attempt at running the example, so this test asserts each graph parses,
validates, and executes end-to-end without error.

Parametrised by glob so adding a new example file is enough — no test
code changes required.

A second check runs over the same graphs: every node ``type`` must be an
EXACT key in the node registry. ``validate_graph`` is deliberately lenient
about that and the canvas is not — see
``test_chapter_graph_node_types_match_the_palette_exactly``.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from app.config import settings
from app.core.graph_engine import execute_graph, validate_graph
from app.core.node_registry import NodeRegistry
from app.core.plugin_loader import install_plugin_finder

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLUGIN_ROOT = _REPO_ROOT / "plugins"


def _discover_chapter_graphs() -> list[Path]:
    return sorted(_PLUGIN_ROOT.glob("*/examples/**/graph.json"))


_GRAPHS = _discover_chapter_graphs()
assert _GRAPHS, "chapter graph smoke suite discovered no plugin examples"

# Node types that pull a real dataset, train for multiple epochs, download
# weights, or otherwise take longer than a few seconds. These graphs ship as
# fully-working teaching examples but are skipped in the fast smoke test —
# students run them manually. The unit tests for the underlying nodes already
# cover correctness.
_SLOW_NODE_TYPES = {
    "Dataset",
    "DataLoader",
    "TrainingLoop",
    "preset:Training Pipeline",
    "ModelLoader",
    "ModelSaver",
    "HuggingFaceDataset",
    "KaggleDataset",
    "Inference",
    # Needs the sentence-embeddings pack; graphs are still validated. Listed
    # even though no chapter example uses it yet -- CI has no pack cache, so
    # a pack example added later would fail at the gate rather than run, and
    # a machine that DOES have the pack would load half a gigabyte of weights
    # inside the fast smoke suite.
    "TextEmbedding",
}


def _is_slow(payload: dict) -> bool:
    return any(n.get("type") in _SLOW_NODE_TYPES for n in payload.get("nodes", []))


@pytest.mark.parametrize(
    "graph_path",
    _GRAPHS,
    ids=[p.relative_to(_PLUGIN_ROOT).as_posix() for p in _GRAPHS],
)
def test_chapter_graph_executes(graph_path: Path):
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = payload["nodes"]
    edges = payload["edges"]

    errors = validate_graph(nodes, edges)
    assert not errors, f"validate_graph errors for {graph_path}: {errors}"

    if _is_slow(payload):
        pytest.skip(
            "Graph pulls a real dataset / trains a model — validated structurally, "
            "manual run required for full execution."
        )

    # Some example graphs reference data files via paths relative to the
    # backend cwd (CSVReader's default is ``data/samples/iris.csv``).
    # ``cdui test`` always launches pytest from backend/, but a contributor
    # running ``pytest`` directly from the repo root would otherwise hit a
    # FileNotFoundError. Hop cwd just for this test to keep both invocations
    # working — restore afterwards so other tests aren't affected.
    backend_dir = Path(__file__).resolve().parents[1]
    prev_cwd = Path.cwd()
    os.chdir(backend_dir)
    try:
        # execute_graph is async; pytest-asyncio is set up for async fixtures
        # but plain ``asyncio.run`` is the simplest pattern when we only need
        # a one-shot execution per graph.
        asyncio.run(execute_graph(nodes, edges, error_mode="fail_fast"))
    finally:
        os.chdir(prev_cwd)


# ── every example type is a type the palette actually has ─────────────────

#: Container node types that are not registry entries and are resolved
#: elsewhere: presets come from ``preset_registry`` (with a graph-embedded
#: fallback for presets the server has never seen), subgraph instances are
#: expanded from the graph's own ``subgraphs`` list.
_NON_REGISTRY_PREFIXES = ("preset:", "subgraph:")

#: Every pack in the repo, by manifest directory — the same auto-discovery
#: the graph glob above uses, so a new pack is covered without editing this
#: file.
_IN_REPO_PACKS = sorted(m.parent.name for m in _PLUGIN_ROOT.glob("*/cdui.plugin.toml"))


@pytest.fixture(scope="module")
def palette() -> set[str]:
    """The registry keys an install of the in-repo packs would offer.

    Built here rather than read off the session-wide registry singleton on
    purpose. The singleton's contents depend on what earlier tests did to it
    — ``test_plugin_api`` runs the app lifespan against a lockfile naming
    three packs, which drops the others — and a check about whether examples
    match the palette must not be able to pass or fail on collection order.

    It is also the point of the check: the keys are produced by the real
    discovery path, ``install_plugin_finder`` → ``discover``, so they are
    exactly the strings ``/api/nodes`` would publish. Built-in nodes are
    included because pack examples wire pack nodes to them; user custom nodes
    are not, because a shipped example must never depend on one.
    """
    reg = NodeRegistry()
    reg.discover(settings.NODES_DIR, "app.nodes")
    lockfile = {
        "schema": 1,
        "plugins": {
            pack: {"source_kind": "builtin", "source": pack, "enabled": True}
            for pack in _IN_REPO_PACKS
        },
    }
    for ns in install_plugin_finder(
        _PLUGIN_ROOT,
        _REPO_ROOT / "_phantom_user_root_for_tests",  # never read
        lockfile,
    ):
        reg.discover(ns.nodes_dir, ns.package_name, plugin_id=ns.plugin_id)

    keys = set(reg.nodes)
    assert _IN_REPO_PACKS, "found no packs under plugins/ -- the scan is broken"
    assert any(":" in k for k in keys), (
        "no pack node was discovered -- this check would pass vacuously"
    )
    return keys


@pytest.mark.parametrize(
    "graph_path",
    _GRAPHS,
    ids=[p.relative_to(_PLUGIN_ROOT).as_posix() for p in _GRAPHS],
)
def test_chapter_graph_node_types_match_the_palette_exactly(
    graph_path: Path, palette: set[str]
):
    """Every node type in a pack example is a key the palette actually has.

    ``test_chapter_graph_executes`` above does NOT cover this.
    ``registry.get`` falls back to a suffix scan, so a graph asking for a
    bare ``Edu-KNN`` while the registry holds ``foundations:Edu-KNN``
    validates and executes server-side and looks entirely healthy from here.

    The canvas has no such fallback. ``resolveSerializedNodes`` does an exact
    ``Map.get(node.type)`` over the definitions from ``/api/nodes`` and, on a
    miss, substitutes an EMPTY definition: the node renders as a blank box
    badged "Utility" with no ports at all, and because the handles do not
    exist every edge touching it is silently dropped. Nothing is logged and
    nothing fails — the student just gets a broken-looking example.

    So this guards the whole class rather than one spelling of it: any pack
    whose registry key stops agreeing with the id its examples were written
    against — a rename, a typo, or an id that the loader spells one way and
    the manifest another — fails here instead of on a student's screen.
    """
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    unresolved = sorted(
        {
            node_type
            for node in payload.get("nodes", [])
            for node_type in [str(node.get("type", ""))]
            if not node_type.startswith(_NON_REGISTRY_PREFIXES)
            and node_type not in palette
        }
    )
    assert not unresolved, (
        f"{graph_path} uses node types that are not registry keys: "
        f"{unresolved}. The canvas resolves a node type by exact match, so "
        f"each of these renders as an empty box with no ports. Plugin nodes "
        f"must be written qualified, as \"<plugin-id>:<NodeName>\" — the "
        f"plugin id exactly as its cdui.plugin.toml spells it."
    )
