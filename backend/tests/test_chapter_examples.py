"""Smoke-test every chapter plugin example by actually executing its graph.

Each chapter has at least one ``plugins/c{N}/examples/C{N}-{M}/<name>/graph.json``
that the textbook references. A broken graph wastes the student's first
attempt at running the example, so this test asserts each graph parses,
validates, and executes end-to-end without error.

Parametrised by glob so adding a new example file is enough — no test
code changes required.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from app.core.graph_engine import execute_graph, validate_graph

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLUGIN_ROOT = _REPO_ROOT / "plugins"


def _discover_chapter_graphs() -> list[Path]:
    return sorted(_PLUGIN_ROOT.glob("c?/examples/**/graph.json"))


_GRAPHS = _discover_chapter_graphs()

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
