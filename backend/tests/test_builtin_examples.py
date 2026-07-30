"""Smoke-test every builtin example by validating and (when fast) executing it.

The empty-canvas gallery offers every ``examples/**/graph.json`` as a
one-click starting point, so a broken graph wastes the user's very first
run. This suite asserts each builtin graph parses and validates, and
executes the fast ones end-to-end without error.

Mirrors ``test_chapter_examples.py`` (the plugin-example twin): graphs
containing dataset/training/weight-IO nodes are validated structurally but
skipped for execution — they ship as fully-working examples that users run
manually (and the underlying nodes have their own unit tests).

Parametrised by glob so adding a new example directory is enough — no test
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
_EXAMPLES_ROOT = _REPO_ROOT / "examples"


def _discover_builtin_graphs() -> list[Path]:
    return sorted(_EXAMPLES_ROOT.rglob("graph.json"))


_GRAPHS = _discover_builtin_graphs()
assert _GRAPHS, "builtin example smoke suite discovered no examples"

# Node types that pull a real dataset, train for multiple epochs, or load/save
# weights — longer than a few seconds or dependent on prior runs. Same list as
# test_chapter_examples.py. This correctly skips execution for the three
# training examples and the MNIST inference example (which needs weights from
# a prior training run); everything else — including all Model_Architecture
# graphs — must execute.
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
    ids=[p.relative_to(_EXAMPLES_ROOT).as_posix() for p in _GRAPHS],
)
def test_builtin_graph_executes(graph_path: Path):
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = payload["nodes"]
    edges = payload["edges"]

    errors = validate_graph(nodes, edges)
    assert not errors, f"validate_graph errors for {graph_path}: {errors}"

    if _is_slow(payload):
        pytest.skip(
            "Graph pulls a real dataset / trains a model / loads saved weights — "
            "validated structurally, manual run required for full execution."
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
        asyncio.run(execute_graph(nodes, edges, error_mode="fail_fast"))
    finally:
        os.chdir(prev_cwd)
