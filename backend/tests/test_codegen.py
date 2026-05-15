"""Tests for the Python export feature (codegen + /api/graph/export)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest


# ── Helpers ─────────────────────────────────────────────────────────


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _load_example(rel_path: str) -> dict:
    return json.loads((EXAMPLES_DIR / rel_path).read_text(encoding="utf-8"))


def _compile_check(script: str) -> None:
    """Assert the generated script parses cleanly."""
    compile(script, "<generated>", "exec")


# ── Codegen unit tests ───────────────────────────────────────────────


def test_codegen_module_imports():
    """The codegen module itself must import without SyntaxError."""
    from app.core import codegen  # noqa: F401


def test_generate_python_legacy_sequential():
    """Legacy flat-list layers spec (still on saved user graphs) must work."""
    from app.core.codegen import generate_python

    nodes = [
        {
            "id": "start",
            "type": "Start",
            "position": {"x": 0, "y": 0},
            "data": {"params": {}},
        },
        {
            "id": "model-1",
            "type": "SequentialModel",
            "position": {"x": 0, "y": 0},
            "data": {
                "params": {
                    "layers": json.dumps([
                        {"type": "Linear", "in_features": 10, "out_features": 5},
                        {"type": "ReLU"},
                        {"type": "Linear", "in_features": 5, "out_features": 2},
                    ])
                }
            },
        },
    ]
    edges = [
        {"id": "t1", "source": "start", "target": "model-1",
         "sourceHandle": "trigger", "type": "trigger"},
    ]
    order = ["start", "model-1"]
    script = generate_python(nodes, edges, order, name="legacy-test")
    _compile_check(script)
    assert "nn.Sequential(" in script
    assert "nn.Linear(in_features=10, out_features=5)" in script
    assert "nn.ReLU(inplace=True)" in script


def test_generate_python_v2_sequential_simple_chain():
    """v2 graph spec with a simple chain must emit nn.Sequential."""
    from app.core.codegen import generate_python

    v2_layers = {
        "version": 2,
        "nodes": [
            {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
            {"id": "l1", "type": "Linear", "params": {"in_features": 4, "out_features": 8}},
            {"id": "r1", "type": "ReLU"},
            {"id": "l2", "type": "Linear", "params": {"in_features": 8, "out_features": 3}},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        "edges": [
            {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "l1"},
            {"id": "e2", "source": "l1", "target": "r1"},
            {"id": "e3", "source": "r1", "target": "l2"},
            {"id": "e4", "source": "l2", "target": "out", "targetHandle": "p_y"},
        ],
    }
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
        {
            "id": "m",
            "type": "SequentialModel",
            "position": {"x": 0, "y": 0},
            "data": {"params": {"layers": json.dumps(v2_layers)}},
        },
    ]
    edges = [
        {"id": "t1", "source": "start", "target": "m",
         "sourceHandle": "trigger", "type": "trigger"},
    ]
    order = ["start", "m"]
    script = generate_python(nodes, edges, order, name="v2-chain")
    _compile_check(script)
    assert "nn.Sequential(" in script
    assert "nn.Linear(in_features=4, out_features=8)" in script
    assert "nn.Linear(in_features=8, out_features=3)" in script


def test_generate_python_inference_no_syntax_error():
    """Regression: the f-string in _gen_inference must produce a parseable script."""
    from app.core.codegen import generate_python

    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
        {
            "id": "m",
            "type": "SequentialModel",
            "position": {"x": 0, "y": 0},
            "data": {
                "params": {
                    "layers": json.dumps([
                        {"type": "Linear", "in_features": 4, "out_features": 2}
                    ])
                }
            },
        },
        {
            "id": "inf",
            "type": "Inference",
            "position": {"x": 0, "y": 0},
            "data": {"params": {"device": "cpu"}},
        },
    ]
    edges = [
        {"id": "t1", "source": "start", "target": "m",
         "sourceHandle": "trigger", "type": "trigger"},
        {"id": "e1", "source": "m", "target": "inf",
         "sourceHandle": "model", "targetHandle": "model"},
    ]
    order = ["start", "m", "inf"]
    script = generate_python(nodes, edges, order, name="inf-test")
    _compile_check(script)
    # The output line should be a valid f-string in the generated script
    assert 'print(f"Output shape: {' in script


# ── Endpoint integration test ────────────────────────────────────────


@pytest.mark.asyncio
async def test_export_endpoint_expands_presets_and_returns_runnable_script(test_client):
    """/api/graph/export must expand preset:* nodes before codegen."""
    graph = _load_example("Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json")

    resp = await test_client.post("/api/graph/export", json=graph)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    script = body["script"]

    # 1. Must parse as valid Python.
    _compile_check(script)

    # 2. Must include the actual training pieces (proves preset expansion worked).
    assert "import torch" in script
    assert "DataLoader(" in script
    assert "optim.Adam(" in script
    assert "nn.CrossEntropyLoss" in script
    assert "for epoch in range(5)" in script
    assert "torch.save(" in script

    # 3. Must NOT contain the unexpanded preset marker or a "implement manually"
    #    line for any of the nodes we know how to handle.
    assert "preset:Training Pipeline" not in script
    assert "no codegen template" not in script


@pytest.mark.asyncio
async def test_export_endpoint_validates_graph(test_client):
    """Invalid graphs must return 400 with errors."""
    bad = {
        "name": "bad",
        "nodes": [
            {"id": "1", "type": "Loss", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
            {"id": "2", "type": "DoesNotExist", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
        ],
        "edges": [],
    }
    resp = await test_client.post("/api/graph/export", json=bad)
    assert resp.status_code == 400
