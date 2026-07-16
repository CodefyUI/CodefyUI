"""Tests for the Python export endpoint and generated headless runner."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
EXAMPLE_GRAPHS = sorted(EXAMPLES_DIR.rglob("graph.json"))


def _load_example(rel_path: str) -> dict:
    return json.loads((EXAMPLES_DIR / rel_path).read_text(encoding="utf-8"))


def _compile_check(script: str) -> None:
    """Assert the generated script parses cleanly."""
    compile(script, "<generated>", "exec")


def _embedded_graph(script: str) -> dict:
    """Read GRAPH_JSON without importing or executing the generated file."""
    module = ast.parse(script)
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "GRAPH_JSON"
            for target in statement.targets
        ):
            return json.loads(ast.literal_eval(statement.value))
    raise AssertionError("generated script has no GRAPH_JSON assignment")


def _run_exported_script(
    script: str,
    tmp_path: Path,
    *args: str,
    installed_plugins: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    script_path = tmp_path / "exported_graph.py"
    script_path.write_text(script, encoding="utf-8")
    user_data_dir = tmp_path / "user-data"
    if installed_plugins:
        plugins_dir = user_data_dir / "plugins"
        plugins_dir.mkdir(parents=True, exist_ok=True)
        lockfile = {
            "schema": 1,
            "plugins": {
                plugin_id: {
                    "source_kind": "builtin",
                    "source": plugin_id,
                    "enabled": True,
                }
                for plugin_id in installed_plugins
            },
        }
        (plugins_dir / "installed.json").write_text(
            json.dumps(lockfile),
            encoding="utf-8",
        )
    env = os.environ.copy()
    env["CODEFYUI_USER_DATA_DIR"] = str(user_data_dir)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-I", str(script_path), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )


def test_codegen_module_imports():
    """The codegen module itself must import without SyntaxError."""
    from app.core import codegen  # noqa: F401


def test_generate_python_treats_all_graph_fields_as_data():
    """UI UUIDs, keywords, quotes, and Windows paths cannot become source."""
    from app.core.codegen import generate_python

    hostile_name = 'bad """ name\nnext line'
    nodes = [
        {
            "id": "class",
            "type": "Start",
            "position": {"x": 0, "y": 0},
            "data": {"params": {}},
        },
        {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "type": "FuturePluginNode",
            "position": {"x": 0, "y": 0},
            "data": {
                "params": {
                    "label": 'quote " and triple """',
                    "path": r"C:\models\weights.pt",
                }
            },
        },
    ]
    edges = [
        {
            "id": "trigger",
            "source": "class",
            "target": "123e4567-e89b-12d3-a456-426614174000",
            "sourceHandle": "trigger",
            "targetHandle": "",
            "type": "trigger",
        }
    ]

    script = generate_python(nodes, edges, list(reversed([n["id"] for n in nodes])), name=hostile_name)

    _compile_check(script)
    assert _embedded_graph(script) == {
        "name": hostile_name,
        "nodes": nodes,
        "edges": edges,
    }
    assert "no codegen template" not in script
    assert "# TODO" not in script


@pytest.mark.parametrize(
    "graph_path",
    EXAMPLE_GRAPHS,
    ids=[p.relative_to(EXAMPLES_DIR).as_posix() for p in EXAMPLE_GRAPHS],
)
@pytest.mark.asyncio
async def test_every_official_example_exports_compilable_runner(
    test_client,
    graph_path: Path,
):
    """Every shipped root example must return syntactically valid Python."""
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    response = await test_client.post("/api/graph/export", json=graph)

    assert response.status_code == 200, response.text
    script = response.json()["script"]
    _compile_check(script)
    assert "no codegen template" not in script
    assert "# TODO" not in script


@pytest.mark.asyncio
async def test_exported_runner_executes_unsupported_nodes_from_temp_cwd(
    test_client,
    tmp_path: Path,
):
    """API -> .py -> fresh subprocess executes through the real graph engine."""
    graph = {
        "name": 'runtime " graph',
        "nodes": [
            {"id": "class", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
            {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "type": "TensorCreate",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"shape": "2", "fill": "ones"}},
            },
            {
                "id": "9abc4567-e89b-12d3-a456-426614174001",
                "type": "TensorCreate",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"shape": "2", "fill": "full", "value": 2.0}},
            },
            {
                "id": "7-add",
                "type": "Add",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"alpha": 1.0}},
            },
            {
                "id": "print-output",
                "type": "Print",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"label": 'sum " C:\\tmp'}},
            },
        ],
        "edges": [
            {
                "id": "trigger-a",
                "source": "class",
                "target": "123e4567-e89b-12d3-a456-426614174000",
                "sourceHandle": "trigger",
                "targetHandle": "",
                "type": "trigger",
            },
            {
                "id": "trigger-b",
                "source": "class",
                "target": "9abc4567-e89b-12d3-a456-426614174001",
                "sourceHandle": "trigger",
                "targetHandle": "",
                "type": "trigger",
            },
            {
                "id": "a-to-add",
                "source": "123e4567-e89b-12d3-a456-426614174000",
                "target": "7-add",
                "sourceHandle": "tensor",
                "targetHandle": "tensor_a",
                "type": "data",
            },
            {
                "id": "b-to-add",
                "source": "9abc4567-e89b-12d3-a456-426614174001",
                "target": "7-add",
                "sourceHandle": "tensor",
                "targetHandle": "tensor_b",
                "type": "data",
            },
            {
                "id": "add-to-print",
                "source": "7-add",
                "target": "print-output",
                "sourceHandle": "tensor",
                "targetHandle": "value",
                "type": "data",
            },
        ],
    }

    response = await test_client.post("/api/graph/export", json=graph)
    assert response.status_code == 200, response.text

    completed = _run_exported_script(
        response.json()["script"],
        tmp_path,
        "--device",
        "cpu",
    )

    assert completed.returncode == 0, completed.stderr
    assert 'sum " C:\\tmp' in completed.stdout
    assert "tensor([3., 3.])" in completed.stdout
    assert "completed on cpu" in completed.stderr


@pytest.mark.asyncio
async def test_exported_runner_graph_input_output_json_round_trip(
    test_client,
    tmp_path: Path,
):
    graph = {
        "name": "contract-runner",
        "nodes": [
            {"id": "start", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
            {
                "id": "input",
                "type": "GraphInput",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"name": "amount", "type": "integer", "required": True, "default": "1"}},
            },
            {
                "id": "print",
                "type": "Print",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"label": "contract value"}},
            },
            {
                "id": "output",
                "type": "GraphOutput",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"name": "result"}},
            },
        ],
        "edges": [
            {
                "id": "trigger",
                "source": "start",
                "target": "input",
                "sourceHandle": "trigger",
                "targetHandle": "",
                "type": "trigger",
            },
            {
                "id": "value",
                "source": "input",
                "target": "print",
                "sourceHandle": "value",
                "targetHandle": "value",
                "type": "data",
            },
            {
                "id": "printed-value",
                "source": "print",
                "target": "output",
                "sourceHandle": "value",
                "targetHandle": "value",
                "type": "data",
            },
        ],
    }

    response = await test_client.post("/api/graph/export", json=graph)
    assert response.status_code == 200, response.text
    completed = _run_exported_script(
        response.json()["script"],
        tmp_path,
        "--device",
        "cpu",
        "--inputs-json",
        json.dumps({"amount": 7}),
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"result": 7}
    assert "[contract value] 7" in completed.stderr


@pytest.mark.asyncio
async def test_exported_runner_executes_installed_plugin_node(
    test_client,
    tmp_path: Path,
):
    """The runner discovers active plugins through the normal lockfile."""
    graph_path = (
        REPO_ROOT
        / "plugins"
        / "foundations"
        / "examples"
        / "Classical"
        / "Column-Stats-101"
        / "graph.json"
    )
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    response = await test_client.post("/api/graph/export", json=graph)
    assert response.status_code == 200, response.text

    missing = _run_exported_script(
        response.json()["script"],
        tmp_path,
        "--device",
        "cpu",
    )
    assert missing.returncode == 2
    assert "Unknown node type: foundations:Edu-ColumnStats" in missing.stderr

    completed = _run_exported_script(
        response.json()["script"],
        tmp_path,
        "--device",
        "cpu",
        installed_plugins=("foundations",),
    )
    assert completed.returncode == 0, completed.stderr
    assert "[Per-column mean]" in completed.stdout
    assert "[Per-column std]" in completed.stdout
    assert "completed on cpu" in completed.stderr


@pytest.mark.asyncio
async def test_export_endpoint_preserves_presets_for_runtime_expansion(test_client):
    graph = _load_example("Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json")
    response = await test_client.post("/api/graph/export", json=graph)

    assert response.status_code == 200, response.text
    embedded = _embedded_graph(response.json()["script"])
    embedded_types = {node["type"] for node in embedded["nodes"]}
    assert "preset:Training Pipeline" in embedded_types
    assert "TrainingLoop" not in embedded_types


@pytest.mark.asyncio
async def test_exported_runner_retains_embedded_preset_sibling_roots(
    test_client,
    tmp_path: Path,
):
    """Runtime expansion must keep a preset's untriggered sibling sources."""
    graph = {
        "name": "portable-preset-runner",
        "nodes": [
            {
                "id": "start",
                "type": "Start",
                "position": {"x": 0, "y": 0},
                "data": {"params": {}},
            },
            {
                "id": "seed",
                "type": "TensorCreate",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"shape": "2", "fill": "ones"}},
            },
            {
                "id": "portable",
                "type": "preset:Portable Add",
                "position": {"x": 0, "y": 0},
                "data": {"params": {}, "internalParams": {}},
            },
            {
                "id": "print",
                "type": "Print",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"label": "preset sum"}},
            },
        ],
        "edges": [
            {
                "id": "trigger",
                "source": "start",
                "target": "seed",
                "sourceHandle": "trigger",
                "targetHandle": "",
                "type": "trigger",
            },
            {
                "id": "seed-to-preset",
                "source": "seed",
                "target": "portable",
                "sourceHandle": "tensor",
                "targetHandle": "tensor_a",
                "type": "data",
            },
            {
                "id": "preset-to-print",
                "source": "portable",
                "target": "print",
                "sourceHandle": "tensor",
                "targetHandle": "value",
                "type": "data",
            },
        ],
        "presets": [
            {
                "preset_name": "Portable Add",
                "category": "Test",
                "description": "",
                "tags": [],
                "nodes": [
                    {
                        "id": "right",
                        "type": "TensorCreate",
                        "params": {"shape": "2", "fill": "full", "value": 2.0},
                    },
                    {
                        "id": "add",
                        "type": "Add",
                        "params": {"alpha": 1.0},
                    },
                ],
                "edges": [
                    {
                        "source": "right",
                        "sourceHandle": "tensor",
                        "target": "add",
                        "targetHandle": "tensor_b",
                    },
                ],
                "exposed_inputs": [
                    {
                        "name": "tensor_a",
                        "internal_node": "add",
                        "internal_port": "tensor_a",
                        "data_type": "TENSOR",
                        "description": "",
                    },
                ],
                "exposed_outputs": [
                    {
                        "name": "tensor",
                        "internal_node": "add",
                        "internal_port": "tensor",
                        "data_type": "TENSOR",
                        "description": "",
                    },
                ],
                "exposed_params": [],
            },
        ],
    }

    response = await test_client.post("/api/graph/export", json=graph)
    assert response.status_code == 200, response.text
    embedded = _embedded_graph(response.json()["script"])
    assert embedded["nodes"][2]["type"] == "preset:Portable Add"
    assert embedded["presets"][0]["preset_name"] == "Portable Add"

    completed = _run_exported_script(
        response.json()["script"],
        tmp_path,
        "--device",
        "cpu",
    )
    assert completed.returncode == 0, completed.stderr
    assert "[preset sum] tensor([3., 3.])" in completed.stdout


@pytest.mark.asyncio
async def test_export_ignores_disconnected_draft_cycle(
    test_client,
    tmp_path: Path,
):
    """Export preflight matches runtime pruning of disconnected drafts."""
    graph = {
        "name": "draft-cycle",
        "nodes": [
            {
                "id": "start",
                "type": "Start",
                "position": {"x": 0, "y": 0},
                "data": {"params": {}},
            },
            {
                "id": "tensor",
                "type": "TensorCreate",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"shape": "1", "fill": "ones"}},
            },
            {
                "id": "draft-a",
                "type": "Print",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"label": "draft a"}},
            },
            {
                "id": "draft-b",
                "type": "Print",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"label": "draft b"}},
            },
        ],
        "edges": [
            {
                "id": "trigger",
                "source": "start",
                "target": "tensor",
                "sourceHandle": "trigger",
                "targetHandle": "",
                "type": "trigger",
            },
            {
                "id": "draft-ab",
                "source": "draft-a",
                "target": "draft-b",
                "sourceHandle": "value",
                "targetHandle": "value",
                "type": "data",
            },
            {
                "id": "draft-ba",
                "source": "draft-b",
                "target": "draft-a",
                "sourceHandle": "value",
                "targetHandle": "value",
                "type": "data",
            },
        ],
    }

    response = await test_client.post("/api/graph/export", json=graph)
    assert response.status_code == 200, response.text
    completed = _run_exported_script(
        response.json()["script"],
        tmp_path,
        "--device",
        "cpu",
    )
    assert completed.returncode == 0, completed.stderr
    assert "draft a" not in completed.stdout
    assert "draft b" not in completed.stdout


@pytest.mark.asyncio
async def test_export_endpoint_ignores_note_nodes_and_incident_edges(test_client):
    graph = {
        "name": "annotated",
        "nodes": [
            {"id": "start", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
            {
                "id": "tensor",
                "type": "TensorCreate",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"shape": "1", "fill": "zeros"}},
            },
            {
                "id": "annotation",
                "type": "note",
                "position": {"x": 0, "y": 0},
                "data": {"noteContent": "teaching note"},
            },
        ],
        "edges": [
            {
                "id": "trigger",
                "source": "start",
                "target": "tensor",
                "sourceHandle": "trigger",
                "targetHandle": "",
                "type": "trigger",
            },
            {
                "id": "annotation-edge",
                "source": "tensor",
                "target": "annotation",
                "sourceHandle": "tensor",
                "targetHandle": "value",
                "type": "data",
            },
        ],
    }

    response = await test_client.post("/api/graph/export", json=graph)
    assert response.status_code == 200, response.text
    embedded = _embedded_graph(response.json()["script"])
    assert {node["id"] for node in embedded["nodes"]} == {"start", "tensor"}
    assert {edge["id"] for edge in embedded["edges"]} == {"trigger"}


@pytest.mark.asyncio
async def test_export_endpoint_rejects_non_note_dangling_edges(test_client):
    graph = {
        "name": "dangling",
        "nodes": [
            {
                "id": "start",
                "type": "Start",
                "position": {"x": 0, "y": 0},
                "data": {"params": {}},
            },
            {
                "id": "tensor",
                "type": "TensorCreate",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"shape": "1", "fill": "zeros"}},
            },
        ],
        "edges": [
            {
                "id": "trigger",
                "source": "start",
                "target": "tensor",
                "sourceHandle": "trigger",
                "targetHandle": "",
                "type": "trigger",
            },
            {
                "id": "dangling",
                "source": "tensor",
                "target": "ghost",
                "sourceHandle": "tensor",
                "targetHandle": "value",
                "type": "data",
            },
        ],
    }

    response = await test_client.post("/api/graph/export", json=graph)
    assert response.status_code == 400
    assert "Edge references missing node" in response.json()["detail"]


@pytest.mark.asyncio
async def test_export_endpoint_validates_graph(test_client):
    bad = {
        "name": "bad",
        "nodes": [
            {"id": "1", "type": "Loss", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
            {"id": "2", "type": "DoesNotExist", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
        ],
        "edges": [],
    }
    response = await test_client.post("/api/graph/export", json=bad)
    assert response.status_code == 400
