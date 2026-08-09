"""Tests for the Python export endpoint and generated node-function runner."""

from __future__ import annotations

import ast
import json
import os
import re
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


def _node_functions(script: str) -> list[tuple[str, str, str]]:
    """Return ``(func_name, node_id, node_type)`` per generated node function.

    Reads the AST without importing or executing the generated file; entries
    come back in file order (grouped by flow, sequence number in the name).
    """
    module = ast.parse(script)
    found: list[tuple[str, str, str]] = []
    for statement in module.body:
        if not isinstance(statement, ast.FunctionDef):
            continue
        if not re.fullmatch(r"n\d+_\w+", statement.name):
            continue
        for call in ast.walk(statement):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_call"
            ):
                node_type = call.args[0].value
                node_id = call.args[1].value
                found.append((statement.name, node_id, node_type))
                break
    return found


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
    hostile_label = 'quote " and triple """'
    hostile_path = r"C:\models\weights.pt"
    nodes = [
        {
            "id": "class",
            "type": "Start",
            "position": {"x": 0, "y": 0},
            "data": {"params": {}},
        },
        {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "type": "_TestSource",
            "position": {"x": 0, "y": 0},
            "data": {
                "params": {
                    "label": hostile_label,
                    "path": hostile_path,
                    "val": "import os  # data, not code",
                }
            },
        },
        # These two ids collide after sanitization ("x-1" and "x_1" both
        # slug to x_1) — the generator must dedupe the flow locals.
        {
            "id": "x-1",
            "type": "_TestSource",
            "position": {"x": 0, "y": 0},
            "data": {"params": {"val": 'x"""1'}},
        },
        {
            "id": "x_1",
            "type": "Print",
            "position": {"x": 0, "y": 0},
            "data": {"params": {"label": "collide"}},
        },
    ]
    edges = [
        {
            "id": "t1",
            "source": "class",
            "target": "123e4567-e89b-12d3-a456-426614174000",
            "sourceHandle": "trigger",
            "targetHandle": "",
            "type": "trigger",
        },
        {
            "id": "t2",
            "source": "class",
            "target": "x-1",
            "sourceHandle": "trigger",
            "targetHandle": "",
            "type": "trigger",
        },
        {
            "id": "d1",
            "source": "x-1",
            "target": "x_1",
            "sourceHandle": "value",
            "targetHandle": "value",
            "type": "data",
        },
    ]

    script = generate_python(nodes, edges, name=hostile_name)

    _compile_check(script)
    assert "GRAPH_JSON" not in script
    assert "no codegen template" not in script
    assert "# TODO" not in script
    # Hostile strings survive only as escaped literals, never as source.
    assert ascii(hostile_name) in script
    assert ascii(hostile_label) in script
    assert ascii(hostile_path) in script
    assert ascii("import os  # data, not code") in script
    funcs = _node_functions(script)
    assert {node_id for _, node_id, _ in funcs} == {n["id"] for n in nodes}
    # Results stay keyed by the REAL node ids...
    assert "results['class']" in script
    assert "results['x-1']" in script
    assert "results['x_1']" in script
    # ...while the flow locals are sanitized, deduped identifiers.
    assert "x_1_2 = results['x_1']" in script


def test_generate_python_sanitizes_hostile_port_names(monkeypatch):
    """Port names that are not valid identifiers stay literal dict keys."""
    from app.core.codegen import generate_python
    from app.core.node_base import BaseNode, DataType, PortDefinition
    from app.core.node_registry import registry

    class _HostilePortsNode(BaseNode):
        NODE_NAME = "_HostilePorts"
        CATEGORY = "Test"
        DESCRIPTION = "Ports that are not valid Python identifiers"

        @classmethod
        def define_inputs(cls) -> list[PortDefinition]:
            return [
                PortDefinition(
                    name="weird port!", data_type=DataType.ANY, optional=True
                ),
                PortDefinition(
                    name="class", data_type=DataType.ANY, optional=True
                ),
                PortDefinition(
                    name="123 go", data_type=DataType.ANY, optional=True
                ),
            ]

        @classmethod
        def define_outputs(cls) -> list[PortDefinition]:
            return [PortDefinition(name="out port", data_type=DataType.ANY)]

        def execute(self, inputs, params):
            return {"out port": inputs.get("weird port!")}

    monkeypatch.setitem(registry._nodes, "_HostilePorts", _HostilePortsNode)

    nodes = [
        {
            "id": "start",
            "type": "Start",
            "position": {"x": 0, "y": 0},
            "data": {"params": {}},
        },
        {
            "id": "src",
            "type": "_TestSource",
            "position": {"x": 0, "y": 0},
            "data": {"params": {}},
        },
        {
            "id": "weird",
            "type": "_HostilePorts",
            "position": {"x": 0, "y": 0},
            "data": {"params": {}},
        },
    ]
    edges = [
        {
            "id": "t1",
            "source": "start",
            "target": "src",
            "sourceHandle": "trigger",
            "targetHandle": "",
            "type": "trigger",
        },
        {
            "id": "d1",
            "source": "src",
            "target": "weird",
            "sourceHandle": "value",
            "targetHandle": "weird port!",
            "type": "data",
        },
        {
            "id": "d2",
            "source": "src",
            "target": "weird",
            "sourceHandle": "value",
            "targetHandle": "class",
            "type": "data",
        },
        {
            "id": "d3",
            "source": "src",
            "target": "weird",
            "sourceHandle": "value",
            "targetHandle": "123 go",
            "type": "data",
        },
    ]

    script = generate_python(nodes, edges, name="hostile ports")

    _compile_check(script)
    # Function parameters are sanitized identifiers; the inputs dict keys
    # are the real port-name strings.
    assert "weird_port=_ABSENT" in script
    assert "class_=_ABSENT" in script
    assert "v_123_go=_ABSENT" in script
    assert "'weird port!': weird_port" in script
    assert "'class': class_" in script
    assert "'123 go': v_123_go" in script


def test_generated_script_structure_and_flow_grouping():
    """One def per node, per-flow functions, run_graph/main, no GRAPH_JSON."""
    from app.core.codegen import generate_python

    nodes = [
        {"id": "start1", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
        {
            "id": "a",
            "type": "TensorCreate",
            "position": {"x": 0, "y": 0},
            "data": {"params": {"shape": "1", "fill": "ones"}},
        },
        {
            "id": "p",
            "type": "Print",
            "position": {"x": 0, "y": 0},
            "data": {"params": {"label": "flow one"}},
        },
        {"id": "start2", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
        {
            "id": "c",
            "type": "TensorCreate",
            "position": {"x": 0, "y": 0},
            "data": {"params": {"shape": "1", "fill": "zeros"}},
        },
    ]
    edges = [
        {
            "id": "t1",
            "source": "start1",
            "target": "a",
            "sourceHandle": "trigger",
            "targetHandle": "",
            "type": "trigger",
        },
        {
            "id": "d1",
            "source": "a",
            "target": "p",
            "sourceHandle": "tensor",
            "targetHandle": "value",
            "type": "data",
        },
        {
            "id": "t2",
            "source": "start2",
            "target": "c",
            "sourceHandle": "trigger",
            "targetHandle": "",
            "type": "trigger",
        },
    ]

    script = generate_python(nodes, edges, name="structure")

    _compile_check(script)
    assert "GRAPH_JSON" not in script
    names = [
        statement.name
        for statement in ast.parse(script).body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    node_funcs = [name for name in names if re.fullmatch(r"n\d+_\w+", name)]
    assert len(node_funcs) == len(nodes)
    for expected in ("flow_1", "flow_2", "run_graph", "_parser", "_run", "main"):
        assert expected in names
    assert "flow_3" not in names

    # Weakly-connected components group into flows, first-seen order.
    flow_1_src = script[script.index("def flow_1") : script.index("def flow_2")]
    assert "results['start1']" in flow_1_src
    assert "results['a']" in flow_1_src
    assert "results['p']" in flow_1_src
    flow_2_src = script[script.index("def flow_2") : script.index("def run_graph")]
    assert "results['start2']" in flow_2_src
    assert "results['c']" in flow_2_src

    run_graph_src = script[script.index("def run_graph") :]
    assert run_graph_src.index("flow_1(ctx, results, provided)") < run_graph_src.index(
        "flow_2(ctx, results, provided)"
    )


def test_node_function_sequence_matches_engine_topological_sort():
    """The nNN sequence numbers replay exactly the engine's execution order."""
    from app.core.codegen import generate_python
    from app.core.graph_engine import prepare_executable_graph, topological_sort

    nodes = [
        {"id": "s1", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
        {
            "id": "a1",
            "type": "TensorCreate",
            "position": {"x": 0, "y": 0},
            "data": {"params": {"shape": "1", "fill": "ones"}},
        },
        {"id": "s2", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
        {
            "id": "b1",
            "type": "TensorCreate",
            "position": {"x": 0, "y": 0},
            "data": {"params": {"shape": "1", "fill": "zeros"}},
        },
        {
            "id": "a2",
            "type": "Print",
            "position": {"x": 0, "y": 0},
            "data": {"params": {"label": "a"}},
        },
        {
            "id": "b2",
            "type": "Print",
            "position": {"x": 0, "y": 0},
            "data": {"params": {"label": "b"}},
        },
    ]
    edges = [
        {"id": "t1", "source": "s1", "target": "a1", "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
        {"id": "t2", "source": "s2", "target": "b1", "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
        {"id": "d1", "source": "a1", "target": "a2", "sourceHandle": "tensor", "targetHandle": "value", "type": "data"},
        {"id": "d2", "source": "b1", "target": "b2", "sourceHandle": "tensor", "targetHandle": "value", "type": "data"},
    ]

    script = generate_python(nodes, edges, name="ordering")

    exec_nodes, exec_edges, _ = prepare_executable_graph(
        nodes, edges, preset_fallback={}
    )
    expected = topological_sort(exec_nodes, exec_edges)
    funcs = sorted(
        _node_functions(script),
        key=lambda entry: int(re.match(r"n(\d+)_", entry[0]).group(1)),
    )
    assert [node_id for _, node_id, _ in funcs] == expected


def test_example_graphs_discovered():
    """Guard: the examples parametrization below must never be empty."""
    assert EXAMPLE_GRAPHS


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
    """API -> .py -> fresh subprocess executes through the real node classes."""
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


def _contract_runner_graph() -> dict:
    """Start -> GraphInput(amount) -> Print -> GraphOutput(result)."""
    return {
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


@pytest.mark.asyncio
async def test_exported_runner_graph_input_output_json_round_trip(
    test_client,
    tmp_path: Path,
):
    response = await test_client.post("/api/graph/export", json=_contract_runner_graph())
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
async def test_exported_runner_uses_graph_input_default_without_input_flags(
    test_client,
    tmp_path: Path,
):
    """No --inputs flags: the GraphInput node falls back to its default."""
    response = await test_client.post("/api/graph/export", json=_contract_runner_graph())
    assert response.status_code == 200, response.text
    completed = _run_exported_script(
        response.json()["script"],
        tmp_path,
        "--device",
        "cpu",
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"result": 1}
    assert "[contract value] 1" in completed.stderr


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
async def test_export_expands_presets_into_node_functions_at_export_time(test_client):
    """Preset internals become real node functions in the emitted script."""
    graph = _load_example("Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json")
    response = await test_client.post("/api/graph/export", json=graph)

    assert response.status_code == 200, response.text
    script = response.json()["script"]
    _compile_check(script)
    assert "GRAPH_JSON" not in script
    # The preset node itself is gone; its internals are expanded functions
    # annotated with their origin.
    assert "'preset:Training Pipeline'" not in script
    assert "'TrainingLoop'" in script
    assert "# from preset 'Training Pipeline'" in script
    expanded_types = {node_type for _, _, node_type in _node_functions(script)}
    assert "TrainingLoop" in expanded_types


def test_generate_python_scrubs_secrets_from_expanded_preset_internals(monkeypatch):
    """A SECRET param inside a server-side preset definition must never be
    emitted.  Export-time expansion reads registry definitions the route's
    payload scrub never saw (hand-edited files, or files written before
    preset saving scrubbed secrets), so generate_python re-scrubs the
    expanded graph itself."""
    from app.core.codegen import generate_python
    from app.core.preset_registry import preset_registry
    from app.schemas.models import InternalNodeSchema, PresetDefinition

    secret = "sk-REGISTRY-HAND-EDITED-SECRET"
    monkeypatch.setitem(
        preset_registry._presets,
        "SecretChat",
        PresetDefinition(
            preset_name="SecretChat",
            category="Test",
            description="",
            nodes=[
                InternalNodeSchema(
                    id="chat",
                    type="LLMChat",
                    params={"provider": "ChatGPT API", "openai_api_key": secret},
                )
            ],
            edges=[],
            exposed_inputs=[
                {
                    "name": "text",
                    "internal_node": "chat",
                    "internal_port": "text",
                    "data_type": "STRING",
                    "description": "",
                }
            ],
            exposed_outputs=[],
            exposed_params=[],
        ),
    )

    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
        {
            "id": "p1",
            "type": "preset:SecretChat",
            "position": {"x": 0, "y": 0},
            "data": {"params": {}, "internalParams": {}},
        },
    ]
    edges = [
        {
            "id": "t1",
            "source": "start",
            "target": "p1",
            "sourceHandle": "trigger",
            "targetHandle": "text",
            "type": "trigger",
        },
    ]

    script = generate_python(nodes, edges, name="leak-test")
    _compile_check(script)
    assert secret not in script
    assert "'openai_api_key': ''" in script
    # The preset was triggered, so its internal node really is in the script.
    assert any(
        node_type == "LLMChat" for _, _, node_type in _node_functions(script)
    )


def test_exported_runner_timeout_stops_scheduling_remaining_nodes(tmp_path: Path):
    """After --timeout expires, no further node may start (engine parity:
    cancellation is cooperative and observed at node boundaries)."""
    from app.core.codegen import generate_python

    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
        {
            "id": "slow",
            "type": "TensorCreate",
            "position": {"x": 0, "y": 0},
            "data": {"params": {"shape": "8192,8192", "fill": "randn"}},
        },
        {
            "id": "late",
            "type": "Print",
            "position": {"x": 0, "y": 0},
            "data": {"params": {"label": "late-print"}},
        },
    ]
    edges = [
        {
            "id": "t1",
            "source": "start",
            "target": "slow",
            "sourceHandle": "trigger",
            "targetHandle": "",
            "type": "trigger",
        },
        {
            "id": "d1",
            "source": "slow",
            "target": "late",
            "sourceHandle": "tensor",
            "targetHandle": "value",
            "type": "data",
        },
    ]
    script = generate_python(nodes, edges, name="timeout-test")

    completed = _run_exported_script(
        script,
        tmp_path,
        "--device",
        "cpu",
        "--timeout",
        "0.05",
    )
    assert completed.returncode == 1, completed.stderr
    assert "Graph execution failed:" in completed.stderr
    assert "--timeout reached before node 'late'" in completed.stderr
    assert "[late-print]" not in completed.stdout
    assert "[late-print]" not in completed.stderr


@pytest.mark.asyncio
async def test_exported_runner_retains_embedded_preset_sibling_roots(
    test_client,
    tmp_path: Path,
):
    """Export-time expansion must keep a preset's untriggered sibling sources."""
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
    script = response.json()["script"]
    # Expanded at export time: the internal nodes (including the untriggered
    # sibling root "right") are real functions; no preset node remains.
    assert "'portable__right'" in script
    assert "'portable__add'" in script
    assert "'preset:Portable Add'" not in script

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
    script = response.json()["script"]
    # Drafts are pruned at export time — they leave no trace in the script.
    assert "'draft-a'" not in script
    assert "'draft-b'" not in script
    completed = _run_exported_script(
        script,
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
    script = response.json()["script"]
    funcs = _node_functions(script)
    assert {node_id for _, node_id, _ in funcs} == {"start", "tensor"}
    assert "'annotation'" not in script
    assert "teaching note" not in script


@pytest.mark.asyncio
async def test_multi_edge_fan_in_matches_engine_last_edge_wins(
    test_client,
    tmp_path: Path,
):
    """Two edges into one targetHandle: engine and script pick the same one."""
    from app.core import api_contract
    from app.core.execution_context import ExecutionContext
    from app.core.graph_engine import execute_graph

    graph = {
        "name": "fan-in",
        "nodes": [
            {"id": "start", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
            {
                "id": "ones",
                "type": "TensorCreate",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"shape": "2", "fill": "ones"}},
            },
            {
                "id": "twos",
                "type": "TensorCreate",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"shape": "2", "fill": "full", "value": 2.0}},
            },
            {
                "id": "picked",
                "type": "Print",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"label": "picked"}},
            },
            {
                "id": "out",
                "type": "GraphOutput",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"name": "result"}},
            },
        ],
        "edges": [
            {"id": "t1", "source": "start", "target": "ones", "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
            {"id": "t2", "source": "start", "target": "twos", "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
            {"id": "first", "source": "ones", "target": "picked", "sourceHandle": "tensor", "targetHandle": "value", "type": "data"},
            {"id": "second", "source": "twos", "target": "picked", "sourceHandle": "tensor", "targetHandle": "value", "type": "data"},
            {"id": "to-out", "source": "picked", "target": "out", "sourceHandle": "value", "targetHandle": "value", "type": "data"},
        ],
    }

    context = ExecutionContext(
        device="cpu", weights_persistent=False, graph_id="parity-fan-in"
    )
    engine_results = await execute_graph(
        graph["nodes"], graph["edges"], context=context
    )
    contract = api_contract.derive_contract(graph["nodes"])
    collected, missing = api_contract.collect_outputs(contract, engine_results)
    assert not missing
    engine_payload = json.loads(
        json.dumps(
            {
                name: api_contract.serialize_output(value)
                for name, value in collected.items()
            }
        )
    )
    # Engine semantics: the LAST edge in edges[] order wins.
    assert engine_payload["result"]["values"] == [2.0, 2.0]

    response = await test_client.post("/api/graph/export", json=graph)
    assert response.status_code == 200, response.text
    script = response.json()["script"]
    # _pick candidates are emitted in reverse edge order.
    assert "_pick(_port(twos, 'tensor'), _port(ones, 'tensor'))" in script

    completed = _run_exported_script(script, tmp_path, "--device", "cpu")
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == engine_payload


@pytest.mark.asyncio
async def test_absent_source_output_key_stays_absent_like_engine(
    test_client,
    tmp_path: Path,
):
    """A wired-but-unproduced source port yields NO input key, as in the engine.

    Split(chunks=3) on a 2-element tensor produces only chunk_0/chunk_1, so
    the edge from chunk_2 delivers nothing. Add must then fail with the
    engine's exact KeyError — a naive exporter passing None instead of
    omitting the key would fail differently (TypeError), or worse, succeed.
    """
    from app.core.execution_context import ExecutionContext
    from app.core.graph_engine import execute_graph

    graph = {
        "name": "absent-port",
        "nodes": [
            {"id": "start", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
            {
                "id": "seed",
                "type": "TensorCreate",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"shape": "2", "fill": "ones"}},
            },
            {
                "id": "split",
                "type": "Split",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"chunks": 3, "dim": 0}},
            },
            {
                "id": "7-add",
                "type": "Add",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"alpha": 1.0}},
            },
        ],
        "edges": [
            {"id": "t1", "source": "start", "target": "seed", "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
            {"id": "d1", "source": "seed", "target": "split", "sourceHandle": "tensor", "targetHandle": "tensor", "type": "data"},
            {"id": "d2", "source": "split", "target": "7-add", "sourceHandle": "chunk_2", "targetHandle": "tensor_a", "type": "data"},
            {"id": "d3", "source": "split", "target": "7-add", "sourceHandle": "chunk_0", "targetHandle": "tensor_b", "type": "data"},
        ],
    }

    context = ExecutionContext(
        device="cpu", weights_persistent=False, graph_id="parity-absent"
    )
    with pytest.raises(KeyError) as excinfo:
        await execute_graph(graph["nodes"], graph["edges"], context=context)
    assert str(excinfo.value) == "'tensor_a'"

    response = await test_client.post("/api/graph/export", json=graph)
    assert response.status_code == 200, response.text
    completed = _run_exported_script(
        response.json()["script"], tmp_path, "--device", "cpu"
    )
    assert completed.returncode == 1
    assert "[7-add] error: 'tensor_a'" in completed.stderr
    assert "Graph execution failed: 'tensor_a'" in completed.stderr


@pytest.mark.asyncio
async def test_engine_and_script_agree_on_deterministic_graph(
    test_client,
    tmp_path: Path,
):
    """execute_graph in-process and the exported script return identical JSON."""
    from app.core import api_contract
    from app.core.execution_context import ExecutionContext
    from app.core.graph_engine import execute_graph

    graph = {
        "name": "deterministic",
        "nodes": [
            {"id": "start", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
            {
                "id": "a",
                "type": "TensorCreate",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"shape": "2", "fill": "ones"}},
            },
            {
                "id": "b",
                "type": "TensorCreate",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"shape": "2", "fill": "full", "value": 2.0}},
            },
            {
                "id": "add",
                "type": "Add",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"alpha": 3.0}},
            },
            {
                "id": "out",
                "type": "GraphOutput",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"name": "result"}},
            },
        ],
        "edges": [
            {"id": "t1", "source": "start", "target": "a", "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
            {"id": "t2", "source": "start", "target": "b", "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
            {"id": "d1", "source": "a", "target": "add", "sourceHandle": "tensor", "targetHandle": "tensor_a", "type": "data"},
            {"id": "d2", "source": "b", "target": "add", "sourceHandle": "tensor", "targetHandle": "tensor_b", "type": "data"},
            {"id": "d3", "source": "add", "target": "out", "sourceHandle": "tensor", "targetHandle": "value", "type": "data"},
        ],
    }

    context = ExecutionContext(
        device="cpu", weights_persistent=False, graph_id="parity-arith"
    )
    engine_results = await execute_graph(
        graph["nodes"], graph["edges"], context=context
    )
    contract = api_contract.derive_contract(graph["nodes"])
    collected, missing = api_contract.collect_outputs(contract, engine_results)
    assert not missing
    engine_payload = json.loads(
        json.dumps(
            {
                name: api_contract.serialize_output(value)
                for name, value in collected.items()
            }
        )
    )
    assert engine_payload["result"]["values"] == [7.0, 7.0]  # 1 + 3.0 * 2

    response = await test_client.post("/api/graph/export", json=graph)
    assert response.status_code == 200, response.text
    completed = _run_exported_script(
        response.json()["script"], tmp_path, "--device", "cpu"
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == engine_payload


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


# ── Bypass in the exported script (core#128) ─────────────────────────────


def _bypass_export_graph(bypass: bool = True) -> tuple[list[dict], list[dict]]:
    """Start -> TensorCreate -> Dropout(bypassed?) -> Print."""
    dropout: dict = {
        "id": "drop",
        "type": "Dropout",
        "position": {"x": 0, "y": 0},
        "data": {"params": {"p": 0.5}},
    }
    if bypass:
        dropout["data"]["bypassed"] = True
    nodes = [
        {"id": "s1", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
        {
            "id": "make",
            "type": "TensorCreate",
            "position": {"x": 0, "y": 0},
            "data": {"params": {"shape": "2,2", "fill": "ones"}},
        },
        dropout,
        {
            "id": "out",
            "type": "Print",
            "position": {"x": 0, "y": 0},
            "data": {"params": {"label": "tail"}},
        },
    ]
    edges = [
        {"id": "t1", "source": "s1", "target": "make", "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
        {"id": "d1", "source": "make", "target": "drop", "sourceHandle": "tensor", "targetHandle": "tensor", "type": "data"},
        {"id": "d2", "source": "drop", "target": "out", "sourceHandle": "tensor", "targetHandle": "value", "type": "data"},
    ]
    return nodes, edges


def test_export_omits_a_bypassed_node_and_wires_the_pass_through():
    from app.core.codegen import generate_python

    script = generate_python(*_bypass_export_graph(), name="bypassed")
    _compile_check(script)

    emitted = _node_functions(script)
    assert "drop" not in {node_id for _, node_id, _ in emitted}
    assert {node_id for _, node_id, _ in emitted} == {"s1", "make", "out"}

    # Print is called with the tensor TensorCreate produced, not via Dropout.
    make_local = next(name for name, nid, _ in emitted if nid == "make")
    assert make_local  # sanity: the source node did get a function
    assert "_port(make, 'tensor')" in script

    # ...and the export says so, in a commented-out pass-through assignment.
    assert "# BYPASSED node 'drop' ('Dropout')" in script
    assert "#     drop['tensor'] = _port(make, 'tensor')" in script


def test_export_keeps_a_bypassed_node_when_the_flag_is_off():
    from app.core.codegen import generate_python

    script = generate_python(*_bypass_export_graph(bypass=False), name="plain")
    _compile_check(script)

    assert "drop" in {node_id for _, node_id, _ in _node_functions(script)}
    assert "BYPASSED" not in script


def test_export_refuses_a_bypass_with_no_compatible_input():
    from app.core.codegen import generate_python
    from app.core.graph_engine import GraphValidationError

    nodes = [
        {"id": "s1", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
        {"id": "ds", "type": "Dataset", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
        {
            "id": "dl",
            "type": "DataLoader",
            "position": {"x": 0, "y": 0},
            "data": {"params": {}, "bypassed": True},
        },
        {"id": "loop", "type": "TrainingLoop", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
    ]
    edges = [
        {"id": "t1", "source": "s1", "target": "ds", "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
        {"id": "d1", "source": "ds", "target": "dl", "sourceHandle": "dataset", "targetHandle": "dataset", "type": "data"},
        {"id": "d2", "source": "dl", "target": "loop", "sourceHandle": "dataloader", "targetHandle": "dataloader", "type": "data"},
    ]
    with pytest.raises(GraphValidationError, match="no type-compatible input"):
        generate_python(nodes, edges, name="broken")


def test_export_notes_a_bypassed_node_that_leads_nowhere():
    """A muted leaf has no consumer to sit above, so it gets its own note."""
    from app.core.codegen import generate_python

    nodes, edges = _bypass_export_graph()
    # Drop the Dropout -> Print edge: nothing consumes the bypass any more.
    edges = [e for e in edges if e["id"] != "d2"]
    nodes = [n for n in nodes if n["id"] != "out"]

    script = generate_python(nodes, edges, name="leaf")
    _compile_check(script)
    assert "# BYPASSED node 'drop' ('Dropout')" in script
    # Its input IS still wired (TensorCreate -> drop); what is missing is a
    # consumer, and the note has to say that and not the opposite.
    assert "#     (nothing downstream consumed it)" in script


# ── Inline source params (core#131) ──────────────────────────────────────


_SCRIPT_SOURCE = (
    'import statistics\n'
    '\n'
    'def run(inputs, params):\n'
    '    """Per-channel mean. Quotes: \' and " and \'\'\' stay data."""\n'
    "    values = list(inputs['in1'])\n"
    '    return {"out1": statistics.mean(values)}\n'
)


def _script_export_graph(code: str) -> tuple[list[dict], list[dict]]:
    nodes = [
        {"id": "s1", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
        {
            "id": "py",
            "type": "PythonScript",
            "position": {"x": 0, "y": 0},
            "data": {"params": {"code": code, "input_ports": 1, "output_ports": 1}},
        },
        {
            "id": "out",
            "type": "Print",
            "position": {"x": 0, "y": 0},
            "data": {"params": {"label": "mean"}},
        },
    ]
    edges = [
        {"id": "t1", "source": "s1", "target": "py", "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
        {"id": "d1", "source": "py", "target": "out", "sourceHandle": "out1", "targetHandle": "value", "type": "data"},
    ]
    return nodes, edges


def _exported_params(script: str, node_id: str) -> dict:
    """Evaluate the ``params`` literal of one node function, without running it."""
    module = ast.parse(script)
    for statement in module.body:
        if not isinstance(statement, ast.FunctionDef):
            continue
        matches = any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "_call"
            and call.args[1].value == node_id
            for call in ast.walk(statement)
        )
        if not matches:
            continue
        for inner in statement.body:
            if (
                isinstance(inner, ast.Assign)
                and isinstance(inner.targets[0], ast.Name)
                and inner.targets[0].id == "params"
            ):
                return ast.literal_eval(inner.value)
    raise AssertionError(f"no params assignment found for node {node_id!r}")


def test_code_param_round_trips_through_the_exported_script():
    """The script the node runs is recoverable from the export, byte for byte."""
    from app.core.codegen import generate_python

    nodes, edges = _script_export_graph(_SCRIPT_SOURCE)
    script = generate_python(nodes, edges, name="stats")

    _compile_check(script)
    assert _exported_params(script, "py")["code"] == _SCRIPT_SOURCE


def test_code_param_is_emitted_one_source_line_at_a_time():
    """Readable, and still the live value -- not a comment beside a blob."""
    from app.core.codegen import generate_python

    nodes, edges = _script_export_graph(_SCRIPT_SOURCE)
    script = generate_python(nodes, edges, name="stats")

    assert "the script this node runs, verbatim" in script
    for source_line in _SCRIPT_SOURCE.splitlines(keepends=True):
        assert f"            {ascii(source_line)}" in script
    # One line per literal, so no line of the export is a wall of escapes.
    assert ascii(_SCRIPT_SOURCE) not in script


def test_code_param_containing_triple_quotes_cannot_escape_the_literal():
    """The injection invariant holds for the one param that IS source."""
    from app.core.codegen import generate_python

    hostile = "'''\nimport os\nos.system('boom')\n'''\ndef run(i, p):\n    return 1\n"
    nodes, edges = _script_export_graph(hostile)
    script = generate_python(nodes, edges, name="hostile")

    _compile_check(script)
    assert _exported_params(script, "py")["code"] == hostile
    # It is data: the emitted file never contains a bare os.system call.
    assert "\nos.system" not in script
    assert "    os.system" not in script


def test_empty_code_param_stays_an_ordinary_literal():
    """An empty string must not become an empty tuple."""
    from app.core.codegen import generate_python

    nodes, edges = _script_export_graph("")
    script = generate_python(nodes, edges, name="empty")

    _compile_check(script)
    assert _exported_params(script, "py")["code"] == ""


def test_nodes_without_a_code_param_keep_the_one_line_params_literal():
    """The expanded form is scoped to CODE params, not every node."""
    from app.core.codegen import generate_python

    nodes, edges = _script_export_graph(_SCRIPT_SOURCE)
    script = generate_python(nodes, edges, name="stats")

    assert "    params = {'label': 'mean'}" in script


# ── advanced params round-trip (core#134) ─────────────────────────────────


def _advanced_params_graph() -> tuple[list[dict], list[dict]]:
    """One node per training role, each with its advanced params set."""
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "data", "type": "SyntheticShapes", "data": {"params": {}}},
        {"id": "model", "type": "SequentialModel", "data": {"params": {}}},
        {"id": "opt", "type": "Optimizer", "data": {"params": {
            "type": "AdamW", "lr": 0.002, "weight_decay": 0.01,
            "betas": "0.85, 0.995", "eps": 1e-06, "amsgrad": True,
        }}},
        {"id": "loss", "type": "Loss", "data": {"params": {
            "type": "CrossEntropyLoss", "label_smoothing": 0.1,
            "reduction": "sum", "weight": "1, 5", "ignore_index": 7,
        }}},
        {"id": "loader", "type": "DataLoader", "data": {"params": {
            "batch_size": 16, "shuffle": True, "num_workers": 2,
            "pin_memory": True, "drop_last": True,
            "persistent_workers": True, "prefetch_factor": 4,
        }}},
        {"id": "train", "type": "TrainingLoop", "data": {"params": {
            "epochs": 3, "device": "cpu", "max_steps": 25,
            "log_interval": 5, "deterministic": True,
        }}},
    ]

    def edge(source, source_handle, target, target_handle):
        return {
            "id": f"{source}.{source_handle}->{target}.{target_handle}",
            "source": source, "sourceHandle": source_handle,
            "target": target, "targetHandle": target_handle,
        }

    edges = [
        {"id": "t1", "source": "start", "target": "data",
         "sourceHandle": "trigger", "type": "trigger"},
        {"id": "t2", "source": "start", "target": "model",
         "sourceHandle": "trigger", "type": "trigger"},
        {"id": "t3", "source": "start", "target": "loss",
         "sourceHandle": "trigger", "type": "trigger"},
        edge("data", "dataset", "loader", "dataset"),
        edge("loader", "dataloader", "train", "dataloader"),
        edge("model", "model", "opt", "model"),
        edge("model", "model", "train", "model"),
        edge("opt", "optimizer", "train", "optimizer"),
        edge("loss", "loss_fn", "train", "loss_fn"),
    ]
    return nodes, edges


def test_advanced_params_round_trip_through_the_exported_script():
    """Hidden in the UI, identical on the wire.

    Codegen bakes each node's params as a literal dict and the real node
    class consumes it, so this is a round-trip assertion rather than a
    translation one -- and it is exactly the assertion that fails if a
    future "only export what the user changed" optimisation appears.
    """
    from app.core.codegen import generate_python

    nodes, edges = _advanced_params_graph()
    script = generate_python(nodes, edges, name="advanced-params")
    _compile_check(script)

    for node_id in ("opt", "loss", "loader", "train"):
        node = next(n for n in nodes if n["id"] == node_id)
        exported = _exported_params(script, node_id)
        assert exported == node["data"]["params"], node_id


@pytest.mark.asyncio
async def test_advanced_params_survive_a_real_graph_save_and_load(
    test_client, tmp_path, monkeypatch,
):
    """Through the ACTUAL persistence path, not a json round-trip of a dict.

    The first version of this test asserted ``json.loads(json.dumps(x)) == x``,
    which is true of any dict and touches no repo code — it would not have
    noticed if saving started dropping params. This one goes through
    ``POST /api/graph/save`` and ``GET /api/graph/load/{name}``, where the
    secret scrubber and the project-mode file split actually live.
    """
    from app.config import settings

    monkeypatch.setattr(settings, "GRAPHS_DIR", tmp_path / "graphs")
    nodes, edges = _advanced_params_graph()

    saved = await test_client.post("/api/graph/save", json={
        "name": "advanced-params-roundtrip", "nodes": nodes, "edges": edges,
    })
    assert saved.status_code == 200, saved.text

    loaded = await test_client.get("/api/graph/load/advanced-params-roundtrip")
    assert loaded.status_code == 200, loaded.text

    restored = {n["id"]: n for n in loaded.json()["nodes"]}
    for node_id in ("opt", "loss", "loader", "train"):
        expected = next(n for n in nodes if n["id"] == node_id)
        assert restored[node_id]["data"]["params"] == expected["data"]["params"], node_id


# ── transform chains (core#136) ──────────────────────────────────────────


def _transform_chain_graph(dataset_path: Path) -> tuple[list[dict], list[dict]]:
    """ImageFolder + a four-step chain + a batch, printed.

    Deliberately DETERMINISTIC (no augmentation): the exported script is
    compared against the in-process engine byte for byte, and a random crop
    would make the two runs legitimately differ.
    """
    def node(node_id: str, node_type: str, **params) -> dict:
        return {"id": node_id, "type": node_type,
                "position": {"x": 0, "y": 0}, "data": {"params": params}}

    def data_edge(source: str, source_handle: str,
                  target: str, target_handle: str) -> dict:
        return {"id": f"{source}->{target}", "source": source, "target": target,
                "sourceHandle": source_handle, "targetHandle": target_handle,
                "type": "data"}

    nodes = [
        node("start", "Start"),
        node("resize", "ResizeTransform", size=2, interpolation="bilinear"),
        node("tensor", "ToTensorTransform"),
        node("norm", "NormalizeTransform", preset="Half", mean="0.5", std="0.5"),
        node("folder", "ImageFolderDataset",
             path=str(dataset_path), split="(none)"),
        node("batch", "DatasetBatch", batch_size=2, start_index=0),
        node("print", "Print", label="chain"),
    ]
    edges = [
        {"id": "t1", "source": "start", "target": "resize",
         "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
        data_edge("resize", "transform", "tensor", "transform"),
        data_edge("tensor", "transform", "norm", "transform"),
        data_edge("norm", "transform", "folder", "eval_transform"),
        data_edge("folder", "dataset", "batch", "dataset"),
        data_edge("batch", "images", "print", "value"),
    ]
    return nodes, edges


def _image_folder_fixture(tmp_path: Path) -> Path:
    from PIL import Image

    root = tmp_path / "glyphs"
    for index, name in enumerate(("cat", "dog")):
        folder = root / name
        folder.mkdir(parents=True)
        Image.new("RGB", (4, 4), (index * 90, 30, 200)).save(folder / "0.png")
    return root


def test_export_emits_the_transform_chain_in_dependency_order(tmp_path: Path):
    """Each step becomes its own ``_call``, ordered by the edges.

    The chain has no branches, so a codegen that emitted the steps in graph
    declaration order rather than topological order would still produce
    valid Python -- and a pipeline in the wrong order.
    """
    from app.core.codegen import generate_python

    nodes, edges = _transform_chain_graph(_image_folder_fixture(tmp_path))
    script = generate_python(nodes, edges, name="transform chain")
    _compile_check(script)

    emitted = [node_type for _fn, _id, node_type in _node_functions(script)]
    for step in ("ResizeTransform", "ToTensorTransform", "NormalizeTransform",
                 "ImageFolderDataset"):
        assert step in emitted, step
    assert (emitted.index("ResizeTransform")
            < emitted.index("ToTensorTransform")
            < emitted.index("NormalizeTransform")
            < emitted.index("ImageFolderDataset"))


def test_export_carries_the_transform_params_verbatim(tmp_path: Path):
    from app.core.codegen import generate_python

    nodes, edges = _transform_chain_graph(_image_folder_fixture(tmp_path))
    script = generate_python(nodes, edges, name="transform chain")
    assert "'preset': 'Half'" in script
    assert "'size': 2" in script


def test_the_exported_transform_chain_builds_the_pipeline_it_should(
        tmp_path: Path):
    """core#136 review, M-7. Equality against a hand-written Compose.

    The old assertions here checked node ORDER and the presence of two
    param substrings, which is why mutating ``ResizeTransform`` to emit a
    ``CenterCrop`` -- a node that IS in this graph -- left all 64 tests in
    this file green. The acceptance criterion is "codegen emits an
    equivalent ``transforms.Compose``", so the pipeline the exported script
    actually builds is compared against one written out by hand here, step
    by step, with the arguments spelled out.

    The name carries "transform" on purpose (core#136 re-review, N-5). The
    acceptance criterion's own command is ``pytest tests/test_codegen.py -k
    transform``, and under the first spelling of this name that filter
    deselected the one test that catches the mutation -- so the obvious way
    to check this area reported ``10 passed`` while the file was red.
    """
    from torchvision import transforms as T

    from app.core.codegen import generate_python

    nodes, edges = _transform_chain_graph(_image_folder_fixture(tmp_path))
    script = generate_python(nodes, edges, name="transform chain")
    results = _run_generated_module(script)

    expected = T.Compose([
        T.Resize((2, 2), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])
    assert repr(results["norm"]["transform"]) == repr(expected)


def _run_generated_module(script: str, **context_overrides):
    """Execute a generated script's ``run_graph`` in this process.

    The generated file is a module whose ``main()`` is behind the usual
    ``__main__`` guard, so its runtime prelude and node functions can be
    driven directly. That keeps these tests about what the emitted code
    BUILDS rather than about anything the CLI prints -- and means no
    debug-only flag has to exist in shipped exports to make it observable.

    The context is built from the script's OWN baked ``GRAPH_SEED`` /
    ``GRAPH_DETERMINISTIC``, which is what its CLI uses as the default, so a
    test reading the result is reading what a plain invocation would do.
    """
    from app.core.execution_context import ExecutionContext

    module: dict = {"__name__": "generated_export"}
    exec(compile(script, "<generated>", "exec"), module)  # noqa: S102
    module["_RT"] = module["_load_runtime"](None)
    module["_RT"].initialize_runtime()
    settings = {
        "seed": module.get("GRAPH_SEED"),
        "deterministic": bool(module.get("GRAPH_DETERMINISTIC", False)),
    }
    settings.update(context_overrides)
    return module["run_graph"](ExecutionContext(**settings), {})


# ── the export carries the seed (core#136 review, M-6) ───────────────────
#
# Before this, ``grep -c seed codegen.py`` returned 0: the exported script
# built an unseeded ExecutionContext, so derive_seed returned None,
# seed_pipeline returned the pipeline unwrapped, no SeededAugmentation was
# ever installed, and three invocations of an exported augmenting graph gave
# three different results -- while data-augmentation.md promised "the same
# seed produces the same crops, flips and colour shifts, every time".
#
# The codegen graph above is deliberately deterministic, so it structurally
# could not have noticed. These use a RANDOM chain on purpose.


def _augmenting_graph(dataset_path: Path) -> tuple[list[dict], list[dict]]:
    def node(node_id: str, node_type: str, **params) -> dict:
        return {"id": node_id, "type": node_type,
                "position": {"x": 0, "y": 0}, "data": {"params": params}}

    def data_edge(source: str, source_handle: str,
                  target: str, target_handle: str) -> dict:
        return {"id": f"{source}->{target}", "source": source, "target": target,
                "sourceHandle": source_handle, "targetHandle": target_handle,
                "type": "data"}

    nodes = [
        node("start", "Start"),
        node("flip", "RandomHorizontalFlip", p=0.5),
        node("tensor", "ToTensorTransform"),
        node("folder", "ImageFolderDataset",
             path=str(dataset_path), split="(none)"),
        node("batch", "DatasetBatch", batch_size=8, start_index=0),
        node("print", "Print", label="aug"),
    ]
    edges = [
        {"id": "t1", "source": "start", "target": "flip",
         "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
        data_edge("flip", "transform", "tensor", "transform"),
        data_edge("tensor", "transform", "folder", "train_transform"),
        data_edge("folder", "dataset", "batch", "dataset"),
        data_edge("batch", "images", "print", "value"),
    ]
    return nodes, edges


def _asymmetric_fixture(tmp_path: Path) -> Path:
    """Images a horizontal flip actually changes."""
    import torch
    from PIL import Image

    root = tmp_path / "asym"
    folder = root / "cat"
    folder.mkdir(parents=True)
    generator = torch.Generator().manual_seed(17)
    for index in range(8):
        pixels = torch.randint(0, 256, (4, 4, 3), generator=generator,
                               dtype=torch.uint8)
        Image.fromarray(pixels.numpy(), mode="RGB").save(
            folder / f"{index}.png")
    return root


def test_the_export_bakes_the_canvas_seed_as_the_cli_default(tmp_path: Path):
    from app.core.codegen import generate_python

    nodes, edges = _augmenting_graph(_asymmetric_fixture(tmp_path))
    seeded = generate_python(nodes, edges, name="aug", seed=4321,
                             deterministic=True)
    _compile_check(seeded)
    assert "GRAPH_SEED = 4321" in seeded
    assert "GRAPH_DETERMINISTIC = True" in seeded

    unseeded = generate_python(nodes, edges, name="aug")
    assert "GRAPH_SEED = None" in unseeded
    assert "GRAPH_DETERMINISTIC = False" in unseeded


def test_an_exported_seeded_graph_installs_the_augmentation_wrapper(
        tmp_path: Path):
    """The mechanism M-6 was missing, at the point it goes missing.

    The context here comes from the script's OWN baked ``GRAPH_SEED``, so
    an export that forgot to bake it leaves ``derive_seed`` returning None
    and the pipeline unwrapped.
    """
    from app.nodes.data.transforms._base import SeededAugmentation

    from app.core.codegen import generate_python

    nodes, edges = _augmenting_graph(_asymmetric_fixture(tmp_path))
    script = generate_python(nodes, edges, name="aug", seed=4321)
    results = _run_generated_module(script)
    assert isinstance(results["folder"]["dataset"].transform,
                      SeededAugmentation)

    unseeded = generate_python(nodes, edges, name="aug")
    plain = _run_generated_module(unseeded)
    assert not isinstance(plain["folder"]["dataset"].transform,
                          SeededAugmentation)


@pytest.mark.asyncio
async def test_an_exported_seeded_graph_reproduces_the_engine(tmp_path: Path):
    """The whole promise, end to end and across a process boundary.

    Three invocations of the exported script in fresh ``-I`` subprocesses,
    plus one in-process engine run, all have to agree -- and a different
    seed has to disagree, or the assertion above it would hold for a script
    that had simply stopped augmenting.
    """
    from app.core.codegen import generate_python
    from app.core.execution_context import ExecutionContext
    from app.core.graph_engine import execute_graph

    nodes, edges = _augmenting_graph(_asymmetric_fixture(tmp_path))
    outputs = await execute_graph(
        nodes, edges, context=ExecutionContext(seed=4321))
    in_process = outputs["print"]["__log__"]

    script = generate_python(nodes, edges, name="aug", seed=4321)
    runs = []
    for attempt in range(3):
        export_dir = tmp_path / f"export{attempt}"
        export_dir.mkdir()
        completed = _run_exported_script(script, export_dir, "--device", "cpu")
        assert completed.returncode == 0, completed.stderr
        runs.append(completed.stdout)

    assert runs[0] == runs[1] == runs[2], "the export is not reproducible"
    assert in_process in runs[0], (
        "the exported script does not reproduce the canvas")

    other = tmp_path / "other"
    other.mkdir()
    different = _run_exported_script(
        generate_python(nodes, edges, name="aug", seed=1),
        other, "--device", "cpu")
    assert different.returncode == 0, different.stderr
    assert different.stdout != runs[0], (
        "two different seeds produced the same augmentation, so the seed is "
        "not reaching the transform")


#: A seed value nothing in the run produces on its own. The probe graph
#: installs it as torch's seed and the test asks whether it SURVIVED, so it
#: only has to be distinguishable from ``derive_seed(4321, "probe")`` -- and
#: the test asserts that separation rather than assuming it.
_SEED_SENTINEL = 987_654_321


def _seed_probe_graph() -> tuple[list[dict], list[dict]]:
    """Start -> install a sentinel seed -> report torch's seed -> Print.

    Two script nodes rather than one, and that is the whole mechanism: the
    export re-seeds before EVERY node (``codegen._call``), so a sentinel
    installed by the first node is overwritten before the second one runs
    exactly when seeding is active, and survives untouched exactly when it
    is not. Reading ``torch.initial_seed()`` from the second node therefore
    reports which of the two happened, as a value rather than as a sample.

    The data edge between them is there for the ordering, not the value; the
    probe ignores what it is handed.
    """
    def script(node_id: str, code: str, **extra) -> dict:
        return {
            "id": node_id,
            "type": "PythonScript",
            "position": {"x": 0, "y": 0},
            "data": {"params": {
                "code": code,
                "input_ports": 1,
                "output_ports": 1,
                "input_types": "ANY",
                "output_types": "ANY",
                **extra,
            }},
        }

    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {"params": {}}},
        script("sentinel", (
            "import torch\n"
            "\n"
            "\n"
            "def run(inputs, params):\n"
            f"    torch.manual_seed({_SEED_SENTINEL})\n"
            "    return {'out1': 0}\n"
        )),
        script("probe", (
            "import torch\n"
            "\n"
            "\n"
            "def run(inputs, params):\n"
            "    return {'out1': torch.initial_seed()}\n"
        )),
        {"id": "print", "type": "Print", "position": {"x": 0, "y": 0},
         "data": {"params": {"label": "seed"}}},
    ]
    edges = [
        {"id": "t1", "source": "start", "target": "sentinel",
         "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
        {"id": "e1", "source": "sentinel", "target": "probe",
         "sourceHandle": "out1", "targetHandle": "in1", "type": "data"},
        {"id": "e2", "source": "probe", "target": "print",
         "sourceHandle": "out1", "targetHandle": "value", "type": "data"},
    ]
    return nodes, edges


def test_no_seed_leaves_an_exported_run_on_torchs_own_entropy(tmp_path: Path):
    """``--no-seed`` means what "no seed" means everywhere else.

    Asserted as two equalities against values known BEFORE either run, which
    is the point of #277. The assertion this replaced ran the same graph
    twice with ``--no-seed`` and required the two outputs to differ -- a bet
    that two independent draws would not collide, and on that graph the draw
    was eight independent coin flips (``RandomHorizontalFlip(p=0.5)`` over a
    batch of eight), so the collision it eventually hit on the 2.2.0 release
    PR had probability 1/256 per run. Nothing here samples: with ``--no-seed``
    the probe must report exactly the sentinel the node before it installed,
    and with the baked seed it must report exactly ``derive_seed(4321,
    "probe")``. Both sides are fixed values, so this test either passes on
    every run or fails on every run.
    """
    from app.core.codegen import generate_python
    from app.core.seeding import derive_seed

    seeded_expectation = derive_seed(4321, "probe")
    # Otherwise a run that seeded and a run that did not would report the
    # same number and both assertions below would hold vacuously.
    assert seeded_expectation != _SEED_SENTINEL

    nodes, edges = _seed_probe_graph()
    script = generate_python(nodes, edges, name="seed probe", seed=4321)

    unseeded_dir = tmp_path / "unseeded"
    unseeded_dir.mkdir()
    unseeded = _run_exported_script(
        script, unseeded_dir, "--device", "cpu", "--no-seed")
    assert unseeded.returncode == 0, unseeded.stderr

    seeded_dir = tmp_path / "seeded"
    seeded_dir.mkdir()
    seeded = _run_exported_script(script, seeded_dir, "--device", "cpu")
    assert seeded.returncode == 0, seeded.stderr

    assert f"[seed] {_SEED_SENTINEL}" in unseeded.stdout, (
        "--no-seed re-seeded the run: the sentinel installed by the previous "
        f"node did not survive to the probe. stdout: {unseeded.stdout!r}")
    assert f"[seed] {seeded_expectation}" in seeded.stdout, (
        "the baked GRAPH_SEED did not reach the node: the probe should report "
        f"derive_seed(4321, 'probe'). stdout: {seeded.stdout!r}")


@pytest.mark.asyncio
async def test_exported_transform_chain_matches_the_engine(tmp_path: Path):
    """The round trip: engine result == exported-script result.

    Nothing is hand-computed. The chain runs once through ``execute_graph``
    and once through a fresh ``-I`` subprocess driving the generated script,
    and the tensor ``Print`` saw has to be the same text both times.
    """
    from app.core.codegen import generate_python
    from app.core.execution_context import ExecutionContext
    from app.core.graph_engine import execute_graph

    nodes, edges = _transform_chain_graph(_image_folder_fixture(tmp_path))
    outputs = await execute_graph(nodes, edges, context=ExecutionContext())
    in_process = outputs["print"]["__log__"]
    assert in_process.startswith("[chain] ")
    # The chain really ran: 2 images, 3 channels, resized 4x4 -> 2x2.
    assert "torch.Size" not in in_process  # Print renders values, not shapes
    assert outputs["batch"]["images"].shape == (2, 3, 2, 2)

    script = generate_python(nodes, edges, name="transform chain")
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    completed = _run_exported_script(script, export_dir, "--device", "cpu")
    assert completed.returncode == 0, completed.stderr
    # stdout, not stderr: the runner only redirects Print to stderr when the
    # graph has a GraphOutput node reserving stdout for its JSON.
    assert in_process in completed.stdout


# ── Subgraph export (core#137) ───────────────────────────────────────────


def _subgraph_export_graph() -> tuple[list[dict], list[dict], list[dict]]:
    """Two instances of one subgraph, chained, feeding a Print.

    Two instances on purpose: it is the case where a per-DEFINITION function
    would be wrong (each instance has its own node ids and its own results
    slots) and where a shared mutable structure would show up as one instance
    overwriting the other's outputs.
    """
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {"params": {}}},
        {"id": "src", "type": "TensorCreate", "position": {"x": 1, "y": 0},
         "data": {"params": {"shape": "1,2", "fill": "full", "value": 1.0}}},
        {"id": "one", "type": "subgraph:double", "position": {"x": 2, "y": 0},
         "data": {"params": {}}},
        {"id": "two", "type": "subgraph:double", "position": {"x": 3, "y": 0},
         "data": {"params": {}}},
        {"id": "out", "type": "Print", "position": {"x": 4, "y": 0},
         "data": {"params": {"label": "TOTAL"}}},
    ]
    edges = [
        {"id": "t", "source": "start", "target": "src",
         "sourceHandle": "trigger", "targetHandle": "__trigger",
         "type": "trigger"},
        {"id": "e1", "source": "src", "target": "one",
         "sourceHandle": "tensor", "targetHandle": "in", "type": "data"},
        {"id": "e2", "source": "one", "target": "two",
         "sourceHandle": "out", "targetHandle": "in", "type": "data"},
        {"id": "e3", "source": "two", "target": "out",
         "sourceHandle": "out", "targetHandle": "value", "type": "data"},
    ]
    subgraphs = [{
        "id": "double",
        "name": "Double",
        "nodes": [
            {"id": "mul", "type": "ScalarMultiply",
             "position": {"x": 0, "y": 0},
             "data": {"params": {"scalar": 2.0}}},
            {"id": "avg", "type": "Mean", "position": {"x": 1, "y": 0},
             "data": {"params": {"dim": "-1", "keepdim": True}}},
        ],
        "edges": [
            {"id": "i", "source": "mul", "target": "avg",
             "sourceHandle": "tensor", "targetHandle": "tensor",
             "type": "data"},
        ],
        "interface": {
            "inputs": [
                {"port": "in", "innerNode": "mul", "innerPort": "tensor"},
            ],
            "outputs": [
                {"port": "out", "innerNode": "avg", "innerPort": "tensor"},
            ],
            "triggerTargets": ["mul"],
        },
    }]
    return nodes, edges, subgraphs


def test_export_emits_one_function_per_subgraph_instance():
    from app.core.codegen import generate_python

    nodes, edges, subgraphs = _subgraph_export_graph()
    script = generate_python(nodes, edges, subgraphs=subgraphs)
    _compile_check(script)

    module = ast.parse(script)
    names = [
        stmt.name for stmt in module.body
        if isinstance(stmt, ast.FunctionDef)
        and stmt.name.startswith("subgraph_")
    ]
    assert len(names) == 2, names
    assert all(name.startswith("subgraph_double") for name in names), names

    # The instance node type never reaches the script -- it was expanded.
    assert "'subgraph:double'" not in script
    # ... and each expanded node says where it came from.
    assert "# from subgraph 'double' (node 'one')" in script
    assert "# from subgraph 'double' (node 'two')" in script


def test_a_subgraph_function_holds_its_members_and_the_flow_calls_it_once():
    from app.core.codegen import generate_python

    nodes, edges, subgraphs = _subgraph_export_graph()
    script = generate_python(nodes, edges, subgraphs=subgraphs)
    module = ast.parse(script)
    bodies = {
        stmt.name: ast.unparse(stmt)
        for stmt in module.body
        if isinstance(stmt, ast.FunctionDef)
    }
    subgraph_names = sorted(n for n in bodies if n.startswith("subgraph_"))
    flow_body = bodies["flow_1"]

    for name in subgraph_names:
        # Called exactly once from the flow.
        assert flow_body.count(f"{name}(ctx, results, provided)") == 1

    # Members are inside the subgraph function, not the flow.
    first = bodies[subgraph_names[0]]
    assert "results['one/mul']" in first
    assert "results['one/avg']" in first
    assert "results['one/mul']" not in flow_body
    # The flow reads the block's output through results, since the local for
    # an inner node does not exist in the flow's scope.
    assert "results['two/avg']" in flow_body


def test_exported_subgraph_script_actually_runs(tmp_path):
    """The generated program must RUN, not merely compile."""
    from app.core.codegen import generate_python

    nodes, edges, subgraphs = _subgraph_export_graph()
    script = generate_python(
        nodes, edges, name="SubgraphDemo", subgraphs=subgraphs
    )
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    completed = _run_exported_script(script, export_dir, "--device", "cpu")
    assert completed.returncode == 0, completed.stderr
    # 1.0 -> x2 -> mean -> x2 -> mean == 4.0, printed by the Print node.
    assert "TOTAL" in completed.stdout, completed.stdout
    assert "4." in completed.stdout, completed.stdout


async def test_exported_subgraph_script_agrees_with_the_engine(tmp_path):
    """Same graph, two runners, same number."""
    import re as _re

    from app.core.codegen import generate_python
    from app.core.graph_engine import execute_graph

    nodes, edges, subgraphs = _subgraph_export_graph()
    engine = await execute_graph(nodes, edges, subgraphs=subgraphs)
    engine_value = float(engine["two/avg"]["tensor"].reshape(-1)[0])

    script = generate_python(nodes, edges, subgraphs=subgraphs)
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    completed = _run_exported_script(script, export_dir, "--device", "cpu")
    assert completed.returncode == 0, completed.stderr
    numbers = _re.findall(r"-?\d+\.\d*", completed.stdout)
    assert numbers, completed.stdout
    assert float(numbers[-1]) == pytest.approx(engine_value)


# ── Subgraph instances that cannot be ONE call (core#137 review) ─────────
#
# Emitting an instance as a single call at the position of its first member
# is only correct when the whole block is schedulable there. Three graphs
# below break that in three different ways; all three run in the engine, so
# the exported script has to run too.


def _subgraph_function_names(script: str) -> list[str]:
    module = ast.parse(script)
    return [
        stmt.name for stmt in module.body
        if isinstance(stmt, ast.FunctionDef)
        and stmt.name.startswith("subgraph_")
    ]


def _outside_node_in_the_middle_graph():
    """An outside node reads one member and feeds another.

    The editor refuses to COLLAPSE this shape, but two edge drags after a
    legal collapse recreate it, and an imported / plugin-built graph never
    asked the editor. 1 -> x2 (in) -> x10 (outside) -> x5 (in) == 100.
    """
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {"params": {}}},
        {"id": "src", "type": "TensorCreate", "position": {"x": 1, "y": 0},
         "data": {"params": {"shape": "1,2", "fill": "full", "value": 1.0}}},
        {"id": "blk", "type": "subgraph:ab", "position": {"x": 2, "y": 0},
         "data": {"params": {}}},
        {"id": "mid", "type": "ScalarMultiply", "position": {"x": 3, "y": 0},
         "data": {"params": {"scalar": 10.0}}},
        {"id": "out", "type": "Print", "position": {"x": 4, "y": 0},
         "data": {"params": {"label": "TOTAL"}}},
    ]
    edges = [
        {"id": "t", "source": "start", "target": "src",
         "sourceHandle": "trigger", "targetHandle": "__trigger",
         "type": "trigger"},
        {"id": "e1", "source": "src", "target": "blk",
         "sourceHandle": "tensor", "targetHandle": "inA", "type": "data"},
        {"id": "e2", "source": "blk", "target": "mid",
         "sourceHandle": "outA", "targetHandle": "tensor", "type": "data"},
        {"id": "e3", "source": "mid", "target": "blk",
         "sourceHandle": "tensor", "targetHandle": "inB", "type": "data"},
        {"id": "e4", "source": "blk", "target": "out",
         "sourceHandle": "outB", "targetHandle": "value", "type": "data"},
    ]
    subgraphs = [{
        "id": "ab",
        "name": "AB",
        "nodes": [
            {"id": "a", "type": "ScalarMultiply", "position": {"x": 0, "y": 0},
             "data": {"params": {"scalar": 2.0}}},
            {"id": "b", "type": "ScalarMultiply", "position": {"x": 1, "y": 0},
             "data": {"params": {"scalar": 5.0}}},
        ],
        "edges": [],
        "interface": {
            "inputs": [
                {"port": "inA", "innerNode": "a", "innerPort": "tensor"},
                {"port": "inB", "innerNode": "b", "innerPort": "tensor"},
            ],
            "outputs": [
                {"port": "outA", "innerNode": "a", "innerPort": "tensor"},
                {"port": "outB", "innerNode": "b", "innerPort": "tensor"},
            ],
            "triggerTargets": ["a"],
        },
    }]
    return nodes, edges, subgraphs


def test_exported_script_runs_when_an_outside_node_sits_inside_a_subgraph(
    tmp_path,
):
    """The reviewer's repro: hoisting reads 'mid' before it exists."""
    from app.core.codegen import generate_python

    nodes, edges, subgraphs = _outside_node_in_the_middle_graph()
    script = generate_python(
        nodes, edges, name="NonConvex", subgraphs=subgraphs
    )
    _compile_check(script)
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    completed = _run_exported_script(script, export_dir, "--device", "cpu")
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "TOTAL" in completed.stdout, completed.stdout
    assert "100." in completed.stdout, completed.stdout


def test_a_subgraph_an_outside_node_sits_inside_is_emitted_inline(tmp_path):
    """No function for it, and the script says why -- silence is worse."""
    from app.core.codegen import generate_python

    nodes, edges, subgraphs = _outside_node_in_the_middle_graph()
    script = generate_python(nodes, edges, subgraphs=subgraphs)

    assert _subgraph_function_names(script) == []
    # The members are still named as the block's, and the flow explains the
    # degradation rather than silently flattening.
    assert "# from subgraph 'ab' (node 'blk')" in script
    module = ast.parse(script)
    flow = next(
        stmt for stmt in module.body
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "flow_1"
    )
    body = ast.unparse(flow)
    assert "results['blk/a']" in body
    assert "results['blk/b']" in body
    assert "inlined" in script and "'blk'" in script


async def test_inlined_subgraph_export_agrees_with_the_engine(tmp_path):
    """The engine runs this graph fine; the export must produce its number."""
    import re as _re

    from app.core.codegen import generate_python
    from app.core.graph_engine import execute_graph

    nodes, edges, subgraphs = _outside_node_in_the_middle_graph()
    engine = await execute_graph(nodes, edges, subgraphs=subgraphs)
    engine_value = float(engine["blk/b"]["tensor"].reshape(-1)[0])

    script = generate_python(nodes, edges, subgraphs=subgraphs)
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    completed = _run_exported_script(script, export_dir, "--device", "cpu")
    assert completed.returncode == 0, completed.stderr + completed.stdout
    numbers = _re.findall(r"-?\d+\.\d*", completed.stdout)
    assert numbers, completed.stdout
    assert float(numbers[-1]) == pytest.approx(engine_value)


def _split_flow_subgraph_graph():
    """One instance whose members land in two disconnected flows.

    ``side`` prints on its own; only ``a`` is wired to the outside world, so
    ``_split_flows`` puts {src, blk/a, out} and {blk/mk, blk/side} in
    different components -- and a per-flow "already emitted" set would call
    the block's function once in each.

    Two Start markers, one per half: a single one would trigger both halves
    and weld them into a single component, and no Start at all is refused
    ("Graph has no entry points") before codegen ever sees the graph.
    """
    nodes = [
        {"id": "start1", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {"params": {}}},
        {"id": "start2", "type": "Start", "position": {"x": 0, "y": 2},
         "data": {"params": {}}},
        {"id": "src", "type": "TensorCreate", "position": {"x": 1, "y": 0},
         "data": {"params": {"shape": "1,2", "fill": "full", "value": 1.0}}},
        {"id": "blk", "type": "subgraph:pair", "position": {"x": 2, "y": 0},
         "data": {"params": {}}},
        {"id": "out", "type": "Print", "position": {"x": 3, "y": 0},
         "data": {"params": {"label": "MAIN"}}},
    ]
    edges = [
        {"id": "t1", "source": "start1", "target": "src",
         "sourceHandle": "trigger", "targetHandle": "__trigger",
         "type": "trigger"},
        {"id": "t2", "source": "start2", "target": "blk",
         "sourceHandle": "trigger", "targetHandle": "__trigger",
         "type": "trigger"},
        {"id": "e1", "source": "src", "target": "blk",
         "sourceHandle": "tensor", "targetHandle": "in", "type": "data"},
        {"id": "e2", "source": "blk", "target": "out",
         "sourceHandle": "out", "targetHandle": "value", "type": "data"},
    ]
    subgraphs = [{
        "id": "pair",
        "name": "Pair",
        "nodes": [
            {"id": "a", "type": "ScalarMultiply", "position": {"x": 0, "y": 0},
             "data": {"params": {"scalar": 2.0}}},
            {"id": "mk", "type": "TensorCreate", "position": {"x": 0, "y": 1},
             "data": {"params": {"shape": "1,1", "fill": "full",
                                 "value": 7.0}}},
            {"id": "side", "type": "Print", "position": {"x": 1, "y": 1},
             "data": {"params": {"label": "SIDE"}}},
        ],
        "edges": [
            {"id": "i", "source": "mk", "target": "side",
             "sourceHandle": "tensor", "targetHandle": "value",
             "type": "data"},
        ],
        "interface": {
            "inputs": [
                {"port": "in", "innerNode": "a", "innerPort": "tensor"},
            ],
            "outputs": [
                {"port": "out", "innerNode": "a", "innerPort": "tensor"},
            ],
            # Only the free half: triggering `a` too would put start2 in the
            # same component as everything else.
            "triggerTargets": ["mk"],
        },
    }]
    return nodes, edges, subgraphs


def test_a_subgraph_split_across_flows_runs_each_member_exactly_once(tmp_path):
    """Two flows must not each call the same block function."""
    from app.core.codegen import generate_python

    nodes, edges, subgraphs = _split_flow_subgraph_graph()
    script = generate_python(
        nodes, edges, name="SplitFlow", subgraphs=subgraphs
    )
    _compile_check(script)
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    completed = _run_exported_script(script, export_dir, "--device", "cpu")
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert completed.stdout.count("SIDE") == 1, completed.stdout
    assert completed.stdout.count("MAIN") == 1, completed.stdout
    assert _subgraph_function_names(script) == []


def _late_input_subgraph_graph():
    """A CONVEX block whose outside input is ordered after its first member.

    ``mk`` has no inputs, so Kahn puts it early; ``use`` is fed through
    ``step``, which lands AFTER it. Nothing outside the block is both fed by
    it and feeding it -- the block is convex -- yet hoisting both members to
    ``mk``'s position still reads ``step`` before it is assigned.
    """
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {"params": {}}},
        {"id": "src", "type": "TensorCreate", "position": {"x": 1, "y": 0},
         "data": {"params": {"shape": "1,2", "fill": "full", "value": 1.0}}},
        {"id": "step", "type": "ScalarMultiply", "position": {"x": 2, "y": 0},
         "data": {"params": {"scalar": 4.0}}},
        {"id": "blk", "type": "subgraph:late", "position": {"x": 3, "y": 0},
         "data": {"params": {}}},
        {"id": "sink", "type": "Print", "position": {"x": 4, "y": 1},
         "data": {"params": {"label": "MK"}}},
        {"id": "out", "type": "Print", "position": {"x": 4, "y": 0},
         "data": {"params": {"label": "USE"}}},
    ]
    edges = [
        {"id": "t1", "source": "start", "target": "src",
         "sourceHandle": "trigger", "targetHandle": "__trigger",
         "type": "trigger"},
        # Keeps mk in the same weakly-connected component as everything else,
        # so this graph tests ordering and NOT the split-flow case.
        {"id": "t2", "source": "start", "target": "blk",
         "sourceHandle": "trigger", "targetHandle": "__trigger",
         "type": "trigger"},
        {"id": "e1", "source": "src", "target": "step",
         "sourceHandle": "tensor", "targetHandle": "tensor", "type": "data"},
        {"id": "e2", "source": "step", "target": "blk",
         "sourceHandle": "tensor", "targetHandle": "in", "type": "data"},
        {"id": "e3", "source": "blk", "target": "sink",
         "sourceHandle": "mk_out", "targetHandle": "value", "type": "data"},
        {"id": "e4", "source": "blk", "target": "out",
         "sourceHandle": "use_out", "targetHandle": "value", "type": "data"},
    ]
    subgraphs = [{
        "id": "late",
        "name": "Late",
        "nodes": [
            {"id": "mk", "type": "TensorCreate", "position": {"x": 0, "y": 0},
             "data": {"params": {"shape": "1,1", "fill": "full",
                                 "value": 3.0}}},
            {"id": "use", "type": "ScalarMultiply",
             "position": {"x": 0, "y": 1},
             "data": {"params": {"scalar": 2.0}}},
        ],
        "edges": [],
        "interface": {
            "inputs": [
                {"port": "in", "innerNode": "use", "innerPort": "tensor"},
            ],
            "outputs": [
                {"port": "mk_out", "innerNode": "mk", "innerPort": "tensor"},
                {"port": "use_out", "innerNode": "use", "innerPort": "tensor"},
            ],
            "triggerTargets": ["mk"],
        },
    }]
    return nodes, edges, subgraphs


def test_exported_script_runs_when_a_subgraph_input_arrives_late(tmp_path):
    """Convexity alone is not enough: this block is convex and still breaks."""
    from app.core.codegen import generate_python

    nodes, edges, subgraphs = _late_input_subgraph_graph()
    script = generate_python(
        nodes, edges, name="LateInput", subgraphs=subgraphs
    )
    _compile_check(script)
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    completed = _run_exported_script(script, export_dir, "--device", "cpu")
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "MK" in completed.stdout, completed.stdout
    assert "3." in completed.stdout, completed.stdout
    assert "USE" in completed.stdout, completed.stdout
    assert "8." in completed.stdout, completed.stdout


def test_a_self_referential_container_map_does_not_hang_the_exporter(
    monkeypatch,
):
    """A malformed expansion map must finish, not spin.

    The exporter walks ``internal node -> container`` to find the OUTERMOST
    container. A map that points at itself, or in a ring, used to make that
    walk loop forever -- an export that never returns and never errors.
    Runs on a worker thread so a regression fails the test instead of
    hanging the suite.
    """
    import threading

    from app.core import codegen as codegen_module
    from app.core.codegen import generate_python

    real = codegen_module.prepare_executable_graph

    def poisoned(*args, **kwargs):
        exec_nodes, exec_edges, mapping = real(*args, **kwargs)
        ids = [node["id"] for node in exec_nodes]
        mapping[ids[0]] = ids[0]                    # contains itself
        mapping[ids[1]], mapping[ids[2]] = ids[2], ids[1]  # contain each other
        return exec_nodes, exec_edges, mapping

    monkeypatch.setattr(codegen_module, "prepare_executable_graph", poisoned)

    nodes, edges, subgraphs = _subgraph_export_graph()
    produced: list[str] = []
    worker = threading.Thread(
        target=lambda: produced.append(
            generate_python(nodes, edges, subgraphs=subgraphs)
        ),
        daemon=True,
    )
    worker.start()
    worker.join(15)
    assert not worker.is_alive(), "generate_python spun on a container ring"
    assert produced, "generate_python failed instead of returning a script"
    _compile_check(produced[0])


def test_the_convex_case_is_still_grouped_into_one_function_per_instance():
    """The new check must not take grouping from graphs that deserve it."""
    from app.core.codegen import generate_python

    nodes, edges, subgraphs = _subgraph_export_graph()
    script = generate_python(nodes, edges, subgraphs=subgraphs)
    assert len(_subgraph_function_names(script)) == 2
    assert "inlined" not in script
