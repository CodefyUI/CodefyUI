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
