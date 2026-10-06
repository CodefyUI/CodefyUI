"""An exported script carries a copy of each preset its Map nodes run.

Found during the browser e2e of the fixes for #618-#625. A Map whose
``subgraph`` named the student's own preset ran on the canvas, and its
exported script, run the way the judge runs it -- ``python -I x.py`` in an
empty folder with no CODEFYUI_* variable -- stopped with ``Subgraph '<name>'
not found``. Preset CARDS are expanded at export time, but Map looks its
preset up by name when it runs, and the judge's machine has none of the
presets a student saved. The script now carries a copy of each such preset
(``_PRESET_COPIES``) and uses it before any preset of its name where it runs.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.core import runtime as runtime_module
from app.core.codegen import generate_python
from app.core.execution_context import ExecutionContext
from app.core.graph_engine import (
    GraphValidationError,
    build_preset_fallback,
    execute_graph,
    validate_graph,
)
from app.core.preset_registry import preset_registry
from app.schemas.models import PresetDefinition

RUN_TIMEOUT_S = 120

#: Names no installation has, so the judge's registry cannot answer for them.
BODY = "F13 map body"
OUTER = "F13 map of maps"


def _print_chain(name: str, *labels: str) -> PresetDefinition:
    """A preset of Print nodes, one per label, wired in a chain. Map feeds
    each item to the first and collects the last one's pass-through."""
    ids = [f"node_{index}" for index in range(len(labels))]
    return PresetDefinition.model_validate({
        "preset_name": name,
        "category": "Custom",
        "nodes": [
            {"id": node_id, "type": "Print", "params": {"label": label}}
            for node_id, label in zip(ids, labels)
        ],
        "edges": [
            {"source": a, "sourceHandle": "value", "target": b, "targetHandle": "value"}
            for a, b in zip(ids, ids[1:])
        ],
        "exposed_inputs": [
            {"name": "value", "internal_node": ids[0], "internal_port": "value"}],
        "exposed_outputs": [
            {"name": "value", "internal_node": ids[-1], "internal_port": "value"}],
    })


def _map_preset(name: str, runs: str) -> PresetDefinition:
    """A preset that is one Map running the preset *runs* over its input."""
    return PresetDefinition.model_validate({
        "preset_name": name,
        "category": "Custom",
        "nodes": [{"id": "node_0", "type": "Map", "params": {"subgraph": runs}}],
        "edges": [],
        "exposed_inputs": [
            {"name": "items", "internal_node": "node_0", "internal_port": "items"}],
        "exposed_outputs": [
            {"name": "results", "internal_node": "node_0", "internal_port": "results"}],
    })


def _node(node_id: str, node_type: str, **params) -> dict:
    return {"id": node_id, "type": node_type, "position": {"x": 0, "y": 0},
            "data": {"params": params}}


def _edge(edge_id: str, source: str, source_handle: str, target: str,
          target_handle: str) -> dict:
    return {"id": edge_id, "source": source, "sourceHandle": source_handle,
            "target": target, "targetHandle": target_handle, "type": "data"}


def _map_graph(preset: str, items: str = "[1, 2]") -> tuple[list[dict], list[dict]]:
    """The e2e's canvas: Start -> a script returning *items* -> Map running
    *preset* -> Print."""
    nodes = [
        _node("start", "Start"),
        _node("items", "PythonScript",
              code=f"def run(inputs, params):\n    return {{'out1': {items}}}\n",
              input_ports=1, output_ports=1),
        _node("map", "Map", subgraph=preset),
        _node("show", "Print", label="results"),
    ]
    edges = [
        {"id": "t", "source": "start", "sourceHandle": "trigger",
         "target": "items", "targetHandle": "", "type": "trigger"},
        _edge("a", "items", "out1", "map", "items"),
        _edge("b", "map", "results", "show", "value"),
    ]
    return nodes, edges


@pytest.fixture
def install(monkeypatch):
    """Install a preset in this process's registry, as Export as Subgraph
    installs one on the student's machine. Undone after the test, together
    with whatever else was put under that name meanwhile."""
    def install(preset: PresetDefinition) -> None:
        monkeypatch.setitem(preset_registry._presets, preset.preset_name, preset)
    return install


async def _canvas_stdout(nodes: list[dict], edges: list[dict], capsys) -> str:
    """What a Run of the canvas prints, run by the engine itself."""
    capsys.readouterr()
    await execute_graph(nodes, edges, context=ExecutionContext())
    return capsys.readouterr().out


def _run_like_the_judge(
    script: str,
    tmp_path: Path,
    *,
    installed_there: tuple[PresetDefinition, ...] = (),
) -> subprocess.CompletedProcess[str]:
    """``python -I x.py`` in an empty folder, with no argument and none of
    this process's CODEFYUI_* variables: the user data dir and the user
    presets folder are empty ones, as on a machine CodefyUI was just
    installed on. *installed_there* are preset files to put in that folder.
    """
    folder = tmp_path / "judge"
    folder.mkdir(parents=True)
    (folder / "x.py").write_text(script, encoding="utf-8")
    presets_dir = tmp_path / "judge-user-presets"
    presets_dir.mkdir()
    for index, preset in enumerate(installed_there):
        (presets_dir / f"preset_{index}.json").write_text(
            preset.model_dump_json(), encoding="utf-8")
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.upper().startswith("CODEFYUI_")
    }
    env["CODEFYUI_USER_DATA_DIR"] = str(tmp_path / "judge-user-data")
    env["CODEFYUI_USER_PRESETS_DIR"] = str(presets_dir)
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, "-I", "x.py"],
        cwd=folder,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=RUN_TIMEOUT_S,
        check=False,
    )


def _copies(script: str) -> list[dict] | None:
    """The script's ``_PRESET_COPIES``, read as data; None without one."""
    for statement in ast.parse(script).body:
        if (
            isinstance(statement, ast.Assign)
            and [getattr(t, "id", None) for t in statement.targets]
            == ["_PRESET_COPIES"]
        ):
            return ast.literal_eval(statement.value)
    return None


# -- The judge runs what the canvas ran --------------------------------------


@pytest.mark.asyncio
async def test_a_map_on_the_students_own_preset_runs_where_it_is_not_installed(
    tmp_path: Path, capsys, install,
):
    """The e2e failure, end to end: same stdout as the canvas, exit 0."""
    install(_print_chain(BODY, "first", "last"))
    nodes, edges = _map_graph(BODY)
    canvas = await _canvas_stdout(nodes, edges, capsys)
    # The body ran once per item, so the comparison below is about something.
    assert canvas == (
        "[first] 1\n[last] 1\n[first] 2\n[last] 2\n[results] [1, 2]\n")

    run = _run_like_the_judge(generate_python(nodes, edges, name="map"), tmp_path)

    assert run.returncode == 0, run.stderr
    assert run.stdout == canvas


@pytest.mark.asyncio
async def test_the_copy_runs_in_place_of_an_installed_preset_of_its_name(
    tmp_path: Path, capsys, install,
):
    """The copy in the file comes first: a preset of the same name installed
    where the script runs does not change what it prints."""
    install(_print_chain(BODY, "first", "last"))
    nodes, edges = _map_graph(BODY)
    canvas = await _canvas_stdout(nodes, edges, capsys)
    script = generate_python(nodes, edges, name="map")
    impostor = _print_chain(BODY, "impostor")

    run = _run_like_the_judge(script, tmp_path / "with-copy", installed_there=(impostor,))
    # The same file without the call that puts its copies in place: proof
    # that the judge's registry really has the impostor under that name.
    without = _run_like_the_judge(
        script.replace("    _use_preset_copies()\n", ""),
        tmp_path / "without-copy", installed_there=(impostor,))

    assert run.returncode == 0, run.stderr
    assert run.stdout == canvas
    assert without.returncode == 0, without.stderr
    assert "[impostor] 1" in without.stdout


@pytest.mark.asyncio
async def test_a_preset_whose_body_holds_a_map_carries_that_maps_preset_too(
    tmp_path: Path, capsys, install,
):
    install(_print_chain(BODY, "inner"))
    install(_map_preset(OUTER, runs=BODY))
    nodes, edges = _map_graph(OUTER, items="[[1, 2], [3]]")
    canvas = await _canvas_stdout(nodes, edges, capsys)
    assert canvas == (
        "[inner] 1\n[inner] 2\n[inner] 3\n[results] [[1, 2], [3]]\n")
    script = generate_python(nodes, edges, name="nested")

    run = _run_like_the_judge(script, tmp_path)

    assert [preset["preset_name"] for preset in _copies(script)] == [OUTER, BODY]
    assert run.returncode == 0, run.stderr
    assert run.stdout == canvas


# -- What is copied, and what is refused --------------------------------------


def _unknown_presets(errors: list[str]) -> list[tuple[str, str]]:
    """``(node_id, preset)`` of each ``unknown_preset`` finding in *errors*."""
    return [
        (issue.node_id, issue.params["preset"])
        for issue in errors
        if getattr(issue, "code", None) == "unknown_preset"
    ]


def test_a_map_naming_a_preset_this_server_does_not_have_refuses_the_export():
    """Refused before a script is written: the canvas cannot run that Map
    either, and a script that fails on the judge's machine is worse."""
    nodes, edges = _map_graph("no such preset")

    with pytest.raises(GraphValidationError) as refused:
        generate_python(nodes, edges)

    assert str(refused.value) == (
        "Unknown preset: no such preset (named by the 'subgraph' setting of "
        "node map)")


def test_validation_names_the_map_whose_preset_is_missing(install):
    """Run and Export both ask validation first and show its findings in the
    user's language: ``unknown_preset``, the code a card naming a missing
    preset gets, on the node the canvas shows."""
    nodes, edges = _map_graph("no such preset")
    assert _unknown_presets(validate_graph(nodes, edges)) == [("map", "no such preset")]

    # Installed: nothing to say. An empty name names no preset.
    install(_print_chain(BODY, "first"))
    for name in (BODY, ""):
        nodes, edges = _map_graph(name)
        assert _unknown_presets(validate_graph(nodes, edges)) == []

    # The graph's own presets[] does not count: Map never reads it.
    nodes, edges = _map_graph("only the graph has it")
    owned = _print_chain("only the graph has it", "owned").model_dump()
    assert _unknown_presets(validate_graph(
        nodes, edges, preset_fallback=build_preset_fallback([owned]),
    )) == [("map", "only the graph has it")]

    # A Map inside a preset a Map runs, named as Map names its body's nodes.
    install(_map_preset(OUTER, runs="no such preset"))
    nodes, edges = _map_graph(OUTER)
    assert _unknown_presets(validate_graph(nodes, edges)) == [
        ("map__node_0", "no such preset")]

    # A Map in a block, named as the block's inner node.
    nodes, edges = _map_graph("no such preset")
    map_node = nodes.pop(2)
    block = {
        "id": "mapper", "name": "Mapper", "nodes": [map_node], "edges": [],
        "interface": {
            "inputs": [{"port": "items", "innerNode": "map", "innerPort": "items"}],
            "outputs": [{"port": "results", "innerNode": "map", "innerPort": "results"}],
        },
    }
    nodes.append(_node("blk", "subgraph:mapper"))
    edges = [edges[0], _edge("a", "items", "out1", "blk", "items"),
             _edge("b", "blk", "results", "show", "value")]
    assert _unknown_presets(validate_graph(nodes, edges, subgraphs=[block])) == [
        ("blk/map", "no such preset")]


@pytest.mark.asyncio
async def test_the_validate_route_gives_run_and_export_the_coded_finding(test_client):
    """What the toolbar's Run and Export show their toasts from."""
    nodes, edges = _map_graph("no such preset")

    response = await test_client.post(
        "/api/graph/validate", json={"nodes": nodes, "edges": edges})

    body = response.json()
    assert response.status_code == 200
    assert body["valid"] is False
    assert {
        "code": "unknown_preset",
        "node_id": "map",
        "params": {"preset": "no such preset"},
        "message": "Unknown preset: no such preset (named by the 'subgraph' "
                   "setting of node map)",
    } in body["issues"]


@pytest.mark.asyncio
async def test_a_run_is_refused_before_any_node_runs(capsys):
    """Refused up front, as a card naming a missing preset is, rather than
    failing at the Map after the nodes before it ran."""
    nodes, edges = _map_graph("no such preset")
    nodes[1]["data"]["params"]["code"] = (
        "def run(inputs, params):\n    print('items ran')\n    return {'out1': [1]}\n")

    with pytest.raises(GraphValidationError, match="Unknown preset: no such preset"):
        await execute_graph(nodes, edges, context=ExecutionContext())

    assert "items ran" not in capsys.readouterr().out


def test_map_reads_the_installed_preset_never_the_graphs_own_copy(install):
    """On the canvas Map looks among the installed presets only (presets.md,
    "Portable definitions"), so the export copies the installed one, and a
    name only the graph's own ``presets[]`` has is refused like any other."""
    owned = _print_chain(BODY, "owned").model_dump()
    nodes, edges = _map_graph(BODY)
    with pytest.raises(GraphValidationError):
        generate_python(nodes, edges, presets=[owned])

    install(_print_chain(BODY, "installed"))
    script = generate_python(nodes, edges, presets=[owned])
    [copied] = _copies(script)
    assert [node["params"]["label"] for node in copied["nodes"]] == ["installed"]


@pytest.mark.asyncio
async def test_the_export_route_carries_the_copy_and_refuses_a_missing_preset(
    test_client, install,
):
    install(_print_chain(BODY, "first"))
    nodes, edges = _map_graph(BODY)
    response = await test_client.post(
        "/api/graph/export", json={"name": "map", "nodes": nodes, "edges": edges})
    assert response.status_code == 200, response.text
    assert [p["preset_name"] for p in _copies(response.json()["script"])] == [BODY]

    nodes, edges = _map_graph("no such preset")
    response = await test_client.post(
        "/api/graph/export", json={"name": "map", "nodes": nodes, "edges": edges})
    assert response.status_code == 400
    assert "Unknown preset: no such preset" in response.json()["detail"]


@pytest.mark.asyncio
async def test_an_absolute_path_in_a_copied_preset_is_warned_about(test_client, install):
    """The script carries the copy with its file params, so a path in one
    pins it to this machine as much as a path on the canvas (#557)."""
    path = r"C:\Users\student01\Desktop\grades.csv"
    reader = _print_chain(BODY, "first").model_dump()
    reader["nodes"].append({"id": "csv", "type": "CSVReader", "params": {"path": path}})
    install(PresetDefinition.model_validate(reader))
    install(_map_preset(OUTER, runs=BODY))

    for preset, node_id, levels in (
        (BODY, "map__csv", [{"node_id": "map", "label": BODY}]),
        (OUTER, "map__node_0__csv", [{"node_id": "map", "label": OUTER},
                                     {"node_id": "map__node_0", "label": BODY}]),
    ):
        nodes, edges = _map_graph(preset)
        response = await test_client.post(
            "/api/graph/export", json={"name": "map", "nodes": nodes, "edges": edges})
        assert response.status_code == 200, response.text
        assert response.json()["warnings"] == [{
            "code": "absolute_path", "node_id": node_id, "label": "CSVReader",
            "param": "path", "value": path, "containers": levels,
        }]


def test_a_graph_that_names_no_preset_gets_no_new_text():
    """Exam scripts are compared by their output, and graphs with no Map
    preset must export exactly as before: nothing of this is emitted."""
    plain_nodes, plain_edges = _map_graph(BODY)
    plain_nodes[2]["data"]["params"]["subgraph"] = ""

    for nodes in (plain_nodes, [n for n in plain_nodes if n["id"] != "map"]):
        ids = {node["id"] for node in nodes}
        edges = [e for e in plain_edges if e["source"] in ids and e["target"] in ids]
        script = generate_python(nodes, edges, name="plain")
        assert _copies(script) is None
        assert "_use_preset_copies" not in script
        assert "_PRESET_COPIES" not in script


def test_a_copied_presets_node_types_are_checked_at_startup(install):
    """Discovery loads a bundled plugin pack only for the types the script
    lists, and names a missing type before any node runs: so the types
    inside a copied preset are listed with the graph's own."""
    preset = _print_chain(BODY, "first").model_dump()
    preset["nodes"].append({"id": "zeros", "type": "TensorCreate", "params": {}})
    install(PresetDefinition.model_validate(preset))
    nodes, edges = _map_graph(BODY)

    script = generate_python(nodes, edges)

    [line] = [line for line in script.splitlines()
              if line.startswith("_REQUIRED_NODE_TYPES = ")]
    required = ast.literal_eval(line.split(" = ", 1)[1])
    assert required == ["Map", "Print", "PythonScript", "Start", "TensorCreate"]


def test_a_secret_in_a_copied_preset_is_blanked_and_the_installed_one_kept(install):
    secret = "sk-F13-COPIED-PRESET-SECRET"
    preset = PresetDefinition.model_validate({
        "preset_name": BODY,
        "nodes": [{"id": "chat", "type": "LLMChat",
                   "params": {"provider": "ChatGPT API", "openai_api_key": secret}}],
        "edges": [],
        "exposed_inputs": [{"name": "text", "internal_node": "chat", "internal_port": "text"}],
        "exposed_outputs": [{"name": "text", "internal_node": "chat", "internal_port": "text"}],
    })
    install(preset)
    nodes, edges = _map_graph(BODY)

    script = generate_python(nodes, edges)

    assert secret not in script
    assert _copies(script)[0]["nodes"][0]["params"]["openai_api_key"] == ""
    assert preset_registry.get(BODY).nodes[0].params["openai_api_key"] == secret


def test_a_map_inside_a_block_or_a_card_carries_its_preset(install):
    """A Map sits anywhere a node can: in a collapsed block (the graph's
    ``subgraphs[]``) or inside a card whose definition the graph owns
    (``presets[]``). Both are opened before the export looks."""
    install(_print_chain(BODY, "first"))
    nodes, edges = _map_graph(BODY)
    map_node = nodes.pop(2)

    block_nodes = [*nodes, _node("blk", "subgraph:mapper")]
    block_edges = [edges[0], _edge("a", "items", "out1", "blk", "items"),
                   _edge("b", "blk", "results", "show", "value")]
    block = {
        "id": "mapper",
        "name": "Mapper",
        "nodes": [map_node],
        "edges": [],
        "interface": {
            "inputs": [{"port": "items", "innerNode": "map", "innerPort": "items"}],
            "outputs": [{"port": "results", "innerNode": "map", "innerPort": "results"}],
        },
    }
    in_block = generate_python(block_nodes, block_edges, subgraphs=[block])

    card_nodes = [*nodes, _node("card", "preset:F13 card")]
    card_edges = [edges[0], _edge("a", "items", "out1", "card", "items"),
                  _edge("b", "card", "results", "show", "value")]
    card = _map_preset("F13 card", runs=BODY).model_dump()
    in_card = generate_python(card_nodes, card_edges, presets=[card])

    for script in (in_block, in_card):
        assert [preset["preset_name"] for preset in _copies(script)] == [BODY]


def test_a_hostile_preset_name_stays_data():
    """The name reaches the file only as an ``ascii()`` literal."""
    name = "x'] + __import__('os').getcwd() + ['\\\n\N{LATIN SMALL LETTER E WITH ACUTE}"
    preset = _print_chain(name, "first")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(preset_registry._presets, name, preset)
        nodes, edges = _map_graph(name)
        script = generate_python(nodes, edges)

    compile(script, "<generated>", "exec")
    assert script.isascii()
    assert _copies(script)[0]["preset_name"] == name


# -- The runtime side ---------------------------------------------------------


def test_use_preset_copies_puts_each_copy_in_place_of_an_installed_one(install):
    # `install` also takes the copy out again after the test.
    install(_print_chain(BODY, "installed"))

    runtime_module.use_preset_copies([_print_chain(BODY, "copy").model_dump()])

    assert preset_registry.get(BODY).nodes[0].params == {"label": "copy"}


def test_a_script_run_by_a_codefyui_without_the_new_call_runs_as_before(
    install, monkeypatch,
):
    """A file exported now, run by a CodefyUI older than this change: its
    Map looks among the presets installed there, as every file did before."""
    install(_print_chain(BODY, "installed"))
    nodes, edges = _map_graph(BODY)
    module: dict = {"__name__": "exported_graph_on_an_older_backend"}
    exec(compile(generate_python(nodes, edges), "<exported graph>", "exec"), module)  # noqa: S102
    monkeypatch.setitem(preset_registry._presets, BODY, _print_chain(BODY, "older"))
    monkeypatch.delattr(runtime_module, "use_preset_copies")

    assert module["_use_preset_copies"]() is None
    assert preset_registry.get(BODY).nodes[0].params == {"label": "older"}
