"""The exported Python script's contract with course graders (#558).

Course exams grade the file Export Python produces on a traditional online
judge: the grader runs ``python <file>.py`` in a folder holding the test
files, with no arguments and no environment variables set, compares stdout,
and reads the same file with ``ast`` to check which nodes a student used.
Four properties of the exported script carry that, and 3.0.0 is the release
later ones must stay compatible with, so each has a test here that fails when
it changes:

1. Shape. Every node is one function whose last statement is
   ``return _call('<Type>', '<node id>', <params>, ctx, ...)``, and
   ``<params>`` takes one of two forms: the name ``params``, bound by a
   top-level ``params = <pure literal>`` line in the same function, or the
   literal ``{}`` inline when the node has no params. A grader's reader
   starts each function's params at ``{}`` and takes the top-level
   assignment when there is one, so the two forms read alike. GraphInput's
   function also rebinds ``params`` inside ``if value is not _ABSENT:`` to
   merge a value given with --inputs-json; that statement is not a literal
   and not part of the contract, so a reader takes the top-level assignment.
   A node inside a preset card or a block is emitted under its expanded id,
   ``<card id>__<inner id>`` or ``<instance id>/<inner id>``, with the params
   it runs with. ``ast.parse`` + ``ast.literal_eval`` recover every node's
   type, id and params without running anything.
2. Output. With a GraphOutput, stdout holds exactly one non-empty line, one
   JSON object with sorted keys, because a judge may compare the whole of
   stdout. A tensor is ``{"__type__": "tensor", "dtype", "shape", "values"}``.
   Print text and the completion line go to stderr.
3. Exit codes. 0 on success, 1 when a node fails at run time, 2 when setup
   fails: a node type this install lacks, or a GraphOutput no path from Start
   reaches.
4. Files. CSVReader, ImageReader and DocumentLoader find a bare file name in
   the working directory.

Graphs and data here are synthetic and written by the tests. Exam questions,
answers and hidden test data must never enter this public repository.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.node_registry import registry

GRAPH_NAME = "export-contract"
SCRIPT_NAME = "contract_graph.py"
CSV_NAME = "contract_table.csv"
IMAGE_NAME = "contract_pixels.png"
NOTES_NAME = "contract_notes.txt"
NOTES_TEXT = "Read from the working directory."
PRINT_MARKER = "print-goes-to-stderr"

SCRIPT_PARAMS = {
    "code": (
        "def run(inputs, params):\n"
        "    return {'out1': inputs['in1'] * inputs['in2']}\n"
    ),
    "input_ports": 2,
    "output_ports": 1,
    "input_types": "TENSOR",
    "output_types": "ANY",
}

#: The whole answer line for the data `_write_data_files` writes. A judge may
#: compare this line as text, so separators, the dtype spelling and float
#: formatting are part of the contract along with the values.
EXPECTED_LAST_LINE = (
    '{"notes": ["Read from the working directory."], '
    '"pixels": {"__type__": "tensor", "dtype": "torch.float32", '
    '"shape": [1, 2, 2], "values": [[[0.0, 1.0], [1.0, 0.0]]]}, '
    '"table": {"__type__": "tensor", "dtype": "torch.float32", '
    '"shape": [2, 2], "values": [[1.5, -2.25], [3.0, 4.0]]}}'
)

#: Torch alone takes seconds to import, and a cold Windows runner is slow.
RUN_TIMEOUT_S = 120

#: Where a stray network call goes: nothing listens on the discard port, so
#: the call fails at once instead of hanging or quietly reaching the internet,
#: which a grading machine may not have.
DEAD_PROXY = "http://127.0.0.1:9"

#: A plugin node type, registered only in the process that exports.
EXPORTER_ONLY_TYPE = "contract_plugin:OnlyOnTheExportingMachine"

#: The boxed-up variant of the contract graph (`_boxed_contract_graph`).
PRESET_NAME = "Export Contract Peek"
PRESET_DEFAULT_LABEL = "the definition's own label"
PEEK_LABEL = "peek-through-a-preset"
SUBGRAPH_ID = "scale-and-show"
#: Not ASCII, so the source spells it escaped and literal_eval has to undo
#: that for the expanded id to come back exactly.
SHOW_ID = "show-結果"


class _ExporterOnlyNode(BaseNode):
    """Stands in for a plugin the grading machine never installed."""

    NODE_NAME = "OnlyOnTheExportingMachine"
    CATEGORY = "Test"
    DESCRIPTION = "Registered in the exporting process and nowhere else"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        return {"value": None}


# ── Graphs and data ──────────────────────────────────────────────────────


def _node(node_id: str, node_type: str, params: dict[str, Any]) -> dict:
    return {
        "id": node_id,
        "type": node_type,
        "position": {"x": 0, "y": 0},
        "data": {"params": params},
    }


def _trigger(source: str, target: str) -> dict:
    return {
        "id": f"trigger-{target}",
        "source": source,
        "target": target,
        "sourceHandle": "trigger",
        "targetHandle": "",
        "type": "trigger",
    }


def _wire(source: str, source_port: str, target: str, target_port: str) -> dict:
    return {
        "id": f"{source}.{source_port}-{target}.{target_port}",
        "source": source,
        "target": target,
        "sourceHandle": source_port,
        "targetHandle": target_port,
        "type": "data",
    }


def _contract_graph() -> dict:
    """An exam-shaped graph that reaches every way codegen emits a node.

    Start has no params, so its ``_call`` gets ``{}`` inline. The three
    readers name bare files. GraphInput is the one function that rebinds
    ``params`` after the literal, and PythonScript's CODE param is emitted one
    source line per string literal. The GraphOutputs are declared in an order
    that is NOT sorted, or a runner that dropped ``sort_keys`` would still
    pass, and two descriptions are Chinese, as a student in Taiwan would type
    them.
    """
    return {
        "name": GRAPH_NAME,
        "nodes": [
            _node("start", "Start", {}),
            _node("csv-7f3a", "CSVReader", {
                "path": CSV_NAME,
                "target_column": "",
                "include_columns": "",
                "skip_header": True,
            }),
            _node("image-41c2", "ImageReader", {
                "path": IMAGE_NAME, "mode": "L", "resize": 0,
            }),
            _node("docs-9b10", "DocumentLoader", {
                "source": "uploaded_file",
                "directory": "data/samples/rag",
                "recursive": False,
                "file": NOTES_NAME,
                "max_docs": 0,
            }),
            _node("factor-5d2e", "GraphInput", {
                "name": "factor",
                "type": "number",
                "required": False,
                "default": "2",
                "description": "放大倍數",
            }),
            _node("script-c0de", "PythonScript", dict(SCRIPT_PARAMS)),
            _node("print-e1", "Print", {"label": PRINT_MARKER}),
            _node("out-table", "GraphOutput", {
                "name": "table", "description": "特徵表格",
            }),
            _node("out-pixels", "GraphOutput", {"name": "pixels", "description": ""}),
            _node("out-notes", "GraphOutput", {"name": "notes", "description": ""}),
        ],
        "edges": [
            _trigger("start", "csv-7f3a"),
            _trigger("start", "image-41c2"),
            _trigger("start", "docs-9b10"),
            _trigger("start", "factor-5d2e"),
            _wire("csv-7f3a", "tensor", "script-c0de", "in1"),
            _wire("factor-5d2e", "value", "script-c0de", "in2"),
            _wire("script-c0de", "out1", "print-e1", "value"),
            _wire("csv-7f3a", "tensor", "out-table", "value"),
            _wire("image-41c2", "tensor", "out-pixels", "value"),
            _wire("docs-9b10", "texts", "out-notes", "value"),
        ],
    }


def _boxed_contract_graph() -> dict:
    """The contract graph with two parts boxed up, as a student might.

    PythonScript and Print are collapsed into a block, and the image passes
    through a preset card -- one Print, which hands its value on unchanged --
    on its way to its GraphOutput. Both definitions travel in the request,
    the way the editor sends a document's own. The card's internalParams
    override the definition's label, so the params the node runs with are not
    the definition's. Nothing computed changes, so the answer must not either.
    """
    graph = _contract_graph()
    boxed = {"script-c0de", "print-e1"}
    graph["nodes"] = [node for node in graph["nodes"] if node["id"] not in boxed]
    graph["edges"] = [
        edge for edge in graph["edges"]
        if edge["source"] not in boxed
        and edge["target"] not in boxed
        and edge["target"] != "out-pixels"
    ]
    card = _node("card-3b", f"preset:{PRESET_NAME}", {})
    card["data"]["internalParams"] = {"peek": {"label": PEEK_LABEL}}
    graph["nodes"] += [card, _node("block-7e", f"subgraph:{SUBGRAPH_ID}", {})]
    graph["edges"] += [
        _wire("image-41c2", "tensor", "card-3b", "value"),
        _wire("card-3b", "value", "out-pixels", "value"),
        _wire("csv-7f3a", "tensor", "block-7e", "in1"),
        _wire("factor-5d2e", "value", "block-7e", "in2"),
    ]

    def exposed(name: str) -> dict:
        return {
            "name": name,
            "internal_node": "peek",
            "internal_port": "value",
            "data_type": "ANY",
            "description": "",
        }

    graph["presets"] = [{
        "preset_name": PRESET_NAME,
        "category": "Custom",
        "description": "",
        "tags": [],
        "nodes": [{
            "id": "peek", "type": "Print", "params": {"label": PRESET_DEFAULT_LABEL},
        }],
        "edges": [],
        "exposed_inputs": [exposed("value")],
        "exposed_outputs": [exposed("value")],
        "exposed_params": [],
    }]
    graph["subgraphs"] = [{
        "id": SUBGRAPH_ID,
        "name": "Scale and show",
        "description": "",
        "nodes": [
            _node("mul", "PythonScript", dict(SCRIPT_PARAMS)),
            _node(SHOW_ID, "Print", {"label": PRINT_MARKER}),
        ],
        "edges": [_wire("mul", "out1", SHOW_ID, "value")],
        "interface": {
            "inputs": [
                {"port": "in1", "innerNode": "mul", "innerPort": "in1",
                 "data_type": "TENSOR"},
                {"port": "in2", "innerNode": "mul", "innerPort": "in2",
                 "data_type": "TENSOR"},
            ],
            "outputs": [],
            "triggerTargets": [],
        },
    }]
    return graph


def _write_data_files(folder: Path, *, csv: bool = True) -> None:
    """The test files a grader puts beside the script, as exact bytes."""
    folder.mkdir(parents=True, exist_ok=True)
    if csv:
        (folder / CSV_NAME).write_bytes(b"a,b\n1.5,-2.25\n3,4\n")
    image = Image.new("L", (2, 2))
    image.putdata([0, 255, 255, 0])
    image.save(folder / IMAGE_NAME)
    (folder / NOTES_NAME).write_bytes(NOTES_TEXT.encode("ascii"))


# ── Export and run ───────────────────────────────────────────────────────


async def _export(client, graph: dict) -> str:
    """The script the editor's Export Python button would download."""
    response = await client.post("/api/graph/export", json=graph)
    assert response.status_code == 200, response.text
    return response.json()["script"]


@dataclass
class _Run:
    returncode: int
    stdout: str
    stderr: str


def _grader_env(home: Path) -> dict[str, str]:
    """This process's environment as a grading machine would have it.

    No CODEFYUI_* variable survives -- a grader sets none, and an inherited
    CODEFYUI_PROJECT_DIR alone would move where relative paths resolve.
    CodefyUI's own state then goes to *home*: the user data dir, and empty
    upload stores, so a file a developer once uploaded under one of these
    names cannot answer for the copy in the working directory. data_root()
    follows MODELS_DIR, so anything a run writes lands in *home* too.
    """
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.upper().startswith("CODEFYUI_")
    }
    env["CODEFYUI_USER_DATA_DIR"] = str(home / "user-data")
    env["CODEFYUI_DATA_FILES_DIR"] = str(home / "data" / "files")
    env["CODEFYUI_IMAGES_DIR"] = str(home / "data" / "images")
    env["CODEFYUI_MODELS_DIR"] = str(home / "data" / "models")
    env["CODEFYUI_MEDIA_DIR"] = str(home / "data" / "media")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        env[name] = DEAD_PROXY
        if os.name != "nt":
            # POSIX names are case-sensitive and urllib reads the lowercase
            # spelling first; on Windows the two are one variable.
            env[name.lower()] = DEAD_PROXY
    env.pop("NO_PROXY", None)
    env.pop("no_proxy", None)
    return env


def _run_like_a_grader(script: str, folder: Path, home: Path) -> _Run:
    """``python <file>.py`` inside *folder*, the way the grader runs it."""
    # Bytes, so the file is exactly what the browser would save.
    (folder / SCRIPT_NAME).write_bytes(script.encode("utf-8"))
    completed = subprocess.run(
        # -I so no PYTHON* variable of the developer's steers the run: a
        # PYTHONPATH naming another checkout would pick which `app` the script
        # imports, and PYTHONUTF8 would change how stdout is encoded. A grader
        # sets none. The one other difference, the script's folder on
        # sys.path, holds only data here.
        [sys.executable, "-I", SCRIPT_NAME],
        cwd=folder,
        env=_grader_env(home),
        capture_output=True,
        timeout=RUN_TIMEOUT_S,
        check=False,
    )
    return _Run(
        returncode=completed.returncode,
        # Strict on stdout, which the grader parses; stderr only carries
        # diagnostics, and a temp path in it need not be ASCII.
        stdout=completed.stdout.decode("utf-8"),
        stderr=completed.stderr.decode("utf-8", errors="replace"),
    )


def _answer_line(run: _Run) -> str:
    """The answer, which must be all there is on stdout: a judge may compare
    the whole stream, so one stray line ahead of it fails every submission."""
    lines = [line for line in run.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, (
        f"stdout must hold the answer line alone, got {len(lines)} lines:\n"
        f"{run.stdout}\nstderr:\n{run.stderr}")
    return lines[0]


# ── Reading the source the way a grader does ─────────────────────────────


def _returned_call(statement: ast.stmt) -> ast.Call | None:
    """The ``_call(...)`` a node function returns; None for any other def."""
    if not isinstance(statement, ast.FunctionDef) or not statement.body:
        return None
    last = statement.body[-1]
    if (
        isinstance(last, ast.Return)
        and isinstance(last.value, ast.Call)
        and isinstance(last.value.func, ast.Name)
        and last.value.func.id == "_call"
    ):
        return last.value
    return None


def _params_assignments(function: ast.FunctionDef) -> list[ast.Assign]:
    """``params = ...`` statements directly in the function body.

    Top level only, which leaves out GraphInput's rebinding inside its
    ``if``: that one is not a literal, and a grader passes no arguments.
    """
    return [
        statement
        for statement in function.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "params"
    ]


def _read_like_a_grader(script: str) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """``{node id: {"type", "params"}}``, by ast + literal_eval only.

    Nothing is executed or imported. Returns the view and the shape problems
    found, each naming the function it is in.
    """
    view: dict[str, dict[str, Any]] = {}
    problems: list[str] = []

    def literal(node: ast.AST, where: str) -> Any:
        try:
            return ast.literal_eval(node)
        except ValueError:
            problems.append(f"{where} is not a pure literal: {ast.unparse(node)}")
            return None

    for statement in ast.parse(script).body:
        call = _returned_call(statement)
        if call is None:
            continue
        where = f"def {statement.name}"
        if len(call.args) < 4:
            problems.append(f"{where}: _call has {len(call.args)} positional arguments")
            continue
        type_arg, id_arg, params_arg, ctx_arg = call.args[:4]
        node_type = literal(type_arg, f"{where}: the node type")
        node_id = literal(id_arg, f"{where}: the node id")
        if not (isinstance(node_type, str) and isinstance(node_id, str)):
            problems.append(f"{where}: the node type and id must be strings")
            continue
        if not (isinstance(ctx_arg, ast.Name) and ctx_arg.id == "ctx"):
            problems.append(f"{where}: the fourth argument is not ctx")

        assignments = _params_assignments(statement)
        if isinstance(params_arg, ast.Name) and params_arg.id == "params":
            # Form one: the literal a top-level `params = ...` line binds.
            if len(assignments) == 1:
                params = literal(assignments[0].value, f"{where}: params")
            else:
                problems.append(
                    f"{where}: {len(assignments)} top-level params assignments, not 1")
                params = None
        else:
            # Form two: a node with no params passes the literal {} inline.
            params = literal(params_arg, f"{where}: the inline params")
            if assignments:
                problems.append(f"{where}: assigns params but does not pass them")
            elif params is not None and params != {}:
                problems.append(f"{where}: non-empty params passed inline")

        if node_id in view:
            problems.append(f"{where}: node {node_id!r} already has a function")
        view[node_id] = {"type": node_type, "params": params}
    return view, problems


# ── 1. Shape ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_grader_reads_every_node_back_without_running_it(test_client):
    """Each node's type, id and params, from the source alone.

    Both forms of params are read the way a grader reads them: ``{}`` unless
    a top-level ``params = <literal>`` line binds them, and never from
    GraphInput's rebinding inside its ``if``.
    """
    graph = _contract_graph()
    script = await _export(test_client, graph)

    # A grader opens the file with its platform's default encoding.
    assert script.isascii(), "the exported script has a non-ASCII character"

    view, problems = _read_like_a_grader(script)
    assert not problems, "\n".join(problems)
    assert view == {
        node["id"]: {"type": node["type"], "params": node["data"]["params"]}
        for node in graph["nodes"]
    }


@pytest.mark.asyncio
async def test_nodes_in_presets_and_blocks_read_back_under_expanded_ids(test_client):
    """A card's inner node is ``<card id>__<inner id>``, a block's is
    ``<instance id>/<inner id>``, each with the params it runs with: the
    definition's, under the card's internalParams."""
    graph = _boxed_contract_graph()
    script = await _export(test_client, graph)
    assert script.isascii(), "the exported script has a non-ASCII character"
    # The block's Print id is spelled escaped in the source, so getting it
    # back below is a real round trip through literal_eval.
    assert ascii(f"block-7e/{SHOW_ID}") in script

    view, problems = _read_like_a_grader(script)
    assert not problems, "\n".join(problems)
    on_canvas = {
        node["id"]: {"type": node["type"], "params": node["data"]["params"]}
        for node in graph["nodes"]
        if not node["type"].startswith(("preset:", "subgraph:"))
    }
    assert view == {
        **on_canvas,
        "card-3b__peek": {"type": "Print", "params": {"label": PEEK_LABEL}},
        "block-7e/mul": {"type": "PythonScript", "params": SCRIPT_PARAMS},
        "block-7e/show-結果": {"type": "Print", "params": {"label": PRINT_MARKER}},
    }


# ── 2. Output, and 4. bare file names ────────────────────────────────────


@pytest.mark.asyncio
async def test_the_answer_is_the_only_stdout_line_as_sorted_json(
    test_client, tmp_path: Path,
):
    graph = _contract_graph()
    script = await _export(test_client, graph)
    folder = tmp_path / "grader"
    _write_data_files(folder)

    run = _run_like_a_grader(script, folder, tmp_path / "home")

    # A reader that stopped looking in the working directory fails here,
    # naming its file in stderr.
    assert run.returncode == 0, run.stderr
    last = _answer_line(run)

    key_orders: list[list[str]] = []

    def remember_key_order(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        key_orders.append([key for key, _ in pairs])
        return dict(pairs)

    answer = json.loads(last, object_pairs_hook=remember_key_order)
    assert isinstance(answer, dict), f"the last stdout line is not an object: {last}"

    declared = [
        node["data"]["params"]["name"]
        for node in graph["nodes"]
        if node["type"] == "GraphOutput"
    ]
    assert declared != sorted(declared), (
        "declare the outputs unsorted, or sort_keys goes unchecked")
    assert list(answer) == sorted(declared)
    unsorted = [keys for keys in key_orders if keys != sorted(keys)]
    assert not unsorted, f"objects whose keys are not sorted: {unsorted}"

    for name in ("table", "pixels"):
        assert set(answer[name]) == {"__type__", "dtype", "shape", "values"}, answer[name]
        assert answer[name]["__type__"] == "tensor"
        assert answer[name]["dtype"] == "torch.float32", answer[name]["dtype"]

    # 4: each value is the file in the working directory, read by bare name.
    assert answer["table"]["values"] == [[1.5, -2.25], [3.0, 4.0]], (
        "CSVReader did not read the CSV in the working directory")
    assert answer["pixels"]["values"] == [[[0.0, 1.0], [1.0, 0.0]]], (
        "ImageReader did not read the PNG in the working directory")
    assert answer["notes"] == [NOTES_TEXT], (
        "DocumentLoader did not read the text file in the working directory")

    assert last == EXPECTED_LAST_LINE

    # Print text, and the "... completed on cpu (N node results)." line.
    assert f"[{PRINT_MARKER}]" in run.stderr
    assert PRINT_MARKER not in run.stdout
    assert "completed on" in run.stderr
    assert "completed on" not in run.stdout


@pytest.mark.asyncio
async def test_presets_and_blocks_leave_the_answer_unchanged(
    test_client, tmp_path: Path,
):
    script = await _export(test_client, _boxed_contract_graph())
    folder = tmp_path / "grader"
    _write_data_files(folder)

    run = _run_like_a_grader(script, folder, tmp_path / "home")

    assert run.returncode == 0, run.stderr
    assert _answer_line(run) == EXPECTED_LAST_LINE
    # Both boxes ran, the card with its own label rather than its
    # definition's, and both printed to stderr only.
    assert f"[{PEEK_LABEL}]" in run.stderr
    assert PRESET_DEFAULT_LABEL not in run.stderr
    assert f"[{PRINT_MARKER}]" in run.stderr
    assert PEEK_LABEL not in run.stdout
    assert PRINT_MARKER not in run.stdout


# ── 3. Exit codes ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_node_that_fails_at_run_time_exits_1(test_client, tmp_path: Path):
    script = await _export(test_client, _contract_graph())
    folder = tmp_path / "grader"
    _write_data_files(folder, csv=False)

    run = _run_like_a_grader(script, folder, tmp_path / "home")

    assert run.returncode == 1, run.stderr
    # Failed for the reason this test arranged: the CSV is missing.
    assert CSV_NAME in run.stderr


@pytest.mark.asyncio
async def test_a_node_type_this_install_lacks_exits_2(
    test_client, tmp_path: Path, monkeypatch,
):
    # Known to the exporting process and not to the one running the script,
    # like a plugin node the grading machine never installed.
    monkeypatch.setitem(registry._nodes, EXPORTER_ONLY_TYPE, _ExporterOnlyNode)
    graph = _contract_graph()
    graph["nodes"].append(_node("plugin-node", EXPORTER_ONLY_TYPE, {}))
    graph["edges"].append(_trigger("start", "plugin-node"))
    script = await _export(test_client, graph)
    folder = tmp_path / "grader"
    _write_data_files(folder)

    run = _run_like_a_grader(script, folder, tmp_path / "home")

    assert run.returncode == 2, run.stderr
    assert EXPORTER_ONLY_TYPE in run.stderr
    assert PRINT_MARKER not in run.stderr, "nodes ran before setup failed"


@pytest.mark.asyncio
async def test_a_graph_output_no_path_from_start_reaches_exits_2(
    test_client, tmp_path: Path,
):
    graph = _contract_graph()
    # Start no longer triggers the DocumentLoader, so nothing reaches the
    # GraphOutput named "notes". The export still succeeds; the script has
    # to refuse it.
    graph["edges"] = [
        edge for edge in graph["edges"]
        if not (edge["type"] == "trigger" and edge["target"] == "docs-9b10")
    ]
    script = await _export(test_client, graph)
    folder = tmp_path / "grader"
    _write_data_files(folder)

    run = _run_like_a_grader(script, folder, tmp_path / "home")

    assert run.returncode == 2, run.stderr
    assert "notes" in run.stderr
    assert PRINT_MARKER not in run.stderr, "nodes ran before setup failed"
