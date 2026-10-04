"""Export Python warns about file params that hold an absolute path (#557).

A value picked from an upload dropdown is a bare file name, which the reader
nodes look up wherever the exported script runs -- so the script still works
in a folder that holds the file beside it, which is how a course exam grades
it. A typed ``C:\\Users\\student01\\Desktop\\grades.csv`` is embedded as
written and the script exits 1 on any other machine.

The export still succeeds. The response carries a ``warnings`` list beside
the script, one entry per such param, naming the node the user sees and the
presets/blocks it sits in. The script itself is not touched.
"""

import pytest

ABSOLUTE = [
    r"C:\Users\student01\Desktop\grades.csv",
    "C:/Users/student01/Desktop/grades.csv",
    r"d:\exam\grades.csv",
    r"\\fileserver\share\grades.csv",
    "//fileserver/share/grades.csv",
    "/home/student01/grades.csv",
    # Rooted at the current drive on Windows: no more portable than "/...".
    r"\Users\student01\grades.csv",
    # CSVReader and DocumentLoader strip the value before opening it.
    r"  C:\exam\grades.csv",
]

NOT_ABSOLUTE = [
    "grades.csv",
    "data/grades.csv",
    r"data\grades.csv",
    "./grades.csv",
    r"..\exam\grades.csv",
    "",
]

WINDOWS_PATH = r"C:\Users\student01\Desktop\grades.csv"


def _start():
    return {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
            "data": {"params": {}}}


def _trigger(target, target_handle=""):
    return {"id": f"t-{target}", "source": "start", "target": target,
            "sourceHandle": "trigger", "targetHandle": target_handle,
            "type": "trigger"}


def _reader_graph(node_type="CSVReader", params=None, *, label=None):
    """Start -> one reader node, so the reader is what the script runs."""
    data = {"params": dict(params or {})}
    if label is not None:
        data["label"] = label
    return {
        "name": "path-warnings",
        "nodes": [
            _start(),
            {"id": "reader", "type": node_type,
             "position": {"x": 200, "y": 0}, "data": data},
        ],
        "edges": [_trigger("reader")],
    }


def _preset(name="Grade Loader", inner_type="CSVReader", params=None):
    """A graph-owned preset whose one inner node takes the Start trigger."""
    return {
        "preset_name": name,
        "category": "Test",
        "description": "",
        "tags": [],
        "nodes": [{"id": "csv", "type": inner_type,
                   "params": dict(params or {"path": "grades.csv"})}],
        "edges": [],
        "exposed_inputs": [{"name": "trigger", "internal_node": "csv",
                            "internal_port": "", "data_type": "TRIGGER",
                            "description": ""}],
        "exposed_outputs": [],
        "exposed_params": [],
    }


def _block(block_id, name, inner_nodes, trigger_targets):
    return {
        "id": block_id,
        "name": name,
        "description": "",
        "nodes": inner_nodes,
        "edges": [],
        "interface": {"inputs": [], "outputs": [],
                      "triggerTargets": trigger_targets},
    }


async def _export(test_client, graph):
    resp = await test_client.post("/api/graph/export", json=graph)
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ABSOLUTE)
async def test_each_absolute_form_warns_and_still_exports(test_client, value):
    body = await _export(test_client, _reader_graph(params={"path": value}))

    assert body["script"]
    assert body["warnings"] == [{
        "code": "absolute_path",
        "node_id": "reader",
        "label": "CSVReader",
        "param": "path",
        "value": value,
        "containers": [],
    }]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", NOT_ABSOLUTE)
async def test_a_bare_name_or_relative_path_does_not_warn(test_client, value):
    body = await _export(test_client, _reader_graph(params={"path": value}))

    assert body["warnings"] == []


@pytest.mark.asyncio
async def test_the_warning_names_a_renamed_node_by_its_label(test_client):
    body = await _export(test_client, _reader_graph(
        params={"path": WINDOWS_PATH}, label="Exam grades"))

    assert [w["label"] for w in body["warnings"]] == ["Exam grades"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("node_type", "params", "param"), [
    ("ImageReader", {"path": WINDOWS_PATH}, "path"),
    ("ModelLoader", {"path": WINDOWS_PATH}, "path"),
    ("DocumentLoader", {"source": "uploaded_file", "file": WINDOWS_PATH},
     "file"),
    ("TextCorpusDataset", {"source": "local_file", "local_path": WINDOWS_PATH},
     "local_path"),
])
async def test_every_file_param_type_is_checked(
    test_client, node_type, params, param,
):
    body = await _export(test_client, _reader_graph(node_type, params))

    assert [(w["node_id"], w["label"], w["param"], w["value"])
            for w in body["warnings"]] == [
        ("reader", node_type, param, WINDOWS_PATH)]


@pytest.mark.asyncio
@pytest.mark.parametrize(("node_type", "params"), [
    # The upload slot is hidden while the node reads a folder, so the node
    # never opens it -- and the user cannot even see the field to fix it.
    ("DocumentLoader", {"source": "directory", "file": WINDOWS_PATH}),
    ("TextCorpusDataset", {"source": "huggingface",
                           "local_path": WINDOWS_PATH}),
])
async def test_a_file_param_the_node_does_not_use_does_not_warn(
    test_client, node_type, params,
):
    body = await _export(test_client, _reader_graph(node_type, params))

    assert body["warnings"] == []


@pytest.mark.asyncio
async def test_only_params_declared_as_files_are_checked(test_client):
    """A STRING param is not a file param, whatever it holds.

    DocumentLoader's ``directory`` is a folder path typed into a text field;
    the param's definition is what decides, not its name or its value.
    """
    body = await _export(test_client, _reader_graph("DocumentLoader", {
        "source": "directory", "directory": r"C:\Users\student01\notes",
    }))

    assert body["warnings"] == []


@pytest.mark.asyncio
async def test_a_node_the_script_does_not_run_does_not_warn(test_client):
    """Drafts and bypassed nodes are not in the script, so neither is their path."""
    graph = _reader_graph(params={"path": "grades.csv"})
    graph["nodes"] += [
        # A draft: nothing triggers or reads it, so the export prunes it.
        {"id": "draft", "type": "CSVReader", "position": {"x": 0, "y": 200},
         "data": {"params": {"path": WINDOWS_PATH}}},
        # Bypassed: muted on the canvas, emitted only as a comment.
        {"id": "muted", "type": "CSVReader", "position": {"x": 0, "y": 400},
         "data": {"params": {"path": WINDOWS_PATH}, "bypassed": True}},
    ]
    graph["edges"].append(_trigger("muted"))

    body = await _export(test_client, graph)

    assert body["warnings"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["override", "definition default"])
async def test_a_path_inside_a_preset_is_found_and_named(test_client, where):
    """Override (the preset's config dialog) or the definition's own default."""
    override = {"csv": {"path": WINDOWS_PATH}} if where == "override" else {}
    default = {"path": WINDOWS_PATH if where != "override" else "grades.csv"}
    graph = {
        "name": "preset-path",
        "nodes": [
            _start(),
            {"id": "p1", "type": "preset:Grade Loader",
             "position": {"x": 200, "y": 0},
             "data": {"params": {}, "internalParams": override}},
        ],
        "edges": [_trigger("p1", "trigger")],
        "presets": [_preset(params=default)],
    }

    body = await _export(test_client, graph)

    assert body["warnings"] == [{
        "code": "absolute_path",
        "node_id": "p1__csv",
        "label": "CSVReader",
        "param": "path",
        "value": WINDOWS_PATH,
        "containers": [{"node_id": "p1", "label": "Grade Loader"}],
    }]


@pytest.mark.asyncio
async def test_a_path_inside_a_block_is_found_and_named(test_client):
    graph = {
        "name": "block-path",
        "nodes": [
            _start(),
            {"id": "blk", "type": "subgraph:loader",
             "position": {"x": 200, "y": 0}, "data": {"params": {}}},
        ],
        "edges": [_trigger("blk")],
        "subgraphs": [_block("loader", "Loader", [
            {"id": "csv", "type": "CSVReader", "position": {"x": 0, "y": 0},
             "data": {"params": {"path": WINDOWS_PATH}, "label": "Grades"}},
        ], ["csv"])],
    }

    body = await _export(test_client, graph)

    assert body["warnings"] == [{
        "code": "absolute_path",
        "node_id": "blk/csv",
        "label": "Grades",
        "param": "path",
        "value": WINDOWS_PATH,
        "containers": [{"node_id": "blk", "label": "Loader"}],
    }]


@pytest.mark.asyncio
async def test_a_nested_path_names_every_level_outermost_first(test_client):
    """A block in a block, and a preset in a block: the user opens each level."""
    graph = {
        "name": "nested-path",
        "nodes": [
            _start(),
            {"id": "outer", "type": "subgraph:outer",
             "position": {"x": 200, "y": 0}, "data": {"params": {}}},
            {"id": "loader", "type": "subgraph:with_preset",
             "position": {"x": 200, "y": 200}, "data": {"params": {}}},
        ],
        "edges": [_trigger("outer"), _trigger("loader", "trigger")],
        "subgraphs": [
            _block("outer", "Outer", [
                {"id": "nest", "type": "subgraph:inner",
                 "position": {"x": 0, "y": 0}, "data": {"params": {}}},
            ], ["nest"]),
            _block("inner", "Inner", [
                {"id": "csv", "type": "CSVReader",
                 "position": {"x": 0, "y": 0},
                 "data": {"params": {"path": "/home/student01/a.csv"}}},
            ], ["csv"]),
            _block("with_preset", "Loader", [
                {"id": "p1", "type": "preset:Grade Loader",
                 "position": {"x": 0, "y": 0},
                 "data": {"params": {}, "internalParams": {
                     "csv": {"path": WINDOWS_PATH}}}},
            ], ["p1"]),
        ],
        "presets": [_preset()],
    }

    body = await _export(test_client, graph)

    assert [(w["node_id"], w["containers"]) for w in body["warnings"]] == [
        ("outer/nest/csv", [
            {"node_id": "outer", "label": "Outer"},
            {"node_id": "outer/nest", "label": "Inner"},
        ]),
        ("loader/p1__csv", [
            {"node_id": "loader", "label": "Loader"},
            {"node_id": "loader/p1", "label": "Grade Loader"},
        ]),
    ]


@pytest.mark.asyncio
async def test_a_preset_nested_in_a_preset_names_both(test_client):
    outer = _preset(name="Outer Preset")
    outer["nodes"] = [{"id": "p2", "type": "preset:Grade Loader", "params": {}}]
    outer["exposed_inputs"][0]["internal_node"] = "p2"
    outer["exposed_inputs"][0]["internal_port"] = "trigger"
    graph = {
        "name": "preset-in-preset",
        "nodes": [
            _start(),
            {"id": "p1", "type": "preset:Outer Preset",
             "position": {"x": 200, "y": 0},
             "data": {"params": {}, "internalParams": {}}},
        ],
        "edges": [_trigger("p1", "trigger")],
        "presets": [outer, _preset(params={"path": "/srv/exam/grades.csv"})],
    }

    body = await _export(test_client, graph)

    assert [(w["node_id"], w["value"], w["containers"])
            for w in body["warnings"]] == [
        ("p1__p2__csv", "/srv/exam/grades.csv", [
            {"node_id": "p1", "label": "Outer Preset"},
            {"node_id": "p1__p2", "label": "Grade Loader"},
        ]),
    ]


@pytest.mark.asyncio
async def test_the_script_is_exactly_what_the_exporter_writes(test_client):
    """The warning rides beside the script; the script does not change.

    Graders parse the exported file, so the path stays embedded as written
    and nothing about the warning reaches the source.
    """
    from app.core.codegen import generate_python
    from app.schemas import GraphExportRequest

    graph = _reader_graph(params={"path": WINDOWS_PATH})
    body = await _export(test_client, graph)

    request = GraphExportRequest(**graph)
    expected = generate_python(
        [n.model_dump() for n in request.nodes],
        [e.model_dump() for e in request.edges],
        name=request.name,
        presets=[],
        subgraphs=[],
    )
    assert body["warnings"]
    assert body["script"] == expected
    assert ascii(WINDOWS_PATH) in body["script"]


@pytest.mark.asyncio
async def test_a_fault_in_the_check_never_costs_the_export(
    test_client, monkeypatch,
):
    """Advice, not a gate: if deciding what to warn about breaks, the user
    still gets the script, with no warnings rather than a 500."""
    def broken(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "app.api.routes_graph._absolute_path_warnings", broken)

    body = await _export(test_client, _reader_graph(params={"path": WINDOWS_PATH}))

    assert body["script"]
    assert body["warnings"] == []
