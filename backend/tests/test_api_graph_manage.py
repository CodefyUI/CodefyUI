"""Managing saved graphs: DELETE /{name}, POST /rename, and /list's `modified`.

Conventions follow test_api_graph_run.py: `_graphs_dir` monkeypatches
`settings.GRAPHS_DIR` at a temp dir so no test can reach a real
`backend/data/graphs`. Project mode gets a fixture of its own (the shape
test_graph_save_load_project.py uses) because `LAYOUT_DIR` is derived from
`PROJECT_DIR` rather than settable on its own.

Traversal payloads are percent-encoded for the reason test_api_data_files.py
spells out at length: httpx removes `..` segments from a URL before sending
it, so an unencoded payload never reaches the handler and the test would pass
with `_sanitize_name` deleted. Only the backslash payload can be tested
through the URL at all -- `{name}` is a single path segment, so a decoded
forward slash is a router 404 -- which is why the `../` payloads live in the
rename tests, where names arrive in a JSON body that nothing normalises.
"""

import json
import os

import pytest


@pytest.fixture
def _graphs_dir(tmp_path, monkeypatch):
    """Isolate saved graphs per test (pattern: test_api_graph.py)."""
    monkeypatch.setattr("app.config.settings.PROJECT_DIR", None)
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def project_settings(tmp_path, monkeypatch):
    """Project mode, with GRAPHS_DIR one level below `tmp_path` -- which is
    also what gives the traversal tests somewhere OUTSIDE it to put a decoy.
    """
    monkeypatch.setattr("app.config.settings.PROJECT_DIR", tmp_path)
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path / "graphs")
    (tmp_path / "graphs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "layout").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _graph(name="demo", x=10):
    """The smallest thing /save will write: one node, no edges."""
    return {
        "name": name,
        "description": "",
        "nodes": [
            {"id": "a", "type": "Dataset", "position": {"x": x, "y": 0},
             "data": {"params": {"name": "MNIST"}}},
        ],
        "edges": [],
        "presets": [],
        "segmentGroups": [],
    }


async def _save(client, graph):
    resp = await client.post("/api/graph/save", json=graph)
    assert resp.status_code == 200, resp.text


def _verbatim(name: str) -> str:
    """Encode *name* so the route receives it character for character.

    Only the dots need it (test_api_data_files.py): `.` is what triggers a
    client's dot-segment removal, and `%2e` is decoded back before the handler
    sees it.
    """
    return name.replace(".", "%2e")


# ── DELETE /api/graph/{name} ─────────────────────────────────────────────


async def test_delete_removes_the_file(test_client, _graphs_dir):
    await _save(test_client, _graph(name="demo"))
    assert (_graphs_dir / "demo.json").exists()

    resp = await test_client.delete("/api/graph/demo")
    assert resp.status_code == 200, resp.text
    assert not (_graphs_dir / "demo.json").exists()
    assert (await test_client.get("/api/graph/list")).json() == []


async def test_delete_404_when_there_was_nothing(test_client, _graphs_dir):
    """Not a quiet success: the panel showed the user a row, and "another tab
    deleted it first" is something they should get to see."""
    resp = await test_client.delete("/api/graph/never-saved")
    assert resp.status_code == 404, resp.text
    assert resp.json()["detail"] == "Graph 'never-saved' not found"


async def test_delete_removes_both_halves_in_project_mode(
    test_client, project_settings,
):
    await _save(test_client, _graph(name="demo"))
    logic = project_settings / "graphs" / "demo.graph.json"
    layout = project_settings / "layout" / "demo.layout.json"
    assert logic.exists() and layout.exists()

    resp = await test_client.delete("/api/graph/demo")
    assert resp.status_code == 200, resp.text
    assert not logic.exists()
    assert not layout.exists()


async def test_delete_removes_a_legacy_single_file(test_client, project_settings):
    """A project that has not saved since the split still stores `<name>.json`;
    resolve_graph_file hands the delete exactly that file, so there is no
    second unlink to forget."""
    legacy = project_settings / "graphs" / "old.json"
    legacy.write_text(json.dumps(_graph(name="old")))

    resp = await test_client.delete("/api/graph/old")
    assert resp.status_code == 200, resp.text
    assert not legacy.exists()


async def test_delete_of_an_ambiguous_pair_409s(test_client, project_settings):
    """Both forms of one base: delete refuses exactly as /load does, naming
    both files. Guessing here means destroying the wrong one."""
    (project_settings / "graphs" / "dup.graph.json").write_text("{}")
    (project_settings / "graphs" / "dup.json").write_text("{}")

    resp = await test_client.delete("/api/graph/dup")
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert "dup.graph.json" in detail
    assert "dup.json" in detail
    assert (project_settings / "graphs" / "dup.graph.json").exists()
    assert (project_settings / "graphs" / "dup.json").exists()


async def test_delete_traversal_never_leaves_graphs_dir(
    test_client, project_settings,
):
    """`..\\outside` reaches the handler on both platforms (no forward slash,
    so it matches `{name}`), and `_sanitize_name` reduces it to underscores
    inside GRAPHS_DIR. The echoed name in the detail is what proves the
    handler ran at all rather than the router answering first."""
    outside = project_settings / "outside.json"
    outside.write_text(json.dumps(_graph(name="outside")))

    resp = await test_client.delete(f"/api/graph/{_verbatim('..')}\\outside")
    assert resp.status_code == 404, resp.text
    assert resp.json()["detail"] == "Graph '..\\outside' not found"
    assert outside.exists()


# ── POST /api/graph/rename ───────────────────────────────────────────────


async def test_rename_moves_the_file(test_client, _graphs_dir):
    await _save(test_client, _graph(name="before"))

    resp = await test_client.post(
        "/api/graph/rename", json={"from": "before", "to": "after"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["file"] == "after"
    assert not (_graphs_dir / "before.json").exists()
    assert (_graphs_dir / "after.json").exists()


async def test_rename_rewrites_the_name_inside_the_file(test_client, _graphs_dir):
    """The list shows each file's own `name`, so a rename that moved only the
    file would leave a row still calling itself by the old title."""
    await _save(test_client, _graph(name="before"))

    resp = await test_client.post(
        "/api/graph/rename", json={"from": "before", "to": "after"})
    assert resp.status_code == 200, resp.text
    assert json.loads((_graphs_dir / "after.json").read_text())["name"] == "after"
    listed = (await test_client.get("/api/graph/list")).json()
    assert [(g["name"], g["file"]) for g in listed] == [("after", "after")]


async def test_rename_404_when_the_source_is_gone(test_client, _graphs_dir):
    resp = await test_client.post(
        "/api/graph/rename", json={"from": "never-saved", "to": "x"})
    assert resp.status_code == 404, resp.text
    assert resp.json()["detail"] == "Graph 'never-saved' not found"
    assert not (_graphs_dir / "x.json").exists()


async def test_rename_onto_an_existing_graph_409s(test_client, _graphs_dir):
    await _save(test_client, _graph(name="one"))
    await _save(test_client, _graph(name="two"))

    resp = await test_client.post(
        "/api/graph/rename", json={"from": "one", "to": "two"})
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"] == "Graph 'two' already exists"
    assert (_graphs_dir / "one.json").exists()
    assert (_graphs_dir / "two.json").exists()


async def test_rename_onto_a_legacy_file_of_the_same_base_409s(
    test_client, project_settings,
):
    """The collision check resolves the target the way a READ would, so
    renaming a canonical pair onto a base that exists only in the legacy form
    is refused too -- otherwise both forms of one base would end up on disk,
    which is the state GraphAmbiguityError exists to forbid."""
    await _save(test_client, _graph(name="pair"))
    (project_settings / "graphs" / "taken.json").write_text(
        json.dumps(_graph(name="taken")))

    resp = await test_client.post(
        "/api/graph/rename", json={"from": "pair", "to": "taken"})
    assert resp.status_code == 409, resp.text
    assert (project_settings / "graphs" / "pair.graph.json").exists()
    assert not (project_settings / "graphs" / "taken.graph.json").exists()


async def test_rename_moves_the_layout_half_in_project_mode(
    test_client, project_settings,
):
    await _save(test_client, _graph(name="before"))

    resp = await test_client.post(
        "/api/graph/rename", json={"from": "before", "to": "after"})
    assert resp.status_code == 200, resp.text
    graphs, layout = project_settings / "graphs", project_settings / "layout"
    assert not (graphs / "before.graph.json").exists()
    assert not (layout / "before.layout.json").exists()
    assert (graphs / "after.graph.json").exists()
    assert (layout / "after.layout.json").exists()
    # Positions live only in the layout file, so a graph that still loads at
    # x=10 is proof the layout half followed rather than being left orphaned.
    loaded = (await test_client.get("/api/graph/load/after")).json()
    assert loaded["name"] == "after"
    assert loaded["nodes"][0]["position"] == {"x": 10, "y": 0}


@pytest.mark.parametrize("reserved", ["weird.graph", "weird.layout"])
async def test_rename_to_a_reserved_name_is_refused(
    test_client, project_settings, reserved,
):
    """The guard /save applies, applied to the other way a name gets chosen."""
    await _save(test_client, _graph(name="demo"))

    resp = await test_client.post(
        "/api/graph/rename", json={"from": "demo", "to": reserved})
    assert resp.status_code == 400, resp.text
    assert "reserved" in resp.json()["detail"].lower()
    # Refused before anything moved: the graph is still where it was, and no
    # sanitized would-be target (weird_graph.graph.json) was written.
    assert [p.name for p in (project_settings / "graphs").iterdir()] == [
        "demo.graph.json"]


@pytest.mark.parametrize("payload", ["../outside", "..\\outside", "a/../../outside"])
async def test_rename_from_a_traversal_is_refused(
    test_client, project_settings, payload,
):
    """Names arrive in a JSON body here, so these reach the handler verbatim
    on every platform -- no encoding, no client-side normalisation."""
    outside = project_settings / "outside.json"
    outside.write_text(json.dumps(_graph(name="outside")))

    resp = await test_client.post(
        "/api/graph/rename", json={"from": payload, "to": "stolen"})
    assert resp.status_code == 404, resp.text
    assert outside.exists()
    assert not (project_settings / "graphs" / "stolen.graph.json").exists()


async def test_rename_to_a_traversal_stays_inside_graphs_dir(
    test_client, project_settings,
):
    """The write side of the same guard: the target is sanitized too, so
    `../escaped` becomes underscores inside GRAPHS_DIR rather than a file one
    directory up."""
    await _save(test_client, _graph(name="demo"))

    resp = await test_client.post(
        "/api/graph/rename", json={"from": "demo", "to": "../escaped"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["file"] == "___escaped"
    assert (project_settings / "graphs" / "___escaped.graph.json").exists()
    assert not (project_settings / "escaped.json").exists()
    assert not (project_settings / "escaped.graph.json").exists()


# ── GET /api/graph/list: `modified` ──────────────────────────────────────


async def test_list_reports_the_file_mtime(test_client, _graphs_dir):
    await _save(test_client, _graph(name="demo"))

    listed = (await test_client.get("/api/graph/list")).json()
    assert len(listed) == 1
    # `name` and `file` are what existing clients read; `modified` is additive.
    assert listed[0]["name"] == "demo"
    assert listed[0]["file"] == "demo"
    assert listed[0]["modified"] == (_graphs_dir / "demo.json").stat().st_mtime


async def test_list_reports_the_logic_mtime_not_the_layout(
    test_client, project_settings,
):
    """A drag rewrites only the layout half. A list sorted most-recent-first
    must not read that as an edit to the graph."""
    await _save(test_client, _graph(name="demo"))
    logic = project_settings / "graphs" / "demo.graph.json"
    layout = project_settings / "layout" / "demo.layout.json"
    newer = logic.stat().st_mtime + 3600
    os.utime(layout, (newer, newer))

    listed = (await test_client.get("/api/graph/list")).json()
    assert listed[0]["modified"] == logic.stat().st_mtime
    assert listed[0]["modified"] < newer
