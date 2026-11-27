"""Phase 4B review round 1: a graph folder disturbed from outside.

Symbolic links (item 1), a save killed half way (2), bytes that are not
UTF-8 (3), one bad graph in the list (4), names (6), a case-only rename (7)
and the per-file size cap (8).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.core import graph_storage as gs
from app.core.graph_storage import GraphStore, read_tree

from .test_graph_storage_dir import LONG_CODE, _payload, _snapshot, project  # noqa: F401


def _symlink(link: Path, target: Path, directory: bool = False) -> None:
    try:
        os.symlink(target, link, target_is_directory=directory)
    except (OSError, NotImplementedError):
        pytest.skip("this platform does not allow creating symlinks here")


def _case_insensitive(directory: Path) -> bool:
    probe = directory / "CaseProbe"
    probe.mkdir(exist_ok=True)
    try:
        return (directory / "cASEpROBE").exists()
    finally:
        probe.rmdir()


# ── 1. Symbolic links ────────────────────────────────────────────────────

async def test_links_inside_a_graph_folder_are_never_followed(
        project, test_client, tmp_path_factory):  # noqa: F811
    await test_client.post("/api/graph/save", json=_payload())
    outside = tmp_path_factory.mktemp("outside")
    (outside / "id_rsa").write_text("SECRET KEY")
    (outside / "leak.json").write_text(json.dumps(
        {"id": "leak", "type": "X", "data": {"params": {"v": "SECRET JSON"}}}))
    (outside / "dir").mkdir()
    (outside / "dir" / "x.json").write_text(json.dumps({"id": "x", "type": "X", "data": {}}))
    nodes = project / "graphs" / "demo" / "nodes"
    (nodes / "py.code.py").unlink()
    _symlink(nodes / "py.code.py", outside / "id_rsa")
    _symlink(nodes / "leak.json", outside / "leak.json")
    _symlink(project / "graphs" / "demo" / "subgraphs", outside / "dir", directory=True)
    r = await test_client.get("/api/graph/load/demo")
    assert r.status_code == 200
    assert "SECRET" not in r.text
    kinds = [(x["kind"], x["path"]) for x in r.json()["storage_repairs"]]
    assert ("symlink_ignored", "nodes/leak.json") in kinds
    assert ("symlink_ignored", "nodes/py.code.py") in kinds
    assert ("symlink_ignored", "subgraphs/") in kinds


async def test_a_graph_or_layout_that_links_out_of_the_project_is_refused(
        project, test_client, tmp_path_factory):  # noqa: F811
    # The project is the fixture's tmp_path, so "outside" is another folder.
    outside = tmp_path_factory.mktemp("elsewhere")
    (outside / "g.graph.json").write_text(json.dumps({"name": "SECRET", "nodes": [], "edges": []}))
    _symlink(project / "graphs" / "v1.graph.json", outside / "g.graph.json")
    r = await test_client.get("/api/graph/load/v1")
    assert r.status_code == 422 and "outside the project" in r.text
    assert "SECRET" not in r.text
    # A folder graph whose folder is a link out of the project.
    (outside / "f").mkdir()
    store = GraphStore(outside / "tmpgraphs", outside / "tmplayout", new_graph_format="dir")
    store.write("f", _payload(name="f"))
    _symlink(project / "graphs" / "f", outside / "tmpgraphs" / "f", directory=True)
    r = await test_client.get("/api/graph/load/f")
    assert r.status_code == 422 and "outside the project" in r.text
    # And a layout file linking out.
    await test_client.post("/api/graph/save", json=_payload(name="ok"))
    layout = project / "layout" / "ok.layout.json"
    (outside / "lay.json").write_text(layout.read_text())
    layout.unlink()
    _symlink(layout, outside / "lay.json")
    assert (await test_client.get("/api/graph/load/ok")).status_code == 422
    listed = {g["file"]: g for g in (await test_client.get("/api/graph/list")).json()}
    assert "outside the project" in listed["v1"]["error"]
    assert "outside the project" in listed["f"]["error"]


# ── 2. A save killed half way ────────────────────────────────────────────

@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_a_save_that_dies_half_way_is_reported_on_the_next_read(project, monkeypatch, fail_at):  # noqa: F811
    store = GraphStore.current()
    store.write("demo", _payload())
    graph = _payload(value="NEW")
    graph["nodes"].append({"id": "extra", "type": "X", "position": {"x": 0, "y": 0},
                           "data": {"params": {}}})
    real = gs._atomic_write_bytes
    calls = {"n": 0}

    def flaky(path, data):
        if path.name != gs.SAVE_MARKER:
            calls["n"] += 1
            if calls["n"] == fail_at:
                raise OSError("disk full")
        return real(path, data)

    monkeypatch.setattr(gs, "_atomic_write_bytes", flaky)
    with pytest.raises(OSError):
        store.write("demo", graph)
    monkeypatch.setattr(gs, "_atomic_write_bytes", real)
    loaded = store.read("demo")
    assert "interrupted_save" in [r["kind"] for r in loaded["storage_repairs"]]
    # The marker is not part of the format and a good save clears it.
    assert gs.SAVE_MARKER not in read_tree(project / "graphs" / "demo")
    store.write("demo", graph)
    assert not (project / "graphs" / "demo" / gs.SAVE_MARKER).exists()
    assert "storage_repairs" not in store.read("demo")


def test_the_marker_is_gitignored_in_new_projects():
    from app.core.project import PROJECT_GITIGNORE

    assert "graphs/**/.saving" in PROJECT_GITIGNORE.splitlines()


# ── 3. Bytes that are not UTF-8 ──────────────────────────────────────────

def test_bytes_that_are_not_utf8_are_reported_not_replaced_silently(project):  # noqa: F811
    store = GraphStore.current()
    store.write("demo", _payload())
    folder = project / "graphs" / "demo"
    (folder / "nodes" / "py.code.py").write_bytes(b"x = 1\n\xff\xfe\n" + b"y\n" * 300)
    edges = (folder / "edges.jsonl").read_bytes()
    (folder / "edges.jsonl").write_bytes(edges.replace(b'"d"', b'"d\xff"', 1))
    node = folder / "nodes" / "start.json"
    node.write_bytes(node.read_bytes().replace(b'"Start"', b'"St\xe9rt"'))
    before = _snapshot(project)
    repairs = store.read("demo")["storage_repairs"]
    bad = {r["path"] for r in repairs if r["kind"] == "invalid_text_encoding"}
    assert bad == {"nodes/py.code.py", "edges.jsonl", "nodes/start.json"}
    assert any("offset 6" in r["detail"] for r in repairs if r["path"] == "nodes/py.code.py")
    assert _snapshot(project) == before


# ── 4. One bad graph in the list ─────────────────────────────────────────

async def test_an_unreadable_index_is_listed_and_opens_with_422(project, test_client):  # noqa: F811
    await test_client.post("/api/graph/save", json=_payload())
    await test_client.post("/api/graph/save", json=_payload(name="fine"))
    (project / "graphs" / "demo" / "index.json").write_text("[1, 2]")
    (project / "graphs" / "not-a-graph").mkdir()  # no index.json: not a graph
    listed = {g["file"]: g for g in (await test_client.get("/api/graph/list")).json()}
    assert set(listed) == {"demo", "fine"}
    assert "not a JSON object" in listed["demo"]["error"]
    r = await test_client.get("/api/graph/load/demo")
    assert r.status_code == 422


# ── 6. Names ──────────────────────────────────────────────────────────────

async def test_a_long_name_is_a_400_not_a_500(project, test_client):  # noqa: F811
    r = await test_client.post("/api/graph/save", json=_payload(name="x" * 300))
    assert r.status_code == 400 and "at most 100" in r.text
    r = await test_client.post("/api/graph/save", json=_payload(name="x" * 100))
    assert r.status_code == 200
    r = await test_client.post("/api/graph/rename", json={"from": "x" * 100, "to": "y" * 101})
    assert r.status_code == 400


@pytest.mark.parametrize("name,stem", [("CON", "CON_"), ("aux", "aux_"), ("lpt1", "lpt1_"),
                                       ("aux.txt", "aux_txt"), ("nodes.", "nodes_"),
                                       ("Con ", "Con_")])
def test_windows_reserved_names_get_an_underscore(name, stem):
    assert gs.sanitize_graph_name(name) == stem


async def test_an_existing_graph_under_a_reserved_name_still_opens(project, test_client):  # noqa: F811
    (project / "codefyui.project.toml").write_text('[project]\nname = "t"\n')
    GraphStore.current().write("CON", _payload(name="CON"))  # as an older build wrote it
    r = await test_client.get("/api/graph/load/CON")
    assert r.status_code == 200 and r.json()["name"] == "CON"
    r = await test_client.post("/api/graph/save", json=_payload(name="aux"))
    assert r.json()["file"] == "aux_"


# ── 7. A case-only rename ────────────────────────────────────────────────

@pytest.mark.parametrize("fmt", ["dir", "file"])
async def test_a_rename_that_changes_only_letter_case(project, test_client, fmt):  # noqa: F811
    if not _case_insensitive(project):
        pytest.skip("the filesystem is case-sensitive")
    (project / "codefyui.project.toml").write_text(
        f'[project]\nname = "t"\n[storage]\ngraph_format = "{fmt}"\n')
    await test_client.post("/api/graph/save", json=_payload(name="beta"))
    r = await test_client.post("/api/graph/rename", json={"from": "beta", "to": "Beta"})
    assert r.status_code == 200, r.text
    names = sorted(p.name for p in (project / "graphs").iterdir())
    assert names == (["Beta"] if fmt == "dir" else ["Beta.graph.json"])
    assert sorted(p.name for p in (project / "layout").iterdir()) == ["Beta.layout.json"]
    assert (await test_client.get("/api/graph/load/Beta")).json()["name"] == "Beta"


# ── 8. The size cap ──────────────────────────────────────────────────────

def test_a_file_past_the_cap_refuses_the_graph(project, monkeypatch):  # noqa: F811
    monkeypatch.setattr(gs, "MAX_FILE_BYTES", 2048)
    store = GraphStore.current()
    store.write("demo", _payload())
    (project / "graphs" / "demo" / "nodes" / "py.code.py").write_text("x\n" * 2000)
    with pytest.raises(gs.GraphUnreadable, match="py.code.py"):
        store.read("demo")
    # And a save that would write one is refused before anything is written.
    graph = _payload()
    graph["nodes"][2]["data"]["params"]["code"] = "y\n" * 2000
    with pytest.raises(gs.StorageCheckFailed, match="over the 2048"):
        store.write("other", graph)
    assert not (project / "graphs" / "other").exists()


# ── cdui run on a node file ──────────────────────────────────────────────

def test_a_node_file_is_not_taken_for_a_graph(project):  # noqa: F811
    GraphStore.current().write("demo", _payload())
    with pytest.raises(ValueError, match="one file of the graph folder"):
        gs.load_graph_document(project / "graphs" / "demo" / "nodes" / "ds.json")
