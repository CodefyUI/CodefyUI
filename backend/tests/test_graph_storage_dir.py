"""Phase 4B: graphs stored as folders (reviews/phase-4b-design.md).

The pure layer (``explode`` / ``implode``), the order rule, repairs, the
round-trip check that refuses to write, and the API: a save writes only what
changed, a load reports repairs without writing them back.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import pytest

from app.api import routes_graph
from app.core import graph_storage as gs
from app.core.graph_storage import (
    O_STEP,
    GraphStore,
    StorageCheckFailed,
    assign_orders,
    explode,
    file_name_for,
    implode,
    read_tree,
)

# ── A seeded generator (no hypothesis dependency) ─────────────────────────

WEIRD_IDS = [
    "a", "A", "conv-1", "x/y", "../up", "名前", "名前", "é", "é",
    "CON", "aux.txt", "trailing.", "trailing ", ".hidden", "x" * 300,
    "a:b", "tab\tid", "Node", "node", "NODE", "lpt9", "-", "_", "..",
]
LONG_CODE = "import torch\n" + "".join(f"x{i} = {i}\n" for i in range(80))


def _node(rng: random.Random, node_id: str) -> dict:
    params: dict = {"n": rng.randint(0, 9), "s": "short"}
    roll = rng.random()
    if roll < 0.25:
        params["code"] = LONG_CODE + node_id
    elif roll < 0.35:
        params["prompt"] = "line\n" * 200
    elif roll < 0.45:
        params["looks_like_marker"] = {"$file": "nope.txt"}
    elif roll < 0.5:
        params["nested_marker"] = {"$literal": {"$file": "x"}}
    elif roll < 0.55:
        params["crlf"] = "a\r\nb" * 300
    return {"id": node_id, "type": rng.choice(["Start", "Conv2d", "PythonScript"]),
            "data": {"params": params, "label": f"L{node_id[:5]}"}}


def make_graph(seed: int, *, nodes: int = 12, blocks: int = 2) -> dict:
    rng = random.Random(seed)
    pool = WEIRD_IDS + [f"n{i}" for i in range(nodes)]
    ids = list(dict.fromkeys(rng.sample(pool, k=min(len(pool), nodes))))
    node_list = [_node(rng, i) for i in ids]
    edges, seen = [], set()
    for k in range(nodes * 2):
        a, b = rng.sample(ids, 2)
        key = (a, "out", b, f"in{k % 3}")
        if key in seen:
            continue
        seen.add(key)
        edges.append({"id": f"e{k}", "source": a, "target": b, "sourceHandle": "out",
                      "targetHandle": f"in{k % 3}", "type": rng.choice(["data", "trigger"])})
    subgraphs = []
    for b in range(blocks):
        inner_ids = [f"in{b}-{k}" for k in range(3)] + ["A", "a"]
        subgraphs.append({
            "id": rng.choice(["blk", "Blk", "blk/2", "CON"]) + str(b),
            "name": f"Block {b}", "description": "",
            "nodes": [_node(rng, i) for i in inner_ids],
            "edges": [{"id": "ie", "source": inner_ids[0], "target": inner_ids[1],
                       "sourceHandle": "o", "targetHandle": "i", "type": "data"}],
            "interface": {"inputs": [], "outputs": [], "triggerTargets": []},
        })
    return {
        "format_version": 1, "name": f"G{seed}", "description": "desc",
        "nodes": node_list, "edges": edges,
        "presets": [{"preset_name": n, "nodes": [], "edges": []}
                    for n in rng.sample(["p", "P", "p/q", "名"], 3)],
        "subgraphs": subgraphs,
        "settings": {"seed": seed},
    }


@pytest.mark.parametrize("seed", range(40))
def test_round_trip_is_exact_including_order(seed):
    graph = make_graph(seed)
    files = explode(graph)
    back, repairs = implode(files)
    assert repairs == []
    assert back == graph
    assert [n["id"] for n in back["nodes"]] == [n["id"] for n in graph["nodes"]]
    assert [e["id"] for e in back["edges"]] == [e["id"] for e in graph["edges"]]
    # Saving the loaded graph again writes nothing new.
    assert explode(back, files) == files


@pytest.mark.parametrize("seed", range(10))
def test_file_names_are_portable_and_unique(seed):
    files = explode(make_graph(seed))
    folded = [name.casefold() for name in files]
    assert len(set(folded)) == len(folded)
    for rel in files:
        for part in rel.split("/"):
            assert part and not part.startswith(".") and not part.endswith((".", " "))
            assert part.split(".")[0].upper() not in gs._WINDOWS_RESERVED
            assert all(c.isascii() and (c.isalnum() or c in "._-") for c in part), rel
        assert len("graphs/some-graph/" + rel) <= gs.PATH_LIMIT + 60


def test_case_only_ids_get_distinct_files():
    taken: set[str] = set()
    first = file_name_for("Node", taken)
    second = file_name_for("node", taken)
    third = file_name_for("NODE", taken)
    assert len({first.casefold(), second.casefold(), third.casefold()}) == 3
    assert first == "Node"  # a safe name is kept as it is


def test_a_long_id_gets_a_hash_name():
    name = file_name_for("x" * 300, set(), dir_len=100)
    assert len(name) == 16


def test_code_params_are_files_with_line_diffs():
    graph = make_graph(1)
    graph["nodes"][0]["data"]["params"]["code"] = LONG_CODE
    files = explode(graph)
    code_files = [rel for rel in files if rel.endswith(".code.py")]
    assert code_files
    assert any(files[rel] == LONG_CODE.encode() for rel in code_files)
    node_doc = json.loads(files[[r for r in files if r.startswith("nodes/") and r.endswith(".json")
                                 and json.loads(files[r])["id"] == graph["nodes"][0]["id"]][0]])
    assert set(node_doc["data"]["params"]["code"]) == {"$file"}


def test_text_with_crlf_stays_inline():
    graph = make_graph(2)
    graph["nodes"][0]["data"]["params"]["code"] = "a\r\nb\n" * 200
    files = explode(graph)
    assert implode(files)[0] == graph
    assert not any(rel.startswith("nodes/") and rel.endswith(".py")
                   and b"\r" in files[rel] for rel in files)


# ── Order ─────────────────────────────────────────────────────────────────

def test_new_items_go_after_the_largest():
    assert assign_orders(["a", "b", "c"], {}) == [O_STEP, 2 * O_STEP, 3 * O_STEP]
    assert assign_orders(["a", "b", "x"], {"a": 1024, "b": 2048}) == [1024, 2048, 3072]


def test_an_insert_takes_the_midpoint_and_keeps_the_rest():
    out = assign_orders(["a", "x", "b"], {"a": 1024, "b": 2048})
    assert out == [1024, 1536, 2048]


def test_a_move_changes_only_the_moved_item():
    prev = {k: (i + 1) * O_STEP for i, k in enumerate("abcde")}
    out = assign_orders(list("abdce"), prev)
    changed = [k for k, o in zip("abdce", out) if prev[k] != o]
    assert len(changed) == 1
    assert out == sorted(out)


def test_no_gap_left_renumbers_everything():
    out = assign_orders(["a", "x", "b"], {"a": 5, "b": 6})
    assert out == [O_STEP, 2 * O_STEP, 3 * O_STEP]


def renumber_threshold() -> int:
    """How many inserts at one place (always right after the first item)
    happen before a renumber."""
    keys = ["first", "last"]
    prev = {"first": O_STEP, "last": 2 * O_STEP}
    for count in range(1, 100):
        keys.insert(1, f"new{count}")
        out = assign_orders(keys, prev)
        if out[0] != O_STEP or out[-1] != 2 * O_STEP:
            return count
        prev = dict(zip(keys, out))
    return 100


def test_ten_inserts_at_one_place_fit_and_the_eleventh_renumbers():
    # Each insert halves the gap below it: 1024 -> 512 -> ... -> 1.
    assert renumber_threshold() == 11


# ── Repairs ───────────────────────────────────────────────────────────────

def _files_of(graph: dict) -> dict[str, bytes]:
    return dict(explode(graph))


def test_an_edge_to_a_missing_node_is_dropped_and_reported():
    graph = make_graph(3, blocks=0)
    files = _files_of(graph)
    victim = next(rel for rel in files if rel.startswith("nodes/") and rel.endswith(".json"))
    victim_id = json.loads(files[victim])["id"]
    del files[victim]
    back, repairs = implode(files)
    assert victim_id not in [n["id"] for n in back["nodes"]]
    assert all(victim_id not in (e["source"], e["target"]) for e in back["edges"])
    kinds = {r.kind for r in repairs}
    if any(victim_id in (e["source"], e["target"]) for e in graph["edges"]):
        assert "edge_missing_node" in kinds


def test_duplicate_edges_ids_text_files_and_bad_json_are_repairs():
    graph = make_graph(4, blocks=0)
    graph["nodes"][0]["data"]["params"]["code"] = LONG_CODE
    files = _files_of(graph)
    lines = files["edges.jsonl"].decode().splitlines()
    files["edges.jsonl"] = ("\n".join(lines + [lines[0]]) + "\n").encode()
    node_rel = next(rel for rel in sorted(files) if rel.startswith("nodes/") and rel.endswith(".json"))
    files["nodes/zz-copy.json"] = files[node_rel]
    files["nodes/broken.json"] = b"{not json"
    text = next(rel for rel in files if rel.endswith(".code.py"))
    del files[text]
    back, repairs = implode(files)
    kinds = sorted(r.kind for r in repairs)
    assert {"duplicate_edge", "duplicate_id", "unreadable_file", "missing_text_file"} <= set(kinds)
    assert any(p.get("code") == "" for n in back["nodes"]
               for p in [n["data"].get("params", {})] if "code" in p)


def test_a_missing_or_unreadable_index_is_an_error_not_a_repair():
    files = _files_of(make_graph(5))
    files["index.json"] = b"nope"
    with pytest.raises(ValueError):
        implode(files)
    del files["index.json"]
    with pytest.raises(ValueError):
        implode(files)


def test_an_unknown_storage_version_is_refused():
    files = _files_of(make_graph(6))
    index = json.loads(files["index.json"])
    index["storage_version"] = 3
    files["index.json"] = json.dumps(index).encode()
    with pytest.raises(LookupError):
        implode(files)


def test_a_hand_written_node_without_o_goes_last():
    files = _files_of(make_graph(7, blocks=0))
    files["nodes/zzz-hand.json"] = json.dumps(
        {"id": "hand", "type": "X", "data": {}}).encode()
    back, repairs = implode(files)
    assert back["nodes"][-1]["id"] == "hand" and repairs == []


# ── The store and the API ─────────────────────────────────────────────────

@pytest.fixture
def project(monkeypatch, tmp_path):
    monkeypatch.setattr(routes_graph.settings, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(routes_graph.settings, "GRAPHS_DIR", tmp_path / "graphs")
    (tmp_path / "graphs").mkdir()
    (tmp_path / "layout").mkdir()
    (tmp_path / "codefyui.project.toml").write_text(
        '[project]\nname = "t"\n\n[storage]\ngraph_format = "dir"\n', encoding="utf-8")
    return tmp_path


def _payload(name="demo", x=10, value="MNIST"):
    return {
        "name": name, "description": "",
        "nodes": [
            {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
             "data": {"params": {}}},
            {"id": "ds", "type": "Dataset", "position": {"x": x, "y": 50},
             "data": {"params": {"name": value}}},
            {"id": "py", "type": "PythonScript", "position": {"x": 5, "y": 90},
             "data": {"params": {"code": LONG_CODE}}},
        ],
        "edges": [
            {"id": "t", "source": "start", "target": "ds", "sourceHandle": "trigger",
             "targetHandle": "", "type": "trigger"},
            {"id": "d", "source": "ds", "target": "py", "sourceHandle": "value",
             "targetHandle": "x", "type": "data"},
        ],
        "presets": [], "segmentGroups": [],
    }


def _snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in root.rglob("*") if p.is_file() and ".git" not in p.parts}


def _changed(before: dict, after: dict) -> list[str]:
    return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))


async def test_a_new_graph_in_a_dir_project_is_a_folder(project, test_client):
    r = await test_client.post("/api/graph/save", json=_payload())
    assert r.status_code == 200, r.text
    folder = project / "graphs" / "demo"
    assert (folder / "index.json").is_file()
    index = json.loads((folder / "index.json").read_text())
    assert index["storage_version"] == 2 and index["format_version"] == 1
    assert (folder / "nodes" / "py.code.py").read_text() == LONG_CODE
    assert (project / "layout" / "demo.layout.json").is_file()
    loaded = (await test_client.get("/api/graph/load/demo")).json()
    assert [n["id"] for n in loaded["nodes"]] == ["start", "ds", "py"]
    assert loaded["nodes"][2]["data"]["params"]["code"] == LONG_CODE
    assert loaded["nodes"][1]["position"] == {"x": 10, "y": 50}
    assert loaded["format_version"] == 1
    assert "storage_repairs" not in loaded


async def test_saving_an_unchanged_graph_writes_nothing(project, test_client):
    await test_client.post("/api/graph/save", json=_payload())
    before = _snapshot(project)
    loaded = (await test_client.get("/api/graph/load/demo")).json()
    loaded.pop("layout_missing", None)
    r = await test_client.post("/api/graph/save", json={**loaded, "file": "demo"})
    assert r.status_code == 200
    assert _changed(before, _snapshot(project)) == []


async def test_one_param_one_file_and_a_move_only_the_layout(project, test_client):
    await test_client.post("/api/graph/save", json=_payload())
    before = _snapshot(project)
    await test_client.post("/api/graph/save", json={**_payload(value="CIFAR"), "file": "demo"})
    assert _changed(before, _snapshot(project)) == ["graphs/demo/nodes/ds.json"]
    before = _snapshot(project)
    await test_client.post("/api/graph/save", json={**_payload(value="CIFAR", x=99), "file": "demo"})
    assert _changed(before, _snapshot(project)) == ["layout/demo.layout.json"]


async def test_a_new_node_is_one_new_file_plus_the_layout(project, test_client):
    await test_client.post("/api/graph/save", json=_payload())
    before = _snapshot(project)
    graph = _payload()
    graph["nodes"].append({"id": "extra", "type": "X", "position": {"x": 1, "y": 2},
                           "data": {"params": {}}})
    await test_client.post("/api/graph/save", json={**graph, "file": "demo"})
    assert _changed(before, _snapshot(project)) == [
        "graphs/demo/nodes/extra.json", "layout/demo.layout.json"]


async def test_the_round_trip_check_refuses_to_write(project, test_client, monkeypatch):
    await test_client.post("/api/graph/save", json=_payload())
    before = _snapshot(project)
    real = gs.implode

    def lossy(files):
        logic, repairs = real(files)
        logic["nodes"] = logic["nodes"][:-1]
        return logic, repairs

    monkeypatch.setattr(gs, "implode", lossy)
    r = await test_client.post("/api/graph/save", json={**_payload(value="new"), "file": "demo"})
    assert r.status_code == 500
    assert "nothing was written" in r.json()["detail"]
    assert _changed(before, _snapshot(project)) == []
    with pytest.raises(StorageCheckFailed):
        GraphStore.current().write("demo", _payload(value="new"))


async def test_repairs_are_reported_on_load_and_never_written(project, test_client):
    await test_client.post("/api/graph/save", json=_payload())
    (project / "graphs" / "demo" / "nodes" / "ds.json").unlink()
    before = _snapshot(project)
    loaded = (await test_client.get("/api/graph/load/demo")).json()
    kinds = {r["kind"] for r in loaded["storage_repairs"]}
    assert "edge_missing_node" in kinds
    assert [n["id"] for n in loaded["nodes"]] == ["start", "py"]
    assert _changed(before, _snapshot(project)) == []


async def test_leftovers_a_folder_cannot_hold_are_dropped_and_said(project, test_client):
    graph = _payload()
    graph["edges"].append({"id": "dup", "source": "ds", "target": "py",
                           "sourceHandle": "value", "targetHandle": "x", "type": "data"})
    graph["edges"].append({"id": "ghost", "source": "nowhere", "target": "py",
                           "sourceHandle": "v", "targetHandle": "x", "type": "data"})
    r = await test_client.post("/api/graph/save", json=graph)
    assert r.status_code == 200, r.text
    assert len(r.json()["dropped"]) == 2
    assert r.json()["dropped_edges"] == [{"block": None, "id": "dup"},
                                         {"block": None, "id": "ghost"}]
    loaded = (await test_client.get("/api/graph/load/demo")).json()
    assert [e["id"] for e in loaded["edges"]] == ["t", "d"]


async def test_list_rename_delete_and_mtime(project, test_client):
    await test_client.post("/api/graph/save", json=_payload())
    listed = (await test_client.get("/api/graph/list")).json()
    assert [(g["file"], g["name"]) for g in listed] == [("demo", "demo")]
    node_file = project / "graphs" / "demo" / "nodes" / "ds.json"
    import os
    os.utime(node_file, (4_000_000_000, 4_000_000_000))
    assert (await test_client.get("/api/graph/list")).json()[0]["modified"] == 4_000_000_000
    r = await test_client.post("/api/graph/rename", json={"from": "demo", "to": "renamed"})
    assert r.status_code == 200, r.text
    assert (project / "graphs" / "renamed" / "index.json").is_file()
    assert not (project / "graphs" / "demo").exists()
    assert (project / "layout" / "renamed.layout.json").is_file()
    assert (await test_client.get("/api/graph/load/renamed")).json()["name"] == "renamed"
    r = await test_client.delete("/api/graph/renamed")
    assert r.status_code == 200
    assert not (project / "graphs" / "renamed").exists()
    assert not (project / "layout" / "renamed.layout.json").exists()


async def test_an_existing_v1_graph_keeps_its_form(project, test_client):
    (project / "codefyui.project.toml").write_text('[project]\nname = "t"\n')
    await test_client.post("/api/graph/save", json=_payload(name="old"))
    assert (project / "graphs" / "old.graph.json").is_file()
    (project / "codefyui.project.toml").write_text(
        '[project]\nname = "t"\n[storage]\ngraph_format = "dir"\n')
    await test_client.post("/api/graph/save", json={**_payload(name="old", value="x"), "file": "old"})
    assert (project / "graphs" / "old.graph.json").is_file()
    assert not (project / "graphs" / "old").exists()


async def test_a_folder_beside_a_file_is_ambiguous(project, test_client):
    await test_client.post("/api/graph/save", json=_payload())
    (project / "graphs" / "demo.graph.json").write_text("{}")
    for resp in (await test_client.get("/api/graph/load/demo"),
                 await test_client.post("/api/graph/save", json={**_payload(), "file": "demo"})):
        assert resp.status_code == 409
        assert "older CodefyUI" in resp.text
    # The list still answers, with every other graph, and this one marked.
    await test_client.post("/api/graph/save", json=_payload(name="other"))
    listed = (await test_client.get("/api/graph/list")).json()
    assert {g["file"] for g in listed} == {"demo", "other"}
    bad = next(g for g in listed if g["file"] == "demo")
    assert "older CodefyUI" in bad["error"]
    assert "error" not in next(g for g in listed if g["file"] == "other")


async def test_a_newer_storage_version_is_listed_but_not_opened(project, test_client):
    await test_client.post("/api/graph/save", json=_payload())
    index_path = project / "graphs" / "demo" / "index.json"
    index = json.loads(index_path.read_text())
    index["storage_version"] = 3
    index_path.write_text(json.dumps(index))
    listed = (await test_client.get("/api/graph/list")).json()
    assert listed[0]["newer"] is True
    assert (await test_client.get("/api/graph/load/demo")).status_code == 409
    r = await test_client.post("/api/graph/save", json={**_payload(), "file": "demo"})
    assert r.status_code == 409
    assert json.loads(index_path.read_text())["storage_version"] == 3


async def test_v1_and_v2_publish_the_same_document(project, test_client, app_db):
    (project / "codefyui.project.toml").write_text('[project]\nname = "t"\n')
    graph = {
        "name": "echo", "description": "",
        "nodes": [
            {"id": "start", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
            {"id": "gi", "type": "GraphInput", "position": {"x": 9, "y": 9},
             "data": {"params": {"name": "x", "type": "string", "required": True,
                                 "default": "", "description": ""}}},
            {"id": "out", "type": "GraphOutput", "position": {"x": 5, "y": 5},
             "data": {"params": {"name": "y", "description": ""}}},
        ],
        "edges": [
            {"id": "t1", "source": "start", "target": "gi", "sourceHandle": "trigger",
             "targetHandle": "", "type": "trigger"},
            {"id": "d1", "source": "gi", "target": "out", "sourceHandle": "value",
             "targetHandle": "value", "type": "data"},
        ],
    }
    await test_client.post("/api/graph/save", json={**graph, "name": "v1g"})
    (project / "codefyui.project.toml").write_text(
        '[project]\nname = "t"\n[storage]\ngraph_format = "dir"\n')
    await test_client.post("/api/graph/save", json={**graph, "name": "v1g", "file": "v2g"})
    assert (project / "graphs" / "v2g").is_dir()
    for slug, source in (("one", "v1g"), ("two", "v2g")):
        r = await test_client.post(f"/api/apps/{slug}/publish",
                                   json={"graph": source, "create": True})
        assert r.status_code == 200, r.text

    def _snaps(conn):
        return [row[0] for row in conn.execute(
            "SELECT graph_json FROM app_versions ORDER BY app_id")]

    one, two = await app_db.run(_snaps)
    assert json.loads(one) == json.loads(two)


def test_a_folder_reads_back_through_the_cli_loader(project):
    store = GraphStore.current()
    store.write("demo", _payload())
    folder = project / "graphs" / "demo"
    repairs: list = []
    by_folder = gs.load_graph_document(folder, repairs)
    by_index = gs.load_graph_document(folder / "index.json")
    assert by_folder == by_index == store.read_logic("demo")
    assert repairs == []
    assert gs.graph_folder_of(project / "codefyui.project.toml") is None


def test_unknown_files_in_the_folder_are_left_alone(project):
    store = GraphStore.current()
    store.write("demo", _payload())
    note = project / "graphs" / "demo" / "README.md"
    note.write_text("mine")
    store.write("demo", copy.deepcopy(_payload(value="changed")))
    assert note.read_text() == "mine"
    assert "README.md" not in read_tree(project / "graphs" / "demo")


# ── requires ──────────────────────────────────────────────────────────────

def test_requires_lists_plugin_and_custom_types_without_versions():
    providers = {"Start": "builtin", "pk:Embed": "plugin:pk", "MyNode": "custom"}
    graph = {"nodes": [{"id": "a", "type": "Start"}, {"id": "b", "type": "pk:Embed"}],
             "subgraphs": [{"id": "s", "nodes": [{"id": "c", "type": "MyNode"}]}],
             "presets": []}
    out = gs.derive_requires(graph, None, providers.get)
    assert out == {"codefyui": gs.REQUIRES_CODEFYUI,
                   "plugins": {"pk": {"nodes": ["pk:Embed"]}},
                   "mods": {"nodes": ["MyNode"]}}


def test_a_type_this_machine_lacks_keeps_its_old_requirement():
    previous = {"codefyui": ">=3.4", "plugins": {"gone": {"nodes": ["gone:X"]}},
                "mods": {"nodes": ["OldMod"]}}
    graph = {"nodes": [{"id": "a", "type": "gone:X"}, {"id": "b", "type": "OldMod"},
                       {"id": "c", "type": "Unknown"}]}
    out = gs.derive_requires(graph, previous, lambda t: None)
    assert out["plugins"] == {"gone": {"nodes": ["gone:X"]}}
    assert out["mods"] == {"nodes": ["OldMod"]}
    # A type no longer in the graph is not carried along.
    assert gs.derive_requires({"nodes": []}, previous, lambda t: None) == {
        "codefyui": gs.REQUIRES_CODEFYUI}


def test_a_saved_folder_records_requires(project):
    GraphStore.current().write("demo", _payload())
    index = json.loads((project / "graphs" / "demo" / "index.json").read_text())
    assert index["requires"]["codefyui"] == gs.REQUIRES_CODEFYUI


# ── Scale (the report's numbers come from .scratch; this keeps the shape) ──

def test_a_thousand_nodes_save_only_what_changed(tmp_path):
    import time

    store = GraphStore(tmp_path / "graphs", tmp_path / "layout", new_graph_format="dir")
    (tmp_path / "layout").mkdir()
    nodes = [{"id": f"n{i}", "type": "Conv2d", "position": {"x": i, "y": 0},
              "data": {"params": {"n": i, **({"code": LONG_CODE} if i % 20 == 0 else {})}}}
             for i in range(1000)]
    edges = [{"id": f"e{i}", "source": f"n{i}", "target": f"n{i + 1}", "sourceHandle": "o",
              "targetHandle": "i", "type": "data"} for i in range(999)]
    graph = {"name": "big", "description": "", "nodes": nodes, "edges": edges}
    store.write("big", graph)
    started = time.perf_counter()
    again = store.write("big", graph)
    elapsed = time.perf_counter() - started
    assert again.written == [] and again.deleted == []
    # Generous for a slow CI runner; measured about 0.06 s on a laptop.
    assert elapsed < 5
    graph["nodes"][500]["data"]["params"]["n"] = -1
    one = store.write("big", graph)
    assert [Path(p).name for p in one.written] == ["n500.json"]
    graph["nodes"].insert(10, {"id": "added", "type": "X", "position": {"x": 0, "y": 0},
                               "data": {"params": {}}})
    added = store.write("big", graph)
    assert sorted(Path(p).name for p in added.written) == ["added.json", "big.layout.json"]
    assert store.read_logic("big")["nodes"][10]["id"] == "added"
