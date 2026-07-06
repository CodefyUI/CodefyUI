"""Project-mode save/load round-trip (spec 10): the pair is written, a drag
only changes layout/, a param edit only changes graphs/, and load merges.

Also covers two endpoint-level gaps flagged by Task 2's reviewer (carried
into this task's spec): `/list`'s double-suffix stripping + both-forms 409
exercised over real HTTP (not just the path-resolution helpers), and the
reserved-name save guard's actual status code in both modes.
"""

import json

import pytest

from app.api import routes_graph


@pytest.fixture
def project_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(routes_graph.settings, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(routes_graph.settings, "GRAPHS_DIR", tmp_path / "graphs")
    (tmp_path / "graphs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "layout").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _graph(param_val="MNIST", x=10, name="demo"):
    return {
        "name": name,
        "description": "",
        "nodes": [
            {"id": "a", "type": "Dataset", "position": {"x": x, "y": 0},
             "data": {"params": {"name": param_val}}},
        ],
        "edges": [],
        "presets": [],
        "segmentGroups": [],
    }


async def test_save_writes_pair(project_settings, test_client):
    r = await test_client.post("/api/graph/save", json=_graph())
    assert r.status_code == 200
    assert (project_settings / "graphs" / "demo.graph.json").exists()
    assert (project_settings / "layout" / "demo.layout.json").exists()


async def test_drag_touches_only_layout(project_settings, test_client):
    await test_client.post("/api/graph/save", json=_graph(x=10))
    logic1 = (project_settings / "graphs" / "demo.graph.json").read_text()
    layout1 = (project_settings / "layout" / "demo.layout.json").read_text()
    # Same params, moved node.
    await test_client.post("/api/graph/save", json=_graph(x=999))
    logic2 = (project_settings / "graphs" / "demo.graph.json").read_text()
    layout2 = (project_settings / "layout" / "demo.layout.json").read_text()
    assert logic1 == logic2          # logic file untouched by a drag
    assert layout1 != layout2        # only the layout file changed


async def test_param_edit_touches_only_logic(project_settings, test_client):
    await test_client.post("/api/graph/save", json=_graph(param_val="MNIST"))
    logic1 = (project_settings / "graphs" / "demo.graph.json").read_text()
    layout1 = (project_settings / "layout" / "demo.layout.json").read_text()
    await test_client.post("/api/graph/save", json=_graph(param_val="CIFAR10"))
    logic2 = (project_settings / "graphs" / "demo.graph.json").read_text()
    layout2 = (project_settings / "layout" / "demo.layout.json").read_text()
    assert layout1 == layout2        # layout file untouched by a param edit
    assert logic1 != logic2          # only the logic file changed


async def test_load_merges_pair(project_settings, test_client):
    await test_client.post("/api/graph/save", json=_graph(x=42))
    r = await test_client.get("/api/graph/load/demo")
    assert r.status_code == 200
    data = r.json()
    assert data["nodes"][0]["position"] == {"x": 42, "y": 0}
    assert data["layout_missing"] is False


async def test_load_missing_layout_flags(project_settings, test_client):
    await test_client.post("/api/graph/save", json=_graph())
    (project_settings / "layout" / "demo.layout.json").unlink()
    r = await test_client.get("/api/graph/load/demo")
    data = r.json()
    assert data["layout_missing"] is True
    assert "position" not in data["nodes"][0]


async def test_legacy_upgrades_on_save(project_settings, test_client):
    legacy = project_settings / "graphs" / "demo.json"
    legacy.write_text(json.dumps(_graph(x=7)))
    # Loads via the legacy path (embedded positions).
    r = await test_client.get("/api/graph/load/demo")
    assert r.json()["nodes"][0]["position"] == {"x": 7, "y": 0}
    # First save upgrades to the pair and removes the legacy file.
    await test_client.post("/api/graph/save", json=_graph(x=7))
    assert (project_settings / "graphs" / "demo.graph.json").exists()
    assert not legacy.exists()


# -- Carried from Task 2 review: endpoint-level /list + reserved-name gaps --


async def test_list_project_mode_strips_double_suffix(project_settings, test_client):
    """Mixed canonical `.graph.json` + legacy `.json` names: /list must not
    leak the double suffix (`Path("canon.graph.json").stem` is "canon.graph",
    not "canon" -- the endpoint has to strip the whole ".graph.json")."""
    (project_settings / "graphs" / "canon.graph.json").write_text(
        json.dumps({"name": "Canon Graph", "nodes": [], "edges": []})
    )
    (project_settings / "graphs" / "legacyonly.json").write_text(
        json.dumps({"name": "Legacy Only", "nodes": [], "edges": []})
    )
    r = await test_client.get("/api/graph/list")
    assert r.status_code == 200
    by_file = {g["file"]: g["name"] for g in r.json()}
    assert by_file == {"canon": "Canon Graph", "legacyonly": "Legacy Only"}
    assert "canon.graph" not in by_file  # double suffix must not leak


async def test_list_project_mode_ambiguous_pair_409(project_settings, test_client):
    """Both `x.json` and `x.graph.json` coexisting for the same base name is
    never silently resolved -- /list 409s naming BOTH files."""
    (project_settings / "graphs" / "dup.graph.json").write_text("{}")
    (project_settings / "graphs" / "dup.json").write_text("{}")
    r = await test_client.get("/api/graph/list")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert "dup.graph.json" in detail
    assert "dup.json" in detail


async def test_save_reserved_name_400_in_project_mode(project_settings, test_client):
    """A graph literally named `weird.graph` collides with the split suffix
    in project mode -- rejected outright, nothing written."""
    r = await test_client.post("/api/graph/save", json=_graph(name="weird.graph"))
    assert r.status_code == 400
    assert "reserved" in r.json()["detail"].lower()
    # Sanitized would-be targets (weird_graph.graph.json / weird_graph.json)
    # must not exist either -- the guard fires before any write is attempted.
    assert list((project_settings / "graphs").iterdir()) == []


async def test_save_reserved_name_200_in_non_project_mode(test_client, tmp_path, monkeypatch):
    """The same name is unremarkable outside a project: no split ever
    happens, so '.graph' cannot collide with anything -- sanitized straight
    to `weird_graph.json`, same as any other punctuation (byte-for-byte
    refactor guard)."""
    monkeypatch.setattr(routes_graph.settings, "PROJECT_DIR", None)
    monkeypatch.setattr(routes_graph.settings, "GRAPHS_DIR", tmp_path)
    r = await test_client.post("/api/graph/save", json=_graph(name="weird.graph"))
    assert r.status_code == 200
    assert (tmp_path / "weird_graph.json").exists()
