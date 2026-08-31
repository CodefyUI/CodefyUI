"""A renamed node keeps its name through save and load (#342).

``data.label`` is written by the editor's serializer from plugin API v5 on,
and the backend has to carry it. Nothing in ``schemas/models.py`` needs to
change for that -- ``NodeData.data`` is ``dict[str, Any]`` with no ``extra=``
config -- but "nothing needs to change" is a claim, and this is the test that
makes it one you can check.

Project mode gets its own case because ``split_graph`` DOES strip keys out of
a node's ``data``: the four note-geometry keys, and only from note nodes. A
regular node's ``label`` is logic and belongs in the reviewable graph file.
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


def _graph(label=None, name="labelled"):
    data = {"params": {"name": "MNIST"}}
    if label is not None:
        data["label"] = label
    return {
        "name": name,
        "description": "",
        "nodes": [
            {"id": "a", "type": "Dataset", "position": {"x": 10, "y": 0}, "data": data},
        ],
        "edges": [],
        "presets": [],
        "segmentGroups": [],
    }


async def test_label_round_trips_in_single_file_mode(test_client, tmp_path, monkeypatch):
    monkeypatch.setattr(routes_graph.settings, "PROJECT_DIR", None)
    monkeypatch.setattr(routes_graph.settings, "GRAPHS_DIR", tmp_path)

    r = await test_client.post("/api/graph/save", json=_graph(label="Training set"))
    assert r.status_code == 200

    r = await test_client.get("/api/graph/load/labelled")
    assert r.status_code == 200
    assert r.json()["nodes"][0]["data"]["label"] == "Training set"


async def test_a_graph_without_a_label_stays_without_one(test_client, tmp_path, monkeypatch):
    monkeypatch.setattr(routes_graph.settings, "PROJECT_DIR", None)
    monkeypatch.setattr(routes_graph.settings, "GRAPHS_DIR", tmp_path)

    await test_client.post("/api/graph/save", json=_graph())
    r = await test_client.get("/api/graph/load/labelled")
    assert "label" not in r.json()["nodes"][0]["data"]


async def test_label_is_logic_in_project_mode(project_settings, test_client):
    await test_client.post("/api/graph/save", json=_graph(label="Training set"))

    logic = json.loads((project_settings / "graphs" / "labelled.graph.json").read_text())
    layout = json.loads((project_settings / "layout" / "labelled.layout.json").read_text())
    # The reviewable file is where a rename belongs: it is a change to what the
    # graph MEANS, not to where it sits.
    assert logic["nodes"][0]["data"]["label"] == "Training set"
    assert "label" not in json.dumps(layout)

    r = await test_client.get("/api/graph/load/labelled")
    assert r.json()["nodes"][0]["data"]["label"] == "Training set"


async def test_renaming_a_node_dirties_only_the_logic_file(project_settings, test_client):
    await test_client.post("/api/graph/save", json=_graph(label="Training set"))
    logic1 = (project_settings / "graphs" / "labelled.graph.json").read_text()
    layout1 = (project_settings / "layout" / "labelled.layout.json").read_text()

    await test_client.post("/api/graph/save", json=_graph(label="Validation set"))
    logic2 = (project_settings / "graphs" / "labelled.graph.json").read_text()
    layout2 = (project_settings / "layout" / "labelled.layout.json").read_text()

    assert layout1 == layout2
    assert logic1 != logic2
