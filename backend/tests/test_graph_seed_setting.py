"""The run seed is saved with the graph as ``settings.seed`` (2.8.9).

Until 2.8.9 the seed was a property of the RUN only: ``/save`` dropped any
``settings`` block without a device, and project mode copied ``device``
alone, so a graph never carried the seed it was built and checked with. The
seed now follows the ``settings.device`` pattern end to end: written only when
set, bounded to the run path's ``0..MAX_SEED`` (422 otherwise), kept by the
project split and merge, and baked into an export as ``GRAPH_SEED`` when the
request has no top-level ``seed`` of its own.
"""

import json

import pytest

from app.api import routes_graph
from app.core.project import merge_graph, split_graph
from app.core.seeding import MAX_SEED


def _graph(name="seeded", settings=None):
    graph = {
        "name": name,
        "nodes": [
            {"id": "start", "type": "Start",
             "position": {"x": 0, "y": 0}, "data": {"params": {}}},
            {"id": "flip", "type": "RandomHorizontalFlip",
             "position": {"x": 200, "y": 0}, "data": {"params": {"p": 0.5}}},
        ],
        "edges": [
            {"id": "t1", "source": "start", "target": "flip",
             "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
        ],
    }
    if settings is not None:
        graph["settings"] = settings
    return graph


@pytest.fixture
def graphs_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    return tmp_path


# ── /save and /load (non-project mode) ──────────────────────────────────


async def test_save_writes_the_seed_and_no_device_key(test_client, graphs_dir):
    resp = await test_client.post(
        "/api/graph/save", json=_graph(settings={"seed": 7}))
    assert resp.status_code == 200, resp.text
    written = json.loads((graphs_dir / "seeded.json").read_text())
    # Neither key may be written as null: an unset device stays absent.
    assert written["settings"] == {"seed": 7}

    resp = await test_client.get("/api/graph/load/seeded")
    assert resp.status_code == 200
    assert resp.json()["settings"] == {"seed": 7}


async def test_save_writes_seed_zero(test_client, graphs_dir):
    """0 is a seed. A truthiness check would drop it."""
    resp = await test_client.post(
        "/api/graph/save", json=_graph(settings={"seed": 0}))
    assert resp.status_code == 200, resp.text
    written = json.loads((graphs_dir / "seeded.json").read_text())
    assert written["settings"] == {"seed": 0}


async def test_save_writes_device_and_seed_together(test_client, graphs_dir):
    resp = await test_client.post(
        "/api/graph/save",
        json=_graph(settings={"device": "cuda:1", "seed": MAX_SEED}))
    assert resp.status_code == 200, resp.text
    written = json.loads((graphs_dir / "seeded.json").read_text())
    assert written["settings"] == {"device": "cuda:1", "seed": MAX_SEED}


@pytest.mark.parametrize("settings", [
    None,
    {},
    {"seed": None},
    {"device": None, "seed": None},
    {"device": "", "seed": None},
])
async def test_save_with_neither_writes_no_settings(
    test_client, graphs_dir, settings,
):
    """A graph with no device and no seed keeps the file shape it had."""
    resp = await test_client.post(
        "/api/graph/save", json=_graph(settings=settings))
    assert resp.status_code == 200, resp.text
    written = json.loads((graphs_dir / "seeded.json").read_text())
    assert "settings" not in written


@pytest.mark.parametrize("seed", [-1, MAX_SEED + 1, 1.5])
@pytest.mark.parametrize("route", ["save", "validate", "export"])
async def test_a_seed_the_run_path_refuses_is_refused(
    test_client, graphs_dir, route, seed,
):
    resp = await test_client.post(
        f"/api/graph/{route}", json=_graph(settings={"seed": seed}))
    assert resp.status_code == 422, resp.text
    assert not (graphs_dir / "seeded.json").exists()


@pytest.mark.parametrize("seed", ["7", True, 2.0])
@pytest.mark.parametrize("route", ["save", "validate", "export"])
async def test_a_seed_that_is_not_an_integer_is_refused_not_coerced(
    test_client, graphs_dir, route, seed,
):
    """Strict: ``"7"``, ``true`` and ``2.0`` would otherwise be written or
    baked in as 7, 1 and 2, while the canvas reads none of them as a seed."""
    resp = await test_client.post(
        f"/api/graph/{route}", json=_graph(settings={"seed": seed}))
    assert resp.status_code == 422, resp.text
    assert not (graphs_dir / "seeded.json").exists()


@pytest.mark.parametrize("seed", ["7", True, 2.0])
async def test_export_refuses_a_run_seed_that_is_not_an_integer(
    test_client, seed,
):
    resp = await test_client.post(
        "/api/graph/export", json={**_graph(), "seed": seed})
    assert resp.status_code == 422, resp.text


# ── project mode: the logic file carries the seed ──────────────────────


@pytest.mark.parametrize("seed", [0, 7])
def test_project_split_and_merge_keep_the_seed(seed):
    payload = {**_graph(), "settings": {"seed": seed}}
    logic, layout = split_graph(payload)
    # Logic, never layout -- it is git-tracked graph content like the device.
    assert logic["settings"] == {"seed": seed}
    assert "settings" not in layout

    merged, _missing = merge_graph(logic, layout)
    assert merged["settings"] == {"seed": seed}


def test_project_split_and_merge_keep_device_and_seed_together():
    payload = {**_graph(), "settings": {"device": "mps", "seed": 3}}
    logic, layout = split_graph(payload)
    assert logic["settings"] == {"device": "mps", "seed": 3}
    merged, _missing = merge_graph(logic, layout)
    assert merged["settings"] == {"device": "mps", "seed": 3}


@pytest.mark.parametrize("settings", [
    None, {}, {"seed": None}, {"device": None, "seed": None},
])
def test_project_split_writes_no_settings_without_device_or_seed(settings):
    payload = {**_graph(), "settings": settings}
    logic, _layout = split_graph(payload)
    assert "settings" not in logic
    merged, _missing = merge_graph(logic, None)
    assert "settings" not in merged


async def test_project_save_and_load_round_trip_the_seed(
    test_client, monkeypatch, tmp_path,
):
    monkeypatch.setattr(routes_graph.settings, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(routes_graph.settings, "GRAPHS_DIR", tmp_path / "graphs")
    (tmp_path / "graphs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "layout").mkdir(parents=True, exist_ok=True)

    resp = await test_client.post(
        "/api/graph/save", json=_graph(settings={"seed": 0}))
    assert resp.status_code == 200, resp.text
    logic = json.loads(
        (tmp_path / "graphs" / "seeded.graph.json").read_text())
    assert logic["settings"] == {"seed": 0}

    resp = await test_client.get("/api/graph/load/seeded")
    assert resp.status_code == 200
    assert resp.json()["settings"] == {"seed": 0}


# ── /export: the graph's seed is the script's default ──────────────────


async def test_export_bakes_the_graph_seed(test_client):
    resp = await test_client.post(
        "/api/graph/export", json=_graph(settings={"seed": 5}))
    assert resp.status_code == 200, resp.text
    assert "GRAPH_SEED = 5" in resp.json()["script"]


async def test_export_bakes_graph_seed_zero(test_client):
    resp = await test_client.post(
        "/api/graph/export", json=_graph(settings={"seed": 0}))
    assert resp.status_code == 200, resp.text
    assert "GRAPH_SEED = 0" in resp.json()["script"]


async def test_export_top_level_seed_wins_over_the_graph_seed(test_client):
    """The request's ``seed`` is the run's setting; it beats the file's."""
    body = {**_graph(settings={"seed": 5}), "seed": 9}
    resp = await test_client.post("/api/graph/export", json=body)
    assert resp.status_code == 200, resp.text
    script = resp.json()["script"]
    assert "GRAPH_SEED = 9" in script
    assert "GRAPH_SEED = 5" not in script


async def test_export_without_any_seed_is_unseeded(test_client):
    resp = await test_client.post(
        "/api/graph/export", json=_graph(settings={"device": "cpu"}))
    assert resp.status_code == 200, resp.text
    assert "GRAPH_SEED = None" in resp.json()["script"]
