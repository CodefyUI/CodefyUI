"""Tests for the graph API endpoints."""

import pytest


@pytest.mark.asyncio
async def test_health(test_client):
    resp = await test_client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["nodes_loaded"] >= 1


@pytest.mark.asyncio
async def test_validate_valid_graph(test_client, sample_graph):
    resp = await test_client.post("/api/graph/validate", json=sample_graph)
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is True
    assert data["errors"] == []


@pytest.mark.asyncio
async def test_validate_invalid_graph(test_client):
    graph = {
        "nodes": [
            {"id": "1", "type": "Loss", "position": {"x": 0, "y": 0}, "data": {}},
            {"id": "2", "type": "Conv2d", "position": {"x": 0, "y": 0}, "data": {}},
        ],
        "edges": [
            {"id": "e1", "source": "1", "target": "2", "sourceHandle": "loss_fn", "targetHandle": "tensor"},
        ],
        "name": "bad-graph",
    }
    resp = await test_client.post("/api/graph/validate", json=graph)
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is False
    assert len(data["errors"]) > 0


@pytest.mark.asyncio
async def test_save_and_load_roundtrip(test_client, sample_graph, tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    # Save
    resp = await test_client.post("/api/graph/save", json=sample_graph)
    assert resp.status_code == 200
    assert "path" in resp.json()

    # Load
    resp = await test_client.get("/api/graph/load/test-graph")
    assert resp.status_code == 200
    loaded = resp.json()
    assert loaded["name"] == "test-graph"
    assert len(loaded["nodes"]) == 3  # Start + _TestSource + Print

    # List
    resp = await test_client.get("/api/graph/list")
    assert resp.status_code == 200
    graphs = resp.json()
    assert any(g["name"] == "test-graph" for g in graphs)


def test_sanitize_name_helper():
    from app.api.routes_graph import _sanitize_name

    assert _sanitize_name("my-graph_2") == "my-graph_2"
    assert _sanitize_name("a b.c/d") == "a_b_c_d"


def test_graph_path_helper(monkeypatch, tmp_path):
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    from app.api.routes_graph import _graph_path

    p = _graph_path("weird name")
    assert p == tmp_path / "weird_name.json"
