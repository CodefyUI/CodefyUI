"""Tests for the graph API endpoints."""

import json

import pytest

from app.core.preset_registry import preset_registry
from app.schemas.models import InternalNodeSchema, PresetDefinition


@pytest.fixture
def _secret_preset():
    """Register a preset with an inner LLMChat (declares SECRET params) so
    the secret scrub / lint can resolve its inner node types. Cleaned up
    after the test to leave the global registry as discovered."""
    preset = PresetDefinition(
        preset_name="SecretChat",
        category="Test",
        description="",
        nodes=[InternalNodeSchema(id="chat", type="LLMChat", params={})],
        edges=[],
        exposed_inputs=[],
        exposed_outputs=[],
        exposed_params=[],
    )
    preset_registry._presets["SecretChat"] = preset
    try:
        yield preset
    finally:
        preset_registry._presets.pop("SecretChat", None)


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


@pytest.mark.asyncio
async def test_save_scrubs_secret_params(test_client, tmp_path, monkeypatch):
    """Item 1d: a filled SECRET param (LLMChat api key) is blanked before the
    graph file is written — secrets never persist to disk, even if a client
    bypasses the editor's own stripping."""
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    graph = {
        "name": "secret-graph",
        "nodes": [
            {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
             "data": {"params": {}}},
            {"id": "llm", "type": "LLMChat", "position": {"x": 0, "y": 0},
             "data": {"params": {
                 "provider": "ChatGPT API",
                 "openai_api_key": "sk-super-secret",
                 "anthropic_api_key": "sk-ant-secret",
                 "model": "gpt-5.2",
             }}},
        ],
        "edges": [],
    }
    resp = await test_client.post("/api/graph/save", json=graph)
    assert resp.status_code == 200

    saved = json.loads((tmp_path / "secret-graph.json").read_text())
    params = next(n for n in saved["nodes"] if n["id"] == "llm")["data"]["params"]
    assert params["openai_api_key"] == ""
    assert params["anthropic_api_key"] == ""
    # Non-secret params on the same node survive untouched.
    assert params["model"] == "gpt-5.2"
    assert params["provider"] == "ChatGPT API"


@pytest.mark.asyncio
async def test_save_scrubs_preset_embedded_secret(
    test_client, tmp_path, monkeypatch, _secret_preset,
):
    """C1: a hand-written graph with a secret baked into a preset node's
    internalParams is scrubbed before the file is written."""
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    graph = {
        "name": "secret-preset-graph",
        "nodes": [
            {"id": "p1", "type": "preset:SecretChat",
             "position": {"x": 0, "y": 0},
             "data": {"internalParams": {
                 "chat": {"openai_api_key": "sk-leaked-in-preset",
                          "model": "gpt-5.2"},
             }}},
        ],
        "edges": [],
    }
    resp = await test_client.post("/api/graph/save", json=graph)
    assert resp.status_code == 200

    saved = json.loads((tmp_path / "secret-preset-graph.json").read_text())
    inner = saved["nodes"][0]["data"]["internalParams"]["chat"]
    assert inner["openai_api_key"] == ""     # secret blanked
    assert inner["model"] == "gpt-5.2"       # non-secret override kept
    assert "sk-leaked-in-preset" not in json.dumps(saved)


@pytest.mark.asyncio
async def test_export_scrubs_secret_params(test_client):
    """M4: exported Python never echoes a SECRET param value (codegen dumps
    raw params in a comment for node types with no template, e.g. LLMChat)."""
    graph = {
        "name": "export-secret",
        "nodes": [
            {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
             "data": {"params": {}}},
            {"id": "llm", "type": "LLMChat", "position": {"x": 200, "y": 0},
             "data": {"params": {
                 "provider": "ChatGPT API", "model": "gpt-5.2",
                 "openai_api_key": "sk-export-secret",
             }}},
        ],
        "edges": [
            {"id": "t1", "source": "start", "target": "llm",
             "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
        ],
    }
    resp = await test_client.post("/api/graph/export", json=graph)
    assert resp.status_code == 200, resp.text
    script = resp.json()["script"]
    assert "sk-export-secret" not in script


@pytest.mark.asyncio
async def test_save_unknown_node_type_left_untouched(
    test_client, tmp_path, monkeypatch,
):
    """A node type the registry doesn't know carries no known secret params,
    so its data is written verbatim (the scrub is a no-op for it) and the
    save still succeeds."""
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    graph = {
        "name": "unknown-graph",
        "nodes": [
            {"id": "x", "type": "TotallyUnknownNode", "position": {"x": 0, "y": 0},
             "data": {"params": {"api_key": "kept-verbatim", "foo": "bar"}}},
        ],
        "edges": [],
    }
    resp = await test_client.post("/api/graph/save", json=graph)
    assert resp.status_code == 200

    saved = json.loads((tmp_path / "unknown-graph.json").read_text())
    assert saved["nodes"][0]["data"]["params"] == {
        "api_key": "kept-verbatim", "foo": "bar",
    }


@pytest.mark.asyncio
async def test_save_and_load_roundtrips_segment_groups(
    test_client, tmp_path, monkeypatch,
):
    """Item 4: segmentGroups is persisted and returned by load (previously
    silently dropped by the GraphData schema)."""
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    graph = {
        "name": "seg-graph",
        "nodes": [],
        "edges": [],
        "segmentGroups": [
            {"id": "g1", "headNodeId": "a", "tailNodeId": "b"},
            {"id": "g2", "headNodeId": "c", "tailNodeId": "d"},
        ],
    }
    resp = await test_client.post("/api/graph/save", json=graph)
    assert resp.status_code == 200

    resp = await test_client.get("/api/graph/load/seg-graph")
    assert resp.status_code == 200
    loaded = resp.json()
    assert loaded["segmentGroups"] == [
        {"id": "g1", "headNodeId": "a", "tailNodeId": "b"},
        {"id": "g2", "headNodeId": "c", "tailNodeId": "d"},
    ]
