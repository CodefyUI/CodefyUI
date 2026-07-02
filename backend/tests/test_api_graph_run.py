"""Tests for POST /api/graph/run/{name} and GET /api/graph/contract/{name}."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.run_output_store import RunOutputStore
from app.main import app

ENVELOPE_KEYS = {"status", "run_id", "graph", "device", "outputs", "error", "timing"}


# ── config: body cap setting ─────────────────────────────────────────────


def test_max_run_body_bytes_default():
    assert settings.MAX_RUN_BODY_BYTES == 64 * 1024 * 1024


def test_max_run_body_bytes_env_override(monkeypatch):
    monkeypatch.setenv("CODEFYUI_MAX_RUN_BODY_BYTES", "1024")
    from app.config import Settings

    assert Settings().MAX_RUN_BODY_BYTES == 1024


# ── envelope builders ────────────────────────────────────────────────

from app.api.routes_graph_run import build_envelope, error_response  # noqa: E402


def test_build_envelope_has_all_keys_with_nulls():
    env = build_envelope(status="ok", run_id="r1", graph="g", outputs={"y": 1})
    assert set(env.keys()) == ENVELOPE_KEYS
    assert env["status"] == "ok"
    assert env["run_id"] == "r1"
    assert env["graph"] == "g"
    assert env["device"] is None
    assert env["outputs"] == {"y": 1}
    assert env["error"] is None
    assert env["timing"] is None


def test_error_response_mirrors_status_and_keeps_all_keys():
    resp = error_response(
        409, run_id="r2", graph="g", code="invalid_contract",
        message="broken", details=["p1", "p2"],
    )
    assert resp.status_code == 409
    body = json.loads(resp.body)
    assert set(body.keys()) == ENVELOPE_KEYS
    assert body["status"] == "error"
    assert body["run_id"] == "r2"
    assert body["outputs"] is None
    assert body["error"] == {
        "code": "invalid_contract", "message": "broken",
        "node_id": None, "details": ["p1", "p2"],
    }
    assert body["timing"] is None


def test_error_response_carries_device_node_id_timing_when_known():
    resp = error_response(
        500, run_id="r3", graph="g", code="execution_error",
        message="node blew up", device="cpu", node_id="n7",
        timing={"total_s": 1.02},
    )
    body = json.loads(resp.body)
    assert body["device"] == "cpu"
    assert body["error"]["node_id"] == "n7"
    assert body["timing"] == {"total_s": 1.02}


# ── shared fixtures + graph builders for endpoint tests ──────────────────


@pytest.fixture(autouse=True)
def _graphs_dir(tmp_path, monkeypatch):
    """Isolate saved graphs per test (pattern: test_api_graph.py)."""
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    return tmp_path


def _echo_graph(
    name: str = "echo-graph",
    *,
    input_name: str = "x",
    output_name: str = "y",
    input_type: str = "string",
    required: bool = True,
    default: str = "",
) -> dict:
    """Start -> GraphInput -> GraphOutput; the minimal contract-complete graph."""
    return {
        "name": name,
        "description": "",
        "nodes": [
            {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
             "data": {"params": {}}},
            {"id": "gi", "type": "GraphInput", "position": {"x": 200, "y": 0},
             "data": {"params": {
                 "name": input_name, "type": input_type, "required": required,
                 "default": default, "description": "",
             }}},
            {"id": "out", "type": "GraphOutput", "position": {"x": 400, "y": 0},
             "data": {"params": {"name": output_name, "description": ""}}},
        ],
        "edges": [
            {"id": "t1", "source": "start", "target": "gi",
             "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
            {"id": "d1", "source": "gi", "target": "out",
             "sourceHandle": "value", "targetHandle": "value", "type": "data"},
        ],
    }


async def _save_graph(client, graph: dict) -> None:
    resp = await client.post("/api/graph/save", json=graph)
    assert resp.status_code == 200, resp.text


# ── GET /api/graph/contract/{name} ───────────────────────────────────────


@pytest.mark.asyncio
async def test_contract_endpoint_schema(test_client):
    graph = _echo_graph(name="contract-demo")
    # A typed source gives output type derivation a real port to follow:
    # TextInput.text is STRING.
    graph["nodes"].append({"id": "ti", "type": "TextInput",
                           "position": {"x": 0, "y": 200},
                           "data": {"params": {"value": "hi"}}})
    graph["nodes"].append({"id": "out2", "type": "GraphOutput",
                           "position": {"x": 400, "y": 200},
                           "data": {"params": {"name": "text_out", "description": ""}}})
    graph["edges"].append({"id": "t2", "source": "start", "target": "ti",
                           "sourceHandle": "trigger", "targetHandle": "",
                           "type": "trigger"})
    graph["edges"].append({"id": "d2", "source": "ti", "target": "out2",
                           "sourceHandle": "text", "targetHandle": "value",
                           "type": "data"})
    await _save_graph(test_client, graph)

    resp = await test_client.get("/api/graph/contract/contract-demo")
    assert resp.status_code == 200
    body = resp.json()
    assert body["graph"] == "contract-demo"
    assert body["problems"] == []
    # default is null for required inputs — never advertise a default the
    # API will not apply.
    assert body["inputs"] == [
        {"name": "x", "type": "string", "required": True,
         "default": None, "description": ""},
    ]
    by_name = {o["name"]: o for o in body["outputs"]}
    assert by_name["y"]["type"] == "ANY"            # fed by GraphInput's ANY port
    assert by_name["text_out"]["type"] == "STRING"  # derived from TextInput.text


@pytest.mark.asyncio
async def test_contract_optional_default_parsed(test_client):
    await _save_graph(test_client, _echo_graph(
        name="contract-optional", input_type="number",
        required=False, default="2.5",
    ))
    resp = await test_client.get("/api/graph/contract/contract-optional")
    body = resp.json()
    assert body["inputs"][0]["required"] is False
    assert body["inputs"][0]["default"] == 2.5


@pytest.mark.asyncio
async def test_contract_reports_problems_nonfatally(test_client):
    await _save_graph(test_client, _echo_graph(
        name="contract-broken", input_name="has space",
    ))
    resp = await test_client.get("/api/graph/contract/contract-broken")
    assert resp.status_code == 200  # problems are reported here, enforced at /run
    assert any("is invalid" in p for p in resp.json()["problems"])


@pytest.mark.asyncio
async def test_contract_unconnected_output_is_any_plus_problem(test_client):
    graph = _echo_graph(name="contract-dangling")
    graph["edges"] = [e for e in graph["edges"] if e["id"] != "d1"]  # cut gi->out
    await _save_graph(test_client, graph)
    resp = await test_client.get("/api/graph/contract/contract-dangling")
    body = resp.json()
    assert body["outputs"][0]["type"] == "ANY"
    assert any("no incoming connection" in p for p in body["problems"])


@pytest.mark.asyncio
async def test_contract_404_missing_and_strict_name(test_client, _graphs_dir):
    resp = await test_client.get("/api/graph/contract/never-saved")
    assert resp.status_code == 404
    # Strict name: the file exists under the SANITIZED name, but the raw
    # name mismatches -> 404, never alias to a different file.
    await _save_graph(test_client, _echo_graph(name="strict.name"))
    assert (_graphs_dir / "strict_name.json").exists()
    resp = await test_client.get("/api/graph/contract/strict.name")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Graph 'strict.name' not found"}
