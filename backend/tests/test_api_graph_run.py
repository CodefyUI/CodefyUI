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


# ── POST /api/graph/run/{name}: happy path ───────────────────────────────


@pytest.mark.asyncio
async def test_run_happy_path_all_envelope_keys(test_client):
    await _save_graph(test_client, _echo_graph(name="run-echo"))
    resp = await test_client.post("/api/graph/run/run-echo",
                                  json={"inputs": {"x": "hello"}})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == ENVELOPE_KEYS
    assert body["status"] == "ok"
    assert body["run_id"]  # non-null, non-empty
    assert body["graph"] == "run-echo"
    assert body["device"] == "cpu"  # resolve_device(None) echo
    assert body["outputs"] == {"y": "hello"}
    assert body["error"] is None
    assert body["timing"]["total_s"] >= 0


@pytest.mark.asyncio
async def test_run_body_optional_uses_default(test_client):
    await _save_graph(test_client, _echo_graph(
        name="run-default", required=False, default="fallback",
    ))
    resp = await test_client.post("/api/graph/run/run-default")  # no body at all
    assert resp.status_code == 200
    assert resp.json()["outputs"] == {"y": "fallback"}


@pytest.mark.asyncio
async def test_run_empty_json_body_equals_absent_body(test_client):
    await _save_graph(test_client, _echo_graph(
        name="run-empty", required=False, default="fallback",
    ))
    resp = await test_client.post("/api/graph/run/run-empty", json={})
    assert resp.status_code == 200
    assert resp.json()["outputs"] == {"y": "fallback"}


@pytest.mark.asyncio
async def test_run_device_echo(test_client):
    await _save_graph(test_client, _echo_graph(name="run-device"))
    resp = await test_client.post("/api/graph/run/run-device",
                                  json={"inputs": {"x": "hi"}, "device": "cpu"})
    assert resp.status_code == 200
    assert resp.json()["device"] == "cpu"


@pytest.mark.asyncio
async def test_run_typed_inputs_roundtrip(test_client):
    cases = [
        ("run-num", "number", 3, 3.0),
        ("run-int", "integer", 3.0, 3),
        ("run-bool", "boolean", True, True),
        ("run-json", "json", {"k": [1, 2]}, {"k": [1, 2]}),
    ]
    for graph_name, input_type, sent, expected in cases:
        await _save_graph(test_client, _echo_graph(
            name=graph_name, input_type=input_type,
        ))
        resp = await test_client.post(f"/api/graph/run/{graph_name}",
                                      json={"inputs": {"x": sent}})
        assert resp.status_code == 200, resp.text
        assert resp.json()["outputs"] == {"y": expected}


@pytest.mark.asyncio
async def test_run_content_type_not_used_for_dispatch(test_client):
    # A valid JSON body with a wrong Content-Type header still parses
    # (behavior pinned by the spec).
    await _save_graph(test_client, _echo_graph(name="run-ctype"))
    resp = await test_client.post(
        "/api/graph/run/run-ctype",
        content=b'{"inputs": {"x": "hi"}}',
        headers={"Content-Type": "text/plain"},
    )
    assert resp.status_code == 200
    assert resp.json()["outputs"] == {"y": "hi"}


# ── test-support nodes (registered directly, conftest _TestSource pattern) ─


class _SlowPassNode(BaseNode):
    """Sleeps `seconds` in the executor thread, then passes value through."""

    NODE_NAME = "_SlowPass"
    CATEGORY = "Test"
    DESCRIPTION = "Sleeps, then passes through"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        time.sleep(float(params.get("seconds", 2.0)))
        return {"value": inputs.get("value")}


class _BoomNode(BaseNode):
    """Raises on execute — drives the execution_error taxonomy row."""

    NODE_NAME = "_Boom"
    CATEGORY = "Test"
    DESCRIPTION = "Raises RuntimeError"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom: intentional test failure")


class _OpaqueNode(BaseNode):
    """Emits a non-serializable object — drives unserializable_output."""

    NODE_NAME = "_Opaque"
    CATEGORY = "Test"
    DESCRIPTION = "Emits object()"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        return {"value": object()}


class _BigTensorNode(BaseNode):
    """Emits a 65,537-element tensor — drives output_too_large."""

    NODE_NAME = "_BigTensor"
    CATEGORY = "Test"
    DESCRIPTION = "Emits a tensor over the serialization cap"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        return {"value": torch.zeros(65537)}


@pytest.fixture(autouse=True)
def _register_test_nodes():
    """Same direct-injection pattern conftest uses for _TestSource."""
    from app.core.node_registry import registry

    registry._nodes["_SlowPass"] = _SlowPassNode
    registry._nodes["_Boom"] = _BoomNode
    registry._nodes["_Opaque"] = _OpaqueNode
    registry._nodes["_BigTensor"] = _BigTensorNode
    yield


def _chain_graph(name: str, middle_type: str,
                 middle_params: dict | None = None) -> dict:
    """Start -> GraphInput -> <middle> -> GraphOutput."""
    g = _echo_graph(name=name)
    g["nodes"].insert(2, {"id": "mid", "type": middle_type,
                          "position": {"x": 300, "y": 0},
                          "data": {"params": middle_params or {}}})
    g["edges"] = [
        {"id": "t1", "source": "start", "target": "gi",
         "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
        {"id": "d1", "source": "gi", "target": "mid",
         "sourceHandle": "value", "targetHandle": "value", "type": "data"},
        {"id": "d2", "source": "mid", "target": "out",
         "sourceHandle": "value", "targetHandle": "value", "type": "data"},
    ]
    return g


# ── POST /run error taxonomy ─────────────────────────────────────────────

from app.core.graph_engine import GraphValidationError  # noqa: E402


@pytest.mark.asyncio
async def test_run_404_missing_graph(test_client):
    resp = await test_client.post("/api/graph/run/never-saved", json={})
    assert resp.status_code == 404
    body = resp.json()
    assert set(body.keys()) == ENVELOPE_KEYS
    assert body["status"] == "error"
    assert body["error"]["code"] == "graph_not_found"
    assert body["run_id"]                    # never null, even on rejections
    assert body["device"] is None            # rejected before device resolution
    assert body["timing"] is None            # execution never attempted


@pytest.mark.asyncio
async def test_run_404_strict_name(test_client, _graphs_dir):
    # The file exists under the SANITIZED name; the raw name mismatches.
    await _save_graph(test_client, _echo_graph(name="strict.run"))
    assert (_graphs_dir / "strict_run.json").exists()
    resp = await test_client.post("/api/graph/run/strict.run",
                                  json={"inputs": {"x": "hi"}})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "graph_not_found"


@pytest.mark.asyncio
async def test_run_500_graph_unreadable(test_client, _graphs_dir):
    (_graphs_dir / "corrupt.json").write_text("{not json")
    resp = await test_client.post("/api/graph/run/corrupt", json={})
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "graph_unreadable"


@pytest.mark.asyncio
async def test_run_409_invalid_contract(test_client):
    await test_client.post("/api/graph/save",
                           json=_echo_graph(name="bad-contract",
                                            input_name="has space"))
    resp = await test_client.post("/api/graph/run/bad-contract", json={})
    assert resp.status_code == 409
    body = resp.json()
    assert body["error"]["code"] == "invalid_contract"
    assert any("is invalid" in d for d in body["error"]["details"])


@pytest.mark.asyncio
async def test_run_409_no_entry_points(test_client):
    graph = _echo_graph(name="no-entry")
    graph["nodes"] = [n for n in graph["nodes"] if n["type"] != "Start"]
    graph["edges"] = [e for e in graph["edges"] if e["type"] != "trigger"]
    await _save_graph(test_client, graph)
    resp = await test_client.post("/api/graph/run/no-entry", json={})
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "no_entry_points"


@pytest.mark.asyncio
async def test_run_409_untriggered_input(test_client):
    graph = _echo_graph(name="untriggered")
    # Retarget the trigger away from the GraphInput to a bystander node.
    graph["nodes"].append({"id": "src", "type": "_TestSource",
                           "position": {"x": 0, "y": 200},
                           "data": {"params": {}}})
    graph["edges"][0]["target"] = "src"
    await _save_graph(test_client, graph)
    resp = await test_client.post("/api/graph/run/untriggered",
                                  json={"inputs": {"x": "hi"}})
    assert resp.status_code == 409
    body = resp.json()
    assert body["error"]["code"] == "untriggered_input"
    assert body["error"]["details"] == ["x"]


@pytest.mark.asyncio
async def test_run_409_unreachable_output(test_client):
    graph = _echo_graph(name="unreachable")
    graph["nodes"].append({"id": "src2", "type": "_TestSource",
                           "position": {"x": 0, "y": 200},
                           "data": {"params": {}}})
    graph["nodes"].append({"id": "out2", "type": "GraphOutput",
                           "position": {"x": 400, "y": 200},
                           "data": {"params": {"name": "y2", "description": ""}}})
    graph["edges"].append({"id": "d9", "source": "src2", "target": "out2",
                           "sourceHandle": "value", "targetHandle": "value",
                           "type": "data"})
    await _save_graph(test_client, graph)
    resp = await test_client.post("/api/graph/run/unreachable",
                                  json={"inputs": {"x": "hi"}})
    assert resp.status_code == 409
    body = resp.json()
    assert body["error"]["code"] == "unreachable_output"
    assert body["error"]["details"] == ["y2"]


@pytest.mark.asyncio
async def test_run_409_invalid_graph_static(test_client):
    graph = _echo_graph(name="bad-port")
    graph["nodes"].append({"id": "pr", "type": "Print",
                           "position": {"x": 300, "y": 200},
                           "data": {"params": {}}})
    graph["edges"].append({"id": "d8", "source": "gi", "target": "pr",
                           "sourceHandle": "value", "targetHandle": "bogus",
                           "type": "data"})
    await _save_graph(test_client, graph)
    resp = await test_client.post("/api/graph/run/bad-port",
                                  json={"inputs": {"x": "hi"}})
    assert resp.status_code == 409
    body = resp.json()
    assert body["error"]["code"] == "invalid_graph"
    assert any("bogus" in d for d in body["error"]["details"])


@pytest.mark.asyncio
async def test_run_409_runtime_graph_validation_error_safety_net(
    test_client, monkeypatch,
):
    await _save_graph(test_client, _echo_graph(name="runtime-gve"))

    async def _boom(*args, **kwargs):
        raise GraphValidationError("preset trigger edge dangling after expansion")

    monkeypatch.setattr("app.api.routes_graph_run.execute_graph", _boom)
    resp = await test_client.post("/api/graph/run/runtime-gve",
                                  json={"inputs": {"x": "hi"}})
    assert resp.status_code == 409
    body = resp.json()
    assert body["error"]["code"] == "invalid_graph"
    assert body["error"]["details"] == [
        "preset trigger edge dangling after expansion",
    ]
    assert body["timing"] is not None  # execution WAS attempted


@pytest.mark.asyncio
async def test_run_422_input_errors_aggregated(test_client):
    await _save_graph(test_client, _echo_graph(name="agg-errors",
                                               input_type="number"))
    resp = await test_client.post("/api/graph/run/agg-errors",
                                  json={"inputs": {"x": "NaN-ish", "typo": 1}})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "invalid_input"
    by_input = {d["input"]: d["reason"] for d in body["error"]["details"]}
    assert by_input["typo"] == "unknown input name"
    assert "expected number" in by_input["x"]


@pytest.mark.asyncio
async def test_run_422_missing_required(test_client):
    await _save_graph(test_client, _echo_graph(name="missing-req"))
    resp = await test_client.post("/api/graph/run/missing-req", json={})
    assert resp.status_code == 422
    assert resp.json()["error"]["details"] == [
        {"input": "x", "reason": "missing required input"},
    ]


@pytest.mark.asyncio
async def test_run_422_malformed_bodies(test_client):
    await _save_graph(test_client, _echo_graph(name="malformed"))
    # Bad JSON.
    resp = await test_client.post("/api/graph/run/malformed",
                                  content=b"{not json",
                                  headers={"Content-Type": "application/json"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_input"
    # Non-dict body.
    resp = await test_client.post("/api/graph/run/malformed", json=[1, 2])
    assert resp.status_code == 422
    assert any(d["field"] == "body" for d in resp.json()["error"]["details"])
    # Non-dict inputs.
    resp = await test_client.post("/api/graph/run/malformed",
                                  json={"inputs": [1]})
    assert resp.status_code == 422
    assert any(d["field"] == "inputs" for d in resp.json()["error"]["details"])
    # Wrong-typed / out-of-range fields.
    for bad in ({"timeout_s": "fast"}, {"timeout_s": 0}, {"timeout_s": 5000},
                {"timeout_s": True}, {"device": 3}, {"record_outputs": "yes"}):
        resp = await test_client.post("/api/graph/run/malformed", json=bad)
        assert resp.status_code == 422, bad
        assert resp.json()["error"]["code"] == "invalid_input"


@pytest.mark.asyncio
async def test_run_413_payload_too_large(test_client, monkeypatch):
    await _save_graph(test_client, _echo_graph(name="too-big"))
    monkeypatch.setattr("app.config.settings.MAX_RUN_BODY_BYTES", 16)
    resp = await test_client.post("/api/graph/run/too-big",
                                  json={"inputs": {"x": "0123456789abcdef0123"}})
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "payload_too_large"


@pytest.mark.asyncio
async def test_run_500_execution_error_carries_node_id(test_client):
    await _save_graph(test_client, _chain_graph("boom-graph", "_Boom"))
    resp = await test_client.post("/api/graph/run/boom-graph",
                                  json={"inputs": {"x": "hi"}})
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"]["code"] == "execution_error"
    assert body["error"]["node_id"] == "mid"
    assert "boom: intentional test failure" in body["error"]["message"]
    assert body["timing"] is not None


@pytest.mark.asyncio
async def test_run_500_timeout(test_client):
    await _save_graph(test_client, _chain_graph("slow-graph", "_SlowPass",
                                                {"seconds": 3.0}))
    resp = await test_client.post("/api/graph/run/slow-graph",
                                  json={"inputs": {"x": "hi"}, "timeout_s": 1})
    assert resp.status_code == 500  # never 504 — 504 belongs to intermediaries
    body = resp.json()
    assert set(body.keys()) == ENVELOPE_KEYS
    assert body["status"] == "error"
    assert body["error"]["code"] == "timeout"
    assert body["timing"]["total_s"] >= 0.9
    # The in-flight executor thread finishes in the background (documented
    # limitation); nothing further to assert here.


@pytest.mark.asyncio
async def test_run_500_output_not_produced_safety_net(test_client, monkeypatch):
    await _save_graph(test_client, _echo_graph(name="net-missing"))
    monkeypatch.setattr("app.core.api_contract.collect_outputs",
                        lambda contract, result: ({}, ["y"]))
    resp = await test_client.post("/api/graph/run/net-missing",
                                  json={"inputs": {"x": "hi"}})
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"]["code"] == "output_not_produced"
    assert body["error"]["details"] == ["y"]


@pytest.mark.asyncio
async def test_run_500_unserializable_output(test_client):
    await _save_graph(test_client, _chain_graph("opaque-graph", "_Opaque"))
    resp = await test_client.post("/api/graph/run/opaque-graph",
                                  json={"inputs": {"x": "hi"}})
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"]["code"] == "unserializable_output"
    assert body["error"]["details"][0]["output"] == "y"


@pytest.mark.asyncio
async def test_run_500_output_too_large(test_client):
    await _save_graph(test_client, _chain_graph("big-graph", "_BigTensor"))
    resp = await test_client.post("/api/graph/run/big-graph",
                                  json={"inputs": {"x": "hi"}})
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"]["code"] == "output_too_large"
    assert "65537" in body["error"]["details"][0]["reason"]


@pytest.mark.asyncio
async def test_run_403_without_token_is_out_of_envelope():
    # The auth middleware fires before the route; its 403 does NOT carry
    # the envelope (documented out-of-envelope response).
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport,
                           base_url=f"http://127.0.0.1:{settings.PORT}") as anon:
        resp = await anon.post("/api/graph/run/anything", json={})
    assert resp.status_code == 403
    body = resp.json()
    assert body == {"detail": "Missing or invalid X-CodefyUI-Token header"}
    assert "run_id" not in body


# ── record_outputs + concurrency + disconnect policy ─────────────────────


@pytest.mark.asyncio
async def test_record_outputs_retrievable_by_run_id(test_client):
    # Store set explicitly — the lifespan does not run under ASGITransport
    # (pattern: test_routes_execution_outputs.py).
    app.state.run_output_store = RunOutputStore(max_runs=5)
    await _save_graph(test_client, _echo_graph(name="recorded"))
    resp = await test_client.post(
        "/api/graph/run/recorded",
        json={"inputs": {"x": "keep me"}, "record_outputs": True},
    )
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    listing = await test_client.get(f"/api/execution/outputs/{run_id}")
    assert listing.status_code == 200
    entries = listing.json()
    assert any(e["node_id"] == "out" and e["port"] == "value" for e in entries)


@pytest.mark.asyncio
async def test_record_outputs_without_store_still_succeeds(test_client):
    # getattr fallback: absent attribute means recording is skipped, not a 500.
    had_store = hasattr(app.state, "run_output_store")
    saved = getattr(app.state, "run_output_store", None)
    if had_store:
        delattr(app.state, "run_output_store")
    try:
        await _save_graph(test_client, _echo_graph(name="no-store"))
        resp = await test_client.post(
            "/api/graph/run/no-store",
            json={"inputs": {"x": "hi"}, "record_outputs": True},
        )
        assert resp.status_code == 200
        assert resp.json()["outputs"] == {"y": "hi"}
    finally:
        if had_store:
            app.state.run_output_store = saved


@pytest.mark.asyncio
async def test_record_outputs_default_off(test_client):
    app.state.run_output_store = RunOutputStore(max_runs=5)
    await _save_graph(test_client, _echo_graph(name="not-recorded"))
    resp = await test_client.post("/api/graph/run/not-recorded",
                                  json={"inputs": {"x": "hi"}})
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]
    listing = await test_client.get(f"/api/execution/outputs/{run_id}")
    assert listing.status_code == 404  # nothing recorded by default


@pytest.mark.asyncio
async def test_concurrent_runs_are_independent(test_client):
    await _save_graph(test_client, _echo_graph(name="concurrent"))
    resp_a, resp_b = await asyncio.gather(
        test_client.post("/api/graph/run/concurrent",
                         json={"inputs": {"x": "alpha"}}),
        test_client.post("/api/graph/run/concurrent",
                         json={"inputs": {"x": "beta"}}),
    )
    assert resp_a.status_code == 200 and resp_b.status_code == 200
    assert resp_a.json()["outputs"] == {"y": "alpha"}
    assert resp_b.json()["outputs"] == {"y": "beta"}
    assert resp_a.json()["run_id"] != resp_b.json()["run_id"]


@pytest.mark.asyncio
async def test_disconnect_does_not_cancel_run(test_client):
    """Spec-normative disconnect policy: a client disconnect never cancels
    the run; only the timeout stops a run. Cancelling the client request
    task mid-run simulates the disconnect; the surviving run is observable
    through the record_outputs store."""
    store = RunOutputStore(max_runs=5)
    app.state.run_output_store = store
    await _save_graph(test_client, _chain_graph("survives", "_SlowPass",
                                                {"seconds": 1.0}))

    request_task = asyncio.create_task(test_client.post(
        "/api/graph/run/survives",
        json={"inputs": {"x": "still here"}, "record_outputs": True},
    ))
    await asyncio.sleep(0.3)   # let the run reach the _SlowPass node
    request_task.cancel()      # the client drops the connection
    with pytest.raises(asyncio.CancelledError):
        await request_task

    # The shielded run keeps going: poll the store until the GraphOutput
    # node's value lands (written only when the run reaches the end).
    deadline = time.monotonic() + 10.0
    recorded = None
    while time.monotonic() < deadline:
        for run_id in await store.list_runs():
            value = await store.get(run_id, "out", "value")
            if value is not None:
                recorded = value
                break
        if recorded is not None:
            break
        await asyncio.sleep(0.1)
    assert recorded == "still here"
