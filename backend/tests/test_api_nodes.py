"""Tests for the nodes API endpoints."""

import pytest


@pytest.mark.asyncio
async def test_list_nodes(test_client):
    resp = await test_client.get("/api/nodes")
    assert resp.status_code == 200
    nodes = resp.json()
    assert isinstance(nodes, list)
    assert len(nodes) >= 1
    # Check structure of first node
    node = nodes[0]
    assert "node_name" in node
    assert "category" in node
    assert "inputs" in node
    assert "outputs" in node
    assert "params" in node


@pytest.mark.asyncio
async def test_get_specific_node(test_client):
    resp = await test_client.get("/api/nodes/Conv2d")
    assert resp.status_code == 200
    node = resp.json()
    assert node["node_name"] == "Conv2d"
    assert node["category"] == "CNN"


@pytest.mark.asyncio
async def test_get_nonexistent_node(test_client):
    resp = await test_client.get("/api/nodes/DoesNotExist")
    assert resp.status_code == 404


# ── Script validation (core#131) ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_validate_script_accepts_an_allowlisted_script(test_client):
    resp = await test_client.post(
        "/api/nodes/script/validate",
        json={"code": "import numpy\n\ndef run(inputs, params):\n    return 1\n"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["error"] is None
    assert body["defines_run"] is True
    assert "numpy" in body["allowed_modules"]


@pytest.mark.asyncio
async def test_validate_script_rejects_a_network_import_with_the_policy_message(test_client):
    resp = await test_client.post(
        "/api/nodes/script/validate",
        json={"code": "import requests\n\ndef run(inputs, params):\n    return 1\n"},
    )
    # A rejection is a normal answer while typing, not an HTTP failure.
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert "requests" in body["error"]
    assert "custom node" in body["error"].lower()
    assert body["line"] == 1


@pytest.mark.asyncio
async def test_validate_script_reports_the_offending_line(test_client):
    resp = await test_client.post(
        "/api/nodes/script/validate",
        json={"code": "def run(inputs, params):\n    x = 1\n    return eval('2')\n"},
    )
    body = resp.json()
    assert body["ok"] is False
    assert body["line"] == 3


@pytest.mark.asyncio
async def test_validate_script_reports_a_syntax_error_line(test_client):
    resp = await test_client.post(
        "/api/nodes/script/validate",
        json={"code": "def run(inputs, params)\n    return 1\n"},
    )
    body = resp.json()
    assert body["ok"] is False
    assert body["line"] == 1


@pytest.mark.asyncio
async def test_validate_script_flags_a_missing_entry_point(test_client):
    resp = await test_client.post(
        "/api/nodes/script/validate",
        json={"code": "x = 1\n"},
    )
    body = resp.json()
    assert body["ok"] is True          # nothing about it violates the policy
    assert body["defines_run"] is False  # but it would fail at run time


@pytest.mark.asyncio
async def test_validate_script_accepts_an_empty_body(test_client):
    resp = await test_client.post("/api/nodes/script/validate", json={})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


@pytest.mark.asyncio
async def test_validate_script_refuses_an_oversized_script(test_client):
    from app.core.script_policy import MAX_SCRIPT_CHARS

    resp = await test_client.post(
        "/api/nodes/script/validate",
        json={"code": "x = 1\n" * (MAX_SCRIPT_CHARS // 4)},
    )
    body = resp.json()
    assert body["ok"] is False
    assert "limit" in body["error"]


@pytest.mark.asyncio
async def test_python_script_node_is_registered_with_a_code_param(test_client):
    resp = await test_client.get("/api/nodes/PythonScript")
    assert resp.status_code == 200
    node = resp.json()
    assert node["category"] == "Utility"
    params = {p["name"]: p for p in node["params"]}
    assert params["code"]["param_type"] == "code"
    assert params["code"]["default"].startswith("def run(")
    # The palette template shows the default one-in / one-out shape.
    assert [p["name"] for p in node["inputs"]] == ["in1"]
    assert [p["name"] for p in node["outputs"]] == ["out1"]
