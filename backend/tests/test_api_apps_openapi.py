"""GET /api/apps/{slug}/openapi.json — a COMPLETE standalone OpenAPI 3.1
document for the ACTIVE version (spec Section 6.1): never a fragment
(openapi-generator chokes on partials), contract-typed requestBody, 9-key
envelope schema, bearerAuth, servers from the validated Host, and the
x-codefyui-curl {powershell, bash} snippet object."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.main import app

SLUG = "openapi-app"


@pytest.fixture(autouse=True)
def _graphs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    return tmp_path


def _typed_graph(name: str = "openapi-src") -> dict:
    """Start triggering three typed GraphInputs (string + image required,
    json optional with default "{}") + one GraphOutput fed from x."""
    return {
        "name": name,
        "description": "",
        "nodes": [
            {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
             "data": {"params": {}}},
            {"id": "gi-x", "type": "GraphInput",
             "position": {"x": 200, "y": 0},
             "data": {"params": {"name": "x", "type": "string",
                                 "required": True, "default": "",
                                 "description": ""}}},
            {"id": "gi-photo", "type": "GraphInput",
             "position": {"x": 200, "y": 150},
             "data": {"params": {"name": "photo", "type": "image",
                                 "required": True, "default": "",
                                 "description": ""}}},
            {"id": "gi-j", "type": "GraphInput",
             "position": {"x": 200, "y": 300},
             "data": {"params": {"name": "j", "type": "json",
                                 "required": False, "default": "{}",
                                 "description": ""}}},
            {"id": "out", "type": "GraphOutput",
             "position": {"x": 400, "y": 0},
             "data": {"params": {"name": "echo", "description": ""}}},
        ],
        "edges": [
            {"id": "t1", "source": "start", "target": "gi-x",
             "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
            {"id": "t2", "source": "start", "target": "gi-photo",
             "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
            {"id": "t3", "source": "start", "target": "gi-j",
             "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
            {"id": "d1", "source": "gi-x", "target": "out",
             "sourceHandle": "value", "targetHandle": "value", "type": "data"},
        ],
    }


async def _publish(client, slug: str, graph: dict) -> None:
    resp = await client.post("/api/graph/save", json=graph)
    assert resp.status_code == 200, resp.text
    resp = await client.post(
        f"/api/apps/{slug}/publish",
        json={"graph": graph["name"], "create": True},
    )
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_openapi_document_is_complete_and_typed(test_client, app_db):
    await _publish(test_client, SLUG, _typed_graph())
    resp = await test_client.get(f"/api/apps/{SLUG}/openapi.json")
    assert resp.status_code == 200
    doc = resp.json()

    # COMPLETE standalone document — never a fragment.
    for key in ("openapi", "info", "paths", "components", "security"):
        assert key in doc, key
    assert doc["openapi"] == "3.1.0"
    assert doc["info"]["version"] == "1"
    assert doc["servers"] == [
        {"url": f"http://127.0.0.1:{settings.PORT}/api/apps/{SLUG}"},
    ]

    post = doc["paths"]["/invoke"]["post"]
    body_schema = post["requestBody"]["content"]["application/json"]["schema"]
    inputs_schema = body_schema["properties"]["inputs"]
    # Contract type table (spec 6.1 + api_contract.INPUT_TYPES).
    assert inputs_schema["properties"]["x"] == {"type": "string"}
    assert inputs_schema["properties"]["photo"] == {
        "type": "string", "contentEncoding": "base64",
    }
    assert inputs_schema["properties"]["j"] == {"default": {}}  # json -> {}
    assert inputs_schema["required"] == ["x", "photo"]

    envelope = doc["components"]["schemas"]["RunEnvelope"]
    assert set(envelope["required"]) == {
        "status", "run_id", "graph", "app", "version",
        "device", "outputs", "error", "timing",
    }
    # Codegen strictness: the error/timing subschemas declare their own
    # required arrays too (every key always present when non-null).
    assert envelope["properties"]["error"]["required"] == [
        "code", "message", "node_id", "details",
    ]
    assert envelope["properties"]["timing"]["required"] == ["total_s"]
    ref = {"$ref": "#/components/schemas/RunEnvelope"}
    assert post["responses"]["200"]["content"]["application/json"]["schema"] \
        == ref
    assert post["responses"]["default"]["content"]["application/json"]["schema"] \
        == ref
    assert doc["components"]["securitySchemes"]["bearerAuth"] == {
        "type": "http", "scheme": "bearer",
        "description": "CodefyUI API key (cdui_...)",
    }
    assert doc["security"] == [{"bearerAuth": []}]


@pytest.mark.asyncio
async def test_openapi_curl_object_carries_both_shells(test_client, app_db):
    await _publish(test_client, SLUG, _typed_graph())
    doc = (await test_client.get(f"/api/apps/{SLUG}/openapi.json")).json()
    curl = doc["x-codefyui-curl"]
    assert set(curl.keys()) == {"powershell", "bash"}
    base = f"http://127.0.0.1:{settings.PORT}/api/apps/{SLUG}/invoke"
    for snippet in curl.values():
        assert base in snippet
        assert "Authorization: Bearer" in snippet
        assert '--data "@payload.json"' in snippet
    # PowerShell aliases `curl` to Invoke-WebRequest — always curl.exe.
    assert curl["powershell"].startswith("curl.exe")
    assert curl["bash"].startswith("curl ")


@pytest.mark.asyncio
async def test_openapi_auth_either_credential_and_error_codes(
    test_client, app_db,
):
    await _publish(test_client, SLUG, _typed_graph())
    key = (await test_client.post(
        "/api/keys", json={"name": "openapi"})).json()

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url=f"http://127.0.0.1:{settings.PORT}",
    ) as anon:
        ok = await anon.get(
            f"/api/apps/{SLUG}/openapi.json",
            headers={"Authorization": f"Bearer {key['token']}"},
        )
        assert ok.status_code == 200            # API key alone
        denied = await anon.get(f"/api/apps/{SLUG}/openapi.json")
        assert denied.status_code == 401        # neither credential
        assert set(denied.json().keys()) == {"detail"}

    # Session token alone (test_client default headers).
    assert (await test_client.get(
        f"/api/apps/{SLUG}/openapi.json")).status_code == 200

    resp = await test_client.get("/api/apps/ghost/openapi.json")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "app_not_found"

    await test_client.post(f"/api/apps/{SLUG}/unpublish")
    resp = await test_client.get(f"/api/apps/{SLUG}/openapi.json")
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "app_unpublished"


@pytest.mark.asyncio
async def test_openapi_tracks_the_active_version(test_client, app_db):
    graph = _typed_graph()
    await _publish(test_client, SLUG, graph)
    resp = await test_client.post(
        f"/api/apps/{SLUG}/publish", json={"graph": graph["name"]})
    assert resp.status_code == 200
    doc = (await test_client.get(f"/api/apps/{SLUG}/openapi.json")).json()
    assert doc["info"]["version"] == "2"
    await test_client.post(f"/api/apps/{SLUG}/activate",
                           json={"version": 1})
    doc = (await test_client.get(f"/api/apps/{SLUG}/openapi.json")).json()
    assert doc["info"]["version"] == "1"        # the ACTIVE version's doc
