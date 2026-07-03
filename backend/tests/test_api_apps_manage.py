"""Tests for the /api/apps management surface (spec Section 6.2):
publish / versions / record_io / activate / unpublish / delete."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.main import app

SLUG = "demo-app"


@pytest.fixture(autouse=True)
def _graphs_dir(tmp_path, monkeypatch):
    """Isolate saved graphs per test (pattern: test_api_graph_run.py)."""
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    return tmp_path


def _echo_graph(
    name: str = "pub-src",
    *,
    input_name: str = "x",
    output_name: str = "y",
    input_type: str = "string",
    required: bool = True,
    default: str = "",
) -> dict:
    """Start -> GraphInput -> GraphOutput (duplicated from
    test_api_graph_run.py so the test modules stay independent)."""
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


async def _publish(client, slug: str, graph_name: str, **overrides) -> dict:
    payload = {"graph": graph_name, "create": True, **overrides}
    resp = await client.post(f"/api/apps/{slug}/publish", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


# ── publish (Task 6) ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_publish_new_slug_without_create_flag_404(test_client, app_db):
    await _save_graph(test_client, _echo_graph())
    resp = await test_client.post(f"/api/apps/{SLUG}/publish",
                                  json={"graph": "pub-src"})
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "app_not_found"
    # Typo-proofing: the rejected publish created nothing.
    count = await app_db.run(lambda conn: conn.execute(
        "SELECT COUNT(*) FROM apps").fetchone()[0])
    assert count == 0


@pytest.mark.asyncio
async def test_publish_create_then_republish_versions(test_client, app_db):
    await _save_graph(test_client, _echo_graph())
    first = await _publish(test_client, SLUG, "pub-src", note="first cut")
    assert first == {
        "slug": SLUG, "version": 1, "active": True, "created": True,
        "graph_name": "pub-src", "note": "first cut",
    }
    second = await _publish(test_client, SLUG, "pub-src")
    assert second["version"] == 2
    assert second["created"] is False
    assert second["active"] is True
    assert second["note"] is None

    def _db_state(conn: sqlite3.Connection):
        app_row = conn.execute(
            "SELECT active_version, graph_name FROM apps WHERE slug = ?",
            (SLUG,)).fetchone()
        versions = conn.execute(
            "SELECT version, note FROM app_versions ORDER BY version",
        ).fetchall()
        return dict(app_row), [tuple(v) for v in versions]

    app_row, versions = await app_db.run(_db_state)
    assert app_row["active_version"] == 2   # publish activates immediately
    assert versions == [(1, "first cut"), (2, None)]


@pytest.mark.asyncio
async def test_publish_invalid_slugs_422(test_client, app_db):
    await _save_graph(test_client, _echo_graph())
    for bad in ("UPPER", "9starts-with-digit", "has space", "has_underscore",
                "-leading-dash", "a" * 65):
        resp = await test_client.post(
            f"/api/apps/{bad}/publish",
            json={"graph": "pub-src", "create": True},
        )
        assert resp.status_code == 422, bad
        assert resp.json()["detail"]["code"] == "invalid_slug", bad


@pytest.mark.asyncio
async def test_publish_missing_graph_strict_name_404(test_client, app_db):
    resp = await test_client.post(
        f"/api/apps/{SLUG}/publish",
        json={"graph": "never-saved", "create": True},
    )
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "graph_not_found"

    # Strict-name rule (routes_graph_run.py behavior): the file exists
    # under the SANITIZED name; the raw name must never alias to it.
    await _save_graph(test_client, _echo_graph(name="strict.pub"))
    resp = await test_client.post(
        f"/api/apps/{SLUG}/publish",
        json={"graph": "strict.pub", "create": True},
    )
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "graph_not_found"


@pytest.mark.asyncio
async def test_publish_preflight_blocks_bad_graphs_with_stage1_codes(
    test_client, app_db,
):
    # invalid_contract: bad input name.
    await _save_graph(test_client, _echo_graph(name="bad-name",
                                               input_name="has space"))
    resp = await test_client.post(
        "/api/apps/pre-a/publish", json={"graph": "bad-name", "create": True})
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["code"] == "invalid_contract"
    assert any("is invalid" in d for d in detail["details"])

    # untriggered_input: trigger retargeted away from the GraphInput.
    graph = _echo_graph(name="untriggered-pub")
    graph["nodes"].append({"id": "src", "type": "_TestSource",
                           "position": {"x": 0, "y": 200},
                           "data": {"params": {}}})
    graph["edges"][0]["target"] = "src"
    await _save_graph(test_client, graph)
    resp = await test_client.post(
        "/api/apps/pre-b/publish",
        json={"graph": "untriggered-pub", "create": True})
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "untriggered_input"

    # no_entry_points: Start removed.
    graph = _echo_graph(name="no-entry-pub")
    graph["nodes"] = [n for n in graph["nodes"] if n["type"] != "Start"]
    graph["edges"] = [e for e in graph["edges"] if e["type"] != "trigger"]
    await _save_graph(test_client, graph)
    resp = await test_client.post(
        "/api/apps/pre-c/publish",
        json={"graph": "no-entry-pub", "create": True})
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "no_entry_points"

    # unreachable_output: an output fed only by an untriggered source.
    graph = _echo_graph(name="unreach-pub")
    graph["nodes"].append({"id": "src2", "type": "_TestSource",
                           "position": {"x": 0, "y": 200},
                           "data": {"params": {}}})
    graph["nodes"].append({"id": "out2", "type": "GraphOutput",
                           "position": {"x": 400, "y": 200},
                           "data": {"params": {"name": "y2",
                                               "description": ""}}})
    graph["edges"].append({"id": "d9", "source": "src2", "target": "out2",
                           "sourceHandle": "value", "targetHandle": "value",
                           "type": "data"})
    await _save_graph(test_client, graph)
    resp = await test_client.post(
        "/api/apps/pre-d/publish",
        json={"graph": "unreach-pub", "create": True})
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "unreachable_output"

    # invalid_graph: bogus target handle.
    graph = _echo_graph(name="badport-pub")
    graph["nodes"].append({"id": "pr", "type": "Print",
                           "position": {"x": 300, "y": 200},
                           "data": {"params": {}}})
    graph["edges"].append({"id": "d8", "source": "gi", "target": "pr",
                           "sourceHandle": "value", "targetHandle": "bogus",
                           "type": "data"})
    await _save_graph(test_client, graph)
    resp = await test_client.post(
        "/api/apps/pre-e/publish",
        json={"graph": "badport-pub", "create": True})
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "invalid_graph"

    # No rejected pre-flight published anything.
    count = await app_db.run(lambda conn: conn.execute(
        "SELECT COUNT(*) FROM apps").fetchone()[0])
    assert count == 0


@pytest.mark.asyncio
async def test_publish_stores_contract_document_and_exact_snapshot(
    test_client, app_db, _graphs_dir,
):
    await _save_graph(test_client, _echo_graph())
    original_bytes = (_graphs_dir / "pub-src.json").read_text()
    await _publish(test_client, SLUG, "pub-src")

    def _stored(conn: sqlite3.Connection):
        row = conn.execute(
            "SELECT graph_json, contract_json FROM app_versions "
            "WHERE version = 1").fetchone()
        return row["graph_json"], json.loads(row["contract_json"])

    graph_json, contract_doc = await app_db.run(_stored)
    assert graph_json == original_bytes   # EXACT saved-file bytes (Decision B)
    assert contract_doc["graph"] == "pub-src"
    assert contract_doc["inputs"] == [{
        "name": "x", "type": "string", "required": True,
        "default": None, "description": "",
    }]
    assert contract_doc["outputs"][0]["name"] == "y"


@pytest.mark.asyncio
async def test_publish_requires_session_token(app_db):
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url=f"http://127.0.0.1:{settings.PORT}",
    ) as anon:
        resp = await anon.post(f"/api/apps/{SLUG}/publish",
                               json={"graph": "pub-src", "create": True})
    assert resp.status_code == 403
