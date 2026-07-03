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


# ── management: list / versions / record_io / activate / unpublish /
#    delete (Task 7) ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_apps_fields(test_client, app_db):
    await _save_graph(test_client, _echo_graph())
    await _publish(test_client, "app-one", "pub-src")
    await _publish(test_client, "app-one", "pub-src")
    await _publish(test_client, "app-two", "pub-src", record_io=False)

    resp = await test_client.get("/api/apps")
    assert resp.status_code == 200
    rows = resp.json()
    assert [r["slug"] for r in rows] == ["app-one", "app-two"]
    one = rows[0]
    assert set(one.keys()) == {
        "slug", "graph_name", "active_version", "versions_count",
        "record_io", "created_at", "updated_at",
    }
    assert one["active_version"] == 2
    assert one["versions_count"] == 2
    assert one["record_io"] is True
    assert rows[1]["record_io"] is False


@pytest.mark.asyncio
async def test_versions_list_marks_active_and_echoes_note(test_client, app_db):
    await _save_graph(test_client, _echo_graph())
    await _publish(test_client, SLUG, "pub-src", note="v1 note")
    await _publish(test_client, SLUG, "pub-src")

    resp = await test_client.get(f"/api/apps/{SLUG}/versions")
    assert resp.status_code == 200
    rows = resp.json()
    assert [(r["version"], r["active"], r["note"]) for r in rows] == [
        (2, True, None), (1, False, "v1 note"),
    ]
    assert all(set(r.keys()) == {
        "version", "source_graph_name", "note", "created_at", "active",
    } for r in rows)

    resp = await test_client.get("/api/apps/nope/versions")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "app_not_found"


@pytest.mark.asyncio
async def test_patch_record_io_flips_without_new_version(test_client, app_db):
    await _save_graph(test_client, _echo_graph())
    await _publish(test_client, SLUG, "pub-src")
    resp = await test_client.patch(f"/api/apps/{SLUG}",
                                   json={"record_io": False})
    assert resp.status_code == 200
    assert resp.json() == {"slug": SLUG, "record_io": False}

    rows = (await test_client.get("/api/apps")).json()
    assert rows[0]["record_io"] is False
    assert rows[0]["versions_count"] == 1   # app state, not version state

    resp = await test_client.patch("/api/apps/nope",
                                   json={"record_io": True})
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "app_not_found"


@pytest.mark.asyncio
async def test_activate_any_version_including_from_unpublished(
    test_client, app_db,
):
    await _save_graph(test_client, _echo_graph())
    await _publish(test_client, SLUG, "pub-src")
    await _publish(test_client, SLUG, "pub-src")

    # Activate subsumes rollback — no separate rollback route.
    resp = await test_client.post(f"/api/apps/{SLUG}/activate",
                                  json={"version": 1})
    assert resp.status_code == 200
    assert resp.json() == {"slug": SLUG, "active_version": 1}

    # Unpublish keeps versions; activate-after-unpublish restores service.
    resp = await test_client.post(f"/api/apps/{SLUG}/unpublish")
    assert resp.status_code == 200
    assert resp.json() == {"slug": SLUG, "active_version": None}
    rows = (await test_client.get(f"/api/apps/{SLUG}/versions")).json()
    assert len(rows) == 2 and not any(r["active"] for r in rows)

    resp = await test_client.post(f"/api/apps/{SLUG}/activate",
                                  json={"version": 2})
    assert resp.status_code == 200
    rows = (await test_client.get(f"/api/apps/{SLUG}/versions")).json()
    assert [(r["version"], r["active"]) for r in rows] == [
        (2, True), (1, False),
    ]

    resp = await test_client.post(f"/api/apps/{SLUG}/activate",
                                  json={"version": 99})
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "version_not_found"

    resp = await test_client.post("/api/apps/nope/activate",
                                  json={"version": 1})
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "app_not_found"


@pytest.mark.asyncio
async def test_delete_cascades_versions_and_runs_and_prunes_lock(
    test_client, app_db,
):
    import asyncio as aio

    await _save_graph(test_client, _echo_graph())
    await _publish(test_client, SLUG, "pub-src")

    # Seed a runs row directly (the invoke route arrives in PR3).
    def _seed_run(conn: sqlite3.Connection) -> None:
        app_id = conn.execute("SELECT id FROM apps WHERE slug = ?",
                              (SLUG,)).fetchone()[0]
        conn.execute(
            "INSERT INTO runs (run_id, app_id, version, api_key_id, status, "
            "node_timings_json, inputs_json, outputs_json, created_at) "
            "VALUES ('seed-run', ?, 1, NULL, 'ok', '{}', '{}', '{}', "
            "'2026-01-01T00:00:00.000000Z')",
            (app_id,),
        )

    await app_db.run(_seed_run)
    app.state.app_locks[SLUG] = aio.Lock()

    resp = await test_client.delete(f"/api/apps/{SLUG}")
    assert resp.status_code == 200
    assert resp.json() == {"slug": SLUG, "deleted": True}

    def _counts(conn: sqlite3.Connection) -> tuple[int, int, int]:
        return (
            conn.execute("SELECT COUNT(*) FROM apps").fetchone()[0],
            conn.execute("SELECT COUNT(*) FROM app_versions").fetchone()[0],
            conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
        )

    assert await app_db.run(_counts) == (0, 0, 0)   # FK cascade
    assert SLUG not in app.state.app_locks           # lock entry pruned

    resp = await test_client.delete(f"/api/apps/{SLUG}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_snapshot_immutable_after_canvas_resave(
    test_client, app_db, _graphs_dir,
):
    graph = _echo_graph()
    await _save_graph(test_client, graph)
    original_bytes = (_graphs_dir / "pub-src.json").read_text()
    await _publish(test_client, SLUG, "pub-src")

    # Canvas re-save with different content (renamed output).
    graph["nodes"][2]["data"]["params"]["name"] = "renamed"
    await _save_graph(test_client, graph)
    assert (_graphs_dir / "pub-src.json").read_text() != original_bytes

    def _snapshot(conn: sqlite3.Connection) -> str:
        return conn.execute(
            "SELECT graph_json FROM app_versions WHERE version = 1",
        ).fetchone()[0]

    # The stored snapshot is the EXACT pre-edit file bytes (Decision B).
    # Invoke-behavior pinning of the same property lands with the invoke
    # route in PR3 (test_api_apps_invoke.py).
    assert await app_db.run(_snapshot) == original_bytes


@pytest.mark.asyncio
async def test_management_get_routes_reject_anonymous(app_db):
    # The auth_guard GET gap, closed at route level.
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url=f"http://127.0.0.1:{settings.PORT}",
    ) as anon:
        assert (await anon.get("/api/apps")).status_code == 403
        assert (await anon.get("/api/apps/x/versions")).status_code == 403
        assert (await anon.delete("/api/apps/x")).status_code == 403
