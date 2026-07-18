"""Publish provenance (spec 9): migration 002 columns, publish fields, the
in-handler git_commit validator, and the versions + OpenAPI surfaces."""

import pytest

from app.core.db import Database


def _echo_graph(name="echo-graph"):
    return {
        "name": name, "description": "",
        "nodes": [
            {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
             "data": {"params": {}}},
            {"id": "gi", "type": "GraphInput", "position": {"x": 1, "y": 0},
             "data": {"params": {"name": "x", "type": "string",
                                 "required": True, "default": "",
                                 "description": ""}}},
            {"id": "out", "type": "GraphOutput", "position": {"x": 2, "y": 0},
             "data": {"params": {"name": "y", "description": ""}}},
        ],
        "edges": [
            {"id": "t1", "source": "start", "target": "gi",
             "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
            {"id": "d1", "source": "gi", "target": "out",
             "sourceHandle": "value", "targetHandle": "value", "type": "data"},
        ],
    }


@pytest.fixture
def graphs_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    return tmp_path


def _table_columns(db: Database, table: str) -> set[str]:
    rows = db._conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r["name"] for r in rows}


def _user_version(db: Database) -> int:
    return db._conn.execute("PRAGMA user_version").fetchone()[0]


def test_migration_002_columns_and_idempotent(tmp_path):
    db = Database(tmp_path / "p.db")
    db.connect()
    assert _user_version(db) == 2
    cols = _table_columns(db, "app_versions")
    assert "git_commit" in cols and "git_dirty" in cols
    db.close()
    # Re-connecting an already-migrated DB is a no-op (user_version gate).
    db2 = Database(tmp_path / "p.db")
    db2.connect()
    assert _user_version(db2) == 2
    db2.close()


def test_migration_002_upgrades_a_v1_db(tmp_path, monkeypatch):
    from app.core import db as dbmod
    from app.core.migrations import MIGRATION_001, MIGRATION_002
    monkeypatch.setattr(dbmod, "MIGRATIONS", [MIGRATION_001])
    d1 = Database(tmp_path / "u.db")
    d1.connect()
    assert _user_version(d1) == 1
    assert "git_commit" not in _table_columns(d1, "app_versions")

    # Seed a published row under the v1 schema (no git_commit/git_dirty
    # columns exist yet) so the in-place upgrade is exercised against real
    # data, not just an empty schema -- a pre-migration row must survive
    # the ALTER TABLE and read back with the new columns NULL (unknown),
    # never silently defaulted to 0/False/"".
    now = "2026-01-01T00:00:00.000000Z"
    d1._conn.execute(
        "INSERT INTO apps (id, slug, graph_name, active_version, "
        "record_io, created_at, updated_at) "
        "VALUES (1, 'legacy-app', 'g', 1, 1, ?, ?)", (now, now),
    )
    d1._conn.execute(
        "INSERT INTO app_versions (app_id, version, graph_json, "
        "contract_json, source_graph_name, note, created_at) "
        "VALUES (1, 1, '{}', '{}', 'g', 'pre-migration note', ?)", (now,),
    )
    d1.close()

    monkeypatch.setattr(dbmod, "MIGRATIONS", [MIGRATION_001, MIGRATION_002])
    d2 = Database(tmp_path / "u.db")
    d2.connect()
    assert _user_version(d2) == 2
    assert "git_commit" in _table_columns(d2, "app_versions")

    row = d2._conn.execute(
        "SELECT note, source_graph_name, git_commit, git_dirty "
        "FROM app_versions WHERE app_id = 1 AND version = 1",
    ).fetchone()
    assert row["note"] == "pre-migration note"        # old data intact
    assert row["source_graph_name"] == "g"
    assert row["git_commit"] is None                  # NULL = unknown
    assert row["git_dirty"] is None
    d2.close()


async def _save_echo(client):
    r = await client.post("/api/graph/save", json=_echo_graph())
    assert r.status_code == 200, r.text


async def test_publish_records_and_surfaces_provenance(test_client, app_db, graphs_dir):
    await _save_echo(test_client)
    commit = "a" * 40
    r = await test_client.post("/api/apps/svc/publish", json={
        "graph": "echo-graph", "create": True,
        "git_commit": commit, "git_dirty": True})
    assert r.status_code == 200, r.text
    assert r.json()["git_commit"] == commit

    row = (await test_client.get("/api/apps/svc/versions")).json()[0]
    assert row["git_commit"] == commit
    assert row["git_dirty"] is True

    info = (await test_client.get("/api/apps/svc/openapi.json")).json()["info"]
    assert info["x-codefyui-git-commit"] == commit
    assert info["x-codefyui-git-dirty"] is True


async def test_publish_clean_tree_records_dirty_false(test_client, app_db,
                                                      graphs_dir):
    """git_dirty=False is a REAL value, not absence (issue #88): int(False)=0
    must round-trip as false -- never collapse to null -- through the publish
    response, the versions rows, and the OpenAPI info block (the 0-is-not-None
    branch of _openapi_document)."""
    await _save_echo(test_client)
    commit = "f" * 40
    r = await test_client.post("/api/apps/svc4/publish", json={
        "graph": "echo-graph", "create": True,
        "git_commit": commit, "git_dirty": False})
    assert r.status_code == 200, r.text
    assert r.json()["git_dirty"] is False

    row = (await test_client.get("/api/apps/svc4/versions")).json()[0]
    assert row["git_commit"] == commit
    assert row["git_dirty"] is False

    info = (await test_client.get("/api/apps/svc4/openapi.json")).json()["info"]
    assert info["x-codefyui-git-commit"] == commit
    assert info["x-codefyui-git-dirty"] is False


async def test_publish_without_provenance(test_client, app_db, graphs_dir):
    await _save_echo(test_client)
    r = await test_client.post("/api/apps/svc2/publish",
                               json={"graph": "echo-graph", "create": True})
    assert r.status_code == 200
    row = (await test_client.get("/api/apps/svc2/versions")).json()[0]
    assert row["git_commit"] is None
    assert row["git_dirty"] is None
    info = (await test_client.get("/api/apps/svc2/openapi.json")).json()["info"]
    assert "x-codefyui-git-commit" not in info
    assert "x-codefyui-git-dirty" not in info


async def test_publish_invalid_git_commit_422(test_client, app_db, graphs_dir):
    await _save_echo(test_client)
    r = await test_client.post("/api/apps/svc3/publish", json={
        "graph": "echo-graph", "create": True, "git_commit": "NOT-hex-!!"})
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "invalid_git_commit"
