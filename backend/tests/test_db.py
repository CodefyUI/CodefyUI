"""Tests for app.core.db.Database + app.core.migrations (Stage-2 storage).

Pins the Decision-A2 connection contract: check_same_thread=False +
isolation_level=None are MANDATORY (the default isolation level silently
de-atomizes DDL migrations — proven by test during spec review), one
shared asyncio.Lock, explicit BEGIN IMMEDIATE transactions, and the
WAL / busy_timeout / foreign_keys pragmas.
"""

from __future__ import annotations

import logging
import sqlite3

import pytest

from app.core.db import Database, utc_now_iso
from app.core.migrations import MIGRATIONS, iter_statements


def test_connect_migrates_empty_file(tmp_path):
    db = Database(tmp_path / "codefyui.db")
    db.connect()
    try:
        assert db._conn.execute("PRAGMA user_version").fetchone()[0] \
            == len(MIGRATIONS) == 1
        names = {
            r[0] for r in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {"apps", "app_versions", "api_keys", "runs"} <= names
    finally:
        db.close()


def test_connect_applies_pragmas(tmp_path):
    db = Database(tmp_path / "codefyui.db")
    db.connect()
    try:
        assert db._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert db._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert db._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    finally:
        db.close()


def test_reopen_is_idempotent(tmp_path):
    path = tmp_path / "codefyui.db"
    first = Database(path)
    first.connect()
    first.close()
    second = Database(path)
    second.connect()  # nothing left to migrate; must not raise
    try:
        assert second._conn.execute("PRAGMA user_version").fetchone()[0] \
            == len(MIGRATIONS)
    finally:
        second.close()


def test_failed_migration_rolls_back_atomically(tmp_path, monkeypatch):
    # With the DEFAULT isolation level the first CREATE would auto-commit
    # and survive the failure — the mandatory isolation_level=None +
    # explicit BEGIN IMMEDIATE make the whole migration atomic.
    monkeypatch.setattr(
        "app.core.db.MIGRATIONS",
        [
            "CREATE TABLE mig_ok (x INTEGER);\n"
            "CREATE TABLE mig_ok (x INTEGER);\n"  # duplicate -> fails
        ],
    )
    path = tmp_path / "broken.db"
    db = Database(path)
    with pytest.raises(sqlite3.OperationalError):
        db.connect()
    conn = sqlite3.connect(str(path))
    try:
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert "mig_ok" not in tables  # rolled back, never half-applied
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
    finally:
        conn.close()


def test_iter_statements_handles_comments_and_multistatement():
    script = (
        "-- leading comment\n"
        "CREATE TABLE a (x INTEGER);\n"
        "CREATE INDEX idx_a ON a(x);  -- trailing comment\n"
    )
    statements = list(iter_statements(script))
    assert len(statements) == 2
    assert "CREATE TABLE a" in statements[0]
    assert "CREATE INDEX idx_a" in statements[1]


def test_utc_now_iso_is_sortable_utc():
    stamp = utc_now_iso()
    assert stamp.endswith("Z")
    assert len(stamp) == len("2026-07-03T00:00:00.000000Z")
    assert stamp > "2026-01-01T00:00:00.000000Z"  # lexicographic order works


@pytest.mark.asyncio
async def test_run_seam_executes_and_returns(tmp_path):
    db = Database(tmp_path / "codefyui.db")
    db.connect()
    try:
        def _insert(conn: sqlite3.Connection) -> int:
            cur = conn.execute(
                "INSERT INTO api_keys (name, prefix, token_hash, created_at) "
                "VALUES (?, ?, ?, ?)",
                ("t", "cdui_abcdefg", "h" * 64, utc_now_iso()),
            )
            return cur.lastrowid

        key_id = await db.run(_insert)
        assert key_id == 1

        def _count(conn: sqlite3.Connection) -> int:
            return conn.execute("SELECT COUNT(*) FROM api_keys").fetchone()[0]

        assert await db.run(_count) == 1
    finally:
        db.close()


@pytest.mark.asyncio
async def test_run_raises_when_not_connected(tmp_path):
    db = Database(tmp_path / "codefyui.db")
    with pytest.raises(RuntimeError, match="not connected"):
        await db.run(lambda conn: None)


# ── retention ────────────────────────────────────────────────────────────


def _seed_app_and_run(conn: sqlite3.Connection, run_id: str,
                      created_at: str) -> None:
    """FK-satisfying seed rows (direct insert, not the invoke path — the
    NULL api_key_id is allowed by schema, reserved for Stage-3 editor
    invokes)."""
    now = utc_now_iso()
    conn.execute(
        "INSERT OR IGNORE INTO apps (id, slug, graph_name, active_version, "
        "record_io, created_at, updated_at) VALUES (1, 'prune-app', 'g', 1, "
        "1, ?, ?)",
        (now, now),
    )
    conn.execute(
        "INSERT OR IGNORE INTO app_versions (app_id, version, graph_json, "
        "contract_json, source_graph_name, note, created_at) "
        "VALUES (1, 1, '{}', '{}', 'g', NULL, ?)",
        (now,),
    )
    conn.execute(
        "INSERT INTO runs (run_id, app_id, version, api_key_id, status, "
        "node_timings_json, inputs_json, outputs_json, created_at) "
        "VALUES (?, 1, 1, NULL, 'ok', '{}', '{}', '{}', ?)",
        (run_id, created_at),
    )


@pytest.mark.asyncio
async def test_prune_disabled_at_default_zero(tmp_path):
    db = Database(tmp_path / "codefyui.db")
    db.connect()
    try:
        await db.run(lambda conn: _seed_app_and_run(
            conn, "old-run", "2000-01-01T00:00:00.000000Z"))
        assert await db.prune_runs(0, force=True) == 0
        count = await db.run(lambda conn: conn.execute(
            "SELECT COUNT(*) FROM runs").fetchone()[0])
        assert count == 1  # keep forever by default
    finally:
        db.close()


@pytest.mark.asyncio
async def test_prune_deletes_only_older_and_logs_loudly(tmp_path, caplog):
    db = Database(tmp_path / "codefyui.db")
    db.connect()
    try:
        def _seed(conn: sqlite3.Connection) -> None:
            _seed_app_and_run(conn, "ancient", "2000-01-01T00:00:00.000000Z")
            _seed_app_and_run(conn, "fresh", utc_now_iso())

        await db.run(_seed)
        with caplog.at_level(logging.WARNING, logger="app.core.db"):
            pruned = await db.prune_runs(30, force=True)
        assert pruned == 1
        assert "pruned 1 runs older than 30d" in caplog.text
        assert "CODEFYUI_RUNS_RETENTION_DAYS=0" in caplog.text
        remaining = await db.run(lambda conn: [r[0] for r in conn.execute(
            "SELECT run_id FROM runs").fetchall()])
        assert remaining == ["fresh"]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_prune_piggyback_rate_limited_to_hourly(tmp_path):
    db = Database(tmp_path / "codefyui.db")
    db.connect()
    try:
        await db.run(lambda conn: _seed_app_and_run(
            conn, "ancient", "2000-01-01T00:00:00.000000Z"))
        assert await db.prune_runs(30) == 1     # first non-forced call prunes
        await db.run(lambda conn: _seed_app_and_run(
            conn, "ancient-2", "2000-01-01T00:00:00.000000Z"))
        assert await db.prune_runs(30) == 0     # within the hour -> no-op
        assert await db.prune_runs(30, force=True) == 1  # force bypasses
    finally:
        db.close()
