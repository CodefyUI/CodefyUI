"""Tests for app.core.db.Database + app.core.migrations (Stage-2 storage).

Pins the Decision-A2 connection contract: check_same_thread=False +
isolation_level=None are MANDATORY (the default isolation level silently
de-atomizes DDL migrations — proven by test during spec review), one
shared asyncio.Lock, explicit BEGIN IMMEDIATE transactions, and the
WAL / busy_timeout / foreign_keys pragmas.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import sqlite3
import threading

import pytest

from app.core.db import Database, transaction, utc_now_iso
from app.core.migrations import MIGRATIONS, iter_statements


def test_connect_migrates_empty_file(tmp_path):
    db = Database(tmp_path / "codefyui.db")
    db.connect()
    try:
        assert db._conn.execute("PRAGMA user_version").fetchone()[0] \
            == len(MIGRATIONS) == 4
        names = {
            r[0] for r in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        # Publish (001/002), the Run Service store (003) and sweeps (004).
        # The literal count above is a deliberate tripwire: adding a
        # migration should make someone extend this set too. Per-table
        # schema assertions live with each subsystem (test_provenance /
        # test_run_store).
        assert {"apps", "app_versions", "api_keys", "runs"} <= names
        assert {"exec_runs", "exec_run_metrics", "exec_run_events",
                "exec_run_artifacts", "sweeps"} <= names
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


# ── transaction() ────────────────────────────────────────────────────────


def test_transaction_commits_on_success(tmp_path):
    db = Database(tmp_path / "codefyui.db")
    db.connect()
    try:
        with transaction(db._conn) as conn:
            conn.execute(
                "INSERT INTO api_keys (name, prefix, token_hash, created_at) "
                "VALUES ('t', 'cdui_a', ?, ?)", ("h" * 64, utc_now_iso()))
        assert db._conn.in_transaction is False   # no dangling transaction
        assert db._conn.execute(
            "SELECT COUNT(*) FROM api_keys").fetchone()[0] == 1
    finally:
        db.close()


def test_transaction_rolls_back_every_statement_on_failure(tmp_path):
    # The multi-statement guarantee: the first INSERT must not survive a
    # failure in the second (this is exactly what RunStore's batched metric
    # flush and cursor allocation rely on).
    db = Database(tmp_path / "codefyui.db")
    db.connect()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            with transaction(db._conn) as conn:
                conn.execute(
                    "INSERT INTO api_keys (name, prefix, token_hash, "
                    "created_at) VALUES ('a', 'p', ?, ?)",
                    ("h" * 64, utc_now_iso()))
                conn.execute(
                    "INSERT INTO api_keys (name, prefix, token_hash, "
                    "created_at) VALUES ('b', 'p', ?, ?)",   # duplicate hash
                    ("h" * 64, utc_now_iso()))
        assert db._conn.in_transaction is False   # no dangling transaction
        assert db._conn.execute(
            "SELECT COUNT(*) FROM api_keys").fetchone()[0] == 0
    finally:
        db.close()


def test_transaction_takes_the_write_lock_up_front(tmp_path):
    # IMMEDIATE, not DEFERRED: the lock must be held before the block's
    # first read, which is what makes a read-then-write cursor allocation
    # safe against a second connection.
    db = Database(tmp_path / "codefyui.db")
    db.connect()
    other = sqlite3.connect(str(tmp_path / "codefyui.db"),
                            isolation_level=None)
    other.execute("PRAGMA busy_timeout=0")
    try:
        with transaction(db._conn) as conn:
            conn.execute("SELECT COUNT(*) FROM api_keys").fetchone()
            with pytest.raises(sqlite3.OperationalError, match="locked|busy"):
                other.execute("BEGIN IMMEDIATE")
    finally:
        other.close()
        db.close()


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


def test_iter_statements_semicolon_inside_comment_does_not_split():
    # A ';' inside a `--` comment must not be mistaken for a statement
    # terminator (sqlite3.complete_statement is comment-aware) — the
    # leading comment stays attached to the CREATE TABLE it precedes.
    script = (
        "-- this comment has a ; semicolon that must not split\n"
        "CREATE TABLE a (x INTEGER);\n"
        "CREATE INDEX idx_a ON a(x);\n"
    )
    statements = list(iter_statements(script))
    assert len(statements) == 2
    assert "CREATE TABLE a" in statements[0]
    assert "CREATE INDEX idx_a" in statements[1]


def test_iter_statements_ignores_trailing_comment_only_tail():
    # A dangling trailing comment with no SQL after it (and no terminating
    # semicolon) must never be yielded as a bogus pseudo-statement.
    script = (
        "CREATE TABLE a (x INTEGER);\n"
        "-- dangling trailing comment, no SQL after this\n"
    )
    statements = list(iter_statements(script))
    assert len(statements) == 1
    assert "CREATE TABLE a" in statements[0]


def test_iter_statements_ignores_trailing_block_comment_tail():
    # Same guard for SQL's other comment syntax: a trailing /* ... */
    # block (terminated or dangling unterminated) is not a statement.
    for tail in (
        "/* terminated block comment; with a semicolon inside */\n",
        "/* dangling unterminated block comment\n   spanning lines\n",
        "-- line comment\n/* then a block */  \n",
    ):
        script = "CREATE TABLE a (x INTEGER);\n" + tail
        statements = list(iter_statements(script))
        assert len(statements) == 1, tail
        assert "CREATE TABLE a" in statements[0]


def test_iter_statements_yields_sql_after_block_comment_in_tail():
    # Conservative direction of the same guard: real SQL hiding after a
    # block comment in the unterminated tail must still be yielded (and
    # then fail loudly downstream if malformed) — never silently dropped.
    script = (
        "CREATE TABLE a (x INTEGER);\n"
        "/* comment */ CREATE TABLE b (y INTEGER)\n"  # no semicolon
    )
    statements = list(iter_statements(script))
    assert len(statements) == 2
    assert "CREATE TABLE b" in statements[1]


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
async def test_cancelling_run_waits_for_the_worker_thread(tmp_path):
    # A cancelled db.run must NOT release the lock while its fn is still
    # executing: the abandoned thread would keep driving BEGIN/COMMIT on the
    # SHARED connection, and the next caller's write would be swallowed by
    # someone else's ROLLBACK, swept into their COMMIT, or refused with
    # "cannot start a transaction within a transaction".
    db = Database(tmp_path / "codefyui.db")
    db.connect()
    inside = threading.Event()
    release = threading.Event()

    def _slow(conn: sqlite3.Connection) -> None:
        with transaction(conn):
            conn.execute(
                "INSERT INTO api_keys (name, prefix, token_hash, created_at) "
                "VALUES ('slow', 'p', ?, ?)", ("a" * 64, utc_now_iso()))
            inside.set()
            release.wait(10)

    try:
        task = asyncio.create_task(db.run(_slow))
        await asyncio.to_thread(inside.wait, 10)   # thread is mid-transaction
        task.cancel()
        await asyncio.sleep(0.05)                  # let cancellation settle
        assert db._lock.locked()                   # lock NOT handed over yet
        assert not task.done()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert db._conn.in_transaction is False

        # The next caller gets a quiet connection and its write survives.
        def _next(conn: sqlite3.Connection) -> int:
            with transaction(conn):
                conn.execute(
                    "INSERT INTO api_keys (name, prefix, token_hash, "
                    "created_at) VALUES ('next', 'p', ?, ?)",
                    ("b" * 64, utc_now_iso()))
            return conn.execute(
                "SELECT COUNT(*) FROM api_keys").fetchone()[0]

        assert await db.run(_next) == 2
    finally:
        release.set()
        db.close()


@pytest.mark.asyncio
async def test_shutdown_sweep_cannot_cancel_the_worker(tmp_path):
    # asyncio's teardown cancels EVERY entry of all_tasks(). If the worker
    # were a Task (what ensure_future(to_thread(...)) builds) the sweep
    # would cancel it directly: done() flips True while the thread is still
    # mid-transaction and the lock goes early -- the same bug as above
    # through a second door. run_in_executor hands back a Future, which the
    # sweep cannot see.
    db = Database(tmp_path / "codefyui.db")
    db.connect()
    inside = threading.Event()
    release = threading.Event()

    def _slow(conn: sqlite3.Connection) -> None:
        with transaction(conn):
            conn.execute(
                "INSERT INTO api_keys (name, prefix, token_hash, created_at) "
                "VALUES ('slow', 'p', ?, ?)", ("a" * 64, utc_now_iso()))
            inside.set()
            release.wait(10)

    try:
        task = asyncio.create_task(db.run(_slow))
        await asyncio.to_thread(inside.wait, 10)

        current = asyncio.current_task()
        for pending in asyncio.all_tasks():
            if pending is not current:
                pending.cancel()
        await asyncio.sleep(0.05)
        assert db._lock.locked()          # connection still private
        assert db._conn.in_transaction    # ...and the thread still owns it

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert db._conn.in_transaction is False
        assert await db.run(lambda c: c.execute(
            "SELECT COUNT(*) FROM api_keys").fetchone()[0]) == 1
    finally:
        release.set()
        db.close()


@pytest.mark.asyncio
async def test_cancelled_run_does_not_leak_an_unretrieved_exception(
        tmp_path, caplog):
    # End-to-end guard on the fn-raises-AND-caller-cancelled path: the lock
    # comes back, CancelledError wins over the worker's error, and asyncio
    # logs nothing. Honest about its own reach: this passes with or without
    # Database.run's explicit worker.exception(), because shield's
    # _inner_done_callback already retrieves the inner exception when the
    # outer was cancelled. It is the OBSERVABLE that is pinned here (no
    # "never retrieved" noise, whatever the mechanism), so a refactor that
    # drops the shield and forgets the retrieval is caught.
    db = Database(tmp_path / "codefyui.db")
    db.connect()
    inside = threading.Event()
    release = threading.Event()

    def _boom(conn: sqlite3.Connection) -> None:
        inside.set()
        release.wait(10)
        raise sqlite3.OperationalError("worker blew up")

    try:
        with caplog.at_level(logging.ERROR, logger="asyncio"):
            task = asyncio.create_task(db.run(_boom))
            await asyncio.to_thread(inside.wait, 10)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert db._lock.locked() is False
            del task
            gc.collect()
            await asyncio.sleep(0)      # let any handler callback land
        assert "never retrieved" not in caplog.text
    finally:
        release.set()
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
