"""Tests for app.core.run_store + migration 003 (Run Service storage, #119).

Four concerns, in order:

1. Migration 003 applies cleanly on a FRESH database and as an in-place
   UPGRADE of the shipped v2 schema, without touching the publish
   subsystem's own (unrelated) ``runs`` table.
2. Typed CRUD round-trips for runs / metrics / events / artifacts.
3. THE hard requirement: ``append_event`` hands out a per-run cursor that
   is gapless AND monotonic under concurrent asyncio writers, and every
   caller gets back the cursor its own event actually landed on.
4. Retention (``prune``/``delete_run``) and provenance capture.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from app.core.db import Database
from app.core.migrations import MIGRATION_001, MIGRATION_002, MIGRATIONS
from app.core.run_store import (
    ACTIVE_STATUSES,
    ARTIFACT_KIND_CHECKPOINT,
    ARTIFACT_KIND_EXPORT,
    ARTIFACT_KIND_IMAGE,
    RUN_STATUSES,
    TERMINAL_STATUSES,
    MetricPoint,
    RunProvenance,
    RunRecord,
    RunStore,
)

EXEC_TABLES = {
    "exec_runs",
    "exec_run_metrics",
    "exec_run_events",
    "exec_run_artifacts",
}

GRAPH = {
    "name": "demo",
    "nodes": [
        {"id": "a", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {"params": {}}},
    ],
    "edges": [],
}
OPTIONS = {"device": "cpu", "seed": 7, "record_outputs": True,
           "lane": "queued"}


# ── fixtures / helpers ────────────────────────────────────────────────────


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
def store(db):
    return RunStore(db)


def _tables(database: Database) -> set[str]:
    return {r[0] for r in database._conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()}


def _index_names(database: Database) -> set[str]:
    return {r[0] for r in database._conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index' "
        "AND name IS NOT NULL").fetchall()}


def _columns(database: Database, table: str) -> list[str]:
    return [r["name"] for r in database._conn.execute(
        f"PRAGMA table_info({table})").fetchall()]


def _user_version(database: Database) -> int:
    return database._conn.execute("PRAGMA user_version").fetchone()[0]


def _indexed_columns(database: Database, table: str) -> set[tuple[str, ...]]:
    """Column tuples of every index on *table* (name-independent shape check)."""
    out: set[tuple[str, ...]] = set()
    for row in database._conn.execute(f"PRAGMA index_list({table})").fetchall():
        cols = database._conn.execute(
            f"PRAGMA index_info({row['name']})").fetchall()
        out.add(tuple(c["name"] for c in cols))
    return out


async def _make_run(store: RunStore, **kwargs) -> RunRecord:
    kwargs.setdefault("graph_snapshot", GRAPH)
    kwargs.setdefault("options", OPTIONS)
    kwargs.setdefault("provenance", RunProvenance())
    return await store.create_run(**kwargs)


# ── 1. migrations ─────────────────────────────────────────────────────────


def test_fresh_install_creates_the_four_exec_tables(db):
    assert _user_version(db) == len(MIGRATIONS) == 3
    assert EXEC_TABLES <= _tables(db)


def test_fresh_install_keeps_the_publish_tables_untouched(db):
    # The naming decision under test: `exec_runs` is a NEW namespace, so the
    # Stage-2 publish tables -- including its own, unrelated `runs` table --
    # must survive migration 003 byte-for-byte.
    assert {"apps", "app_versions", "api_keys", "runs"} <= _tables(db)
    assert _columns(db, "runs") == [
        "run_id", "app_id", "version", "api_key_id", "status", "error_code",
        "error_message", "error_node_id", "device", "total_s",
        "node_timings_json", "inputs_json", "outputs_json", "created_at",
    ]


def test_exec_runs_has_the_full_column_list(db):
    assert _columns(db, "exec_runs") == [
        "id", "name", "graph_snapshot", "options", "status", "error",
        "queue_key", "created_at", "started_at", "finished_at",
        "git_commit", "git_dirty", "plugin_pins",
    ]


def test_child_tables_have_the_specified_columns(db):
    assert _columns(db, "exec_run_metrics") == [
        "id", "run_id", "node_id", "name", "step", "value", "ts"]
    assert _columns(db, "exec_run_events") == [
        "id", "run_id", "cursor", "type", "payload", "ts"]
    assert _columns(db, "exec_run_artifacts") == [
        "id", "run_id", "kind", "path", "meta", "created_at"]


def test_required_indexes_exist(db):
    assert ("run_id", "name", "step") in _indexed_columns(db, "exec_run_metrics")
    assert ("run_id", "cursor") in _indexed_columns(db, "exec_run_events")
    assert ("run_id",) in {t[:1] for t in _indexed_columns(db, "exec_run_artifacts")}
    names = _index_names(db)
    assert "idx_exec_runs_created" in names
    assert "idx_exec_runs_status_created" in names


def test_upgrade_from_shipped_v2_db_preserves_publish_data(tmp_path, monkeypatch):
    """In-place upgrade of a real v2 DB with rows in it (test_provenance.py
    pattern): user_version 2 -> 3, new tables appear, old rows survive."""
    from app.core import db as dbmod

    monkeypatch.setattr(dbmod, "MIGRATIONS", [MIGRATION_001, MIGRATION_002])
    old = Database(tmp_path / "u.db")
    old.connect()
    assert _user_version(old) == 2
    assert not (EXEC_TABLES & _tables(old))
    now = "2026-01-01T00:00:00.000000Z"
    old._conn.execute(
        "INSERT INTO apps (id, slug, graph_name, active_version, record_io, "
        "created_at, updated_at) VALUES (1, 'legacy', 'g', 1, 1, ?, ?)",
        (now, now),
    )
    old._conn.execute(
        "INSERT INTO runs (run_id, app_id, version, api_key_id, status, "
        "created_at) VALUES ('legacy-invoke', 1, 1, NULL, 'ok', ?)", (now,),
    )
    old.close()

    monkeypatch.setattr(dbmod, "MIGRATIONS", list(MIGRATIONS))
    new = Database(tmp_path / "u.db")
    new.connect()
    try:
        assert _user_version(new) == 3
        assert EXEC_TABLES <= _tables(new)
        row = new._conn.execute(
            "SELECT status, created_at FROM runs WHERE run_id = 'legacy-invoke'"
        ).fetchone()
        assert row["status"] == "ok"            # publish history intact
        assert row["created_at"] == now
        assert new._conn.execute(
            "SELECT COUNT(*) FROM exec_runs").fetchone()[0] == 0
    finally:
        new.close()


def test_reconnecting_a_v3_db_is_a_noop(tmp_path):
    first = Database(tmp_path / "idem.db")
    first.connect()
    first.close()
    second = Database(tmp_path / "idem.db")
    second.connect()          # must not raise "table already exists"
    try:
        assert _user_version(second) == len(MIGRATIONS)
        assert EXEC_TABLES <= _tables(second)
    finally:
        second.close()


def test_the_run_list_order_is_served_by_an_index(db):
    """Guards the ASC-index choice in migration 003. `created_at DESC,
    rowid DESC` is satisfied by scanning an ASC index backwards; flipping
    the index to DESC silently degrades every Runs-panel poll to a full
    scan plus a sort, which no functional test would notice."""
    plan = " ".join(r["detail"] for r in db._conn.execute(
        "EXPLAIN QUERY PLAN SELECT id FROM exec_runs "
        "ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?", (1, 0),
    ).fetchall())
    assert "idx_exec_runs_created" in plan
    assert "TEMP B-TREE" not in plan


def test_status_filtered_list_order_is_index_served_for_one_status(db):
    """...and honestly is NOT for several.

    A multi-value IN merges two index ranges, which cannot preserve a
    global created_at order, so sqlite sorts. Accepted: that query exists
    to find the handful of runs still in flight. Asserted rather than left
    implicit so the ASC-index rationale does not overclaim.
    """
    def _plan(marks: str, params: tuple) -> str:
        return " ".join(r["detail"] for r in db._conn.execute(
            f"EXPLAIN QUERY PLAN SELECT id FROM exec_runs WHERE status IN "
            f"({marks}) ORDER BY created_at DESC, rowid DESC LIMIT ?", params,
        ).fetchall())

    one = _plan("?", ("queued", 1))
    assert "idx_exec_runs_status_created" in one
    assert "TEMP B-TREE" not in one

    two = _plan("?,?", ("queued", "running", 1))
    assert "idx_exec_runs_status_created" in two
    assert "TEMP B-TREE" in two          # documented, accepted trade-off


def test_event_replay_and_cursor_allocation_are_index_served(db):
    for sql, params in (
        ("SELECT cursor FROM exec_run_events WHERE run_id = ? AND cursor > ? "
         "ORDER BY cursor", ("r", 0)),
        ("SELECT COALESCE(MAX(cursor), 0) + 1 FROM exec_run_events "
         "WHERE run_id = ?", ("r",)),
    ):
        plan = " ".join(r["detail"] for r in db._conn.execute(
            "EXPLAIN QUERY PLAN " + sql, params).fetchall())
        assert "SCAN exec_run_events" not in plan   # never a full scan
        assert "TEMP B-TREE" not in plan


def test_metric_reads_are_index_served(db):
    """The chart queries, in the exact form get_metrics/list_metric_names
    emit. `ORDER BY name, step, id` is chosen to match the index; a future
    reorder to `ts` or `step, name` would still be correct but would sort
    the whole series on every poll of a live run."""
    for sql, params in (
        ("SELECT run_id, node_id, name, step, value, ts FROM exec_run_metrics "
         "WHERE run_id = ? AND name = ? ORDER BY name, step, id", ("r", "a")),
        ("SELECT DISTINCT name FROM exec_run_metrics WHERE run_id = ? "
         "ORDER BY name", ("r",)),
    ):
        plan = " ".join(r["detail"] for r in db._conn.execute(
            "EXPLAIN QUERY PLAN " + sql, params).fetchall())
        assert "idx_exec_run_metrics_series" in plan
        assert "SCAN exec_run_metrics" not in plan
        assert "TEMP B-TREE" not in plan


def test_status_vocabulary_is_not_pinned_in_sql(db):
    """No CHECK constraint on `status`: SQLite cannot ALTER one away, and
    the queue/sweep issues may add lifecycle states. Enforcement is the
    DAO's frozenset instead (asserted by the create/finish tests)."""
    ddl = db._conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'exec_runs'").fetchone()[0]
    assert "CHECK" not in ddl.upper()


def test_schema_stays_additive_for_the_sweep_column(db):
    """#140 will add `sweep_id` -- prove a plain ALTER TABLE ADD COLUMN is
    enough (no table rebuild, no data migration)."""
    db._conn.execute("ALTER TABLE exec_runs ADD COLUMN sweep_id TEXT")
    assert "sweep_id" in _columns(db, "exec_runs")


# ── 2. run CRUD ───────────────────────────────────────────────────────────


async def test_create_and_get_run_round_trip(store):
    created = await _make_run(store, name="mnist", queue_key="cpu")
    assert created.status == "queued"
    assert created.name == "mnist"
    assert created.queue_key == "cpu"
    assert created.options == OPTIONS
    assert created.started_at is None and created.finished_at is None
    assert created.error is None
    assert len(created.id) == 32           # uuid4().hex, like publish runs

    fetched = await store.get_run(created.id)
    assert fetched == created


async def test_graph_snapshot_is_fetched_separately_from_the_row(store):
    # RunRecord deliberately omits the (potentially megabyte) snapshot so
    # list_runs() stays cheap for the Runs panel's polling.
    run = await _make_run(store)
    assert not hasattr(run, "graph_snapshot")
    assert await store.get_graph_snapshot(run.id) == GRAPH
    assert await store.get_graph_snapshot("nope") is None


async def test_get_run_returns_none_for_unknown_id(store):
    assert await store.get_run("does-not-exist") is None


async def test_create_run_defaults_options_to_empty_dict(store):
    run = await store.create_run(graph_snapshot=GRAPH,
                                 provenance=RunProvenance())
    assert run.options == {}
    assert (await store.get_run(run.id)).options == {}


async def test_create_run_accepts_an_explicit_run_id(store):
    run = await _make_run(store, run_id="fixed-id")
    assert run.id == "fixed-id"
    assert (await store.get_run("fixed-id")) == run


async def test_create_run_rejects_an_unknown_status(store):
    with pytest.raises(ValueError, match="status"):
        await _make_run(store, status="wat")


async def test_create_run_accepts_an_already_running_status(store):
    # A caller that starts work immediately should not have to insert
    # `queued` first and correct it a microsecond later.
    run = await _make_run(store, status="running")
    assert (await store.get_run(run.id)).status == "running"


async def test_duplicate_run_id_is_rejected_loudly(store):
    await _make_run(store, run_id="dup")
    with pytest.raises(sqlite3.IntegrityError):
        await _make_run(store, run_id="dup")


async def test_non_ascii_graph_and_options_survive_the_round_trip(store):
    graph = {"name": "中文圖", "nodes": [], "edges": []}
    run = await store.create_run(
        graph_snapshot=graph, options={"note": "訓練"},
        name="實驗 1", provenance=RunProvenance())
    assert await store.get_graph_snapshot(run.id) == graph
    reread = await store.get_run(run.id)
    assert reread.name == "實驗 1"
    assert reread.options == {"note": "訓練"}


async def test_list_runs_is_newest_first_and_filters_by_status(store):
    a = await _make_run(store, name="a")
    b = await _make_run(store, name="b")
    c = await _make_run(store, name="c")
    await store.mark_running(b.id)
    await store.mark_finished(c.id, "succeeded")

    assert [r.id for r in await store.list_runs()] == [c.id, b.id, a.id]
    assert [r.id for r in await store.list_runs(status="running")] == [b.id]
    assert [r.id for r in await store.list_runs(
        status=["queued", "running"])] == [b.id, a.id]
    assert [r.id for r in await store.list_runs(limit=2)] == [c.id, b.id]
    assert [r.id for r in await store.list_runs(limit=2, offset=2)] == [a.id]


async def test_list_runs_rejects_an_unknown_status_filter(store):
    with pytest.raises(ValueError, match="status"):
        await store.list_runs(status="finished")


async def test_list_runs_with_an_empty_status_filter_returns_nothing(store):
    # `IN ()` is a syntax error, so the empty filter must short-circuit
    # rather than reach sqlite.
    await _make_run(store)
    assert await store.list_runs(status=[]) == []


async def test_list_runs_ties_break_on_insertion_order(store, monkeypatch):
    # Same created_at for every row -> ordering must still be deterministic
    # (newest inserted first), never sqlite's arbitrary scan order.
    monkeypatch.setattr("app.core.run_store.utc_now_iso",
                        lambda: "2026-01-01T00:00:00.000000Z")
    ids = [(await _make_run(store)).id for _ in range(5)]
    assert [r.id for r in await store.list_runs()] == list(reversed(ids))


async def test_mark_running_sets_status_and_started_at(store):
    run = await _make_run(store)
    assert await store.mark_running(run.id, queue_key="cuda:0") is True
    updated = await store.get_run(run.id)
    assert updated.status == "running"
    assert updated.started_at is not None
    assert updated.queue_key == "cuda:0"
    assert updated.finished_at is None


async def test_mark_running_keeps_the_first_started_at(store):
    run = await _make_run(store)
    await store.mark_running(run.id)
    first = (await store.get_run(run.id)).started_at
    await store.mark_running(run.id)
    assert (await store.get_run(run.id)).started_at == first


async def test_mark_finished_records_terminal_status_and_error(store):
    run = await _make_run(store)
    await store.mark_running(run.id)
    assert await store.mark_finished(run.id, "failed", error="boom") is True
    done = await store.get_run(run.id)
    assert done.status == "failed"
    assert done.error == "boom"
    assert done.finished_at is not None


async def test_cancelling_a_queued_run_leaves_started_at_null(store):
    # Dequeue-cancel: the run never ran, so it must not claim a start time.
    run = await _make_run(store)
    await store.mark_finished(run.id, "cancelled")
    cancelled = await store.get_run(run.id)
    assert cancelled.started_at is None
    assert cancelled.finished_at is not None


async def test_a_late_cancel_cannot_rewrite_a_finished_run(store):
    """The Stop-button race: the user hits cancel as the last epoch lands.

    Without a precondition the late writer wins and a run that actually
    succeeded is filed forever as cancelled -- with a finished_at from the
    wrong moment.
    """
    run = await _make_run(store)
    await store.mark_running(run.id)
    assert await store.mark_finished(run.id, "succeeded") is True
    settled = await store.get_run(run.id)

    assert await store.mark_finished(run.id, "cancelled") is False
    unchanged = await store.get_run(run.id)
    assert unchanged.status == "succeeded"
    assert unchanged.finished_at == settled.finished_at
    assert unchanged.error is None


async def test_mark_finished_can_be_forced_with_an_explicit_expected(store):
    run = await _make_run(store)
    await store.mark_finished(run.id, "succeeded")
    assert await store.mark_finished(
        run.id, "failed", error="found later",
        expected=["succeeded"]) is True
    fixed = await store.get_run(run.id)
    assert (fixed.status, fixed.error) == ("failed", "found later")

    assert await store.mark_finished(run.id, "failed", expected=[]) is False
    with pytest.raises(ValueError, match="status"):
        await store.mark_finished(run.id, "failed", expected=["nope"])


async def test_mark_finished_rejects_a_non_terminal_status(store):
    run = await _make_run(store)
    with pytest.raises(ValueError, match="terminal"):
        await store.mark_finished(run.id, "running")


async def test_mark_helpers_return_false_for_unknown_runs(store):
    assert await store.mark_running("ghost") is False
    assert await store.mark_finished("ghost", "succeeded") is False


async def test_interrupt_active_runs_clears_zombie_states(store):
    queued = await _make_run(store)
    running = await _make_run(store)
    finished = await _make_run(store)
    await store.mark_running(running.id)
    await store.mark_finished(finished.id, "succeeded")

    assert await store.interrupt_active_runs() == 2
    assert (await store.get_run(running.id)).status == "interrupted"
    assert (await store.get_run(queued.id)).status == "interrupted"
    assert (await store.get_run(finished.id)).status == "succeeded"
    assert (await store.get_run(running.id)).finished_at is not None
    assert await store.interrupt_active_runs() == 0   # idempotent


async def test_interrupt_active_runs_can_be_narrowed_to_running(store):
    queued = await _make_run(store)
    running = await _make_run(store)
    await store.mark_running(running.id)
    assert await store.interrupt_active_runs(statuses=["running"]) == 1
    assert (await store.get_run(queued.id)).status == "queued"
    assert await store.interrupt_active_runs(statuses=[]) == 0
    with pytest.raises(ValueError, match="status"):
        await store.interrupt_active_runs(statuses=["nope"])


async def test_interrupt_active_runs_refuses_to_rewrite_finished_history(store):
    # It rewrites status AND finished_at, so a terminal status must not be
    # reachable through the statuses= escape hatch.
    done = await _make_run(store)
    await store.mark_finished(done.id, "succeeded")
    with pytest.raises(ValueError, match="not an active status"):
        await store.interrupt_active_runs(statuses=["succeeded"])
    assert (await store.get_run(done.id)).status == "succeeded"


async def test_count_runs_ignores_paging(store):
    for _ in range(3):
        await _make_run(store)
    running = await _make_run(store)
    await store.mark_running(running.id)

    assert await store.count_runs() == 4
    assert await store.count_runs(status="queued") == 3
    assert await store.count_runs(status=["queued", "running"]) == 4
    assert await store.count_runs(status=[]) == 0
    assert len(await store.list_runs(limit=2)) == 2      # paging unaffected
    with pytest.raises(ValueError, match="status"):
        await store.count_runs(status="bogus")


def test_status_constants_partition_the_vocabulary():
    assert ACTIVE_STATUSES | TERMINAL_STATUSES == RUN_STATUSES
    assert not (ACTIVE_STATUSES & TERMINAL_STATUSES)
    assert RUN_STATUSES == {"queued", "running", "succeeded", "failed",
                            "cancelled", "interrupted"}


# ── 3. events: the gapless cursor ─────────────────────────────────────────


async def test_append_event_returns_a_one_based_cursor(store):
    run = await _make_run(store)
    assert await store.append_event(run.id, "execution_start") == 1
    assert await store.append_event(run.id, "node_status", {"n": "a"}) == 2


async def test_cursors_are_gapless_and_monotonic_under_concurrent_writers(store):
    """THE acceptance criterion. 100 coroutines append at once; every
    caller must get a distinct cursor, the set must be exactly 1..100,
    and the payload stored at cursor C must be the one whose caller was
    handed C (no cross-assignment)."""
    run = await _make_run(store)
    cursors = await asyncio.gather(*(
        store.append_event(run.id, "log", {"i": i}) for i in range(100)
    ))
    assert sorted(cursors) == list(range(1, 101))

    events = await store.get_events(run.id)
    assert [e.cursor for e in events] == list(range(1, 101))
    by_cursor = {e.cursor: e.payload["i"] for e in events}
    for i, cursor in enumerate(cursors):
        assert by_cursor[cursor] == i


async def test_append_event_reads_and_writes_in_one_immediate_transaction(
        store, db):
    """Layer 2 of the cursor argument, which the gather test cannot see.

    The gather test is serialised by Database.run's lock before any SQL
    runs, so it passes even with no transaction at all. What actually
    protects the read-modify-write from a SECOND connection is that the
    SELECT and the INSERT sit inside one BEGIN IMMEDIATE -- assert exactly
    that statement order.
    """
    run = await _make_run(store)
    statements: list[str] = []
    db._conn.set_trace_callback(statements.append)
    try:
        await store.append_event(run.id, "log", {"i": 1})
    finally:
        db._conn.set_trace_callback(None)

    trimmed = [" ".join(s.split()).upper() for s in statements]
    assert trimmed[0].startswith("BEGIN IMMEDIATE")   # not DEFERRED, not absent
    assert trimmed[-1].startswith("COMMIT")
    body = " ".join(trimmed)
    assert body.index("MAX(CURSOR)") < body.index("INSERT INTO")


async def test_cursors_are_independent_per_run(store):
    a = await _make_run(store)
    b = await _make_run(store)
    interleaved = await asyncio.gather(*(
        store.append_event(run.id, "log", {"i": i})
        for i in range(20) for run in (a, b)
    ))
    assert sorted(interleaved) == sorted(list(range(1, 21)) * 2)
    assert await store.latest_cursor(a.id) == 20
    assert await store.latest_cursor(b.id) == 20


async def test_latest_cursor_is_zero_before_any_event(store):
    run = await _make_run(store)
    assert await store.latest_cursor(run.id) == 0
    assert await store.latest_cursor("ghost") == 0


async def test_get_events_replays_after_a_cursor_without_gaps_or_dupes(store):
    run = await _make_run(store)
    for i in range(10):
        await store.append_event(run.id, "log", {"i": i})
    tail = await store.get_events(run.id, after_cursor=6)
    assert [e.cursor for e in tail] == [7, 8, 9, 10]
    assert [e.payload["i"] for e in tail] == [6, 7, 8, 9]
    assert await store.get_events(run.id, after_cursor=10) == []

    page = await store.get_events(run.id, after_cursor=0, limit=3)
    assert [e.cursor for e in page] == [1, 2, 3]


async def test_event_record_carries_type_payload_and_timestamp(store):
    run = await _make_run(store)
    await store.append_event(run.id, "node_status",
                             {"node_id": "a", "status": "running"},
                             ts="2026-01-01T00:00:00.000000Z")
    event = (await store.get_events(run.id))[0]
    assert event.run_id == run.id
    assert event.type == "node_status"
    assert event.payload == {"node_id": "a", "status": "running"}
    assert event.ts == "2026-01-01T00:00:00.000000Z"


async def test_event_payload_may_be_omitted(store):
    run = await _make_run(store)
    await store.append_event(run.id, "execution_complete")
    assert (await store.get_events(run.id))[0].payload is None


async def test_append_event_for_an_unknown_run_is_rejected(store):
    with pytest.raises(sqlite3.IntegrityError):
        await store.append_event("ghost", "log")


def test_duplicate_cursor_is_rejected_by_the_unique_constraint(db):
    """Backstop for the serialized-writer argument: even if a second
    connection ever bypassed Database.run's lock, a lost update would
    raise instead of silently duplicating a cursor."""
    now = "2026-01-01T00:00:00.000000Z"
    db._conn.execute(
        "INSERT INTO exec_runs (id, graph_snapshot, options, status, "
        "created_at) VALUES ('r', '{}', '{}', 'running', ?)", (now,))
    db._conn.execute(
        "INSERT INTO exec_run_events (run_id, cursor, type, ts) "
        "VALUES ('r', 1, 'log', ?)", (now,))
    with pytest.raises(sqlite3.IntegrityError):
        db._conn.execute(
            "INSERT INTO exec_run_events (run_id, cursor, type, ts) "
            "VALUES ('r', 1, 'log', ?)", (now,))


# ── 4. metrics ────────────────────────────────────────────────────────────


async def test_log_metric_round_trip(store):
    run = await _make_run(store)
    await store.log_metric(run.id, "train_loss", 0.5, 1, node_id="loop")
    metric = (await store.get_metrics(run.id))[0]
    assert metric.run_id == run.id
    assert metric.name == "train_loss"
    assert metric.value == pytest.approx(0.5)
    assert metric.step == 1
    assert metric.node_id == "loop"
    assert metric.ts


async def test_get_metrics_filters_by_name_and_orders_by_step(store):
    run = await _make_run(store)
    for step in (3, 1, 2):
        await store.log_metric(run.id, "train_loss", step / 10, step)
        await store.log_metric(run.id, "val_loss", step, step)
    series = await store.get_metrics(run.id, name="train_loss")
    assert [m.step for m in series] == [1, 2, 3]
    assert [m.value for m in series] == pytest.approx([0.1, 0.2, 0.3])
    assert len(await store.get_metrics(run.id)) == 6
    assert [m.step for m in await store.get_metrics(
        run.id, name="val_loss", after_step=1)] == [2, 3]
    assert [m.step for m in await store.get_metrics(
        run.id, name="train_loss", limit=2)] == [1, 2]


async def test_log_metrics_batch_inserts_every_point(store):
    run = await _make_run(store)
    points = [MetricPoint("train_loss", 1.0 / (s + 1), s) for s in range(50)]
    assert await store.log_metrics(run.id, points) == 50
    assert len(await store.get_metrics(run.id, name="train_loss")) == 50
    assert await store.log_metrics(run.id, []) == 0


async def test_log_metrics_batch_is_atomic(store):
    """One bad point rolls the whole batch back -- the high-frequency
    consumer never leaves a half-written flush behind."""
    run = await _make_run(store)
    points = [MetricPoint("ok", 1.0, 0), MetricPoint("bad", 1.0, None)]
    with pytest.raises(sqlite3.IntegrityError):
        await store.log_metrics(run.id, points)
    assert await store.get_metrics(run.id) == []


async def test_list_metric_names_is_sorted_and_deduped(store):
    run = await _make_run(store)
    await store.log_metrics(run.id, [
        MetricPoint("val_loss", 1.0, 0), MetricPoint("train_loss", 2.0, 0),
        MetricPoint("train_loss", 1.5, 1),
    ])
    assert await store.list_metric_names(run.id) == ["train_loss", "val_loss"]
    assert await store.list_metric_names("ghost") == []


async def test_metric_for_an_unknown_run_is_rejected(store):
    with pytest.raises(sqlite3.IntegrityError):
        await store.log_metric("ghost", "train_loss", 1.0, 0)


async def test_a_diverged_loss_is_stored_as_none_not_an_error(store):
    # sqlite binds NaN as NULL, so a NOT NULL column would reject it with a
    # constraint error -- and because the flush is all-or-nothing, would
    # discard every good point around it. A diverged loss is the single
    # thing a user most wants to SEE on the chart.
    run = await _make_run(store)
    points = [MetricPoint("train_loss", 0.5, 0),
              MetricPoint("train_loss", float("nan"), 1),
              MetricPoint("train_loss", float("inf"), 2),
              MetricPoint("train_loss", float("-inf"), 3),
              MetricPoint("train_loss", 0.25, 4)]
    assert await store.log_metrics(run.id, points) == 5
    series = await store.get_metrics(run.id, name="train_loss")
    assert [m.step for m in series] == [0, 1, 2, 3, 4]
    assert [m.value for m in series] == [0.5, None, None, None, 0.25]


async def test_latest_metrics_answers_a_whole_page_of_runs_at_once(store):
    """The Runs table's per-row summary — one query, not one per row."""
    first = await _make_run(store)
    second = await _make_run(store)
    barren = await _make_run(store)
    await store.log_metrics(first.id, [
        MetricPoint("train_loss", 2.0, 1), MetricPoint("train_loss", 0.5, 3),
        MetricPoint("train_loss", 1.0, 2), MetricPoint("val_loss", 0.9, 3),
    ])
    await store.log_metrics(second.id, [MetricPoint("lr", 0.01, 7)])

    finals = await store.latest_metrics([first.id, second.id, barren.id])
    assert finals[first.id] == {"train_loss": 0.5, "val_loss": 0.9}
    assert finals[second.id] == {"lr": 0.01}
    # A run with no metrics is simply absent, not a run_id mapped to {}.
    assert barren.id not in finals
    # Scoping is real: asking for one run does not leak the other's series.
    assert await store.latest_metrics([second.id]) == {second.id: {"lr": 0.01}}
    assert await store.latest_metrics([]) == {}
    assert await store.latest_metrics(["ghost"]) == {}


async def test_latest_metrics_omits_a_series_whose_last_point_diverged(store):
    """NaN reads back as NULL; reporting it as 0.0 would look like success."""
    run = await _make_run(store)
    await store.log_metrics(run.id, [
        MetricPoint("train_loss", 0.5, 1),
        MetricPoint("val_loss", 0.4, 1),
        MetricPoint("train_loss", float("nan"), 2),
    ])
    assert await store.latest_metrics([run.id]) == {run.id: {"val_loss": 0.4}}


async def test_get_metrics_limit_requires_a_name(store):
    run = await _make_run(store)
    await store.log_metrics(run.id, [MetricPoint("a", 1.0, 0),
                                     MetricPoint("b", 1.0, 0)])
    with pytest.raises(ValueError, match="limit requires name"):
        await store.get_metrics(run.id, limit=1)


async def test_event_payloads_never_store_non_json_tokens(store):
    # json.dumps defaults to allow_nan=True and emits the bare tokens NaN /
    # Infinity, which JSON.parse throws on -- and the WS attach path replays
    # stored payloads verbatim to a browser.
    run = await _make_run(store)
    await store.append_event(run.id, "progress", {
        "loss": float("nan"), "lr": float("inf"),
        "losses": [1.0, float("-inf")], "epoch": 3})
    raw = await store.db.run(lambda conn: conn.execute(
        "SELECT payload FROM exec_run_events WHERE run_id = ?",
        (run.id,)).fetchone()[0])
    assert "NaN" not in raw and "Infinity" not in raw
    assert json.loads(raw) == {"loss": None, "lr": None,
                               "losses": [1.0, None], "epoch": 3}
    assert (await store.get_events(run.id))[0].payload["epoch"] == 3


async def test_artifact_meta_and_options_are_also_json_safe(store):
    run = await store.create_run(graph_snapshot={"t": [float("nan")]},
                                 options={"lr": float("inf")},
                                 provenance=RunProvenance())
    assert (await store.get_run(run.id)).options == {"lr": None}
    assert await store.get_graph_snapshot(run.id) == {"t": [None]}
    art = await store.add_artifact(run.id, "checkpoint", "c.pt",
                                   meta={"loss": float("nan")})
    assert (await store.list_artifacts(run.id))[0].meta == {"loss": None}
    assert art.id > 0


async def test_returned_records_equal_what_a_reread_gives(store):
    """The RETURN VALUE must be sanitized too, not just the stored row.

    #120 serialises the record it got back straight into its response, so
    a create_run(...) that still held the caller's raw NaN would put the
    non-JSON token back on the wire that this store just removed.
    """
    created = await store.create_run(
        graph_snapshot={}, options={"lr": float("nan"), "epochs": 3},
        provenance=RunProvenance())
    assert created.options == {"lr": None, "epochs": 3}
    assert created == await store.get_run(created.id)

    art = await store.add_artifact(created.id, "checkpoint", "c.pt",
                                   meta={"loss": float("-inf"), "epoch": 1})
    assert art.meta == {"loss": None, "epoch": 1}
    assert art == (await store.list_artifacts(created.id))[0]


async def test_create_run_does_not_alias_the_callers_dict(store):
    options = {"device": "cpu"}
    created = await store.create_run(graph_snapshot={}, options=options,
                                     provenance=RunProvenance())
    options["device"] = "cuda"          # caller keeps using its own dict
    assert created.options == {"device": "cpu"}
    assert (await store.get_run(created.id)).options == {"device": "cpu"}


# ── 5. artifacts ──────────────────────────────────────────────────────────


def test_artifact_kind_constants_are_the_documented_literals():
    assert (ARTIFACT_KIND_CHECKPOINT, ARTIFACT_KIND_EXPORT,
            ARTIFACT_KIND_IMAGE) == ("checkpoint", "export", "image")


async def test_add_and_list_artifacts(store):
    run = await _make_run(store)
    art = await store.add_artifact(
        run.id, ARTIFACT_KIND_CHECKPOINT, "models/interrupt.pt",
        meta={"epoch": 3, "batch": 12})
    assert art.id > 0
    assert art.kind == "checkpoint"
    assert art.path == "models/interrupt.pt"
    assert art.meta == {"epoch": 3, "batch": 12}
    assert art.created_at

    listed = await store.list_artifacts(run.id)
    assert listed == [art]


async def test_list_artifacts_filters_by_kind(store):
    run = await _make_run(store)
    await store.add_artifact(run.id, "checkpoint", "a.pt")
    await store.add_artifact(run.id, "image", "b.png")
    assert [a.path for a in await store.list_artifacts(run.id, kind="image")] \
        == ["b.png"]
    assert len(await store.list_artifacts(run.id)) == 2
    assert await store.list_artifacts("ghost") == []


async def test_artifact_meta_defaults_to_none(store):
    run = await _make_run(store)
    art = await store.add_artifact(run.id, "export", "out.py")
    assert art.meta is None


async def test_artifact_for_an_unknown_run_is_rejected(store):
    with pytest.raises(sqlite3.IntegrityError):
        await store.add_artifact("ghost", "checkpoint", "a.pt")


# ── 6. retention ──────────────────────────────────────────────────────────


async def _child_counts(database: Database, run_id: str) -> tuple[int, int, int]:
    def _count(conn: sqlite3.Connection) -> tuple[int, int, int]:
        return tuple(
            conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
            for table in ("exec_run_metrics", "exec_run_events",
                          "exec_run_artifacts")
        )
    return await database.run(_count)


async def _populate(store: RunStore, run: RunRecord) -> None:
    await store.log_metric(run.id, "train_loss", 1.0, 0)
    await store.append_event(run.id, "execution_start")
    await store.add_artifact(run.id, "checkpoint", "c.pt")


async def test_delete_run_cascades_to_children(store, db):
    run = await _make_run(store)
    await _populate(store, run)
    assert await _child_counts(db, run.id) == (1, 1, 1)

    assert await store.delete_run(run.id) is True
    assert await store.get_run(run.id) is None
    assert await _child_counts(db, run.id) == (0, 0, 0)
    assert await store.delete_run(run.id) is False


async def test_prune_keeps_the_newest_n_and_cascades(store, db):
    runs = []
    for _ in range(5):
        run = await _make_run(store)
        await _populate(store, run)
        await store.mark_finished(run.id, "succeeded")
        runs.append(run)

    assert await store.prune(keep_last=2) == 3
    remaining = {r.id for r in await store.list_runs()}
    assert remaining == {runs[3].id, runs[4].id}
    assert await _child_counts(db, runs[0].id) == (0, 0, 0)
    assert await _child_counts(db, runs[4].id) == (1, 1, 1)
    assert await store.prune(keep_last=2) == 0      # nothing left to drop


async def test_prune_never_deletes_queued_or_running_runs(store):
    old_active = await _make_run(store, name="active")
    await store.mark_running(old_active.id)
    old_queued = await _make_run(store, name="queued")
    for _ in range(3):
        done = await _make_run(store)
        await store.mark_finished(done.id, "succeeded")

    assert await store.prune(keep_last=0) == 3
    survivors = {r.name for r in await store.list_runs()}
    assert survivors == {"active", "queued"}


async def test_prune_rejects_a_negative_keep_last(store):
    with pytest.raises(ValueError, match="keep_last"):
        await store.prune(keep_last=-1)


async def test_publish_retention_does_not_touch_exec_runs(store, db):
    """Database.prune_runs targets the publish `runs` table only -- the two
    same-named concepts must never prune each other."""
    run = await _make_run(store)
    await store.mark_finished(run.id, "succeeded")
    db._conn.execute(
        "UPDATE exec_runs SET created_at = '2000-01-01T00:00:00.000000Z'")
    assert await db.prune_runs(1, force=True) == 0
    assert await store.get_run(run.id) is not None


# ── 7. provenance ─────────────────────────────────────────────────────────


async def test_create_run_persists_captured_provenance(store, monkeypatch):
    monkeypatch.setattr(
        "app.core.run_store.git_provenance", lambda d: ("a" * 40, True))
    monkeypatch.setattr(
        "app.core.run_store.load_lockfile",
        lambda: {"schema": 1, "plugins": {
            "demo": {"source_kind": "github_url", "source": "o/r",
                     "ref": "v1", "sha": "b" * 40, "enabled": True,
                     "manifest": {"version": "0.2.0"}}}})
    monkeypatch.setattr("app.core.run_store.settings.PROJECT_DIR",
                        "/some/project")

    run = await store.create_run(graph_snapshot=GRAPH)
    assert run.git_commit == "a" * 40
    assert run.git_dirty is True
    assert run.plugin_pins == {"demo": {
        "source_kind": "github_url", "source": "o/r", "ref": "v1",
        "sha": "b" * 40, "version": "0.2.0"}}

    reread = await store.get_run(run.id)
    assert reread.git_commit == run.git_commit
    assert reread.git_dirty is True                  # INTEGER 1 -> True
    assert reread.plugin_pins == run.plugin_pins


async def test_provenance_skips_git_outside_project_mode(store, monkeypatch):
    called: list[object] = []
    monkeypatch.setattr("app.core.run_store.git_provenance",
                        lambda d: called.append(d) or ("x", False))
    monkeypatch.setattr("app.core.run_store.load_lockfile",
                        lambda: {"schema": 1, "plugins": {}})
    monkeypatch.setattr("app.core.run_store.settings.PROJECT_DIR", None)

    run = await store.create_run(graph_snapshot=GRAPH)
    assert called == []                    # no subprocess in non-project mode
    assert run.git_commit is None
    assert run.git_dirty is None
    assert run.plugin_pins == {}


def test_capture_binds_the_real_git_helper():
    """Wiring check without spawning git.

    Asserting `capture(x) == git_provenance(x)` against the live checkout
    would be both tautological and flaky: `dirty` is whatever the working
    tree looks like at that instant, so anything writing into the repo
    between the two calls reddens the test. Prove the binding by identity;
    the companion test below proves the argument.
    """
    from app.core import project, run_store

    assert run_store.git_provenance is project.git_provenance


def test_capture_passes_the_project_dir_as_a_path(monkeypatch):
    seen: list[object] = []
    monkeypatch.setattr("app.core.run_store.git_provenance",
                        lambda d: (seen.append(d) or ("a" * 40, False)))
    monkeypatch.setattr("app.core.run_store.load_lockfile",
                        lambda: {"schema": 1, "plugins": {}})
    monkeypatch.setattr("app.core.run_store.settings.PROJECT_DIR",
                        r"C:\projects\demo")

    from pathlib import Path
    assert RunProvenance.capture().git_commit == "a" * 40
    assert seen == [Path(r"C:\projects\demo")]     # str coerced to Path

    seen.clear()
    RunProvenance.capture(Path("/explicit"))       # explicit beats settings
    assert seen == [Path("/explicit")]


@pytest.fixture
def no_git(monkeypatch):
    """Keep the plugin-pin tests off real `git` subprocesses."""
    monkeypatch.setattr("app.core.run_store.git_provenance",
                        lambda d: (None, None))


def test_provenance_capture_tolerates_a_corrupt_lockfile(
        monkeypatch, tmp_path, no_git):
    monkeypatch.setattr("app.core.run_store.load_lockfile",
                        lambda: {"schema": 1, "plugins": "not-a-dict"})
    assert RunProvenance.capture(tmp_path).plugin_pins == {}


def test_provenance_capture_skips_disabled_plugins(
        monkeypatch, tmp_path, no_git):
    monkeypatch.setattr("app.core.run_store.load_lockfile", lambda: {
        "schema": 1,
        "plugins": {
            "on": {"source_kind": "builtin", "source": "on"},
            "off": {"source_kind": "builtin", "source": "off",
                    "enabled": False},
        },
    })
    pins = RunProvenance.capture(tmp_path).plugin_pins
    assert set(pins) == {"on"}
    assert pins["on"] == {"source_kind": "builtin", "source": "on",
                          "ref": "", "sha": "", "version": None}


async def test_explicit_provenance_overrides_capture(store, monkeypatch):
    monkeypatch.setattr("app.core.run_store.git_provenance",
                        lambda d: pytest.fail("capture must be skipped"))
    run = await store.create_run(
        graph_snapshot=GRAPH,
        provenance=RunProvenance(git_commit="c" * 40, git_dirty=False,
                                 plugin_pins={}))
    assert run.git_commit == "c" * 40
    assert (await store.get_run(run.id)).git_dirty is False
