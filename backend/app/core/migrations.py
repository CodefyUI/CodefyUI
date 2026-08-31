"""SQLite schema migrations.

``MIGRATIONS[i]`` moves ``PRAGMA user_version`` from ``i`` to ``i + 1``.
Append-only: NEVER edit a shipped migration — add a new list entry.
Timestamps are ISO-8601 UTC TEXT (``app.core.db.utc_now_iso``).

001/002 are the Stage-2 publish subsystem; 003 is the Run Service store
(see the naming-decision block above ``MIGRATION_003``); 004 is the sweep
schema (#140).
"""

from __future__ import annotations

import sqlite3
from typing import Iterator

MIGRATION_001 = """
CREATE TABLE apps (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  slug            TEXT NOT NULL UNIQUE,
  graph_name      TEXT NOT NULL,            -- source graph at last publish (informational)
  active_version  INTEGER,                  -- NULL = unpublished. app-enforced ref to app_versions.version
  record_io       INTEGER NOT NULL DEFAULT 1,
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL
);
CREATE TABLE app_versions (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  app_id            INTEGER NOT NULL REFERENCES apps(id) ON DELETE CASCADE,
  version           INTEGER NOT NULL,       -- 1,2,3... per app
  graph_json        TEXT NOT NULL,          -- immutable snapshot (exact saved-file bytes)
  contract_json     TEXT NOT NULL,          -- derived contract at publish. feeds openapi.json
  source_graph_name TEXT NOT NULL,
  note              TEXT,                   -- optional publish note. immutable version metadata
  created_at        TEXT NOT NULL,
  UNIQUE (app_id, version)
);
CREATE TABLE api_keys (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  name         TEXT NOT NULL,
  prefix       TEXT NOT NULL,               -- first 12 chars, display only
  token_hash   TEXT NOT NULL UNIQUE,        -- sha256 hex of the full token
  created_at   TEXT NOT NULL,
  last_used_at TEXT,
  revoked_at   TEXT                          -- NULL = active (soft revoke keeps runs.api_key_id meaningful)
);
CREATE TABLE runs (
  run_id            TEXT PRIMARY KEY,        -- same uuid4 hex as the envelope run_id
  app_id            INTEGER NOT NULL REFERENCES apps(id) ON DELETE CASCADE,
  version           INTEGER NOT NULL,        -- row exists iff resolution succeeded, so never NULL
  api_key_id        INTEGER REFERENCES api_keys(id),
                    -- no ON DELETE action: api_keys rows are soft-revoked, never deleted.
                    -- NULL is RESERVED for future editor-originated invokes (Stage-3 test
                    -- button). Stage 2 never writes NULL here.
  status            TEXT NOT NULL,           -- "ok" | "error"
  error_code        TEXT, error_message TEXT, error_node_id TEXT,
  device            TEXT,
  total_s           REAL,
  node_timings_json TEXT,                    -- {"<node_id>": seconds}
  inputs_json       TEXT,                    -- capped/redacted JSON (marker objects when not stored)
  outputs_json      TEXT,
  created_at        TEXT NOT NULL
);
CREATE INDEX idx_runs_app_created ON runs(app_id, created_at DESC);
CREATE INDEX idx_runs_created     ON runs(created_at);
"""

MIGRATION_002 = """
ALTER TABLE app_versions ADD COLUMN git_commit TEXT;
ALTER TABLE app_versions ADD COLUMN git_dirty INTEGER;
"""

# ── Migration 003: Run Service storage ─────────────────────────────────────
#
# NAMING DECISION (issue #119, acceptance criterion 3) — why ``exec_runs``
# and not ``runs``:
#
# ``runs`` (MIGRATION_001) is already taken, and it is a DIFFERENT concept.
# It is the *published-app invocation* history: every row REQUIRES an
# ``app_id`` (NOT NULL, FK CASCADE to ``apps``) plus a ``version``, its
# status vocabulary is the two-valued ``"ok" | "error"`` REST envelope
# result, and it stores request/response I/O (``inputs_json`` /
# ``outputs_json``, capped at ``settings.RUN_IO_CAP_BYTES``). Editor and
# CLI graph executions have no row there at all.
#
# What this migration adds is the *graph execution* run: no app, a six-state
# lifecycle (queued/running/succeeded/failed/cancelled/interrupted), a graph
# snapshot instead of request I/O, and child tables for metrics, a replay
# event log and artifacts.
#
# Options considered:
#   (a) Reuse ``runs``. Rejected: it would mean relaxing ``app_id`` to
#       nullable, overloading one ``status`` column with two disjoint
#       vocabularies, and rebuilding a SHIPPED table (SQLite cannot drop a
#       NOT NULL constraint in place) — a destructive migration of live
#       user data to save one word of naming.
#   (b) Migrate ``runs`` into the new tables. Rejected for the same reason
#       plus a lossy mapping: publish rows carry per-field I/O and API-key
#       attribution that have no home in the execution schema.
#   (c) Namespace the new tables ``exec_*``. CHOSEN. Purely additive: 001
#       and 002 are untouched, existing publish history keeps working
#       byte-for-byte, and the prefix reads correctly at every call site
#       (``exec_run_events`` is unambiguous next to ``runs``).
#
# Consequence to remember: ``Database.prune_runs`` is the *publish* table's
# retention (``DELETE FROM runs``); ``RunStore.prune`` is this one's. They
# are deliberately separate policies and never touch each other's rows.
#
# Schema notes:
# - ``status`` carries NO SQL CHECK constraint: SQLite cannot ALTER one away
#   later, and the queue/sweep work may add lifecycle states. The closed
#   vocabulary is enforced in ``app.core.run_store`` (RUN_STATUSES), which
#   is the only writer.
# - Child tables use a plain ``INTEGER PRIMARY KEY`` (rowid alias) rather
#   than AUTOINCREMENT: no ``sqlite_sequence`` write per insert, which
#   matters for the high-frequency metric path, and nothing references a
#   metric/event row by id so rowid reuse after a delete is harmless.
# - Everything is additive-friendly: the sweep work adds ``sweep_id`` with a
#   bare ``ALTER TABLE exec_runs ADD COLUMN``.
MIGRATION_003 = """
CREATE TABLE exec_runs (
  id             TEXT PRIMARY KEY,       -- uuid4().hex, like runs.run_id
  name           TEXT,                   -- user-facing label. NULL = unnamed
  graph_snapshot TEXT NOT NULL,          -- JSON. immutable copy of the submitted graph
  options        TEXT NOT NULL,          -- JSON {device, seed, record_outputs, lane}
  status         TEXT NOT NULL,          -- queued|running|succeeded|failed|cancelled|interrupted
  error          TEXT,                   -- failure summary. NULL unless status = failed
  queue_key      TEXT,                   -- resolved device, e.g. cpu / cuda:0. NULL until scheduled
  created_at     TEXT NOT NULL,          -- submit time. the list + retention ordering key
  started_at     TEXT,                   -- NULL while queued
  finished_at    TEXT,                   -- NULL until terminal
  git_commit     TEXT,                   -- project provenance. NULL = unknown, never guessed
  git_dirty      INTEGER,                -- 0/1/NULL, same tri-state as app_versions.git_dirty
  plugin_pins    TEXT                    -- JSON {plugin_id: {source_kind, source, ref, sha, version}}
);
-- ASC, deliberately, even though every read is newest-first. The list
-- order is (created_at DESC, rowid DESC), and sqlite serves that by
-- scanning an ASC index BACKWARDS -- entries are stored (created_at ASC,
-- rowid ASC), so reversed they are exactly (created_at DESC, rowid DESC).
-- A DESC index stores (created_at DESC, rowid ASC): the tie-break points
-- the wrong way, so sqlite falls back to a full scan plus a temp b-tree
-- sort (verified with EXPLAIN QUERY PLAN both ways). Do not "fix" these
-- to DESC.
CREATE INDEX idx_exec_runs_created        ON exec_runs(created_at);
CREATE INDEX idx_exec_runs_status_created ON exec_runs(status, created_at);

CREATE TABLE exec_run_metrics (
  id      INTEGER PRIMARY KEY,
  run_id  TEXT NOT NULL REFERENCES exec_runs(id) ON DELETE CASCADE,
  node_id TEXT,                          -- emitting node. NULL for run-level scalars
  name    TEXT NOT NULL,                 -- series name, e.g. train_loss
  step    INTEGER NOT NULL,              -- epoch or batch index, series-defined
  -- Nullable on purpose. sqlite has no NaN: the driver binds one as NULL,
  -- so NOT NULL here would reject a diverged loss with a constraint error
  -- naming a column the caller never set -- and, because the metric flush
  -- is one all-or-nothing transaction, would discard the whole batch of
  -- good points around it. run_store normalises every non-finite value
  -- (NaN, +/-inf) to NULL and reads it back as None, which is also the
  -- only JSON-representable answer for the chart consumers.
  value   REAL,
  ts      TEXT NOT NULL
);
-- Serves both chart reads: one series (run_id, name -> ordered steps) and
-- the series list (DISTINCT name uses the same index prefix).
CREATE INDEX idx_exec_run_metrics_series ON exec_run_metrics(run_id, name, step);

CREATE TABLE exec_run_events (
  id      INTEGER PRIMARY KEY,
  run_id  TEXT NOT NULL REFERENCES exec_runs(id) ON DELETE CASCADE,
  cursor  INTEGER NOT NULL,              -- 1-based, gapless, monotonic within a run
  type    TEXT NOT NULL,                 -- open vocabulary. the WS message type
  payload TEXT,                          -- JSON. NULL when the type needs no body
  ts      TEXT NOT NULL,
  -- UNIQUE, not a plain index: it is the (run_id, cursor) replay index AND
  -- the backstop that turns a lost cursor update into a loud IntegrityError
  -- instead of two events silently sharing a cursor.
  UNIQUE (run_id, cursor)
);

CREATE TABLE exec_run_artifacts (
  id         INTEGER PRIMARY KEY,
  run_id     TEXT NOT NULL REFERENCES exec_runs(id) ON DELETE CASCADE,
  kind       TEXT NOT NULL,              -- checkpoint|export|image. open vocabulary
  path       TEXT NOT NULL,              -- as written by the node, under the data root
  meta       TEXT,                       -- JSON, e.g. {epoch, batch} for an interrupt checkpoint
  created_at TEXT NOT NULL
);
CREATE INDEX idx_exec_run_artifacts_run ON exec_run_artifacts(run_id, created_at);
"""

# ── Migration 004: sweeps (#140) ──────────────────────────────────────────
#
# A sweep is a parent of N variant runs. What is load-bearing here is that
# the CREATE and the ALTERs ship in ONE entry -- not their order inside it.
# Measured on SQLite 3.50.4 with PRAGMA foreign_keys=ON: both orders commit
# and every subsequent DML behaves identically, because a REFERENCES clause
# added by ADD COLUMN is resolved at DML time, not at DDL time. The real
# hazard is SPLITTING them: an ALTER-only migration commits happily and then
# every INSERT INTO exec_runs -- and, worse, every DELETE FROM exec_runs --
# fails with "no such table: main.sweeps", forever, until the CREATE ships.
# Measured statement by statement: SELECT succeeds, UPDATE of a non-FK
# column succeeds, INSERT and DELETE both fail. The DELETE half is the bad
# one: RunStore.prune's DELETE runs after EVERY run reaches a terminal state
# (run_service.py:2005) and at every startup (main.py:364), so a split
# migration would not merely refuse new runs -- it would raise inside
# retention on every finishing run in the process. Do not write a test
# asserting the ORDER matters -- both orders migrate successfully, so it
# would fail.
#
# RULING 4 is why `variants` is a column and not a pointer: RunStore.prune
# deletes finished runs after EVERY terminal transition, so a sweep that
# only held run ids would rot into dangling references. Each variant's
# chosen params and harvested objective live on this row; the run id stays
# as a link that MAY BE DEAD.
MIGRATION_004 = """
CREATE TABLE sweeps (
  id            TEXT PRIMARY KEY,   -- uuid4().hex, same shape as exec_runs.id
  name          TEXT,               -- user label. NULL = unnamed, per normalize_name
  state         TEXT NOT NULL,      -- running|cancelling|finished|failed. READ-REPAIRED,
                                    -- never pushed: nothing wakes up to advance it, so a
                                    -- sweep that finished while nobody was looking still
                                    -- reads 'running' HERE until the next read, cancel or
                                    -- prune runs the harvest. And 'failed' means the SWEEP
                                    -- broke (the submit loop gave some variant no run at
                                    -- all), never that the training did: a sweep whose
                                    -- variants all failed is 'finished'. Per-variant
                                    -- outcomes live in `variants`, not here.
  method        TEXT NOT NULL,      -- grid|random
  seed          INTEGER,            -- the sweep's own PLANNER seed: it selects WHICH
                                    -- COMBINATIONS EXIST and is not by itself what seeds
                                    -- a graph. Required for random; optional for grid,
                                    -- which enumerates everything and so consumes none.
                                    -- NULL only for a grid sweep that sent none -- RULING
                                    -- 1 records it whenever there is one, because it is
                                    -- also the base the per-variant execution seeds below
                                    -- are derived from.
  seed_variants INTEGER NOT NULL,   -- 0/1. Per-variant EXECUTION seeds -- the kind that
                                    -- reach a run and seed the graph itself -- opt-in and
                                    -- default OFF (RULING 1): seeding every variant would
                                    -- serialise the whole sweep and stall interactive runs
                                    -- alongside it.
  spec          TEXT NOT NULL,      -- JSON. the compiled, normalised sweep_spec
  objective     TEXT NOT NULL,      -- JSON {"metric": str, "direction": "minimize"|"maximize"}.
                                    -- Both keys required at submit; there is NO default
                                    -- metric and no inferred direction, and the pair is
                                    -- immutable in v1 (re-ranking would need child series
                                    -- that retention may already have deleted).
  variants      TEXT NOT NULL,      -- JSON list, one entry per variant, holding that
                                    -- variant's chosen params and its harvested objective.
                                    -- RULING 4: this is what keeps the sweep answerable
                                    -- after retention has deleted its children, which is
                                    -- also why the run id inside an entry is a link that
                                    -- MAY BE DEAD rather than a foreign key.
  error         TEXT,               -- failure summary. NULL unless state = 'failed'
  created_at    TEXT NOT NULL,      -- submit time; the list ordering key
  finished_at   TEXT                -- NULL until state is finished or failed, and written
                                    -- by the same read repair that moves `state`
);
-- ASC deliberately, for the reason spelled out above idx_exec_runs_created:
-- a (created_at DESC, rowid DESC) read is served by scanning an ASC index
-- backwards. Do not "fix" this to DESC.
CREATE INDEX idx_sweeps_created ON sweeps(created_at);

-- Both columns are nullable, and a writer sets both or neither: NULL means
-- "not part of a sweep", the same NULL-is-unknown convention git_commit /
-- git_dirty already use. A `sweep_variant INTEGER NOT NULL DEFAULT 0` would
-- be legal SQLite and would tell every pre-existing run it is variant 0 of
-- nothing.
--
-- They are NOT always NULL together once written, and a reader must not
-- assume it: ON DELETE SET NULL clears `sweep_id` alone, so a run whose
-- sweep row was deleted keeps its `sweep_variant`. Find a sweep's children
-- by `sweep_id IS NOT NULL`, never by `sweep_variant IS NOT NULL`.
--
-- ON DELETE SET NULL, never CASCADE: deleting a sweep row must not delete
-- run history. SQLite cannot add a UNIQUE column via ADD COLUMN, so the
-- (sweep_id, sweep_variant) pairing is a read index, not a constraint;
-- uniqueness is the writer's job -- the submit loop that creates a sweep's
-- children is the only assigner, and gives out each index exactly once.
ALTER TABLE exec_runs ADD COLUMN sweep_id TEXT
  REFERENCES sweeps(id) ON DELETE SET NULL;
ALTER TABLE exec_runs ADD COLUMN sweep_variant INTEGER;
CREATE INDEX idx_exec_runs_sweep ON exec_runs(sweep_id, sweep_variant);
"""

MIGRATIONS: list[str] = [MIGRATION_001, MIGRATION_002, MIGRATION_003,
                         MIGRATION_004]


def _is_comment_only(statement: str) -> bool:
    """True if *statement* has no executable content — only whitespace,
    ``--`` line comments, and ``/* ... */`` block comments (an
    unterminated trailing block comment counts as comment too).

    Used only to guard the final, possibly-unterminated tail fragment in
    :func:`iter_statements` — a script ending in a bare comment (no SQL,
    no semicolon after it) must never be yielded as a pseudo-statement for
    ``Connection.execute`` to choke on. A single sequential scan handles
    both comment syntaxes in order (mirroring sqlite's own tokenizer), so
    ``/*`` inside a line comment or ``--`` inside a block comment cannot
    be misread; anything that is not whitespace or a comment opener makes
    this return False — when in doubt the tail is yielded and sqlite
    fails loudly rather than a statement being dropped silently.
    """
    i, n = 0, len(statement)
    while i < n:
        ch = statement[i]
        if ch in " \t\r\n":
            i += 1
        elif statement.startswith("--", i):
            newline = statement.find("\n", i)
            i = n if newline == -1 else newline + 1
        elif statement.startswith("/*", i):
            end = statement.find("*/", i + 2)
            i = n if end == -1 else end + 2
        else:
            return False
    return True


def iter_statements(script: str) -> Iterator[str]:
    """Split a migration script into single executable statements.

    ``Connection.execute`` runs exactly one statement, and
    ``executescript`` force-COMMITs first — useless inside the explicit
    ``BEGIN IMMEDIATE`` the migration runner holds. Accumulate lines until
    ``sqlite3.complete_statement`` says a full statement (comments and
    string literals understood) is buffered.
    """
    buffer = ""
    for line in script.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            statement = buffer.strip()
            if statement:
                yield statement
            buffer = ""
    tail = buffer.strip()
    if tail and not _is_comment_only(tail):
        yield tail
