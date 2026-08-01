"""SQLite schema migrations.

``MIGRATIONS[i]`` moves ``PRAGMA user_version`` from ``i`` to ``i + 1``.
Append-only: NEVER edit a shipped migration — add a new list entry.
Timestamps are ISO-8601 UTC TEXT (``app.core.db.utc_now_iso``).

001/002 are the Stage-2 publish subsystem; 003 is the Run Service store
(see the naming-decision block above ``MIGRATION_003``).
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
  value   REAL NOT NULL,
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

MIGRATIONS: list[str] = [MIGRATION_001, MIGRATION_002, MIGRATION_003]


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
