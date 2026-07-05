"""Stage-2 SQLite schema migrations.

``MIGRATIONS[i]`` moves ``PRAGMA user_version`` from ``i`` to ``i + 1``.
Append-only: NEVER edit a shipped migration — add a new list entry.
Timestamps are ISO-8601 UTC TEXT (``app.core.db.utc_now_iso``).
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

MIGRATIONS: list[str] = [MIGRATION_001]


def _is_comment_only(statement: str) -> bool:
    """True if *statement* has no executable content once ``--`` line
    comments are stripped from every line.

    Used only to guard the final, possibly-unterminated tail fragment in
    :func:`iter_statements` — a script ending in a bare comment (no SQL,
    no semicolon after it) must never be yielded as a pseudo-statement for
    ``Connection.execute`` to choke on.
    """
    for line in statement.splitlines():
        code = line.split("--", 1)[0]
        if code.strip():
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
