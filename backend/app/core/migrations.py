"""SQLite schema migrations.

``MIGRATIONS[i]`` moves ``PRAGMA user_version`` from ``i`` to ``i + 1``.
Append-only: NEVER edit a shipped migration — add a new list entry.
Timestamps are ISO-8601 UTC TEXT (``app.core.db.utc_now_iso``).

001/002 are the Stage-2 publish subsystem; 003 is the Run Service store
(see the naming-decision block above ``MIGRATION_003``); 004 is the sweep
schema (#140); 005 indexes metrics by producer (#641); 006 records the
environment that executed each run (#344); 007 is the durable study
coordinator (#343); 008 records where each run executes (#769); 009 is
worker enrollment (#780); 010 says where each artifact's file lives
(#782); 011 records a worker's environment (#781); 012 the SSH workers
the control plane launches (#784); 013 is the transfer backend's content
index (#785); 014 records the S3 bucket of an artifact (#791); 015 the
per-device admission policies and each run's graph hash (#788); 016
whether an enrolled worker's own last agent was one-shot (#845).
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

# ── Migration 005: per-producer metric index (#641) ───────────────────────
#
# A sweep objective names a node as well as a series, and its value is the
# last point THAT node logged. idx_exec_run_metrics_series cannot answer it
# by seek: within (run_id, name) its order is step, so finding one node's
# last point, or the list of nodes that logged a name, walks every point of
# every producer. This index puts node_id before step, which makes both a
# seek (``run_store.metric_producers`` / ``producer_last_value``). The
# series index stays: the Runs summary orders by step ACROSS producers,
# which this index cannot serve.
MIGRATION_005 = """
CREATE INDEX idx_exec_run_metrics_producer
  ON exec_run_metrics(run_id, name, node_id, step);
"""

# ── Migration 006: the environment that executed a run (#344) ─────────────
#
# ``git_commit`` / ``plugin_pins`` describe the user's project at SUBMISSION.
# These describe the CodefyUI process that STARTED the run, written by
# ``RunStore.mark_running``:
#
# - ``exec_run_fingerprints`` is content-addressed: one row per distinct
#   environment, keyed by the sha256 of its canonical JSON
#   (``core.env_fingerprint``). A server runs thousands of runs in one
#   environment; storing the same few kilobytes on every row would be
#   waste, and a digest alone could not explain a mismatch. Nothing
#   references it by foreign key and retention does not prune it, so a
#   digest stays resolvable after its runs are deleted.
# - ``exec_runs.fingerprint_digest`` points into it. NULL = the run never
#   started, or started before this migration.
# - ``exec_runs.execution_facts`` is small per-run JSON: what was requested
#   next to what was resolved (device, seed, determinism, torch numerics
#   switches). NULL on the same terms.
MIGRATION_006 = """
CREATE TABLE exec_run_fingerprints (
  digest         TEXT PRIMARY KEY,       -- sha256 hex of the canonical JSON
  schema_version INTEGER NOT NULL,
  fingerprint    TEXT NOT NULL,          -- JSON, canonical form
  created_at     TEXT NOT NULL           -- first time this environment ran
);
ALTER TABLE exec_runs ADD COLUMN fingerprint_digest TEXT;
ALTER TABLE exec_runs ADD COLUMN execution_facts TEXT;
"""

# ── Migration 007: durable studies (#343) ──────────────────────────────────
#
# A study is a server-owned experiment: N variants x R repetitions, each
# repetition run as a queued child of the Run Service, coordinated by
# ``core.study_coordinator`` so it finishes after the browser that submitted
# it has gone and resumes after a restart. Four tables plus two columns on
# ``exec_runs``:
#
# - ``studies``: one row per study. ``status`` is the study's own state
#   machine (``study_store.STUDY_STATUSES``), with no CHECK constraint for
#   the reason ``exec_runs.status`` has none. ``revision`` goes up by one on
#   every write to the study or to one of its attempts, so a reader can tell
#   two snapshots apart without diffing them. ``idempotency_key`` is UNIQUE
#   (NULLs are distinct in SQLite, so an unkeyed submit is unconstrained);
#   ``request_digest`` is the sha256 of the canonical, SECRET-scrubbed
#   request, which is how a retry with the same key is told apart from a
#   key reused for a different study (409).
# - ``study_variants``: the resolved graph of every variant, SECRET-scrubbed
#   like ``exec_runs.graph_snapshot``. Stored here rather than read back off
#   a child run because retention deletes children, and a resume after that
#   still has to submit the variant.
# - ``study_attempts``: one row per (variant, repetition, attempt). The
#   attempt is the unit of recovery: completed attempts are never re-run,
#   an interrupted one is kept as history and a new attempt row is added.
#   ``run_id`` is a link that MAY BE DEAD (no FK), for the reason
#   ``sweeps.variants`` holds its run ids that way: the evidence columns
#   (objective value, final metrics, timestamps, fingerprint digest) are
#   copied onto the row when the child ends, so the study stays answerable
#   after retention has deleted the child.
# - ``study_events``: the study's audit log, with the same gapless per-owner
#   cursor ``exec_run_events`` has, so a client replays it after a cursor.
#
# ``exec_runs.study_id`` / ``study_attempt_id`` stamp every child with the
# attempt that submitted it. That stamp is what recovers a child that was
# submitted when the attempt's own linkage write failed: the coordinator
# finds the run by ``study_attempt_id`` instead of submitting a second one.
# The CREATE and the ALTER ship in ONE migration for the reason spelled out
# above MIGRATION_004 (an ALTER whose REFERENCES target does not exist yet
# makes every INSERT and DELETE on exec_runs fail).
MIGRATION_007 = """
CREATE TABLE studies (
  id              TEXT PRIMARY KEY,       -- uuid4().hex
  name            TEXT,                   -- user label. NULL = unnamed
  status          TEXT NOT NULL,          -- queued|running|cancel_requested|succeeded|failed|cancelled|interrupted
  revision        INTEGER NOT NULL,       -- +1 on every write to the study or its attempts
  objective       TEXT NOT NULL,          -- JSON {node_id, metric, direction}. exact, immutable
  repetitions     INTEGER NOT NULL,       -- scheduled repetitions per variant, >= 1
  seed            INTEGER NOT NULL,       -- study seed. repetition r runs with (seed + r) mod 2^32
  options         TEXT NOT NULL,          -- JSON run options copied into every child (no seed)
  budget          TEXT NOT NULL,          -- JSON {max_runs, max_parallel_runs}
  resume_policy   TEXT NOT NULL,          -- explicit|auto
  idempotency_key TEXT UNIQUE,            -- NULL = submitted without one
  request_digest  TEXT NOT NULL,          -- sha256 of the canonical scrubbed request
  error           TEXT,                   -- why the study failed. NULL otherwise
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL,
  finished_at     TEXT                    -- NULL while the study can still move
);
-- ASC on purpose; see the note above idx_exec_runs_created.
CREATE INDEX idx_studies_created ON studies(created_at);
CREATE INDEX idx_studies_status ON studies(status);

CREATE TABLE study_variants (
  study_id       TEXT NOT NULL REFERENCES studies(id) ON DELETE CASCADE,
  variant_index  INTEGER NOT NULL,        -- 0-based, the position in the request
  label          TEXT NOT NULL,           -- unique within the study
  ops            TEXT,                    -- JSON [{node_id, param, value}] or NULL for a whole graph
  graph_snapshot TEXT NOT NULL,           -- JSON, SECRET-scrubbed, ops applied
  PRIMARY KEY (study_id, variant_index)
);

CREATE TABLE study_attempts (
  id                 TEXT PRIMARY KEY,    -- uuid4().hex. stamped on the child run
  study_id           TEXT NOT NULL REFERENCES studies(id) ON DELETE CASCADE,
  variant_index      INTEGER NOT NULL,
  repetition         INTEGER NOT NULL,    -- 0-based
  attempt            INTEGER NOT NULL,    -- 1-based within (variant, repetition)
  seed               INTEGER NOT NULL,    -- the repetition's seed; every attempt reuses it
  run_id             TEXT,                -- the child. a link that MAY BE DEAD, no FK
  status             TEXT NOT NULL,       -- pending|submitting|queued|running|succeeded|failed|cancelled|interrupted|lost
  error              TEXT,
  created_at         TEXT NOT NULL,
  submitted_at       TEXT,
  started_at         TEXT,                -- copied from the child
  finished_at        TEXT,                -- copied from the child
  objective_value    REAL,                -- the objective producer's last point. NULL = none
  final_metrics      TEXT,                -- JSON {series: last value}, copied from the child
  fingerprint_digest TEXT,                -- the child's environment (#344)
  harvested_at       TEXT,                -- NULL until the evidence was copied
  UNIQUE (study_id, variant_index, repetition, attempt)
);
CREATE INDEX idx_study_attempts_run ON study_attempts(run_id);

CREATE TABLE study_events (
  id       INTEGER PRIMARY KEY,
  study_id TEXT NOT NULL REFERENCES studies(id) ON DELETE CASCADE,
  cursor   INTEGER NOT NULL,              -- 1-based, gapless, monotonic within a study
  type     TEXT NOT NULL,
  payload  TEXT,                          -- JSON
  ts       TEXT NOT NULL,
  UNIQUE (study_id, cursor)
);

ALTER TABLE exec_runs ADD COLUMN study_id TEXT
  REFERENCES studies(id) ON DELETE SET NULL;
ALTER TABLE exec_runs ADD COLUMN study_attempt_id TEXT;
CREATE INDEX idx_exec_runs_study_attempt ON exec_runs(study_attempt_id)
  WHERE study_attempt_id IS NOT NULL;
"""

# ── Migration 008: where a run executes (#769) ────────────────────────────
#
# Part of the control-plane / worker split (#764). Two nullable columns:
#
# - ``worker_id``: the worker that runs it: ``local`` (this process), a
#   worker's name, or ``any`` until such a run is placed (#772). NULL is a
#   row from before this migration and reads back as ``local``
#   (``RunRecord.from_row``), which is what it was.
# - ``devices``: the device set as a JSON array (``target.devices`` as
#   asked, then the devices a worker run was placed on), or NULL when the
#   run asked for one ``device`` the way every run did before. The first
#   entry is the run's device and so its ``queue_key``.
MIGRATION_008 = """
ALTER TABLE exec_runs ADD COLUMN worker_id TEXT;
ALTER TABLE exec_runs ADD COLUMN devices TEXT;
"""

# ── Migration 009: enrolled workers (#780) ────────────────────────────────
#
# Two tables, and no secret in either: every credential is stored as the
# sha256 hex of the secret (``core.worker_store.hash_secret``), the way
# ``api_keys.token_hash`` is.
#
# - ``workers``: one row per enrolled worker agent. ``id`` IS the name (a
#   worker's id is its name everywhere else: ``target.worker``,
#   ``exec_runs.worker_id``, the registry); ``name`` is kept as its own
#   UNIQUE column so the table reads like the spec. ``token_hash`` is the
#   long-lived worker token's hash; a re-enrollment under the same name
#   replaces it, so the old token stops working. ``platform``,
#   ``mounts_json`` and ``devices_json`` are the facts the agent last
#   reported (JSON), so a worker known from an earlier boot can be shown,
#   and targeted, before it reconnects; ``machine_id`` is the digest its
#   last ``hello`` reported (a record only: same-host also needs a loopback
#   peer, see ``ws_worker``). ``enabled`` is 1 for every row this
#   release writes; a revoked worker's row is deleted.
# - ``worker_enrollments``: one-time join tokens. ``id`` is the token's
#   public half (the six characters before the dot); ``uses_left`` counts
#   down to 0 and a used-up or expired row stays listed until deleted.
#   ``worker`` binds a token to one enrolled worker: it is the only kind
#   of token that can enroll a name that is already taken (a re-key).
MIGRATION_009 = """
CREATE TABLE workers (
  id            TEXT PRIMARY KEY,           -- the worker name
  name          TEXT NOT NULL UNIQUE,
  token_hash    TEXT NOT NULL UNIQUE,       -- sha256 hex of the worker token
  platform      TEXT,                       -- JSON {system, release, machine, python}
  mounts_json   TEXT,                       -- JSON {label: path}
  devices_json  TEXT,                       -- JSON [DeviceReport]
  machine_id    TEXT,                       -- hello.machine_id (a digest), last seen
  created_at    TEXT NOT NULL,
  last_seen_at  TEXT,
  enabled       INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE worker_enrollments (
  id          TEXT PRIMARY KEY,             -- the token's public half
  token_hash  TEXT NOT NULL UNIQUE,         -- sha256 hex of the whole token
  expires_at  TEXT NOT NULL,
  uses_left   INTEGER NOT NULL,
  created_at  TEXT NOT NULL,
  label       TEXT,
  worker      TEXT                          -- bound: may enroll (re-key) this name only
);
"""

# ── Migration 010: where an artifact's file lives (#782) ─────────────────
#
# A worker run writes its checkpoints on the worker. With a shared
# filesystem the file is under a mount both hosts hold, at a different
# absolute path on each, so the row records it relative to the mount:
#
# - ``backend``: ``local`` (``path`` is absolute on this machine: every row
#   before this migration, which reads back NULL and means ``local``),
#   ``mount`` (``path`` is relative to the mount ``mount_label`` names),
#   ``transfer`` (#785: ``path`` is the key the worker's upload was stored
#   under, relative to this server's data root, ``core.transfer.key_path``);
#   ``s3`` (#791: ``path`` is the object key in the S3 bucket, ``core.s3``).
# - ``mount_label``: the mount's label, for ``mount`` rows; else NULL.
# - ``worker_id``: the worker whose run produced the file (``local`` for
#   this process), so a path from another machine is never read as one of
#   this machine's. NULL on a row from before and reads back as ``local``.
#
# ``RunStore`` translates a ``mount`` row to this server's path when it is
# read (``core.mounts.translate``).
MIGRATION_010 = """
ALTER TABLE exec_run_artifacts ADD COLUMN backend TEXT;
ALTER TABLE exec_run_artifacts ADD COLUMN mount_label TEXT;
ALTER TABLE exec_run_artifacts ADD COLUMN worker_id TEXT;
"""

# ── Migration 011: a worker's environment (#781) ──────────────────────────
#
# ``workers.fingerprint_digest`` is the environment fingerprint digest the
# worker's last ``hello`` carried (``core.env_fingerprint``), so a worker
# known from an earlier boot shows it before it reconnects. The document
# itself is in ``exec_run_fingerprints``, content-addressed like a run's
# (``RunStore.record_fingerprint``). NULL: never reported, or the last
# connection reported none.
MIGRATION_011 = """
ALTER TABLE workers ADD COLUMN fingerprint_digest TEXT;
"""

# ── Migration 012: SSH workers (#784) ────────────────────────────────────
#
# One row per worker the control plane starts itself over ``ssh -R``
# (``core.ssh_launch``). ``id`` is the worker name (the same id as its
# ``workers`` row once it has enrolled; a row here may exist before that).
# ``target`` is what ``ssh`` is given (``user@host`` or a ``Host`` alias
# from the server account's ssh config), ``port`` its ``-p``, ``identity``
# an optional key path on THIS machine, ``remote_cdui`` the launcher on the
# remote host, ``remote_port`` the remote loopback port of the reverse
# forward (kept so a restart reuses it, and with it the credential the
# agent stored for that address). ``enabled`` 0 keeps the row but no ssh.
# No secret here: tokens travel on ssh's stdin, never stored.
MIGRATION_012 = """
CREATE TABLE ssh_workers (
  id           TEXT PRIMARY KEY,            -- the worker name
  target       TEXT NOT NULL,               -- user@host or an ssh Host alias
  port         INTEGER NOT NULL DEFAULT 22,
  identity     TEXT,                        -- key file on the control plane
  remote_cdui  TEXT NOT NULL,               -- the cdui launcher on the remote
  remote_port  INTEGER,                     -- remote loopback end of ssh -R
  enabled      INTEGER NOT NULL DEFAULT 1,
  created_at   TEXT NOT NULL
);
"""

# ── Migration 013: the transfer backend's content index (#785) ───────────
#
# A run sent to a worker without the shared mount names each upload it
# reads by its sha256, so the worker can cache it by content. Hashing a
# 2 GiB model on every assignment is not free, so the hash is kept here
# per stored file and recomputed only when the file's size or mtime moved
# (``core.transfer.ContentIndex``): ``(kind, name)`` is the upload store
# (``data_file`` / ``image`` / ``model``) and the name the graph uses;
# ``mtime`` is ``st_mtime_ns``. A cache, not a record: a row may name a
# file that is gone, and nothing but the next lookup refreshes it.
MIGRATION_013 = """
CREATE TABLE file_hashes (
  kind    TEXT NOT NULL,
  name    TEXT NOT NULL,
  sha256  TEXT NOT NULL,
  bytes   INTEGER NOT NULL,
  mtime   INTEGER NOT NULL,
  PRIMARY KEY (kind, name)
);
"""

# ── Migration 014: the S3 bucket of an artifact (#791) ────────────────────
#
# An ``s3`` row (``backend = 's3'``) names its object by key (``path``); the
# bucket goes here, so a row keeps naming its object after
# ``CODEFYUI_S3_BUCKET`` changes (its key carries the prefix of its time).
# NULL for every other row, and for every row from before: no row written
# by 3.1.0 is an ``s3`` row, so nothing is backfilled.
MIGRATION_014 = """
ALTER TABLE exec_run_artifacts ADD COLUMN bucket TEXT;
"""

# ── Migration 015: per-device admission policy (#788) ─────────────────────
#
# How a worker's card admits runs: ``exclusive`` (the count limit, one run
# per card, as before) or ``shared`` (memory admission: runs declare a
# budget and share the card while their budgets fit). Keyed by worker NAME
# and device as that worker numbers it (``cuda:1``), so it applies to a
# same-host worker known only in memory as well as to an enrolled one, and
# survives both a reconnect and a restart of this server. No row:
# ``exclusive``. Revoking a worker deletes its rows.
MIGRATION_015 = """
CREATE TABLE worker_device_policies (
  worker      TEXT NOT NULL,                -- the worker name
  device      TEXT NOT NULL,                -- canonical, the worker's numbering
  policy      TEXT NOT NULL,                -- 'exclusive' | 'shared'
  updated_at  TEXT NOT NULL,
  PRIMARY KEY (worker, device)
);

-- The graph a run executed, as memory_admission.graph_hash gives it, for a
-- run that targets a worker (#788). The default memory budget of a run that
-- declares none is the last peak of a succeeded run of the same graph; the
-- index makes that lookup a seek, newest first.
ALTER TABLE exec_runs ADD COLUMN graph_hash TEXT;
CREATE INDEX idx_exec_runs_graph_hash
  ON exec_runs(graph_hash, status, finished_at)
  WHERE graph_hash IS NOT NULL;
"""

# ── Migration 016: a one-shot worker known before it reconnects (#845) ────
#
# ``workers.one_shot`` is 1 when the credential's own agent (``--name``,
# no ``--instance``) last said ``hello.one_shot`` (#790), and 0 when it
# was a regular agent. Instances of the credential (``--instance``) never
# write it: they are known in memory only, and a credential used only by
# instances stays 0 (feed an array through the one-shot pool, not by
# naming the credential). A server that restarted loads it with the row
# (``WorkerRegistry.load_known``), so a run naming the worker before its
# next agent dials in is timed by ``CODEFYUI_WORKER_ONE_SHOT_QUEUE_TIMEOUT_S``
# like any run waiting for a one-shot worker. NULL: never reported (every
# row from before), read as 0.
MIGRATION_016 = """
ALTER TABLE workers ADD COLUMN one_shot INTEGER;
"""

# ── Migration 017: node provenance (#344, hook 4) ────────────────────────
#
# What data and which weights a run read, one row per node that read them,
# for a run submitted with the ``provenance`` option
# (``core.node_provenance``). ``provenance`` is the node's record as JSON
# (source, revision, content identity, row count; unknown facts are
# explicit nulls). The primary key keeps one row per node: a replayed
# worker frame rewrites the row it already wrote. Deleted with its run
# (CASCADE), like the run's metrics and artifacts; the run's environment
# fingerprint is the part that outlives it.
MIGRATION_017 = """
CREATE TABLE exec_run_provenance (
  run_id      TEXT NOT NULL REFERENCES exec_runs(id) ON DELETE CASCADE,
  node_id     TEXT NOT NULL,
  node_type   TEXT NOT NULL,
  kind        TEXT NOT NULL,                -- 'dataset' | 'model'
  provenance  TEXT NOT NULL,                -- JSON, node_provenance's record
  recorded_at TEXT NOT NULL,
  PRIMARY KEY (run_id, node_id)
);
"""

# ── Migration 018: study budgets and plugin-owned studies (#343) ──────────
#
# - ``studies.owner``: the plugin that submitted the study through
#   ``/api/plugins/{plugin_id}/jobs`` (``routes_plugin_jobs``), or NULL for
#   a study submitted to ``/api/studies``. A plugin lists and controls only
#   its own studies there; ``/api/studies`` shows every study. A plugin's
#   idempotency keys are stored as ``plugin:<id>:<key>`` so two plugins
#   cannot collide on a key.
# - ``studies.halt``: JSON ``{budget, limit, used, unfinished_repetitions,
#   at}`` for a study that a server-enforced budget stopped (status
#   ``halted``); NULL otherwise, and cleared by a resume.
# - ``study_attempts.stop_reason``: the budget the coordinator stopped the
#   attempt's child for (``max_run_seconds`` or ``max_active_seconds``),
#   written BEFORE the cancel is sent, so the harvest of the child knows a
#   cancelled child was the study's own decision. NULL otherwise.
MIGRATION_018 = """
ALTER TABLE studies ADD COLUMN owner TEXT;
ALTER TABLE studies ADD COLUMN halt TEXT;
ALTER TABLE study_attempts ADD COLUMN stop_reason TEXT;
CREATE INDEX idx_studies_owner ON studies(owner, created_at)
  WHERE owner IS NOT NULL;
"""

MIGRATIONS: list[str] = [MIGRATION_001, MIGRATION_002, MIGRATION_003,
                         MIGRATION_004, MIGRATION_005, MIGRATION_006,
                         MIGRATION_007, MIGRATION_008, MIGRATION_009,
                         MIGRATION_010, MIGRATION_011,
                         MIGRATION_012, MIGRATION_013,
                         MIGRATION_014, MIGRATION_015,
                         MIGRATION_016, MIGRATION_017,
                         MIGRATION_018]


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
