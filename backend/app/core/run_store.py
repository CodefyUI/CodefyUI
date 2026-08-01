"""Persistence for graph-execution runs — the Run Service store (#119).

Storage only. No scheduling, no engine calls, no HTTP: this module owns the
``exec_runs`` / ``exec_run_metrics`` / ``exec_run_events`` /
``exec_run_artifacts`` tables created by ``migrations.MIGRATION_003`` and
nothing else. The naming decision behind the ``exec_`` prefix (there is an
unrelated, already-shipped ``runs`` table for published-app invocations)
lives in the comment block above ``MIGRATION_003``.

Not to be confused with ``run_output_store.RunOutputStore``, which is the
in-memory LRU of captured node outputs and is lost on restart. This one is
the durable half.

House-style note: the publish subsystem keeps its SQL inline in the route
handlers. A shared DAO is used here instead because five separate consumers
(run service, WS attach/replay, the metric logger, the Runs panel API and
sweeps) read and write the same rows, and a lifecycle spread across five
modules is exactly how status vocabularies drift apart.

Every method goes through ``Database.run`` — one sync function, one worker
thread, one shared ``asyncio.Lock``, one connection. See ``append_event``
for why that plus ``BEGIN IMMEDIATE`` is what makes the event cursor safe.

Each method is individually atomic; they do NOT compose. Two calls are two
transactions, and calling one from inside another's ``fn(conn)`` closure
deadlocks on the non-reentrant lock. When a caller needs several writes to
land together, that belongs in a new method with one ``Database.run`` —
``log_metrics`` is the existing example (a whole flush, one transaction).

Rows in, rows out
-----------------
Reads return frozen dataclasses, not ``sqlite3.Row``: the row objects are
tied to the connection's lifetime and would leak the storage layer into
five consumers. ``dataclasses.asdict`` covers the REST-serialisation side.
``RunRecord`` deliberately omits ``graph_snapshot`` — a run list must not
deserialize a megabyte of graph per row on every poll of the Runs panel;
fetch it explicitly with ``get_graph_snapshot``.
"""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Sequence
from uuid import uuid4

from ..config import settings
from .db import Database, transaction, utc_now_iso
from .plugin_loader import is_enabled, load_lockfile
from .project import git_provenance

# ── status vocabulary ─────────────────────────────────────────────────────
#
# The canonical definition, and the only enforcement point: the schema
# deliberately carries no CHECK constraint (SQLite cannot ALTER one away and
# the queue/sweep work may add states), so validation happens here, on the
# one module that writes the column.
#
# Distinct from the per-NODE status vocabulary on the WebSocket
# (running/completed/cached/skipped/error/progress) and from the publish
# envelope's two-valued ok/error — do not cross-map them by accident.
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_INTERRUPTED = "interrupted"

#: Not yet finished; cannot survive a server restart.
ACTIVE_STATUSES: frozenset[str] = frozenset({STATUS_QUEUED, STATUS_RUNNING})
#: Finished, one way or another. Only these are eligible for retention.
TERMINAL_STATUSES: frozenset[str] = frozenset({
    STATUS_SUCCEEDED, STATUS_FAILED, STATUS_CANCELLED, STATUS_INTERRUPTED,
})
RUN_STATUSES: frozenset[str] = ACTIVE_STATUSES | TERMINAL_STATUSES

# Known artifact kinds. Open vocabulary on purpose (a node pack may register
# its own); listed for discoverability, never validated.
ARTIFACT_KIND_CHECKPOINT = "checkpoint"
ARTIFACT_KIND_EXPORT = "export"
ARTIFACT_KIND_IMAGE = "image"

_RUN_COLUMNS = (
    "id, name, options, status, error, queue_key, created_at, started_at, "
    "finished_at, git_commit, git_dirty, plugin_pins"
)
_METRIC_COLUMNS = "run_id, node_id, name, step, value, ts"
_EVENT_COLUMNS = "run_id, cursor, type, payload, ts"
_ARTIFACT_COLUMNS = "id, run_id, kind, path, meta, created_at"


def _json_safe(value: Any) -> Any:
    """Replace every non-finite float with None, recursively."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _dumps(value: Any) -> str:
    """Compact JSON for a storage column. Always valid JSON.

    ``separators`` trims the whitespace sqlite would otherwise store for
    every graph snapshot and event payload. ``ensure_ascii`` stays at its
    default: escaping keeps a lone surrogate (legal in a JSON string that
    arrived over the wire) from raising ``UnicodeEncodeError`` on insert,
    and the round-trip is byte-identical either way.

    ``allow_nan=False`` is the load-bearing one. Python's default emits the
    bare tokens ``NaN`` / ``Infinity``, which are a CPython extension and
    NOT JSON — ``JSON.parse`` throws on them. A stored event payload is
    replayed verbatim to a browser on attach, so a diverged ``loss`` in a
    progress payload would otherwise poison the whole replay, remotely and
    long after the fact. The strict attempt runs first (no cost when the
    data is clean); only when it trips do we walk the structure and replace
    the offenders with null. Dropping the value beats dropping the event.
    """
    try:
        return json.dumps(value, separators=(",", ":"), allow_nan=False)
    except ValueError:
        return json.dumps(_json_safe(value), separators=(",", ":"),
                          allow_nan=False)


def _loads(raw: str | None) -> Any:
    return None if raw is None else json.loads(raw)


def _tri_state(value: int | None) -> bool | None:
    """SQLite 0/1/NULL -> False/True/None (NULL means *unknown*, not False)."""
    return None if value is None else bool(value)


# ── records ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RunRecord:
    """One ``exec_runs`` row, minus ``graph_snapshot`` (see module docstring)."""

    id: str
    name: str | None
    options: dict[str, Any]
    status: str
    error: str | None
    queue_key: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None
    git_commit: str | None
    git_dirty: bool | None
    plugin_pins: dict[str, Any] | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "RunRecord":
        # `or {}` would quietly turn a corrupt 0 / "" / [] into a plausible
        # empty options dict; an explicit type check keeps damage loud.
        options = _loads(row["options"])
        if not isinstance(options, dict):
            raise ValueError(
                f"exec_runs.options for run {row['id']!r} is "
                f"{type(options).__name__}, expected a JSON object")
        return cls(
            id=row["id"],
            name=row["name"],
            options=options,
            status=row["status"],
            error=row["error"],
            queue_key=row["queue_key"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            git_commit=row["git_commit"],
            git_dirty=_tri_state(row["git_dirty"]),
            plugin_pins=_loads(row["plugin_pins"]),
        )


@dataclass(frozen=True)
class MetricRecord:
    """One stored point of a named series. The read side of ``MetricPoint``.

    ``value`` is None when the producer logged a non-finite number — a
    diverged ``train_loss`` is NaN, and neither sqlite nor JSON can carry
    NaN or +/-inf. The point is still recorded at its step so a chart shows
    a break exactly where the training went bad, instead of the series
    silently ending there.
    """

    run_id: str
    node_id: str | None
    name: str
    step: int
    value: float | None
    ts: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "MetricRecord":
        return cls(run_id=row["run_id"], node_id=row["node_id"],
                   name=row["name"], step=row["step"], value=row["value"],
                   ts=row["ts"])


class MetricPoint(NamedTuple):
    """The write side: one point to append, before the store stamps it.

    Separate from ``MetricRecord`` so the high-frequency producer can build
    a flush batch without inventing a ``run_id``/``ts`` per point.
    """

    name: str
    value: float
    step: int
    node_id: str | None = None
    ts: str | None = None


@dataclass(frozen=True)
class EventRecord:
    """One entry of a run's replay log, addressed by its per-run cursor."""

    run_id: str
    cursor: int
    type: str
    payload: Any
    ts: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "EventRecord":
        return cls(run_id=row["run_id"], cursor=row["cursor"],
                   type=row["type"], payload=_loads(row["payload"]),
                   ts=row["ts"])


@dataclass(frozen=True)
class ArtifactRecord:
    """A file a run produced: checkpoint, export, image."""

    id: int
    run_id: str
    kind: str
    path: str
    meta: Any
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ArtifactRecord":
        return cls(id=row["id"], run_id=row["run_id"], kind=row["kind"],
                   path=row["path"], meta=_loads(row["meta"]),
                   created_at=row["created_at"])


@dataclass(frozen=True)
class RunProvenance:
    """What the workspace looked like when a run was submitted.

    All three fields are tri-state: ``None`` means *not known*, and is
    stored as SQL NULL rather than guessed — the same rule
    ``app_versions.git_commit`` follows.
    """

    git_commit: str | None = None
    git_dirty: bool | None = None
    plugin_pins: dict[str, dict[str, Any]] | None = None

    @classmethod
    def capture(cls, project_dir: Path | str | None = None) -> "RunProvenance":
        """Snapshot the current workspace. Blocking — call in a thread.

        Runs two ``git`` subprocesses (via ``project.git_provenance``) and
        reads the plugin lockfile, so ``create_run`` dispatches it through
        ``asyncio.to_thread`` rather than calling it on the event loop.

        *project_dir* defaults to ``settings.PROJECT_DIR``. Outside project
        mode that is ``None`` and git is skipped entirely: the server's own
        checkout is not the user's project, so reporting its commit would be
        a lie, and spawning git per run to learn nothing is pure cost.
        """
        directory = project_dir if project_dir is not None \
            else settings.PROJECT_DIR
        commit: str | None = None
        dirty: bool | None = None
        if directory is not None:
            commit, dirty = git_provenance(Path(directory))
        return cls(git_commit=commit, git_dirty=dirty,
                   plugin_pins=_capture_plugin_pins())


def _capture_plugin_pins() -> dict[str, dict[str, Any]]:
    """Which plugins were active for this run, from the install lockfile.

    Related to but deliberately different from ``cdui project freeze``:
    freeze emits only what can be re-installed (github pins with a sha) and
    skips builtin/local entries. A run snapshot answers "what was loaded",
    so builtin and local packs are included with whatever identity they
    have. Disabled entries are skipped — they were not loaded.

    Never raises: ``load_lockfile`` already degrades a missing or corrupt
    file to an empty lockfile, and a malformed entry is skipped rather than
    failing the run that was only trying to describe itself.
    """
    plugins = load_lockfile().get("plugins")
    if not isinstance(plugins, dict):
        return {}
    pins: dict[str, dict[str, Any]] = {}
    for plugin_id, entry in plugins.items():
        if not isinstance(entry, dict) or not is_enabled(entry):
            continue
        manifest = entry.get("manifest")
        pins[str(plugin_id)] = {
            "source_kind": str(entry.get("source_kind") or ""),
            "source": str(entry.get("source") or ""),
            "ref": str(entry.get("ref") or ""),
            "sha": str(entry.get("sha") or ""),
            "version": (manifest.get("version")
                        if isinstance(manifest, dict) else None),
        }
    return pins


# ── the store ─────────────────────────────────────────────────────────────


@dataclass
class RunStore:
    """Typed CRUD over the four ``exec_*`` tables.

    Holds no state of its own beyond the ``Database`` it was handed, so it
    is safe to construct per request or once on ``app.state``.
    """

    db: Database

    # ── runs ──────────────────────────────────────────────────────────────

    async def create_run(
        self,
        *,
        graph_snapshot: dict[str, Any],
        options: dict[str, Any] | None = None,
        name: str | None = None,
        status: str = STATUS_QUEUED,
        queue_key: str | None = None,
        run_id: str | None = None,
        provenance: RunProvenance | None = None,
    ) -> RunRecord:
        """Insert a run and return its row.

        *run_id* defaults to a fresh ``uuid4().hex`` (the shape the publish
        ``runs.run_id`` and the WS envelope already use). Passing one is for
        callers that minted the id earlier — a duplicate raises
        ``sqlite3.IntegrityError`` rather than overwriting.

        *provenance* defaults to a fresh ``RunProvenance.capture()`` in a
        worker thread. Pass an explicit one (even an empty
        ``RunProvenance()``) to skip the git/lockfile work.
        """
        if status not in RUN_STATUSES:
            raise ValueError(
                f"unknown run status {status!r}; expected one of "
                f"{sorted(RUN_STATUSES)}")
        if provenance is None:
            provenance = await asyncio.to_thread(RunProvenance.capture)

        record = RunRecord(
            id=run_id or uuid4().hex,
            name=name,
            options=options if options is not None else {},
            status=status,
            error=None,
            queue_key=queue_key,
            created_at=utc_now_iso(),
            started_at=None,
            finished_at=None,
            git_commit=provenance.git_commit,
            git_dirty=provenance.git_dirty,
            plugin_pins=provenance.plugin_pins,
        )
        params = (
            record.id, record.name, _dumps(graph_snapshot),
            _dumps(record.options), record.status, record.queue_key,
            record.created_at, record.git_commit,
            None if record.git_dirty is None else int(record.git_dirty),
            None if record.plugin_pins is None else _dumps(record.plugin_pins),
        )

        def _insert(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO exec_runs (id, name, graph_snapshot, options, "
                "status, queue_key, created_at, git_commit, git_dirty, "
                "plugin_pins) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params,
            )

        await self.db.run(_insert)
        return record

    async def get_run(self, run_id: str) -> RunRecord | None:
        def _select(conn: sqlite3.Connection) -> sqlite3.Row | None:
            return conn.execute(
                f"SELECT {_RUN_COLUMNS} FROM exec_runs WHERE id = ?",
                (run_id,),
            ).fetchone()

        row = await self.db.run(_select)
        return None if row is None else RunRecord.from_row(row)

    async def get_graph_snapshot(self, run_id: str) -> dict[str, Any] | None:
        """The submitted graph, fetched on demand (not part of ``RunRecord``)."""
        def _select(conn: sqlite3.Connection) -> str | None:
            row = conn.execute(
                "SELECT graph_snapshot FROM exec_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            return None if row is None else row["graph_snapshot"]

        raw = await self.db.run(_select)
        return None if raw is None else json.loads(raw)

    async def list_runs(
        self,
        *,
        status: str | Sequence[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RunRecord]:
        """Newest first. Ties break on insertion order, never scan order.

        *status* accepts one status or several (the Runs panel asks for
        ``queued|running`` on mount). An unknown status raises rather than
        quietly returning nothing; an EMPTY sequence is the honest empty
        filter and returns nothing (``IN ()`` is not valid SQL).

        A single-status filter is fully index-ordered; a multi-status one
        merges two index ranges and pays a sort. That is accepted — the
        query exists to find the handful of runs still in flight.
        """
        where, params = self._status_filter(status)
        if where is None:
            return []
        params += [limit, offset]

        def _select(conn: sqlite3.Connection) -> list[sqlite3.Row]:
            return conn.execute(
                f"SELECT {_RUN_COLUMNS} FROM exec_runs{where} "
                "ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                params,
            ).fetchall()

        return [RunRecord.from_row(r) for r in await self.db.run(_select)]

    async def count_runs(
        self, *, status: str | Sequence[str] | None = None,
    ) -> int:
        """Total matching runs, ignoring limit/offset.

        The paging companion to ``list_runs``: a table cannot render
        "1-50 of 213" or size a scrollbar from a page alone.
        """
        where, params = self._status_filter(status)
        if where is None:
            return 0

        def _count(conn: sqlite3.Connection) -> int:
            return conn.execute(
                f"SELECT COUNT(*) FROM exec_runs{where}", params).fetchone()[0]

        return await self.db.run(_count)

    @staticmethod
    def _status_filter(
        status: str | Sequence[str] | None,
    ) -> tuple[str | None, list[Any]]:
        """(WHERE fragment, params) for a status filter.

        Returns ``(None, [])`` for an explicitly EMPTY filter — ``IN ()`` is
        not valid SQL, so callers short-circuit to an empty result instead.
        """
        if status is None:
            return "", []
        wanted = (status,) if isinstance(status, str) else tuple(status)
        if not wanted:
            return None, []
        unknown = sorted(set(wanted) - RUN_STATUSES)
        if unknown:
            raise ValueError(
                f"unknown run status {unknown}; expected one of "
                f"{sorted(RUN_STATUSES)}")
        return f" WHERE status IN ({','.join('?' * len(wanted))})", list(wanted)

    async def mark_running(
        self, run_id: str, *, queue_key: str | None = None,
    ) -> bool:
        """Move a run to ``running``. False when no such run exists.

        ``started_at`` is written once (COALESCE): a re-mark — a resumed or
        retried run — must not erase when the work actually began. A
        *queue_key* of None likewise leaves any previously resolved device
        in place.
        """
        stamp = utc_now_iso()

        def _update(conn: sqlite3.Connection) -> int:
            return conn.execute(
                "UPDATE exec_runs SET status = ?, "
                "started_at = COALESCE(started_at, ?), "
                "queue_key = COALESCE(?, queue_key) WHERE id = ?",
                (STATUS_RUNNING, stamp, queue_key, run_id),
            ).rowcount

        return await self.db.run(_update) > 0

    async def mark_finished(
        self, run_id: str, status: str, *, error: str | None = None,
    ) -> bool:
        """Move a run to a terminal status and stamp ``finished_at``.

        *error* is written as given, including None — the finishing caller
        holds the authoritative outcome. ``started_at`` is left alone, so a
        queued run cancelled before it ever ran keeps a NULL there.
        """
        if status not in TERMINAL_STATUSES:
            raise ValueError(
                f"{status!r} is not a terminal status; expected one of "
                f"{sorted(TERMINAL_STATUSES)}")
        stamp = utc_now_iso()

        def _update(conn: sqlite3.Connection) -> int:
            return conn.execute(
                "UPDATE exec_runs SET status = ?, error = ?, finished_at = ? "
                "WHERE id = ?",
                (status, error, stamp, run_id),
            ).rowcount

        return await self.db.run(_update) > 0

    async def interrupt_active_runs(
        self, *, statuses: Sequence[str] = tuple(sorted(ACTIVE_STATUSES)),
    ) -> int:
        """Crash recovery: retire rows no process is driving any more.

        Call once at startup. Both active states are covered by default —
        nothing resumes a ``running`` row after a restart, and a ``queued``
        row would otherwise wait forever on a scheduler that lost its queue.
        Narrow it with *statuses* if a caller can genuinely re-adopt one.
        Idempotent: a second call finds nothing.

        *statuses* is validated against ACTIVE_STATUSES, not the full
        vocabulary: this rewrites ``status`` AND ``finished_at``, so letting
        a terminal state through would let a startup call silently falsify
        completed history.
        """
        wanted = tuple(statuses)
        if not wanted:
            return 0
        unknown = sorted(set(wanted) - ACTIVE_STATUSES)
        if unknown:
            raise ValueError(
                f"{unknown} is not an active status; expected a subset of "
                f"{sorted(ACTIVE_STATUSES)}")
        stamp = utc_now_iso()

        def _update(conn: sqlite3.Connection) -> int:
            return conn.execute(
                "UPDATE exec_runs SET status = ?, finished_at = ? "
                f"WHERE status IN ({','.join('?' * len(wanted))})",
                (STATUS_INTERRUPTED, stamp, *wanted),
            ).rowcount

        return await self.db.run(_update)

    # ── events ────────────────────────────────────────────────────────────

    async def append_event(
        self,
        run_id: str,
        event_type: str,
        payload: Any = None,
        *,
        ts: str | None = None,
    ) -> int:
        """Append to a run's replay log; return the cursor it landed on.

        Cursors are per run, 1-based, gapless and strictly increasing, which
        is what lets a re-attaching client say "everything after N" and get
        back exactly the events it missed, once each.

        Why that holds under concurrent writers
        ---------------------------------------
        Allocating a cursor is a read-modify-write (``MAX(cursor) + 1`` then
        INSERT), the classic lost-update shape. Three things stack:

        1. In-process serialisation. Every call funnels through
           ``Database.run``, which holds ONE ``asyncio.Lock`` for the whole
           ``asyncio.to_thread`` hop onto ONE connection — including when
           the caller is CANCELLED mid-call, which is the case that used to
           hand the connection over with a transaction still open (see
           ``Database.run``). Gathered coroutines cannot interleave their
           statements. ``test_cursors_are_gapless_and_monotonic_under_
           concurrent_writers`` pins this layer, plus the cursor-to-caller
           mapping; note it CANNOT detect a missing transaction, because
           the lock alone already serialises the 100 writers.
        2. Cross-connection serialisation. ``BEGIN IMMEDIATE`` takes the
           write lock BEFORE the SELECT, so even a second connection on the
           same file (another process, a future pool) cannot commit between
           this read and this write. DEFERRED would not: it would upgrade
           the lock only at the INSERT, having already read a stale MAX.
           Pinned by ``test_append_event_reads_and_writes_in_one_immediate_
           transaction`` here and by
           ``test_transaction_takes_the_write_lock_up_front`` in test_db.
        3. A constraint backstop. ``UNIQUE (run_id, cursor)`` means that if
           1 and 2 were ever both defeated, the loser raises
           ``IntegrityError`` — a failed append, never two events silently
           sharing a cursor and a replay that skips one of them.

        Not covered: an append whose caller is cancelled after the COMMIT
        still leaves a durable event, and the cursor is lost with the
        raise. A producer that retries must expect that its previous
        attempt may already be in the log.

        *payload* is any JSON-serialisable value; None stores SQL NULL.
        Pass *ts* when the caller needs the exact stamp it will fan out to
        subscribers, otherwise the store stamps it.
        """
        stamp = ts or utc_now_iso()
        encoded = None if payload is None else _dumps(payload)

        def _append(conn: sqlite3.Connection) -> int:
            with transaction(conn):
                cursor = conn.execute(
                    "SELECT COALESCE(MAX(cursor), 0) + 1 FROM exec_run_events "
                    "WHERE run_id = ?", (run_id,),
                ).fetchone()[0]
                conn.execute(
                    "INSERT INTO exec_run_events (run_id, cursor, type, "
                    "payload, ts) VALUES (?, ?, ?, ?, ?)",
                    (run_id, cursor, event_type, encoded, stamp),
                )
                return cursor

        return await self.db.run(_append)

    async def get_events(
        self, run_id: str, *, after_cursor: int = 0, limit: int | None = None,
    ) -> list[EventRecord]:
        """Replay in cursor order, strictly after *after_cursor*.

        ``after_cursor=0`` is the whole log (cursors start at 1), which is
        what a fresh attach asks for.
        """
        sql = (f"SELECT {_EVENT_COLUMNS} FROM exec_run_events "
               "WHERE run_id = ? AND cursor > ? ORDER BY cursor")
        params: list[Any] = [run_id, after_cursor]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        def _select(conn: sqlite3.Connection) -> list[sqlite3.Row]:
            return conn.execute(sql, params).fetchall()

        return [EventRecord.from_row(r) for r in await self.db.run(_select)]

    async def latest_cursor(self, run_id: str) -> int:
        """Highest cursor issued for *run_id*; 0 when it has no events.

        Because cursors are gapless this doubles as the event count.
        """
        def _select(conn: sqlite3.Connection) -> int:
            return conn.execute(
                "SELECT COALESCE(MAX(cursor), 0) FROM exec_run_events "
                "WHERE run_id = ?", (run_id,),
            ).fetchone()[0]

        return await self.db.run(_select)

    # ── metrics ───────────────────────────────────────────────────────────

    async def log_metric(
        self,
        run_id: str,
        name: str,
        value: float,
        step: int,
        *,
        node_id: str | None = None,
        ts: str | None = None,
    ) -> None:
        """Append one point. Prefer ``log_metrics`` for anything in a loop."""
        await self.log_metrics(
            run_id, [MetricPoint(name, value, step, node_id, ts)])

    async def log_metrics(
        self, run_id: str, points: Iterable[MetricPoint],
    ) -> int:
        """Append a batch in ONE transaction; return how many landed.

        The training-loop consumer flushes bursts, and per-point
        transactions would mean one fsync each. All-or-nothing on purpose: a
        bad point rolls the whole flush back rather than leaving a partial
        one behind for a chart to interpolate through.

        A non-finite value (NaN from a diverged loss, +/-inf) is stored as
        NULL and reads back as ``MetricRecord.value is None``. It must not
        be an error: all-or-nothing means one NaN would otherwise discard
        every good point flushed alongside it, at exactly the moment the
        user most needs to see the chart.
        """
        rows = [
            (run_id, p.node_id, p.name, p.step,
             p.value if math.isfinite(p.value) else None,
             p.ts or utc_now_iso())
            for p in points
        ]
        if not rows:
            return 0

        def _insert(conn: sqlite3.Connection) -> int:
            with transaction(conn):
                return conn.executemany(
                    "INSERT INTO exec_run_metrics (run_id, node_id, name, "
                    "step, value, ts) VALUES (?, ?, ?, ?, ?, ?)",
                    rows,
                ).rowcount

        return await self.db.run(_insert)

    async def get_metrics(
        self,
        run_id: str,
        *,
        name: str | None = None,
        after_step: int | None = None,
        limit: int | None = None,
    ) -> list[MetricRecord]:
        """Points ordered ``(name, step)`` — chart order, and index order.

        *limit* requires *name*. The ordering groups by series, so a raw row
        cap over several interleaved series would return the whole
        alphabetically-first one and NOTHING of the rest, with no signal
        that anything was dropped. Cap a series, not a run.
        """
        if limit is not None and name is None:
            raise ValueError(
                "limit requires name: capping across series would silently "
                "drop whole series (order is name, step)")
        sql = (f"SELECT {_METRIC_COLUMNS} FROM exec_run_metrics "
               "WHERE run_id = ?")
        params: list[Any] = [run_id]
        if name is not None:
            sql += " AND name = ?"
            params.append(name)
        if after_step is not None:
            sql += " AND step > ?"
            params.append(after_step)
        sql += " ORDER BY name, step, id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        def _select(conn: sqlite3.Connection) -> list[sqlite3.Row]:
            return conn.execute(sql, params).fetchall()

        return [MetricRecord.from_row(r) for r in await self.db.run(_select)]

    async def list_metric_names(self, run_id: str) -> list[str]:
        """Distinct series names for a run — the chart legend."""
        def _select(conn: sqlite3.Connection) -> list[sqlite3.Row]:
            return conn.execute(
                "SELECT DISTINCT name FROM exec_run_metrics "
                "WHERE run_id = ? ORDER BY name", (run_id,),
            ).fetchall()

        return [r["name"] for r in await self.db.run(_select)]

    # ── artifacts ─────────────────────────────────────────────────────────

    async def add_artifact(
        self, run_id: str, kind: str, path: str, *, meta: Any = None,
    ) -> ArtifactRecord:
        """Register a file a run produced. *kind* is an open vocabulary."""
        created_at = utc_now_iso()
        encoded = None if meta is None else _dumps(meta)

        def _insert(conn: sqlite3.Connection) -> int:
            cur = conn.execute(
                "INSERT INTO exec_run_artifacts (run_id, kind, path, meta, "
                "created_at) VALUES (?, ?, ?, ?, ?)",
                (run_id, kind, path, encoded, created_at),
            )
            # lastrowid is Optional in the driver's typing; after a
            # successful single-row INSERT it is always the new rowid.
            return int(cur.lastrowid or 0)

        artifact_id = await self.db.run(_insert)
        return ArtifactRecord(id=artifact_id, run_id=run_id, kind=kind,
                              path=path, meta=meta, created_at=created_at)

    async def list_artifacts(
        self, run_id: str, *, kind: str | None = None,
    ) -> list[ArtifactRecord]:
        sql = (f"SELECT {_ARTIFACT_COLUMNS} FROM exec_run_artifacts "
               "WHERE run_id = ?")
        params: list[Any] = [run_id]
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY created_at, id"

        def _select(conn: sqlite3.Connection) -> list[sqlite3.Row]:
            return conn.execute(sql, params).fetchall()

        return [ArtifactRecord.from_row(r) for r in await self.db.run(_select)]

    # ── retention ─────────────────────────────────────────────────────────

    async def delete_run(self, run_id: str) -> bool:
        """Delete one run. Metrics, events and artifacts cascade with it.

        The cascade is the schema's (``ON DELETE CASCADE`` +
        ``PRAGMA foreign_keys=ON`` from ``Database.connect``), not a manual
        three-table sweep that a future table would silently escape.
        Artifact FILES are left on disk — this store never owned them.
        """
        def _delete(conn: sqlite3.Connection) -> int:
            return conn.execute(
                "DELETE FROM exec_runs WHERE id = ?", (run_id,)).rowcount

        return await self.db.run(_delete) > 0

    async def prune(self, *, keep_last: int) -> int:
        """Keep the *keep_last* newest runs, delete the older ones.

        Returns how many rows went (children cascade). Two rules:

        - Active runs are never deleted. A queued or running row is live
          state, not history; retention that could delete one would be a
          scheduler bug generator.
        - Active runs still occupy slots in the keep window, so *keep_last*
          stays a straightforward cap on the table rather than a moving
          target that depends on how many runs happen to be in flight.

        ``keep_last=0`` therefore means "drop all finished runs", not "drop
        everything". Age-based retention (the publish table's
        ``Database.prune_runs`` model) can be added alongside this later;
        the two are independent policies over different tables.
        """
        if keep_last < 0:
            raise ValueError(f"keep_last must be >= 0, got {keep_last}")
        active = tuple(sorted(ACTIVE_STATUSES))
        params = (*active, keep_last)

        def _prune(conn: sqlite3.Connection) -> int:
            with transaction(conn):
                return conn.execute(
                    "DELETE FROM exec_runs WHERE status NOT IN "
                    f"({','.join('?' * len(active))}) AND rowid NOT IN "
                    "(SELECT rowid FROM exec_runs "
                    "ORDER BY created_at DESC, rowid DESC LIMIT ?)",
                    params,
                ).rowcount

        return await self.db.run(_prune)
