"""Durable study rows (#343): studies, their variants, attempts and events.

Built to ``run_store``'s rules, for the same reasons: every method goes
through ``Database.run``; each method is individually atomic; **methods do
not compose** (calling one from inside another's ``fn(conn)`` closure
deadlocks on the non-reentrant lock). Reads return frozen dataclasses.

Every write that changes a study or one of its attempts does three things in
ONE transaction: the change itself, ``revision = revision + 1`` on the study
row, and an entry in ``study_events``. That is what makes the event log an
audit trail rather than a best-effort notification stream: there is no state
a reader can observe that the log does not explain, and no log entry for a
change that did not land.

The coordinator (``core.study_coordinator``) decides WHAT to write; this
module only knows how to write it atomically and how to read it back. The
ranking rule is here too, as a pure function, because it is a property of
the stored evidence and nothing else.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from .db import Database, transaction, utc_now_iso
from .run_store import (
    STATUS_QUEUED,
    STATUS_RUNNING,
    TERMINAL_STATUSES,
    RunRecord,
    last_metric_values,
    producer_last_value,
)

logger = logging.getLogger(__name__)

# ── the study state machine ───────────────────────────────────────────────
#
#   queued -> running -> succeeded            every scheduled repetition ended
#                     -> failed               the coordinator could not finish
#                                             the schedule (a submit refused,
#                                             the run budget exhausted)
#                     -> cancel_requested -> cancelled
#                     -> interrupted          a server restart, or a child
#                                             stopped outside the study
#   interrupted | cancelled | failed -> running   explicit resume
#
# ``succeeded`` is about the SCHEDULE, not the training: a study whose
# repetitions all ran is ``succeeded`` even if some of them failed, and the
# failed repetitions make their variant ineligible for ranking. ``failed`` is
# reserved for the study itself breaking, the way a sweep's ``failed`` is.
STUDY_QUEUED = "queued"
STUDY_RUNNING = "running"
STUDY_CANCEL_REQUESTED = "cancel_requested"
STUDY_SUCCEEDED = "succeeded"
STUDY_FAILED = "failed"
STUDY_CANCELLED = "cancelled"
STUDY_INTERRUPTED = "interrupted"

STUDY_ACTIVE: frozenset[str] = frozenset({
    STUDY_QUEUED, STUDY_RUNNING, STUDY_CANCEL_REQUESTED})
STUDY_RESUMABLE: frozenset[str] = frozenset({
    STUDY_INTERRUPTED, STUDY_CANCELLED, STUDY_FAILED})
STUDY_STATUSES: frozenset[str] = STUDY_ACTIVE | STUDY_RESUMABLE | {
    STUDY_SUCCEEDED}

# ── the attempt state machine ─────────────────────────────────────────────
#
#   pending -> submitting -> queued -> running -> succeeded | failed
#                                              -> cancelled | interrupted
#           submitting -> lost       the submission never created a run
#
# ``submitting`` is written WITH the run id the coordinator is about to use,
# before the child is submitted: the linkage exists before the child does.
ATTEMPT_PENDING = "pending"
ATTEMPT_SUBMITTING = "submitting"
ATTEMPT_QUEUED = STATUS_QUEUED
ATTEMPT_RUNNING = STATUS_RUNNING
ATTEMPT_SUCCEEDED = "succeeded"
ATTEMPT_FAILED = "failed"
ATTEMPT_CANCELLED = "cancelled"
ATTEMPT_INTERRUPTED = "interrupted"
ATTEMPT_LOST = "lost"

#: A child exists or may exist and is not finished.
ATTEMPT_ACTIVE: frozenset[str] = frozenset({
    ATTEMPT_SUBMITTING, ATTEMPT_QUEUED, ATTEMPT_RUNNING})
#: The repetition is DONE: its outcome is evidence and it is never re-run.
ATTEMPT_FINAL: frozenset[str] = frozenset({ATTEMPT_SUCCEEDED, ATTEMPT_FAILED})
#: The repetition did not complete; a resume gives it a new attempt.
ATTEMPT_RETRYABLE: frozenset[str] = frozenset({
    ATTEMPT_CANCELLED, ATTEMPT_INTERRUPTED, ATTEMPT_LOST})
ATTEMPT_STATUSES: frozenset[str] = (
    ATTEMPT_ACTIVE | ATTEMPT_FINAL | ATTEMPT_RETRYABLE | {ATTEMPT_PENDING})

RESUME_EXPLICIT = "explicit"
RESUME_AUTO = "auto"
RESUME_POLICIES: frozenset[str] = frozenset({RESUME_EXPLICIT, RESUME_AUTO})

DIRECTION_MINIMIZE = "minimize"
DIRECTION_MAXIMIZE = "maximize"

_STUDY_COLUMNS = (
    "id, name, status, revision, objective, repetitions, seed, options, "
    "budget, resume_policy, idempotency_key, request_digest, error, "
    "created_at, updated_at, finished_at")
_ATTEMPT_COLUMNS = (
    "id, study_id, variant_index, repetition, attempt, seed, run_id, status, "
    "error, created_at, submitted_at, started_at, finished_at, "
    "objective_value, final_metrics, fingerprint_digest, harvested_at")
_EVENT_COLUMNS = "study_id, cursor, type, payload, ts"


class IdempotencyConflict(ValueError):
    """An idempotency key reused for a different request. REST maps to 409."""

    def __init__(self, key: str, study_id: str) -> None:
        super().__init__(
            f"idempotency key {key!r} was already used for study {study_id} "
            "with a different request; use a new key for a new study")
        self.key = key
        self.study_id = study_id


class StudyStateError(ValueError):
    """A transition the study's current status does not allow. -> 409."""


def _dumps(value: Any) -> str:
    """Compact JSON; NaN refused (a study row is durable, see sweep_store)."""
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def _loads(raw: str | None) -> Any:
    return None if raw is None else json.loads(raw)


def _finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) \
            and math.isfinite(value):
        return float(value)
    return None


# ── records ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StudyRecord:
    id: str
    name: str | None
    status: str
    revision: int
    objective: dict[str, Any]
    repetitions: int
    seed: int
    options: dict[str, Any]
    budget: dict[str, Any]
    resume_policy: str
    idempotency_key: str | None
    request_digest: str
    error: str | None
    created_at: str
    updated_at: str
    finished_at: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "StudyRecord":
        return cls(
            id=row["id"], name=row["name"], status=row["status"],
            revision=row["revision"], objective=_loads(row["objective"]),
            repetitions=row["repetitions"], seed=row["seed"],
            options=_loads(row["options"]), budget=_loads(row["budget"]),
            resume_policy=row["resume_policy"],
            idempotency_key=row["idempotency_key"],
            request_digest=row["request_digest"], error=row["error"],
            created_at=row["created_at"], updated_at=row["updated_at"],
            finished_at=row["finished_at"])


@dataclass(frozen=True)
class StudyVariantRecord:
    study_id: str
    index: int
    label: str
    ops: list[dict[str, Any]] | None


@dataclass(frozen=True)
class AttemptRecord:
    id: str
    study_id: str
    variant_index: int
    repetition: int
    attempt: int
    seed: int
    run_id: str | None
    status: str
    error: str | None
    created_at: str
    submitted_at: str | None
    started_at: str | None
    finished_at: str | None
    objective_value: float | None
    final_metrics: dict[str, float] | None
    fingerprint_digest: str | None
    harvested_at: str | None

    @property
    def slot(self) -> tuple[int, int]:
        return (self.variant_index, self.repetition)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "AttemptRecord":
        return cls(
            id=row["id"], study_id=row["study_id"],
            variant_index=row["variant_index"],
            repetition=row["repetition"], attempt=row["attempt"],
            seed=row["seed"], run_id=row["run_id"], status=row["status"],
            error=row["error"], created_at=row["created_at"],
            submitted_at=row["submitted_at"], started_at=row["started_at"],
            finished_at=row["finished_at"],
            objective_value=row["objective_value"],
            final_metrics=_loads(row["final_metrics"]),
            fingerprint_digest=row["fingerprint_digest"],
            harvested_at=row["harvested_at"])

    def as_json(self) -> dict[str, Any]:
        return {
            "attempt_id": self.id, "variant_index": self.variant_index,
            "repetition": self.repetition, "attempt": self.attempt,
            "seed": self.seed, "run_id": self.run_id, "status": self.status,
            "error": self.error, "created_at": self.created_at,
            "submitted_at": self.submitted_at, "started_at": self.started_at,
            "finished_at": self.finished_at,
            "objective_value": self.objective_value,
            "final_metrics": self.final_metrics,
            "fingerprint_digest": self.fingerprint_digest,
            "harvested_at": self.harvested_at,
        }


@dataclass(frozen=True)
class StudyEventRecord:
    study_id: str
    cursor: int
    type: str
    payload: Any
    ts: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "StudyEventRecord":
        return cls(study_id=row["study_id"], cursor=row["cursor"],
                   type=row["type"], payload=_loads(row["payload"]),
                   ts=row["ts"])


@dataclass(frozen=True)
class NewVariant:
    """One resolved variant, as the route hands it to ``create_study``."""

    label: str
    ops: list[dict[str, Any]] | None
    graph: dict[str, Any]


# ── connection-level helpers (usable inside any open transaction) ────────


def _select_study(conn: sqlite3.Connection,
                  study_id: str) -> StudyRecord | None:
    row = conn.execute(f"SELECT {_STUDY_COLUMNS} FROM studies WHERE id = ?",
                       (study_id,)).fetchone()
    return None if row is None else StudyRecord.from_row(row)


def _select_attempts(conn: sqlite3.Connection,
                     study_id: str) -> list[AttemptRecord]:
    return [AttemptRecord.from_row(row) for row in conn.execute(
        f"SELECT {_ATTEMPT_COLUMNS} FROM study_attempts WHERE study_id = ? "
        "ORDER BY variant_index, repetition, attempt", (study_id,))]


def _touch(conn: sqlite3.Connection, study_id: str, stamp: str) -> int:
    """``revision + 1`` and ``updated_at``; returns the new revision."""
    conn.execute(
        "UPDATE studies SET revision = revision + 1, updated_at = ? "
        "WHERE id = ?", (stamp, study_id))
    row = conn.execute("SELECT revision FROM studies WHERE id = ?",
                       (study_id,)).fetchone()
    return 0 if row is None else row["revision"]


def _event(conn: sqlite3.Connection, study_id: str, event_type: str,
           payload: dict[str, Any], stamp: str) -> int:
    """Bump the revision and append one event; returns its cursor.

    The payload always carries the revision the change produced, so a log
    reader can line an event up with the ``GET`` snapshot it explains.
    """
    revision = _touch(conn, study_id, stamp)
    cursor = conn.execute(
        "SELECT COALESCE(MAX(cursor), 0) + 1 FROM study_events "
        "WHERE study_id = ?", (study_id,)).fetchone()[0]
    conn.execute(
        "INSERT INTO study_events (study_id, cursor, type, payload, ts) "
        "VALUES (?, ?, ?, ?, ?)",
        (study_id, cursor, event_type,
         json.dumps({**payload, "revision": revision},
                    separators=(",", ":"), default=str), stamp))
    return cursor


def _attempt_ref(attempt: AttemptRecord | sqlite3.Row) -> dict[str, Any]:
    get = (lambda key: getattr(attempt, key)) \
        if isinstance(attempt, AttemptRecord) else (lambda key: attempt[key])
    return {"attempt_id": get("id"), "variant_index": get("variant_index"),
            "repetition": get("repetition"), "attempt": get("attempt")}


def _harvest_one(conn: sqlite3.Connection, attempt: sqlite3.Row,
                 run: sqlite3.Row, objective: Mapping[str, Any],
                 stamp: str) -> bool:
    """Copy a TERMINAL child's evidence onto its attempt. False = nothing
    to do (already harvested, or the child is not terminal yet).

    Matches by the child's stamp, so it also completes the linkage of an
    attempt whose ``run_id`` write was lost: ``run_id`` is (re)written from
    the child row. The objective is read exactly: the named producer's last
    point of the named series (``run_store.producer_last_value``), never a
    name-only collapse across producers.
    """
    if attempt["harvested_at"] is not None \
            or run["status"] not in TERMINAL_STATUSES:
        return False
    metric = objective.get("metric")
    node_id = objective.get("node_id")
    value = (producer_last_value(conn, run["id"], metric, node_id)
             if isinstance(metric, str) and isinstance(node_id, str)
             else None)
    metrics = last_metric_values(conn, run["id"])
    conn.execute(
        "UPDATE study_attempts SET run_id = ?, status = ?, error = ?, "
        "started_at = ?, finished_at = ?, objective_value = ?, "
        "final_metrics = ?, fingerprint_digest = ?, harvested_at = ? "
        "WHERE id = ? AND harvested_at IS NULL",
        (run["id"], run["status"], run["error"], run["started_at"],
         run["finished_at"], _finite(value), _dumps(metrics),
         run["fingerprint_digest"], stamp, attempt["id"]))
    _event(conn, attempt["study_id"], "attempt_finished", {
        **_attempt_ref(attempt), "run_id": run["id"],
        "status": run["status"], "objective_value": _finite(value),
        "error": run["error"]}, stamp)
    return True


_HARVEST_RUN_COLUMNS = (
    "id, status, error, started_at, finished_at, fingerprint_digest, "
    "study_id, study_attempt_id")


def _harvest_run_rows(conn: sqlite3.Connection,
                      runs: Iterable[sqlite3.Row], stamp: str) -> set[str]:
    """Harvest every terminal child in *runs*; returns the touched study
    ids. Isolated per attempt: one bad row costs only its own attempt."""
    touched: set[str] = set()
    objectives: dict[str, Mapping[str, Any] | None] = {}
    for run in runs:
        attempt_id = run["study_attempt_id"]
        if attempt_id is None:
            continue
        # A SAVEPOINT per attempt, so a failure part-way through one
        # harvest (after its UPDATE, before its event) rolls back that
        # attempt alone and never leaves a change without its log entry.
        conn.execute("SAVEPOINT study_harvest")
        try:
            attempt = conn.execute(
                f"SELECT {_ATTEMPT_COLUMNS} FROM study_attempts WHERE id = ?",
                (attempt_id,)).fetchone()
            if attempt is not None:
                study_id = attempt["study_id"]
                if study_id not in objectives:
                    study = _select_study(conn, study_id)
                    objectives[study_id] = None if study is None \
                        else study.objective
                objective = objectives[study_id]
                if objective is not None and _harvest_one(
                        conn, attempt, run, objective, stamp):
                    touched.add(study_id)
            conn.execute("RELEASE study_harvest")
        except Exception:
            conn.execute("ROLLBACK TO study_harvest")
            conn.execute("RELEASE study_harvest")
            logger.warning("study attempt %s: could not copy the evidence of "
                           "run %s", attempt_id, run["id"], exc_info=True)
    return touched


def harvest_doomed_attempts(conn: sqlite3.Connection, where_clause: str,
                            params: tuple) -> int:
    """Retention's backstop (``RunStore.prune``): copy the evidence of every
    study child about to be deleted onto its attempt, INSIDE prune's
    transaction and with prune's own WHERE clause, so no child can be
    deleted unharvested. Returns how many studies were touched.

    Normally a no-op: the coordinator's terminal listener has already
    harvested the child before ``_finalize`` reached retention.
    """
    runs = conn.execute(
        f"SELECT {_HARVEST_RUN_COLUMNS} FROM exec_runs "
        f"WHERE {where_clause} AND study_attempt_id IS NOT NULL",
        params).fetchall()
    if not runs:
        return 0
    return len(_harvest_run_rows(conn, runs, utc_now_iso()))


# ── the ranking rule ──────────────────────────────────────────────────────


def current_attempts(
    attempts: Iterable[AttemptRecord],
) -> dict[tuple[int, int], AttemptRecord]:
    """The newest attempt of every (variant, repetition) slot."""
    current: dict[tuple[int, int], AttemptRecord] = {}
    for attempt in attempts:
        held = current.get(attempt.slot)
        if held is None or attempt.attempt > held.attempt:
            current[attempt.slot] = attempt
    return current


def rank_study(
    variant_count: int, repetitions: int,
    attempts: Iterable[AttemptRecord], *, direction: str,
) -> list[dict[str, Any]]:
    """Per-variant aggregates and ranks, in variant-index order.

    Eligibility, ported from Graph Copilot's promotion rule and stricter
    than the Sweep v1 table on purpose: a variant is ranked only when EVERY
    one of its scheduled repetitions has a current attempt that SUCCEEDED
    and resolved a finite value for the study's exact objective (producer
    node + metric). A repetition that failed after logging a number makes
    its variant ineligible; its value is still reported as evidence.

    Aggregates are over the current attempts that have a value: ``count``,
    ``mean`` (``math.fsum``, so the result does not depend on summation
    order), and ``std``, the SAMPLE standard deviation (n - 1), None when
    fewer than two values exist.

    Ranks are competition ranks over the eligible means ("1, 1, 3"): a
    variant's rank is one plus the number of eligible variants with a
    STRICTLY better mean. Exactly equal means share a rank and each lists
    the others in ``tied_with``; there is no tie-break at run time, and the
    labelling does not depend on the order variants are listed in.
    """
    if direction not in (DIRECTION_MINIMIZE, DIRECTION_MAXIMIZE):
        raise ValueError(f"unknown objective direction {direction!r}")
    current = current_attempts(attempts)
    rows: list[dict[str, Any]] = []
    for index in range(variant_count):
        values: list[float | None] = []
        reasons: list[str] = []
        for repetition in range(repetitions):
            attempt = current.get((index, repetition))
            value = None if attempt is None else attempt.objective_value
            values.append(value)
            if attempt is None:
                reasons.append(f"repetition {repetition} has no attempt")
            elif attempt.status != ATTEMPT_SUCCEEDED:
                reasons.append(
                    f"repetition {repetition} is {attempt.status}")
            elif value is None:
                reasons.append(
                    f"repetition {repetition} did not record the objective")
        present = [v for v in values if v is not None]
        count = len(present)
        mean = math.fsum(present) / count if count else None
        std = None
        if count >= 2 and mean is not None:
            std = math.sqrt(math.fsum((v - mean) ** 2 for v in present)
                            / (count - 1))
        rows.append({
            "variant_index": index, "values": values, "count": count,
            "mean": mean, "std": std, "eligible": not reasons,
            "ineligible_reasons": reasons, "rank": None, "tie": False,
            "tied_with": [],
        })
    eligible = [row for row in rows if row["eligible"]]
    sign = -1.0 if direction == DIRECTION_MAXIMIZE else 1.0
    for row in eligible:
        better = sum(1 for other in eligible
                     if sign * other["mean"] < sign * row["mean"])
        row["rank"] = better + 1
        row["tied_with"] = [other["variant_index"] for other in eligible
                            if other is not row
                            and other["mean"] == row["mean"]]
        row["tie"] = bool(row["tied_with"])
    return rows


def _select_by_idempotency_key(conn: sqlite3.Connection,
                               key: str) -> StudyRecord | None:
    """The study created with idempotency *key*, or None. Shared by the
    lookup before a submit and the one inside ``create_study``'s insert
    transaction, so the two cannot disagree on what "the same key" is."""
    row = conn.execute(
        f"SELECT {_STUDY_COLUMNS} FROM studies WHERE idempotency_key = ?",
        (key,)).fetchone()
    return None if row is None else StudyRecord.from_row(row)


# ── the store ─────────────────────────────────────────────────────────────


@dataclass
class StudyStore:
    """Typed CRUD over the four study tables. Holds only its ``Database``."""

    db: Database

    async def create_study(
        self, *, name: str | None, objective: dict[str, Any],
        repetitions: int, seed: int, options: dict[str, Any],
        budget: dict[str, Any], resume_policy: str,
        idempotency_key: str | None, request_digest: str,
        variants: Sequence[NewVariant], repetition_seeds: Sequence[int],
    ) -> tuple[StudyRecord, bool]:
        """Insert a study with every variant and N x R pending attempts, in
        ONE transaction; returns ``(study, created)``.

        With an *idempotency_key* that already exists, nothing is written:
        the same *request_digest* returns the existing study with
        ``created=False``, a different one raises ``IdempotencyConflict``.
        The lookup and the insert share the ``BEGIN IMMEDIATE`` transaction,
        so two concurrent submits with one key cannot both create a study.
        """
        study_id = uuid4().hex
        stamp = utc_now_iso()

        def _insert(conn: sqlite3.Connection) -> tuple[StudyRecord, bool]:
            with transaction(conn):
                if idempotency_key is not None:
                    existing = _select_by_idempotency_key(conn,
                                                          idempotency_key)
                    if existing is not None:
                        if existing.request_digest != request_digest:
                            raise IdempotencyConflict(idempotency_key,
                                                      existing.id)
                        return existing, False
                conn.execute(
                    "INSERT INTO studies (id, name, status, revision, "
                    "objective, repetitions, seed, options, budget, "
                    "resume_policy, idempotency_key, request_digest, "
                    "created_at, updated_at) VALUES "
                    "(?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (study_id, name, STUDY_QUEUED, _dumps(objective),
                     repetitions, seed, _dumps(options), _dumps(budget),
                     resume_policy, idempotency_key, request_digest, stamp,
                     stamp))
                for index, variant in enumerate(variants):
                    conn.execute(
                        "INSERT INTO study_variants (study_id, "
                        "variant_index, label, ops, graph_snapshot) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (study_id, index, variant.label,
                         None if variant.ops is None else _dumps(variant.ops),
                         json.dumps(variant.graph, separators=(",", ":"))))
                    for repetition in range(repetitions):
                        conn.execute(
                            "INSERT INTO study_attempts (id, study_id, "
                            "variant_index, repetition, attempt, seed, "
                            "status, created_at) VALUES "
                            "(?, ?, ?, ?, 1, ?, ?, ?)",
                            (uuid4().hex, study_id, index, repetition,
                             repetition_seeds[repetition], ATTEMPT_PENDING,
                             stamp))
                _event(conn, study_id, "study_created", {
                    "status": STUDY_QUEUED,
                    "variants": [v.label for v in variants],
                    "repetitions": repetitions,
                    "attempts": len(variants) * repetitions,
                    "repetition_seeds": list(repetition_seeds),
                    "idempotency_key": idempotency_key}, stamp)
                study = _select_study(conn, study_id)
                assert study is not None
                return study, True

        return await self.db.run(_insert)

    # ── reads ─────────────────────────────────────────────────────────────

    async def get_study(self, study_id: str) -> StudyRecord | None:
        return await self.db.run(lambda conn: _select_study(conn, study_id))

    async def find_by_idempotency_key(self, key: str) -> StudyRecord | None:
        """The study created with idempotency *key*, or None."""
        return await self.db.run(
            lambda conn: _select_by_idempotency_key(conn, key))

    async def list_studies(
        self, *, status: Sequence[str] | None = None, limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[StudyRecord], int]:
        """Newest first, plus the total matching count."""
        where, params = "", []
        if status is not None:
            unknown = sorted(set(status) - STUDY_STATUSES)
            if unknown:
                raise ValueError(f"unknown study status {unknown}; expected "
                                 f"one of {sorted(STUDY_STATUSES)}")
            if not status:
                return [], 0
            where = f" WHERE status IN ({','.join('?' * len(status))})"
            params = list(status)

        def _select(conn: sqlite3.Connection
                    ) -> tuple[list[StudyRecord], int]:
            rows = conn.execute(
                f"SELECT {_STUDY_COLUMNS} FROM studies{where} "
                "ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                [*params, limit, offset]).fetchall()
            total = conn.execute(f"SELECT COUNT(*) FROM studies{where}",
                                 params).fetchone()[0]
            return [StudyRecord.from_row(row) for row in rows], total

        return await self.db.run(_select)

    async def list_study_ids(self, statuses: Iterable[str]) -> list[str]:
        wanted = list(statuses)

        def _select(conn: sqlite3.Connection) -> list[str]:
            return [row[0] for row in conn.execute(
                "SELECT id FROM studies WHERE status IN "
                f"({','.join('?' * len(wanted))}) ORDER BY created_at",
                wanted)]

        return await self.db.run(_select) if wanted else []

    async def list_variants(self, study_id: str) -> list[StudyVariantRecord]:
        def _select(conn: sqlite3.Connection) -> list[sqlite3.Row]:
            return conn.execute(
                "SELECT study_id, variant_index, label, ops FROM "
                "study_variants WHERE study_id = ? ORDER BY variant_index",
                (study_id,)).fetchall()

        return [StudyVariantRecord(study_id=row["study_id"],
                                   index=row["variant_index"],
                                   label=row["label"],
                                   ops=_loads(row["ops"]))
                for row in await self.db.run(_select)]

    async def get_variant_graph(self, study_id: str,
                                index: int) -> dict[str, Any] | None:
        def _select(conn: sqlite3.Connection) -> str | None:
            row = conn.execute(
                "SELECT graph_snapshot FROM study_variants WHERE study_id = ? "
                "AND variant_index = ?", (study_id, index)).fetchone()
            return None if row is None else row[0]

        raw = await self.db.run(_select)
        return None if raw is None else json.loads(raw)

    async def list_attempts(self, study_id: str) -> list[AttemptRecord]:
        return await self.db.run(lambda conn: _select_attempts(conn,
                                                                study_id))

    async def attempt_counts(
        self, study_ids: Sequence[str],
    ) -> dict[str, dict[str, int]]:
        """``{study_id: {status: n}}`` over CURRENT attempts, for a list."""
        ids = list(study_ids)
        if not ids:
            return {}

        def _select(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
            out: dict[str, dict[str, int]] = {sid: {} for sid in ids}
            for study_id in ids:
                for attempt in current_attempts(
                        _select_attempts(conn, study_id)).values():
                    counts = out[study_id]
                    counts[attempt.status] = counts.get(attempt.status, 0) + 1
            return out

        return await self.db.run(_select)

    # ── events ────────────────────────────────────────────────────────────

    async def append_event(self, study_id: str, event_type: str,
                           payload: dict[str, Any]) -> int:
        stamp = utc_now_iso()

        def _append(conn: sqlite3.Connection) -> int:
            with transaction(conn):
                return _event(conn, study_id, event_type, payload, stamp)

        return await self.db.run(_append)

    async def get_events(self, study_id: str, *, after_cursor: int = 0,
                         limit: int = 500) -> list[StudyEventRecord]:
        def _select(conn: sqlite3.Connection) -> list[sqlite3.Row]:
            return conn.execute(
                f"SELECT {_EVENT_COLUMNS} FROM study_events WHERE "
                "study_id = ? AND cursor > ? ORDER BY cursor LIMIT ?",
                (study_id, after_cursor, limit)).fetchall()

        return [StudyEventRecord.from_row(row)
                for row in await self.db.run(_select)]

    async def latest_cursor(self, study_id: str) -> int:
        def _select(conn: sqlite3.Connection) -> int:
            return conn.execute(
                "SELECT COALESCE(MAX(cursor), 0) FROM study_events "
                "WHERE study_id = ?", (study_id,)).fetchone()[0]

        return await self.db.run(_select)

    # ── the study's own transitions ───────────────────────────────────────

    async def transition(
        self, study_id: str, *, to: str, expected: Iterable[str],
        event_type: str, payload: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> StudyRecord | None:
        """Move the study to *to* when its status is in *expected*.

        Returns the new row, or None when the guard refused (the study is
        gone or already moved on). ``finished_at`` is stamped when *to* is
        not an active status and cleared when it is (a resume).
        """
        if to not in STUDY_STATUSES:
            raise ValueError(f"unknown study status {to!r}")
        allowed = tuple(expected)
        stamp = utc_now_iso()
        finished = None if to in STUDY_ACTIVE else stamp

        def _update(conn: sqlite3.Connection) -> StudyRecord | None:
            with transaction(conn):
                study = _select_study(conn, study_id)
                if study is None or study.status not in allowed:
                    return None
                conn.execute(
                    "UPDATE studies SET status = ?, error = ?, "
                    "finished_at = ? WHERE id = ?",
                    (to, error, finished, study_id))
                _event(conn, study_id, event_type, {
                    **(payload or {}), "from": study.status, "to": to,
                    **({"error": error} if error else {})}, stamp)
                return _select_study(conn, study_id)

        return await self.db.run(_update)

    async def resume(
        self, study_id: str, *, reason: str,
    ) -> tuple[StudyRecord, list[AttemptRecord]]:
        """Reopen a resumable study: every repetition whose current attempt
        is cancelled, interrupted or lost gets a NEW pending attempt
        (attempt + 1, same seed); completed repetitions are left alone.

        Raises ``StudyStateError`` when the study cannot be resumed, and
        ``KeyError`` when it does not exist.
        """
        stamp = utc_now_iso()

        def _resume(conn: sqlite3.Connection
                    ) -> tuple[StudyRecord, list[AttemptRecord]]:
            with transaction(conn):
                study = _select_study(conn, study_id)
                if study is None:
                    raise KeyError(study_id)
                if study.status not in STUDY_RESUMABLE:
                    raise StudyStateError(
                        f"study {study_id} is {study.status}; only "
                        f"{', '.join(sorted(STUDY_RESUMABLE))} studies can "
                        "be resumed")
                current = current_attempts(_select_attempts(conn, study_id))
                created: list[str] = []
                for slot in sorted(current):
                    attempt = current[slot]
                    if attempt.status not in ATTEMPT_RETRYABLE:
                        continue
                    new_id = uuid4().hex
                    conn.execute(
                        "INSERT INTO study_attempts (id, study_id, "
                        "variant_index, repetition, attempt, seed, status, "
                        "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (new_id, study_id, attempt.variant_index,
                         attempt.repetition, attempt.attempt + 1,
                         attempt.seed, ATTEMPT_PENDING, stamp))
                    created.append(new_id)
                conn.execute(
                    "UPDATE studies SET status = ?, error = NULL, "
                    "finished_at = NULL WHERE id = ?",
                    (STUDY_RUNNING, study_id))
                fresh = {a.id: a for a in _select_attempts(conn, study_id)}
                new_attempts = [fresh[attempt_id] for attempt_id in created]
                completed = sum(1 for a in current.values()
                                if a.status in ATTEMPT_FINAL)
                _event(conn, study_id, "study_resumed", {
                    "from": study.status, "to": STUDY_RUNNING,
                    "reason": reason,
                    "new_attempts": [_attempt_ref(a) for a in new_attempts],
                    "completed_repetitions_kept": completed}, stamp)
                record = _select_study(conn, study_id)
                assert record is not None
                return record, new_attempts

        return await self.db.run(_resume)

    # ── attempt transitions ───────────────────────────────────────────────

    async def _attempt_update(
        self, attempt_id: str, *, expected: Iterable[str], sets: str,
        values: Sequence[Any], event_type: str,
        payload: dict[str, Any],
    ) -> AttemptRecord | None:
        allowed = tuple(expected)
        stamp = utc_now_iso()

        def _update(conn: sqlite3.Connection) -> AttemptRecord | None:
            with transaction(conn):
                row = conn.execute(
                    f"SELECT {_ATTEMPT_COLUMNS} FROM study_attempts "
                    "WHERE id = ?", (attempt_id,)).fetchone()
                if row is None or row["status"] not in allowed:
                    return None
                conn.execute(f"UPDATE study_attempts SET {sets} WHERE id = ?",
                             (*values, attempt_id))
                fresh = AttemptRecord.from_row(conn.execute(
                    f"SELECT {_ATTEMPT_COLUMNS} FROM study_attempts "
                    "WHERE id = ?", (attempt_id,)).fetchone())
                _event(conn, fresh.study_id, event_type, {
                    **_attempt_ref(fresh), "run_id": fresh.run_id,
                    "status": fresh.status, **payload}, stamp)
                return fresh

        return await self.db.run(_update)

    async def begin_submission(self, attempt_id: str,
                               run_id: str) -> AttemptRecord | None:
        """``pending -> submitting`` WITH the run id about to be used.

        Written before the child exists: if the process dies or the next
        write fails after the submit, the attempt already names its child.
        """
        return await self._attempt_update(
            attempt_id, expected=(ATTEMPT_PENDING,),
            sets="status = ?, run_id = ?, submitted_at = ?",
            values=(ATTEMPT_SUBMITTING, run_id, utc_now_iso()),
            event_type="attempt_submitting", payload={})

    async def confirm_submission(self, attempt_id: str,
                                 status: str) -> AttemptRecord | None:
        """``submitting -> queued | running`` once the submit returned."""
        return await self._attempt_update(
            attempt_id, expected=(ATTEMPT_SUBMITTING,), sets="status = ?",
            values=(status,), event_type="attempt_submitted", payload={})

    async def relink(self, attempt_id: str,
                     run: RunRecord) -> AttemptRecord | None:
        """Adopt the child found by its stamp for a ``submitting`` attempt
        (the linkage write after the submit did not land)."""
        status = run.status if run.status in (ATTEMPT_QUEUED,
                                              ATTEMPT_RUNNING) \
            else ATTEMPT_RUNNING
        return await self._attempt_update(
            attempt_id, expected=(ATTEMPT_SUBMITTING,),
            sets="status = ?, run_id = ?", values=(status, run.id),
            event_type="attempt_relinked",
            payload={"child_status": run.status})

    async def mark_lost(self, attempt_id: str, error: str,
                        *, expected: Iterable[str] = (ATTEMPT_SUBMITTING,),
                        ) -> AttemptRecord | None:
        return await self._attempt_update(
            attempt_id, expected=expected, sets="status = ?, error = ?",
            values=(ATTEMPT_LOST, error), event_type="attempt_lost",
            payload={"error": error})

    async def mark_started(self, attempt_id: str,
                           started_at: str | None) -> AttemptRecord | None:
        return await self._attempt_update(
            attempt_id, expected=(ATTEMPT_QUEUED,),
            sets="status = ?, started_at = ?",
            values=(ATTEMPT_RUNNING, started_at),
            event_type="attempt_started", payload={})

    async def add_attempt(self, previous: AttemptRecord,
                          reason: str) -> AttemptRecord | None:
        """A new pending attempt for *previous*'s slot, when *previous* is
        still the slot's current attempt and is retryable."""
        stamp = utc_now_iso()

        def _insert(conn: sqlite3.Connection) -> AttemptRecord | None:
            with transaction(conn):
                newest = conn.execute(
                    "SELECT MAX(attempt) FROM study_attempts WHERE "
                    "study_id = ? AND variant_index = ? AND repetition = ?",
                    (previous.study_id, previous.variant_index,
                     previous.repetition)).fetchone()[0]
                if newest != previous.attempt:
                    return None
                new_id = uuid4().hex
                conn.execute(
                    "INSERT INTO study_attempts (id, study_id, variant_index, "
                    "repetition, attempt, seed, status, created_at) VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?)",
                    (new_id, previous.study_id, previous.variant_index,
                     previous.repetition, previous.attempt + 1,
                     previous.seed, ATTEMPT_PENDING, stamp))
                fresh = AttemptRecord.from_row(conn.execute(
                    f"SELECT {_ATTEMPT_COLUMNS} FROM study_attempts "
                    "WHERE id = ?", (new_id,)).fetchone())
                _event(conn, previous.study_id, "attempt_added", {
                    **_attempt_ref(fresh), "replaces": previous.id,
                    "reason": reason}, stamp)
                return fresh

        return await self.db.run(_insert)

    async def harvest_runs(self, run_ids: Sequence[str]) -> set[str]:
        """Copy the evidence of every TERMINAL study child among *run_ids*
        onto its attempt; returns the study ids that changed."""
        ids = list(dict.fromkeys(run_ids))
        if not ids:
            return set()
        stamp = utc_now_iso()

        def _harvest(conn: sqlite3.Connection) -> set[str]:
            with transaction(conn):
                runs = conn.execute(
                    f"SELECT {_HARVEST_RUN_COLUMNS} FROM exec_runs WHERE id "
                    f"IN ({','.join('?' * len(ids))}) AND "
                    "study_attempt_id IS NOT NULL", ids).fetchall()
                return _harvest_run_rows(conn, runs, stamp)

        return await self.db.run(_harvest)
