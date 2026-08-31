"""Durable sweep rows (#140) — the parent of a set of variant runs.

Built to ``run_store``'s rules, for the same reasons: every method goes
through ``Database.run``; each method is individually atomic; **methods do
not compose** — calling one from inside another's ``fn(conn)`` closure
deadlocks on the non-reentrant lock (``run_store``'s module docstring).
Reads return frozen dataclasses, never ``sqlite3.Row``.

RULING 4: this row OWNS its results. ``RunStore.prune`` deletes finished
runs after every terminal transition and at startup, so a sweep that only
pointed at run ids would rot into dangling references. Each variant's chosen
params and harvested objective live here; the run id stays as a link that
MAY BE DEAD.

``variants`` is a JSON list on the row rather than a ``sweep_variants``
table: ``MAX_SWEEP_RUNS`` bounds it at 32 entries, this codebase already
stores bounded structured JSON in a column (``options``, ``plugin_pins``,
``exec_run_artifacts.meta``), and ranking 32 rows happens in Python, not SQL.
The cost is that a harvest is a read-modify-write of the whole array — safe
because ``Database.run`` is one connection on one worker thread behind one
``asyncio.Lock``, so a SELECT-patch-UPDATE inside a SINGLE ``fn(conn)`` is
atomic against every other database operation in the process.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .db import Database, transaction, utc_now_iso
from .run_store import TERMINAL_STATUSES

# ── state vocabulary ──────────────────────────────────────────────────────
#
# Enforced in Python, the way RUN_STATUSES is: `sweeps.state` carries no SQL
# CHECK constraint, following exec_runs.status' own policy (SQLite cannot
# ALTER one away).
#
# There is deliberately no `cancelled`. A sweep is a CONTAINER, not a run:
# collapsing 32 outcomes into one word loses the only information a
# comparison table exists to show, and a sweep where 30 variants finished and
# 2 were stopped is not "a cancelled sweep". Per-variant detail lives in
# variants[].status.
SWEEP_STATE_RUNNING = "running"
SWEEP_STATE_CANCELLING = "cancelling"
SWEEP_STATE_FINISHED = "finished"
SWEEP_STATE_FAILED = "failed"
SWEEP_STATES: frozenset[str] = frozenset({
    SWEEP_STATE_RUNNING, SWEEP_STATE_CANCELLING,
    SWEEP_STATE_FINISHED, SWEEP_STATE_FAILED,
})

_SWEEP_COLUMNS = (
    "id, name, state, method, seed, seed_variants, spec, objective, "
    "variants, error, created_at, finished_at"
)


def _dumps(value: Any) -> str:
    """JSON for a sweep column. ``allow_nan=False`` deliberately.

    Nothing that reaches this row may be non-finite: the compiler refuses a
    NaN domain value at the door (spec 2.7) and a non-finite metric is
    stored as SQL NULL and read back as None. If one ever did arrive, a
    loud failure here beats ``_json_safe``'s silent rewrite to ``null`` in a
    DURABLE row that nothing deletes.
    """
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class SweepVariant:
    """One entry of ``sweeps.variants``. Everything here is JSON-safe."""

    #: 0-based; the variant's position in the list.
    index: int
    #: Which cell of the cartesian product this is.
    domain_index: int
    #: The child run. None only if creation failed part-way (spec 5.2).
    run_id: str | None
    #: [{"node_id":.., "param":.., "value":..}] in declared order.
    params: list[dict[str, Any]]
    #: The EXECUTION seed, when seed_variants is on. RULING 1.
    seed: int | None
    #: Harvested final objective. None = absent.
    objective: float | None
    #: HARVESTED terminal status. None = not harvested yet. NOT the same
    #: field as a response's "status", which prefers the child's LIVE status
    #: and falls back to this one.
    status: str | None
    #: ISO-8601 Z. None = not harvested yet. This is what stops a variant
    #: that legitimately produced no objective from being re-read forever:
    #: "harvested, no value" is a recorded fact, not a retry.
    harvested_at: str | None

    def as_json(self) -> dict[str, Any]:
        return {
            "index": self.index, "domain_index": self.domain_index,
            "run_id": self.run_id, "params": self.params, "seed": self.seed,
            "objective": self.objective, "status": self.status,
            "harvested_at": self.harvested_at,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "SweepVariant":
        return cls(
            index=raw["index"], domain_index=raw["domain_index"],
            run_id=raw.get("run_id"), params=list(raw.get("params") or []),
            seed=raw.get("seed"), objective=raw.get("objective"),
            status=raw.get("status"), harvested_at=raw.get("harvested_at"),
        )


@dataclass(frozen=True)
class SweepRecord:
    """One ``sweeps`` row."""

    id: str
    name: str | None
    state: str
    method: str
    seed: int | None
    seed_variants: bool
    spec: dict[str, Any]
    objective: dict[str, Any]
    variants: list[SweepVariant]
    error: str | None
    created_at: str
    finished_at: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "SweepRecord":
        variants = json.loads(row["variants"])
        if not isinstance(variants, list):
            # `or []` would quietly turn a corrupt value into a plausible
            # empty sweep; an explicit check keeps damage loud, the rule
            # RunRecord.from_row applies to `options`.
            raise ValueError(
                f"sweeps.variants for sweep {row['id']!r} is "
                f"{type(variants).__name__}, expected a JSON array")
        return cls(
            id=row["id"], name=row["name"], state=row["state"],
            method=row["method"], seed=row["seed"],
            seed_variants=bool(row["seed_variants"]),
            spec=json.loads(row["spec"]),
            objective=json.loads(row["objective"]),
            variants=[SweepVariant.from_json(v) for v in variants],
            error=row["error"], created_at=row["created_at"],
            finished_at=row["finished_at"],
        )


def _select_sweep(conn: sqlite3.Connection,
                  sweep_id: str) -> SweepRecord | None:
    """One row, off an already-open connection.

    A plain function, not a method: it is called both from ``SweepStore``'s
    own ``Database.run`` closures and from ``harvest_doomed``, which runs
    inside ``RunStore.prune``'s transaction and must never open a second
    ``Database.run``.
    """
    row = conn.execute(
        f"SELECT {_SWEEP_COLUMNS} FROM sweeps WHERE id = ?",
        (sweep_id,)).fetchone()
    return None if row is None else SweepRecord.from_row(row)


@dataclass
class SweepStore:
    """Typed CRUD over the ``sweeps`` table.

    Holds no state beyond the ``Database`` it was handed, so it is safe to
    construct per request or once on ``app.state`` — the same contract
    ``RunStore`` states.
    """

    db: Database

    async def create_sweep(
        self,
        *,
        method: str,
        seed: int | None,
        seed_variants: bool,
        spec: dict[str, Any],
        objective: dict[str, Any],
        variants: Sequence[SweepVariant],
        name: str | None = None,
        sweep_id: str | None = None,
    ) -> SweepRecord:
        """Insert a sweep and return its row, ``state='running'``.

        The sweep row is written BEFORE its children: ``exec_runs.sweep_id``
        has an enforced foreign key, so inserting a child first fails with
        ``FOREIGN KEY constraint failed``.

        *variants* carry ``run_id: None`` placeholders; the caller patches
        each id in with :meth:`set_variant_run` as its submit returns.

        **Each variant's ``index`` is assigned here, not taken on trust,
        and this method is the only place that can be.** MIGRATION_004
        spells out why: SQLite cannot add a UNIQUE column via
        ``ADD COLUMN``, so ``(sweep_id, sweep_variant)`` is a read index and
        not a constraint — the database will store two variant 3s without a
        murmur, ``set_variant_run`` would then write one run id into both,
        and two children would answer to the same ``sweep_variant``. There
        is nothing below this line that would notice. Deriving the value
        from the position is not a guess: spec 4.2 defines ``index`` AS the
        entry's position in the list, which makes a duplicate unrepresentable
        rather than merely detectable. ``domain_index`` — which cell of the
        cartesian product this is — is the caller's own datum and is left
        exactly as handed in.
        """
        record = SweepRecord(
            id=sweep_id or uuid4().hex,
            name=name,
            state=SWEEP_STATE_RUNNING,
            method=method,
            seed=seed,
            seed_variants=bool(seed_variants),
            spec=spec,
            objective=objective,
            variants=[replace(v, index=i) for i, v in enumerate(variants)],
            error=None,
            created_at=utc_now_iso(),
            finished_at=None,
        )
        params = (
            record.id, record.name, record.state, record.method, record.seed,
            int(record.seed_variants), _dumps(record.spec),
            _dumps(record.objective),
            _dumps([v.as_json() for v in record.variants]),
            record.created_at,
        )

        def _insert(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO sweeps (id, name, state, method, seed, "
                "seed_variants, spec, objective, variants, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", params)

        await self.db.run(_insert)
        return record

    async def get_sweep(self, sweep_id: str) -> SweepRecord | None:
        return await self.db.run(
            lambda conn: _select_sweep(conn, sweep_id))

    async def set_variant_run(self, sweep_id: str, index: int, *,
                              run_id: str, seed: int | None) -> bool:
        """Patch one variant's ``run_id`` and execution ``seed``.

        Called once per variant as its submit returns, rather than once at
        the end: patching in one call would lose every id if the loop broke
        half-way, which is precisely the case RULING 2 says must stay
        visible. Thirty-two small writes is milliseconds.

        False when the sweep or the index does not exist.
        """
        def _patch(conn: sqlite3.Connection) -> bool:
            with transaction(conn):
                record = _select_sweep(conn, sweep_id)
                if record is None:
                    return False
                patched = [replace(v, run_id=run_id, seed=seed)
                           if v.index == index else v
                           for v in record.variants]
                if patched == record.variants:
                    return False
                conn.execute(
                    "UPDATE sweeps SET variants = ? WHERE id = ?",
                    (_dumps([v.as_json() for v in patched]), sweep_id))
                return True

        return await self.db.run(_patch)

    async def mark_failed(self, sweep_id: str, error: str) -> bool:
        """The submit loop broke part-way: some variants have no run at all.

        ``failed`` on a sweep means something went wrong with the SWEEP, not
        with the training — a sweep whose variants all failed is
        ``finished``, with the detail in ``variants[].status``.
        """
        def _update(conn: sqlite3.Connection) -> int:
            return conn.execute(
                "UPDATE sweeps SET state = ?, error = ?, finished_at = ? "
                "WHERE id = ?",
                (SWEEP_STATE_FAILED, error, utc_now_iso(), sweep_id),
            ).rowcount

        return await self.db.run(_update) > 0

    async def set_state(self, sweep_id: str, state: str) -> bool:
        """Move the sweep to *state*. False when nothing changed.

        Guarded on ``state != 'failed'``: a cancel must never overwrite a
        failed submit loop's record of what went wrong.
        """
        if state not in SWEEP_STATES:
            raise ValueError(
                f"unknown sweep state {state!r}; expected one of "
                f"{sorted(SWEEP_STATES)}")

        def _update(conn: sqlite3.Connection) -> int:
            return conn.execute(
                "UPDATE sweeps SET state = ? WHERE id = ? AND state != ?",
                (state, sweep_id, SWEEP_STATE_FAILED)).rowcount

        return await self.db.run(_update) > 0

    async def harvest(self, sweep_id: str, *,
                      entries: Mapping[int, HarvestEntry],
                      finished: bool) -> SweepRecord | None:
        """Seam A: patch harvested objectives in and settle the state.

        SELECT-patch-UPDATE inside ONE ``fn(conn)``, which is atomic against
        every other database operation in the process. Returns the row as it
        now stands, so a caller never needs a second read.

        Safe against a race with a finishing run because of ``_finalize``'s
        documented ordering: metrics are flushed with ``force=True`` BEFORE
        the terminal event and before ``mark_finished``, so by the time a
        row reads terminal its series are durable.
        """
        stamp = utc_now_iso()

        def _patch(conn: sqlite3.Connection) -> SweepRecord | None:
            with transaction(conn):
                record = _select_sweep(conn, sweep_id)
                if record is None:
                    return None
                _write_variants(
                    conn, record,
                    _apply_entries(record.variants, entries, stamp),
                    finished=finished, stamp=stamp)
                return _select_sweep(conn, sweep_id)

        return await self.db.run(_patch)


# ── the harvest (spec 6.3) ────────────────────────────────────────────────


@dataclass(frozen=True)
class HarvestEntry:
    """What one terminal child contributes to its sweep row."""

    objective: float | None
    status: str


def variant_is_terminal(variant: SweepVariant, child_status: str | None, *,
                        child_exists: bool) -> bool:
    """Spec 4.3's "terminal outcome", defined ONCE and used by both seams.

    True when the variant has **no reachable run** (``run_id`` is None — the
    failed-submit-loop case — or its row is gone and it was never
    harvested), or it carries a harvested terminal status, or its live child
    row is terminal.

    The no-reachable-run clause is load-bearing: without it a variant whose
    run someone deleted by hand would keep its sweep at ``running`` forever.
    """
    if variant.run_id is None:
        return True
    if variant.status is not None:
        return True
    if not child_exists:
        return True
    return child_status in TERMINAL_STATUSES


def rank_variants(
    variants: Sequence[SweepVariant], *, direction: str,
) -> list[tuple[SweepVariant, int | None]]:
    """``(variant, rank)`` in DISPLAY order — best first, then the rest.

    Rankable means ``objective is not None``. Note the important
    NON-exclusion: a **failed** variant that did log the objective before it
    died IS ranked, on the value it reached — hiding a real number because
    the run ended badly is the silent disappearance #140's third acceptance
    criterion forbids.

    Ties break on ``index`` **ascending** in both directions, so the order is
    total and the table does not reshuffle between polls. Unrankable
    variants keep their row with ``rank`` None and are appended in ``index``
    order — never dropped, and never sorted as if None were a number.
    """
    sign = -1 if direction == "maximize" else 1
    rankable = sorted((v for v in variants if v.objective is not None),
                      key=lambda v: (sign * v.objective, v.index))
    unranked = sorted((v for v in variants if v.objective is None),
                      key=lambda v: v.index)
    return ([(variant, i + 1) for i, variant in enumerate(rankable)]
            + [(variant, None) for variant in unranked])


def _apply_entries(variants: Sequence[SweepVariant],
                   entries: Mapping[int, HarvestEntry],
                   stamp: str) -> list[SweepVariant]:
    """Patch harvested values in. Already-harvested variants are left alone,
    which is what makes both seams idempotent."""
    return [
        replace(variant, objective=entries[variant.index].objective,
                status=entries[variant.index].status, harvested_at=stamp)
        if variant.index in entries and variant.harvested_at is None
        else variant
        for variant in variants
    ]


def _write_variants(conn: sqlite3.Connection, record: SweepRecord,
                    variants: Sequence[SweepVariant], *, finished: bool,
                    stamp: str) -> None:
    """One UPDATE: the patched blob plus, when the sweep has just become
    finished, its terminal state.

    ``failed`` is NEVER overwritten, ``finished`` is never re-stamped, and a
    sweep in ``cancelling`` lands on ``finished`` — the cancel is over once
    nothing is active.
    """
    state = record.state
    finished_at = record.finished_at
    if finished and state not in (SWEEP_STATE_FAILED, SWEEP_STATE_FINISHED):
        state = SWEEP_STATE_FINISHED
        finished_at = finished_at or stamp
    conn.execute(
        "UPDATE sweeps SET variants = ?, state = ?, finished_at = ? "
        "WHERE id = ?",
        (_dumps([v.as_json() for v in variants]), state, finished_at,
         record.id))


def _last_metric_value(conn: sqlite3.Connection, run_id: str,
                       name: str | None) -> float | None:
    """The LAST point of one series.

    ``ORDER BY step DESC, id DESC LIMIT 1`` is the identical rule
    ``RunStore.latest_metrics`` uses (``_LATEST_METRICS_SQL``), so seam A
    and seam B can never disagree about a variant's objective. Covered by
    ``idx_exec_run_metrics_series``, so this is one seek per doomed child.

    A NULL value is a non-finite point (a diverged loss) and reads back as
    None — exactly as ``latest_metrics`` omits that series.
    """
    if not name:
        return None
    row = conn.execute(
        "SELECT value FROM exec_run_metrics WHERE run_id = ? AND name = ? "
        "ORDER BY step DESC, id DESC LIMIT 1", (run_id, name)).fetchone()
    return None if row is None else row["value"]


def harvest_doomed(conn: sqlite3.Connection, where_clause: str,
                   params: tuple) -> int:
    """Seam B: copy the final objective + status of every run about to be
    deleted into its sweep row. Returns how many variants were harvested.

    Runs INSIDE the caller's transaction and takes a CONNECTION, never a
    ``Database``: a second ``Database.run`` here would deadlock on the
    non-reentrant lock (``run_store``'s module docstring). A function
    operating on an already-open connection does not violate that rule.

    *where_clause* and *params* are ``RunStore.prune``'s own, verbatim, and
    this runs immediately before its DELETE — the same discipline the
    checkpoint-path read already uses, so a row cannot become eligible
    between the two statements and be seen by one and deleted by the other.
    **Retention cannot delete a child its sweep has not already harvested.**

    Covers retention ONLY. ``DELETE /api/runs/{id}`` calls ``delete_run``,
    not ``prune``, so a hand-deleted child that no read had harvested loses
    its objective; the design then reports that variant honestly as
    ``missing`` with a null objective (spec 10.12 files the fix).
    """
    doomed = conn.execute(
        "SELECT id, sweep_id, sweep_variant, status FROM exec_runs "
        f"WHERE {where_clause} AND sweep_id IS NOT NULL", params).fetchall()
    if not doomed:
        return 0

    doomed_ids = {row["id"] for row in doomed}
    by_sweep: dict[str, list[sqlite3.Row]] = {}
    for row in doomed:
        by_sweep.setdefault(row["sweep_id"], []).append(row)

    stamp = utc_now_iso()
    harvested = 0
    for sweep_id, rows in by_sweep.items():
        record = _select_sweep(conn, sweep_id)
        if record is None:
            continue
        metric = record.objective.get("metric")
        entries = {
            row["sweep_variant"]: HarvestEntry(
                objective=_last_metric_value(conn, row["id"], metric),
                status=row["status"])
            for row in rows if row["sweep_variant"] is not None
        }
        patched = _apply_entries(record.variants, entries, stamp)
        # Every child that will STILL EXIST after the DELETE. The doomed
        # ones are excluded on purpose: they are terminal by definition
        # (prune never deletes an active run) and they have just been
        # harvested, so `variant.status is not None` already answers for
        # them.
        live = {row["id"]: row["status"] for row in conn.execute(
            "SELECT id, status FROM exec_runs WHERE sweep_id = ?",
            (sweep_id,)) if row["id"] not in doomed_ids}
        finished = all(
            variant_is_terminal(variant, live.get(variant.run_id),
                                child_exists=variant.run_id in live)
            for variant in patched)
        _write_variants(conn, record, patched, finished=finished,
                        stamp=stamp)
        harvested += len(entries)
    return harvested
