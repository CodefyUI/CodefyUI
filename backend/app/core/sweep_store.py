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
import logging
import sqlite3
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .db import Database, transaction, utc_now_iso
from .run_store import (
    TERMINAL_STATUSES,
    metric_producers,
    producer_last_value,
)

logger = logging.getLogger(__name__)

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

#: **A stop was REQUESTED. It is not a promise that one will happen** (#404).
#:
#: What it asserts: at least one child was still active when
#: ``POST /api/sweeps/{id}/cancel`` reached it, and every such child was
#: asked to stop. What it does NOT assert: that any of them will.
#: Cancellation is cooperative all the way down — ``RunService.cancel`` sets
#: a flag on the ``ExecutionContext`` and never calls ``Task.cancel``,
#: because there is no safe way to interrupt arbitrary third-party node code
#: and killing the task outright is what leaves half-written rows and wedged
#: CUDA state. A node that polls ``context.should_stop()`` stops within a
#: batch; **a node that never polls it never stops, and this sweep stays
#: here for as long as that node runs** — which can be hours, and in
#: principle forever.
#:
#: There is deliberately NO timeout, and none should be added. The only way
#: to enforce a deadline is to kill a training run mid-step, which costs the
#: user the very thing the run existed to produce and, on a GPU, can leave
#: the device unusable until the process dies. A state that is honest about
#: being indefinite beats a deadline that is dishonest about being safe.
#:
#: How it ENDS: not by elapsed time, but by the children. ``_write_variants``
#: moves the sweep to ``finished`` on the first harvest (either seam) at
#: which every variant is terminal — the cancel is over once nothing is
#: active, whether the children stopped because they were asked to or
#: because they were going to finish anyway. A server restart also ends it:
#: ``RunService.recover_interrupted`` retires the abandoned rows, and the
#: next read settles the sweep.
#:
#: How a READER tells a slow stop from a stuck one: never by how long the
#: state has been showing. ``GET /api/sweeps/{id}`` returns per-variant
#: ``status`` and the ``counts`` tally built from it, so
#: ``counts["running"] + counts["queued"]`` is exactly what the stop is
#: still waiting on, by name and by run id. A client that renders
#: ``cancelling`` as a spinner with no such count is promising something
#: this state does not.
SWEEP_STATE_CANCELLING = "cancelling"

SWEEP_STATE_FINISHED = "finished"
SWEEP_STATE_FAILED = "failed"
SWEEP_STATES: frozenset[str] = frozenset({
    SWEEP_STATE_RUNNING, SWEEP_STATE_CANCELLING,
    SWEEP_STATE_FINISHED, SWEEP_STATE_FAILED,
})

#: The sweep is over and nothing will move it again: ``finished`` is stamped
#: once (re-stamping would drag ``finished_at`` forward on every poll) and
#: ``failed`` is the sweep's own record of what went wrong and is never
#: overwritten. Both halves are the same question, so they are one set
#: rather than a tuple repeated at each site.
#:
#: The complement matters more than the set does: ``running`` and
#: ``cancelling`` are BOTH still-going states, and classifying ``cancelling``
#: with them rather than with the terminal pair is the machine-readable half
#: of what it means (see above). A client that stops polling on
#: ``cancelling`` has filed a sweep as over while its children still run.
SWEEP_SETTLED_STATES: frozenset[str] = frozenset({
    SWEEP_STATE_FINISHED, SWEEP_STATE_FAILED,
})

# The objective's ranking direction, enforced the same way and in the same
# place, so a route validating the submitted field imports this rather than
# re-spelling the two strings. There is no default and none is inferred:
# spec 6.2 requires the field because `val_loss` and `val_accuracy` are both
# plausible objectives and rank in OPPOSITE directions, and guessing from a
# user-chosen series name is exactly the heuristic that would be wrong
# silently.
SWEEP_DIRECTION_MINIMIZE = "minimize"
SWEEP_DIRECTION_MAXIMIZE = "maximize"
SWEEP_DIRECTIONS: frozenset[str] = frozenset({
    SWEEP_DIRECTION_MINIMIZE, SWEEP_DIRECTION_MAXIMIZE,
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
    #: The node whose series the objective was read from (#641): the
    #: objective's own ``node_id``, or the ONE node that logged the metric
    #: for a name-only objective. None when no node did, when the only
    #: producer is the run-level series, or when the name was ambiguous.
    #: Harvested with the value, so the identity outlives the child.
    objective_node_id: str | None = None
    #: Set only when a name-only objective was AMBIGUOUS in this run: every
    #: node that logged the metric, sorted (``None`` is the run-level
    #: series). The variant then has no objective and no rank (#641).
    ambiguous_producers: list[str | None] | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "index": self.index, "domain_index": self.domain_index,
            "run_id": self.run_id, "params": self.params, "seed": self.seed,
            "objective": self.objective, "status": self.status,
            "harvested_at": self.harvested_at,
            "objective_node_id": self.objective_node_id,
            "ambiguous_producers": self.ambiguous_producers,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "SweepVariant":
        # The two #641 keys are absent from a row harvested before them,
        # which reads as "no producer recorded" -- the truth for that row.
        ambiguous = raw.get("ambiguous_producers")
        return cls(
            index=raw["index"], domain_index=raw["domain_index"],
            run_id=raw.get("run_id"), params=list(raw.get("params") or []),
            seed=raw.get("seed"), objective=raw.get("objective"),
            status=raw.get("status"), harvested_at=raw.get("harvested_at"),
            objective_node_id=raw.get("objective_node_id"),
            ambiguous_producers=(None if ambiguous is None
                                 else list(ambiguous)),
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

    async def read_harvest(
        self, children: Mapping[int, tuple[str, str]],
        objective: Mapping[str, Any],
    ) -> dict[int, HarvestEntry]:
        """Seam A's read: ``{variant index: HarvestEntry}`` for terminal
        children, one ``Database.run`` for all of them.

        *children* maps a variant index to its child's ``(run_id,
        status)``. Each objective is read by :func:`harvest_entry`, the
        function seam B runs inside ``RunStore.prune``, so the two seams
        share the rule itself rather than agreeing on results (#404, #641).
        """
        if not children:
            return {}

        def _read(conn: sqlite3.Connection) -> dict[int, HarvestEntry]:
            return {index: harvest_entry(conn, run_id, status, objective)
                    for index, (run_id, status) in children.items()}

        return await self.db.run(_read)

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
    #: See ``SweepVariant.objective_node_id``.
    objective_node_id: str | None = None
    #: See ``SweepVariant.ambiguous_producers``.
    ambiguous_producers: list[str | None] | None = None


def objective_selector(
    objective: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    """``(metric, node_id)`` off a stored objective, legacy rows included.

    A sweep created before #641 stored ``{"metric", "direction"}`` only. It
    reads as ``node_id`` None, i.e. NAME-ONLY selection with the ambiguity
    rule of :func:`read_objective`. That is the compatibility decision for
    already-created sweeps: their stored row is left as it is (no
    migration rewrites it), a variant already harvested keeps the value it
    was given, and every variant harvested from now on is read by the
    same rule as a new name-only sweep. Choosing a producer for an old
    sweep would need a guess #641 forbids.
    """
    metric = objective.get("metric")
    node_id = objective.get("node_id")
    return (metric if isinstance(metric, str) and metric else None,
            node_id if isinstance(node_id, str) and node_id else None)


def harvest_entry(conn: sqlite3.Connection, run_id: str, status: str,
                  objective: Mapping[str, Any]) -> HarvestEntry:
    """One terminal child's harvest, as :func:`read_objective` reads it."""
    value, node_id, ambiguous = read_objective(conn, run_id, objective)
    return HarvestEntry(objective=value, status=status,
                        objective_node_id=node_id,
                        ambiguous_producers=ambiguous)


def read_objective(
    conn: sqlite3.Connection, run_id: str, objective: Mapping[str, Any],
) -> tuple[float | None, str | None, list[str | None] | None]:
    """A sweep objective's ``(value, producer, ambiguous producers)`` for
    ONE run -- THE rule, used by both harvest seams (#404, #641).

    * **node_id named**: the last point of that node's series, by
      ``step DESC, id DESC`` WITHIN the node. Another node logging the
      same name at a higher step, or later at the same step, cannot reach
      it. A flattened inner id (``block/inner``) is just a node id here.
    * **name only, one producer**: that producer's last point, exactly as
      if it had been named. A sweep made before #641 is in this case when
      its graph has one producer, so it ranks as it always did.
    * **name only, several producers**: no value, and the producers are
      returned so the sweep can say why. Picking one by step or by write
      order is what #641 removes; a reduction across producers would need
      a rule of its own and none is inferred.

    The value is None as well when the series' last point is non-finite:
    a diverged loss is unranked, never a fabricated 0.0. An empty or absent
    metric reads nothing: a sweeps row is durable and outlives the route
    validation that refuses one.

    A plain function on a CONNECTION, because seam B runs it inside
    ``RunStore.prune``'s transaction (see ``_select_sweep``).
    """
    metric, node_id = objective_selector(objective)
    if metric is None:
        return None, None, None
    if node_id is not None:
        return (producer_last_value(conn, run_id, metric, node_id), node_id,
                None)
    producers = metric_producers(conn, run_id, metric)
    if len(producers) > 1:
        return None, None, producers
    if not producers:
        return None, None, None
    return (producer_last_value(conn, run_id, metric, producers[0]),
            producers[0], None)


def variant_is_terminal(variant: SweepVariant, child_status: str | None, *,
                        child_exists: bool) -> bool:
    """Spec 4.3's "terminal outcome", defined ONCE and used by both seams.

    True when the variant carries a harvested terminal status, or its run
    row is **gone** (spec 5.3's ``missing``), or its live child row is
    terminal. The gone-row clause is load-bearing: without it a variant
    whose run someone deleted by hand would keep its sweep at ``running``
    forever.

    **A variant with no ``run_id`` at all is NOT terminal** (spec 4.3, as
    corrected during this task — the earlier "no reachable run" wording
    covered it and that was a real bug). During ``POST /api/sweeps`` the
    later variants legitimately carry ``run_id: null`` until the submit loop
    reaches them, and a prune fires on the same event loop after every run
    finishes — so counting them as done stamps a sweep ``finished`` on its
    first child, and ``_write_variants`` never re-stamps a sweep that
    already reads ``finished``. The wrong answer would be permanent. The
    submit loop BREAKING is the other case, and it does not need this
    clause: spec 5.2 has the route call :meth:`SweepStore.mark_failed`
    itself, and neither seam ever overwrites ``failed``. That makes it a
    DUTY on the route — a submit loop that neither fills in every ``run_id``
    nor calls ``mark_failed`` leaves its sweep at ``running`` for good.

    The harvested-status check comes FIRST so a variant seam B harvested by
    ``sweep_variant`` before its ``set_variant_run`` landed still counts.
    """
    if variant.status is not None:
        return True
    if variant.run_id is None:
        return False
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

    An unrecognised *direction* RAISES, as ``set_state`` does for its own
    vocabulary. Falling back to minimize would present the worst variant of
    a ``maximize`` sweep as the best one, with nothing on screen to say so.
    """
    if direction not in SWEEP_DIRECTIONS:
        raise ValueError(
            f"unknown sweep direction {direction!r}; expected one of "
            f"{sorted(SWEEP_DIRECTIONS)}")
    sign = -1 if direction == SWEEP_DIRECTION_MAXIMIZE else 1
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
                status=entries[variant.index].status, harvested_at=stamp,
                objective_node_id=entries[variant.index].objective_node_id,
                ambiguous_producers=entries[variant.index].ambiguous_producers)
        if variant.index in entries and variant.harvested_at is None
        else variant
        for variant in variants
    ]


def _write_variants(conn: sqlite3.Connection, record: SweepRecord,
                    variants: Sequence[SweepVariant], *, finished: bool,
                    stamp: str) -> None:
    """One UPDATE: the patched blob plus, when the sweep has just become
    finished, its terminal state.

    A SETTLED sweep keeps the state it has: ``failed`` is never overwritten
    and ``finished`` is never re-stamped. That is one question, so it is
    asked once, against ``SWEEP_SETTLED_STATES``.

    A sweep in ``cancelling`` is NOT settled and does land on ``finished``
    here — this line is the only way out of that state, and it is driven by
    the children being terminal rather than by any deadline (see
    ``SWEEP_STATE_CANCELLING``).
    """
    state = record.state
    finished_at = record.finished_at
    if finished and state not in SWEEP_SETTLED_STATES:
        state = SWEEP_STATE_FINISHED
        finished_at = finished_at or stamp
    conn.execute(
        "UPDATE sweeps SET variants = ?, state = ?, finished_at = ? "
        "WHERE id = ?",
        (_dumps([v.as_json() for v in variants]), state, finished_at,
         record.id))


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

    A sweep whose harvest RAISES still has its children deleted, and #404
    asked what should be left behind. The answer is
    :func:`_record_harvest_failure`: the row says why its table is empty.
    Pinning the children instead — skipping their delete until the harvest
    succeeds — was the alternative, and it is the one option this module
    cannot take: the failures that reach the handler below are mostly
    permanent (an unreadable ``variants`` blob does not heal), so a single
    corrupt cell would exempt its children from retention for good, and
    with them their metrics, their checkpoint files and their TensorBoard
    directories. That is bookkeeping blocking retention, which is the one
    thing this path must never do.
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
        # ISOLATED PER SWEEP. `prune` sweeps the whole table in one pass, so
        # the sweeps in this loop have nothing to do with one another: one
        # unreadable `variants` cell must cost its own sweep its results and
        # NOTHING ELSE. Catching one level up instead would keep retention
        # alive but unwind the loop, and every healthy sweep in the pass
        # would lose the numbers RULING 4 exists to preserve while its
        # children were deleted in the same transaction. The DELETE proceeds
        # either way: retention is unattended and irreplaceable, and must
        # never be blocked by bookkeeping.
        try:
            harvested += _harvest_one_sweep(conn, sweep_id, rows, doomed_ids,
                                            stamp)
        except Exception as exc:
            logger.warning(
                "retention: could not harvest sweep %s, so its variants keep "
                "whatever was harvested before; every other sweep in this "
                "pass is unaffected and the delete proceeds", sweep_id,
                exc_info=True)
            _record_harvest_failure(conn, sweep_id, rows, exc)
    return harvested


#: How much of a failed harvest's own exception text is kept on the row.
#: ``sweeps.error`` is durable, nothing deletes it, and it is serialised
#: into every later ``GET /api/sweeps/{id}`` body — a pathological ``str``
#: must not become this sweep's permanent payload.
_HARVEST_ERROR_DETAIL_MAX = 300


def _record_harvest_failure(conn: sqlite3.Connection, sweep_id: str,
                            rows: Sequence[sqlite3.Row],
                            exc: BaseException) -> None:
    """Write WHY a sweep's results are missing, onto the sweep row (#404).

    The residue the per-sweep isolation leaves behind: the children were
    deleted unharvested, so the numbers are gone for good, and without this
    the row is indistinguishable from a sweep that simply never produced
    anything. Both render as an empty comparison table. A reader who cannot
    tell those apart will go looking for a bug in their graph.

    ``sweeps.error`` is the field, and it reaches a reader for free —
    ``GET /api/sweeps/{id}`` already puts it in every response body, so no
    route, no column and no migration is involved.

    **The STATE is deliberately left alone.** A failed harvest says nothing
    about which variants are terminal, and ``failed`` on a sweep is
    reserved for a broken SUBMIT loop (:meth:`SweepStore.mark_failed`) —
    claiming it here would both overstate what is known and, because
    neither seam ever overwrites ``failed``, permanently prevent a sweep
    from being stamped ``finished`` over a bookkeeping error that may well
    have been transient.

    ``error IS NULL`` guards the write for the same reason
    :meth:`SweepStore.set_state` guards on ``failed``: a submit loop's own
    record of what went wrong outranks a later retention note, and where
    two retention passes both failed, the FIRST one is when the results
    were actually lost.

    **Never raises.** It runs inside ``RunStore.prune``'s transaction, one
    statement before the DELETE, so a throw here would abort retention for
    every run in the pass — the exact failure the isolation above exists to
    prevent, reintroduced by the note about it. Bookkeeping about
    bookkeeping is still bookkeeping.
    """
    detail = f"{type(exc).__name__}: {exc}"
    if len(detail) > _HARVEST_ERROR_DETAIL_MAX:
        detail = detail[:_HARVEST_ERROR_DETAIL_MAX - 3] + "..."
    message = (
        f"retention deleted {len(rows)} finished run(s) of this sweep "
        f"before the harvest could copy their objectives onto this row, so "
        f"those results are gone for good ({detail})")
    try:
        conn.execute(
            "UPDATE sweeps SET error = ? WHERE id = ? AND error IS NULL",
            (message, sweep_id))
    except Exception:
        logger.warning(
            "retention: could not record the failed harvest on sweep %s "
            "either; the delete still proceeds", sweep_id, exc_info=True)


def _harvest_one_sweep(conn: sqlite3.Connection, sweep_id: str,
                       rows: Sequence[sqlite3.Row], doomed_ids: set[str],
                       stamp: str) -> int:
    """One sweep's share of seam B; returns how many entries it harvested.

    Split out so :func:`harvest_doomed` can isolate a failure to the sweep
    that caused it. The only write is the single ``_write_variants`` UPDATE
    at the end, so a sweep that raises anywhere above leaves its row exactly
    as it was rather than half-patched.
    """
    record = _select_sweep(conn, sweep_id)
    if record is None:
        return 0
    entries = {
        row["sweep_variant"]: harvest_entry(conn, row["id"], row["status"],
                                            record.objective)
        for row in rows if row["sweep_variant"] is not None
    }
    patched = _apply_entries(record.variants, entries, stamp)
    # Every child that will STILL EXIST after the DELETE. The doomed ones
    # are excluded on purpose: they are terminal by definition (prune never
    # deletes an active run) and they have just been harvested, so
    # `variant.status is not None` already answers for them.
    live = {row["id"]: row["status"] for row in conn.execute(
        "SELECT id, status FROM exec_runs WHERE sweep_id = ?",
        (sweep_id,)) if row["id"] not in doomed_ids}
    finished = all(
        variant_is_terminal(variant, live.get(variant.run_id),
                            child_exists=variant.run_id in live)
        for variant in patched)
    _write_variants(conn, record, patched, finished=finished, stamp=stamp)
    return len(entries)
