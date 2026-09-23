"""The three Sweep v1 backend loose ends (#404).

A. A harvest that raises loses its sweep's results SILENTLY. The delete
   proceeds either way -- retention must never be blocked by bookkeeping --
   so the residue is a sweep row that reads as if it simply forgot. It now
   records WHY on its own row.
B. ``cancelling`` is an acknowledgement, not a promise: cancellation is
   cooperative, so a node that never polls ``should_stop()`` holds a sweep
   there indefinitely and no timeout will ever move it. The state's meaning
   is now written down at its definition point, and the half a machine can
   read -- "cancelling is not settled" -- is a set this module asserts on.
C. "The last point of the named series" had TWO implementations. They now
   share one, and the agreement is asserted over the three shapes that
   could have pulled them apart: many series, a NULL last point, and a run
   ``latest_metrics`` does not return at all.

Fixtures and helpers are local copies rather than imports from
test_sweeps.py, following the house rule that module states in its own
header: test modules stay independent.
"""

from __future__ import annotations

import logging
import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest

from app.api import routes_sweeps
from app.core.db import Database
from app.core.run_store import (
    STATUS_RUNNING,
    TERMINAL_STATUSES,
    MetricPoint,
    RunProvenance,
    RunStore,
    last_metric_values,
)
from app.core.sweep_store import (
    SWEEP_SETTLED_STATES,
    SWEEP_STATE_CANCELLING,
    SWEEP_STATE_FAILED,
    SWEEP_STATE_FINISHED,
    SWEEP_STATE_RUNNING,
    SWEEP_STATES,
    HarvestEntry,
    SweepStore,
    SweepVariant,
    _last_metric_value,
    _record_harvest_failure,
    variant_is_terminal,
)


# ── fixtures and helpers ──────────────────────────────────────────────────


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
def sweeps(db):
    return SweepStore(db)


@pytest.fixture
def runs(db):
    return RunStore(db)


def _variant(index: int, **overrides) -> SweepVariant:
    fields: dict[str, Any] = {
        "index": index, "domain_index": index, "run_id": None,
        "params": [{"node_id": "probe", "param": "lr", "value": 0.1}],
        "seed": None, "objective": None, "status": None,
        "harvested_at": None,
    }
    fields.update(overrides)
    return SweepVariant(**fields)


async def _new_sweep(store: SweepStore, count: int = 1, **overrides):
    fields: dict[str, Any] = {
        "method": "grid", "seed": 1, "seed_variants": False,
        "spec": {"method": "grid", "params": []},
        "objective": {"metric": "val_loss", "direction": "minimize"},
        "variants": [_variant(i) for i in range(count)],
        "name": "lr sweep",
    }
    fields.update(overrides)
    return await store.create_sweep(**fields)


async def _child(store: RunStore, sweep_id: str, index: int, *,
                 status: str = "succeeded",
                 points: list[MetricPoint] | None = None) -> str:
    """One child run, logged point by point.

    Points are flushed ONE AT A TIME: write order is the tie-break for two
    points that share a step, so batching them into a single flush would
    make that half of the rule untestable.
    """
    record = await store.create_run(
        graph_snapshot={"nodes": [], "edges": []}, options={},
        provenance=RunProvenance(), sweep_id=sweep_id, sweep_variant=index)
    for point in (points or []):
        await store.log_metrics(record.id, [point])
    if status in TERMINAL_STATUSES:
        await store.mark_finished(record.id, status)
    elif status == STATUS_RUNNING:
        await store.mark_running(record.id)
    return record.id


async def _attach(sweeps: SweepStore, runs: RunStore, sweep_id: str,
                  index: int, **kwargs) -> str:
    run_id = await _child(runs, sweep_id, index, **kwargs)
    await sweeps.set_variant_run(sweep_id, index, run_id=run_id, seed=None)
    return run_id


async def _seam_a(runs: RunStore, sweeps: SweepStore, sweep_id: str):
    """``routes_sweeps._harvested_sweep``'s rule, minus the HTTP.

    Copied deliberately rather than reached through the app: seam A is a
    STORE-level rule (harvest what is terminal, settle when everything is)
    and these tests are about that rule, not about FastAPI. The expression
    that reads the objective --
    ``metrics.get(run_id, {}).get(metric) if metric else None`` -- is the
    route's own, verbatim, because that is the exact level at which the two
    seams have to agree.
    """
    sweep = await sweeps.get_sweep(sweep_id)
    children = {record.id: record
                for record in await runs.list_runs_by_sweep(sweep_id)}
    metrics = await runs.latest_metrics(list(children))
    metric = sweep.objective.get("metric")
    entries: dict[int, HarvestEntry] = {}
    for variant in sweep.variants:
        if variant.harvested_at is not None or variant.run_id is None:
            continue
        child = children.get(variant.run_id)
        if child is None or child.status not in TERMINAL_STATUSES:
            continue
        entries[variant.index] = HarvestEntry(
            objective=(metrics.get(variant.run_id, {}).get(metric)
                       if metric else None),
            status=child.status)
    finished = all(
        variant.index in entries
        or variant_is_terminal(
            variant,
            children[variant.run_id].status
            if variant.run_id in children else None,
            child_exists=variant.run_id in children)
        for variant in sweep.variants)
    return await sweeps.harvest(sweep_id, entries=entries, finished=finished)


async def _seam_a_objective(runs: RunStore, run_id: str,
                            metric: str | None) -> float | None:
    """Seam A's answer for ONE run, at the objective level."""
    metrics = await runs.latest_metrics([run_id])
    return metrics.get(run_id, {}).get(metric) if metric else None


async def _corrupt_variants(db: Database, sweep_id: str) -> None:
    """Make one ``sweeps`` row unreadable -- a JSON object where the blob
    must be an array, which ``SweepRecord.from_row`` refuses to paper over
    rather than turning into a plausible empty sweep."""
    await db.run(lambda conn: conn.execute(
        "UPDATE sweeps SET variants = ? WHERE id = ?",
        ('{"not": "a list"}', sweep_id)))


# ── A: a harvest that raises says so on the sweep row ─────────────────────


async def test_a_sweep_whose_harvest_failed_records_why_on_its_own_row(
        db, sweeps, runs):
    """The residue #404 names: results gone, and nothing on the row to say so.

    The DELETE proceeds -- that is the non-negotiable half, because
    retention is unattended, irreversible and the only thing bounding this
    table, its cascaded children, checkpoint files and TensorBoard logdirs.
    So the sweep's numbers really are gone for good, and the only question
    left is whether a reader can tell the difference between "the harvest
    could not run" and "this sweep never produced anything". Without a
    record they look identical: an empty comparison table either way.
    """
    broken = await _new_sweep(sweeps, count=1)
    run_id = await _attach(sweeps, runs, broken.id, 0,
                           points=[MetricPoint("val_loss", 0.7, 0)])
    await _corrupt_variants(db, broken.id)

    assert await runs.prune(keep_last=0) == 1        # retention not blocked
    assert await runs.get_run(run_id) is None        # the child really went

    # The blob is still unreadable, so the row is read as a row.
    row = await db.run(lambda conn: conn.execute(
        "SELECT state, error FROM sweeps WHERE id = ?",
        (broken.id,)).fetchone())
    assert row["error"] is not None
    assert "harvest" in row["error"]
    # Names the cause, not just "something went wrong".
    assert "expected a JSON array" in row["error"]
    # The lifecycle is untouched: the harvest failing says nothing about
    # which variants are terminal, and `failed` is reserved for a sweep
    # whose SUBMIT loop broke.
    assert row["state"] == SWEEP_STATE_RUNNING


async def test_the_recorded_harvest_failure_reaches_a_reader_of_the_row(
        sweeps, runs, monkeypatch):
    """The readable half: a harvest can fail without the blob being corrupt.

    A transient sqlite error, or a variant entry the patch cannot re-encode,
    leaves a sweep whose row still parses -- and that is the case a user
    actually meets, because ``GET /api/sweeps/{id}`` serialises
    ``sweep.error`` into every response body. So the recorded reason has to
    survive a normal read, not only a raw SELECT.
    """
    sweep = await _new_sweep(sweeps, count=1)
    await _attach(sweeps, runs, sweep.id, 0,
                  points=[MetricPoint("val_loss", 0.4, 0)])

    def _boom(conn, sweep_id, rows, doomed_ids, stamp):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("app.core.sweep_store._harvest_one_sweep", _boom)
    assert await runs.prune(keep_last=0) == 1

    fetched = await sweeps.get_sweep(sweep.id)
    assert fetched.error is not None
    assert "OperationalError" in fetched.error
    assert "database is locked" in fetched.error
    # A count, so the reader knows how much was lost rather than just that
    # something was.
    assert "1" in fetched.error


async def test_a_recorded_harvest_failure_never_overwrites_a_failed_submit(
        db, sweeps, runs):
    """``mark_failed``'s record outranks this one, the rule ``set_state``
    already follows.

    A sweep whose submit loop broke has variants with no run at all, and
    THAT is why its table is short. Overwriting it with a later retention
    note would replace the first cause with the second one.
    """
    sweep = await _new_sweep(sweeps, count=1)
    await _attach(sweeps, runs, sweep.id, 0,
                  points=[MetricPoint("val_loss", 0.4, 0)])
    await sweeps.mark_failed(sweep.id, "run service is shutting down")
    await _corrupt_variants(db, sweep.id)

    assert await runs.prune(keep_last=0) == 1
    row = await db.run(lambda conn: conn.execute(
        "SELECT state, error FROM sweeps WHERE id = ?",
        (sweep.id,)).fetchone())
    assert row["error"] == "run service is shutting down"
    assert row["state"] == SWEEP_STATE_FAILED


def test_a_failure_to_record_the_failure_cannot_block_the_delete(caplog):
    """Bookkeeping about bookkeeping is still bookkeeping.

    ``_record_harvest_failure`` runs INSIDE ``prune``'s transaction, one
    statement before the DELETE. If it could raise it would abort retention
    for every run in the pass -- exactly the failure #404's first paragraph
    says the isolation exists to prevent, reintroduced by the fix for it.
    """
    class _BrokenConn:
        def execute(self, *args, **kwargs):
            raise sqlite3.OperationalError("no such table: sweeps")

    with caplog.at_level(logging.WARNING, logger="app.core.sweep_store"):
        _record_harvest_failure(_BrokenConn(), "sweep-42", [],
                                ValueError("boom"))
    assert "sweep-42" in caplog.text


async def test_one_unrecordable_sweep_does_not_cost_the_others_theirs(
        db, sweeps, runs):
    """Per-sweep isolation still holds with the record in place.

    The recording write is inside the per-sweep handler, so a second,
    healthy sweep in the same pass must still be harvested normally --
    the property #404 credits as "the right call" and that this change
    must not undo.
    """
    broken = await _new_sweep(sweeps, count=1, name="broken")
    healthy = await _new_sweep(sweeps, count=1, name="healthy")
    for sweep, value in ((broken, 0.7), (healthy, 0.25)):
        await _attach(sweeps, runs, sweep.id, 0,
                      points=[MetricPoint("val_loss", value, 0)])
    await _corrupt_variants(db, broken.id)

    assert await runs.prune(keep_last=0) == 2
    fetched = await sweeps.get_sweep(healthy.id)
    assert fetched.variants[0].objective == 0.25
    assert fetched.error is None                     # nothing to report
    assert fetched.state == SWEEP_STATE_FINISHED


# ── B: cancelling is an acknowledgement, not a promise ────────────────────


def test_cancelling_is_not_a_settled_state():
    """The machine-readable half of what ``cancelling`` means.

    A sweep in ``cancelling`` has not finished and has not failed: a stop
    was REQUESTED, and cancellation is cooperative, so whether it happens
    is up to the node. Classifying it with ``running`` rather than with the
    terminal states is what keeps a client polling instead of filing the
    sweep as over.
    """
    assert SWEEP_SETTLED_STATES == {SWEEP_STATE_FINISHED, SWEEP_STATE_FAILED}
    assert SWEEP_STATE_CANCELLING not in SWEEP_SETTLED_STATES
    assert SWEEP_STATE_RUNNING not in SWEEP_SETTLED_STATES


def test_every_sweep_state_is_deliberately_classified():
    """A new state must be settled or not -- never settled by default.

    Left unclassified, a future state would land on the "still going" side
    by omission, and a sweep that is over would be polled forever.
    """
    assert SWEEP_SETTLED_STATES <= SWEEP_STATES
    assert SWEEP_STATES - SWEEP_SETTLED_STATES == {
        SWEEP_STATE_RUNNING, SWEEP_STATE_CANCELLING}


def test_the_read_path_uses_the_same_settled_set():
    """The read path asks the store which states are settled.

    It used to keep a private ``_SETTLED_STATES`` of its own, so "which
    states are final" had three spellings: this constant, the tuple inside
    ``_write_variants``, and the vocabulary itself. An equality assertion
    over two copies only reports the drift after it has happened; an
    IDENTITY assertion over one object says there is nothing to drift.
    The private name is asserted GONE so it cannot quietly come back.
    """
    assert routes_sweeps.SWEEP_SETTLED_STATES is SWEEP_SETTLED_STATES
    assert not hasattr(routes_sweeps, "_SETTLED_STATES")


async def test_a_cancelling_sweep_whose_child_never_stops_stays_cancelling(
        db, sweeps, runs):
    """No timeout, and no drift toward one.

    #404: a node that ignores ``context.should_stop()`` never stops, so the
    sweep sits at ``cancelling``. That is the honest report -- the run
    really is still running -- and the fix is NOT to invent a deadline,
    because killing a training run mid-step has its own costs. This pins
    the absence: five full harvest-and-prune rounds, and the state moves
    only when the CHILD moves.
    """
    sweep = await _new_sweep(sweeps, count=1)
    run_id = await _attach(sweeps, runs, sweep.id, 0, status=STATUS_RUNNING,
                           points=[MetricPoint("val_loss", 0.4, 0)])
    assert await sweeps.set_state(sweep.id, SWEEP_STATE_CANCELLING)

    for _ in range(5):
        # Retention never deletes an active run, so the child survives and
        # seam B has nothing to harvest.
        assert await runs.prune(keep_last=0) == 0
        record = await _seam_a(runs, sweeps, sweep.id)
        assert record.state == SWEEP_STATE_CANCELLING
        assert record.finished_at is None
        assert record.variants[0].harvested_at is None

    # The one thing that DOES move it: the child becoming terminal.
    assert await runs.mark_finished(run_id, "cancelled")
    record = await _seam_a(runs, sweeps, sweep.id)
    assert record.state == SWEEP_STATE_FINISHED
    assert record.finished_at is not None
    assert record.variants[0].status == "cancelled"
    assert record.variants[0].objective == 0.4     # a stopped run's numbers


async def test_a_cancelling_sweep_settles_on_its_last_child_not_its_first(
        db, sweeps, runs):
    """Half-stopped is still ``cancelling``.

    A cancel is N separate cooperative requests with no transaction across
    them, so a partially-stopped sweep is a reachable, ordinary state. It
    must keep saying ``cancelling`` until the LAST child is terminal.
    """
    sweep = await _new_sweep(sweeps, count=2)
    first = await _attach(sweeps, runs, sweep.id, 0, status=STATUS_RUNNING,
                          points=[MetricPoint("val_loss", 0.8, 0)])
    second = await _attach(sweeps, runs, sweep.id, 1, status=STATUS_RUNNING,
                           points=[MetricPoint("val_loss", 0.2, 0)])
    assert await sweeps.set_state(sweep.id, SWEEP_STATE_CANCELLING)

    assert await runs.mark_finished(first, "cancelled")
    record = await _seam_a(runs, sweeps, sweep.id)
    assert record.state == SWEEP_STATE_CANCELLING
    assert record.variants[0].status == "cancelled"  # harvested all the same
    assert record.variants[1].status is None

    assert await runs.mark_finished(second, "succeeded")
    record = await _seam_a(runs, sweeps, sweep.id)
    assert record.state == SWEEP_STATE_FINISHED


async def test_a_settled_sweep_is_never_re_stamped(db, sweeps, runs):
    """The other half of the classification, through production code.

    ``finished`` is stamped once (re-stamping would move ``finished_at`` on
    every poll) and ``failed`` is never overwritten (it is the sweep's own
    record of what went wrong). Both are exactly "is this state settled?",
    so they are asserted against the set rather than against two literals.
    """
    done = await _new_sweep(sweeps, count=1)
    await _attach(sweeps, runs, done.id, 0,
                  points=[MetricPoint("val_loss", 0.4, 0)])
    settled = await _seam_a(runs, sweeps, done.id)
    assert settled.state == SWEEP_STATE_FINISHED
    stamped_at = settled.finished_at
    assert stamped_at is not None
    again = await sweeps.harvest(done.id, entries={}, finished=True)
    assert again.finished_at == stamped_at

    broke = await _new_sweep(sweeps, count=1)
    await _attach(sweeps, runs, broke.id, 0,
                  points=[MetricPoint("val_loss", 0.4, 0)])
    await sweeps.mark_failed(broke.id, "run service is shutting down")
    after = await _seam_a(runs, sweeps, broke.id)
    assert after.state == SWEEP_STATE_FAILED
    assert after.error == "run service is shutting down"


# ── C: one implementation of "the last point of the named series" ─────────


def test_both_seams_share_one_implementation_of_the_last_point_rule():
    """Not "two implementations that agree" -- ONE.

    Agreement asserted over sample inputs is only ever a statement about
    the samples. The seams cannot diverge on a shape nobody thought of if
    there is nothing to diverge FROM, which is what #404 means by "sharing
    is preferred".
    """
    import app.core.run_store as run_store_module
    import app.core.sweep_store as sweep_store_module

    assert (sweep_store_module.last_metric_values
            is run_store_module.last_metric_values)


async def test_the_two_seams_agree_over_many_series(db, sweeps, runs):
    """Divergence case 1: more than one series in the run.

    The existing agreement test feeds ONE series, which is the shape on
    which a leapfrog over every series and a single-name seek cannot
    disagree. With several series, a per-name seek that leaked the wrong
    name -- or a leapfrog whose per-series subquery drifted -- shows up
    here and nowhere else. The series are built so the answer is not at
    either end of the write order and not the best value either: steps
    arrive out of order, and the last step is a tie broken by write order.
    """
    sweep = await _new_sweep(sweeps, count=1)
    run_id = await _attach(sweeps, runs, sweep.id, 0, points=[
        MetricPoint("accuracy", 0.10, 5),
        MetricPoint("val_loss", 0.9, 1),
        MetricPoint("train_loss", 9.9, 7),
        MetricPoint("val_loss", 0.1, 3, "node-a"),   # best, and NOT last
        MetricPoint("accuracy", 0.99, 1),
        MetricPoint("val_loss", 0.7, 0),
        MetricPoint("val_loss", 0.5, 3, "node-b"),   # same step, later write
        MetricPoint("val_loss", 0.3, 2),
        MetricPoint("train_loss", 1.1, 2),
    ])

    seam_a = await _seam_a_objective(runs, run_id, "val_loss")
    # Every OTHER series is answered by the same rule, so a leak between
    # them would show up as a wrong number here rather than a missing one.
    assert await runs.latest_metrics([run_id]) == {run_id: {
        "accuracy": 0.10, "train_loss": 9.9, "val_loss": 0.5}}

    assert await runs.prune(keep_last=0) == 1     # seam B, no prior harvest
    seam_b = (await sweeps.get_sweep(sweep.id)).variants[0].objective
    assert seam_b == seam_a == 0.5


async def test_the_two_seams_agree_when_the_last_point_is_null(db, sweeps,
                                                               runs):
    """Divergence case 2: a NULL last point on one side.

    A diverged loss is stored as SQL NULL. ``latest_metrics`` OMITS that
    series from its map (rendering it as 0.0 would show a diverged run as
    the best one); the seam-B seek reads the NULL back as None. Those are
    the same answer spelled two ways, and the place they have to agree is
    the objective, so that is where this asserts. The run keeps a second,
    healthy series, so it is the SERIES that is missing from seam A's
    answer and not the run.
    """
    sweep = await _new_sweep(sweeps, count=1)
    run_id = await _attach(sweeps, runs, sweep.id, 0, points=[
        MetricPoint("val_loss", 0.2, 0),             # finite, but NOT last
        MetricPoint("val_loss", float("nan"), 1),    # diverged, and last
        MetricPoint("accuracy", 0.5, 1),
    ])

    seam_a = await _seam_a_objective(runs, run_id, "val_loss")
    assert seam_a is None
    # The run itself is still answered -- only the diverged series is gone.
    assert await runs.latest_metrics([run_id]) == {run_id: {"accuracy": 0.5}}

    assert await runs.prune(keep_last=0) == 1
    variant = (await sweeps.get_sweep(sweep.id)).variants[0]
    assert variant.objective is seam_a is None
    # Harvested, no value -- a recorded fact, not a retry, and emphatically
    # not a fabricated 0.0 that would rank BEST under minimize.
    assert variant.status == "succeeded"
    assert variant.harvested_at is not None


async def test_the_two_seams_agree_on_a_run_seam_a_never_returns(db, sweeps,
                                                                 runs):
    """Divergence case 3: a run ``latest_metrics`` omits ENTIRELY.

    It drops a run whose every series is unusable, and a run that logged
    nothing at all was never in the map to begin with. Seam A's caller then
    reads the objective off a missing key; seam B reads it off a missing
    row. Both shapes, because they fail on different sides of the same
    ``.get``.
    """
    sweep = await _new_sweep(sweeps, count=2)
    diverged = await _attach(sweeps, runs, sweep.id, 0, points=[
        MetricPoint("val_loss", float("inf"), 0)])   # its ONLY series
    silent = await _attach(sweeps, runs, sweep.id, 1)   # no metrics at all

    assert await runs.latest_metrics([diverged, silent]) == {}
    seam_a = [await _seam_a_objective(runs, run_id, "val_loss")
              for run_id in (diverged, silent)]
    assert seam_a == [None, None]

    assert await runs.prune(keep_last=0) == 2
    fetched = await sweeps.get_sweep(sweep.id)
    assert [v.objective for v in fetched.variants] == seam_a
    assert [v.status for v in fetched.variants] == ["succeeded", "succeeded"]


async def test_seam_b_reads_exactly_what_the_shared_rule_returns(db, sweeps,
                                                                 runs):
    """The seam-B half of the sharing, on one connection, case by case.

    ``_last_metric_value`` adds exactly one thing to the shared rule: an
    empty or absent metric name answers None rather than looking anything
    up. Everything else must be the shared map's own answer, including for
    a series that does not exist and for a run that does not exist.
    """
    sweep = await _new_sweep(sweeps, count=1)
    run_id = await _attach(sweeps, runs, sweep.id, 0, points=[
        MetricPoint("val_loss", 0.4, 0),
        MetricPoint("val_loss", 0.2, 2),
        MetricPoint("accuracy", float("nan"), 1),
    ])

    def _compare(conn: sqlite3.Connection) -> list[tuple[Any, Any]]:
        shared = last_metric_values(conn, run_id)
        return [(shared.get(name), _last_metric_value(conn, run_id, name))
                for name in ("val_loss", "accuracy", "never_logged")]

    assert await db.run(_compare) == [(0.2, 0.2), (None, None), (None, None)]

    def _edges(conn: sqlite3.Connection) -> list[Any]:
        return [_last_metric_value(conn, "no-such-run", "val_loss"),
                _last_metric_value(conn, run_id, None),
                _last_metric_value(conn, run_id, "")]

    assert await db.run(_edges) == [None, None, None]


async def test_the_read_path_answers_none_for_an_empty_metric_name(
        db, sweeps, runs):
    """Seam A, through the real route helper, on the one name seam B guards.

    #483: ``_last_metric_value`` answers None for an empty name, and the read
    path looked ``""`` up like any other key. A node can log a series named
    ``""`` (``log_metric`` stores ``str(name)`` unchecked), and a sweeps row
    carrying ``"metric": ""`` outlives the route validation that would have
    refused it. Both seams write the objective onto that row, so both have to
    give the same answer.
    """
    sweep = await _new_sweep(
        sweeps, count=1, objective={"metric": "", "direction": "minimize"})
    run_id = await _attach(sweeps, runs, sweep.id, 0,
                           points=[MetricPoint("", 0.3, 0)])
    # The series really is there, so a None below is the rule answering and
    # not a lookup that missed.
    assert await runs.latest_metrics([run_id]) == {run_id: {"": 0.3}}

    record, _children, _metrics = await routes_sweeps._harvested_sweep(
        SimpleNamespace(store=runs), sweeps, sweep.id)
    variant = record.variants[0]
    assert variant.objective is None
    # Harvested with no value, which is what seam B records for it.
    assert variant.status == "succeeded"
    assert variant.harvested_at is not None
