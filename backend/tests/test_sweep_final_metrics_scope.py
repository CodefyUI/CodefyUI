"""A sweep variant's ``final_metrics`` prefer the objective's node (#841).

The 3.1.0 E2E audit (U23) swept ``lr`` of one branch of the Weight decay
example, which trains two models side by side (``train-plain`` and
``train-decay``, each logging ``lr`` and ``train_loss``). The objective was
``train_loss`` of ``train-plain`` and ranked correctly, but every variant's
"Final metrics" read ``lr 0.01``: the run-wide summary collapses a name two
nodes log into whichever wrote last, which was the OTHER branch.

The summary stays run-wide, so a series only another node logged (an
Evaluate node's ``val_acc``) is kept, and the objective node's own last
values replace the run-wide ones for every name it logged. These tests pin
that overlay at the store, at the sweeps route (JSON, the level the Runs
panel reads) and its warning, and at the study harvest, which had the same
collapse.

Fixtures and helpers are local, following the house rule that test modules
stay independent.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api import routes_sweeps
from app.core.db import Database
from app.core.run_store import (
    MetricPoint,
    RunProvenance,
    RunStore,
    last_metric_values,
    objective_metric_values,
)
from app.core.sweep_store import SweepStore, SweepVariant

MIN = "minimize"


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


@pytest.fixture
async def http(runs, sweeps):
    api = FastAPI()
    api.include_router(routes_sweeps.router)
    api.state.run_service = SimpleNamespace(store=runs)
    api.state.sweep_store = sweeps
    async with AsyncClient(transport=ASGITransport(app=api),
                           base_url="http://test") as client:
        yield client


SWEPT = [0.003, 0.001]


def _variant(index: int) -> SweepVariant:
    return SweepVariant(
        index=index, domain_index=index, run_id=None,
        params=[{"node_id": "opt-plain", "param": "lr",
                 "value": SWEPT[index]}],
        seed=None, objective=None, status=None, harvested_at=None)


async def _sweep(sweeps: SweepStore, objective: dict[str, Any]):
    return await sweeps.create_sweep(
        method="grid", seed=None, seed_variants=False,
        spec={"method": "grid", "params": [
            {"node_id": "opt-plain", "param": "lr", "domain": SWEPT}]},
        objective=objective,
        variants=[_variant(i) for i in range(len(SWEPT))],
        name="weight decay")


def _two_branches(lr: float, loss: float) -> list[MetricPoint]:
    """The audit's run: the swept branch first, the fixed branch (lr 0.01)
    written LAST at the same steps, so the run-wide summary reads it. Two
    more nodes each log a series no other node logs."""
    points: list[MetricPoint] = []
    for step in (0, 1):
        points += [
            MetricPoint("lr", lr, step, "train-plain"),
            MetricPoint("train_loss", loss + 1 - step, step, "train-plain"),
            MetricPoint("lr", 0.01, step, "train-decay"),
            MetricPoint("train_loss", 0.058 + 1 - step, step, "train-decay"),
        ]
    points.append(MetricPoint("mse", 0.04, 0, "mse-plain"))
    points.append(MetricPoint("val_acc", 0.8, 1, "eval-plain"))
    return points


async def _child(runs: RunStore, sweeps: SweepStore, sweep_id: str,
                 index: int, points: list[MetricPoint]) -> str:
    record = await runs.create_run(
        graph_snapshot={"nodes": [], "edges": []}, options={},
        provenance=RunProvenance(), sweep_id=sweep_id, sweep_variant=index)
    for point in points:
        await runs.log_metrics(record.id, [point])
    await runs.mark_finished(record.id, "succeeded")
    await sweeps.set_variant_run(sweep_id, index, run_id=record.id, seed=None)
    return record.id


async def _get(http: AsyncClient, sweep_id: str) -> dict[str, Any]:
    response = await http.get(f"/api/sweeps/{sweep_id}")
    assert response.status_code == 200, response.text
    return response.json()


async def test_a_named_objective_node_wins_its_names_and_keeps_the_rest(
        runs, sweeps, http):
    """U23: each variant shows ITS swept lr and its own loss, never the
    fixed branch's lr 0.01, and still shows what other nodes logged."""
    sweep = await _sweep(sweeps, {"metric": "train_loss", "direction": MIN,
                                  "node_id": "train-plain"})
    run_ids = [await _child(runs, sweeps, sweep.id, i,
                            _two_branches(lr, 0.05 + i / 100))
               for i, lr in enumerate(SWEPT)]

    body = await _get(http, sweep.id)
    by_index = {v["index"]: v for v in body["variants"]}
    for index, lr in enumerate(SWEPT):
        variant = by_index[index]
        assert variant["final_metrics"] == {
            "lr": lr, "train_loss": pytest.approx(0.05 + index / 100),
            "mse": 0.04, "val_acc": 0.8}
        assert variant["final_metrics_node_id"] == "train-plain"
        assert variant["objective"] == pytest.approx(0.05 + index / 100)
    # The Runs list summary of the same run is unchanged: run-wide.
    assert (await runs.latest_metrics([run_ids[0]]))[run_ids[0]]["lr"] == 0.01


async def test_a_name_only_single_producer_is_preferred_once_harvested(
        runs, sweeps, http):
    """A name-only objective one node logged: the harvest names the node,
    and the rest of the summary is the run's."""
    sweep = await _sweep(sweeps, {"metric": "mse", "direction": MIN})
    await _child(runs, sweeps, sweep.id, 0, _two_branches(0.003, 0.05))

    variant = next(v for v in (await _get(http, sweep.id))["variants"]
                   if v["index"] == 0)
    assert variant["objective_node_id"] == "mse-plain"
    assert variant["final_metrics_node_id"] == "mse-plain"
    assert variant["final_metrics"] == {
        "mse": 0.04, "val_acc": 0.8, "lr": 0.01,
        "train_loss": pytest.approx(0.058)}


async def test_an_ambiguous_objective_keeps_the_run_wide_summary(
        runs, sweeps, http):
    """Several producers and no node named: the variant is unranked and its
    summary is the plain run-wide one, labelled as such by a null node."""
    sweep = await _sweep(sweeps, {"metric": "train_loss", "direction": MIN})
    await _child(runs, sweeps, sweep.id, 0, _two_branches(0.003, 0.05))

    variant = next(v for v in (await _get(http, sweep.id))["variants"]
                   if v["index"] == 0)
    assert variant["ambiguous_producers"] == ["train-decay", "train-plain"]
    assert variant["final_metrics_node_id"] is None
    assert variant["final_metrics"]["lr"] == 0.01
    assert variant["final_metrics"]["mse"] == 0.04


async def test_an_unstarted_variant_has_no_summary_and_no_node(
        runs, sweeps, http):
    sweep = await _sweep(sweeps, {"metric": "train_loss", "direction": MIN,
                                  "node_id": "train-plain"})
    variant = next(v for v in (await _get(http, sweep.id))["variants"]
                   if v["index"] == 1)
    assert variant["run_id"] is None
    assert "final_metrics" not in variant
    assert "final_metrics_node_id" not in variant


async def test_a_named_node_that_ranked_nothing_is_warned_with_the_runs_series(
        runs, sweeps, http):
    """The warning lists every series the run recorded, run-wide, and names
    the nodes that did log the objective's metric."""
    sweep = await _sweep(sweeps, {"metric": "train_loss", "direction": MIN,
                                  "node_id": "eval-plain"})
    await _child(runs, sweeps, sweep.id, 0, _two_branches(0.003, 0.05))

    body = await _get(http, sweep.id)
    assert body["objective_warning"] == (
        "no variant recorded a metric named 'train_loss' from node "
        "'eval-plain'; 'train_loss' was recorded by train-decay, "
        "train-plain; the series recorded across this sweep were: lr, mse, "
        "train_loss, val_acc")


async def test_a_named_node_that_diverged_is_named_with_the_reason(
        runs, sweeps, http):
    """#848 review: the named node DID log the metric, but its last point
    was non-finite. The warning says so instead of pointing only at the
    other branch, which would be the wrong fix."""
    sweep = await _sweep(sweeps, {"metric": "train_loss", "direction": MIN,
                                  "node_id": "train-plain"})
    points = _two_branches(0.003, 0.05)
    points.append(MetricPoint("train_loss", float("nan"), 2, "train-plain"))
    await _child(runs, sweeps, sweep.id, 0, points)

    warning = (await _get(http, sweep.id))["objective_warning"]
    assert warning.startswith(
        "no variant recorded a metric named 'train_loss' from node "
        "'train-plain'; 'train-plain' logged 'train_loss', but its last "
        "value was not finite (NaN or infinite); 'train_loss' was also "
        "recorded by train-decay; ")


async def test_a_cancel_and_the_csv_do_not_read_the_metrics(
        runs, sweeps, http, monkeypatch):
    """#848 review: neither shows ``final_metrics``, so neither pays for
    the summary read."""
    sweep = await _sweep(sweeps, {"metric": "train_loss", "direction": MIN,
                                  "node_id": "train-plain"})
    await _child(runs, sweeps, sweep.id, 0, _two_branches(0.003, 0.05))
    reads: list[Any] = []
    real = runs.latest_metrics

    async def counted(*args, **kwargs):
        reads.append(args)
        return await real(*args, **kwargs)

    monkeypatch.setattr(runs, "latest_metrics", counted)
    response = await http.get(f"/api/sweeps/{sweep.id}?format=csv")
    assert response.status_code == 200
    assert reads == []

    async def no_such_run(_run_id):
        return None

    service = http._transport.app.state.run_service  # type: ignore[attr-defined]
    service.cancel = no_such_run
    response = await http.post(f"/api/sweeps/{sweep.id}/cancel")
    assert response.status_code == 200, response.text
    assert reads == []
    await _get(http, sweep.id)
    assert len(reads) == 1


async def test_a_named_node_whose_metric_nobody_logged_names_no_producer(
        runs, sweeps, http):
    sweep = await _sweep(sweeps, {"metric": "val_loss", "direction": MIN,
                                  "node_id": "eval-plain"})
    await _child(runs, sweeps, sweep.id, 0, _two_branches(0.003, 0.05))

    warning = (await _get(http, sweep.id))["objective_warning"]
    assert "was recorded by" not in warning
    assert warning.endswith("lr, mse, train_loss, val_acc")


async def test_objective_metric_values_overlays_the_node_on_the_run(db, runs):
    record = await runs.create_run(
        graph_snapshot={"nodes": [], "edges": []}, options={},
        provenance=RunProvenance())
    await runs.log_metrics(record.id, [
        MetricPoint("lr", 0.003, 0, "train-plain"),
        MetricPoint("train_loss", 0.5, 0, "train-plain"),
        MetricPoint("train_loss", float("nan"), 1, "train-plain"),
        MetricPoint("lr", 0.01, 5, "train-decay"),
        MetricPoint("train_loss", 0.4, 5, "train-decay"),
        MetricPoint("acc", 0.9, 0, None),
    ])
    # lr: the node's own; train_loss: the node diverged, so no number even
    # though train-decay has one; acc: only the run level logged it, kept.
    assert await db.run(lambda conn: objective_metric_values(
        conn, record.id, "train-plain")) == {"lr": 0.003, "acc": 0.9}
    # A node that logged nothing leaves the run-wide summary as it is.
    assert await db.run(lambda conn: objective_metric_values(
        conn, record.id, "nobody")) == await db.run(
        lambda conn: last_metric_values(conn, record.id))
    # Overlaid and plain runs in ONE call.
    other = await runs.create_run(
        graph_snapshot={"nodes": [], "edges": []}, options={},
        provenance=RunProvenance())
    await runs.log_metrics(other.id, [MetricPoint("lr", 0.2, 0, "x")])
    assert await runs.latest_metrics(
        [record.id, other.id], producers={record.id: "train-plain"}) == {
        record.id: {"lr": 0.003, "acc": 0.9}, other.id: {"lr": 0.2}}


async def test_objective_metric_values_seeks_and_never_scans(db, runs):
    """The overlay is ONE statement per run, and every touch of the metrics
    table in it is an index SEARCH -- the series leapfrog, and a producer
    seek per name -- so its cost tracks the series, never the points."""
    record = await runs.create_run(
        graph_snapshot={"nodes": [], "edges": []}, options={},
        provenance=RunProvenance())
    await runs.log_metrics(record.id, [
        MetricPoint(name, 1.0, step, node)
        for name in ("loss", "lr", "val_acc") for node in ("a", "b")
        for step in range(20)])

    executed: list[str] = []

    def _run(conn: sqlite3.Connection) -> dict[str, float]:
        conn.set_trace_callback(executed.append)
        try:
            return objective_metric_values(conn, record.id, "a")
        finally:
            conn.set_trace_callback(None)

    assert await db.run(_run) == {"loss": 1.0, "lr": 1.0, "val_acc": 1.0}
    reads = [sql for sql in executed if "exec_run_metrics" in sql]
    # ONE statement per run, whatever the number of series (#848 review).
    assert len(reads) == 1, executed

    def _plans(conn: sqlite3.Connection) -> list[str]:
        return [" ".join(row["detail"] for row in conn.execute(
            "EXPLAIN QUERY PLAN " + sql).fetchall()) for sql in reads]

    for plan in await db.run(_plans):
        assert "SCAN exec_run_metrics" not in plan, plan
        assert "TEMP B-TREE" not in plan, plan
        assert "SEARCH" in plan, plan
        assert "idx_exec_run_metrics_producer" in plan, plan
