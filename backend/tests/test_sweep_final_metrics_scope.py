"""A sweep variant's ``final_metrics`` come from the objective's node (#841).

The 3.1.0 E2E audit (U23) swept ``lr`` of one branch of the Weight decay
example, which trains two models side by side (``train-plain`` and
``train-decay``, each logging ``lr`` and ``train_loss``). The objective was
``train_loss`` of ``train-plain`` and ranked correctly, but every variant's
"Final metrics" read ``lr 0.01``: the run-wide summary collapses a name two
nodes log into whichever wrote last, which was the OTHER branch.

These tests pin the scoped summary at the store, at the sweeps route (JSON,
the level the Runs panel reads) and at the study harvest, which had the same
collapse.

Fixtures and helpers are local, following the house rule that test modules
stay independent.
"""

from __future__ import annotations

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
    producer_metric_values,
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
    written LAST at the same steps, so the run-wide summary reads it."""
    points: list[MetricPoint] = []
    for step in (0, 1):
        points += [
            MetricPoint("lr", lr, step, "train-plain"),
            MetricPoint("train_loss", loss + 1 - step, step, "train-plain"),
            MetricPoint("lr", 0.01, step, "train-decay"),
            MetricPoint("train_loss", 0.058 + 1 - step, step, "train-decay"),
        ]
    points.append(MetricPoint("mse", 0.04, 0, "mse-plain"))
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


async def test_a_named_objective_node_scopes_final_metrics_to_that_branch(
        runs, sweeps, http):
    """U23: each variant shows ITS swept lr and its own loss, never the
    fixed branch's lr 0.01."""
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
            "lr": lr, "train_loss": pytest.approx(0.05 + index / 100)}
        assert variant["final_metrics_node_id"] == "train-plain"
        assert variant["objective"] == pytest.approx(0.05 + index / 100)
    # The Runs list summary of the same run is unchanged: run-wide.
    assert (await runs.latest_metrics([run_ids[0]]))[run_ids[0]]["lr"] == 0.01


async def test_a_name_only_single_producer_scopes_once_harvested(
        runs, sweeps, http):
    """A name-only objective one node logged scopes to that node: the
    harvest is what names it."""
    sweep = await _sweep(sweeps, {"metric": "mse", "direction": MIN})
    await _child(runs, sweeps, sweep.id, 0, _two_branches(0.003, 0.05))

    variant = next(v for v in (await _get(http, sweep.id))["variants"]
                   if v["index"] == 0)
    assert variant["objective_node_id"] == "mse-plain"
    assert variant["final_metrics_node_id"] == "mse-plain"
    assert variant["final_metrics"] == {"mse": 0.04}


async def test_an_ambiguous_objective_keeps_the_run_wide_summary(
        runs, sweeps, http):
    """Several producers and no node named: the variant is unranked and its
    summary stays run-wide, labelled as such by a null node."""
    sweep = await _sweep(sweeps, {"metric": "train_loss", "direction": MIN})
    await _child(runs, sweeps, sweep.id, 0, _two_branches(0.003, 0.05))

    variant = next(v for v in (await _get(http, sweep.id))["variants"]
                   if v["index"] == 0)
    assert variant["ambiguous_producers"] == ["train-decay", "train-plain"]
    assert variant["final_metrics_node_id"] is None
    assert variant["final_metrics"]["lr"] == 0.01
    assert variant["final_metrics"]["mse"] == 0.04


async def test_an_unstarted_variant_has_no_summary_and_no_scope(
        runs, sweeps, http):
    sweep = await _sweep(sweeps, {"metric": "train_loss", "direction": MIN,
                                  "node_id": "train-plain"})
    variant = next(v for v in (await _get(http, sweep.id))["variants"]
                   if v["index"] == 1)
    assert variant["run_id"] is None
    assert "final_metrics" not in variant
    assert "final_metrics_node_id" not in variant


async def test_producer_metric_values_skips_other_nodes_and_non_finite(
        db, runs):
    record = await runs.create_run(
        graph_snapshot={"nodes": [], "edges": []}, options={},
        provenance=RunProvenance())
    await runs.log_metrics(record.id, [
        MetricPoint("lr", 0.003, 0, "train-plain"),
        MetricPoint("train_loss", 0.5, 0, "train-plain"),
        MetricPoint("train_loss", float("nan"), 1, "train-plain"),
        MetricPoint("lr", 0.01, 5, "train-decay"),
        MetricPoint("acc", 0.9, 0, None),
    ])
    assert await db.run(lambda conn: producer_metric_values(
        conn, record.id, "train-plain")) == {"lr": 0.003}
    assert await db.run(lambda conn: producer_metric_values(
        conn, record.id, "nobody")) == {}
    # Scoped and unscoped runs in ONE call.
    other = await runs.create_run(
        graph_snapshot={"nodes": [], "edges": []}, options={},
        provenance=RunProvenance())
    await runs.log_metrics(other.id, [MetricPoint("lr", 0.2, 0, "x")])
    assert await runs.latest_metrics(
        [record.id, other.id], producers={record.id: "train-decay"}) == {
        record.id: {"lr": 0.01}, other.id: {"lr": 0.2}}
