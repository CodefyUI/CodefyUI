"""A sweep objective is a node AND a metric (#641).

Before #641 both harvest seams read the objective off the Runs summary,
which collapses a series name several nodes log into one number: the
highest step across all producers, then the last write. With

    | producer  | name     | step | value |
    | trainer_a | val_loss | 100  | 0.8   |
    | trainer_b | val_loss | 1    | 0.1   |

a sweep ranking by "val_loss" read 0.8 whatever the user meant, and with
equal steps the insertion order chose. These tests pin the replacement
rule (``sweep_store.read_objective``) at the store, at both seams, and at
the route's JSON and CSV.

Fixtures and helpers are local, following the house rule that test
modules stay independent.
"""

from __future__ import annotations

import csv
import io
import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from app.api import routes_sweeps
from app.api.routes_runs import _CSV_BOM
from app.core.db import Database
from app.core.run_store import (
    MetricPoint,
    RunProvenance,
    RunStore,
    metric_producers,
    producer_last_value,
)
from app.core.sweep_store import SweepStore, SweepVariant, read_objective

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
    """The sweeps router alone, over the two stores: GET needs nothing
    else, and seam A reads only ``service.store``."""
    api = FastAPI()
    api.include_router(routes_sweeps.router)
    api.state.run_service = SimpleNamespace(store=runs)
    api.state.sweep_store = sweeps
    async with AsyncClient(transport=ASGITransport(app=api),
                           base_url="http://test") as client:
        yield client


def _variant(index: int) -> SweepVariant:
    return SweepVariant(
        index=index, domain_index=index, run_id=None,
        params=[{"node_id": "trainer_b", "param": "lr", "value": index}],
        seed=None, objective=None, status=None, harvested_at=None)


async def _sweep(sweeps: SweepStore, objective: dict[str, Any],
                 count: int = 1):
    return await sweeps.create_sweep(
        method="grid", seed=None, seed_variants=False,
        spec={"method": "grid", "params": [
            {"node_id": "trainer_b", "param": "lr", "domain": [0, 1]}]},
        objective=objective, variants=[_variant(i) for i in range(count)],
        name="producers")


async def _child(runs: RunStore, sweeps: SweepStore, sweep_id: str,
                 index: int, points: list[MetricPoint],
                 status: str = "succeeded") -> str:
    """One finished child, its points flushed ONE AT A TIME so write order
    is exactly the order given (the same-step tie-break depends on it)."""
    record = await runs.create_run(
        graph_snapshot={"nodes": [], "edges": []}, options={},
        provenance=RunProvenance(), sweep_id=sweep_id, sweep_variant=index)
    for point in points:
        await runs.log_metrics(record.id, [point])
    await runs.mark_finished(record.id, status)
    await sweeps.set_variant_run(sweep_id, index, run_id=record.id, seed=None)
    return record.id


def _issue_fixture(*, a_step: int = 100, b_first: bool = False
                   ) -> list[MetricPoint]:
    """The table in #641, in either insertion order."""
    a = MetricPoint("val_loss", 0.8, a_step, "trainer_a")
    b = MetricPoint("val_loss", 0.1, 1, "trainer_b")
    return [b, a] if b_first else [a, b]


async def _seam_a(http: AsyncClient, sweep_id: str) -> dict[str, Any]:
    response = await http.get(f"/api/sweeps/{sweep_id}")
    assert response.status_code == 200, response.text
    return response.json()


async def _seam_b(runs: RunStore, sweeps: SweepStore, sweep_id: str):
    """Retention with no prior read: the children are deleted here."""
    await runs.prune(keep_last=0)
    assert await runs.list_runs_by_sweep(sweep_id) == []
    return (await sweeps.get_sweep(sweep_id)).variants


def _read(db: Database, run_id: str, objective: dict[str, Any]):
    return db.run(lambda conn: read_objective(conn, run_id, objective))


# ── the rule itself ───────────────────────────────────────────────────────


async def test_the_issue_fixture_reads_trainer_b_whatever_trainer_a_did(
        db, runs, sweeps):
    """Acceptance 1: trainer_b.val_loss is 0.1, independent of trainer_a's
    step count -- the Runs summary still says 0.8, which is the bug."""
    sweep = await _sweep(sweeps, {"metric": "val_loss", "direction": MIN,
                                  "node_id": "trainer_b"})
    for a_step in (1, 100, 10_000):
        run_id = await _child(runs, sweeps, sweep.id, 0,
                              _issue_fixture(a_step=a_step))
        assert await _read(db, run_id, sweep.objective) == (
            0.1, "trainer_b", None)
        assert await _read(db, run_id, {
            "metric": "val_loss", "direction": MIN,
            "node_id": "trainer_a"}) == (0.8, "trainer_a", None)
    # The compact Runs summary is deliberately unchanged.
    assert (await runs.latest_metrics([run_id]))[run_id] == {"val_loss": 0.8}


@pytest.mark.parametrize("b_first", [False, True])
async def test_same_step_insertion_order_cannot_change_the_producer(
        db, runs, sweeps, b_first):
    """Acceptance 2: with both at step 1, the write order picked the value
    before #641. Now a named producer reads its own series either way, and
    a name-only objective is ambiguous either way, with the same list."""
    sweep = await _sweep(sweeps, {"metric": "val_loss", "direction": MIN})
    run_id = await _child(runs, sweeps, sweep.id, 0,
                          _issue_fixture(a_step=1, b_first=b_first))
    for node, value in (("trainer_a", 0.8), ("trainer_b", 0.1)):
        assert await _read(db, run_id, {"metric": "val_loss",
                                        "direction": MIN,
                                        "node_id": node}) == (value, node,
                                                              None)
    assert await _read(db, run_id, sweep.objective) == (
        None, None, ["trainer_a", "trainer_b"])


async def test_a_single_producer_name_only_objective_still_ranks(
        db, runs, sweeps, http):
    """Acceptance 4, and the legacy-sweep decision: a stored objective with
    no node_id key is name-only, and with one producer it reads that
    producer's last point and records which node that was."""
    legacy = await _sweep(sweeps, {"metric": "val_loss", "direction": MIN},
                          count=2)
    assert "node_id" not in legacy.objective
    await _child(runs, sweeps, legacy.id, 0, [
        MetricPoint("val_loss", 0.5, 0, "trainer_b"),
        MetricPoint("val_loss", 0.3, 1, "trainer_b")])
    await _child(runs, sweeps, legacy.id, 1, [
        MetricPoint("val_loss", 0.2, 0, "trainer_b")])

    body = await _seam_a(http, legacy.id)
    assert body["objective"] == {"metric": "val_loss", "direction": MIN,
                                 "node_id": None}
    assert [(v["index"], v["rank"], v["objective"], v["objective_node_id"],
             v["ambiguous_producers"]) for v in body["variants"]] == [
        (1, 1, 0.2, "trainer_b", None), (0, 2, 0.3, "trainer_b", None)]
    assert "objective_warning" not in body


async def test_an_inner_node_id_is_an_ordinary_producer(db, runs, sweeps):
    """Acceptance 7a: a node inside a block logs under its FLATTENED id
    (``graph_engine.SUBGRAPH_SEPARATOR``), and the objective names it."""
    sweep = await _sweep(sweeps, {"metric": "val_loss", "direction": MIN,
                                  "node_id": "block1/trainer"})
    run_id = await _child(runs, sweeps, sweep.id, 0, [
        MetricPoint("val_loss", 0.9, 5, "block1"),
        MetricPoint("val_loss", 0.4, 2, "block1/trainer"),
        MetricPoint("val_loss", 0.7, 9, "block2/trainer")])
    assert await _read(db, run_id, sweep.objective) == (
        0.4, "block1/trainer", None)
    assert await db.run(lambda conn: metric_producers(
        conn, run_id, "val_loss")) == ["block1", "block1/trainer",
                                       "block2/trainer"]


async def test_a_non_finite_last_value_is_unranked_not_an_earlier_point(
        db, runs, sweeps):
    """Acceptance 7b: the chosen producer's LAST point diverged. No value --
    neither its earlier finite point nor the other producer's number."""
    sweep = await _sweep(sweeps, {"metric": "val_loss", "direction": MIN,
                                  "node_id": "trainer_b"})
    run_id = await _child(runs, sweeps, sweep.id, 0, [
        MetricPoint("val_loss", 0.2, 0, "trainer_b"),
        MetricPoint("val_loss", float("nan"), 1, "trainer_b"),
        MetricPoint("val_loss", 0.8, 100, "trainer_a")])
    assert await _read(db, run_id, sweep.objective) == (
        None, "trainer_b", None)
    variants = await _seam_b(runs, sweeps, sweep.id)
    assert variants[0].objective is None
    assert variants[0].objective_node_id == "trainer_b"


async def test_the_run_level_series_counts_as_a_producer(db, runs, sweeps):
    """A point logged with no node is one more producer of the name, so it
    cannot hide an ambiguity, and it reads alone when it is the only one."""
    sweep = await _sweep(sweeps, {"metric": "loss", "direction": MIN})
    both = await _child(runs, sweeps, sweep.id, 0, [
        MetricPoint("loss", 1.0, 0), MetricPoint("loss", 2.0, 0, "n")])
    assert await _read(db, both, sweep.objective) == (None, None, [None, "n"])
    alone = await _child(runs, sweeps, sweep.id, 0, [
        MetricPoint("loss", 1.0, 0), MetricPoint("loss", 3.0, 4)])
    assert await _read(db, alone, sweep.objective) == (3.0, None, None)
    assert await db.run(lambda conn: producer_last_value(
        conn, alone, "loss", None)) == 3.0
    assert await db.run(lambda conn: producer_last_value(
        conn, alone, "loss", "absent")) is None


# ── both seams, JSON, CSV ─────────────────────────────────────────────────


async def test_ambiguous_name_only_is_explained_and_unranked_in_json_and_csv(
        db, runs, sweeps, http):
    """Acceptance 3 and 6: a name-only objective two nodes log is unranked,
    the producers are named per variant, the sweep says why -- even though
    another variant ranked -- and the CSV carries the same identity."""
    sweep = await _sweep(sweeps, {"metric": "val_loss", "direction": MIN},
                         count=2)
    await _child(runs, sweeps, sweep.id, 0, _issue_fixture())
    await _child(runs, sweeps, sweep.id, 1, [
        MetricPoint("val_loss", 0.3, 1, "trainer_b")])

    body = await _seam_a(http, sweep.id)
    by_index = {v["index"]: v for v in body["variants"]}
    assert by_index[0]["rank"] is None and by_index[0]["objective"] is None
    assert by_index[0]["ambiguous_producers"] == ["trainer_a", "trainer_b"]
    assert by_index[0]["objective_node_id"] is None
    assert (by_index[1]["rank"], by_index[1]["objective"],
            by_index[1]["objective_node_id"]) == (1, 0.3, "trainer_b")
    assert "trainer_a, trainer_b" in body["objective_warning"]
    assert "objective.node_id" in body["objective_warning"]

    text = (await http.get(f"/api/sweeps/{sweep.id}?format=csv")).text
    rows = list(csv.DictReader(io.StringIO(text.lstrip(_CSV_BOM))))
    assert [(r["variant_index"], r["rank"], r["objective"],
             r["objective_metric"], r["objective_node_id"],
             r["ambiguous_producers"]) for r in rows] == [
        ("1", "1", "0.3", "val_loss", "trainer_b", ""),
        ("0", "", "", "val_loss", "", "trainer_a;trainer_b")]


@pytest.mark.parametrize("seam", ["read", "retention"])
async def test_both_seams_harvest_the_same_identity_and_value(
        db, runs, sweeps, http, seam):
    """Acceptance 5 and 6: whichever seam harvests -- a GET, or retention
    deleting the children unread -- the row keeps the same value and the
    same producer, and the JSON and CSV read them back after the children
    are gone."""
    sweep = await _sweep(sweeps, {"metric": "val_loss", "direction": MIN,
                                  "node_id": "trainer_b"}, count=2)
    await _child(runs, sweeps, sweep.id, 0, _issue_fixture())
    await _child(runs, sweeps, sweep.id, 1, _issue_fixture(b_first=True))
    if seam == "read":
        await _seam_a(http, sweep.id)
    await _seam_b(runs, sweeps, sweep.id)

    stored = (await sweeps.get_sweep(sweep.id)).variants
    assert [(v.objective, v.objective_node_id) for v in stored] == [
        (0.1, "trainer_b"), (0.1, "trainer_b")]

    body = await _seam_a(http, sweep.id)
    assert body["objective"]["node_id"] == "trainer_b"
    assert [(v["run_exists"], v["objective"], v["objective_node_id"])
            for v in body["variants"]] == [(False, 0.1, "trainer_b")] * 2
    assert body["best"]["objective"] == 0.1
    text = (await http.get(f"/api/sweeps/{sweep.id}?format=csv")).text
    rows = list(csv.DictReader(io.StringIO(text.lstrip(_CSV_BOM))))
    assert {(r["objective"], r["objective_node_id"]) for r in rows} == {
        ("0.1", "trainer_b")}


async def test_an_ambiguity_survives_retention(db, runs, sweeps, http):
    sweep = await _sweep(sweeps, {"metric": "val_loss", "direction": MIN})
    await _child(runs, sweeps, sweep.id, 0, _issue_fixture())
    variants = await _seam_b(runs, sweeps, sweep.id)
    assert variants[0].ambiguous_producers == ["trainer_a", "trainer_b"]
    body = await _seam_a(http, sweep.id)
    assert body["variants"][0]["ambiguous_producers"] == [
        "trainer_a", "trainer_b"]
    assert "objective_warning" in body


async def test_a_named_producer_that_logged_nothing_is_warned_by_node(
        db, runs, sweeps, http):
    sweep = await _sweep(sweeps, {"metric": "val_loss", "direction": MIN,
                                  "node_id": "trainer_c"})
    await _child(runs, sweeps, sweep.id, 0, _issue_fixture())
    body = await _seam_a(http, sweep.id)
    assert body["variants"][0]["rank"] is None
    assert body["objective_warning"].startswith(
        "no variant recorded a metric named 'val_loss' from node "
        "'trainer_c'")


# ── the request model and the evidence ────────────────────────────────────


def test_the_objective_model_accepts_a_node_and_refuses_a_blank_one():
    model = routes_sweeps.SweepObjectiveModel
    assert model(metric=" val_loss ", direction=MIN,
                 node_id=" block/inner ").model_dump() == {
        "metric": "val_loss", "direction": MIN, "node_id": "block/inner"}
    assert model(metric="val_loss", direction=MIN).node_id is None
    for bad in ("", "   "):
        with pytest.raises(ValidationError):
            model(metric="val_loss", direction=MIN, node_id=bad)


async def test_producers_by_run_is_the_evidence_and_is_index_served(
        db, runs, sweeps):
    sweep = await _sweep(sweeps, {"metric": "val_loss", "direction": MIN})
    run_id = await _child(runs, sweeps, sweep.id, 0, [
        *_issue_fixture(), MetricPoint("lr", 0.1, 0)])
    assert await runs.metric_producers_by_run([run_id, "ghost"]) == {
        run_id: {"lr": [None], "val_loss": ["trainer_a", "trainer_b"]}}
    assert await runs.metric_producers_by_run([]) == {}

    def _plans(conn: sqlite3.Connection) -> list[str]:
        statements = [
            ("SELECT value FROM exec_run_metrics WHERE run_id = ? AND name = ? "
             "AND node_id = ? ORDER BY step DESC, id DESC LIMIT 1",
             (run_id, "val_loss", "trainer_b")),
            ("SELECT MIN(node_id) FROM exec_run_metrics WHERE run_id = ? "
             "AND name = ? AND node_id > ?", (run_id, "val_loss", "")),
        ]
        return [" ".join(row["detail"] for row in conn.execute(
            "EXPLAIN QUERY PLAN " + sql, params)) for sql, params in statements]

    for plan in await db.run(_plans):
        assert "idx_exec_run_metrics_producer" in plan, plan
        assert "SCAN exec_run_metrics" not in plan, plan
        assert "TEMP B-TREE" not in plan, plan
