"""A sweep over a classical or an RL graph ranks its variants (#711).

Before #711 only TrainingLoop and a few evaluation nodes called
``context.log_metric``. The issue's repro, a sweep of
``examples/Classical/Iris-Sklearn-KNN`` over ``knn.n_neighbors`` with
objective ``accuracy``, finished with ``final_metrics == {}`` for every
variant and ``objective_warning: no variant recorded a metric named
'accuracy'``. These tests run that sweep, and one over the shipped
Reward-Hacking example, through the real ``POST /api/sweeps`` route and the
real queue, and check the ranked table against an independent reference.

Fixtures are local, following the house rule that test modules stay
independent (test_api_sweeps.py builds the same service the same way).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core.auth import TOKEN_HEADER, session_token
from app.core.db import Database
from app.core.run_service import QueueLimits, RunService
from app.core.run_store import RunStore
from app.core.sweep_store import SweepStore
from app.main import app
from app.nodes.rl.bradley_terry_node import BradleyTerryTrainNode
from app.nodes.rl.preference_dataset_node import PreferenceDatasetNode
from tests._example_graphs import graph_nodes

BASE_URL = f"http://127.0.0.1:{settings.PORT}"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = _REPO_ROOT / "examples"
_IRIS_CSV = _REPO_ROOT / "backend" / "data" / "samples" / "iris.csv"

#: Picked so the four accuracies differ (0.967, 0.933, 0.9, 0.667 on the
#: shipped split), which makes the expected order unique.
_KNN_DOMAIN = [90, 5, 60, 25]
#: Holdout accuracy falls as the planted shortcut grows (the example's
#: whole point), so the expected order is 0, 1, 3.
_SHORTCUT_DOMAIN = [3.0, 0.0, 1.0]


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
async def client(db):
    store = RunStore(db)
    service = RunService(
        store, shutdown_grace_s=5.0,
        limits=QueueLimits(cpu=2, gpu=1, interactive=2, overrides=()))
    app.state.db = db
    app.state.run_service = service
    app.state.sweep_store = SweepStore(db)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url=BASE_URL,
            headers={TOKEN_HEADER: session_token()},
        ) as http:
            yield http, store
    finally:
        await service.shutdown()
        for attribute in ("db", "run_service", "sweep_store"):
            if hasattr(app.state, attribute):
                delattr(app.state, attribute)


def _example(relative: str) -> dict[str, Any]:
    graph = json.loads((_EXAMPLES / relative / "graph.json").read_text())
    return {"nodes": graph_nodes(graph), "edges": graph["edges"]}


def _node(graph: dict[str, Any], node_id: str) -> dict[str, Any]:
    return next(n for n in graph["nodes"] if n["id"] == node_id)


def _iris_graph() -> dict[str, Any]:
    """The issue's example, read from the absolute CSV path and with uniform
    KNN votes, so the reference below is a plain sklearn KNN."""
    graph = _example("Classical/Iris-Sklearn-KNN")
    _node(graph, "csv")["data"]["params"]["path"] = str(_IRIS_CSV)
    _node(graph, "knn")["data"]["params"]["weights"] = "uniform"
    return graph


def _iris_reference(n_neighbors: int) -> float:
    """The example's pipeline in pandas + sklearn, outside CodefyUI."""
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier

    df = pd.read_csv(_IRIS_CSV)
    x = df.iloc[:, [2, 3]].to_numpy(dtype="float32")
    y = [str(v) for v in df["species"]]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y)
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="uniform")
    knn.fit((x_train - mean) / std, y_train)
    return float(np.mean(knn.predict((x_test - mean) / std) == np.array(y_test)))


async def _run_sweep(http: AsyncClient, store: RunStore,
                     body: dict[str, Any]) -> dict[str, Any]:
    response = await http.post("/api/sweeps", json=body)
    assert response.status_code == 201, response.text
    sweep_id = response.json()["sweep_id"]
    deadline = time.monotonic() + 60.0
    for child in await store.list_runs_by_sweep(sweep_id):
        while (await store.get_run(child.id)).finished_at is None:
            assert time.monotonic() < deadline, "a sweep variant never finished"
            await asyncio.sleep(0.05)
    result = await http.get(f"/api/sweeps/{sweep_id}")
    assert result.status_code == 200, result.text
    return result.json()


def _knn_value(variant: dict[str, Any]) -> int:
    return next(p["value"] for p in variant["params"] if p["param"] == "n_neighbors")


async def test_the_iris_knn_sweep_ranks_its_variants_by_accuracy(client):
    """#711's repro and its first acceptance criterion."""
    http, store = client
    body = await _run_sweep(http, store, {
        "base_graph": _iris_graph(),
        "sweep_spec": {"method": "grid", "params": [
            {"node_id": "knn", "param": "n_neighbors", "values": _KNN_DOMAIN}]},
        "objective": {"metric": "accuracy", "direction": "maximize"},
        "options": {"device": "cpu"},
        "name": "iris knn",
    })

    assert body["counts"]["succeeded"] == len(_KNN_DOMAIN)
    assert "objective_warning" not in body
    variants = body["variants"]
    for variant in variants:
        assert "accuracy" in variant["final_metrics"], variant
        assert variant["objective_node_id"] == "acc"
        assert variant["objective"] == pytest.approx(
            _iris_reference(_knn_value(variant)))

    expected = sorted(_KNN_DOMAIN, key=_iris_reference, reverse=True)
    assert [_knn_value(v) for v in variants] == expected
    assert [v["rank"] for v in variants] == [1, 2, 3, 4]
    assert body["best"]["index"] == variants[0]["index"]


async def test_two_accuracy_nodes_are_ambiguous_until_the_objective_names_one(
        client):
    """#711 acceptance 3: a second Accuracy in the graph triggers #641's
    ambiguity message, and ``objective.node_id`` resolves it."""
    http, store = client
    graph = _iris_graph()
    graph["nodes"].append({"id": "acc-2", "type": "Accuracy",
                           "data": {"params": {}}})
    # Train-on-train accuracy: a second, different number of the same name.
    graph["edges"] += [
        {"id": "a2-p", "source": "split", "sourceHandle": "y_train",
         "target": "acc-2", "targetHandle": "predictions"},
        {"id": "a2-l", "source": "split", "sourceHandle": "y_train",
         "target": "acc-2", "targetHandle": "labels"},
    ]
    spec = {"method": "grid", "params": [
        {"node_id": "knn", "param": "n_neighbors", "values": [90, 5]}]}

    ambiguous = await _run_sweep(http, store, {
        "base_graph": graph, "sweep_spec": spec,
        "objective": {"metric": "accuracy", "direction": "maximize"},
        "options": {"device": "cpu"}, "name": "two accuracies"})
    assert all(v["rank"] is None for v in ambiguous["variants"])
    assert "more than one node" in ambiguous["objective_warning"]
    assert "objective.node_id" in ambiguous["objective_warning"]

    named = await _run_sweep(http, store, {
        "base_graph": graph, "sweep_spec": spec,
        "objective": {"metric": "accuracy", "direction": "maximize",
                      "node_id": "acc"},
        "options": {"device": "cpu"}, "name": "named accuracy"})
    assert "objective_warning" not in named
    assert [_knn_value(v) for v in named["variants"]] == [5, 90]
    assert named["variants"][0]["objective"] == pytest.approx(_iris_reference(5))


def _holdout_reference(shortcut: float) -> float:
    """BradleyTerryTrain's own last holdout accuracy, run outside the
    engine with the example's params."""
    graph = _example("RL/Reward-Hacking-Shortcut")
    pair_params = dict(_node(graph, "pairs-hack")["data"]["params"],
                       shortcut_strength=shortcut)
    pairs = PreferenceDatasetNode().execute({}, pair_params)
    out = BradleyTerryTrainNode().execute(
        pairs, _node(graph, "rm-hack")["data"]["params"])
    return float(out["holdout_accuracy"][-1])


async def test_the_reward_hacking_example_can_be_swept_on_holdout_accuracy(
        client):
    """#711 acceptance 2: an RL example swept on a recorded metric. The
    example has two BradleyTerryTrain nodes, so the objective names one."""
    http, store = client
    body = await _run_sweep(http, store, {
        "base_graph": _example("RL/Reward-Hacking-Shortcut"),
        "sweep_spec": {"method": "grid", "params": [
            {"node_id": "pairs-hack", "param": "shortcut_strength",
             "values": _SHORTCUT_DOMAIN}]},
        "objective": {"metric": "holdout_accuracy", "direction": "maximize",
                      "node_id": "rm-hack"},
        "options": {"device": "cpu"},
        "name": "shortcut strength",
    })

    assert body["counts"]["succeeded"] == len(_SHORTCUT_DOMAIN)
    assert "objective_warning" not in body
    variants = body["variants"]
    shortcuts = [v["params"][0]["value"] for v in variants]
    assert shortcuts == [0.0, 1.0, 3.0]
    assert [v["rank"] for v in variants] == [1, 2, 3]
    for variant, shortcut in zip(variants, shortcuts):
        assert variant["objective"] == pytest.approx(
            _holdout_reference(shortcut))
        assert {"train_loss", "train_accuracy", "holdout_accuracy"} <= set(
            variant["final_metrics"])
