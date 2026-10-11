"""Headless runs take the graph's ``settings.seed`` (#704).

The canvas and Export Python read ``settings.seed``; ``run_graph.py``,
``POST /api/runs``, ``POST /api/graph/run/{name}`` and app invoke read only
``settings.device``, so a graph saved with a seed gave different numbers on
every headless run. The seed now follows the device rule: an explicit seed
wins, an omitted one is the graph's, and an explicit null (``--no-seed``,
``"seed": null``) is an unseeded run.

The numeric tests draw from a node that calls ``random.random()``, which the
engine seeds per node from ``(run seed, node id)`` on a seeded run only.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

import pytest

import run_graph
from app.core.db import Database
from app.core.device_utils import graph_settings_seed
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.node_registry import registry
from app.core.run_service import RunService, RunSubmitError, normalize_graph
from app.core.run_store import RunStore
from app.core.seeding import MAX_SEED


class _SeedDrawNode(BaseNode):
    """Passes nothing through: returns one draw from Python's ``random``."""

    NODE_NAME = "_SeedDraw"
    CATEGORY = "Test"
    DESCRIPTION = "One draw from random.random()"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY,
                               optional=True)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs: dict[str, Any],
                params: dict[str, Any]) -> dict[str, Any]:
        return {"value": random.random()}


@pytest.fixture(autouse=True)
def _register_draw_node():
    registry._nodes["_SeedDraw"] = _SeedDrawNode
    yield
    registry._nodes.pop("_SeedDraw", None)


# ── graph_settings_seed ───────────────────────────────────────────────────


@pytest.mark.parametrize("graph, expected", [
    ({"settings": {"seed": 0}}, 0),
    ({"settings": {"seed": 7, "device": "cpu"}}, 7),
    ({"settings": {"seed": MAX_SEED}}, MAX_SEED),
    ({"settings": {"device": "cpu"}}, None),
    ({"settings": {"seed": None}}, None),
    ({}, None),
    (None, None),
])
def test_graph_settings_seed(graph, expected):
    assert graph_settings_seed(graph) == expected


@pytest.mark.parametrize("value", ["7", True, 2.0, -1, MAX_SEED + 1])
def test_graph_settings_seed_drops_an_invalid_value_with_a_warning(
        caplog, value):
    with caplog.at_level(logging.WARNING, logger="app.core.device_utils"):
        assert graph_settings_seed({"settings": {"seed": value}}) is None
    assert "settings.seed" in caplog.text


# ── run_graph.py ──────────────────────────────────────────────────────────


def _cli_graph(seed: int | None) -> dict:
    graph = {
        "name": "draw",
        "nodes": [
            {"id": "start", "type": "Start", "data": {"params": {}}},
            {"id": "draw", "type": "_SeedDraw", "data": {"params": {}}},
        ],
        "edges": [
            {"id": "e", "source": "start", "target": "draw",
             "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
        ],
    }
    if seed is not None:
        graph["settings"] = {"seed": seed}
    return graph


@pytest.fixture
def cli_draws(monkeypatch):
    """The value each ``run_graph.run`` call drew, from the real engine."""
    draws: list[float] = []
    real = run_graph.execute_graph

    async def recording(*args, **kwargs):
        outputs = await real(*args, **kwargs)
        draws.append(outputs["draw"]["value"])
        return outputs

    monkeypatch.setattr(run_graph, "execute_graph", recording)
    return draws


def _write(tmp_path, graph: dict, name: str = "g.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps(graph), encoding="utf-8")
    return str(path)


async def test_cli_reproduces_a_seeded_graph_without_a_flag(
        tmp_path, cli_draws):
    path = _write(tmp_path, _cli_graph(0))
    await run_graph.run(path)
    await run_graph.run(path)
    assert cli_draws[0] == cli_draws[1]
    # The graph's seed IS the run seed: the same numbers as --seed 0.
    await run_graph.run(path, seed=0)
    assert cli_draws[2] == cli_draws[0]


async def test_cli_seed_flag_overrides_the_graph_seed(tmp_path, cli_draws):
    await run_graph.run(_write(tmp_path, _cli_graph(0)), seed=1)
    await run_graph.run(_write(tmp_path, _cli_graph(1), "one.json"))
    await run_graph.run(_write(tmp_path, _cli_graph(0), "zero.json"))
    assert cli_draws[0] == cli_draws[1]
    assert cli_draws[0] != cli_draws[2]


async def test_cli_no_seed_runs_a_seeded_graph_unseeded(
        tmp_path, cli_draws, caplog):
    path = _write(tmp_path, _cli_graph(0))
    with caplog.at_level(logging.INFO, logger="codefyui.cli"):
        await run_graph.run(path)
    assert "Seed: 0 (graph)" in caplog.text
    await run_graph.run(path, no_seed=True)
    await run_graph.run(path, no_seed=True)
    assert cli_draws[1] != cli_draws[0]
    assert cli_draws[2] != cli_draws[1]


def test_cli_seed_and_no_seed_are_exclusive(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv",
                        ["run_graph.py", "g.json", "--seed", "1", "--no-seed"])
    with pytest.raises(SystemExit) as excinfo:
        run_graph.main()
    assert excinfo.value.code == 2


# ── POST /api/runs (RunService.submit) ────────────────────────────────────


def _run_graph(seed: int | None) -> dict:
    graph = {
        "nodes": [
            {"id": "start", "type": "Start", "data": {"params": {}}},
            {"id": "draw", "type": "_SeedDraw", "data": {"params": {}}},
        ],
        "edges": [
            {"id": "e", "source": "start", "target": "draw",
             "sourceHandle": "trigger", "type": "trigger"},
        ],
    }
    if seed is not None:
        graph["settings"] = {"seed": seed}
    return graph


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
def store(db):
    return RunStore(db)


@pytest.fixture
async def service(store):
    svc = RunService(store, shutdown_grace_s=2.0)
    try:
        yield svc
    finally:
        await svc.shutdown()


async def _finished(store: RunStore, run_id: str):
    import asyncio
    import time

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        record = await store.get_run(run_id)
        if record.finished_at is not None:
            return record
        await asyncio.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish")


async def test_submit_without_a_seed_records_the_graph_seed(store, service):
    submitted = await service.submit(_run_graph(5))
    record = await _finished(store, submitted.run_id)
    assert record.status == "succeeded"
    assert record.options["seed"] == 5
    snapshot = await store.get_graph_snapshot(submitted.run_id)
    assert snapshot["settings"] == {"seed": 5}


async def test_submit_takes_graph_seed_zero(store, service):
    submitted = await service.submit(_run_graph(0), options={"lane": "queued"})
    record = await _finished(store, submitted.run_id)
    assert record.options["seed"] == 0


@pytest.mark.parametrize("options, expected", [
    ({"seed": 3}, 3),
    # The explicit way to ask for an unseeded run of a seeded graph.
    ({"seed": None}, None),
])
async def test_submit_option_seed_beats_the_graph_seed(
        store, service, options, expected):
    submitted = await service.submit(_run_graph(5), options=options)
    record = await _finished(store, submitted.run_id)
    assert record.options["seed"] == expected


async def test_submit_without_any_seed_stays_unseeded(store, service):
    submitted = await service.submit(_run_graph(None))
    record = await _finished(store, submitted.run_id)
    assert record.options["seed"] is None


@pytest.mark.parametrize("seed", ["7", True, 2.0, -1, MAX_SEED + 1])
def test_normalize_graph_refuses_an_invalid_settings_seed(seed):
    with pytest.raises(RunSubmitError, match="settings.seed"):
        normalize_graph({**_run_graph(None), "settings": {"seed": seed}})


# ── POST /api/graph/run/{name} ────────────────────────────────────────────


def _contract_graph(name: str, seed: int | None) -> dict:
    """Start -> GraphInput -> _SeedDraw -> GraphOutput."""
    graph = {
        "name": name,
        "description": "",
        "nodes": [
            {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
             "data": {"params": {}}},
            {"id": "gi", "type": "GraphInput", "position": {"x": 200, "y": 0},
             "data": {"params": {"name": "x", "type": "string",
                                 "required": False, "default": "",
                                 "description": ""}}},
            {"id": "draw", "type": "_SeedDraw", "position": {"x": 300, "y": 0},
             "data": {"params": {}}},
            {"id": "out", "type": "GraphOutput", "position": {"x": 400, "y": 0},
             "data": {"params": {"name": "y", "description": ""}}},
        ],
        "edges": [
            {"id": "t1", "source": "start", "target": "gi",
             "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
            {"id": "d1", "source": "gi", "target": "draw",
             "sourceHandle": "value", "targetHandle": "value", "type": "data"},
            {"id": "d2", "source": "draw", "target": "out",
             "sourceHandle": "value", "targetHandle": "value", "type": "data"},
        ],
    }
    if seed is not None:
        graph["settings"] = {"seed": seed}
    return graph


@pytest.fixture
def graphs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    return tmp_path


async def _function_draw(client, name: str, body: dict | None = None) -> Any:
    resp = await client.post(f"/api/graph/run/{name}", json=body or {})
    assert resp.status_code == 200, resp.text
    return resp.json()["outputs"]["y"]


async def test_graph_as_a_function_takes_the_graph_seed(
        test_client, graphs_dir):
    resp = await test_client.post("/api/graph/save",
                                  json=_contract_graph("seeded", 4))
    assert resp.status_code == 200, resp.text
    first = await _function_draw(test_client, "seeded")
    assert await _function_draw(test_client, "seeded") == first
    # An explicit seed wins, and seed 4 named in the body is the same run.
    assert await _function_draw(test_client, "seeded", {"seed": 4}) == first
    assert await _function_draw(test_client, "seeded", {"seed": 9}) != first
    # "seed": null is an unseeded run of the seeded graph.
    unseeded = [await _function_draw(test_client, "seeded", {"seed": None})
                for _ in range(2)]
    assert unseeded[0] != unseeded[1]


@pytest.mark.parametrize("seed", ["4", True, 1.5, -1, MAX_SEED + 1])
async def test_graph_as_a_function_refuses_a_bad_seed(
        test_client, graphs_dir, seed):
    resp = await test_client.post("/api/graph/save",
                                  json=_contract_graph("seeded", None))
    assert resp.status_code == 200, resp.text
    resp = await test_client.post("/api/graph/run/seeded", json={"seed": seed})
    assert resp.status_code == 422
    assert resp.json()["error"]["details"][0]["field"] == "seed"


# ── POST /api/apps/{slug}/invoke ──────────────────────────────────────────


async def test_app_invoke_takes_the_snapshot_seed(
        test_client, app_db, graphs_dir):
    graph = _contract_graph("app-src", 4)
    resp = await test_client.post("/api/graph/save", json=graph)
    assert resp.status_code == 200, resp.text
    resp = await test_client.post("/api/apps/seeded-app/publish",
                                  json={"graph": "app-src", "create": True})
    assert resp.status_code == 200, resp.text
    key = (await test_client.post("/api/keys", json={"name": "t"})).json()
    headers = {"Authorization": f"Bearer {key['token']}"}

    async def invoke(body: dict) -> Any:
        resp = await test_client.post("/api/apps/seeded-app/invoke",
                                      json=body, headers=headers)
        assert resp.status_code == 200, resp.text
        return resp.json()["outputs"]["y"]

    first = await invoke({"inputs": {}})
    assert await invoke({"inputs": {}}) == first
    # The same value graph-as-a-function gives for the same graph and seed.
    assert await _function_draw(test_client, "app-src") == first
    assert await invoke({"inputs": {}, "seed": 9}) != first
    unseeded = [await invoke({"inputs": {}, "seed": None}) for _ in range(2)]
    assert unseeded[0] != unseeded[1]
