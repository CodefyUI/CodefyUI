"""Tests for /api/runs — the Run Service REST surface (#120).

The three acceptance criteria of the issue are stated here at the level a
user actually meets them (HTTP), on top of the service-level versions in
test_run_service.py:

1. Submit, kill the submitting client, the run still completes and stays
   fully queryable.
2. Two concurrent submits do not cancel each other.
3. A row left ``running`` by a dead process reads ``interrupted``.

Plus the parts of the contract only the HTTP layer has: cursor pagination,
long-poll semantics, CSV metrics export, auth, and the 503 when the service
is not wired up.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import time
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core.auth import TOKEN_HEADER, session_token
from app.core.db import Database
from app.core.node_base import (
    MEDIA_IMAGE,
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from app.core.node_registry import registry
from app.core.run_service import QueueLimits, RunService
from app.core.run_store import RunProvenance, RunStore
from app.main import app

BASE_URL = f"http://127.0.0.1:{settings.PORT}"


# ── test nodes (duplicated so test modules stay independent) ─────────────


class _ApiSlowNode(BaseNode):
    NODE_NAME = "_ApiSlow"
    CATEGORY = "Test"
    DESCRIPTION = "Sleeps, then passes through"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [ParamDefinition(name="seconds", param_type=ParamType.FLOAT,
                                default=0.3)]

    def execute(self, inputs: dict[str, Any],
                params: dict[str, Any]) -> dict[str, Any]:
        time.sleep(float(params.get("seconds", 0.3)))
        return {"value": inputs.get("value")}


class _ApiMetricsNode(BaseNode):
    NODE_NAME = "_ApiMetrics"
    CATEGORY = "Test"
    DESCRIPTION = "Emits epoch progress events"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any],
                progress_callback=None) -> dict[str, Any]:
        for epoch in (1, 2):
            if progress_callback:
                progress_callback({"event": "epoch", "epoch": epoch,
                                   "loss": 1.0 / epoch, "note": "text,with,commas"})
        return {"value": inputs.get("value")}


class _ApiImageNode(BaseNode):
    NODE_NAME = "_ApiImage"
    CATEGORY = "Test"
    DESCRIPTION = "Emits a declared base64 image"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="value", data_type=DataType.ANY),
            PortDefinition(name="image", data_type=DataType.ANY,
                           media=MEDIA_IMAGE),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [ParamDefinition(name="size", param_type=ParamType.INT,
                                default=64)]

    def execute(self, inputs: dict[str, Any],
                params: dict[str, Any]) -> dict[str, Any]:
        return {"value": inputs.get("value"),
                "image": "A" * int(params.get("size", 64))}


_TEST_NODES = {
    "_ApiSlow": _ApiSlowNode,
    "_ApiMetrics": _ApiMetricsNode,
    "_ApiImage": _ApiImageNode,
}


@pytest.fixture(autouse=True)
def _register_test_nodes():
    registry._nodes.update(_TEST_NODES)
    yield
    for name in _TEST_NODES:
        registry._nodes.pop(name, None)


# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
async def run_app(tmp_path):
    """Per-test Database + RunService on app.state, torn down in the REAL order.

    Teardown mirrors the lifespan exactly — drain the service, THEN close the
    database — so every test in this module also exercises the shutdown
    ordering that keeps a run's last writes from racing ``db.close()``.
    """
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    store = RunStore(database)
    service = RunService(store, shutdown_grace_s=5.0)
    app.state.db = database
    app.state.run_service = service
    try:
        yield service
    finally:
        await service.shutdown()
        database.close()
        for attribute in ("db", "run_service"):
            if hasattr(app.state, attribute):
                delattr(app.state, attribute)


@pytest.fixture
async def client(run_app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url=BASE_URL,
        headers={TOKEN_HEADER: session_token()},
    ) as http:
        yield http


def _new_client() -> AsyncClient:
    """A SEPARATE connection to the same app — a different browser tab."""
    return AsyncClient(
        transport=ASGITransport(app=app), base_url=BASE_URL,
        headers={TOKEN_HEADER: session_token()},
    )


def _graph(middle_type: str | None = None,
           params: dict[str, Any] | None = None) -> dict[str, Any]:
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "src", "type": "_TestSource", "data": {"params": {"val": "hi"}}},
        {"id": "print", "type": "Print", "data": {"params": {"label": "out"}}},
    ]
    edges = [
        {"id": "et", "source": "start", "target": "src",
         "sourceHandle": "trigger", "type": "trigger"},
    ]
    if middle_type is None:
        edges.append({"id": "e1", "source": "src", "target": "print",
                      "sourceHandle": "value", "targetHandle": "value"})
    else:
        nodes.insert(2, {"id": "mid", "type": middle_type,
                         "data": {"params": params or {}}})
        edges.append({"id": "e1", "source": "src", "target": "mid",
                      "sourceHandle": "value", "targetHandle": "value"})
        edges.append({"id": "e2", "source": "mid", "target": "print",
                      "sourceHandle": "value", "targetHandle": "value"})
    return {"nodes": nodes, "edges": edges}


async def _poll_until_terminal(http: AsyncClient, run_id: str,
                               timeout: float = 15.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = await http.get(f"/api/runs/{run_id}")
        assert response.status_code == 200, response.text
        body = response.json()
        if body["finished_at"] is not None:
            return body
        await asyncio.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish within {timeout}s")


# ── acceptance criterion 1 ────────────────────────────────────────────────


async def test_run_completes_after_the_submitting_client_is_killed(run_app):
    """Submit, close the client mid-run, then query everything from a new one."""
    submitter = _new_client()
    response = await submitter.post("/api/runs", json={
        "graph": _graph("_ApiSlow", {"seconds": 0.4}),
        "options": {"record_outputs": True},
        "name": "detached",
    })
    assert response.status_code == 200, response.text
    body = response.json()
    run_id = body["run_id"]
    assert body["status"] == "running"

    # The client goes away while the run is still in flight.
    await submitter.aclose()
    assert run_app.is_active(run_id), "run was already over; test proves nothing"

    async with _new_client() as watcher:
        record = await _poll_until_terminal(watcher, run_id)
        assert record["status"] == "succeeded"
        assert record["error"] is None
        assert record["name"] == "detached"

        events = (await watcher.get(f"/api/runs/{run_id}/events")).json()
        types = [event["type"] for event in events["events"]]
        assert types[0] == "execution_start"
        assert types[-1] == "execution_complete"
        assert events["cursor"] == events["events"][-1]["cursor"]
        assert events["status"] == "succeeded"

        metrics = (await watcher.get(f"/api/runs/{run_id}/metrics")).json()
        assert metrics["run_id"] == run_id
        assert metrics["names"] == []


# ── acceptance criterion 2 ────────────────────────────────────────────────


async def test_two_concurrent_submits_do_not_cancel_each_other(client):
    first, second = await asyncio.gather(
        client.post("/api/runs",
                    json={"graph": _graph("_ApiSlow", {"seconds": 0.3})}),
        client.post("/api/runs",
                    json={"graph": _graph("_ApiSlow", {"seconds": 0.05})}),
    )
    ids = [first.json()["run_id"], second.json()["run_id"]]
    assert ids[0] != ids[1]

    records = await asyncio.gather(*(
        _poll_until_terminal(client, run_id) for run_id in ids))
    assert [record["status"] for record in records] == ["succeeded", "succeeded"]


# ── acceptance criterion 3 ────────────────────────────────────────────────


async def test_restart_shows_interrupted_not_running(client, run_app):
    """A row from a process that died mid-run must never read ``running``."""
    orphan = await run_app.store.create_run(
        graph_snapshot=_graph(), options={}, provenance=RunProvenance())
    await run_app.store.mark_running(orphan.id)
    assert (await client.get(f"/api/runs/{orphan.id}")).json()["status"] == "running"

    # The process restarts: a fresh service over the same database.
    reborn = RunService(run_app.store)
    assert await reborn.recover_interrupted() == 1

    body = (await client.get(f"/api/runs/{orphan.id}")).json()
    assert body["status"] == "interrupted"
    assert body["finished_at"] is not None
    assert body["active"] is False


# ── submit ────────────────────────────────────────────────────────────────


async def test_submit_records_normalized_options(client):
    response = await client.post("/api/runs", json={
        "graph": _graph(), "options": {"seed": 3, "lane": "gpu"}})
    run_id = response.json()["run_id"]
    record = await _poll_until_terminal(client, run_id)
    # The full effective configuration, not just what the client sent, so a
    # stored run stays readable without knowing which defaults were in force.
    assert record["options"] == {
        "device": "cpu", "seed": 3, "lane": "gpu", "graph_id": "",
        "error_mode": "fail_fast", "max_retries": 0, "record_outputs": False,
        "verbose": False, "weights_persistent": False, "backward_mode": False,
        "auto_backward": False,
    }
    assert record["queue_key"] == "cpu"


@pytest.mark.parametrize("payload,expected", [
    ({"graph": {"nodes": []}}, 400),
    ({"graph": {"edges": []}}, 400),
    ({"graph": _graph(), "options": {"devcie": "cuda"}}, 400),   # bad key
    ({"graph": _graph(), "options": {"device": "cudda"}}, 400),  # bad value
    ({"graph": _graph(), "options": {"seed": -5}}, 400),
    ({"graph": _graph(), "options": "nope"}, 400),
    ({"graph": _graph(), "name": "x" * 65}, 400),
    ({"options": {}}, 422),                       # graph is required
    ({"graph": "not-an-object"}, 422),
])
async def test_submit_rejects_bad_input(client, payload, expected):
    response = await client.post("/api/runs", json=payload)
    assert response.status_code == expected, response.text


async def test_submit_requires_the_session_token(run_app):
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url=BASE_URL) as anonymous:
        response = await anonymous.post("/api/runs", json={"graph": _graph()})
        assert response.status_code == 403
        cancel = await anonymous.post("/api/runs/whatever/cancel")
        assert cancel.status_code == 403


async def test_submit_during_shutdown_is_503(client, run_app):
    await run_app.shutdown()
    response = await client.post("/api/runs", json={"graph": _graph()})
    assert response.status_code == 503
    assert "shutting down" in response.json()["detail"]


async def test_endpoints_503_without_a_wired_service():
    # Restore whatever was on app.state: the module-level app is shared with
    # every other test module, so a test that strips it must put it back
    # even when the assertions fail.
    saved = {name: getattr(app.state, name)
             for name in ("db", "run_service") if hasattr(app.state, name)}
    for attribute in ("db", "run_service"):
        if hasattr(app.state, attribute):
            delattr(app.state, attribute)
    try:
        async with _new_client() as http:
            assert (await http.get("/api/runs")).status_code == 503
            assert (await http.get("/api/runs/abc/artifacts")).status_code == 503
            assert (await http.delete("/api/runs/abc")).status_code == 503
            assert (await http.post(
                "/api/runs", json={"graph": _graph()})).status_code == 503
    finally:
        for name, value in saved.items():
            setattr(app.state, name, value)


# ── list / get ────────────────────────────────────────────────────────────


async def test_list_is_newest_first_and_filterable(client):
    ids = []
    for _ in range(3):
        response = await client.post("/api/runs", json={"graph": _graph()})
        run_id = response.json()["run_id"]
        await _poll_until_terminal(client, run_id)
        ids.append(run_id)

    body = (await client.get("/api/runs")).json()
    assert body["total"] == 3
    assert body["limit"] == 50 and body["offset"] == 0
    assert [run["id"] for run in body["runs"]] == list(reversed(ids))
    # queue_position is part of the contract from day one; nothing queues yet.
    assert all(run["queue_position"] is None for run in body["runs"])

    succeeded = (await client.get("/api/runs?status=succeeded")).json()
    assert succeeded["total"] == 3
    empty = (await client.get("/api/runs?status=failed")).json()
    assert empty["runs"] == [] and empty["total"] == 0
    paged = (await client.get("/api/runs?limit=1&offset=1")).json()
    assert [run["id"] for run in paged["runs"]] == [ids[1]]
    assert paged["total"] == 3


async def test_list_reports_queue_position_for_queued_rows(client, run_app):
    """The field is computed, not hardcoded null — #123 inherits it working."""
    first = await run_app.store.create_run(graph_snapshot=_graph(), options={},
                                           provenance=RunProvenance())
    second = await run_app.store.create_run(graph_snapshot=_graph(), options={},
                                            provenance=RunProvenance())
    body = (await client.get("/api/runs?status=queued")).json()
    positions = {run["id"]: run["queue_position"] for run in body["runs"]}
    assert positions == {first.id: 1, second.id: 2}


async def test_queue_position_is_counted_within_each_device(client, run_app):
    """Per queue_key, not global (#123): a CPU run is not "fourth" because
    three CUDA runs happen to have been submitted first."""
    made = []
    for queue_key in ("cuda:0", "cuda:0", "cpu", "cuda:1", "cpu"):
        made.append((queue_key, await run_app.store.create_run(
            graph_snapshot=_graph(), options={}, queue_key=queue_key,
            provenance=RunProvenance())))

    body = (await client.get("/api/runs?status=queued")).json()
    positions = {run["id"]: run["queue_position"] for run in body["runs"]}
    assert [positions[record.id] for _key, record in made] == [1, 2, 1, 1, 2]


async def test_list_rejects_an_unknown_status(client):
    assert (await client.get("/api/runs?status=nope")).status_code == 400


async def test_get_run_reports_the_event_cursor(client):
    run_id = (await client.post("/api/runs",
                                json={"graph": _graph()})).json()["run_id"]
    record = await _poll_until_terminal(client, run_id)
    events = (await client.get(f"/api/runs/{run_id}/events")).json()
    assert record["last_cursor"] == events["events"][-1]["cursor"]
    assert record["active"] is False
    assert "graph_snapshot" not in record       # never on a list/detail row


async def test_get_unknown_run_is_404(client):
    for path in ("", "/events", "/metrics"):
        response = await client.get(f"/api/runs/nope{path}")
        assert response.status_code == 404, path


# ── cancel ────────────────────────────────────────────────────────────────


async def test_cancel_stops_a_running_run(client):
    run_id = (await client.post("/api/runs", json={
        "graph": _graph("_ApiSlow", {"seconds": 0.4})})).json()["run_id"]
    response = await client.post(f"/api/runs/{run_id}/cancel")
    assert response.status_code == 200
    assert response.json() == {"run_id": run_id, "status": "running",
                               "cancelled": True}

    record = await _poll_until_terminal(client, run_id)
    assert record["status"] == "cancelled"
    events = (await client.get(f"/api/runs/{run_id}/events")).json()
    assert events["events"][-1]["type"] == "execution_stopped"
    assert events["events"][-1]["payload"] == {"reason": "cancelled"}


async def test_cancel_of_a_finished_run_is_a_no_op(client):
    run_id = (await client.post("/api/runs",
                                json={"graph": _graph()})).json()["run_id"]
    await _poll_until_terminal(client, run_id)
    response = await client.post(f"/api/runs/{run_id}/cancel")
    assert response.status_code == 200
    assert response.json() == {"run_id": run_id, "status": "succeeded",
                               "cancelled": False}


async def test_cancel_unknown_run_is_404(client):
    assert (await client.post("/api/runs/nope/cancel")).status_code == 404


# ── events ────────────────────────────────────────────────────────────────


async def test_events_paginate_by_cursor(client):
    run_id = (await client.post("/api/runs",
                                json={"graph": _graph()})).json()["run_id"]
    await _poll_until_terminal(client, run_id)

    everything = (await client.get(f"/api/runs/{run_id}/events")).json()
    cursors = [event["cursor"] for event in everything["events"]]
    assert cursors == list(range(1, len(cursors) + 1))

    tail = (await client.get(
        f"/api/runs/{run_id}/events?cursor={cursors[1]}")).json()
    assert [event["cursor"] for event in tail["events"]] == cursors[2:]

    exhausted = (await client.get(
        f"/api/runs/{run_id}/events?cursor={cursors[-1]}")).json()
    assert exhausted["events"] == []
    assert exhausted["cursor"] == cursors[-1]   # cursor never goes backwards

    capped = (await client.get(f"/api/runs/{run_id}/events?limit=2")).json()
    assert len(capped["events"]) == 2


async def test_events_long_poll_returns_immediately_on_a_dead_run(client):
    run_id = (await client.post("/api/runs",
                                json={"graph": _graph()})).json()["run_id"]
    await _poll_until_terminal(client, run_id)
    last = (await client.get(f"/api/runs/{run_id}/events")).json()["cursor"]

    started = time.monotonic()
    response = await client.get(
        f"/api/runs/{run_id}/events?cursor={last}&wait=5")
    assert response.status_code == 200
    assert response.json()["events"] == []
    assert time.monotonic() - started < 2.0, "long poll hung on a finished run"


async def test_events_long_poll_wakes_on_the_next_event(client):
    run_id = (await client.post("/api/runs", json={
        "graph": _graph("_ApiSlow", {"seconds": 0.3})})).json()["run_id"]
    first = (await client.get(f"/api/runs/{run_id}/events?wait=5")).json()
    assert first["events"], "no events at all"

    started = time.monotonic()
    later = await client.get(
        f"/api/runs/{run_id}/events?cursor={first['cursor']}&wait=10")
    elapsed = time.monotonic() - started
    assert later.json()["events"], "timed out instead of waking"
    assert elapsed < 9.0, "woke on the timeout, not on the event"
    await _poll_until_terminal(client, run_id)


async def test_oversized_image_events_are_served_elided(client):
    """The DB never stores the blob, so the API never serves it."""
    run_id = (await client.post("/api/runs", json={
        "graph": _graph("_ApiImage", {"size": 200_000})})).json()["run_id"]
    await _poll_until_terminal(client, run_id)

    body = (await client.get(f"/api/runs/{run_id}/events")).json()
    images = [entry
              for event in body["events"] if event["type"] == "node_status"
              for entry in (event["payload"].get("outputs") or [])
              if entry.get("output_kind") == "image"]
    assert images, "the declared image port produced no entry at all"
    assert images[0]["elided"] is True
    assert images[0]["port"] == "image"        # placeholder still placeable
    assert images[0]["bytes"] > 200_000
    assert len(response_text := json.dumps(body)) < 200_000, response_text[:200]


async def test_events_response_is_byte_bounded(client, monkeypatch):
    """`limit` alone must not multiply into an unbounded response body.

    A short page is normal — the cursor says where to resume — so the whole
    log is still reachable, one bounded page at a time, with no gaps.
    """
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "RUN_EVENTS_RESPONSE_CAP_BYTES", 400)
    run_id = (await client.post("/api/runs", json={
        "graph": _graph("_ApiImage", {"size": 300})})).json()["run_id"]
    await _poll_until_terminal(client, run_id)

    seen: list[dict[str, Any]] = []
    cursor = 0
    for _ in range(50):
        page = (await client.get(
            f"/api/runs/{run_id}/events?cursor={cursor}&limit=2000")).json()
        if not page["events"]:
            break
        assert len(page["events"]) < 8, "the byte budget did not bound the page"
        # The cursor tracks what was RETURNED, so resuming skips nothing.
        assert page["cursor"] == page["events"][-1]["cursor"]
        seen.extend(page["events"])
        cursor = page["cursor"]

    assert len(seen) >= 8, "the run should have produced more than one page"
    assert [event["cursor"] for event in seen] == list(range(1, len(seen) + 1))
    assert seen[-1]["type"] == "execution_complete"


async def test_events_reject_an_out_of_range_wait(client):
    run_id = (await client.post("/api/runs",
                                json={"graph": _graph()})).json()["run_id"]
    await _poll_until_terminal(client, run_id)
    assert (await client.get(
        f"/api/runs/{run_id}/events?wait=600")).status_code == 422
    assert (await client.get(
        f"/api/runs/{run_id}/events?cursor=-1")).status_code == 422


# ── metrics ───────────────────────────────────────────────────────────────


async def test_metrics_json_and_name_filter(client):
    run_id = (await client.post("/api/runs", json={
        "graph": _graph("_ApiMetrics")})).json()["run_id"]
    await _poll_until_terminal(client, run_id)

    body = (await client.get(f"/api/runs/{run_id}/metrics")).json()
    assert body["names"] == ["loss"]            # "note" is a string, not a series
    assert [(point["step"], point["value"]) for point in body["metrics"]] == [
        (1, 1.0), (2, 0.5)]
    assert body["metrics"][0]["node_id"] == "mid"

    filtered = (await client.get(
        f"/api/runs/{run_id}/metrics?name=loss")).json()
    assert len(filtered["metrics"]) == 2
    assert (await client.get(
        f"/api/runs/{run_id}/metrics?name=nope")).json()["metrics"] == []


async def test_metrics_csv_export(client):
    run_id = (await client.post("/api/runs", json={
        "graph": _graph("_ApiMetrics")})).json()["run_id"]
    await _poll_until_terminal(client, run_id)

    response = await client.get(f"/api/runs/{run_id}/metrics?format=csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert run_id in response.headers["content-disposition"]

    rows = list(csv.reader(io.StringIO(response.text)))
    assert rows[0] == ["run_id", "node_id", "name", "step", "value", "ts"]
    assert len(rows) == 3
    assert [row[2] for row in rows[1:]] == ["loss", "loss"]
    assert [row[3] for row in rows[1:]] == ["1", "2"]
    assert float(rows[1][4]) == 1.0


async def test_list_rows_carry_the_last_value_of_every_series(client):
    """The Runs table's "final loss" column, without a request per row."""
    with_metrics = (await client.post("/api/runs", json={
        "graph": _graph("_ApiMetrics")})).json()["run_id"]
    without = (await client.post("/api/runs",
                                 json={"graph": _graph()})).json()["run_id"]
    for run_id in (with_metrics, without):
        await _poll_until_terminal(client, run_id)

    rows = {run["id"]: run for run in (await client.get("/api/runs")).json()["runs"]}
    # _ApiMetrics logs loss 1/epoch for two epochs -> last value is 0.5.
    assert rows[with_metrics]["final_metrics"] == {"loss": 0.5}
    assert rows[without]["final_metrics"] == {}
    # The detail endpoint agrees with the row rather than always saying {}.
    detail = (await client.get(f"/api/runs/{with_metrics}")).json()
    assert detail["final_metrics"] == {"loss": 0.5}


async def test_metrics_reject_an_unknown_format(client):
    run_id = (await client.post("/api/runs",
                                json={"graph": _graph()})).json()["run_id"]
    await _poll_until_terminal(client, run_id)
    assert (await client.get(
        f"/api/runs/{run_id}/metrics?format=xml")).status_code == 422


# ── artifacts ─────────────────────────────────────────────────────────────


async def test_artifacts_list_is_oldest_first_and_filterable_by_kind(client,
                                                                     run_app):
    run_id = (await client.post("/api/runs",
                                json={"graph": _graph()})).json()["run_id"]
    await _poll_until_terminal(client, run_id)
    await run_app.store.add_artifact(run_id, "checkpoint", "runs/a/epoch1.pt",
                                     meta={"epoch": 1})
    await run_app.store.add_artifact(run_id, "export", "runs/a/model.onnx")

    body = (await client.get(f"/api/runs/{run_id}/artifacts")).json()
    assert body["run_id"] == run_id
    assert [a["kind"] for a in body["artifacts"]] == ["checkpoint", "export"]
    assert body["artifacts"][0]["path"] == "runs/a/epoch1.pt"
    assert body["artifacts"][0]["meta"] == {"epoch": 1}
    assert body["artifacts"][1]["meta"] is None
    assert isinstance(body["artifacts"][0]["id"], int)
    assert body["artifacts"][0]["created_at"]

    filtered = (await client.get(
        f"/api/runs/{run_id}/artifacts?kind=checkpoint")).json()
    assert [a["path"] for a in filtered["artifacts"]] == ["runs/a/epoch1.pt"]


async def test_artifacts_of_an_unknown_kind_are_an_empty_list_not_a_400(client):
    """The kind vocabulary is open, so an unknown one is simply no rows."""
    run_id = (await client.post("/api/runs",
                                json={"graph": _graph()})).json()["run_id"]
    await _poll_until_terminal(client, run_id)
    response = await client.get(f"/api/runs/{run_id}/artifacts?kind=nope")
    assert response.status_code == 200
    assert response.json()["artifacts"] == []


async def test_artifacts_of_an_unknown_run_is_404(client):
    assert (await client.get("/api/runs/nope/artifacts")).status_code == 404


# ── delete ────────────────────────────────────────────────────────────────


async def test_delete_removes_a_finished_run_and_cascades_to_its_children(
        client, run_app):
    run_id = (await client.post("/api/runs", json={
        "graph": _graph("_ApiMetrics")})).json()["run_id"]
    await _poll_until_terminal(client, run_id)
    await run_app.store.add_artifact(run_id, "checkpoint", "runs/a/final.pt")
    assert (await client.get(f"/api/runs/{run_id}/metrics")).json()["metrics"]
    assert (await client.get(f"/api/runs/{run_id}/events")).json()["events"]

    response = await client.delete(f"/api/runs/{run_id}")
    assert response.status_code == 200
    assert response.json() == {"run_id": run_id, "deleted": True}

    # The row is gone, and so is everything that pointed at it — asserted
    # through the store rather than the (now 404-ing) endpoints, because a
    # cascade that silently failed would look identical from outside.
    assert await run_app.store.get_run(run_id) is None
    assert await run_app.store.get_metrics(run_id) == []
    assert await run_app.store.get_events(run_id) == []
    assert await run_app.store.list_artifacts(run_id) == []
    for path in ("", "/events", "/metrics", "/artifacts"):
        assert (await client.get(
            f"/api/runs/{run_id}{path}")).status_code == 404, path


async def test_delete_drops_the_run_from_the_list(client):
    keep = (await client.post("/api/runs",
                              json={"graph": _graph(), "name": "keep"})
            ).json()["run_id"]
    drop = (await client.post("/api/runs",
                              json={"graph": _graph(), "name": "drop"})
            ).json()["run_id"]
    for run_id in (keep, drop):
        await _poll_until_terminal(client, run_id)

    assert (await client.delete(f"/api/runs/{drop}")).status_code == 200
    body = (await client.get("/api/runs")).json()
    assert [run["id"] for run in body["runs"]] == [keep]
    assert body["total"] == 1


async def test_delete_refuses_a_running_run_with_409(client, run_app):
    run_id = (await client.post("/api/runs", json={
        "graph": _graph("_ApiSlow", {"seconds": 0.4})})).json()["run_id"]
    # The run is in flight: the service, not the row, is the authority.
    assert run_app.is_active(run_id)
    response = await client.delete(f"/api/runs/{run_id}")
    assert response.status_code == 409
    assert "cancel it first" in response.json()["detail"]

    # Refusing must not have damaged it — it still finishes normally.
    assert (await client.get(f"/api/runs/{run_id}")).status_code == 200
    await client.post(f"/api/runs/{run_id}/cancel")
    await _poll_until_terminal(client, run_id)
    assert (await client.delete(f"/api/runs/{run_id}")).status_code == 200


async def test_delete_refuses_a_queued_run_with_409(client, run_app):
    """A queued run's place in the FIFO lives in memory, not in the row.

    Runs against a one-slot service so the second submit is genuinely
    parked; limits are passed explicitly rather than monkeypatched onto
    ``settings`` because ``QueueLimits.from_settings`` is read once, when the
    service is built — which the fixture already did.
    """
    single = RunService(run_app.store, shutdown_grace_s=5.0,
                        limits=QueueLimits(cpu=1, gpu=1, interactive=1))
    app.state.run_service = single
    try:
        first = (await client.post("/api/runs", json={
            "graph": _graph("_ApiSlow", {"seconds": 0.4})})).json()["run_id"]
        second = (await client.post("/api/runs", json={
            "graph": _graph("_ApiSlow", {"seconds": 0.4})})).json()
        assert second["status"] == "queued"
        queued_id = second["run_id"]
        assert single.is_queued(queued_id)

        response = await client.delete(f"/api/runs/{queued_id}")
        assert response.status_code == 409
        assert "cancel it first" in response.json()["detail"]

        for run_id in (first, queued_id):
            await client.post(f"/api/runs/{run_id}/cancel")
            await _poll_until_terminal(client, run_id)
        # Cancelled is terminal, so the same row deletes cleanly now.
        assert (await client.delete(f"/api/runs/{queued_id}")).status_code == 200
    finally:
        await single.shutdown()
        app.state.run_service = run_app


async def test_delete_unknown_run_is_404(client):
    assert (await client.delete("/api/runs/nope")).status_code == 404


async def test_delete_twice_is_a_404_the_second_time(client):
    run_id = (await client.post("/api/runs",
                                json={"graph": _graph()})).json()["run_id"]
    await _poll_until_terminal(client, run_id)
    assert (await client.delete(f"/api/runs/{run_id}")).status_code == 200
    assert (await client.delete(f"/api/runs/{run_id}")).status_code == 404


async def test_delete_requires_the_session_token(client):
    """DELETE is mutating, so auth_guard must reject an unauthenticated one."""
    run_id = (await client.post("/api/runs",
                                json={"graph": _graph()})).json()["run_id"]
    await _poll_until_terminal(client, run_id)
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url=BASE_URL) as anonymous:
        assert (await anonymous.delete(
            f"/api/runs/{run_id}")).status_code == 403
    # Still there — the refusal was not a silent success.
    assert (await client.get(f"/api/runs/{run_id}")).status_code == 200


# ── openapi / auth-surface sanity ─────────────────────────────────────────


def test_runs_routes_are_not_under_an_auth_exempt_prefix():
    """/api/runs relies on auth_guard, so it must NOT be prefix-exempt —
    otherwise its mutating routes would sail through with no auth at all."""
    from app.main import _prefix_exempt

    for path in ("/api/runs", "/api/runs/abc", "/api/runs/abc/cancel",
                 "/api/runs/abc/artifacts"):
        assert not _prefix_exempt(path), path


def test_runs_routes_appear_in_the_openapi_document():
    paths = app.openapi()["paths"]
    assert "/api/runs" in paths
    assert "/api/runs/{run_id}" in paths
    assert "/api/runs/{run_id}/cancel" in paths
    assert "/api/runs/{run_id}/events" in paths
    assert "/api/runs/{run_id}/metrics" in paths
    assert "/api/runs/{run_id}/artifacts" in paths
    # DELETE shares the detail path with GET rather than adding a new one.
    assert set(paths["/api/runs/{run_id}"]) == {"get", "delete"}
