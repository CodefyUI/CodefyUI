"""Sweeps and studies with ``options.target`` over fake worker agents (#777).

Every child of a sweep or a study inherits the parent's ``options.target``;
with ``target.worker == "any"`` each child is placed on its own free
``(worker, device)`` slot, so two agents with one card each run two
variants at once. The fake agent speaks ``core.worker_protocol`` by hand,
like ``test_ws_worker.py`` (duplicated rather than imported: test modules
stay independent).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from httpx import AsyncClient
from httpx_ws import aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport

from app.core import worker_protocol as wp
from app.core.auth import TOKEN_HEADER, TOKEN_QUERY_PARAM, session_token
from app.core.db import Database
from app.core.run_service import QueueLimits, RunService
from app.core.run_store import RunStore
from app.core.study_coordinator import StudyCoordinator
from app.core.study_store import StudyStore
from app.core.sweep_store import SweepStore
from app.core.workers import WorkerRegistry
from app.main import app

BASE = "http://127.0.0.1:8000"
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
WS_PATH = f"{wp.WORKER_WS_PATH}?{TOKEN_QUERY_PARAM}={session_token()}"


def _graph() -> dict[str, Any]:
    return {"nodes": [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "x", "type": "TensorCreate",
         "data": {"params": {"shape": "2", "fill": "full", "value": 0.0}}}],
        "edges": [{"id": "t", "source": "start", "target": "x",
                   "sourceHandle": "trigger", "type": "trigger"}]}


@pytest.fixture
async def env(tmp_path):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    registry = WorkerRegistry(heartbeat_s=10)
    store = RunStore(database)
    service = RunService(
        store, workers=registry, shutdown_grace_s=5.0,
        limits=QueueLimits(cpu=2, gpu=1, interactive=2, overrides=()))
    coordinator = StudyCoordinator(service, StudyStore(database))
    app.state.db = database
    app.state.run_service = service
    app.state.worker_registry = registry
    app.state.sweep_store = SweepStore(database)
    app.state.study_coordinator = coordinator
    try:
        yield service, coordinator
    finally:
        await coordinator.shutdown()
        await service.shutdown()
        await registry.shutdown()
        database.close()
        for attribute in ("db", "run_service", "worker_registry",
                          "sweep_store", "study_coordinator"):
            if hasattr(app.state, attribute):
                delattr(app.state, attribute)


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGIWebSocketTransport(app=app),
                       base_url=BASE, headers={TOKEN_HEADER: session_token()})


class FakeAgent:
    def __init__(self, ws: Any) -> None:
        self.ws = ws

    async def send(self, frame: Any) -> None:
        await self.ws.send_text(wp.encode_frame(frame))

    async def recv(self, timeout: float = 10.0) -> Any:
        raw = await asyncio.wait_for(self.ws.receive_text(), timeout)
        return wp.decode_control_frame(raw)

    async def assignment(self, timeout: float = 10.0) -> wp.Assign:
        frame = await self.recv(timeout)
        assert isinstance(frame, wp.Assign), frame
        return frame

    async def nothing(self, timeout: float = 0.3) -> None:
        with pytest.raises(asyncio.TimeoutError):
            await self.recv(timeout)

    async def hello(self, name: str, devices: tuple = ("cpu",)) -> Any:
        await self.send(wp.Hello(name=name, version="t", devices=[
            wp.DeviceReport(value=d, kind=d.split(":")[0]) for d in devices]))
        return await self.recv()

    async def finish(self, assign: wp.Assign, *, metric: str | None = None,
                     value: float = 0.0) -> None:
        run_id = assign.run_id
        await self.send(wp.RunStarted(run_id=run_id, seq=1, pid=1))
        seq = 2
        if metric is not None:
            await self.send(wp.Metrics(run_id=run_id, seq=seq, points=[
                wp.MetricPointReport(name=metric, value=value, step=1,
                                     node_id="x")]))
            seq += 1
        await self.send(wp.RunFinished(run_id=run_id, seq=seq,
                                       status="succeeded"))


async def _terminal(service: RunService, run_id: str,
                    timeout: float = 10.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = await service.store.get_run(run_id)
        if record is not None and record.status in TERMINAL \
                and run_id not in service.active_run_ids():
            return record
        await asyncio.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish")


async def _until(predicate: Any, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not await predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met")
        await asyncio.sleep(0.02)


def _sweep_body(values: list[float], **options: Any) -> dict[str, Any]:
    return {
        "base_graph": _graph(),
        "sweep_spec": {"method": "grid", "params": [
            {"node_id": "x", "param": "value", "values": values}]},
        "objective": {"metric": "val_loss", "direction": "minimize"},
        "options": options,
    }


# ── sweeps ────────────────────────────────────────────────────────────────


async def test_a_target_any_sweep_spreads_its_variants_over_free_slots(env):
    """The acceptance criterion: four variants, two agents with one card
    each, two variants running at once on different (worker, device)
    slots; each freed slot takes the next variant."""
    service, _ = env
    async with _client() as client, \
            aconnect_ws(WS_PATH, client) as ws_a, \
            aconnect_ws(WS_PATH, client) as ws_b:
        a, b = FakeAgent(ws_a), FakeAgent(ws_b)
        await a.hello("a-box", ("cpu", "cuda:0"))
        await b.hello("b-box", ("cpu", "cuda:0"))
        response = await client.post("/api/sweeps", json=_sweep_body(
            [1.0, 2.0, 3.0, 4.0], device="cuda",
            target={"worker": "any"}))
        assert response.status_code == 201, response.text
        sweep = response.json()
        ids = [v["run_id"] for v in sweep["variants"]]

        first_a, first_b = await a.assignment(), await b.assignment()
        # Two variants at once, one per worker, each on that worker's card.
        assert [first_a.run_id, first_b.run_id] == ids[:2]
        assert (first_a.device, first_b.device) == ("cuda:0", "cuda:0")
        for run_id in ids[2:]:
            assert (await service.store.get_run(run_id)).status == "queued"
        assert service.queue_snapshot() == {"any/cuda": ids[2:]}
        await a.nothing()
        await b.nothing()

        detail = (await client.get(f"/api/sweeps/{sweep['sweep_id']}")).json()
        by_run = {v["run_id"]: v for v in detail["variants"]}
        assert (by_run[ids[0]]["worker_id"], by_run[ids[0]]["devices"],
                by_run[ids[0]]["queue_key"]) == ("a-box", ["cuda:0"],
                                                 "a-box/cuda:0")
        assert by_run[ids[1]]["worker_id"] == "b-box"
        # Waiting: still `any`, on the shared `any/cuda` line.
        assert (by_run[ids[2]]["worker_id"], by_run[ids[2]]["queue_key"]) \
            == ("any", "any/cuda")

        # A freed slot on b-box serves the next variant there.
        await b.finish(first_b, metric="val_loss", value=2.0)
        third = await b.assignment()
        assert third.run_id == ids[2]
        await a.finish(first_a, metric="val_loss", value=1.0)
        fourth = await a.assignment()
        assert fourth.run_id == ids[3]
        await b.finish(third, metric="val_loss", value=3.0)
        await a.finish(fourth, metric="val_loss", value=4.0)
        records = [await _terminal(service, r) for r in ids]

        assert [r.status for r in records] == ["succeeded"] * 4
        assert [r.worker_id for r in records] == [
            "a-box", "b-box", "b-box", "a-box"]
        detail = (await client.get(f"/api/sweeps/{sweep['sweep_id']}")).json()
    assert detail["state"] == "finished"
    assert [v["run_id"] for v in detail["variants"]] == ids  # rank order
    assert [v["worker_id"] for v in detail["variants"]] == [
        "a-box", "b-box", "b-box", "a-box"]
    assert service._slots == {}


async def test_seeded_variants_on_workers_still_run_side_by_side(env):
    """``seed_variants`` makes every variant a seeded run; on a worker each
    one is a subprocess with RNGs of its own, so they do not take this
    process's seed gate and two of them still run at once."""
    service, _ = env
    async with _client() as client, \
            aconnect_ws(WS_PATH, client) as ws_a, \
            aconnect_ws(WS_PATH, client) as ws_b:
        a, b = FakeAgent(ws_a), FakeAgent(ws_b)
        await a.hello("a-box", ("cuda:0",))
        await b.hello("b-box", ("cuda:0",))
        body = _sweep_body([1.0, 2.0], device="cuda",
                           target={"worker": "any"})
        body["sweep_spec"]["seed"] = 7
        body["seed_variants"] = True
        response = await client.post("/api/sweeps", json=body)
        assert response.status_code == 201, response.text
        first, second = await a.assignment(), await b.assignment()
        assert (first.options["seed"], second.options["seed"]) == (7, 8)
        await a.finish(first)
        await b.finish(second)
        for assign in (first, second):
            assert (await _terminal(service, assign.run_id)).status \
                == "succeeded"


async def test_a_named_worker_target_is_inherited_by_every_variant(env):
    service, _ = env
    async with _client() as client, aconnect_ws(WS_PATH, client) as ws:
        agent = FakeAgent(ws)
        await agent.hello("box", ("cpu", "cuda:0", "cuda:1"))
        response = await client.post("/api/sweeps", json=_sweep_body(
            [1.0, 2.0, 3.0], device="cuda", target={"worker": "box"}))
        assert response.status_code == 201, response.text
        ids = [v["run_id"] for v in response.json()["variants"]]
        # A bare kind takes each free card in turn; the third waits.
        assigned = [await agent.assignment(), await agent.assignment()]
        assert [(x.run_id, x.device) for x in assigned] == [
            (ids[0], "cuda:0"), (ids[1], "cuda:1")]
        await agent.nothing()
        await agent.finish(assigned[0])
        third = await agent.assignment()
        assert (third.run_id, third.device) == (ids[2], "cuda:0")
        for assign in (assigned[1], third):
            await agent.finish(assign)
        for run_id in ids:
            record = await _terminal(service, run_id)
            assert (record.status, record.worker_id) == ("succeeded", "box")


@pytest.mark.parametrize("options, needle", [
    ({"target": {"worker": "gpu-box"}}, "has not connected"),
    ({"device": "cuda:3", "target": {"worker": "any"}},
     "neither this server nor a known worker offers cuda:3"),
    ({"record_outputs": True, "target": {"worker": "any"}},
     "record_outputs"),
])
async def test_a_target_the_service_would_refuse_is_a_400_with_no_rows(
        env, options, needle):
    service, _ = env
    async with _client() as client:
        response = await client.post("/api/sweeps",
                                     json=_sweep_body([1.0, 2.0], **options))
        assert response.status_code == 400, response.text
        assert needle in response.json()["detail"]
    assert await service.store.list_runs() == []
    count = service.store.db.run(
        lambda conn: conn.execute("SELECT COUNT(*) FROM sweeps").fetchone()[0])
    assert await count == 0


# ── studies ───────────────────────────────────────────────────────────────


def _study_body(**extra: Any) -> dict[str, Any]:
    return {
        "objective": {"node_id": "x", "metric": "val_loss",
                      "direction": "minimize"},
        "base_graph": _graph(),
        "variants": [
            {"label": "low", "ops": [
                {"node_id": "x", "param": "value", "value": 1.0}]},
            {"label": "high", "ops": [
                {"node_id": "x", "param": "value", "value": 2.0}]}],
        "repetitions": 2,
        "seed": 11,
        **extra,
    }


async def test_a_study_honours_max_parallel_runs_across_workers(env):
    """Seeded children on two workers: ``max_parallel_runs: 2`` puts two of
    them on two cards at once (in-process they would execute one at a
    time behind the seed gate), and never a third."""
    service, coordinator = env
    async with _client() as client, \
            aconnect_ws(WS_PATH, client) as ws_a, \
            aconnect_ws(WS_PATH, client) as ws_b:
        a, b = FakeAgent(ws_a), FakeAgent(ws_b)
        await a.hello("a-box", ("cuda:0",))
        await b.hello("b-box", ("cuda:0",))
        response = await client.post("/api/studies", json=_study_body(
            options={"device": "cuda", "target": {"worker": "any"}},
            budget={"max_parallel_runs": 2}))
        assert response.status_code == 201, response.text
        study_id = response.json()["study_id"]

        first_a, first_b = await a.assignment(), await b.assignment()
        # Repetition-major order: both variants of repetition 0, seed 11.
        assert (first_a.options["seed"], first_b.options["seed"]) == (11, 11)
        assert first_a.options["target"] == {"worker": "any",
                                             "devices": None}
        await a.nothing()
        await b.nothing()

        values = {first_a.run_id: 1.0, first_b.run_id: 2.0}
        await a.finish(first_a, metric="val_loss", value=values[first_a.run_id])
        await b.finish(first_b, metric="val_loss", value=values[first_b.run_id])
        # Repetition 1 follows, again two at once; which agent takes which
        # child depends on which slot freed first.
        second = await asyncio.gather(
            a.assignment(), b.assignment())
        assert {x.options["seed"] for x in second} == {12}
        for agent, assign in zip((a, b), second):
            await agent.finish(assign, metric="val_loss", value=1.0)

        async def _settled() -> bool:
            study = await coordinator.store.get_study(study_id)
            return study is not None and study.status == "succeeded"
        await _until(_settled)
    runs = [await service.store.get_run(x.run_id)
            for x in (first_a, first_b, *second)]
    assert sorted(r.worker_id for r in runs) == [
        "a-box", "a-box", "b-box", "b-box"]


async def test_a_study_target_the_service_would_refuse_is_a_400(env):
    service, coordinator = env
    async with _client() as client:
        response = await client.post("/api/studies", json=_study_body(
            options={"target": {"worker": "gpu-box"}}))
        assert response.status_code == 400, response.text
        assert "has not connected" in response.json()["detail"]
    studies, total = await coordinator.store.list_studies()
    assert (studies, total) == ([], 0)


async def test_an_idempotent_retry_returns_the_study_after_its_worker_left(
        env):
    """The refusal is for NEW studies: a retry of one that exists returns
    it, whatever the workers are doing now."""
    service, coordinator = env
    body = _study_body(options={"device": "cuda",
                                "target": {"worker": "box"}},
                       idempotency_key="k-777")
    async with _client() as client:
        async with aconnect_ws(WS_PATH, client) as ws:
            agent = FakeAgent(ws)
            await agent.hello("box", ("cuda:0",))
            created = await client.post("/api/studies", json=body)
            assert created.status_code == 201, created.text
            await agent.assignment()
        retry = await client.post("/api/studies", json=body)
        assert retry.status_code == 200, retry.text
        assert retry.json()["study_id"] == created.json()["study_id"]
        assert retry.json()["created"] is False
        # The lookup comes before the placement check, so the key reused
        # for a different request is a 409 even when that request's target
        # would be refused.
        conflict = await client.post("/api/studies", json={
            **body, "options": {"target": {"worker": "nobody"}}})
        assert conflict.status_code == 409, conflict.text
        # A new study for the known-but-away worker is accepted and waits.
        waiting = await client.post("/api/studies", json={
            **body, "idempotency_key": "k-other"})
        assert waiting.status_code == 201, waiting.text
        # An unknown worker is refused.
        other = await client.post("/api/studies", json={
            **body, "options": {"target": {"worker": "nobody"}},
            "idempotency_key": "k-third"})
        assert other.status_code == 400
        assert "has not connected" in other.json()["detail"]


# ── #777 review: one submit rule, one batch check, study placement ───────


async def test_a_sweep_for_a_worker_that_went_away_waits_for_it(env):
    """A worker this server has seen is waited for, not refused, so a sweep
    submitted (or still submitting) while it is away queues every variant
    and runs them once it is back."""
    service, _ = env
    async with _client() as client:
        async with aconnect_ws(WS_PATH, client) as ws:
            await FakeAgent(ws).hello("box", ("cuda:0",))
        response = await client.post("/api/sweeps", json=_sweep_body(
            [1.0, 2.0], device="cuda", target={"worker": "box"}))
        assert response.status_code == 201, response.text
        ids = [v["run_id"] for v in response.json()["variants"]]
        row = (await client.get(f"/api/runs/{ids[0]}")).json()
        assert (row["status"], row["queue_reason"]) == (
            "queued", "waiting for worker box to reconnect")
        async with aconnect_ws(WS_PATH, client) as ws:
            agent = FakeAgent(ws)
            await agent.hello("box", ("cuda:0",))
            for run_id in ids:
                assign = await agent.assignment()
                assert assign.run_id == run_id
                await agent.finish(assign)
                assert (await _terminal(service, run_id)).status \
                    == "succeeded"


async def test_the_batch_check_runs_once_per_distinct_settings(env,
                                                                monkeypatch):
    service, _ = env
    calls: list[Any] = []
    real = service._prepare

    def _spy(graph: Any, options: Any, **kw: Any) -> Any:
        calls.append((graph.get("settings"), kw))
        return real(graph, options, **kw)

    monkeypatch.setattr(service, "_prepare", _spy)
    graphs = [_graph(), _graph(),
              {**_graph(), "settings": {"device": "cpu"}},
              {**_graph(), "settings": {"device": "cpu"}}]
    service.check_placeable_batch(graphs, {})
    assert calls == [(None, {"batch_child": True}),
                     ({"device": "cpu"}, {"batch_child": True})]

    async with _client() as client:
        calls.clear()
        body = _study_body()
        body["variants"].append({"label": "on-cpu", "graph": {
            **_graph(), "settings": {"device": "cpu"}}})
        response = await client.post("/api/studies", json=body)
        assert response.status_code == 201, response.text
        # Two distinct settings among three variants: two checks, before
        # any child was submitted.
        assert [c[0] for c in calls[:2]] == [None, {"device": "cpu"}]
        assert all(c[1] == {"batch_child": True} for c in calls[:2])


async def test_the_batch_check_refuses_an_interactive_lane_and_a_shutdown(
        env):
    service, _ = env
    from app.core.run_service import RunServiceUnavailable, RunSubmitError
    with pytest.raises(RunSubmitError, match="interactive"):
        service.check_placeable_batch([_graph()], {"lane": "interactive"})
    service._shutting_down = True
    try:
        with pytest.raises(RunServiceUnavailable):
            service.check_placeable_batch([_graph()], {})
        async with _client() as client:
            sweep = await client.post("/api/sweeps", json=_sweep_body([1.0]))
            assert sweep.status_code == 503, sweep.text
            study = await client.post("/api/studies", json=_study_body())
            assert study.status_code == 503, study.text
    finally:
        service._shutting_down = False


async def test_the_study_detail_says_where_each_child_runs(env):
    service, coordinator = env
    async with _client() as client, \
            aconnect_ws(WS_PATH, client) as ws_a, \
            aconnect_ws(WS_PATH, client) as ws_b:
        a, b = FakeAgent(ws_a), FakeAgent(ws_b)
        await a.hello("a-box", ("cuda:0",))
        await b.hello("b-box", ("cuda:0",))
        response = await client.post("/api/studies", json=_study_body(
            repetitions=1,
            options={"device": "cuda", "target": {"worker": "any"}},
            budget={"max_parallel_runs": 2}))
        assert response.status_code == 201, response.text
        study_id = response.json()["study_id"]
        first_a, first_b = await a.assignment(), await b.assignment()

        async def _placed() -> bool:
            detail = (await client.get(f"/api/studies/{study_id}")).json()
            return all(v["repetitions"][0]["current"].get("worker_id")
                       not in (None, "any") for v in detail["variants"])
        await _until(_placed)
        detail = (await client.get(f"/api/studies/{study_id}")).json()
        placed = {v["repetitions"][0]["current"]["run_id"]:
                  v["repetitions"][0]["current"] for v in detail["variants"]}
        assert (placed[first_a.run_id]["worker_id"],
                placed[first_a.run_id]["devices"],
                placed[first_a.run_id]["queue_key"]) == (
            "a-box", ["cuda:0"], "a-box/cuda:0")
        assert placed[first_b.run_id]["worker_id"] == "b-box"
        attempts = [a for v in detail["variants"] for a in v["attempts"]]
        assert {a["worker_id"] for a in attempts} == {"a-box", "b-box"}
        await a.finish(first_a)
        await b.finish(first_b)
        for assign in (first_a, first_b):
            await _terminal(service, assign.run_id)
        # A child whose run row is gone reports no placement.
        await service.store.delete_run(first_a.run_id)
        detail = (await client.get(f"/api/studies/{study_id}")).json()
        gone = [a for v in detail["variants"] for a in v["attempts"]
                if a["run_id"] == first_a.run_id]
        assert gone and "worker_id" not in gone[0]
