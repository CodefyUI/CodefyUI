"""Durable studies (#343): the store, the coordinator and /api/studies.

The acceptance bar from the issue, each as its own test: a study runs N x R
attempts as queued child runs; an idempotent retry creates nothing and a
key reused for another payload is a 409; completion ranks with exact ties
labelled as ties and a variant with a failed repetition ineligible; a cancel
at a (variant, repetition) boundary keeps the completed attempts; a resume
after a simulated restart (a NEW RunService over the same database) skips
the completed repetitions and gives the interrupted one a new attempt,
verified by run counts; a child whose linkage write failed is recovered
without a duplicate; events replay from a cursor; and retention pruning of
the children leaves the attempt evidence readable.

The probe node and its gates follow ``test_api_sweeps.py``'s, duplicated on
purpose so the test modules stay independent.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core.auth import TOKEN_HEADER, session_token
from app.core.db import Database
from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from app.core.node_registry import registry
from app.core.run_service import QueueLimits, RunService
from app.core.run_store import RunStore
from app.core.study_coordinator import StudyCoordinator, repetition_seed
from app.core.study_store import (
    ATTEMPT_FAILED,
    ATTEMPT_SUCCEEDED,
    AttemptRecord,
    StudyStore,
    rank_study,
)
from app.main import app

BASE_URL = f"http://127.0.0.1:{settings.PORT}"


# ── the probe node ────────────────────────────────────────────────────────


class _Probe:
    """Start log, gates and scripted failures. Mutated from engine worker
    threads, hence the lock."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.started: list[str] = []
        self.gates: dict[str, tuple[threading.Event, threading.Event]] = {}
        self.fail_next: dict[str, int] = {}

    def enter(self, tag: str) -> bool:
        """Record a start; True when this run of *tag* must fail."""
        with self.lock:
            self.started.append(tag)
            remaining = self.fail_next.get(tag, 0)
            if remaining:
                self.fail_next[tag] = remaining - 1
                return True
            return False

    def hold(self, tag: str) -> threading.Event:
        """Every run of *tag* blocks until the returned event is set."""
        entered, release = threading.Event(), threading.Event()
        self.gates[tag] = (entered, release)
        return release

    async def wait_entered(self, tag: str, timeout: float = 10.0) -> None:
        entered, _release = self.gates[tag]
        if not await asyncio.to_thread(entered.wait, timeout):
            raise AssertionError(f"{tag!r} never reached its hold point")

    def count(self, tag: str) -> int:
        with self.lock:
            return self.started.count(tag)


_probe = _Probe()


class _StudyProbeNode(BaseNode):
    """Logs ``val_loss = value`` (plus a seed-dependent jitter when asked),
    a known function of its params, so a ranking test asserts exact
    numbers."""

    NODE_NAME = "_StudyProbe"
    CATEGORY = "Test"
    DESCRIPTION = "Logs a known val_loss"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="tag", param_type=ParamType.STRING,
                            default="a"),
            ParamDefinition(name="value", param_type=ParamType.FLOAT,
                            default=1.0),
            ParamDefinition(name="jitter", param_type=ParamType.BOOL,
                            default=False),
            ParamDefinition(name="api_key", param_type=ParamType.SECRET,
                            default=""),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], *,
                context: Any = None) -> dict[str, Any]:
        tag = str(params.get("tag", "a"))
        must_fail = _probe.enter(tag)
        gate = _probe.gates.get(tag)
        if gate is not None:
            entered, release = gate
            entered.set()
            if not release.wait(timeout=15.0):
                raise TimeoutError(f"probe hold for {tag!r} never released")
        value = float(params.get("value", 1.0))
        if params.get("jitter"):
            value += random.random()
        if context is not None:
            context.log_metric("val_loss", value, step=0)
            context.log_metric("train_loss", value * 2, step=0)
        if must_fail:
            # AFTER logging: a failed repetition that recorded a number
            # must still make its variant ineligible.
            raise RuntimeError("probe failed on purpose")
        return {"value": inputs.get("value")}


@pytest.fixture(autouse=True)
def _register_probe():
    registry._nodes["_StudyProbe"] = _StudyProbeNode
    yield
    registry._nodes.pop("_StudyProbe", None)


@pytest.fixture(autouse=True)
def probe():
    global _probe
    _probe = _Probe()
    return _probe


# ── fixtures ──────────────────────────────────────────────────────────────


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
async def make_env(db, store):
    """Build (service, coordinator) pairs over ONE database and wire
    app.state the way the lifespan does; tear down in the lifespan's order
    (coordinator, then service)."""
    built: list[tuple[RunService, StudyCoordinator]] = []

    def _make(*, cpu: int = 2, retention: int | None = None,
              shutdown_grace_s: float = 5.0
              ) -> tuple[RunService, StudyCoordinator]:
        service = RunService(
            store, shutdown_grace_s=shutdown_grace_s,
            retention_keep_last=retention,
            limits=QueueLimits(cpu=cpu, gpu=1, interactive=2, overrides=()))
        coordinator = StudyCoordinator(service, StudyStore(db))
        built.append((service, coordinator))
        app.state.db = db
        app.state.run_service = service
        app.state.study_coordinator = coordinator
        return service, coordinator

    try:
        yield _make
    finally:
        for service, coordinator in built:
            await coordinator.shutdown()
            await service.shutdown()
        for attribute in ("db", "run_service", "study_coordinator"):
            if hasattr(app.state, attribute):
                delattr(app.state, attribute)


@pytest.fixture
async def env(make_env):
    return make_env()


@pytest.fixture
async def client(env):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url=BASE_URL,
        headers={TOKEN_HEADER: session_token()},
    ) as http:
        yield http


# ── helpers ───────────────────────────────────────────────────────────────


def _graph(**params: Any) -> dict[str, Any]:
    """Start --trigger--> _TestSource --value--> _StudyProbe."""
    return {
        "nodes": [
            {"id": "start", "type": "Start", "data": {"params": {}}},
            {"id": "src", "type": "_TestSource",
             "data": {"params": {"val": "hi"}}},
            {"id": "probe", "type": "_StudyProbe",
             "data": {"params": params}},
        ],
        "edges": [
            {"id": "et", "source": "start", "target": "src",
             "sourceHandle": "trigger", "type": "trigger"},
            {"id": "e1", "source": "src", "target": "probe",
             "sourceHandle": "value", "targetHandle": "value"},
        ],
    }


def _ops(tag: str, value: float, **extra: Any) -> list[dict[str, Any]]:
    ops = [{"node_id": "probe", "param": "tag", "value": tag},
           {"node_id": "probe", "param": "value", "value": value}]
    ops += [{"node_id": "probe", "param": key, "value": val}
            for key, val in extra.items()]
    return ops


def _body(variants: list[tuple[str, float]] | None = None, *,
          repetitions: int = 2, **overrides: Any) -> dict[str, Any]:
    variants = variants or [("A", 1.0), ("B", 2.0)]
    body: dict[str, Any] = {
        "name": "lr study",
        "objective": {"node_id": "probe", "metric": "val_loss",
                      "direction": "minimize"},
        "base_graph": _graph(),
        "variants": [{"label": label, "ops": _ops(label, value)}
                     for label, value in variants],
        "repetitions": repetitions,
        "seed": 7,
        "options": {"device": "cpu"},
    }
    body.update(overrides)
    return body


async def _wait_status(client: AsyncClient, study_id: str, *statuses: str,
                       timeout: float = 20.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    body: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = await client.get(f"/api/studies/{study_id}")
        assert response.status_code == 200, response.text
        body = response.json()
        if body["status"] in statuses:
            return body
        await asyncio.sleep(0.02)
    raise AssertionError(f"study {study_id} never reached {statuses}; last "
                         f"status {body.get('status')!r}")


async def _study_runs(store: RunStore, study_id: str) -> list[Any]:
    def _select(conn):
        return [row[0] for row in conn.execute(
            "SELECT id FROM exec_runs WHERE study_id = ? ORDER BY created_at",
            (study_id,))]

    ids = await store.db.run(_select)
    return [await store.get_run(run_id) for run_id in ids]


def _attempts(body: dict[str, Any], variant: int,
              repetition: int) -> list[dict[str, Any]]:
    return [a for a in body["variants"][variant]["attempts"]
            if a["repetition"] == repetition]


async def _events(client: AsyncClient, study_id: str,
                  cursor: int = 0) -> list[dict[str, Any]]:
    response = await client.get(f"/api/studies/{study_id}/events",
                                params={"cursor": cursor})
    assert response.status_code == 200, response.text
    return response.json()["events"]


# ── the ranking rule (pure) ───────────────────────────────────────────────


def _attempt(variant: int, repetition: int, status: str,
             value: float | None, attempt: int = 1) -> AttemptRecord:
    return AttemptRecord(
        id=f"{variant}-{repetition}-{attempt}", study_id="s",
        variant_index=variant, repetition=repetition, attempt=attempt,
        seed=0, run_id=None, status=status, error=None, created_at="t",
        submitted_at=None, started_at=None, finished_at=None,
        objective_value=value, final_metrics=None, fingerprint_digest=None,
        harvested_at=None)


def test_exact_ties_are_labelled_independently_of_order():
    attempts = [
        _attempt(0, 0, ATTEMPT_SUCCEEDED, 1.0),
        _attempt(0, 1, ATTEMPT_SUCCEEDED, 3.0),
        _attempt(1, 0, ATTEMPT_SUCCEEDED, 3.0),
        _attempt(1, 1, ATTEMPT_SUCCEEDED, 1.0),
        _attempt(2, 0, ATTEMPT_SUCCEEDED, 5.0),
        _attempt(2, 1, ATTEMPT_SUCCEEDED, 5.0),
    ]
    forward = rank_study(3, 2, attempts, direction="minimize")
    backward = rank_study(3, 2, list(reversed(attempts)),
                          direction="minimize")
    assert forward == backward
    assert [row["rank"] for row in forward] == [1, 1, 3]
    assert forward[0]["tie"] and forward[0]["tied_with"] == [1]
    assert forward[1]["tied_with"] == [0]
    assert not forward[2]["tie"]
    assert forward[0]["mean"] == 2.0
    # Sample standard deviation (n - 1).
    assert forward[0]["std"] == pytest.approx(2 ** 0.5)
    maximize = rank_study(3, 2, attempts, direction="maximize")
    assert [row["rank"] for row in maximize] == [2, 2, 1]


def test_a_failed_repetition_makes_its_variant_ineligible():
    attempts = [
        _attempt(0, 0, ATTEMPT_SUCCEEDED, 1.0),
        _attempt(0, 1, ATTEMPT_FAILED, 0.1),       # logged, then failed
        _attempt(1, 0, ATTEMPT_SUCCEEDED, 2.0),
        _attempt(1, 1, ATTEMPT_SUCCEEDED, 2.0),
    ]
    rows = rank_study(2, 2, attempts, direction="minimize")
    assert rows[0]["eligible"] is False and rows[0]["rank"] is None
    assert rows[0]["ineligible_reasons"] == ["repetition 1 is failed"]
    # Its numbers are still reported as evidence.
    assert rows[0]["values"] == [1.0, 0.1]
    assert rows[1]["rank"] == 1


def test_only_the_current_attempt_of_a_repetition_counts():
    attempts = [
        _attempt(0, 0, "interrupted", None, attempt=1),
        _attempt(0, 0, ATTEMPT_SUCCEEDED, 4.0, attempt=2),
    ]
    rows = rank_study(1, 1, attempts, direction="minimize")
    assert rows[0]["rank"] == 1 and rows[0]["values"] == [4.0]
    assert rows[0]["std"] is None and rows[0]["count"] == 1


# ── routes ────────────────────────────────────────────────────────────────


def test_studies_routes_are_not_under_an_auth_exempt_prefix():
    from app.main import _prefix_exempt

    for path in ("/api/studies", "/api/studies/x", "/api/studies/x/cancel",
                 "/api/studies/x/resume"):
        assert not _prefix_exempt(path), path


async def test_a_mutating_study_route_without_a_token_is_403(env):
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url=BASE_URL) as http:
        assert (await http.post("/api/studies", json=_body())
                ).status_code == 403
        assert (await http.post("/api/studies/x/resume")).status_code == 403
        unknown = await http.get("/api/studies/x")
        assert unknown.status_code == 404
        assert unknown.json()["detail"] == "study 'x' not found"


async def test_submission_creates_n_by_r_attempts_and_queued_runs(
        client, store, probe):
    created = await client.post("/api/studies", json=_body(
        [("A", 1.0), ("B", 2.0), ("C", 3.0)], repetitions=2,
        budget={"max_parallel_runs": 2}))
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["created"] is True
    assert body["status"] == "running"
    assert sum(len(v["attempts"]) for v in body["variants"]) == 6
    # Repetition-major submission: the first two children are repetition 0.
    submitted = [a for v in body["variants"] for a in v["attempts"]
                 if a["status"] != "pending"]
    assert submitted and all(a["repetition"] == 0 for a in submitted)

    done = await _wait_status(client, body["study_id"], "succeeded")
    runs = await _study_runs(store, body["study_id"])
    assert len(runs) == 6
    assert all(run.status == "succeeded" for run in runs)
    by_attempt = {run.study_attempt_id: run for run in runs}
    for variant in done["variants"]:
        assert [a["attempt"] for a in variant["attempts"]] == [1, 1]
        for attempt in variant["attempts"]:
            run = by_attempt[attempt["attempt_id"]]
            assert run.id == attempt["run_id"]
            # Each repetition gets (seed + r), the same for every variant.
            assert run.options["seed"] == repetition_seed(
                7, attempt["repetition"]) == attempt["seed"]
            assert attempt["objective_value"] == float(
                variant["index"] + 1)
            assert attempt["final_metrics"]["val_loss"] == float(
                variant["index"] + 1)
            assert attempt["fingerprint_digest"] == run.fingerprint_digest
    assert [r["label"] for r in done["rankings"]] == ["A", "B", "C"]
    assert done["counts"]["succeeded"] == 6
    assert probe.count("A") == 2


async def test_an_idempotent_retry_returns_the_same_study_and_creates_nothing(
        client, store):
    body = _body()
    first = await client.post("/api/studies", json=body,
                              headers={"Idempotency-Key": "k-1"})
    assert first.status_code == 201
    second = await client.post("/api/studies", json=body,
                               headers={"Idempotency-Key": "k-1"})
    assert second.status_code == 200
    assert second.json()["created"] is False
    study_id = first.json()["study_id"]
    assert second.json()["study_id"] == study_id
    # The body field is the same key.
    third = await client.post("/api/studies",
                              json={**body, "idempotency_key": "k-1"})
    assert third.status_code == 200 and third.json()["study_id"] == study_id
    await _wait_status(client, study_id, "succeeded")
    listed = (await client.get("/api/studies")).json()
    assert listed["total"] == 1
    assert len(await _study_runs(store, study_id)) == 4


async def test_an_idempotency_key_reused_for_another_payload_is_409(client):
    first = await client.post("/api/studies", json=_body(),
                              headers={"Idempotency-Key": "k-2"})
    assert first.status_code == 201
    other = await client.post("/api/studies", json=_body(repetitions=3),
                              headers={"Idempotency-Key": "k-2"})
    assert other.status_code == 409
    assert first.json()["study_id"] in other.json()["detail"]
    mismatch = await client.post(
        "/api/studies", json={**_body(), "idempotency_key": "x"},
        headers={"Idempotency-Key": "y"})
    assert mismatch.status_code == 400


async def test_completion_ranks_with_an_exact_tie_labelled_as_a_tie(client):
    created = await client.post("/api/studies", json=_body(
        [("A", 1.0), ("B", 1.0), ("C", 0.5)], repetitions=2))
    done = await _wait_status(client, created.json()["study_id"], "succeeded")
    ranks = {row["label"]: row for row in done["rankings"]}
    assert ranks["C"]["rank"] == 1 and not ranks["C"]["tie"]
    assert ranks["A"]["rank"] == ranks["B"]["rank"] == 2
    assert ranks["A"]["tie"] and ranks["A"]["tied_with"] == [1]
    assert ranks["B"]["tie"] and ranks["B"]["tied_with"] == [0]
    a = done["variants"][0]["aggregate"]
    assert a == {"count": 2, "mean": 1.0, "std": 0.0, "values": [1.0, 1.0]}
    finished = [e for e in await _events(client, done["study_id"])
                if e["type"] == "study_finished"]
    assert len(finished) == 1
    assert {row["variant_index"]: row["rank"]
            for row in finished[0]["payload"]["ranking"]} == {0: 2, 1: 2,
                                                              2: 1}


async def test_seed_dependent_values_aggregate_per_repetition(client):
    body = _body(repetitions=3)
    body["variants"] = [{"label": "J", "ops": _ops("J", 1.0, jitter=True)}]
    created = await client.post("/api/studies", json=body)
    done = await _wait_status(client, created.json()["study_id"], "succeeded")
    aggregate = done["variants"][0]["aggregate"]
    assert aggregate["count"] == 3
    assert len(set(aggregate["values"])) == 3       # distinct seeds
    assert aggregate["std"] > 0
    assert done["rankings"][0]["rank"] == 1


async def test_a_variant_with_one_failed_repetition_is_ineligible(
        client, probe):
    probe.fail_next["B"] = 1                  # B's repetition 0 fails
    created = await client.post("/api/studies", json=_body(
        [("A", 2.0), ("B", 1.0)], repetitions=2))
    done = await _wait_status(client, created.json()["study_id"], "succeeded")
    b = done["variants"][1]
    assert b["eligible"] is False and b["rank"] is None
    assert b["ineligible_reasons"] == ["repetition 0 is failed"]
    # The failed repetition DID log 1.0 and would have won the Sweep v1
    # table; it is evidence here, never a rank.
    assert b["aggregate"]["values"] == [1.0, 1.0]
    assert [r["label"] for r in done["rankings"]] == ["A"]
    # A failed repetition is a result, not a retry.
    assert [a["attempt"] for a in b["attempts"]] == [1, 1]


async def test_cancel_at_a_boundary_keeps_completed_attempts_then_resume(
        client, store, probe):
    release = probe.hold("B")
    created = await client.post("/api/studies", json=_body())
    study_id = created.json()["study_id"]
    await probe.wait_entered("B")            # A r0 done, B r0 running

    cancelled = await client.post(f"/api/studies/{study_id}/cancel")
    assert cancelled.status_code == 200, cancelled.text
    body = cancelled.json()
    assert body["status"] == "cancel_requested"
    assert len(body["cancelled_runs"]) == 1
    release.set()
    done = await _wait_status(client, study_id, "cancelled")
    assert _attempts(done, 0, 0)[0]["status"] == "succeeded"
    assert _attempts(done, 1, 0)[0]["status"] == "cancelled"
    assert _attempts(done, 0, 1)[0]["status"] == "pending"
    assert _attempts(done, 1, 1)[0]["status"] == "pending"
    assert len(await _study_runs(store, study_id)) == 2
    # Cancelling again is a no-op.
    again = await client.post(f"/api/studies/{study_id}/cancel")
    assert again.status_code == 200 and again.json()["status"] == "cancelled"

    probe.gates.pop("B")
    resumed = await client.post(f"/api/studies/{study_id}/resume")
    assert resumed.status_code == 200, resumed.text
    final = await _wait_status(client, study_id, "succeeded")
    # Completed A r0 was not re-run; B r0 got attempt 2.
    assert probe.count("A") == 2
    assert [a["status"] for a in _attempts(final, 1, 0)] == [
        "cancelled", "succeeded"]
    assert [a["attempt"] for a in _attempts(final, 0, 0)] == [1]
    assert len(await _study_runs(store, study_id)) == 5
    types = [e["type"] for e in await _events(client, study_id)]
    for expected in ("study_cancel_requested", "study_cancelled",
                     "study_resumed", "study_finished"):
        assert expected in types
    resume = next(e for e in await _events(client, study_id)
                  if e["type"] == "study_resumed")
    assert resume["payload"]["completed_repetitions_kept"] == 1
    assert len(resume["payload"]["new_attempts"]) == 1
    # A succeeded study cannot be cancelled or resumed.
    assert (await client.post(f"/api/studies/{study_id}/cancel")
            ).status_code == 409
    assert (await client.post(f"/api/studies/{study_id}/resume")
            ).status_code == 409


async def _crash(service: RunService, coordinator: StudyCoordinator) -> None:
    """A restart without a graceful run finish: the coordinator stops, the
    service hard-cancels its stuck run task (its row stays `running`)."""
    await coordinator.shutdown()
    await service.shutdown()


@pytest.mark.parametrize("policy", ["explicit", "auto"])
async def test_resume_after_a_restart_skips_completed_repetitions(
        make_env, store, probe, policy):
    service, coordinator = make_env(shutdown_grace_s=0.1)
    release = probe.hold("B")
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url=BASE_URL,
                           headers={TOKEN_HEADER: session_token()}) as http:
        created = await http.post("/api/studies", json=_body(
            resume_policy=policy))
        study_id = created.json()["study_id"]
        await probe.wait_entered("B")
        await _crash(service, coordinator)
        release.set()
        probe.gates.pop("B")
        stuck = [run for run in await _study_runs(store, study_id)
                 if run.status == "running"]
        assert len(stuck) == 1                   # B r0, abandoned mid-run

        # The next boot: a NEW service and coordinator over the same db,
        # in the lifespan's order.
        service2, coordinator2 = make_env()
        assert await service2.recover_interrupted() == 1
        assert await coordinator2.recover() == 1
        if policy == "explicit":
            waiting = (await http.get(f"/api/studies/{study_id}")).json()
            assert waiting["status"] == "interrupted"
            assert _attempts(waiting, 0, 0)[0]["status"] == "succeeded"
            assert _attempts(waiting, 1, 0)[0]["status"] == "interrupted"
            # It waits: nothing new was submitted.
            assert len(await _study_runs(store, study_id)) == 2
            assert (await http.post(f"/api/studies/{study_id}/resume")
                    ).status_code == 200
        final = await _wait_status(http, study_id, "succeeded")

    runs = await _study_runs(store, study_id)
    # A r0 + B r0 (interrupted) + B r0 attempt 2 + A r1 + B r1.
    assert len(runs) == 5
    assert sorted(run.status for run in runs).count("interrupted") == 1
    assert probe.count("A") == 2
    assert [a["status"] for a in _attempts(final, 1, 0)] == [
        "interrupted", "succeeded"]
    assert [a["attempt"] for a in _attempts(final, 0, 0)] == [1]
    events = [e.type for e in await coordinator2.store.get_events(study_id)]
    assert "study_interrupted" in events and "study_resumed" in events


async def test_a_child_whose_linkage_write_failed_is_relinked_not_resubmitted(
        client, store, probe, env, monkeypatch):
    _service, coordinator = env
    original = StudyStore.confirm_submission
    failures = {"left": 1}

    async def flaky(self, attempt_id, status):
        if failures["left"]:
            failures["left"] -= 1
            raise RuntimeError("injected: linkage write lost")
        return await original(self, attempt_id, status)

    monkeypatch.setattr(StudyStore, "confirm_submission", flaky)
    release = probe.hold("A")
    created = await client.post("/api/studies", json=_body(
        [("A", 1.0)], repetitions=2))
    study_id = created.json()["study_id"]
    await probe.wait_entered("A")
    orphan = created.json()["variants"][0]["attempts"][0]
    assert orphan["status"] == "submitting"           # linkage not confirmed

    # The next tick (a GET) finds the child by its stamp.
    seen = (await client.get(f"/api/studies/{study_id}")).json()
    relinked = _attempts(seen, 0, 0)[0]
    assert relinked["status"] in ("queued", "running")
    assert relinked["run_id"] == orphan["run_id"]
    release.set()
    probe.gates.pop("A")
    final = await _wait_status(client, study_id, "succeeded")
    runs = await _study_runs(store, study_id)
    assert len(runs) == 2                            # never resubmitted
    assert [a["attempt"] for a in final["variants"][0]["attempts"]] == [1, 1]
    types = [e["type"] for e in await _events(client, study_id)]
    assert types.count("attempt_relinked") == 1


async def test_a_lost_linkage_is_recovered_across_a_restart(
        make_env, store, probe, monkeypatch):
    """The issue's acceptance case end to end: the child was submitted, its
    linkage write failed, and the server died before anything noticed. The
    next boot finds the child by its stamp, so the repetition is retried
    once (attempt 2) and never runs twice alongside its orphan."""
    service, coordinator = make_env(shutdown_grace_s=0.1)
    original = StudyStore.confirm_submission
    failures = {"left": 1}

    async def flaky(self, attempt_id, status):
        if failures["left"]:
            failures["left"] -= 1
            raise RuntimeError("injected: linkage write lost")
        return await original(self, attempt_id, status)

    monkeypatch.setattr(StudyStore, "confirm_submission", flaky)
    release = probe.hold("A")
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url=BASE_URL,
                           headers={TOKEN_HEADER: session_token()}) as http:
        created = await http.post("/api/studies", json=_body(
            [("A", 1.0)], repetitions=2))
        study_id = created.json()["study_id"]
        orphan = created.json()["variants"][0]["attempts"][0]
        assert orphan["status"] == "submitting"
        await probe.wait_entered("A")
        await _crash(service, coordinator)
        release.set()
        probe.gates.pop("A")

        service2, coordinator2 = make_env()
        await service2.recover_interrupted()
        await coordinator2.recover()
        waiting = (await http.get(f"/api/studies/{study_id}")).json()
        assert waiting["status"] == "interrupted"
        first = _attempts(waiting, 0, 0)[0]
        assert first["status"] == "interrupted"
        assert first["run_id"] == orphan["run_id"]
        assert (await http.post(f"/api/studies/{study_id}/resume")
                ).status_code == 200
        final = await _wait_status(http, study_id, "succeeded")
    runs = await _study_runs(store, study_id)
    assert len(runs) == 3          # the orphan, its retry, repetition 1
    assert [a["attempt"] for a in _attempts(final, 0, 0)] == [1, 2]
    assert [a["attempt"] for a in _attempts(final, 0, 1)] == [1]
    types = [e.type for e in await coordinator2.store.get_events(study_id)]
    assert types.count("attempt_relinked") == 1


async def test_a_submission_that_created_no_run_is_lost_and_retried(
        client, store, env, monkeypatch):
    service, _coordinator = env
    original = RunService.submit
    failures = {"left": 1}

    async def broken(self, *args, **kwargs):
        if failures["left"]:
            failures["left"] -= 1
            raise RuntimeError("injected: died before the row was written")
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(RunService, "submit", broken)
    created = await client.post("/api/studies", json=_body(
        [("A", 1.0)], repetitions=1))
    study_id = created.json()["study_id"]
    final = await _wait_status(client, study_id, "succeeded")
    attempts = final["variants"][0]["attempts"]
    assert [a["status"] for a in attempts] == ["lost", "succeeded"]
    assert len(await _study_runs(store, study_id)) == 1
    lost = [e for e in await _events(client, study_id)
            if e["type"] == "attempt_lost"]
    assert len(lost) == 1 and "stamp" in lost[0]["payload"]["error"]


async def test_the_run_budget_fails_a_study_that_would_exceed_it(
        client, store, monkeypatch):
    original = RunService.submit
    failures = {"left": 1}

    async def broken(self, *args, **kwargs):
        if failures["left"]:
            failures["left"] -= 1
            raise RuntimeError("injected")
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(RunService, "submit", broken)
    created = await client.post("/api/studies", json=_body(
        [("A", 1.0)], repetitions=1, budget={"max_runs": 1}))
    final = await _wait_status(client, created.json()["study_id"], "failed")
    assert "budget.max_runs (1)" in final["error"]
    assert len(await _study_runs(store, final["study_id"])) == 0


async def test_a_child_cancelled_outside_the_study_interrupts_it(
        client, probe):
    release = probe.hold("A")
    created = await client.post("/api/studies", json=_body(
        [("A", 1.0)], repetitions=2))
    study_id = created.json()["study_id"]
    await probe.wait_entered("A")
    run_id = created.json()["variants"][0]["attempts"][0]["run_id"]
    assert (await client.post(f"/api/runs/{run_id}/cancel")
            ).status_code == 200
    release.set()
    probe.gates.pop("A")
    interrupted = await _wait_status(client, study_id, "interrupted")
    assert _attempts(interrupted, 0, 0)[0]["status"] == "cancelled"
    await client.post(f"/api/studies/{study_id}/resume")
    await _wait_status(client, study_id, "succeeded")


async def test_events_replay_after_a_cursor(client):
    created = await client.post("/api/studies", json=_body())
    study_id = created.json()["study_id"]
    await _wait_status(client, study_id, "succeeded")
    everything = await _events(client, study_id)
    assert [e["cursor"] for e in everything] == list(
        range(1, len(everything) + 1))
    assert everything[0]["type"] == "study_created"
    assert everything[-1]["type"] == "study_finished"
    revisions = [e["payload"]["revision"] for e in everything]
    assert revisions == sorted(revisions)
    tail = await _events(client, study_id, cursor=3)
    assert tail == everything[3:]
    # A long poll on a settled study answers at once.
    started = time.monotonic()
    empty = await client.get(f"/api/studies/{study_id}/events",
                             params={"cursor": everything[-1]["cursor"],
                                     "wait": 5})
    assert empty.json()["events"] == []
    assert empty.json()["cursor"] == everything[-1]["cursor"]
    assert time.monotonic() - started < 2.0


async def test_a_long_poll_wakes_when_the_study_moves(client, probe):
    release = probe.hold("A")
    created = await client.post("/api/studies", json=_body(
        [("A", 1.0)], repetitions=1))
    study_id = created.json()["study_id"]
    await probe.wait_entered("A")
    cursor = (await client.get(f"/api/studies/{study_id}")).json()[
        "last_cursor"]
    poll = asyncio.ensure_future(client.get(
        f"/api/studies/{study_id}/events",
        params={"cursor": cursor, "wait": 10}))
    await asyncio.sleep(0.1)
    release.set()
    page = (await poll).json()
    assert page["events"] and page["events"][0]["cursor"] == cursor + 1


@pytest.mark.parametrize("listener", [True, False])
async def test_retention_pruning_leaves_the_attempt_evidence_readable(
        make_env, store, listener):
    service, coordinator = make_env(retention=0)
    if not listener:
        # Retention's own backstop copies the evidence instead.
        service.remove_terminal_listener(coordinator._on_terminal)
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url=BASE_URL,
                           headers={TOKEN_HEADER: session_token()}) as http:
        created = await http.post("/api/studies", json=_body())
        final = await _wait_status(http, created.json()["study_id"],
                                   "succeeded")
    for variant in final["variants"]:
        for attempt in variant["attempts"]:
            assert attempt["run_id"]
            assert await store.get_run(attempt["run_id"]) is None
            assert attempt["status"] == "succeeded"
            assert attempt["objective_value"] == float(variant["index"] + 1)
            assert attempt["final_metrics"]["train_loss"] == 2 * float(
                variant["index"] + 1)
            assert attempt["harvested_at"] and attempt["finished_at"]
            assert attempt["fingerprint_digest"]
    assert [r["label"] for r in final["rankings"]] == ["A", "B"]


async def test_secret_params_never_reach_the_study_rows(client, db):
    body = _body([("A", 1.0)], repetitions=1,
                 base_graph=_graph(api_key="sk-live-123"))
    created = await client.post("/api/studies", json=body)
    await _wait_status(client, created.json()["study_id"], "succeeded")

    def _dump(conn):
        rows = []
        for table in ("studies", "study_variants", "study_events",
                      "study_attempts"):
            rows += [tuple(row) for row in conn.execute(
                f"SELECT * FROM {table}")]
        return repr(rows)

    assert "sk-live-123" not in await db.run(_dump)


async def test_listing_lets_any_client_reattach(client):
    created = await client.post("/api/studies", json=_body())
    study_id = created.json()["study_id"]
    await _wait_status(client, study_id, "succeeded")
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url=BASE_URL) as other:      # no token
        listed = (await other.get("/api/studies",
                                  params={"status": "succeeded"})).json()
        assert [s["study_id"] for s in listed["studies"]] == [study_id]
        assert listed["studies"][0]["counts"] == {"succeeded": 4}
        running = (await other.get("/api/studies", params={
            "status": "queued|running|cancel_requested"})).json()
        assert running["total"] == 0
        assert (await other.get("/api/studies", params={"status": "nope"})
                ).status_code == 400
        detail = (await other.get(f"/api/studies/{study_id}")).json()
        assert detail["status"] == "succeeded"


@pytest.mark.parametrize("mutate, fragment", [
    (lambda b: b.pop("base_graph"), "no base_graph"),
    (lambda b: b["variants"][0]["ops"].append(
        {"node_id": "ghost", "param": "x", "value": 1}),
     "variants[0].ops[2]: no node with id 'ghost'"),
    (lambda b: b["variants"][0]["ops"].append(
        {"node_id": "probe", "param": "api_key", "value": "k"}),
     "secret parameter"),
    (lambda b: b["variants"].append(dict(b["variants"][0])),
     "labels must be unique"),
    (lambda b: b.update(options={"seed": 3}), "options.seed is owned"),
    (lambda b: b.update(options={"lane": "interactive"}),
     "interactive lane"),
    (lambda b: b.update(budget={"max_runs": 3}), "below the 4 runs"),
    (lambda b: b.update(options={"device": "tpu"}), "unknown device"),
])
async def test_invalid_submissions_are_400_and_write_nothing(
        client, mutate, fragment):
    body = _body()
    mutate(body)
    response = await client.post("/api/studies", json=body)
    assert response.status_code == 400, response.text
    assert fragment in response.json()["detail"]
    assert (await client.get("/api/studies")).json()["total"] == 0


async def test_the_objective_needs_an_exact_producer(client):
    body = _body()
    body["objective"].pop("node_id")
    assert (await client.post("/api/studies", json=body)).status_code == 422
    body = _body()
    body["variants"][0] = {"label": "A", "ops": [], "graph": _graph()}
    assert (await client.post("/api/studies", json=body)).status_code == 422
