"""Server-enforced study budgets and the audit's recovery cases (#343).

Increment 4 of the issue: ``max_run_seconds`` stops one child that runs too
long (it ends ``timed_out``, a result), ``max_active_seconds`` and
``max_runs`` halt the study with a durable ``halt`` record and every partial
result kept, and a resume with a raised budget continues it. The budget
timer enforces the time limits with NOBODY reading the study: these tests
watch the event log through the store, which does not tick the study the
way ``GET /api/studies/{id}`` does.

The second half covers the 2026-10-08 audit's recovery acceptance: a retry
with the same idempotency key across a restart and concurrently, a failure
between a submit and its linkage combined with a restart that finds
completed, interrupted and orphaned repetitions at once, and
``expected_revision`` races.

Fixtures and the probe node come from ``test_studies``.
"""

# The fixtures imported from test_studies are re-bound as test arguments.
# ruff: noqa: F811

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.auth import TOKEN_HEADER, session_token
from app.core.study_store import (
    ATTEMPT_FINAL,
    StudyStore,
    attempt_outcome,
    budget_usage,
    scoped_idempotency_key,
    unscoped_idempotency_key,
)
from app.main import app

from .test_studies import (  # noqa: F401  (fixtures)
    BASE_URL,
    _attempt,
    _attempts,
    _body,
    _crash,
    _events,
    _register_probe,
    _study_runs,
    _wait_status,
    client,
    db,
    env,
    make_env,
    probe,
    store,
)


async def _wait_event(coordinator: Any, study_id: str, event_type: str,
                      timeout: float = 15.0) -> dict[str, Any]:
    """The first *event_type* event, read from the store (no tick)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for event in await coordinator.store.get_events(study_id):
            if event.type == event_type:
                return {"type": event.type, "payload": event.payload}
        await asyncio.sleep(0.02)
    raise AssertionError(f"no {event_type} event in study {study_id}")


async def _wait_stored_status(coordinator: Any, study_id: str, *statuses: str,
                              timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        study = await coordinator.store.get_study(study_id)
        if study is not None and study.status in statuses:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"study {study_id} never reached {statuses}")


# ── pure accounting ───────────────────────────────────────────────────────


def _timed(variant: int, repetition: int, status: str, start: str | None,
           end: str | None, attempt: int = 1) -> Any:
    record = _attempt(variant, repetition, status, None, attempt)
    return record.__class__(**{**record.__dict__, "started_at": start,
                               "finished_at": end})


def test_budget_usage_counts_runs_and_seconds_and_names_the_exhausted_one():
    t0 = "2026-10-21T00:00:00.000000Z"
    t10 = "2026-10-21T00:00:10.000000Z"
    now = 1792540820.0     # 2026-10-21T00:00:20Z
    attempts = [
        _timed(0, 0, "succeeded", t0, t10),            # 10 s
        _timed(1, 0, "running", t10, None),            # 10 s so far
        _timed(0, 1, "pending", None, None),
        _timed(1, 1, "lost", None, None),              # submitted, no run
        _timed(1, 1, "pending", None, None, attempt=2),
    ]
    usage = budget_usage({"max_parallel_runs": 2}, attempts, now=now)
    assert usage["runs"] == {"used": 3, "limit": None}
    assert usage["parallel"] == {"active": 1, "limit": 2}
    assert usage["active_seconds"]["used"] == pytest.approx(20.0)
    assert usage["run_seconds"]["longest_active"] == pytest.approx(10.0)
    assert usage["unfinished_repetitions"] == 3
    assert usage["exhausted"] is None
    assert budget_usage({"max_active_seconds": 20}, attempts,
                        now=now)["exhausted"] == "max_active_seconds"
    assert budget_usage({"max_runs": 3}, attempts,
                        now=now)["exhausted"] == "max_runs"
    # Every run submitted, one repetition interrupted: a resume would need
    # one more run, so max_runs is what stops it (the panel's budget form
    # and the resume 409 agree on that).
    interrupted = [_timed(0, 0, "succeeded", t0, t10),
                   _timed(0, 1, "interrupted", t0, t10)]
    assert budget_usage({"max_runs": 2}, interrupted,
                        now=now)["exhausted"] == "max_runs"
    # Nothing left to run: a used-up budget halts nothing.
    done = [_timed(0, 0, "succeeded", t0, t10)]
    assert budget_usage({"max_runs": 1, "max_active_seconds": 1}, done,
                        now=now)["exhausted"] is None


def test_attempt_outcome_and_key_scoping():
    assert attempt_outcome("cancelled", "max_run_seconds") == "timed_out"
    assert attempt_outcome("interrupted", "max_run_seconds") == "timed_out"
    assert attempt_outcome("succeeded", "max_run_seconds") == "succeeded"
    assert attempt_outcome("failed", "max_run_seconds") == "failed"
    assert attempt_outcome("cancelled", "max_active_seconds") == "cancelled"
    assert attempt_outcome("cancelled", None) == "cancelled"
    assert "timed_out" in ATTEMPT_FINAL
    assert scoped_idempotency_key(None, "k") == "k"
    assert scoped_idempotency_key("copilot", "k") == "plugin:copilot:k"
    assert scoped_idempotency_key("copilot", None) is None
    assert unscoped_idempotency_key("copilot", "plugin:copilot:k") == "k"
    assert unscoped_idempotency_key(None, "k") == "k"
    assert unscoped_idempotency_key("copilot", "other") == "other"


def test_migration_018_upgrades_a_v16_database(tmp_path, monkeypatch):
    """A study written before 018 reads back as a core study with no halt,
    and its attempts with no stop reason."""
    from app.core import db as dbmod
    from app.core.db import Database
    from app.core.migrations import MIGRATIONS

    monkeypatch.setattr(dbmod, "MIGRATIONS", list(MIGRATIONS[:16]))
    old = Database(tmp_path / "v16.db")
    old.connect()
    stamp = "2026-10-01T00:00:00.000000Z"
    old._conn.execute(
        "INSERT INTO studies (id, status, revision, objective, repetitions, "
        "seed, options, budget, resume_policy, request_digest, created_at, "
        "updated_at) VALUES ('s1', 'succeeded', 3, '{\"node_id\": \"n\", "
        "\"metric\": \"m\", \"direction\": \"minimize\"}', 1, 0, '{}', "
        "'{\"max_runs\": null, \"max_parallel_runs\": 1}', 'explicit', 'd', "
        "?, ?)", (stamp, stamp))
    old._conn.execute(
        "INSERT INTO study_attempts (id, study_id, variant_index, "
        "repetition, attempt, seed, status, created_at) VALUES "
        "('a1', 's1', 0, 0, 1, 0, 'succeeded', ?)", (stamp,))
    old.close()

    monkeypatch.setattr(dbmod, "MIGRATIONS", list(MIGRATIONS))
    new = Database(tmp_path / "v16.db")
    new.connect()
    try:
        assert new._conn.execute("PRAGMA user_version").fetchone()[0] == 18
        store_ = StudyStore(new)
        study = asyncio.run(store_.get_study("s1"))
        assert (study.owner, study.halt) == (None, None)
        assert study.budget == {"max_runs": None, "max_parallel_runs": 1}
        attempt = asyncio.run(store_.list_attempts("s1"))[0]
        assert attempt.stop_reason is None
        usage = budget_usage(study.budget, [attempt], now=0.0)
        assert usage["exhausted"] is None
        assert usage["run_seconds"]["limit"] is None
    finally:
        new.close()


# ── max_run_seconds ───────────────────────────────────────────────────────


async def test_a_child_past_max_run_seconds_times_out_and_the_study_goes_on(
        client, store, env, probe):
    _service, coordinator = env
    release = probe.hold("B")
    created = await client.post("/api/studies", json=_body(
        repetitions=1, budget={"max_run_seconds": 0.4}))
    assert created.status_code == 201, created.text
    study_id = created.json()["study_id"]
    await probe.wait_entered("B")
    # Nobody reads the study: the budget timer alone stops the child.
    stop = await _wait_event(coordinator, study_id, "attempt_stop_requested")
    assert stop["payload"]["reason"] == "max_run_seconds"
    release.set()
    await _wait_stored_status(coordinator, study_id, "succeeded")

    final = (await client.get(f"/api/studies/{study_id}")).json()
    timed_out = _attempts(final, 1, 0)[0]
    assert timed_out["status"] == "timed_out"
    assert timed_out["stop_reason"] == "max_run_seconds"
    assert final["counts"]["timed_out"] == 1
    assert final["variants"][1]["eligible"] is False
    assert "repetition 0 is timed_out" in final["variants"][1][
        "ineligible_reasons"]
    assert [r["label"] for r in final["rankings"]] == ["A"]
    # A timed-out repetition is a result: it is never re-run.
    assert len(await _study_runs(store, study_id)) == 2
    child = await store.get_run(timed_out["run_id"])
    assert child.status == "cancelled"
    finished = [e for e in await _events(client, study_id)
                if e["type"] == "attempt_finished"
                and e["payload"]["status"] == "timed_out"]
    assert finished[0]["payload"]["run_status"] == "cancelled"
    assert finished[0]["payload"]["stop_reason"] == "max_run_seconds"


# ── max_active_seconds ────────────────────────────────────────────────────


async def test_max_active_seconds_halts_with_partial_results_then_resumes(
        client, store, env, probe):
    _service, coordinator = env
    release = probe.hold("B")
    created = await client.post("/api/studies", json=_body(
        repetitions=2, budget={"max_active_seconds": 0.6}))
    study_id = created.json()["study_id"]
    await probe.wait_entered("B")          # A r0 done, B r0 running
    stop = await _wait_event(coordinator, study_id, "attempt_stop_requested")
    assert stop["payload"]["reason"] == "max_active_seconds"
    release.set()
    await _wait_stored_status(coordinator, study_id, "halted")

    halted = (await client.get(f"/api/studies/{study_id}")).json()
    assert halted["status"] == "halted"
    assert halted["halt"]["budget"] == "max_active_seconds"
    assert halted["halt"]["limit"] == 0.6
    assert halted["halt"]["used"] >= 0.6
    assert halted["halt"]["unfinished_repetitions"] == 3
    assert halted["budget_usage"]["exhausted"] == "max_active_seconds"
    # The partial results are all there.
    assert _attempts(halted, 0, 0)[0]["status"] == "succeeded"
    stopped = _attempts(halted, 1, 0)[0]
    assert stopped["status"] == "cancelled"
    assert stopped["stop_reason"] == "max_active_seconds"
    assert _attempts(halted, 0, 1)[0]["status"] == "pending"
    assert halted["variants"][0]["aggregate"]["count"] == 1
    assert halted["variants"][0]["aggregate"]["values"] == [1.0, None]
    assert halted["rankings"] == []          # nothing complete yet
    event = (await _events(client, study_id))[-1]
    assert event["type"] == "study_halted"
    assert event["payload"]["completed_repetitions_kept"] == 1
    assert len(event["payload"]["ranking"]) == 2
    assert len(await _study_runs(store, study_id)) == 2
    # Halted is not active: a listing of in-flight studies leaves it out.
    active = (await client.get("/api/studies", params={
        "status": "queued|running|cancel_requested"})).json()
    assert active["total"] == 0
    listed = (await client.get("/api/studies",
                               params={"status": "halted"})).json()
    assert listed["studies"][0]["halt"]["budget"] == "max_active_seconds"

    # Resuming with the same budget would halt again at once: 409.
    refused = await client.post(f"/api/studies/{study_id}/resume")
    assert refused.status_code == 409
    assert "max_active_seconds" in refused.json()["detail"]
    probe.gates.pop("B")
    resumed = await client.post(f"/api/studies/{study_id}/resume", json={
        "budget": {"max_active_seconds": 120}})
    assert resumed.status_code == 200, resumed.text
    final = await _wait_status(client, study_id, "succeeded")
    # B r0 got attempt 2; A r0 was not re-run.
    assert [a["status"] for a in _attempts(final, 1, 0)] == [
        "cancelled", "succeeded"]
    assert [a["attempt"] for a in _attempts(final, 0, 0)] == [1]
    assert probe.count("A") == 2
    assert len(await _study_runs(store, study_id)) == 5
    assert final["budget"]["max_active_seconds"] == 120.0
    assert final["budget_usage"]["active_seconds"]["limit"] == 120.0


async def test_a_halted_study_stays_halted_across_a_restart(
        make_env, store, probe):
    service, coordinator = make_env()
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url=BASE_URL,
                           headers={TOKEN_HEADER: session_token()}) as http:
        release = probe.hold("A")
        created = await http.post("/api/studies", json=_body(
            [("A", 1.0)], repetitions=2,
            budget={"max_active_seconds": 0.3}))
        study_id = created.json()["study_id"]
        await probe.wait_entered("A")
        await _wait_event(coordinator, study_id, "attempt_stop_requested")
        release.set()
        await _wait_stored_status(coordinator, study_id, "halted")
        await _crash(service, coordinator)

        service2, coordinator2 = make_env()
        await service2.recover_interrupted()
        assert await coordinator2.recover() == 0
        after = (await http.get(f"/api/studies/{study_id}")).json()
        assert after["status"] == "halted"
        assert after["halt"]["budget"] == "max_active_seconds"
        assert len(await _study_runs(store, study_id)) == 1


async def test_cancelling_a_halted_study_cancels_it(client, env, probe):
    _service, coordinator = env
    release = probe.hold("A")
    created = await client.post("/api/studies", json=_body(
        [("A", 1.0)], repetitions=2, budget={"max_active_seconds": 0.3}))
    study_id = created.json()["study_id"]
    await probe.wait_entered("A")
    await _wait_event(coordinator, study_id, "attempt_stop_requested")
    release.set()
    await _wait_stored_status(coordinator, study_id, "halted")
    cancelled = await client.post(f"/api/studies/{study_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["halt"] is None


@pytest.mark.parametrize("budget, fragment", [
    ({"max_run_seconds": 0}, "greater than 0"),
    ({"max_run_seconds": -1}, "greater than 0"),
    ({"max_active_seconds": "10"}, "max_active_seconds"),
    ({"max_active_seconds": 31 * 24 * 3600}, "less than or equal"),
    ({"deadline": 3}, "Extra inputs"),
])
async def test_invalid_time_budgets_are_422(client, budget, fragment):
    response = await client.post("/api/studies", json=_body(budget=budget))
    assert response.status_code == 422, response.text
    assert fragment in response.text


async def test_a_resume_budget_below_the_schedule_is_409(client, probe):
    release = probe.hold("A")
    created = await client.post("/api/studies", json=_body(
        [("A", 1.0)], repetitions=2))
    study_id = created.json()["study_id"]
    await probe.wait_entered("A")
    await client.post(f"/api/studies/{study_id}/cancel")
    release.set()
    await _wait_status(client, study_id, "cancelled")
    refused = await client.post(f"/api/studies/{study_id}/resume", json={
        "budget": {"max_runs": 1}})
    assert refused.status_code == 409
    assert "below the 2 runs" in refused.json()["detail"]


async def test_unset_time_budgets_keep_the_idempotency_identity(client):
    """A retry that spells the new budget keys as null (or a client from
    before they existed, which leaves them out) is the same request."""
    body = _body()
    first = await client.post("/api/studies", json=body,
                              headers={"Idempotency-Key": "same"})
    spelled = {**body, "budget": {"max_parallel_runs": 1,
                                  "max_run_seconds": None,
                                  "max_active_seconds": None}}
    second = await client.post("/api/studies", json=spelled,
                               headers={"Idempotency-Key": "same"})
    assert second.status_code == 200, second.text
    assert second.json()["study_id"] == first.json()["study_id"]
    other = await client.post("/api/studies", json={
        **body, "budget": {"max_run_seconds": 5}},
        headers={"Idempotency-Key": "same"})
    assert other.status_code == 409


async def test_a_plugin_key_prefix_is_reserved_on_the_core_route(client):
    response = await client.post("/api/studies", json=_body(),
                                 headers={"Idempotency-Key": "plugin:x:k"})
    assert response.status_code == 400
    assert "reserved" in response.json()["detail"]


async def test_an_auto_resume_with_a_used_up_budget_halts_and_startup_goes_on(
        make_env, store, probe):
    """resume_policy auto, every run of max_runs already submitted, and a
    restart interrupts two of them: recover() must not raise (it runs at
    startup); the study halts on max_runs and a raised budget resumes it."""
    service, coordinator = make_env(shutdown_grace_s=0.1)
    release = probe.hold("B")
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url=BASE_URL,
                           headers={TOKEN_HEADER: session_token()}) as http:
        created = await http.post("/api/studies", json=_body(
            resume_policy="auto",
            budget={"max_runs": 4, "max_parallel_runs": 4}))
        study_id = created.json()["study_id"]
        await probe.wait_entered("B")

        async def all_submitted() -> bool:
            attempts = await coordinator.store.list_attempts(study_id)
            return all(a.status != "pending" for a in attempts)

        deadline = time.monotonic() + 10
        while not await all_submitted():
            assert time.monotonic() < deadline
            await asyncio.sleep(0.02)
        await _crash(service, coordinator)
        release.set()
        probe.gates.pop("B")

        service2, coordinator2 = make_env()
        await service2.recover_interrupted()
        await coordinator2.recover()              # does not raise
        halted = (await http.get(f"/api/studies/{study_id}")).json()
        assert halted["status"] == "halted", halted["status"]
        assert halted["halt"]["budget"] == "max_runs"
        assert halted["halt"]["used"] == 4
        before = len(await _study_runs(store, study_id))
        resumed = await http.post(f"/api/studies/{study_id}/resume", json={
            "budget": {"max_runs": 10, "max_parallel_runs": 4}})
        assert resumed.status_code == 200, resumed.text
        final = await _wait_status(http, study_id, "succeeded")
    retried = sum(1 for v in final["variants"] for a in v["attempts"]
                  if a["attempt"] == 2)
    assert len(await _study_runs(store, study_id)) == before + retried
    assert retried >= 1


async def test_a_budget_stop_that_failed_to_send_is_sent_again(
        client, env, probe, monkeypatch):
    service, coordinator = env
    from app.core import study_coordinator as sc
    monkeypatch.setattr(sc, "_STOP_RESEND_S", 0.3)
    original = type(service).cancel
    calls = {"n": 0}

    async def flaky(self, run_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("injected: the cancel was lost")
        return await original(self, run_id)

    monkeypatch.setattr(type(service), "cancel", flaky)
    release = probe.hold("A")
    created = await client.post("/api/studies", json=_body(
        [("A", 1.0)], repetitions=1, budget={"max_run_seconds": 0.3}))
    study_id = created.json()["study_id"]
    await probe.wait_entered("A")
    deadline = time.monotonic() + 10
    while calls["n"] < 2:
        assert time.monotonic() < deadline, "the stop was never sent again"
        await asyncio.sleep(0.05)
    release.set()
    await _wait_stored_status(coordinator, study_id, "succeeded")
    attempt = (await coordinator.store.list_attempts(study_id))[0]
    assert attempt.status == "timed_out"


async def test_the_budget_timer_does_not_spin_while_a_stopped_child_drains(
        client, env, probe, monkeypatch):
    _service, coordinator = env
    ticks = {"n": 0}
    original = type(coordinator).tick

    async def counting(self, study_id):
        ticks["n"] += 1
        return await original(self, study_id)

    monkeypatch.setattr(type(coordinator), "tick", counting)
    release = probe.hold("A")
    created = await client.post("/api/studies", json=_body(
        [("A", 1.0)], repetitions=2, budget={"max_active_seconds": 0.2}))
    study_id = created.json()["study_id"]
    await probe.wait_entered("A")
    await _wait_event(coordinator, study_id, "attempt_stop_requested")
    start = ticks["n"]
    await asyncio.sleep(1.5)          # the child ignores the cancel meanwhile
    assert ticks["n"] - start <= 3, ticks["n"] - start
    release.set()
    await _wait_stored_status(coordinator, study_id, "halted")


# ── expected_revision ─────────────────────────────────────────────────────


async def test_cancel_and_resume_refuse_a_stale_revision(client, probe):
    release = probe.hold("A")
    created = await client.post("/api/studies", json=_body(
        [("A", 1.0)], repetitions=2))
    study_id = created.json()["study_id"]
    await probe.wait_entered("A")
    stale = await client.post(f"/api/studies/{study_id}/cancel",
                              json={"expected_revision": 1})
    assert stale.status_code == 409
    assert "not the expected 1" in stale.json()["detail"]
    # The client's loop: read, act on that revision, re-read on a 409.
    for _ in range(5):
        revision = (await client.get(f"/api/studies/{study_id}")).json()[
            "revision"]
        fresh = await client.post(f"/api/studies/{study_id}/cancel",
                                  json={"expected_revision": revision})
        if fresh.status_code != 409:
            break
    assert fresh.status_code == 200, fresh.text
    release.set()
    probe.gates.pop("A")
    cancelled = await _wait_status(client, study_id, "cancelled")
    stale = await client.post(f"/api/studies/{study_id}/resume", json={
        "expected_revision": cancelled["revision"] + 5})
    assert stale.status_code == 409
    resumed = await client.post(f"/api/studies/{study_id}/resume", json={
        "expected_revision": cancelled["revision"]})
    assert resumed.status_code == 200, resumed.text
    await _wait_status(client, study_id, "succeeded")


# ── the audit's recovery acceptance ───────────────────────────────────────


async def test_an_idempotent_retry_after_a_restart_creates_nothing(
        make_env, store, probe):
    """The client lost the reply, the server restarted, the client retries
    with the same key: it gets the same (now interrupted) study back with
    200, and no child is submitted by the retry."""
    service, coordinator = make_env(shutdown_grace_s=0.1)
    body = _body()
    release = probe.hold("B")
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url=BASE_URL,
                           headers={TOKEN_HEADER: session_token()}) as http:
        first = await http.post("/api/studies", json=body,
                                headers={"Idempotency-Key": "lost-reply"})
        study_id = first.json()["study_id"]
        await probe.wait_entered("B")
        await _crash(service, coordinator)
        release.set()
        probe.gates.pop("B")

        service2, coordinator2 = make_env()
        await service2.recover_interrupted()
        await coordinator2.recover()
        before = len(await _study_runs(store, study_id))
        retry = await http.post("/api/studies", json=body,
                                headers={"Idempotency-Key": "lost-reply"})
        assert retry.status_code == 200, retry.text
        assert retry.json()["created"] is False
        assert retry.json()["study_id"] == study_id
        assert retry.json()["status"] == "interrupted"
        assert len(await _study_runs(store, study_id)) == before == 2
        listed = (await http.get("/api/studies")).json()
        assert listed["total"] == 1


async def test_concurrent_retries_with_one_key_create_one_study(client, store):
    body = _body()
    responses = await asyncio.gather(*(
        client.post("/api/studies", json=body,
                    headers={"Idempotency-Key": "burst"})
        for _ in range(4)))
    assert sorted(r.status_code for r in responses) == [200, 200, 200, 201]
    ids = {r.json()["study_id"] for r in responses}
    assert len(ids) == 1
    study_id = ids.pop()
    await _wait_status(client, study_id, "succeeded")
    assert len(await _study_runs(store, study_id)) == 4


async def test_a_restart_with_completed_interrupted_and_unlinked_repetitions(
        make_env, store, probe, monkeypatch):
    """One restart finds all three at once: A r0 succeeded, B r0 was
    running, and C r0 was submitted while its linkage write failed. The
    restart keeps A, retries B and C once each (relinking C's orphan by
    its stamp first), and no repetition runs twice side by side."""
    service, coordinator = make_env(shutdown_grace_s=0.1)
    original = StudyStore.confirm_submission
    calls = {"n": 0}

    async def flaky(self, attempt_id, status):
        calls["n"] += 1
        if calls["n"] == 3:                 # the third submission: C r0
            raise RuntimeError("injected: linkage write lost")
        return await original(self, attempt_id, status)

    monkeypatch.setattr(StudyStore, "confirm_submission", flaky)
    release = probe.hold("B")
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url=BASE_URL,
                           headers={TOKEN_HEADER: session_token()}) as http:
        created = await http.post("/api/studies", json=_body(
            [("A", 1.0), ("B", 2.0), ("C", 3.0)], repetitions=1,
            budget={"max_parallel_runs": 2}))
        study_id = created.json()["study_id"]
        await probe.wait_entered("B")

        async def c_submitting() -> bool:
            attempts = await coordinator.store.list_attempts(study_id)
            return any(a.variant_index == 2 and a.status == "submitting"
                       and a.run_id for a in attempts)

        deadline = time.monotonic() + 10
        while not await c_submitting():
            assert time.monotonic() < deadline, "C r0 was never submitted"
            await asyncio.sleep(0.02)
        await _crash(service, coordinator)
        release.set()
        probe.gates.pop("B")

        service2, coordinator2 = make_env()
        await service2.recover_interrupted()
        assert await coordinator2.recover() == 1
        waiting = (await http.get(f"/api/studies/{study_id}")).json()
        assert waiting["status"] == "interrupted"
        assert _attempts(waiting, 0, 0)[0]["status"] == "succeeded"
        assert _attempts(waiting, 1, 0)[0]["status"] == "interrupted"
        orphan = _attempts(waiting, 2, 0)[0]
        assert orphan["status"] == "interrupted"
        assert len(await _study_runs(store, study_id)) == 3

        resumed = await http.post(f"/api/studies/{study_id}/resume")
        assert resumed.status_code == 200, resumed.text
        final = await _wait_status(http, study_id, "succeeded")
    runs = await _study_runs(store, study_id)
    assert len(runs) == 5                   # A, B, C orphan, B and C retries
    assert probe.count("A") == 1
    assert [a["attempt"] for a in _attempts(final, 0, 0)] == [1]
    assert [a["status"] for a in _attempts(final, 1, 0)] == [
        "interrupted", "succeeded"]
    assert [a["status"] for a in _attempts(final, 2, 0)] == [
        "interrupted", "succeeded"]
    assert _attempts(final, 2, 0)[0]["run_id"] == orphan["run_id"]
    types = [e.type for e in await coordinator2.store.get_events(study_id)]
    assert types.count("attempt_relinked") == 1
    resume = next(e for e in await coordinator2.store.get_events(study_id)
                  if e.type == "study_resumed")
    assert resume.payload["completed_repetitions_kept"] == 1
    assert len(resume.payload["new_attempts"]) == 2
