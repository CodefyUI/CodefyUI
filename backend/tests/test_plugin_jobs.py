"""Plugin-scoped durable jobs (#343 increment 6): /api/plugins/{id}/jobs.

A job of kind ``study`` is a study owned by the plugin that submitted it:
the plugin sees and controls only its own, its idempotency keys are scoped
to it and required, and every job also shows in ``/api/studies``. These are
the calls Graph Copilot makes once the user approved a plan; the plugin
itself lives in its own repository.
"""

# The fixtures imported from test_studies are re-bound as test arguments.
# ruff: noqa: F811

from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.core import plugin_jobs
from app.core.plugin_jobs import JobKind, get_job_kind, register_job_kind
from app.main import app

from .test_studies import (  # noqa: F401  (fixtures)
    BASE_URL,
    _attempts,
    _body,
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

PLUGIN = "graph-copilot"
JOBS = f"/api/plugins/{PLUGIN}/jobs"


@pytest.fixture(autouse=True)
def lockfile(monkeypatch):
    plugins: dict[str, Any] = {PLUGIN: {"enabled": True},
                               "other": {"enabled": True},
                               "off": {"enabled": False}}
    monkeypatch.setattr(plugin_jobs.plugin_loader, "load_lockfile",
                        lambda: {"plugins": plugins})
    return plugins


def _job(spec: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    return {"kind": "study", "spec_version": 1,
            "spec": spec or _body(), **extra}


async def _job_status(client: AsyncClient, job_id: str, *statuses: str,
                      plugin: str = PLUGIN) -> dict[str, Any]:
    body = await _wait_status(client, job_id, *statuses)
    assert body["owner"] == plugin
    return body


async def test_kinds_name_the_study_kind(client):
    response = await client.get(f"{JOBS}/kinds")
    assert response.status_code == 200
    body = response.json()
    assert body["api_version"] == 1
    assert [k["kind"] for k in body["kinds"]] == ["study"]
    assert body["kinds"][0]["spec_version"] == 1


async def test_a_job_is_a_study_owned_by_the_plugin(client, store):
    response = await client.post(JOBS, json=_job(),
                                 headers={"Idempotency-Key": "plan-1"})
    assert response.status_code == 202, response.text
    job = response.json()
    assert job["created"] is True
    assert (job["plugin_id"], job["kind"], job["spec_version"]) == (
        PLUGIN, "study", 1)
    job_id = job["job_id"]
    assert job["study_id"] == job_id and job["owner"] == PLUGIN
    assert job["idempotency_key"] == "plan-1"          # as the plugin sent it
    assert len(job["request_digest"]) == 64
    done = await _job_status(client, job_id, "succeeded")
    assert [r["label"] for r in done["rankings"]] == ["A", "B"]

    fetched = (await client.get(f"{JOBS}/{job_id}")).json()
    assert fetched["status"] == "succeeded" and fetched["job_id"] == job_id
    listed = (await client.get(JOBS)).json()
    assert [j["job_id"] for j in listed["jobs"]] == [job_id]
    assert listed["jobs"][0]["counts"] == {"succeeded": 4}
    # It is a study like any other: /api/studies lists it, with its owner.
    studies = (await client.get("/api/studies")).json()["studies"]
    assert studies[0]["owner"] == PLUGIN
    events = (await client.get(f"{JOBS}/{job_id}/events",
                               params={"after": 2})).json()
    assert events["events"][0]["cursor"] == 3
    assert events["events"][0]["sequence"] == 3
    assert events["events"][-1]["type"] == "study_finished"
    created = (await client.get(f"{JOBS}/{job_id}/events")).json()[
        "events"][0]
    assert created["payload"]["owner"] == PLUGIN
    assert created["payload"]["idempotency_key"] == "plan-1"
    graph = (await client.get(f"{JOBS}/{job_id}/variants/1/graph")).json()
    assert graph["label"] == "B"
    probe_node = next(n for n in graph["graph"]["nodes"] if n["id"] == "probe")
    assert probe_node["data"]["params"]["value"] == 2.0
    assert (await client.get(f"{JOBS}/{job_id}/variants/9/graph")
            ).status_code == 404
    core = (await client.get(f"/api/studies/{job_id}/variants/0/graph")
            ).json()
    assert core["label"] == "A"
    assert len(await _study_runs(store, job_id)) == 4


async def test_a_retry_with_the_key_returns_the_job_and_another_plugin_is_apart(
        client, store):
    first = await client.post(JOBS, json=_job(),
                              headers={"Idempotency-Key": "k"})
    retry = await client.post(JOBS, json=_job(idempotency_key="k"))
    assert retry.status_code == 200, retry.text
    assert retry.json()["created"] is False
    assert retry.json()["job_id"] == first.json()["job_id"]
    # The same key from another plugin is another job.
    other = await client.post("/api/plugins/other/jobs", json=_job(),
                              headers={"Idempotency-Key": "k"})
    assert other.status_code == 202, other.text
    assert other.json()["job_id"] != first.json()["job_id"]
    # ... and the same key on /api/studies is a third, unrelated study.
    core = await client.post("/api/studies", json=_body(),
                             headers={"Idempotency-Key": "k"})
    assert core.status_code == 201
    # Reused for a different request: 409.
    changed = await client.post(JOBS, json=_job(_body(repetitions=1)),
                                headers={"Idempotency-Key": "k"})
    assert changed.status_code == 409
    assert "already used for job" in changed.json()["detail"]
    for response in (first, other, core):
        await _wait_status(client, response.json()["study_id"], "succeeded")
    assert len(await _study_runs(store, first.json()["job_id"])) == 4


async def test_a_plugin_sees_and_controls_only_its_own_jobs(client, probe):
    release = probe.hold("A")
    created = await client.post(JOBS, json=_job(_body([("A", 1.0)])),
                                headers={"Idempotency-Key": "mine"})
    job_id = created.json()["job_id"]
    await probe.wait_entered("A")
    foreign = "/api/plugins/other/jobs"
    assert (await client.get(f"{foreign}/{job_id}")).status_code == 404
    assert (await client.post(f"{foreign}/{job_id}/cancel")
            ).status_code == 404
    assert (await client.post(f"{foreign}/{job_id}/resume")
            ).status_code == 404
    assert (await client.get(f"{foreign}/{job_id}/events")).status_code == 404
    assert (await client.get(f"{foreign}/{job_id}/variants/0/graph")
            ).status_code == 404
    assert (await client.get(foreign)).json()["total"] == 0
    # A core study is no plugin's job.
    core = await client.post("/api/studies", json=_body([("A", 1.0)]))
    assert (await client.get(f"{JOBS}/{core.json()['study_id']}")
            ).status_code == 404

    cancelled = await client.post(f"{JOBS}/{job_id}/cancel")
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] == "cancel_requested"
    assert len(cancelled.json()["cancelled_runs"]) == 1
    release.set()
    probe.gates.pop("A")
    await _job_status(client, job_id, "cancelled")
    resumed = await client.post(f"{JOBS}/{job_id}/resume", json={
        "budget": {"max_parallel_runs": 1, "max_active_seconds": 60}})
    assert resumed.status_code == 200, resumed.text
    done = await _job_status(client, job_id, "succeeded")
    assert done["budget"]["max_active_seconds"] == 60.0
    assert [a["status"] for a in _attempts(done, 0, 0)] == [
        "cancelled", "succeeded"]
    await _wait_status(client, core.json()["study_id"], "succeeded")


@pytest.mark.parametrize("payload, headers, status, fragment", [
    (_job(), {}, 400, "needs an Idempotency-Key"),
    (_job(idempotency_key="a"), {"Idempotency-Key": "b"}, 400, "differ"),
    ({**_job(), "kind": "llm_plan"}, {"Idempotency-Key": "x"}, 400,
     "unknown job kind"),
    ({**_job(), "spec_version": 2}, {"Idempotency-Key": "x"}, 400,
     "spec_version 1"),
    (_job({**_body(), "repetitions": 0}), {"Idempotency-Key": "x"}, 422,
     "spec"),
    ({**_job(), "extra": 1}, {"Idempotency-Key": "x"}, 422, "extra"),
    (_job({**_body(), "options": {"seed": 1}}), {"Idempotency-Key": "x"},
     400, "options.seed is owned"),
])
async def test_invalid_job_submissions_write_nothing(
        client, payload, headers, status, fragment):
    response = await client.post(JOBS, json=payload, headers=headers)
    assert response.status_code == status, response.text
    assert fragment in response.text
    assert (await client.get("/api/studies")).json()["total"] == 0


async def test_only_an_enabled_installed_plugin_may_submit_or_resume(
        client, lockfile, probe):
    missing = await client.post("/api/plugins/nope/jobs", json=_job(),
                                headers={"Idempotency-Key": "x"})
    assert missing.status_code == 404
    assert "not installed" in missing.json()["detail"]
    disabled = await client.post("/api/plugins/off/jobs", json=_job(),
                                 headers={"Idempotency-Key": "x"})
    assert disabled.status_code == 409
    assert "disabled" in disabled.json()["detail"]

    release = probe.hold("A")
    created = await client.post(JOBS, json=_job(_body([("A", 1.0)])),
                                headers={"Idempotency-Key": "y"})
    job_id = created.json()["job_id"]
    await probe.wait_entered("A")
    lockfile[PLUGIN]["enabled"] = False
    # A disabled plugin's job can still be read and stopped ...
    assert (await client.get(f"{JOBS}/{job_id}")).status_code == 200
    assert (await client.post(f"{JOBS}/{job_id}/cancel")).status_code == 200
    release.set()
    await _wait_status(client, job_id, "cancelled")
    # ... but not resumed.
    refused = await client.post(f"{JOBS}/{job_id}/resume")
    assert refused.status_code == 409
    lockfile[PLUGIN]["enabled"] = True


async def test_a_stale_expected_revision_is_409(client, probe):
    release = probe.hold("A")
    created = await client.post(JOBS, json=_job(_body([("A", 1.0)])),
                                headers={"Idempotency-Key": "rev"})
    job_id = created.json()["job_id"]
    await probe.wait_entered("A")
    stale = await client.post(f"{JOBS}/{job_id}/cancel",
                              json={"expected_revision": 0})
    assert stale.status_code == 409
    release.set()
    probe.gates.pop("A")
    await _wait_status(client, job_id, "succeeded")
    listed = (await client.get(JOBS, params={"status": "succeeded"})).json()
    assert listed["total"] == 1
    assert (await client.get(JOBS, params={"status": "nope"})
            ).status_code == 400


async def test_the_job_routes_need_the_token_to_mutate(env):
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url=BASE_URL) as anonymous:
        response = await anonymous.post(JOBS, json=_job(),
                                        headers={"Idempotency-Key": "x"})
        assert response.status_code == 403
        assert (await anonymous.get(f"{JOBS}/kinds")).status_code == 200


def test_a_job_kind_registers_once():
    assert get_job_kind("study") is not None
    with pytest.raises(ValueError, match="already registered"):
        register_job_kind(JobKind("study", 1, "again"))
