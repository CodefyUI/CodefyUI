"""Runs and app versions belong to a project (MIGRATION_019, phase 4A).

A run records the data root and the project it was created in. Its files
are reachable only while that root is open (409 ``run_project_not_open``),
retention leaves another root's runs alone, and the Runs list shows the
open project's runs unless asked for all. An app version records the
project it was published from and is refused while another one is open
(409 ``app_project_not_open``). Rows from before the migration (NULL)
behave as they always did.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core import db as dbmod
from app.core.auth import TOKEN_HEADER, session_token
from app.core.db import Database
from app.core.migrations import MIGRATIONS
from app.core.run_service import RunService
from app.core.run_store import (
    RUN_PROJECT_NOT_OPEN,
    STATUS_SUCCEEDED,
    RunProvenance,
    RunStore,
    run_project_refusal,
)
from types import SimpleNamespace

from app.main import app

GRAPH = {"nodes": [], "edges": []}


def _open(monkeypatch, root: Path, *, project: bool = True) -> Path:
    """Point the data roots at *root* as if it were the open project."""
    models = root / "assets" / "models"
    models.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "MODELS_DIR", models)
    monkeypatch.setattr(settings, "PROJECT_DIR",
                        root.resolve() if project else None)
    return root.resolve()


@pytest.fixture
def store(tmp_path):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    try:
        yield RunStore(database)
    finally:
        database.close()


async def _run(store: RunStore, status: str = STATUS_SUCCEEDED) -> str:
    record = await store.create_run(graph_snapshot=GRAPH,
                                    provenance=RunProvenance(),
                                    status=status)
    return record.id


async def test_a_run_records_its_data_root_and_project(store, tmp_path,
                                                       monkeypatch):
    project = _open(monkeypatch, tmp_path / "alpha")
    run_id = await _run(store)
    record = await store.get_run(run_id)
    assert record.data_root == str((project / "assets").resolve())
    assert record.project_dir == str(project)
    assert run_project_refusal(record) is None


async def test_a_flat_mode_run_records_no_project(store, tmp_path,
                                                  monkeypatch):
    _open(monkeypatch, tmp_path / "flat", project=False)
    record = await store.get_run(await _run(store))
    assert record.project_dir is None
    assert record.data_root == str((tmp_path / "flat" / "assets").resolve())


async def test_another_open_root_refuses_and_names_the_project(
        store, tmp_path, monkeypatch):
    _open(monkeypatch, tmp_path / "alpha")
    record = await store.get_run(await _run(store))
    _open(monkeypatch, tmp_path / "beta")
    refusal = run_project_refusal(record)
    assert refusal["code"] == RUN_PROJECT_NOT_OPEN
    assert "'alpha'" in refusal["message"]
    assert refusal["details"]["project_dir"] == record.project_dir

    _open(monkeypatch, tmp_path / "flat", project=False)
    flat = await store.get_run(await _run(store))
    _open(monkeypatch, tmp_path / "beta")
    assert "no project" in run_project_refusal(flat)["message"]


async def test_the_list_filters_by_the_open_root_and_keeps_old_rows(
        store, tmp_path, monkeypatch):
    _open(monkeypatch, tmp_path / "alpha")
    alpha = await _run(store)
    _open(monkeypatch, tmp_path / "beta")
    beta = await _run(store)
    legacy = await _run(store)
    store.db._conn.execute(
        "UPDATE exec_runs SET data_root = NULL, project_dir = NULL "
        "WHERE id = ?", (legacy,))

    mine = {r.id for r in await store.list_runs(this_project=True)}
    assert mine == {beta, legacy}
    assert await store.count_runs(this_project=True) == 2
    everything = {r.id for r in await store.list_runs()}
    assert everything == {alpha, beta, legacy}

    _open(monkeypatch, tmp_path / "alpha")
    assert {r.id for r in await store.list_runs(this_project=True)} \
        == {alpha, legacy}
    assert await store.count_runs(status=STATUS_SUCCEEDED,
                                  this_project=True) == 2


async def test_retention_leaves_another_roots_runs_alone(store, tmp_path,
                                                         monkeypatch):
    _open(monkeypatch, tmp_path / "alpha")
    alpha = [await _run(store) for _ in range(3)]
    _open(monkeypatch, tmp_path / "beta")
    beta = [await _run(store) for _ in range(3)]

    await store.prune(keep_last=0)
    left = {r.id for r in await store.list_runs()}
    assert left == set(alpha)

    _open(monkeypatch, tmp_path / "alpha")
    await store.prune(keep_last=0)
    assert await store.list_runs() == []
    assert not set(beta) & {r.id for r in await store.list_runs()}


def test_the_migration_leaves_old_rows_null(tmp_path, monkeypatch):
    path = tmp_path / "v18.db"
    monkeypatch.setattr(dbmod, "MIGRATIONS", list(MIGRATIONS[:18]))
    old = Database(path)
    old.connect()
    old._conn.execute(
        "INSERT INTO exec_runs (id, graph_snapshot, options, status, "
        "created_at) VALUES ('r1', '{}', '{}', 'succeeded', "
        "'2026-01-01T00:00:00Z')")
    old.close()

    monkeypatch.setattr(dbmod, "MIGRATIONS", list(MIGRATIONS))
    new = Database(path)
    new.connect()
    try:
        record = asyncio.run(RunStore(new).get_run("r1"))
        assert (record.data_root, record.project_dir) == (None, None)
        assert run_project_refusal(record) is None
        columns = [r[1] for r in new._conn.execute(
            "PRAGMA table_info(app_versions)")]
        assert columns[-1] == "project_dir"
    finally:
        new.close()


# ── the routes ───────────────────────────────────────────────────────────


@pytest.fixture
async def client(tmp_path):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    service = RunService(RunStore(database), shutdown_grace_s=5.0)
    app.state.db = database
    app.state.run_service = service
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://localhost",
            headers={TOKEN_HEADER: session_token()},
        ) as http:
            yield http, service.store
    finally:
        await service.shutdown()
        database.close()
        for attribute in ("db", "run_service"):
            if hasattr(app.state, attribute):
                delattr(app.state, attribute)


async def test_the_file_routes_refuse_a_run_of_another_project(
        client, tmp_path, monkeypatch):
    http, store = client
    _open(monkeypatch, tmp_path / "alpha")
    run_id = await _run(store)
    await store.add_artifact(run_id, "checkpoint",
                             str(tmp_path / "alpha" / "assets" / "w.pt"))
    _open(monkeypatch, tmp_path / "beta")

    for method, url in (
        ("GET", f"/api/runs/{run_id}/artifacts"),
        ("GET", f"/api/runs/{run_id}/artifacts/1/download"),
        ("DELETE", f"/api/runs/{run_id}"),
    ):
        resp = await http.request(method, url)
        assert resp.status_code == 409, (method, url, resp.text)
        assert resp.json()["detail"]["code"] == RUN_PROJECT_NOT_OPEN

    # What is not a file stays readable: the row, its metrics, the list.
    assert (await http.get(f"/api/runs/{run_id}")).status_code == 200
    assert (await http.get(f"/api/runs/{run_id}/metrics")).status_code == 200
    listed = (await http.get("/api/runs")).json()
    assert listed["total"] == 0
    listed = (await http.get("/api/runs?scope=all")).json()
    assert [r["id"] for r in listed["runs"]] == [run_id]
    assert listed["runs"][0]["project_dir"] == str((tmp_path / "alpha").resolve())
    assert (await http.get("/api/runs?scope=other")).status_code == 422

    # Back in its project, the same routes answer.
    _open(monkeypatch, tmp_path / "alpha")
    resp = await http.get(f"/api/runs/{run_id}/artifacts")
    assert resp.status_code == 200
    assert (await http.get("/api/runs")).json()["total"] == 1
    assert (await http.delete(f"/api/runs/{run_id}")).status_code == 200


async def test_a_run_from_before_projects_stays_reachable(client, tmp_path,
                                                         monkeypatch):
    http, store = client
    _open(monkeypatch, tmp_path / "alpha")
    run_id = await _run(store)
    store.db._conn.execute(
        "UPDATE exec_runs SET data_root = NULL, project_dir = NULL "
        "WHERE id = ?", (run_id,))
    _open(monkeypatch, tmp_path / "beta")
    assert (await http.get(f"/api/runs/{run_id}/artifacts")).status_code == 200
    assert (await http.get("/api/runs")).json()["total"] == 1


def test_an_app_version_records_its_project(tmp_path):
    from app.api.routes_apps import app_project_refusal

    assert app_project_refusal(None) is None
    original = settings.PROJECT_DIR
    try:
        settings.PROJECT_DIR = (tmp_path / "alpha").resolve()
        assert app_project_refusal(str(settings.PROJECT_DIR)) is None
        message = app_project_refusal(str(tmp_path / "beta"))
        assert "'beta'" in message and "not open" in message
        settings.PROJECT_DIR = None
        assert app_project_refusal(str(tmp_path / "beta")) is not None
    finally:
        settings.PROJECT_DIR = original


async def test_invoking_a_version_of_another_project_is_refused(
        test_client, app_db, tmp_path, monkeypatch):
    from .test_api_apps_invoke import _bearer, _echo_graph, _publish

    monkeypatch.setattr(settings, "GRAPHS_DIR", tmp_path / "graphs")
    (tmp_path / "graphs").mkdir()
    key = (await test_client.post("/api/keys", json={"name": "p"})).json()
    _open(monkeypatch, tmp_path / "alpha")
    await _publish(test_client, "scoped", _echo_graph())
    row = app_db._conn.execute(
        "SELECT project_dir FROM app_versions").fetchone()
    assert row["project_dir"] == str((tmp_path / "alpha").resolve())

    async def invoke():
        return await test_client.post(
            "/api/apps/scoped/invoke", json={"inputs": {"x": "hi"}},
            headers=_bearer(key["token"]))

    _open(monkeypatch, tmp_path / "beta")
    resp = await invoke()
    assert resp.status_code == 409, resp.text
    body = resp.json()
    assert body["error"]["code"] == "app_project_not_open"
    assert "'alpha'" in body["error"]["message"]
    assert body["version"] == 1

    _open(monkeypatch, tmp_path / "alpha")
    resp = await invoke()
    assert resp.status_code == 200, resp.text
    assert resp.json()["outputs"] == {"y": "hi"}

    # A version from before projects (or from flat mode) runs anywhere.
    app_db._conn.execute("UPDATE app_versions SET project_dir = NULL")
    _open(monkeypatch, tmp_path / "beta")
    assert (await invoke()).status_code == 200


async def test_resuming_a_study_of_another_project_is_refused(
        client, tmp_path, monkeypatch):
    """A study's new attempts would run in the open project, beside
    children whose files are in another one (review round 1, item 7). The
    check is in ``StudyCoordinator.resume``, so the study route, the plugin
    job route and ``resume_policy: auto`` at start-up all meet it."""
    from app.core.study_coordinator import StudyCoordinator, StudyProjectNotOpen
    from app.core.study_store import StudyStore

    http, store = client
    _open(monkeypatch, tmp_path / "alpha")
    # A child row is all the check reads; the study row itself is not needed.
    await store.db.run(lambda conn: conn.execute("PRAGMA foreign_keys = OFF"))
    await store.create_run(graph_snapshot=GRAPH, provenance=RunProvenance(),
                           status=STATUS_SUCCEEDED, study_id="st-1")
    await store.db.run(lambda conn: conn.execute("PRAGMA foreign_keys = ON"))
    coordinator = StudyCoordinator(app.state.run_service, StudyStore(store.db))
    reached: list[str] = []

    async def get_study(study_id):
        reached.append(study_id)
        return SimpleNamespace(id=study_id, revision=1)

    monkeypatch.setattr(coordinator.store, "get_study", get_study)
    monkeypatch.setattr(app.state, "study_coordinator", coordinator, raising=False)

    _open(monkeypatch, tmp_path / "beta")
    resp = await http.post("/api/studies/st-1/resume")
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "study_project_not_open"
    assert "'alpha'" in detail["message"]
    assert detail["details"]["project_dir"] == str((tmp_path / "alpha").resolve())
    with pytest.raises(StudyProjectNotOpen):
        await coordinator.resume("st-1")

    # In its own project the check passes and the resume goes on (it then
    # stops on the stand-in study, which is not this test's business).
    _open(monkeypatch, tmp_path / "alpha")
    await coordinator._check_resume_project("st-1")
