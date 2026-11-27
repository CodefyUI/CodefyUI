"""Phase 4A: switching the open project, and every reason it is refused."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import settings
from app.core import project_switch, projects
from app.core.db import Database
from app.core.project import PROJECT_MANIFEST, scaffold_project
from app.core.project_switch import (
    PathRefused,
    SwitchRefused,
    check_project_path,
    refused_location,
    switch_project,
)
from app.core.run_service import RunService, RunServiceUnavailable
from app.core.run_store import RunStore

from .test_projects_startup import isolated  # noqa: F401  (fixture)


# ── fakes for app.state ───────────────────────────────────────────────────


class FakeRuns:
    def __init__(self, active=(), queued=(), submitting=0, later_active=None):
        self.active, self.queued, self.submitting = list(active), list(queued), submitting
        self.later_active = later_active
        self.calls = 0
        self.closed: list[str | None] = []

    def active_run_ids(self):
        self.calls += 1
        if self.later_active is not None and self.calls > 1:
            return list(self.later_active)
        return list(self.active)

    def queued_run_ids(self):
        return list(self.queued)

    def submits_in_flight(self):
        return self.submitting

    def close_for_switch(self, reason):
        self.closed.append(reason)

    def reopen_after_switch(self):
        self.closed.append(None)


class FakeStudies:
    def __init__(self, names):
        self.names = names

    async def list_studies(self, *, status, limit):
        rows = [SimpleNamespace(name=n, id=f"id-{n}") for n in self.names]
        return rows[:limit], len(rows)


class FakeStore:
    def __init__(self):
        self.cleared = False

    def reset_all(self):
        self.cleared = True
        return 0

    def clear(self):
        self.cleared = True


class FakeOutputs(FakeStore):
    async def clear(self):  # type: ignore[override]
        self.cleared = True


def _state(**overrides):
    state = SimpleNamespace(
        run_service=FakeRuns(),
        study_coordinator=SimpleNamespace(store=FakeStudies([])),
        plugin_service=SimpleNamespace(current_job_id=lambda: None),
        pack_service=SimpleNamespace(current_job_id=lambda: None),
        git_service=SimpleNamespace(lock=asyncio.Lock(),
                                    network_lock=asyncio.Lock()),
        worker_registry=SimpleNamespace(workers=lambda: []),
        node_state_store=FakeStore(),
        run_output_store=FakeOutputs(),
        port_stats_cache=FakeStore(),
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def _project(path: Path, env: str | None = None) -> Path:
    scaffold_project(path)
    if env is not None:
        (path / ".env").write_text(env, encoding="utf-8")
    return path.resolve()


@pytest.fixture
def two(isolated):  # noqa: F811
    alpha = _project(isolated / "projects" / "alpha")
    beta = _project(isolated / "projects" / "beta")
    projects.activate(alpha)
    projects.record_opened(alpha)
    return alpha, beta


# ── the switch ────────────────────────────────────────────────────────────


async def test_a_switch_moves_every_root_and_clears_the_old_state(two):
    alpha, beta = two
    state = _state()
    result = await switch_project(state, beta)
    assert (result.previous, result.project) == (alpha, beta)
    assert settings.PROJECT_DIR == beta
    assert settings.GRAPHS_DIR == beta / "graphs"
    assert settings.MODELS_DIR == beta / "assets" / "models"
    assert os.environ["CODEFYUI_PROJECT_DIR"] == str(beta)
    assert projects.load_registry()["current"] == str(beta)
    assert state.node_state_store.cleared and state.run_output_store.cleared
    assert state.port_stats_cache.cleared
    # Submits were closed for the switch and opened again afterwards.
    assert state.run_service.closed[0] and state.run_service.closed[-1] is None
    assert not state.git_service.lock.locked()


async def test_an_untrusted_env_is_not_loaded_on_a_switch(two, monkeypatch):
    alpha, _ = two
    gamma = _project(alpha.parent / "gamma", env="P4_SWITCH_VALUE=1\nPATH=/x\n")
    monkeypatch.delenv("P4_SWITCH_VALUE", raising=False)
    result = await switch_project(_state(), gamma)
    assert result.env.state == "untrusted"
    assert result.env.names == ["P4_SWITCH_VALUE", "PATH"]
    assert result.env.blocked == ["PATH"]
    assert "P4_SWITCH_VALUE" not in os.environ


async def test_the_previous_projects_env_goes_away(two, monkeypatch):
    alpha, beta = two
    (alpha / ".env").write_text("P4_FROM_ALPHA=yes\n", encoding="utf-8")
    projects.mark_owned(alpha)
    monkeypatch.delenv("P4_FROM_ALPHA", raising=False)
    projects.activate(alpha)
    assert os.environ["P4_FROM_ALPHA"] == "yes"
    await switch_project(_state(), beta)
    assert "P4_FROM_ALPHA" not in os.environ


@pytest.mark.parametrize("state_overrides, code, words", [
    ({"run_service": FakeRuns(active=["r1"])}, "runs_active", "1 run is running"),
    ({"run_service": FakeRuns(queued=["r1", "r2"])}, "runs_active", "2 runs are queued"),
    ({"run_service": FakeRuns(submitting=1)}, "runs_active", "being submitted"),
    ({"study_coordinator": SimpleNamespace(store=FakeStudies(["lr-grid"]))},
     "study_active", "'lr-grid'"),
    ({"plugin_service": SimpleNamespace(current_job_id=lambda: "job-7")},
     "install_active", "plugin install"),
    ({"pack_service": SimpleNamespace(current_job_id=lambda: "job-8")},
     "install_active", "pack install"),
    ({"worker_registry": SimpleNamespace(workers=lambda: [SimpleNamespace(
        name="gpu-box", status="online", mounts={"data": "/srv/alpha"})])},
     "worker_mount", "worker 'gpu-box'"),
])
async def test_each_busy_reason_refuses_and_changes_nothing(
        two, state_overrides, code, words):
    alpha, beta = two
    state = _state(**state_overrides)
    with pytest.raises(SwitchRefused) as info:
        await switch_project(state, beta)
    assert info.value.code == code
    assert words in info.value.message
    assert settings.PROJECT_DIR == alpha
    assert projects.load_registry()["current"] == str(alpha)
    assert not state.node_state_store.cleared


async def test_an_offline_worker_or_a_lent_mount_does_not_refuse(two):
    _, beta = two
    workers = [SimpleNamespace(name="off", status="offline",
                               mounts={"data": "/srv"}),
               SimpleNamespace(name="lent", status="online", mounts={})]
    await switch_project(_state(worker_registry=SimpleNamespace(
        workers=lambda: workers)), beta)
    assert settings.PROJECT_DIR == beta


async def test_a_git_operation_refuses(two):
    alpha, beta = two
    state = _state()
    await state.git_service.lock.acquire()
    try:
        with pytest.raises(SwitchRefused) as info:
            await switch_project(state, beta)
    finally:
        state.git_service.lock.release()
    assert info.value.code == "git_busy"
    assert settings.PROJECT_DIR == alpha


async def test_a_second_switch_meanwhile_refuses(two):
    alpha, beta = two
    async with project_switch._switch_lock:
        with pytest.raises(SwitchRefused) as info:
            await switch_project(_state(), beta)
    assert info.value.code == "switch_in_progress"
    assert settings.PROJECT_DIR == alpha


async def test_a_run_that_arrives_before_submits_close_is_caught(two):
    alpha, beta = two
    runs = FakeRuns(later_active=["late"])
    state = _state(run_service=runs)
    with pytest.raises(SwitchRefused) as info:
        await switch_project(state, beta)
    assert info.value.code == "runs_active"
    assert settings.PROJECT_DIR == alpha
    # Reopened, and the git lock let go, on the way out.
    assert runs.closed[-1] is None
    assert not state.git_service.lock.locked()


async def test_a_failure_part_way_puts_the_old_project_back(two, monkeypatch):
    alpha, beta = two

    def broken(path):
        raise OSError("disk full")

    monkeypatch.setattr(projects, "record_opened", broken)
    state = _state()
    with pytest.raises(OSError):
        await switch_project(state, beta)
    assert settings.PROJECT_DIR == alpha
    assert settings.GRAPHS_DIR == alpha / "graphs"
    assert os.environ["CODEFYUI_PROJECT_DIR"] == str(alpha)
    assert projects.load_registry()["current"] == str(alpha)
    assert state.run_service.closed[-1] is None


# ── RunService closed for a switch ────────────────────────────────────────


async def test_run_service_refuses_submits_while_closed(tmp_path):
    database = Database(tmp_path / "runs.db")
    database.connect()
    service = RunService(RunStore(database), shutdown_grace_s=1.0)
    try:
        service.close_for_switch("the project is being switched")
        with pytest.raises(RunServiceUnavailable, match="being switched"):
            await service.submit({"nodes": [], "edges": []})
        assert service.submits_in_flight() == 0
        assert service.queued_run_ids() == []
        service.reopen_after_switch()
        with pytest.raises(Exception) as info:
            await service.submit({"nodes": [], "edges": []})
        assert not isinstance(info.value, RunServiceUnavailable) \
            or "switched" not in str(info.value)
        assert service.submits_in_flight() == 0
    finally:
        await service.shutdown()
        database.close()


# ── paths ─────────────────────────────────────────────────────────────────


def test_check_project_path_accepts_a_project(two):
    _, beta = two
    assert check_project_path(str(beta)) == beta


@pytest.mark.parametrize("make, status, code", [
    (lambda root: "relative/dir", 400, "invalid_path"),
    (lambda root: "", 400, "invalid_path"),
    (lambda root: str(root / "missing"), 400, "not_found"),
    (lambda root: str(_file(root / "a-file")), 400, "not_a_directory"),
    (lambda root: str(_dir(root / "plain")), 400, "not_a_project"),
    (lambda root: str(_bad_manifest(root / "broken")), 400, "invalid_manifest"),
    (lambda root: "/etc", 403, "path_refused"),
    (lambda root: str(Path.home()), 403, "path_refused"),
])
def test_check_project_path_refusals(isolated, make, status, code):  # noqa: F811
    with pytest.raises(PathRefused) as info:
        check_project_path(make(isolated))
    assert (info.value.status, info.value.code) == (status, code)


def _file(path: Path) -> Path:
    path.write_text("x", encoding="utf-8")
    return path


def _dir(path: Path) -> Path:
    path.mkdir(parents=True)
    return path


def _bad_manifest(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / PROJECT_MANIFEST).write_text("[project\nname = ", encoding="utf-8")
    return path


def test_refused_locations(isolated, monkeypatch):  # noqa: F811
    from app.config import _user_data_root

    install = project_switch._install_dir()
    assert refused_location(install / "backend" / "data") is not None
    assert refused_location(_user_data_root().resolve() / "x") is not None
    plugins = Path(settings.PLUGINS_USER_DIR).expanduser().resolve()
    assert refused_location(plugins / "some-plugin") is not None
    assert refused_location(Path("/")) is not None
    assert refused_location(Path("/usr/local/project")) is not None
    assert refused_location(isolated / "projects" / "fine") is None
    # A projects root a developer keeps inside the checkout is allowed.
    monkeypatch.setattr(settings, "PROJECTS_ROOT", install / ".scratch-projects")
    assert refused_location(install / ".scratch-projects" / "p") is None
