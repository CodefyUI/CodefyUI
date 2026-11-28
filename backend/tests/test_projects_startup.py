"""Phase 4A: the project registry, trust, and what start-up opens."""

import json
import os
import shutil
from pathlib import Path

import pytest

from app.config import PROJECT_DERIVED_ROOTS, settings
from app.core import dotenv, projects
from app.core.project import PROJECT_MANIFEST, scaffold_project


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """A clean user-data dir, projects root and flat graphs dir; settings,
    os.environ and the applied-.env record restored afterwards."""
    saved = {name: getattr(settings, name) for name in ("PROJECT_DIR", *PROJECT_DERIVED_ROOTS)}
    saved_explicit = settings._explicit_roots
    monkeypatch.setattr(projects, "_activated", None)
    environ = dict(os.environ)
    dotenv._APPLIED.clear()
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path / "user"))
    monkeypatch.setattr(settings, "PROJECTS_ROOT", tmp_path / "projects")
    monkeypatch.setattr(settings, "DEFAULT_PROJECT", True)
    settings._explicit_roots = frozenset()
    settings.derive_project_roots(None)
    flat = tmp_path / "flat-graphs"
    monkeypatch.setattr(projects, "flat_graphs_dir", lambda: flat)
    yield tmp_path
    dotenv._APPLIED.clear()
    os.environ.clear()
    os.environ.update(environ)
    settings._explicit_roots = saved_explicit
    for name, value in saved.items():
        setattr(settings, name, value)


def _project(path: Path, env: str | None = None) -> Path:
    scaffold_project(path)
    if env is not None:
        (path / ".env").write_text(env, encoding="utf-8")
    return path


def test_a_fresh_install_gets_a_default_project(isolated):
    decision = projects.resolve_startup_project()
    target = isolated / "projects" / "workspace"
    assert decision.reason == "default" and decision.created
    assert (target / PROJECT_MANIFEST).is_file()
    if shutil.which("git"):
        assert (target / ".git").is_dir()
    assert settings.PROJECT_DIR == target.resolve()
    assert settings.GRAPHS_DIR == target.resolve() / "graphs"
    assert os.environ["CODEFYUI_PROJECT_DIR"] == str(target.resolve())
    assert projects.load_registry()["current"] == str(target.resolve())
    # The next start reopens it through the registry, and creates nothing.
    again = projects.resolve_startup_project()
    assert again.reason == "registry" and not again.created


def test_old_flat_graphs_keep_flat_mode_and_are_left_alone(isolated):
    flat = isolated / "flat-graphs"
    flat.mkdir()
    (flat / "a.json").write_text('{"nodes": [], "edges": []}', encoding="utf-8")
    (flat / "b.graph.json").write_text('{"nodes": [], "edges": []}', encoding="utf-8")
    before = {p.name: p.read_bytes() for p in flat.iterdir()}
    decision = projects.resolve_startup_project()
    assert decision.reason == "flat-graphs" and decision.flat_graphs == 2
    assert settings.PROJECT_DIR is None
    assert not (isolated / "projects").exists()
    assert {p.name: p.read_bytes() for p in flat.iterdir()} == before


def test_default_project_off_means_flat(isolated, monkeypatch):
    monkeypatch.setattr(settings, "DEFAULT_PROJECT", False)
    assert projects.resolve_startup_project().reason == "flat"
    assert not (isolated / "projects").exists()


def test_a_non_project_folder_in_the_default_place_is_not_touched(isolated):
    target = isolated / "projects" / "workspace"
    target.mkdir(parents=True)
    (target / "notes.txt").write_text("mine", encoding="utf-8")
    decision = projects.resolve_startup_project()
    assert decision.reason == "flat" and decision.warnings
    assert sorted(p.name for p in target.iterdir()) == ["notes.txt"]


def test_an_empty_folder_in_the_default_place_becomes_the_project(isolated):
    (isolated / "projects" / "workspace").mkdir(parents=True)
    assert projects.resolve_startup_project().created


def test_an_explicit_graphs_dir_keeps_flat_mode(isolated):
    settings._explicit_roots = frozenset({"GRAPHS_DIR"})
    assert projects.resolve_startup_project().reason == "flat-pinned"
    assert projects.load_registry()["current"] is None


def test_explicit_project_loads_its_env_without_the_blocklist(isolated):
    proj = _project(isolated / "explicit", env="P4_SECRET=1\nCODEFYUI_X=2\n")
    os.environ.pop("P4_SECRET", None)
    os.environ.pop("CODEFYUI_X", None)
    settings.PROJECT_DIR = proj.resolve()
    decision = projects.resolve_startup_project()
    assert decision.reason == "explicit"
    assert os.environ["P4_SECRET"] == "1" and os.environ["CODEFYUI_X"] == "2"
    assert projects.env_trust_state(proj) == "trusted"
    assert projects.load_registry()["current"] == str(proj.resolve())


def test_explicit_project_never_sets_python_or_loader_variables(isolated, caplog):
    """Phase 4C: even the project named with --project cannot set PYTHON*,
    LD_* or DYLD_* -- they choose which code child processes run. Listed as
    blocked, logged by name (never value), everything else still loaded."""
    proj = _project(isolated / "explicit", env=(
        "P4_SECRET=1\nPYTHONPATH=/evil\npythonstartup=/x\nLD_PRELOAD=/evil.so\n"
        "DYLD_INSERT_LIBRARIES=/evil.dylib\nCODEFYUI_X=2\n"))
    for name in ("P4_SECRET", "PYTHONPATH", "pythonstartup", "LD_PRELOAD",
                 "DYLD_INSERT_LIBRARIES", "CODEFYUI_X"):
        os.environ.pop(name, None)
    settings.PROJECT_DIR = proj.resolve()
    with caplog.at_level("WARNING"):
        projects.resolve_startup_project()
    assert os.environ["P4_SECRET"] == "1" and os.environ["CODEFYUI_X"] == "2"
    for name in ("PYTHONPATH", "pythonstartup", "LD_PRELOAD", "DYLD_INSERT_LIBRARIES"):
        assert name not in os.environ
    assert projects.current_env.blocked == [
        "DYLD_INSERT_LIBRARIES", "LD_PRELOAD", "PYTHONPATH", "pythonstartup"]
    assert "LD_PRELOAD" in caplog.text and "/evil.so" not in caplog.text


def test_registry_project_with_an_untrusted_env_opens_without_it(isolated):
    proj = _project(isolated / "cloned", env="P4_SECRET=1\nPYTHONPATH=/evil\n")
    os.environ.pop("P4_SECRET", None)
    projects.record_opened(proj)
    decision = projects.resolve_startup_project()
    assert decision.reason == "registry"
    status = projects.current_env
    assert status.state == "untrusted"
    assert status.names == ["P4_SECRET", "PYTHONPATH"] and status.blocked == ["PYTHONPATH"]
    assert "P4_SECRET" not in os.environ
    projects.trust_env(proj, projects.env_digest(proj))
    projects.resolve_startup_project()
    assert os.environ["P4_SECRET"] == "1"
    assert os.environ.get("PYTHONPATH") != "/evil"
    assert projects.current_env.blocked == ["PYTHONPATH"]


def test_a_changed_env_needs_trust_again(isolated):
    proj = _project(isolated / "p", env="A=1\n")
    projects.trust_env(proj, projects.env_digest(proj))
    assert projects.env_trust_state(proj) == "trusted"
    (proj / ".env").write_text("A=2\n", encoding="utf-8")
    assert projects.env_trust_state(proj) == "changed"


def test_an_owned_project_is_trusted_whatever_its_env(isolated):
    proj = _project(isolated / "mine", env="A=1\n")
    projects.mark_owned(proj)
    (proj / ".env").write_text("A=2\n", encoding="utf-8")
    assert projects.env_trust_state(proj) == "trusted"


def test_a_vanished_registry_project_falls_through_with_a_warning(isolated, monkeypatch):
    monkeypatch.setattr(settings, "DEFAULT_PROJECT", False)
    gone = _project(isolated / "gone")
    projects.record_opened(gone)
    shutil.rmtree(gone)
    decision = projects.resolve_startup_project()
    assert decision.reason == "flat" and any("gone" in w for w in decision.warnings)


def test_switching_away_takes_the_env_back_out(isolated):
    a = _project(isolated / "a", env="P4_A=1\n")
    projects.mark_owned(a)
    os.environ.pop("P4_A", None)
    projects.activate(a)
    assert os.environ["P4_A"] == "1"
    projects.activate(None)
    assert "P4_A" not in os.environ and "CODEFYUI_PROJECT_DIR" not in os.environ
    assert settings.PROJECT_DIR is None


def test_recent_list_and_removal(isolated):
    a, b = _project(isolated / "a"), _project(isolated / "b")
    projects.record_opened(a)
    projects.record_opened(b)
    projects.record_opened(a)
    data = projects.load_registry()
    assert [e["path"] for e in data["recent"]] == [str(a.resolve()), str(b.resolve())]
    assert data["current"] == str(a.resolve())
    assert projects.remove_recent(str(b)) and not projects.remove_recent(str(b))
    assert (b / PROJECT_MANIFEST).is_file()


def test_a_corrupt_registry_is_moved_aside(isolated):
    path = projects.registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert projects.load_registry() == {"current": None, "recent": []}
    assert list(path.parent.glob("projects.json.corrupt-*"))


def test_registry_and_trust_live_outside_the_project(isolated):
    proj = _project(isolated / "p", env="A=1\n")
    projects.record_opened(proj)
    projects.trust_env(proj, projects.env_digest(proj))
    assert projects.trust_path().parent == Path(os.environ["CODEFYUI_USER_DATA_DIR"])
    assert not any(p.name in ("trust.json", "projects.json") for p in proj.rglob("*"))
    assert json.loads(projects.trust_path().read_text())["version"] == 1
