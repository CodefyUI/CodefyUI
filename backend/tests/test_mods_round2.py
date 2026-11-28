"""Phase 4C review round 1: the owner shortcut needs a baseline (a pull
into an owned project that never had mods must ask), the mods state on
disk vs what was loaded, the scan in mods' own words, and the files that
are never loaded marked as such."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from app.config import settings
from app.core import mods, projects
from app.core.node_registry import registry
from app.core.project import init_new_project_repo, scaffold_project

from .test_projects_startup import isolated  # noqa: F401  (fixture)

GIT = shutil.which("git")
needs_git = pytest.mark.skipif(GIT is None, reason="needs git")


def _node(name: str, marker: Path | None = None) -> str:
    mark = f"open({str(marker)!r}, 'a').write('ran\\n')\n" if marker else ""
    return (
        "from app.core.node_base import BaseNode\n" + mark + "\n\n"
        f"class N(BaseNode):\n    NODE_NAME = {name!r}\n    CATEGORY = 'Project'\n"
        "    @classmethod\n    def define_inputs(cls):\n        return []\n"
        "    @classmethod\n    def define_outputs(cls):\n        return []\n"
        "    def execute(self, inputs, params):\n        return {}\n"
    )


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@e",
                        "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"})


@pytest.fixture(autouse=True)
def _server(monkeypatch):
    """These are the server's rules (the owner shortcut is the server's)."""
    monkeypatch.setattr(mods, "_server_process", True)
    monkeypatch.setattr(mods, "_ran", set())
    mods.unload_project_mods()
    yield
    mods.unload_project_mods()


def _owned_repo(root: Path) -> Path:
    """What /api/projects/create does: scaffold, git init + commit, owned."""
    scaffold_project(root)
    assert init_new_project_repo(root)
    projects.mark_owned(root)
    return root.resolve()


def _collaborator_adds(project: Path, rel: str, text: str, tmp: Path) -> None:
    collab = tmp / f"collab-{project.name}"
    _git(tmp, "clone", "-q", str(project), str(collab))
    _write(collab / "mods" / rel, text)
    _git(collab, "add", "-A")
    _git(collab, "commit", "-qm", "collaborator adds code")
    _git(project, "pull", "-q", str(collab), "HEAD")


# ── (1) the owner shortcut needs a baseline ──────────────────────────────

@needs_git
def test_first_mods_pulled_into_an_owned_project_ask_and_do_not_run(
        isolated, tmp_path_factory):  # noqa: F811
    marker = tmp_path_factory.mktemp("out") / "ran"
    p = _owned_repo(isolated / "projects" / "mine")
    assert mods.load_project_mods(p)["state"] == "none"     # no mods: baseline kept
    _collaborator_adds(p, "nodes/evil.py", _node("Evil", marker), isolated)
    status = mods.load_project_mods(p)
    assert status["state"] == "changed"
    assert status["loaded"] is False
    assert registry.get("project:Evil") is None
    assert not marker.exists()
    assert mods.mods_trust_state(p)["changed"]["added"] == ["nodes/evil.py"]


@needs_git
def test_the_owner_adding_a_file_without_moving_head_is_not_asked(isolated):  # noqa: F811
    p = _owned_repo(isolated / "projects" / "mine")
    mods.load_project_mods(p)
    _write(p / "mods" / "nodes" / "mine.py", _node("Mine"))
    status = mods.load_project_mods(p)
    assert status["state"] == "trusted" and status["nodes"] == ["project:Mine"]
    # Committed by the owner: the load after it re-records the baseline.
    _git(p, "add", "-A")
    _git(p, "commit", "-qm", "mine")
    assert mods.load_project_mods(p)["state"] == "trusted"
    _write(p / "mods" / "nodes" / "mine.py", _node("Mine") + "# more\n")
    assert mods.load_project_mods(p)["state"] == "trusted"


def test_an_owned_project_outside_git_is_not_asked(isolated):  # noqa: F811
    p = isolated / "plain"
    scaffold_project(p)
    projects.mark_owned(p)
    _write(p / "mods" / "nodes" / "mine.py", _node("Mine"))
    assert mods.load_project_mods(p.resolve())["state"] == "trusted"


@needs_git
def test_an_owned_project_without_a_baseline_asks(isolated):  # noqa: F811
    """Owned before this version: no baseline, so mods already there ask."""
    p = isolated / "old"
    scaffold_project(p)
    assert init_new_project_repo(p)
    projects._update_trust(p, owned=True)       # what 4A wrote; no baseline
    _write(p / "mods" / "nodes" / "x.py", _node("X"))
    assert mods.load_project_mods(p.resolve())["state"] == "untrusted"


@needs_git
def test_the_default_project_with_a_remote_asks_for_pulled_mods(
        isolated, tmp_path_factory):  # noqa: F811
    marker = tmp_path_factory.mktemp("out") / "ran"
    decision = projects.resolve_startup_project()
    assert decision.reason == "default" and decision.created
    p = settings.PROJECT_DIR
    assert projects._trust_entries()[projects._key(p)]["owned"] is True
    assert mods.load_project_mods(p)["state"] == "none"
    _collaborator_adds(p, "nodes/evil.py", _node("Evil", marker), isolated)
    assert mods.load_project_mods(p)["state"] == "changed"
    assert not marker.exists()


@needs_git
def test_a_created_project_records_the_baseline_after_its_first_commit(
        isolated):  # noqa: F811
    from app.api.routes_projects import _make_project

    target = isolated / "projects" / "made"
    assert _make_project(target, "made")
    entry = projects._trust_entries()[projects._key(target)]
    head = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    assert entry["mods_head"] == head and entry["mods_sha256"] is None


# ── (2) disk vs loaded ───────────────────────────────────────────────────

def test_a_change_on_disk_is_reported_and_not_loaded(isolated):  # noqa: F811
    p = isolated / "clone"
    scaffold_project(p)
    p = p.resolve()
    _write(p / "mods" / "nodes" / "x.py", _node("X"))
    mods.record_trust(p, mods.walk_mods(p), full=False)
    assert mods.load_project_mods(p)["disk_changed"] is False
    _write(p / "mods" / "nodes" / "x.py", _node("X") + "# pulled\n")
    status = mods.current_status(p)
    assert status["disk_changed"] is True
    assert status["state"] == "trusted"                   # still what runs
    assert registry.get("project:X") is not None          # nothing reloaded
    after = mods.load_project_mods(p)
    assert after["state"] == "changed" and after["disk_changed"] is False


# ── (3) the scan in mods' words ──────────────────────────────────────────

def test_findings_speak_about_mods_not_plugin_manifests():
    net = mods.scan_file("nodes/net.py", b"import json\nimport socket\n")["finding"]
    assert net["kind"] == "import" and net["line"] == 2
    assert net["module"] == "socket" and net["capability"] == "network"
    assert net["message"].startswith("line 2: imports socket: needs network")
    call = mods.scan_file("nodes/c.py", b"x = 1\neval('1')\n")["finding"]
    assert call["kind"] == "refused" and call["line"] == 2
    syntax = mods.scan_file("nodes/s.py", b"def (:\n")["finding"]
    assert syntax["kind"] == "syntax"
    for finding in (net, call, syntax):
        assert "manifest" not in finding["message"]
        assert "plugin" not in finding["message"]


# ── (5) what never loads is marked ───────────────────────────────────────

def test_files_that_never_load_are_marked_and_listed_last(tmp_path):
    p = tmp_path / "p"
    for rel in ("nodes/__init__.py", "__init__.py", "conftest.py", "nodes/evil.pth",
                "nodes/a.py", "presets/x.json", "README.md"):
        _write(p / "mods" / rel, "{}" if rel.endswith(".json") else "x = 1\n")
    scan = mods.scan_mods(p)
    order = [f["path"] for f in scan["files"]]
    assert order[:2] == ["nodes/a.py", "presets/x.json"]
    never = {f["path"] for f in scan["files"] if not f["loads"]}
    assert never == {"nodes/__init__.py", "__init__.py", "conftest.py",
                     "nodes/evil.pth", "README.md"}
    assert all("never loaded" in f["reason"] for f in scan["files"] if not f["loads"])
    assert mods.walk_mods(p).node_files == ["nodes/a.py"]
