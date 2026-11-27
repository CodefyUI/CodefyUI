"""Phase 4A review round 1: a project the server creates starts as a
repository on ``main`` (or the user's ``init.defaultBranch``) with one
commit holding the starter files, and an identity is never written."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from app.core.project import (
    NEW_PROJECT_COMMIT_MESSAGE,
    init_new_project_repo,
    init_project_repo,
    scaffold_project,
)

from .test_api_projects import http  # noqa: F401  (fixture)
from .test_projects_startup import isolated  # noqa: F401  (fixture)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


@pytest.fixture
def gitconfig(tmp_path, monkeypatch):
    """A global git config of the test's own, empty unless a test writes it."""
    path = tmp_path / "gitconfig"
    path.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(path))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME",
                 "GIT_COMMITTER_EMAIL", "EMAIL"):
        monkeypatch.delenv(name, raising=False)
    return path


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()


def test_no_identity_and_no_default_branch(tmp_path, gitconfig):
    target = tmp_path / "lab"
    scaffold_project(target)
    assert init_new_project_repo(target) is True
    assert _git(target, "branch", "--show-current") == "main"
    assert _git(target, "log", "--format=%s|%an|%ae") == \
        f"{NEW_PROJECT_COMMIT_MESSAGE}|CodefyUI|codefyui@localhost"
    assert _git(target, "status", "--porcelain") == ""
    tracked = set(_git(target, "ls-files").splitlines())
    assert "codefyui.project.toml" in tracked and ".gitignore" in tracked
    # The identity was given for that commit only.
    assert gitconfig.read_text(encoding="utf-8") == ""
    assert "user" not in (target / ".git" / "config").read_text(encoding="utf-8")


def test_the_users_branch_and_identity_are_used(tmp_path, gitconfig):
    gitconfig.write_text("[init]\n\tdefaultBranch = trunk\n"
                         "[user]\n\tname = Ada\n\temail = ada@example.invalid\n",
                         encoding="utf-8")
    target = tmp_path / "lab"
    scaffold_project(target)
    init_new_project_repo(target)
    assert _git(target, "branch", "--show-current") == "trunk"
    assert _git(target, "log", "--format=%an") == "Ada"


def test_cdui_project_init_still_makes_no_commit(tmp_path, gitconfig):
    target = tmp_path / "lab"
    scaffold_project(target)
    init_project_repo(target)
    assert subprocess.run(["git", "rev-parse", "HEAD"], cwd=target,
                          capture_output=True).returncode != 0


async def test_create_starts_clean_on_main(http, isolated, gitconfig):  # noqa: F811
    resp = await http.post("/api/projects/create", json={"name": "fresh"})
    assert resp.status_code == 200, resp.text
    target = Path(resp.json()["created"])
    assert _git(target, "branch", "--show-current") == "main"
    assert _git(target, "log", "--format=%s") == NEW_PROJECT_COMMIT_MESSAGE
    assert _git(target, "status", "--porcelain") == ""


async def test_the_default_project_starts_clean_on_main(isolated, gitconfig):  # noqa: F811
    from app.core import projects

    decision = projects.resolve_startup_project()
    assert decision.created
    assert _git(decision.project, "log", "--format=%s") == NEW_PROJECT_COMMIT_MESSAGE
    assert _git(decision.project, "branch", "--show-current") == "main"


async def test_an_import_commits_only_the_starter_files(http, isolated, gitconfig):  # noqa: F811
    flat = isolated / "flat-graphs"
    flat.mkdir()
    (flat / "g.json").write_text('{"nodes": [], "edges": []}', encoding="utf-8")
    resp = await http.post("/api/projects/import", json={"name": "imported"})
    assert resp.status_code == 200, resp.text
    target = Path(resp.json()["created"])
    assert _git(target, "log", "--format=%s") == NEW_PROJECT_COMMIT_MESSAGE
    assert "graphs/g/index.json" not in _git(target, "ls-files").splitlines()
    assert "graphs/g/index.json" in _git(target, "status", "--porcelain", "-uall")
