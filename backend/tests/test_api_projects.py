"""Phase 4A: /api/projects -- open, create, clone, trust, recents, browse."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core import projects
from app.core.auth import TOKEN_HEADER, session_token
from app.core.plugins import git_source
from app.core.project import PROJECT_MANIFEST, scaffold_project
from app.main import app

from .test_project_switch import FakeRuns
from .test_projects_startup import isolated  # noqa: F401  (fixture)

BASE_URL = f"http://127.0.0.1:{settings.PORT}"
GIT = shutil.which("git")


@pytest.fixture
async def http(isolated):  # noqa: F811
    saved = getattr(app.state, "run_service", None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL,
                           headers={TOKEN_HEADER: session_token()}) as client:
        yield client
    app.state.run_service = saved


def _project(path: Path, env: str | None = None) -> Path:
    scaffold_project(path)
    if env is not None:
        (path / ".env").write_text(env, encoding="utf-8")
    return path.resolve()


async def test_listing_in_flat_mode(http):
    body = (await http.get("/api/projects")).json()
    assert body["current"] is None
    assert body["project_id"] == "flat"
    assert body["recent"] == []
    assert body["env"]["names"] == []
    assert body["projects_root"].endswith("projects")
    assert body["browse_allowed"] in (True, False)


async def test_open_switches_and_lists_it_as_current(http, isolated):  # noqa: F811
    target = _project(isolated / "elsewhere" / "alpha")
    resp = await http.post("/api/projects/open", json={"path": str(target)})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["project"]["path"] == str(target)
    assert body["project_id"] == projects.project_id(target)
    assert body["unchanged"] is False
    assert settings.GRAPHS_DIR == target / "graphs"

    listing = (await http.get("/api/projects")).json()
    assert listing["current"]["path"] == str(target)
    assert [r["path"] for r in listing["recent"]] == [str(target)]
    assert listing["recent"][0]["missing"] is False

    again = await http.post("/api/projects/open", json={"path": str(target)})
    assert again.json()["unchanged"] is True


@pytest.mark.parametrize("make, status, code", [
    (lambda root: str(root / "nope"), 400, "not_found"),
    (lambda root: str(_plain(root / "plain")), 400, "not_a_project"),
    (lambda root: "/etc", 403, "path_refused"),
])
async def test_open_refusals(http, isolated, make, status, code):  # noqa: F811
    resp = await http.post("/api/projects/open", json={"path": make(isolated)})
    assert resp.status_code == status, resp.text
    assert resp.json()["detail"]["code"] == code
    assert settings.PROJECT_DIR is None


def _plain(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "notes.txt").write_text("x", encoding="utf-8")
    return path


async def test_open_is_refused_while_a_run_is_active(http, isolated):  # noqa: F811
    target = _project(isolated / "alpha")
    app.state.run_service = FakeRuns(active=["r1"], queued=["r2"])
    resp = await http.post("/api/projects/open", json={"path": str(target)})
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["code"] == "runs_active"
    assert "1 run is running and 1 run is queued" in detail["message"]
    assert settings.PROJECT_DIR is None


async def test_an_untrusted_env_waits_for_consent(http, isolated, monkeypatch):  # noqa: F811
    monkeypatch.delenv("P4_API_TOKEN_NAME", raising=False)
    target = _project(isolated / "cloned-by-hand",
                      env="P4_API_TOKEN_NAME=secret-value\nLD_PRELOAD=/x.so\n"
                          "HF_HOME=/cache\n")
    body = (await http.post("/api/projects/open",
                            json={"path": str(target)})).json()
    env = body["env"]
    assert env["state"] == "untrusted" and env["needs_consent"] is True
    assert env["names"] == ["HF_HOME", "LD_PRELOAD", "P4_API_TOKEN_NAME"]
    assert env["blocked"] == ["LD_PRELOAD"]
    assert "secret-value" not in str(body)
    assert "P4_API_TOKEN_NAME" not in os.environ

    stale = await http.post("/api/projects/trust", json={"sha256": "0" * 64})
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "env_changed"

    trusted = await http.post("/api/projects/trust",
                              json={"sha256": env["sha256"]})
    assert trusted.status_code == 200, trusted.text
    after = trusted.json()
    assert after["env"]["state"] == "trusted"
    assert os.environ["P4_API_TOKEN_NAME"] == "secret-value"
    assert "LD_PRELOAD" not in after["env"]["applied"]
    assert after["env_restart_recommended"] == ["HF_HOME"]

    # Changed contents need consent again.
    (target / ".env").write_text("P4_API_TOKEN_NAME=other\n", encoding="utf-8")
    projects.activate(target)
    assert projects.current_env.state == "changed"


async def test_create_scaffolds_inits_and_opens(http, isolated):  # noqa: F811
    resp = await http.post("/api/projects/create", json={"name": "mnist-lab"})
    assert resp.status_code == 200, resp.text
    target = (isolated / "projects" / "mnist-lab").resolve()
    assert resp.json()["created"] == str(target)
    assert (target / PROJECT_MANIFEST).is_file()
    if GIT:
        assert (target / ".git").is_dir() and resp.json()["git"] is True
    assert settings.PROJECT_DIR == target
    # Created here, so trusted whatever its .env later holds.
    (target / ".env").write_text("P4_OWNED=1\n", encoding="utf-8")
    assert projects.env_trust_state(target) == "trusted"


@pytest.mark.parametrize("name", [
    "has space", ".hidden", "CON", "lpt1.txt", "a/b", "..", "trailing.", "",
    "x" * 65, "é",
])
async def test_create_refuses_a_bad_name(http, name):
    resp = await http.post("/api/projects/create", json={"name": name})
    assert resp.status_code == 400, (name, resp.text)
    assert resp.json()["detail"]["code"] == "invalid_name"


async def test_create_refuses_a_non_empty_folder(http, isolated):  # noqa: F811
    taken = isolated / "projects" / "taken"
    taken.mkdir(parents=True)
    (taken / "keep.txt").write_text("mine", encoding="utf-8")
    resp = await http.post("/api/projects/create", json={"name": "taken"})
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "exists"
    assert sorted(p.name for p in taken.iterdir()) == ["keep.txt"]
    # An empty folder of that name is fine.
    (isolated / "projects" / "empty").mkdir()
    resp = await http.post("/api/projects/create", json={"name": "empty"})
    assert resp.status_code == 200, resp.text


async def test_create_while_busy_makes_nothing(http, isolated):  # noqa: F811
    app.state.run_service = FakeRuns(active=["r1"])
    resp = await http.post("/api/projects/create", json={"name": "later"})
    assert resp.status_code == 409
    assert not (isolated / "projects" / "later").exists()


async def test_recent_entries_can_be_removed(http, isolated):  # noqa: F811
    target = _project(isolated / "alpha")
    await http.post("/api/projects/open", json={"path": str(target)})
    resp = await http.request("DELETE", "/api/projects/recent",
                              json={"path": str(target)})
    assert resp.status_code == 200
    assert projects.load_registry()["recent"] == []
    assert target.is_dir()
    missing = await http.request("DELETE", "/api/projects/recent",
                                 json={"path": str(target)})
    assert missing.status_code == 404


async def test_browse_lists_folders_only(http, isolated, monkeypatch):  # noqa: F811
    from app.api import routes_projects

    monkeypatch.setattr(routes_projects, "_browse_allowed", lambda request: True)
    root = isolated / "projects"
    _project(root / "alpha")
    (root / "plain").mkdir()
    (root / ".hidden").mkdir()
    (root / "file.txt").write_text("x", encoding="utf-8")
    body = (await http.get("/api/projects/browse")).json()
    assert body["dir"] == str(root.resolve())
    assert [(e["name"], e["is_project"]) for e in body["entries"]] == [
        ("alpha", True), ("plain", False)]
    shown = (await http.get("/api/projects/browse",
                            params={"hidden": "true"})).json()
    assert ".hidden" in [e["name"] for e in shown["entries"]]
    inside = (await http.get("/api/projects/browse",
                             params={"dir": str(root / "alpha")})).json()
    assert inside["is_project"] is True
    assert "codefyui.project.toml" not in str(inside["entries"])


async def test_browse_is_refused_off_loopback(http, monkeypatch):
    from app.api import routes_projects

    monkeypatch.setattr(routes_projects, "bound_to_loopback", lambda: False)
    resp = await http.get("/api/projects/browse")
    assert resp.status_code == 403
    assert resp.json()["detail"]["code"] == "browse_refused"


# ── clone ─────────────────────────────────────────────────────────────────


def _git(*args: str, cwd: Path) -> None:
    env = {**os.environ, "GIT_AUTHOR_NAME": "T",
           "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "T",
           "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    subprocess.run([GIT, *args], cwd=cwd, env=env, check=True,
                   capture_output=True)


def _bare(tmp: Path, monkeypatch, *, project: bool = True,
          lfs: bool = False) -> Path:
    if GIT is None:
        pytest.skip("git is not on PATH")
    work = tmp / "work"
    if project:
        scaffold_project(work)
        (work / "graphs" / "g.json").write_text('{"nodes": [], "edges": []}\n',
                                                encoding="utf-8")
    else:
        work.mkdir()
        (work / "README.md").write_text("not a project\n", encoding="utf-8")
    if lfs:
        (work / ".gitattributes").write_text(
            "*.pt filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")
    _git("init", "-q", "-b", "main", cwd=work)
    _git("add", "-A", cwd=work)
    _git("commit", "-q", "-m", "init", cwd=work)
    bare = tmp / "remote.git"
    _git("init", "-q", "--bare", "-b", "main", str(bare), cwd=tmp)
    _git("push", "-q", str(bare), "HEAD:main", cwd=work)
    monkeypatch.setattr(git_source, "_remote_for", lambda url: bare.as_uri())
    monkeypatch.setattr(git_source, "ALLOWED_PROTOCOLS", ("https", "file"))
    return bare


async def test_clone_opens_the_project_untrusted(http, isolated, monkeypatch):  # noqa: F811
    _bare(isolated, monkeypatch, lfs=True)
    resp = await http.post("/api/projects/clone",
                           json={"url": "https://git.example.test/team/vision-lab.git"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    target = (isolated / "projects" / "vision-lab").resolve()
    assert body["cloned"] == str(target)
    assert body["uses_lfs"] is True
    assert len(body["head"]) == 40
    assert (target / "graphs" / "g.json").is_file()
    assert settings.PROJECT_DIR == target
    # Full history, not shallow: it is a repository to keep working in.
    assert not (target / ".git" / "shallow").exists()
    (target / ".env").write_text("P4_CLONED=1\n", encoding="utf-8")
    assert projects.env_trust_state(target) == "untrusted"


async def test_clone_into_an_existing_folder_is_refused(http, isolated, monkeypatch):  # noqa: F811
    _bare(isolated, monkeypatch)
    existing = isolated / "projects" / "vision-lab"
    existing.mkdir(parents=True)
    resp = await http.post("/api/projects/clone",
                           json={"url": "https://git.example.test/team/vision-lab"})
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "exists"
    assert list(existing.iterdir()) == []


async def test_a_clone_that_is_not_a_project_is_removed(http, isolated, monkeypatch):  # noqa: F811
    _bare(isolated, monkeypatch, project=False)
    resp = await http.post("/api/projects/clone",
                           json={"url": "https://git.example.test/team/notes.git"})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "not_a_project"
    assert not (isolated / "projects" / "notes").exists()
    assert settings.PROJECT_DIR is None


async def test_a_failed_clone_leaves_nothing(http, isolated, monkeypatch):  # noqa: F811
    if GIT is None:
        pytest.skip("git is not on PATH")
    monkeypatch.setattr(git_source, "_remote_for",
                        lambda url: (isolated / "missing.git").as_uri())
    monkeypatch.setattr(git_source, "ALLOWED_PROTOCOLS", ("https", "file"))
    resp = await http.post("/api/projects/clone",
                           json={"url": "https://git.example.test/team/gone.git"})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"].startswith("clone_")
    assert not (isolated / "projects" / "gone").exists()


@pytest.mark.parametrize("url", [
    "http://git.example.test/team/x.git",
    "git@git.example.test:team/x.git",
    "file:///tmp/x.git",
    "https://git.example.test/team/../x.git",
])
async def test_clone_takes_https_only(http, url):
    resp = await http.post("/api/projects/clone", json={"url": url})
    assert resp.status_code == 400, (url, resp.text)
    assert resp.json()["detail"]["code"] == "invalid_url"
