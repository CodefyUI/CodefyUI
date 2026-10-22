"""Installing a plugin from a repository on an HTTPS git host other than GitHub.

GitLab, Gitea/Forgejo, Codeberg, Bitbucket, a school's own server: there is
no API to read one manifest from, so the source is cloned with ``git``. The
claim tested here is the same one the local-source tests make -- after the
clone the plugin takes the GitHub road (manifest, gate, consent, staging,
lockfile) -- plus the clone's own rules: nothing typed becomes a git option,
only https, the reviewed commit is the one installed, and a cancel, a
deadline or a clone past the size cap ends it.

The repository is real: a bare repository in ``tmp_path``, reached through
the one seam :mod:`app.core.plugins.git_source` has for it (``_remote_for``
maps the https URL to that directory, and ``file`` joins the allowed
protocols). Skipped with a reason when ``git`` is not on PATH; CI has it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import pytest
from httpx import ASGITransport, AsyncClient

import plugins as plugin_cli
from app.config import settings
from app.core import plugin_loader
from app.core.auth import TOKEN_HEADER, session_token
from app.core.plugins import catalog as catalog_module
from app.core.plugins import deps, flows, git_source, github, lifecycle
from app.core.plugins import inspect as plugin_inspect
from app.core.plugins.errors import (
    GitError,
    GitUnavailable,
    PluginCancelled,
    SourceRefused,
    UnparseableSource,
)
from app.core.plugins.service import PluginService
from app.core.plugins.sources import parse_git_url, parse_source
from app.main import app

GIT = shutil.which("git")
needs_git = pytest.mark.skipif(
    GIT is None, reason="git is not on PATH; installing from a git host needs it")
posix_only = pytest.mark.skipif(
    os.name == "nt", reason="uses a POSIX shell script as a stand-in for git")

BASE_URL = f"http://127.0.0.1:{settings.PORT}"
URL = "https://git.example.test/alice/extras.git"

MANIFEST = dedent("""\
    [plugin]
    id = "hosted-demo"
    name = "Hosted Demo"
    version = "{version}"
    description = "From a self-hosted Gitea."
    schema_version = 1
    """)

NODE = "def triple(x):\n    return 3 * x\n"


# ── guards and roots ───────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def _explode(*_a, **_k):  # pragma: no cover - only runs on a bug
        raise AssertionError("a git-source test tried to reach the network")

    monkeypatch.setattr(github.urllib.request, "urlopen", _explode)


@pytest.fixture(autouse=True)
def no_installer(monkeypatch):
    """The plugin here declares no Python packages; a pip run is a bug.
    (``subprocess.Popen`` itself cannot be fenced off: the clone needs it.)"""
    def _explode(*_a, **_k):  # pragma: no cover - only runs on a bug
        raise AssertionError("a git-source test tried to run an installer")

    monkeypatch.setattr(deps, "install_deps_step", _explode)


@pytest.fixture(autouse=True)
def user_root(tmp_path, monkeypatch) -> Path:
    target = tmp_path / "user" / "plugins"
    target.mkdir(parents=True)
    monkeypatch.setattr(plugin_loader, "plugins_user_root", lambda: target)
    monkeypatch.setattr(plugin_cli, "plugins_user_root", lambda: target)
    monkeypatch.setattr(plugin_cli, "_backend_reload", lambda: False)
    return target


@pytest.fixture(autouse=True)
def builtin_root(tmp_path, monkeypatch) -> Path:
    target = tmp_path / "builtin"
    target.mkdir()
    monkeypatch.setattr(plugin_loader, "plugins_builtin_root", lambda: target)
    monkeypatch.setattr(plugin_cli, "plugins_builtin_root", lambda: target)
    monkeypatch.setattr(catalog_module, "catalog_entries", lambda: {})
    return target


# ── a real repository ──────────────────────────────────────────────────────

#: An empty global config for the test's own git commands, set by ``remote``.
#: A file rather than ``os.devnull``, which is not a path git reads on Windows.
_EMPTY_CONFIG: list[str] = []


def _git_env() -> dict[str, str]:
    """The test's own git, free of the developer's global configuration."""
    env = dict(os.environ)
    env.update(GIT_CONFIG_GLOBAL=_EMPTY_CONFIG[-1], GIT_CONFIG_NOSYSTEM="1",
               GIT_AUTHOR_NAME="Teacher", GIT_AUTHOR_EMAIL="t@example.invalid",
               GIT_COMMITTER_NAME="Teacher", GIT_COMMITTER_EMAIL="t@example.invalid")
    return env


def _run_git(*args: str, cwd: Path) -> str:
    done = subprocess.run([GIT, *args], cwd=cwd, env=_git_env(), check=True,
                          capture_output=True, text=True)
    return done.stdout.strip()


@dataclass
class Remote:
    work: Path
    bare: Path

    def commit(self, version: str, *, extra: dict[str, str] | None = None,
               manifest: str | None = None) -> str:
        (self.work / "cdui.plugin.toml").write_text(
            manifest or MANIFEST.format(version=version), encoding="utf-8")
        for name, text in (extra or {}).items():
            path = self.work / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        _run_git("add", "-A", cwd=self.work)
        _run_git("commit", "-q", "-m", f"v{version}", cwd=self.work)
        _run_git("push", "-q", str(self.bare), "HEAD:main", "--tags", cwd=self.work)
        return _run_git("rev-parse", "HEAD", cwd=self.work)


@pytest.fixture
def remote(tmp_path, monkeypatch) -> Remote:
    if GIT is None:
        pytest.skip("git is not on PATH; installing from a git host needs it")
    config = tmp_path / "gitconfig"
    config.write_text("", encoding="utf-8")
    _EMPTY_CONFIG[:] = [str(config)]
    work = tmp_path / "work"
    (work / "nodes").mkdir(parents=True)
    (work / "nodes" / "triple.py").write_text(NODE, encoding="utf-8")
    _run_git("init", "-q", "-b", "main", cwd=work)
    bare = tmp_path / "remote" / "extras.git"
    bare.parent.mkdir()
    _run_git("init", "-q", "--bare", "-b", "main", str(bare), cwd=tmp_path)
    repo = Remote(work=work, bare=bare)
    repo.first = repo.commit("1.0.0")
    _run_git("tag", "v1.0.0", cwd=work)
    _run_git("push", "-q", str(bare), "--tags", cwd=work)
    monkeypatch.setattr(git_source, "_remote_for", lambda url: bare.as_uri())
    monkeypatch.setattr(git_source, "ALLOWED_PROTOCOLS", ("https", "file"))
    return repo


def _entry(plugin_id: str = "hosted-demo") -> dict | None:
    return plugin_loader.load_lockfile().get("plugins", {}).get(plugin_id)


def _steps(events: list[dict]) -> list[str]:
    return [e["step"] for e in events if e["type"] == "step_started"]


def _install(spec: str, **decisions):
    found = plugin_inspect.inspect_source(spec, lockfile=plugin_loader.load_lockfile())
    plan = flows.plan_from_inspection(found, **decisions)
    events: list[dict] = []
    outcome = flows.install_plugin_live(plan, emit=events.append,
                                        cancel_check=lambda: False)
    return outcome, events


# ── parse_source ───────────────────────────────────────────────────────────

@pytest.mark.parametrize(("spec", "url", "ref"), [
    ("https://gitlab.com/alice/extras", "https://gitlab.com/alice/extras", ""),
    ("https://gitlab.com/alice/extras.git", "https://gitlab.com/alice/extras.git", ""),
    ("https://gitlab.com/alice/extras/", "https://gitlab.com/alice/extras", ""),
    ("https://codeberg.org/alice/extras@v1.2", "https://codeberg.org/alice/extras", "v1.2"),
    ("https://gitlab.com/school/ml/extras@feat/x",
     "https://gitlab.com/school/ml/extras", "feat/x"),
    ("https://git.school.edu:3000/alice/extras.git@main",
     "https://git.school.edu:3000/alice/extras.git", "main"),
    ("https://bitbucket.org/alice/extras@0123abc", "https://bitbucket.org/alice/extras",
     "0123abc"),
    ("HTTPS://gitea.example/alice/extras", "HTTPS://gitea.example/alice/extras", ""),
])
def test_an_https_url_on_another_host_is_a_git_source(spec, url, ref):
    assert parse_source(spec, catalog={"plugins": {}}) == ("git", url, "", ref)


def test_github_keeps_its_own_road():
    assert parse_source("https://github.com/alice/extras",
                        catalog={"plugins": {}}).kind == "github"
    # A deeper github.com URL is still not a repository; it does not fall
    # through to a clone.
    with pytest.raises(UnparseableSource):
        parse_source("https://github.com/alice/extras/tree/main",
                     catalog={"plugins": {}})


@pytest.mark.parametrize("spec", [
    "http://gitlab.com/alice/extras",              # https only
    "ssh://git@gitlab.com/alice/extras.git",       # ssh
    "git@gitlab.com:alice/extras.git",             # scp-like
    "https://token@gitlab.com/alice/extras",       # userinfo
    "https://user:pw@gitlab.com/alice/extras",
    "https://gitlab.com/alice",                    # one segment
    "https://gitlab.com/alice/../bob/extras",      # a segment that walks
    "https://gitlab.com/./alice/extras",
    "https://gitlab.com/alice/extras@../x",        # a ref that walks
    "https://gitlab.com/alice/extras@--upload-pack=x",  # an option as a ref
    "https://gitlab.com/alice/extras?ref=main",    # a query
    "https://gitlab.com/alice/extras#main",
    "https://-gitlab.com/alice/extras",            # not a host name
    "https://gitlab.com/alice/extras@a b",
])
def test_what_is_not_an_https_repository_is_refused(spec):
    with pytest.raises(UnparseableSource):
        parse_source(spec, catalog={"plugins": {}})
    assert parse_git_url(spec) is None


def test_a_file_url_is_a_local_source_and_never_reaches_git():
    assert parse_source("file:///srv/repo", catalog={"plugins": {}}).kind == "local"


def test_the_cli_parser_passes_the_git_kind_through():
    assert plugin_cli.parse_source("https://codeberg.org/a/b@v1") == (
        "git", "https://codeberg.org/a/b", "", "v1")


# ── the clone ──────────────────────────────────────────────────────────────

@needs_git
def test_a_clone_of_the_default_branch_returns_its_commit(remote, tmp_path):
    sha = git_source.clone(URL, "", tmp_path / "c")
    assert sha == remote.first
    assert (tmp_path / "c" / "cdui.plugin.toml").exists()


@needs_git
def test_a_clone_at_a_tag_or_a_branch(remote, tmp_path):
    second = remote.commit("2.0.0")
    assert git_source.clone(URL, "v1.0.0", tmp_path / "tag") == remote.first
    assert git_source.clone(URL, "main", tmp_path / "branch") == second


@needs_git
def test_a_clone_at_a_commit_falls_back_to_a_checkout(remote, tmp_path):
    remote.commit("2.0.0")
    assert git_source.clone(URL, remote.first, tmp_path / "full") == remote.first
    assert git_source.clone(URL, remote.first[:10], tmp_path / "short") == remote.first


@needs_git
def test_a_ref_the_repository_does_not_have_is_not_found(remote, tmp_path):
    with pytest.raises(GitError) as excinfo:
        git_source.clone(URL, "no-such-branch", tmp_path / "c")
    assert excinfo.value.reason == "not_found"
    with pytest.raises(GitError) as excinfo:
        git_source.clone(URL, "deadbeefdeadbeef", tmp_path / "c2")
    assert excinfo.value.reason == "not_found"


@needs_git
def test_a_repository_that_is_not_there_is_not_found(remote, tmp_path, monkeypatch):
    monkeypatch.setattr(git_source, "_remote_for",
                        lambda url: (tmp_path / "missing.git").as_uri())
    with pytest.raises(GitError) as excinfo:
        git_source.clone(URL, "", tmp_path / "c")
    assert excinfo.value.reason == "not_found"
    assert excinfo.value.hint


@needs_git
def test_the_reviewed_commit_is_checked_out_after_the_branch_moved(remote, tmp_path):
    second = remote.commit("2.0.0")
    sha = git_source.clone(URL, "main", tmp_path / "c", want_sha=remote.first)
    assert sha == remote.first
    assert 'version = "1.0.0"' in (tmp_path / "c" / "cdui.plugin.toml").read_text()
    assert sha != second


@needs_git
def test_a_reviewed_commit_that_cannot_be_fetched_is_refused(remote, tmp_path):
    with pytest.raises(SourceRefused) as excinfo:
        git_source.clone(URL, "main", tmp_path / "c", want_sha="1" * 40)
    assert excinfo.value.reason == "changed"


@needs_git
def test_only_the_https_protocol_is_allowed_by_default(remote, tmp_path, monkeypatch):
    """The seam maps the URL to ``file://``; without ``file`` in the allowed
    list, git itself refuses to use it -- which is what stops a redirect or
    a submodule from moving a real clone off https."""
    monkeypatch.setattr(git_source, "ALLOWED_PROTOCOLS", ("https",))
    with pytest.raises(GitError):
        git_source.clone(URL, "", tmp_path / "c")


def test_a_url_that_is_not_https_never_reaches_git(tmp_path, monkeypatch):
    monkeypatch.setattr(git_source, "git_executable", lambda: "/bin/false")
    for url, ref in (("http://gitlab.com/a/b", ""), ("https://gitlab.com/a/b", "-x")):
        with pytest.raises(GitError):
            git_source.clone(url, ref, tmp_path / "c")
    with pytest.raises(GitError):
        git_source.clone("https://gitlab.com/a/b", "", tmp_path / "c", want_sha="HEAD")


def test_no_git_is_a_structured_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(git_source, "git_executable", lambda: None)
    with pytest.raises(GitUnavailable) as excinfo:
        git_source.clone(URL, "", tmp_path / "c")
    assert "git-scm.com" in excinfo.value.hint


def _fake_git(tmp_path: Path, body: str) -> str:
    """A stand-in for git: a shell script whose last argument is the clone
    destination."""
    script = tmp_path / "fakegit"
    script.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


@posix_only
def test_a_clone_past_its_deadline_is_killed(tmp_path, monkeypatch):
    monkeypatch.setattr(git_source, "git_executable",
                        lambda: _fake_git(tmp_path, "sleep 30"))
    monkeypatch.setattr(git_source, "CLONE_TIMEOUT_S", 0.3)
    started = time.monotonic()
    with pytest.raises(GitError) as excinfo:
        git_source.clone(URL, "", tmp_path / "c")
    assert excinfo.value.reason == "timeout"
    assert time.monotonic() - started < 10


@posix_only
def test_a_clone_can_be_cancelled(tmp_path, monkeypatch):
    monkeypatch.setattr(git_source, "git_executable",
                        lambda: _fake_git(tmp_path, "sleep 30"))
    with pytest.raises(PluginCancelled):
        git_source.clone(URL, "", tmp_path / "c", cancel_check=lambda: True)


@posix_only
def test_a_clone_past_the_size_cap_is_killed(tmp_path, monkeypatch):
    body = ('for last; do :; done\nmkdir -p "$last"\n'
            'head -c 4096 /dev/zero > "$last/big"\nsleep 30')
    monkeypatch.setattr(git_source, "git_executable", lambda: _fake_git(tmp_path, body))
    monkeypatch.setattr(github, "MAX_EXTRACTED_BYTES", 1024)
    monkeypatch.setattr(git_source, "SIZE_CHECK_S", 0.0)
    with pytest.raises(SourceRefused) as excinfo:
        git_source.clone(URL, "", tmp_path / "c")
    assert excinfo.value.reason == "too_large"


@posix_only
def test_a_clone_whose_commit_cannot_be_read_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(git_source, "git_executable",
                        lambda: _fake_git(tmp_path, "exit 0"))
    with pytest.raises(GitError) as excinfo:
        git_source.clone(URL, "", tmp_path / "c")
    assert excinfo.value.reason == "failed"


@posix_only
def test_any_other_clone_failure_quotes_git(tmp_path, monkeypatch):
    monkeypatch.setattr(git_source, "git_executable", lambda: _fake_git(
        tmp_path, "echo 'fatal: the remote end hung up unexpectedly' >&2; exit 128"))
    with pytest.raises(GitError) as excinfo:
        git_source.clone(URL, "", tmp_path / "c")
    assert excinfo.value.reason == "failed"
    assert "hung up" in excinfo.value.hint


def test_the_child_git_is_not_pointed_elsewhere(monkeypatch):
    monkeypatch.setenv("GIT_DIR", "/somewhere/else")
    monkeypatch.setenv("GIT_ASKPASS", "/bin/echo")
    env = git_source._env()
    assert "GIT_DIR" not in env and "GIT_ASKPASS" not in env
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    args = git_source._config_args()
    assert args[:2] == ["-c", "protocol.allow=never"]
    assert "core.symlinks=false" in args


# ── inspect and install ────────────────────────────────────────────────────

@needs_git
def test_inspecting_a_git_source_pins_the_commit(remote):
    found = plugin_inspect.inspect_source(URL + "@main", lockfile={})
    assert found.kind == "git"
    assert found.plugin_id == "hosted-demo"
    assert found.url == URL and found.ref == "main"
    assert found.sha == remote.first
    assert found.source == URL + "@main"
    assert found.consent_required is True
    assert found.owner is None and found.path is None


@needs_git
def test_a_git_source_installs_through_every_step(remote, user_root):
    outcome, events = _install(URL)
    assert _steps(events) == ["resolve", "download", "extract", "verify", "stage", "lock"]
    assert outcome.sha == remote.first
    installed = user_root / "hosted-demo"
    assert (installed / "nodes" / "triple.py").read_text() == NODE
    assert not (installed / ".git").exists()
    entry = _entry()
    assert entry["source_kind"] == "git_url"
    assert entry["url"] == URL
    assert entry["ref"] == ""
    assert entry["sha"] == remote.first
    assert entry["source"] == URL


@needs_git
def test_the_install_takes_the_reviewed_commit_not_the_moved_branch(remote, user_root):
    found = plugin_inspect.inspect_source(URL + "@main", lockfile={})
    remote.commit("2.0.0")
    plan = flows.plan_from_inspection(found)
    outcome = flows.install_plugin_live(plan, emit=lambda e: None,
                                        cancel_check=lambda: False)
    assert outcome.sha == remote.first
    assert _entry()["manifest"]["version"] == "1.0.0"


@needs_git
def test_the_gate_scans_a_git_source(remote, user_root):
    from app.core.plugins.errors import PluginInstallError

    remote.commit("1.1.0", extra={"nodes/evil.py": "import subprocess\n"})
    with pytest.raises(PluginInstallError, match="security scan"):
        _install(URL)
    assert not (user_root / "hosted-demo").exists()


@needs_git
def test_a_git_plan_with_no_commit_is_refused(remote):
    from app.core.plugins.errors import PluginInstallError

    found = plugin_inspect.inspect_source(URL, lockfile={})
    plan = flows.InstallPlan(**{**flows.plan_from_inspection(found).__dict__, "sha": None})
    with pytest.raises(PluginInstallError, match="names no commit"):
        flows.install_plugin_live(plan, emit=lambda e: None, cancel_check=lambda: False)


@needs_git
def test_an_update_compares_commits_like_the_github_road(remote):
    _install(URL)
    lockfile = plugin_loader.load_lockfile()
    assert plugin_inspect.updatable_entry("hosted-demo", lockfile=lockfile)
    assert plugin_inspect.inspect_installed("hosted-demo", lockfile=lockfile).up_to_date

    second = remote.commit("2.0.0")
    found = plugin_inspect.inspect_installed("hosted-demo", lockfile=lockfile)
    assert found.up_to_date is False
    assert found.sha == second
    assert found.mode == "update"


@needs_git
def test_an_update_into_another_id_is_refused(remote):
    from app.core.plugins.errors import NotUpdatable

    _install(URL)
    remote.commit("2.0.0", manifest=MANIFEST.format(version="2.0.0").replace(
        'id = "hosted-demo"', 'id = "someone-else"'))
    with pytest.raises(NotUpdatable, match="now installs as"):
        plugin_inspect.inspect_installed("hosted-demo",
                                         lockfile=plugin_loader.load_lockfile())


def test_a_record_with_no_usable_url_is_not_updatable():
    from app.core.plugins.errors import NotUpdatable

    lockfile = {"plugins": {"x": {"source_kind": "git_url", "url": "ssh://h/a/b"}}}
    with pytest.raises(NotUpdatable, match="Cannot tell"):
        plugin_inspect.inspect_installed("x", lockfile=lockfile)


@needs_git
def test_uninstall_deletes_the_clone_and_offers_the_url_again(remote, user_root):
    _install(URL + "@v1.0.0")
    outcome = lifecycle.uninstall_plugin("hosted-demo", builtin_ids=(),
                                         catalog={"plugins": {}})
    assert outcome.files_removed is True
    assert not (user_root / "hosted-demo").exists()
    assert outcome.reinstall_hint == f'cdui plugin install "{URL}@v1.0.0"'


def test_a_git_record_that_no_longer_parses_gets_no_reinstall_hint():
    entry = {"source_kind": "git_url", "url": "ssh://h/a/b", "ref": ""}
    assert lifecycle.reinstall_command("x", entry, catalog={"plugins": {}},
                                       builtin_ids=()) is None
    assert lifecycle.reinstall_command("x", {"source_kind": "git_url"},
                                       catalog={"plugins": {}}, builtin_ids=()) is None


# ── the REST routes ────────────────────────────────────────────────────────

@pytest.fixture
async def client(monkeypatch):
    monkeypatch.setattr(settings, "HOST", "127.0.0.1")
    service = PluginService(reload=lambda: {})
    previous = getattr(app.state, "plugin_service", None)
    app.state.plugin_service = service
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url=BASE_URL,
            headers={TOKEN_HEADER: session_token()},
        ) as http:
            yield http
    finally:
        await service.shutdown()
        if previous is None:
            if hasattr(app.state, "plugin_service"):
                delattr(app.state, "plugin_service")
        else:
            app.state.plugin_service = previous


async def _drain(client, job_id: str) -> tuple[list[dict], str]:
    events: list[dict] = []
    cursor, status = 0, "running"
    deadline = time.monotonic() + 30
    while status == "running":
        assert time.monotonic() < deadline, "job never finished"
        response = await client.get(f"/api/plugins/jobs/{job_id}/events",
                                    params={"cursor": cursor, "wait": 1.0})
        body = response.json()
        events.extend(body["events"])
        cursor, status = body["cursor"], body["status"]
    return events, status


@needs_git
async def test_the_panel_installs_and_updates_a_git_source(client, remote):
    response = await client.post("/api/plugins/inspect", json={"source": URL})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "git"
    assert body["url"] == URL and body["sha"] == remote.first

    response = await client.post("/api/plugins/install",
                                 json={"inspection_id": body["inspection_id"]})
    assert response.status_code == 202, response.text
    events, status = await _drain(client, response.json()["job_id"])
    assert status == "done", events

    rows = {e["id"]: e for e in (await client.get("/api/plugins/catalog")).json()["entries"]}
    row = rows["hosted-demo"]
    assert row["source_kind"] == "git_url"
    assert row["url"] == URL and row["repo"] is None
    assert row["consent_required"] is True

    response = await client.post("/api/plugins/hosted-demo/update")
    assert response.json() == {"status": "up_to_date", "sha": remote.first}

    second = remote.commit("2.0.0")
    response = await client.post("/api/plugins/hosted-demo/update")
    assert response.status_code == 202, response.text
    _events, status = await _drain(client, response.json()["job_id"])
    assert status == "done"
    assert _entry()["sha"] == second


@needs_git
async def test_an_update_that_asks_for_more_is_a_consent_screen(client, remote):
    _install(URL)
    remote.commit("2.0.0", manifest=MANIFEST.format(version="2.0.0")
                  + '\n[security]\ncapabilities = ["network"]\n')
    response = await client.post("/api/plugins/hosted-demo/update")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "needs_consent"
    assert body["capabilities_added"] == ["network"]


@needs_git
async def test_a_repository_that_is_not_there_is_a_404(client, remote, tmp_path, monkeypatch):
    monkeypatch.setattr(git_source, "_remote_for",
                        lambda url: (tmp_path / "missing.git").as_uri())
    response = await client.post("/api/plugins/inspect", json={"source": URL})
    assert response.status_code == 404
    assert response.json()["detail"] == {"code": "git_not_found"}


async def test_no_git_on_the_server_is_a_coded_refusal(client, monkeypatch):
    monkeypatch.setattr(git_source, "git_executable", lambda: None)
    response = await client.post("/api/plugins/inspect", json={"source": URL})
    assert response.status_code == 400
    assert response.json()["detail"] == {"code": "git_unavailable"}


@pytest.mark.parametrize(("reason", "status", "code"), [
    ("timeout", 504, "git_timeout"), ("failed", 502, "git_failed"),
])
async def test_the_other_clone_failures_are_coded(client, monkeypatch, reason, status, code):
    def boom(*_a, **_k):
        raise GitError("nope", reason=reason)

    monkeypatch.setattr(git_source, "clone", boom)
    response = await client.post("/api/plugins/inspect", json={"source": URL})
    assert response.status_code == status
    assert response.json()["detail"] == {"code": code}


async def test_an_update_whose_clone_fails_is_coded_too(client, monkeypatch):
    lockfile = plugin_loader.load_lockfile()
    lockfile.setdefault("plugins", {})["hosted-demo"] = {
        "source_kind": "git_url", "source": URL, "url": URL, "ref": "",
        "sha": "0" * 40, "manifest": {"id": "hosted-demo"}, "capabilities": [],
        "trusted_modules": [], "enabled": True,
    }
    plugin_loader.save_lockfile(lockfile)
    monkeypatch.setattr(git_source, "git_executable", lambda: None)
    response = await client.post("/api/plugins/hosted-demo/update")
    assert response.status_code == 400
    assert response.json()["detail"] == {"code": "git_unavailable"}

    def too_big(*_a, **_k):
        raise SourceRefused("big", reason="too_large")

    monkeypatch.setattr(git_source, "clone", too_big)
    response = await client.post("/api/plugins/hosted-demo/update")
    assert response.status_code == 400
    assert response.json()["detail"]["reason"] == "too_large"


# ── the CLI ────────────────────────────────────────────────────────────────

def _cli_args(*sources: str, **overrides) -> argparse.Namespace:
    base = dict(source=list(sources), force=False, no_confirm=True,
                trust_author=False, accept_capabilities=False)
    base.update(overrides)
    return argparse.Namespace(**base)


@needs_git
def test_cdui_plugin_install_takes_a_git_url(remote, user_root, capsys, monkeypatch):
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    assert plugin_cli.cmd_install(_cli_args(URL + "@v1.0.0")) == 0
    out = capsys.readouterr().out
    assert f"Source: {URL}" in out
    assert f"Installed: hosted-demo ({remote.first[:7]})" in out
    assert _entry()["ref"] == "v1.0.0"


def test_cdui_plugin_install_says_git_is_missing(capsys, monkeypatch):
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    monkeypatch.setattr(git_source, "git_executable", lambda: None)
    assert plugin_cli.cmd_install(_cli_args(URL)) == 1
    captured = capsys.readouterr()
    assert "git-scm.com" in captured.out + captured.err


@needs_git
def test_cdui_plugin_install_reports_a_missing_repository(
        remote, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    monkeypatch.setattr(git_source, "_remote_for",
                        lambda url: (tmp_path / "missing.git").as_uri())
    assert plugin_cli.cmd_install(_cli_args(URL)) == 1
    captured = capsys.readouterr()
    assert "has no repository or ref" in captured.out + captured.err


@needs_git
def test_cdui_plugin_update_follows_a_git_source(remote, capsys, monkeypatch):
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    plugin_cli.cmd_install(_cli_args(URL))
    capsys.readouterr()
    assert plugin_cli.cmd_update(argparse.Namespace(plugin_id="hosted-demo")) == 0
    assert "already up to date" in capsys.readouterr().out

    second = remote.commit("2.0.0")
    assert plugin_cli.cmd_update(argparse.Namespace(plugin_id="hosted-demo")) == 0
    out = capsys.readouterr().out
    assert f"Updating hosted-demo: {remote.first[:7]}" in out
    assert "updated 1" in out
    assert _entry()["sha"] == second


def test_cdui_plugin_update_reports_git_missing_and_a_bad_record(capsys, monkeypatch):
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    lockfile = plugin_loader.load_lockfile()
    lockfile.setdefault("plugins", {})["hosted-demo"] = {
        "source_kind": "git_url", "url": URL, "ref": "", "sha": "0" * 40}
    plugin_loader.save_lockfile(lockfile)
    monkeypatch.setattr(git_source, "git_executable", lambda: None)
    assert plugin_cli.cmd_update(argparse.Namespace(plugin_id="hosted-demo")) == 1
    assert "git-scm.com" in "".join(capsys.readouterr())

    lockfile["plugins"]["hosted-demo"]["url"] = "ssh://h/a/b"
    plugin_loader.save_lockfile(lockfile)
    assert plugin_cli.cmd_update(argparse.Namespace(plugin_id="hosted-demo")) == 1
    assert "Cannot tell which repository" in "".join(capsys.readouterr())


@needs_git
def test_cdui_plugin_info_reads_a_git_source(remote, capsys, monkeypatch):
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    assert plugin_cli.cmd_info(argparse.Namespace(source_or_id=URL)) == 0
    assert "Hosted Demo" in capsys.readouterr().out
    assert _entry() is None


def test_cdui_plugin_info_says_git_is_missing(capsys, monkeypatch):
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    monkeypatch.setattr(git_source, "git_executable", lambda: None)
    assert plugin_cli.cmd_info(argparse.Namespace(source_or_id=URL)) == 1


@needs_git
def test_project_restore_reinstalls_a_git_pin(remote, tmp_path, monkeypatch, capsys):
    import project as project_cli

    monkeypatch.setenv("CODEFYUI_LANG", "en")
    plugin_cli.cmd_install(_cli_args(URL))
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / project_cli.MANIFEST_FILENAME).write_text('[project]\nname = "p"\n',
                                                     encoding="utf-8")
    monkeypatch.setattr(project_cli, "load_lockfile", plugin_loader.load_lockfile)
    assert project_cli.cmd_freeze(argparse.Namespace(dir=str(proj))) == 0
    assert URL in (proj / project_cli.MANIFEST_FILENAME).read_text()

    lifecycle.uninstall_plugin("hosted-demo", builtin_ids=(), catalog={"plugins": {}})
    remote.commit("2.0.0")
    assert project_cli.cmd_restore(argparse.Namespace(dir=str(proj))) == 0
    assert _entry()["sha"] == remote.first
