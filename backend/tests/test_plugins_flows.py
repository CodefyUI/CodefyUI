"""One plugin install, watched from outside: its steps, refusals and rollback.

Everything here goes through the same three calls the CLI and the server
both make -- ``inspect_*``, then ``plan_from_inspection``, then
``install_plugin_live`` -- because the point of the flow is that there is
only one order of events, and a test that reached past it to a private step
would stop noticing when the order changed.

Nothing touches the network, the disk outside ``tmp_path``, or a real
installer. Three autouse guards make that a failure rather than a habit: a
socket, a ``uv`` process, and the user's own plugin directory are each
replaced with something that fails the test if it is reached. What is left
is fast enough that every refusal below gets its own test.

The events are the contract. Steps are asserted as the sequence of ids the
caller would draw a list from -- ``resolve download extract verify deps
stage lock`` -- rather than by counting events, because that sequence is
what the Plugin Center and ``cdui plugin install`` have to agree about.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

from app.core import plugin_loader
from app.core.packs import runner as packs_runner
from app.core.plugins import flows, github
from app.core.plugins import inspect as plugin_inspect
from app.core.plugins.errors import (
    AlreadyInstalled,
    ConsentRequired,
    PluginCancelled,
    PluginInstallError,
    PluginNeedsRestart,
)


# ── guards ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """An install that reaches a socket fails here, not in somebody's CI."""
    def _explode(*_a, **_k):  # pragma: no cover - only runs on a bug
        raise AssertionError("a test in this file tried to reach the network")

    monkeypatch.setattr(github.urllib.request, "urlopen", _explode)


@pytest.fixture(autouse=True)
def no_installer(monkeypatch):
    """And one that reaches ``uv`` fails too.

    ``run_pip`` is faked per test; this is the floor under that, because a
    dependency step that quietly ran the real installer would edit the venv
    the suite is running in.
    """
    def _explode(*_a, **_k):  # pragma: no cover - only runs on a bug
        raise AssertionError("a test in this file tried to run an installer")

    monkeypatch.setattr(packs_runner.subprocess, "Popen", _explode)


@pytest.fixture(autouse=True)
def user_root(tmp_path, monkeypatch) -> Path:
    """The plugin directory and the lockfile, in a directory of our own.

    Autouse: every test in this file either writes a lockfile entry or
    asserts that it did not, and three plugin directories once had to be
    swept out of a developer's real user data because one test forgot to
    ask for this.
    """
    target = tmp_path / "user" / "plugins"
    target.mkdir(parents=True)
    monkeypatch.setattr(plugin_loader, "plugins_user_root", lambda: target)
    return target


@pytest.fixture(autouse=True)
def builtin_root(tmp_path, monkeypatch) -> Path:
    """The packs that ship with the release, and the catalog beside them.

    Patched even for the repository tests: the reserved-id rule reads
    ``registry.json`` out of this directory, and an empty one is how these
    tests stop depending on which packs this release happens to ship.
    """
    target = tmp_path / "builtin"
    target.mkdir()
    monkeypatch.setattr(plugin_loader, "plugins_builtin_root", lambda: target)
    return target


# ── manifests ──────────────────────────────────────────────────────────────

PLAIN = dedent("""\
    [plugin]
    id = "extras"
    name = "Extras"
    version = "1.2.0"
    description = "A third-party pack."
    schema_version = 1
    """)

WITH_DEPS = dedent("""\
    [plugin]
    id = "extras"
    name = "Extras"
    version = "1.2.0"
    schema_version = 1

    [python_deps]
    tinylib = "1.0.0"
    """)

WITH_CAPABILITY = dedent("""\
    [plugin]
    id = "extras"
    name = "Extras"
    version = "1.2.0"
    schema_version = 1

    [security]
    capabilities = ["network"]
    """)

WITH_MODULES = dedent("""\
    [plugin]
    id = "extras"
    name = "Extras"
    version = "1.2.0"
    schema_version = 1

    [security]
    allowed_modules = ["subprocess"]
    """)

BUILTIN = dedent("""\
    [plugin]
    id = "demo-pack"
    name = "Demo Pack"
    version = "0.1.0"
    description = "Ships with the release."
    schema_version = 1
    """)

BUILTIN_WITH_DEPS = BUILTIN + dedent("""
    [python_deps]
    tinylib = "1.0.0"
    """)


# ── helpers ────────────────────────────────────────────────────────────────

def _tarball_of(files: dict[str, str], dest: Path) -> None:
    """Pack ``{path inside the archive: text}`` into a gzipped tar."""
    with tarfile.open(dest, "w:gz") as tf:
        for rel, text in files.items():
            data = text.encode("utf-8")
            member = tarfile.TarInfo(rel)
            member.size = len(data)
            tf.addfile(member, io.BytesIO(data))


@pytest.fixture
def fake_github(monkeypatch):
    """Serve one commit -- a sha, a manifest and a tarball -- with no network.

    *during_download* is called with the flow's own ``cancel_check`` and
    ``progress`` before the tarball is written, which is how a test says
    "the user pressed Stop at 40%" or "this client reports every chunk".
    """
    def _make(
        files: dict[str, str],
        *,
        sha: str = "a" * 40,
        during_download=None,
    ) -> str:
        def _download(owner, repo, commit, dest, *, cancel_check=None, progress=None):
            if during_download is not None:
                during_download(cancel_check=cancel_check, progress=progress)
            _tarball_of(
                {f"{repo}-main/{rel}": text for rel, text in files.items()}, dest
            )

        monkeypatch.setattr(github, "resolve_sha", lambda o, r, ref: sha)
        monkeypatch.setattr(
            github, "fetch_manifest_text",
            lambda o, r, s: files["cdui.plugin.toml"],
        )
        monkeypatch.setattr(github, "download_tarball", _download)
        return sha

    return _make


@pytest.fixture
def fake_pip(monkeypatch):
    """Answer for ``uv`` and record what it was asked to do.

    The constraints file is read INSIDE the call on purpose: it lives in a
    per-job temporary directory that is gone by the time the step returns,
    so a test that looked afterwards would be asserting about a path that no
    longer exists.
    """
    calls: list[dict[str, Any]] = []

    def _make(*, returncode: int = 0, output: tuple[str, ...] = ()):
        def _run_pip(specs, *, constraints_path, emit, cancel_check, cwd, tail=None):
            calls.append({
                "specs": list(specs),
                "constraints_path": constraints_path,
                "constraints_exists": constraints_path.exists(),
                "constraints_text": constraints_path.read_text(encoding="utf-8"),
                "cwd": cwd,
            })
            emit({"type": "log", "line": " ".join(packs_runner.pip_install_argv(
                specs, constraints_path=constraints_path))})
            for line in output:
                if tail is not None:
                    tail.append(line)
                emit({"type": "log", "line": line})
            return returncode

        monkeypatch.setattr(packs_runner, "run_pip", _run_pip)
        return calls

    return _make


def _steps(events: list[dict]) -> list[str]:
    return [e["step"] for e in events if e["type"] == "step_started"]


def _finished(events: list[dict]) -> list[str]:
    return [e["step"] for e in events if e["type"] == "step_done"]


def _lines(events: list[dict]) -> list[str]:
    return [e["line"] for e in events if e["type"] == "log"]


def _entry(plugin_id: str) -> dict[str, Any] | None:
    return plugin_loader.load_lockfile().get("plugins", {}).get(plugin_id)


def _github_plan(*, ref: str = "v1", **decisions) -> flows.InstallPlan:
    found = plugin_inspect.inspect_github(
        "alice", "extras", ref, lockfile=plugin_loader.load_lockfile()
    )
    return flows.plan_from_inspection(found, **decisions)


def _builtin_plan(plugin_id: str = "demo-pack", **decisions) -> flows.InstallPlan:
    found = plugin_inspect.inspect_builtin(
        plugin_id, lockfile=plugin_loader.load_lockfile()
    )
    return flows.plan_from_inspection(found, **decisions)


def _write_builtin(builtin_root: Path, manifest: str, plugin_id: str = "demo-pack"):
    pack = builtin_root / plugin_id
    pack.mkdir()
    (pack / plugin_loader.MANIFEST_FILENAME).write_text(manifest, encoding="utf-8")
    return pack


def _bare_plan(**overrides) -> flows.InstallPlan:
    """A plan built by hand, for the checks that defend against a stale one."""
    fields: dict[str, Any] = dict(
        kind="github", plugin_id="extras", catalog_id=None, owner="alice",
        repo="extras", ref="v1", sha="a" * 40, manifest={},
        granted_capabilities=(), trust_author=False, force=False,
        mode="install", prior=None,
    )
    fields.update(overrides)
    return flows.InstallPlan(**fields)


def _install(plan, *, emit=None, cancel_check=None, **kw):
    return flows.install_plugin_live(
        plan,
        emit=(lambda event: None) if emit is None else emit,
        cancel_check=(lambda: False) if cancel_check is None else cancel_check,
        **kw,
    )


# ── the order of an install ────────────────────────────────────────────────

def test_a_repository_install_runs_every_step_in_order(
    monkeypatch, user_root, fake_github, fake_pip
):
    """resolve, download, extract, verify, deps, stage, lock -- and the sha is
    never resolved a second time, because a branch that moved between the
    preview and the install would substitute a commit nobody looked at."""
    sha = fake_github({"cdui.plugin.toml": WITH_DEPS, "nodes/thing.py": "VALUE = 1\n"})
    fake_pip()
    plan = _github_plan()

    def _never(*_a):  # pragma: no cover - only runs on a bug
        raise AssertionError("an install must not re-resolve the ref")

    monkeypatch.setattr(github, "resolve_sha", _never)

    events: list[dict] = []
    outcome = _install(plan, emit=events.append)

    assert _steps(events) == [
        "resolve", "download", "extract", "verify", "deps", "stage", "lock",
    ]
    assert _finished(events) == _steps(events), "every step that ran also closed"
    assert outcome.plugin_id == "extras"
    assert outcome.sha == sha
    assert outcome.deps_installed == ("tinylib==1.0.0",)
    assert outcome.replaced is False
    assert outcome.plugin_dir == user_root / "extras"
    assert (user_root / "extras" / "nodes" / "thing.py").exists()


def test_a_repository_with_no_dependencies_has_no_deps_step(fake_github):
    """An empty step is a checkbox for work nobody did."""
    fake_github({"cdui.plugin.toml": PLAIN})
    events: list[dict] = []
    _install(_github_plan(), emit=events.append)
    assert "deps" not in _steps(events)


def test_a_builtin_install_resolves_and_records_and_nothing_else(
    builtin_root, user_root
):
    """A pack that shipped in this release has nothing to download, nothing
    to unpack and nobody outside this repository to be asked about."""
    _write_builtin(builtin_root, BUILTIN)
    events: list[dict] = []
    outcome = _install(_builtin_plan(), emit=events.append)

    assert _steps(events) == ["resolve", "lock"]
    assert "built-in pack: gate skipped" in _lines(events)
    assert outcome.plugin_dir is None, "nothing was written, so nothing to name"
    assert outcome.sha is None
    assert _entry("demo-pack")["source_kind"] == "builtin"
    assert not (user_root / "demo-pack").exists()


def test_a_builtin_pack_with_dependencies_still_installs_them(
    builtin_root, fake_pip
):
    _write_builtin(builtin_root, BUILTIN_WITH_DEPS)
    calls = fake_pip()
    events: list[dict] = []
    outcome = _install(_builtin_plan(), emit=events.append)

    assert _steps(events) == ["resolve", "deps", "lock"]
    assert calls[0]["specs"] == ["tinylib==1.0.0"]
    assert outcome.deps_installed == ("tinylib==1.0.0",)


# ── the identity check ─────────────────────────────────────────────────────

def test_a_tarball_that_asks_for_more_than_the_preview_did_is_refused(
    monkeypatch, user_root, fake_github
):
    """The dialog described one manifest. Without this check the installer
    obeys another, and every capability the user unticked is one the tarball
    can simply declare."""
    fake_github({"cdui.plugin.toml": PLAIN})
    plan = _github_plan()
    # The same commit, answering with a different manifest the second time.
    fake_github({"cdui.plugin.toml": WITH_CAPABILITY})

    with pytest.raises(PluginInstallError) as excinfo:
        _install(plan)
    assert "differs from the one you consented to" in str(excinfo.value)
    assert "network" in (excinfo.value.hint or "")
    assert not (user_root / "extras").exists()
    assert _entry("extras") is None


def test_a_tarball_that_installs_under_another_id_is_refused(
    user_root, fake_github
):
    """The id is the lockfile key, the card and the /api/plugins/{id} URL."""
    fake_github({"cdui.plugin.toml": PLAIN})
    plan = _github_plan()
    fake_github({"cdui.plugin.toml": PLAIN.replace('id = "extras"', 'id = "other"')})

    with pytest.raises(PluginInstallError) as excinfo:
        _install(plan)
    assert "'other'" in (excinfo.value.hint or "")
    assert not (user_root / "extras").exists()


def test_a_tarball_that_grew_an_allowed_modules_list_is_refused(fake_github):
    """allowed_modules is the AST gate switched off by name. Growing one
    after the trust question was asked is asking it of nobody."""
    fake_github({"cdui.plugin.toml": PLAIN})
    plan = _github_plan()
    fake_github({"cdui.plugin.toml": WITH_MODULES})

    with pytest.raises(PluginInstallError) as excinfo:
        _install(plan)
    assert "subprocess" in (excinfo.value.hint or "")


def test_trusting_the_author_is_an_answer_about_a_LIST_not_a_flag(fake_github):
    """The user read ["subprocess"] and said yes to that. A tarball shipping
    ["subprocess", "os"] has taken that answer as permission for a list they
    never saw, and every extra name is another door the gate leaves open."""
    fake_github({"cdui.plugin.toml": WITH_MODULES})
    plan = _github_plan(trust_author=True)
    fake_github({"cdui.plugin.toml": WITH_MODULES.replace(
        '["subprocess"]', '["subprocess", "os"]')})

    with pytest.raises(PluginInstallError) as excinfo:
        _install(plan)
    assert "differs from the one you consented to" in str(excinfo.value)
    assert "os" in (excinfo.value.hint or "")
    assert "not consented to" in (excinfo.value.hint or "")


def test_the_module_list_that_was_consented_to_installs(fake_github):
    """The other side of it: the same list, trusted, goes through."""
    fake_github({"cdui.plugin.toml": WITH_MODULES})
    _install(_github_plan(trust_author=True))
    assert _entry("extras")["trusted_modules"] == ["subprocess"]


def test_the_manifest_that_was_consented_to_installs(fake_github, user_root):
    """The other side of the check: a capability that WAS granted, and a
    module list the author is trusted for, go through untouched."""
    fake_github({"cdui.plugin.toml": WITH_CAPABILITY})
    plan = _github_plan(accept_capabilities=["network"])
    _install(plan)
    assert _entry("extras")["capabilities"] == ["network"]


# ── the security gate ──────────────────────────────────────────────────────

def test_an_ast_refusal_names_the_file_in_the_hint(user_root, fake_github):
    """"Refused" without a filename is a plugin the author cannot fix."""
    fake_github({
        "cdui.plugin.toml": PLAIN,
        "nodes/payload.py": "import os\n",
    })
    with pytest.raises(PluginInstallError) as excinfo:
        _install(_github_plan())
    assert "security scan" in str(excinfo.value)
    assert "payload.py" in (excinfo.value.hint or "")
    assert not (user_root / "extras").exists(), "a refused plugin is not staged"


# ── the dependency step ────────────────────────────────────────────────────

def test_the_dependency_step_runs_under_the_package_centers_freeze(
    fake_github, fake_pip
):
    """Add-only: the constraints file pins what this interpreter already
    imported, so a plugin cannot downgrade a package the server is using."""
    fake_github({"cdui.plugin.toml": WITH_DEPS})
    calls = fake_pip()
    _install(_github_plan())

    assert len(calls) == 1
    call = calls[0]
    assert call["specs"] == ["tinylib==1.0.0"]
    assert call["constraints_path"].name == "constraints.txt"
    assert call["constraints_exists"], "written before uv runs, not after"
    assert "==" in call["constraints_text"], "it pins the interpreter it froze"


def test_a_resolver_conflict_asks_for_a_restart_and_quotes_the_command(
    fake_github, fake_pip, user_root
):
    """Not a broken plugin: uv cannot replace a package this process has
    already imported, so the answer is a command to run with the server
    stopped -- one a Windows path and a version spec both survive."""
    fake_github({"cdui.plugin.toml": WITH_DEPS})
    fake_pip(returncode=1, output=("x No solution found when resolving",))

    with pytest.raises(PluginNeedsRestart) as excinfo:
        _install(_github_plan())
    command = excinfo.value.command
    assert command.startswith("uv pip install --python ")
    assert '"tinylib==1.0.0"' in command, "an unquoted spec is not pasteable"
    assert command in (excinfo.value.hint or "")
    assert not (user_root / "extras").exists(), "deps run before anything is staged"
    assert _entry("extras") is None


def test_any_other_pip_failure_keeps_uvs_last_lines(fake_github, fake_pip):
    fake_github({"cdui.plugin.toml": WITH_DEPS})
    fake_pip(returncode=2, output=("error: could not download tinylib",))

    with pytest.raises(PluginInstallError) as excinfo:
        _install(_github_plan())
    assert not isinstance(excinfo.value, PluginNeedsRestart)
    assert "uv exited 2" in str(excinfo.value)
    assert "could not download tinylib" in (excinfo.value.hint or "")


# ── stopping ───────────────────────────────────────────────────────────────

def test_a_cancel_during_the_download_leaves_nothing_behind(
    user_root, fake_github
):
    """Stop has to be felt inside the download, and what it leaves has to be
    nothing: no staging directory, no lockfile entry, no half-install."""
    asked = {"n": 0}

    def _cancel() -> bool:
        asked["n"] += 1
        return asked["n"] >= 2

    def _stop(*, cancel_check, progress):
        if cancel_check():
            raise PluginCancelled("download of alice/extras cancelled")

    fake_github({"cdui.plugin.toml": PLAIN}, during_download=_stop)

    events: list[dict] = []
    with pytest.raises(PluginCancelled):
        _install(_github_plan(), emit=events.append, cancel_check=_cancel)

    assert _steps(events) == ["resolve", "download"]
    assert _finished(events) == ["resolve"], "a cancelled step is not a done one"
    assert not (user_root / flows.STAGING_DIRNAME).exists()
    assert _entry("extras") is None


def test_a_cancel_between_the_copy_and_the_rename_removes_the_copy(
    user_root, fake_github
):
    """The copy is the last moment at which stopping costs nothing. What it
    must not cost is a .staging directory nothing points at -- invisible,
    permanent, and the same size as the plugin."""
    staging = user_root / flows.STAGING_DIRNAME

    def _cancel() -> bool:
        return staging.exists() and any(staging.iterdir())

    fake_github({"cdui.plugin.toml": PLAIN})

    with pytest.raises(PluginCancelled):
        _install(_github_plan(), cancel_check=_cancel)

    assert list(staging.iterdir()) == []
    assert not (user_root / "extras").exists()
    assert _entry("extras") is None


# ── replacing what is there ────────────────────────────────────────────────

def test_a_rename_that_fails_puts_the_previous_install_back(
    monkeypatch, user_root, fake_github
):
    """The window between the two renames is the only moment in an install
    when the user has no plugin at all."""
    installed = user_root / "extras"
    installed.mkdir()
    (installed / "marker.txt").write_text("the copy that was here", encoding="utf-8")

    real_rename = Path.rename

    def _rename(self, target):
        if self.parent.name == flows.STAGING_DIRNAME:
            raise OSError(13, "used by another process")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", _rename)
    fake_github({"cdui.plugin.toml": PLAIN})

    with pytest.raises(PluginInstallError) as excinfo:
        _install(_github_plan(force=True))

    assert "used by another process" in (excinfo.value.hint or "")
    assert (installed / "marker.txt").read_text(encoding="utf-8") == (
        "the copy that was here")
    assert [p.name for p in user_root.glob("extras.old-*")] == []
    assert list((user_root / flows.STAGING_DIRNAME).iterdir()) == []


def test_a_restore_that_fails_too_says_where_the_previous_install_is(
    monkeypatch, user_root, fake_github
):
    """One Windows cause -- something holding the destination open -- breaks
    both renames, so the restore is the failure most likely to happen here.
    Unguarded, its raw OSError replaced the translated one and nobody was
    told that the previous install is still on the disk under another name."""
    installed = user_root / "extras"
    installed.mkdir()
    (installed / "marker.txt").write_text("the copy that was here", encoding="utf-8")

    real_rename = Path.rename

    def _rename(self, target):
        if self.parent.name == flows.STAGING_DIRNAME:
            raise OSError(13, "used by another process")
        if self.name.startswith("extras.old-"):
            raise OSError(13, "and the backup will not move back either")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", _rename)
    fake_github({"cdui.plugin.toml": PLAIN})

    with pytest.raises(PluginInstallError) as excinfo:
        _install(_github_plan(force=True))

    hint = excinfo.value.hint or ""
    assert "used by another process" in hint, "the failure that started it"
    assert "and the backup will not move back either" in hint
    # The one fact that makes this recoverable by hand.
    backups = [p for p in user_root.glob("extras.old-*")]
    assert len(backups) == 1
    assert backups[0].name in hint
    assert (backups[0] / "marker.txt").read_text(encoding="utf-8") == (
        "the copy that was here")
    assert list((user_root / flows.STAGING_DIRNAME).iterdir()) == []


def test_a_previous_install_that_will_not_move_aside_is_a_plugin_failure(
    monkeypatch, user_root, fake_github
):
    """On Windows a directory the editor or a running server still has open
    refuses to be renamed. A raw PermissionError out of an install names
    neither the plugin nor what to close."""
    installed = user_root / "extras"
    installed.mkdir()
    (installed / "marker.txt").write_text("the copy that was here", encoding="utf-8")

    real_rename = Path.rename

    def _rename(self, target):
        if self.name == "extras":
            raise PermissionError(13, "used by another process")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", _rename)
    fake_github({"cdui.plugin.toml": PLAIN})

    with pytest.raises(PluginInstallError) as excinfo:
        _install(_github_plan(force=True))

    assert "move the previous extras aside" in str(excinfo.value)
    assert "used by another process" in (excinfo.value.hint or "")
    assert (installed / "marker.txt").read_text(encoding="utf-8") == (
        "the copy that was here")
    assert list((user_root / flows.STAGING_DIRNAME).iterdir()) == []
    assert _entry("extras") is None


def test_a_copy_that_fails_halfway_leaves_no_staging_directory(
    monkeypatch, user_root, fake_github
):
    """A .staging directory nothing points at is invisible, permanent, and
    the same size as however far the copy got."""
    def _half_a_copy(source, dest, *a, **kw):
        Path(dest).mkdir(parents=True)
        (Path(dest) / "half.txt").write_text("as far as it got", encoding="utf-8")
        raise OSError(28, "no space left on device")

    monkeypatch.setattr(flows.shutil, "copytree", _half_a_copy)
    fake_github({"cdui.plugin.toml": PLAIN})

    with pytest.raises(OSError):
        _install(_github_plan())
    assert list((user_root / flows.STAGING_DIRNAME).iterdir()) == []


def test_force_replaces_an_installed_plugin_and_takes_the_backup_with_it(
    user_root, fake_github
):
    """The backup outlives every step that could still fail, and not one
    step longer: a .old-<timestamp> directory left behind is the size of the
    plugin and nothing will ever look at it again."""
    installed = user_root / "extras"
    installed.mkdir()
    (installed / "marker.txt").write_text("the old copy", encoding="utf-8")
    fake_github({"cdui.plugin.toml": PLAIN})

    outcome = _install(_github_plan(force=True))

    assert outcome.replaced is True
    assert not (installed / "marker.txt").exists(), "replaced, not merged"
    assert (installed / "cdui.plugin.toml").exists()
    assert list(user_root.glob("extras.old-*")) == []


def test_a_plugin_that_is_already_here_is_not_reinstalled_by_accident(
    user_root, fake_github
):
    """A directory with no lockfile entry is still an install: whatever put
    it there, replacing it is a decision somebody has to make."""
    (user_root / "extras").mkdir()
    fake_github({"cdui.plugin.toml": PLAIN})

    with pytest.raises(AlreadyInstalled) as excinfo:
        _install(_github_plan())
    assert excinfo.value.plugin_id == "extras"


def test_a_builtin_pack_that_is_already_installed_needs_force(builtin_root):
    _write_builtin(builtin_root, BUILTIN)
    _install(_builtin_plan())
    with pytest.raises(AlreadyInstalled):
        _install(_builtin_plan())
    _install(_builtin_plan(force=True))  # and force is what says otherwise


def test_installing_again_forgets_that_it_was_uninstalled(builtin_root):
    """Installing a pack by name is the undo for having uninstalled it, so
    the tombstone goes with it -- or `cdui plugin sync` keeps skipping a pack
    that is now installed."""
    _write_builtin(builtin_root, BUILTIN)
    lockfile = plugin_loader.load_lockfile()
    plugin_loader.mark_removed(lockfile, "demo-pack", source_kind="builtin")
    plugin_loader.save_lockfile(lockfile)

    outcome = _install(_builtin_plan())
    assert outcome.tombstone_cleared is True
    assert plugin_loader.removed_ids(plugin_loader.load_lockfile()) == set()


# ── what a finished install writes down ────────────────────────────────────

def test_the_lockfile_entry_of_a_repository_install(user_root, fake_github):
    """The keys ``cdui plugin install`` has always written, in that order:
    this is the same file, read by the same loader."""
    sha = fake_github({"cdui.plugin.toml": WITH_CAPABILITY})
    _install(_github_plan(accept_capabilities=["network"]))

    entry = _entry("extras")
    assert list(entry) == [
        "source_kind", "source", "url", "ref", "sha", "installed_at",
        "manifest", "trusted_modules", "capabilities", "enabled",
    ]
    assert entry["source_kind"] == "github_url"
    assert entry["source"] == "alice/extras@v1"
    assert entry["url"] == "https://github.com/alice/extras"
    assert entry["ref"] == "v1"
    assert entry["sha"] == sha
    assert entry["manifest"]["version"] == "1.2.0"
    assert entry["capabilities"] == ["network"]
    assert entry["trusted_modules"] == []
    assert entry["enabled"] is True


def test_the_lockfile_entry_of_a_builtin_install(builtin_root):
    """The same keys minus the three a repository has and a pack does not.
    ``catalog_id`` is present because a built-in pack IS a catalog row."""
    _write_builtin(builtin_root, BUILTIN)
    _install(_builtin_plan())

    entry = _entry("demo-pack")
    assert list(entry) == [
        "source_kind", "source", "installed_at", "manifest",
        "trusted_modules", "capabilities", "enabled", "catalog_id",
    ]
    assert entry["source_kind"] == "builtin"
    assert entry["source"] == "demo-pack"
    assert entry["catalog_id"] == "demo-pack"
    assert entry["manifest"]["version"] == "0.1.0"
    assert entry["enabled"] is True


def test_a_catalog_row_is_recorded_and_free_text_is_not(fake_github):
    """"Installed from the catalog" is a claim a reader wants to trust, so
    the key is absent rather than null when there was no row."""
    fake_github({"cdui.plugin.toml": PLAIN})
    _install(_github_plan())
    assert "catalog_id" not in _entry("extras")

    plan = _bare_plan(catalog_id="extras", force=True, manifest={"plugin": {}})
    fake_github({"cdui.plugin.toml": PLAIN})
    _install(plan)
    assert _entry("extras")["catalog_id"] == "extras"


# ── plans, and what they refuse before a job exists ────────────────────────

def test_a_capability_nobody_granted_stops_the_plan(fake_github):
    """Server-side, on the caller's thread, where the refusal is still an
    answer to a request rather than a failed install to read events about."""
    fake_github({"cdui.plugin.toml": WITH_CAPABILITY})
    found = plugin_inspect.inspect_github("alice", "extras", "v1", lockfile={})
    with pytest.raises(ConsentRequired) as excinfo:
        flows.plan_from_inspection(found)
    assert excinfo.value.missing_capabilities == ("network",)


def test_an_untrusted_module_list_stops_the_plan(fake_github):
    fake_github({"cdui.plugin.toml": WITH_MODULES})
    found = plugin_inspect.inspect_github("alice", "extras", "v1", lockfile={})
    with pytest.raises(ConsentRequired) as excinfo:
        flows.plan_from_inspection(found)
    assert excinfo.value.allowed_modules == ("subprocess",)
    assert flows.plan_from_inspection(found, trust_author=True).trust_author is True


def test_a_capability_granted_last_time_is_not_a_second_decision(fake_github):
    """An update that asks for what the user already agreed to goes through;
    what it added would not."""
    fake_github({"cdui.plugin.toml": WITH_CAPABILITY})
    lockfile = {"plugins": {"extras": {
        "source_kind": "github_url", "capabilities": ["network"],
        "url": "https://github.com/alice/extras", "sha": "b" * 40,
    }}}
    found = plugin_inspect.inspect_github("alice", "extras", "v1", lockfile=lockfile)
    plan = flows.plan_from_inspection(found)
    assert plan.granted_capabilities == ("network",)
    assert plan.mode == "update"


def test_a_builtin_pack_is_planned_without_asking_anybody(builtin_root):
    """It arrived through a pull request in this repository. Its capabilities
    are recorded all the same -- "which of my plugins reaches the network" is
    a question about every pack, wherever it came from."""
    _write_builtin(builtin_root, BUILTIN.replace(
        'schema_version = 1', 'schema_version = 1\n\n[security]\ncapabilities = ["network"]'))
    plan = _builtin_plan()
    assert plan.granted_capabilities == ("network",)
    assert plan.kind == "builtin"


def test_a_plan_that_names_no_commit_is_refused(user_root):
    """Never re-resolved here: a plan with no sha skipped the inspection, and
    guessing a commit is the substitution a pinned sha exists to prevent."""
    with pytest.raises(PluginInstallError) as excinfo:
        _install(_bare_plan(sha=None))
    assert "names no commit" in str(excinfo.value)


def test_a_reserved_id_is_refused_before_anything_is_downloaded(user_root):
    """Asked again here, not because the inspection forgot: a plan can be
    minutes old, and this is the last point at which refusing is free."""
    with pytest.raises(PluginInstallError) as excinfo:
        _install(_bare_plan(plugin_id="install"))
    assert "reserved" in str(excinfo.value)


# ── progress ───────────────────────────────────────────────────────────────

def test_progress_frames_are_throttled_but_the_last_one_always_arrives(
    monkeypatch, fake_github
):
    """A bar that stops at 97% because the download finished inside a
    throttle window is worse than no bar."""
    clock = {"t": 100.0}
    monkeypatch.setattr(flows, "monotonic", lambda: clock["t"])

    def _report(*, cancel_check, progress):
        progress(0, 100)
        clock["t"] += 0.1
        progress(40, 100)      # inside the window: collapsed into the first
        clock["t"] += 0.1
        progress(100, 100)     # the last frame is forced past the throttle

    fake_github({"cdui.plugin.toml": PLAIN}, during_download=_report)

    events: list[dict] = []
    _install(_github_plan(), emit=events.append)

    frames = [e for e in events if e["type"] == "progress"]
    assert [f["bytes_done"] for f in frames] == [0, 100]
    assert [f["percent"] for f in frames] == [0.0, 100.0]
    assert {f["item"] for f in frames} == {"tarball"}
    assert all(f["bytes_total"] == 100 for f in frames)


def test_a_frame_past_the_interval_reports(monkeypatch, fake_github):
    clock = {"t": 100.0}
    monkeypatch.setattr(flows, "monotonic", lambda: clock["t"])

    def _report(*, cancel_check, progress):
        progress(0, None)
        clock["t"] += flows.PROGRESS_MIN_INTERVAL_S
        progress(40, None)

    fake_github({"cdui.plugin.toml": PLAIN}, during_download=_report)
    events: list[dict] = []
    _install(_github_plan(), emit=events.append)

    frames = [e for e in events if e["type"] == "progress"]
    assert [f["bytes_done"] for f in frames] == [0, 40]
    assert [f["percent"] for f in frames] == [None, None], "no total, no claim"


# ── the client hook ────────────────────────────────────────────────────────

def test_the_github_client_can_be_handed_in(fake_github):
    """``scripts/plugins.py`` re-exports the client under its own module
    attributes precisely so a test can replace them; the flow has to call
    the names the CLI patched, not the ones behind them."""
    class _Client:
        def __init__(self) -> None:
            self.seen: list[str] = []

        def download_tarball(self, owner, repo, sha, dest, **kw):
            self.seen.append(sha)
            _tarball_of({f"{repo}-main/cdui.plugin.toml": PLAIN}, dest)

    fake_github({"cdui.plugin.toml": PLAIN})
    plan = _github_plan()
    client = _Client()
    _install(plan, github=client)

    assert client.seen == ["a" * 40], "the handed-in client is the one that ran"


def test_a_partial_client_falls_back_per_name(fake_github):
    """The CLI re-exports two of the three functions and reaches the rest
    through the module, so requiring the whole surface would make the hook
    unusable by the one caller it exists for."""
    class _OnlyDownload:
        def __init__(self) -> None:
            self.calls = 0

        def download_tarball(self, owner, repo, sha, dest, **kw):
            self.calls += 1
            _tarball_of({f"{repo}-main/cdui.plugin.toml": PLAIN}, dest)

    fake_github({"cdui.plugin.toml": PLAIN})
    client = _OnlyDownload()
    _install(_github_plan(), github=client)
    assert client.calls == 1
    assert _entry("extras") is not None
