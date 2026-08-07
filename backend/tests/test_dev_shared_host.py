"""`cdui stop` and `cdui update` on a machine with more than one user (#250).

Both commands were written for the deployment CodefyUI actually had — one
student, one laptop, one checkout — and both are wrong in the same way once a
second person shares the box:

* ``stop`` swept the whole machine (``pkill -f "uvicorn app.main:app"`` and,
  worse, ``pkill -f vite``), so one person stopping their server stopped
  everyone's training and every unrelated Vite dev server on the host.
* ``update`` never asked whether a server was running before it realigned the
  checkout, deleted ``frontend/dist`` and rewrote the venv underneath it.

ISOLATION, and it matters here more than usual: ``update()`` calls
``shutil.rmtree(DIST_DIR)`` inline in this process. Stubbing ``run`` only
redirects the *subprocesses* — the deletion still happens, against the
developer's real ``frontend/dist``. Every test below therefore redirects
``ROOT``, ``VENV``, ``DIST_DIR`` and both server state files at ``tmp_path``
before calling anything.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import dev  # scripts/dev.py — conftest puts scripts/ on sys.path


# ── ownership matching: is that process ours, and is it THIS install's? ──


@pytest.fixture
def rooted(tmp_path, monkeypatch):
    """ROOT redirected at a scratch checkout directory."""
    root = tmp_path / "codefyui"
    root.mkdir()
    monkeypatch.setattr(dev, "ROOT", root)
    return root


def test_under_root_accepts_the_root_and_its_children(rooted):
    assert dev._under_root(str(rooted))
    assert dev._under_root(str(rooted / "backend"))
    assert dev._under_root(str(rooted / "frontend" / "node_modules"))


def test_under_root_rejects_a_sibling_sharing_the_prefix(rooted, tmp_path):
    """``codefyui-old`` is not inside ``codefyui``. A bare ``startswith``
    would say it is, and would then stop that other install's server."""
    assert not dev._under_root(str(tmp_path / "codefyui-old" / "backend"))
    assert not dev._under_root(str(tmp_path / "somewhere-else"))
    assert not dev._under_root(None)
    assert not dev._under_root("")


def test_matches_our_background_server_by_launch_path(rooted):
    """`cdui start` runs the venv's own uvicorn, so the absolute path in
    argv[0] already identifies the install."""
    cmdline = [str(rooted / "backend" / ".venv" / "bin" / "uvicorn"),
               "app.main:app", "--host", "127.0.0.1", "--port", "8000"]
    assert dev._is_this_install_process(cmdline, None)


def test_matches_dev_mode_vite_by_working_directory(rooted):
    """`cdui dev` starts the frontend in ``<root>/frontend``. A globally
    installed vite has no ROOT path in its argv, so cwd is what settles it."""
    assert dev._is_this_install_process(
        ["node", "/usr/lib/node_modules/vite/bin/vite.js"],
        str(rooted / "frontend"),
    )


def test_rejects_another_installs_server(rooted, tmp_path):
    """The whole point: a colleague's CodefyUI is a CodefyUI, and is not ours."""
    theirs = tmp_path / "home-someone-else" / "CodefyUI"
    cmdline = [str(theirs / "backend" / ".venv" / "bin" / "uvicorn"),
               "app.main:app"]
    assert not dev._is_this_install_process(cmdline, str(theirs / "backend"))


def test_rejects_an_unrelated_vite_dev_server(rooted):
    """`pkill -f vite` was never scoped to CodefyUI at all — it ended any
    Vite any developer on the machine happened to be running."""
    assert not dev._is_this_install_process(
        ["node", "/srv/marketing-site/node_modules/vite/bin/vite.js"],
        "/srv/marketing-site",
    )


def test_rejects_an_unrelated_process_inside_our_checkout(rooted):
    """Living in ROOT is not enough — an editor, a shell or a test runner
    opened here must survive `cdui stop`."""
    assert not dev._is_this_install_process(
        ["python", "-m", "pytest", "tests/"], str(rooted / "backend"))
    assert not dev._is_this_install_process([], str(rooted))


def test_vite_is_matched_as_a_path_component_not_a_substring(rooted):
    """A bare substring test would claim `invite.py` is a dev server. It
    lives in ROOT, so the ownership half would not save it."""
    assert not dev._is_this_install_process(
        ["python", str(rooted / "scripts" / "invite.py")], str(rooted))
    # The spellings that must still match.
    for part in ("vite",
                 str(rooted / "frontend" / "node_modules" / "vite" / "bin"
                     / "vite.js")):
        assert dev._is_this_install_process(["node", part], str(rooted)), part


# ── the psutil scan ──────────────────────────────────────────────────────


class _FakeProc:
    def __init__(self, pid, cmdline, cwd=None, cwd_denied=False):
        self.info = {"pid": pid, "cmdline": cmdline}
        self._cwd = cwd
        self._denied = cwd_denied

    def cwd(self):
        if self._denied:
            raise PermissionError("access denied")
        return self._cwd


class _FakePsutil:
    """Just enough psutil for ``_this_install_pids``."""

    def __init__(self, procs, self_pid, parent_pids=()):
        self._procs = procs
        self._self_pid = self_pid
        self._parents = [SimpleNamespace(pid=p) for p in parent_pids]

    def process_iter(self, attrs=None):
        return list(self._procs)

    def Process(self):  # noqa: N802 — mirrors psutil's own capitalisation
        return SimpleNamespace(pid=self._self_pid, parents=lambda: self._parents)


def _install_fake_psutil(monkeypatch, procs, parent_pids=()):
    fake = _FakePsutil(procs, self_pid=dev.os.getpid(), parent_pids=parent_pids)
    monkeypatch.setitem(sys.modules, "psutil", fake)
    return fake


def test_scan_finds_ours_and_leaves_everything_else(rooted, monkeypatch):
    ours_backend = _FakeProc(
        101, [str(rooted / "backend" / ".venv" / "bin" / "uvicorn"),
              "app.main:app"])
    ours_vite = _FakeProc(
        102, ["node", "vite"], cwd=str(rooted / "frontend"))
    theirs = _FakeProc(
        201, ["/home/bob/CodefyUI/backend/.venv/bin/uvicorn", "app.main:app"],
        cwd="/home/bob/CodefyUI/backend")
    other_vite = _FakeProc(
        202, ["node", "/srv/site/node_modules/vite/bin/vite.js"],
        cwd="/srv/site")
    unrelated = _FakeProc(203, ["sshd", "-D"], cwd="/")
    _install_fake_psutil(
        monkeypatch, [ours_backend, ours_vite, theirs, other_vite, unrelated])

    assert dev._this_install_pids() == [101, 102]


def test_scan_never_returns_our_own_process_or_an_ancestor(rooted, monkeypatch):
    """`cdui stop` runs from inside ROOT. Killing ourselves, or the shell
    that launched us, would be a spectacular way to succeed."""
    me = dev.os.getpid()
    procs = [
        _FakeProc(me, [str(rooted / "x" / "vite")]),
        _FakeProc(7, [str(rooted / "y" / "vite")]),
        _FakeProc(9, [str(rooted / "backend" / "uvicorn"), "app.main:app"]),
    ]
    _install_fake_psutil(monkeypatch, procs, parent_pids=(7,))
    assert dev._this_install_pids() == [9]


def test_scan_skips_a_process_whose_cwd_is_denied(rooted, monkeypatch):
    """Another user's process denies ``cwd()``. Unreadable means not ours."""
    denied = _FakeProc(301, ["node", "vite"], cwd_denied=True)
    _install_fake_psutil(monkeypatch, [denied])
    assert dev._this_install_pids() == []


def test_scan_is_empty_without_psutil(rooted, monkeypatch):
    """No psutil, no identification — and then only the pidfile server is
    stopped. Failing to stop a stray is recoverable; stopping someone
    else's server is not."""
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert dev._this_install_pids() == []


# ── stop(): scoped by default, machine-wide only on --all ────────────────


@pytest.fixture
def stopped(rooted, tmp_path, monkeypatch):
    """`stop()` with every process-ending call captured instead of made."""
    monkeypatch.setattr(dev, "SERVER_PIDFILE", tmp_path / "server.pid")
    monkeypatch.setattr(dev, "SERVER_ADDRFILE", tmp_path / "server.addr")
    shell: list[list[str]] = []
    targeted: list[int] = []
    groups: list[int] = []
    monkeypatch.setattr(dev.subprocess, "run",
                        lambda cmd, **kw: shell.append([str(c) for c in cmd]))
    monkeypatch.setattr(dev, "_terminate_pid", targeted.append)
    monkeypatch.setattr(dev, "_terminate_posix", groups.append)
    monkeypatch.setattr(dev, "_this_install_pids", list)
    monkeypatch.setattr(dev, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(dev, "_has_psutil", lambda: True)
    monkeypatch.setattr(sys, "argv", ["cdui", "stop"])
    return SimpleNamespace(shell=shell, targeted=targeted, groups=groups)


def _machine_wide(shell):
    """Commands that end processes this install does not own."""
    joined = [" ".join(cmd) for cmd in shell]
    return [c for c in joined
            if "pkill" in c or "/IM" in c or "WINDOWTITLE" in c]


def test_stop_does_not_sweep_the_whole_machine(stopped):
    """The regression this whole file exists for."""
    dev.stop()
    assert _machine_wide(stopped.shell) == []


def test_stop_all_still_sweeps_the_whole_machine(stopped, monkeypatch):
    """The old behaviour is preserved, just no longer the default: it is
    still the only thing that catches a server whose checkout is gone."""
    monkeypatch.setattr(sys, "argv", ["cdui", "stop", "--all"])
    dev.stop()
    assert _machine_wide(stopped.shell), stopped.shell


def test_stop_all_says_out_loud_what_it_is_about_to_do(stopped, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["cdui", "stop", "--all"])
    dev.stop()
    assert "--all" in capsys.readouterr().out


def test_stop_still_ends_leftovers_from_this_install(stopped, monkeypatch):
    """A foreground `cdui start` leaves no pidfile, so the scoped sweep is
    what stops it — deleting the sweep outright would have broken that."""
    monkeypatch.setattr(dev, "_this_install_pids", lambda: [111, 222])
    dev.stop()
    assert stopped.targeted == [111, 222]


def test_stop_still_ends_the_pidfile_server(stopped):
    dev.SERVER_PIDFILE.write_text("4242")
    dev.SERVER_ADDRFILE.write_text("127.0.0.1:8000")
    dev.stop()
    hit = (4242 in stopped.groups
           or any("4242" in " ".join(cmd) for cmd in stopped.shell))
    assert hit, stopped.shell
    assert not dev.SERVER_PIDFILE.exists()


def test_stop_does_not_end_the_pidfile_server_twice(stopped, monkeypatch):
    """The scan sees the background server too; it has already been stopped
    through its process group, and signalling it again is noise at best."""
    dev.SERVER_PIDFILE.write_text("4242")
    dev.SERVER_ADDRFILE.write_text("127.0.0.1:8000")
    monkeypatch.setattr(dev, "_this_install_pids", lambda: [4242, 999])
    dev.stop()
    assert stopped.targeted == [999]


def test_stop_admits_when_it_cannot_identify_strays(stopped, monkeypatch, capsys):
    monkeypatch.setattr(dev, "_has_psutil", lambda: False)
    dev.stop()
    assert "psutil" in capsys.readouterr().out


# ── update(): refuse while a server is running ───────────────────────────


@pytest.fixture
def updatable(rooted, tmp_path, monkeypatch):
    """`update()` with every destructive edge redirected at tmp_path.

    ``run`` and ``install`` are captured, and DIST_DIR points at a throwaway
    directory because ``shutil.rmtree`` runs inline in this process.
    ``CODEFYUI_FORCE_BUILD`` keeps update on the build-from-source branch so
    ``_resolve_release_tag()`` never reaches for the network.
    """
    (rooted / ".git").mkdir()
    dist = rooted / "frontend" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(dev, "VENV", rooted / "backend" / ".venv")
    monkeypatch.setattr(dev, "DIST_DIR", dist)
    monkeypatch.setattr(dev, "SERVER_PIDFILE", tmp_path / "server.pid")
    monkeypatch.setattr(dev, "SERVER_ADDRFILE", tmp_path / "server.addr")
    calls: list[list[str]] = []
    monkeypatch.setattr(dev, "run", lambda cmd, **kw: calls.append(list(cmd)))
    monkeypatch.setattr(dev, "install", lambda **kw: calls.append(["install"]))
    monkeypatch.setattr(dev, "_resolve_update_options", lambda argv: ("skip", False))
    monkeypatch.setenv("CODEFYUI_FORCE_BUILD", "1")
    monkeypatch.setattr(sys, "argv", ["cdui", "update"])
    return SimpleNamespace(calls=calls, dist=dist)


def test_update_refuses_while_a_server_is_running(updatable, monkeypatch, capsys):
    """No git checkout, no rmtree, no reinstall — and a message that names
    the command to run first."""
    dev.SERVER_PIDFILE.write_text("4242")
    dev.SERVER_ADDRFILE.write_text("127.0.0.1:8000")
    monkeypatch.setattr(dev, "_pid_alive", lambda pid: True)

    with pytest.raises(SystemExit) as exc:
        dev.update()

    assert exc.value.code == 1
    assert updatable.calls == [], updatable.calls
    assert (updatable.dist / "index.html").exists(), \
        "update deleted the dist of a server that is still serving it"
    message = capsys.readouterr().err
    assert "cdui stop" in message
    assert "4242" in message


def test_update_proceeds_when_no_server_is_running(updatable, monkeypatch):
    monkeypatch.setattr(dev, "_pid_alive", lambda pid: False)
    dev.update()
    assert ["install"] in updatable.calls
    assert any(cmd[:2] == ["git", "checkout"] for cmd in updatable.calls)
    assert not updatable.dist.exists()


def test_update_is_not_blocked_by_a_stale_pidfile(updatable, monkeypatch):
    """A crashed server leaves its pidfile behind. Reusing the liveness check
    `cdui status` already runs means the stale file is cleared, not obeyed —
    otherwise a crash would make the install permanently un-updatable."""
    dev.SERVER_PIDFILE.write_text("4242")
    dev.SERVER_ADDRFILE.write_text("127.0.0.1:8000")
    monkeypatch.setattr(dev, "_pid_alive", lambda pid: False)

    dev.update()

    assert ["install"] in updatable.calls
    assert not dev.SERVER_PIDFILE.exists()


def test_update_help_still_answers_while_a_server_is_running(
        rooted, tmp_path, monkeypatch, capsys):
    """`--help` must not become collateral damage of the refusal: option
    resolution (which exits on --help) runs before the liveness check.

    Built by hand rather than on the `updatable` fixture because the real
    ``_resolve_update_options`` is the thing under test here.
    """
    (rooted / ".git").mkdir()
    dist = rooted / "frontend" / "dist"
    dist.mkdir(parents=True)
    monkeypatch.setattr(dev, "VENV", rooted / "backend" / ".venv")
    monkeypatch.setattr(dev, "DIST_DIR", dist)
    pidfile = tmp_path / "server.pid"
    pidfile.write_text("4242")
    monkeypatch.setattr(dev, "SERVER_PIDFILE", pidfile)
    monkeypatch.setattr(dev, "SERVER_ADDRFILE", tmp_path / "server.addr")
    monkeypatch.setattr(dev, "_pid_alive", lambda pid: True)
    calls: list[list[str]] = []
    monkeypatch.setattr(dev, "run", lambda cmd, **kw: calls.append(list(cmd)))
    monkeypatch.setattr(dev, "install", lambda **kw: calls.append(["install"]))
    monkeypatch.setattr(sys, "argv", ["cdui", "update", "--help"])

    with pytest.raises(SystemExit):
        dev.update()

    assert calls == []
    assert dist.exists()
    assert "cdui update" in capsys.readouterr().out
