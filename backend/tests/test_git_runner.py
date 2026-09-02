"""Starting a git process on behalf of a browser request.

This is the only place in the app that spawns git, so it is the only place
that can get these wrong -- and each of them fails somewhere else, long
after the call that caused it:

* an argument that reaches git as an OPTION rather than as a path or a ref;
* a git that decides to ask for a password, on a server with no terminal,
  and never returns;
* ``text=True``, which would rewrite the CRLF inside a diff and crash on a
  file that is not UTF-8;
* an inherited ``GIT_DIR``, which would silently point every command at
  another repository;
* a timeout that leaves git (and the ssh it started) running, holding the
  index lock the next request needs.

No test here starts a real process: ``subprocess.Popen`` is replaced with a
fake, following ``test_packs_runner.py``.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

from app.core.git import runner
from app.core.git.errors import GitError
from app.core.packs import runner as packs_runner

#: Where the fake git lives. Any absolute-looking string will do; it only
#: has to be recognisable in an argv assertion.
_GIT = "C:/Program Files/Git/cmd/git.exe"


class _FakeProc:
    """A finished (or hung) git process, without the process."""

    def __init__(self, *, stdout: bytes = b"", stderr: bytes = b"",
                 returncode: int = 0, hangs: bool = False):
        self._stdout = stdout
        self._stderr = stderr
        self._returncode = returncode
        self._hangs = hangs
        self.pid = 4242
        self.returncode: int | None = None
        self.communicate_calls: list[tuple] = []

    def communicate(self, data=None, timeout=None):
        self.communicate_calls.append((data, timeout))
        if self._hangs and len(self.communicate_calls) == 1:
            raise subprocess.TimeoutExpired(cmd="git", timeout=timeout or 0)
        self.returncode = self._returncode
        return self._stdout, self._stderr


def _fake_popen(monkeypatch, proc: _FakeProc) -> dict:
    """Install *proc* as the next ``Popen`` result; returns the recorded call."""
    seen: dict = {}

    def _popen(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return proc

    monkeypatch.setattr(subprocess, "Popen", _popen)
    return seen


@pytest.fixture(autouse=True)
def _fresh_caches():
    """Every cache in the runner is process-wide; no test may inherit one."""
    runner._reset_for_tests()
    yield
    runner._reset_for_tests()


@pytest.fixture
def fake_git(monkeypatch):
    """git is on PATH, and the ssh question is already answered.

    The probe for ``core.sshCommand`` is a git call of its own, so a test
    that is not about it would otherwise consume the fake ``Popen`` before
    the command under test got there. Its own tests below run it for real.
    """
    monkeypatch.setattr(runner, "git_executable", lambda: _GIT)
    monkeypatch.setattr(runner, "_ssh_command_configured", lambda: True)


def _run(monkeypatch, proc: _FakeProc, args=("status", "--porcelain=v2"),
         *, cwd: Path | None = None, **kwargs):
    """Call ``run_git`` against *proc*; returns ``(result, recorded call)``."""
    seen = _fake_popen(monkeypatch, proc)
    result = runner.run_git(list(args), cwd=cwd or Path.cwd(), timeout=10,
                            **kwargs)
    return result, seen


# --- the command line ------------------------------------------------------


def test_argv_is_the_fixed_prefix_then_the_callers_arguments(
        monkeypatch, fake_git, tmp_path):
    """The exact command line, including where the caller's ``--`` lands.

    ``core.askPass=`` is the one that is easy to lose and impossible to
    notice: without it, a machine with a credential helper configured opens
    a dialog on the SERVER and the request hangs until someone answers it.
    """
    _, seen = _run(monkeypatch, _FakeProc(),
                   ("add", "-A", "--", "src/a.txt", "b.txt"), cwd=tmp_path)

    assert seen["argv"] == [
        _GIT, "-C", str(tmp_path.resolve()),
        "-c", "core.quotepath=false",
        "-c", "core.askPass=",
        "-c", "color.ui=never",
        "add", "-A", "--", "src/a.txt", "b.txt",
    ]


def test_argv_carries_the_resolved_cwd(monkeypatch, fake_git, tmp_path):
    """``-C`` is an absolute path with no ``..`` left in it."""
    sub = tmp_path / "sub"
    sub.mkdir()

    _, seen = _run(monkeypatch, _FakeProc(), cwd=sub / "..")

    assert seen["argv"][1:3] == ["-C", str(tmp_path.resolve())]


def test_never_uses_a_shell(monkeypatch, fake_git):
    """A shell here would turn a branch name into a command."""
    _, seen = _run(monkeypatch, _FakeProc())

    assert "shell" not in seen["kwargs"]
    assert isinstance(seen["argv"], list)


def test_pipes_are_binary(monkeypatch, fake_git):
    """No text mode: git output is bytes, and both callers decode it once.

    ``text=True`` would translate the CRLF inside a diff hunk (making the
    patch describe a change that is not there) and raise on a file that is
    not valid UTF-8.
    """
    _, seen = _run(monkeypatch, _FakeProc())

    kwargs = seen["kwargs"]
    assert "text" not in kwargs
    assert "universal_newlines" not in kwargs
    assert "encoding" not in kwargs
    assert kwargs["stdout"] == subprocess.PIPE
    assert kwargs["stderr"] == subprocess.PIPE


def test_stdin_is_devnull_without_input(monkeypatch, fake_git):
    """A git that reads stdin gets EOF, not a wait with nobody typing."""
    _, seen = _run(monkeypatch, _FakeProc())

    assert seen["kwargs"]["stdin"] == subprocess.DEVNULL


def test_input_bytes_go_to_stdin(monkeypatch, fake_git):
    """A commit message travels on stdin, never as an argument."""
    proc = _FakeProc()

    _, seen = _run(monkeypatch, proc, ("commit", "-q", "--file=-"),
                   input_bytes="fix: \u4fee\u6b63".encode())

    assert seen["kwargs"]["stdin"] == subprocess.PIPE
    assert proc.communicate_calls[0][0] == "fix: \u4fee\u6b63".encode()


def test_popen_is_not_given_a_cwd(monkeypatch, fake_git, tmp_path):
    """``-C`` does the job, and a ``cwd=`` would confuse two failures.

    A directory Popen cannot enter raises ``FileNotFoundError`` -- the same
    exception a missing git raises -- and would be reported as "git is not
    installed" on a machine where it plainly is.
    """
    _, seen = _run(monkeypatch, _FakeProc(), cwd=tmp_path)

    assert "cwd" not in seen["kwargs"]


# --- the environment -------------------------------------------------------


def test_env_is_non_interactive(monkeypatch, fake_git):
    """Every variable that turns a question into an immediate failure."""
    env = runner.git_env()

    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GCM_INTERACTIVE"] == "never"
    assert env["SSH_ASKPASS_REQUIRE"] == "never"
    assert env["GIT_EDITOR"] == ":"
    assert env["GIT_SEQUENCE_EDITOR"] == ":"
    assert env["LC_ALL"] == "C"
    assert env["GIT_ALLOW_PROTOCOL"] == "https:ssh:file"


@pytest.mark.parametrize("name", runner.REPOSITORY_ENV_VARS)
def test_env_drops_the_repository_variables(monkeypatch, fake_git, name):
    """Any of these would move every command to another repository.

    A server started from a shell inside a git hook inherits ``GIT_DIR``,
    and every ``-C <project>`` in this module would then be ignored.
    """
    monkeypatch.setenv(name, "C:/somewhere/else/.git")

    assert name not in runner.git_env()


@pytest.mark.parametrize("name", ["PYTHONPATH", "PYTHONHOME"])
def test_env_drops_the_python_pointers(monkeypatch, fake_git, name):
    """Inherited from ``pip_env`` -- pinned here because it is a promise the
    git side relies on: a hook written in Python must not be run against
    this server's interpreter or this repository's ``app`` package."""
    monkeypatch.setenv(name, "D:/Github/CodefyUI/backend")

    assert name not in runner.git_env()


def test_env_keeps_the_users_own_ssh_command(monkeypatch):
    """A ``GIT_SSH_COMMAND`` the user set wins, and is not even second-guessed.

    Theirs may name a key, a jump host or a Windows OpenSSH binary. Ours
    knows none of that, so when the environment already answers the
    question the probe must not run at all.
    """
    monkeypatch.setattr(runner, "git_executable", lambda: _GIT)
    monkeypatch.setenv("GIT_SSH_COMMAND", "ssh -i C:/keys/id_ed25519")
    probes: list[int] = []
    monkeypatch.setattr(runner, "_ssh_command_configured",
                        lambda: probes.append(1) or False)

    assert runner.git_env()["GIT_SSH_COMMAND"] == "ssh -i C:/keys/id_ed25519"
    assert probes == []


def test_env_adds_batch_mode_when_nothing_is_configured(monkeypatch):
    """No env var and no ``core.sshCommand``: ssh must fail fast, not ask.

    Without batch mode, ssh prompts for a passphrase or a host-key
    confirmation on a server that has no terminal to answer it.
    """
    monkeypatch.setattr(runner, "git_executable", lambda: _GIT)
    monkeypatch.delenv("GIT_SSH_COMMAND", raising=False)
    # ``git config --get`` exits 1 for "unset", which is an ok_code here.
    _fake_popen(monkeypatch, _FakeProc(returncode=1))

    assert runner.git_env()["GIT_SSH_COMMAND"] == "ssh -oBatchMode=yes"


def test_env_leaves_ssh_alone_when_core_sshcommand_is_set(monkeypatch):
    """A user who configured ssh in git config gets their configuration."""
    monkeypatch.setattr(runner, "git_executable", lambda: _GIT)
    monkeypatch.delenv("GIT_SSH_COMMAND", raising=False)
    _fake_popen(monkeypatch, _FakeProc(stdout=b"ssh -i C:/keys/id_ed25519\n"))

    assert "GIT_SSH_COMMAND" not in runner.git_env()


def test_ssh_probe_runs_once_per_process(monkeypatch):
    """The answer cannot change often enough to be worth a process a poll."""
    monkeypatch.setattr(runner, "git_executable", lambda: _GIT)
    monkeypatch.delenv("GIT_SSH_COMMAND", raising=False)
    calls: list[list[str]] = []

    def _popen(argv, **kwargs):
        calls.append(argv)
        return _FakeProc(returncode=1)

    monkeypatch.setattr(subprocess, "Popen", _popen)

    runner.git_env()
    runner.git_env()

    assert len(calls) == 1
    assert calls[0][-3:] == ["config", "--get", "core.sshCommand"]


def test_ssh_probe_failure_is_not_fatal(monkeypatch):
    """A config the probe cannot read must not break every git command."""
    monkeypatch.setattr(runner, "git_executable", lambda: None)

    assert runner.git_env()["GIT_SSH_COMMAND"] == "ssh -oBatchMode=yes"


def test_read_only_skips_the_index_lock(monkeypatch, fake_git):
    """A status poll must not fight a real operation for the index lock."""
    assert "GIT_OPTIONAL_LOCKS" not in runner.git_env()
    assert runner.git_env(read_only=True)["GIT_OPTIONAL_LOCKS"] == "0"


def test_read_only_reaches_the_process(monkeypatch, fake_git):
    """...and the flag the caller passed to ``run_git`` is the one used."""
    _, seen = _run(monkeypatch, _FakeProc(), read_only=True)

    assert seen["kwargs"]["env"]["GIT_OPTIONAL_LOCKS"] == "0"


# --- the platform ----------------------------------------------------------


def test_windows_creationflags(monkeypatch, fake_git):
    """No console window over the editor, and a killable process GROUP."""
    monkeypatch.setattr(sys, "platform", "win32")

    _, seen = _run(monkeypatch, _FakeProc())

    flags = seen["kwargs"]["creationflags"]
    assert flags & packs_runner.CREATE_NO_WINDOW
    assert flags & packs_runner.CREATE_NEW_PROCESS_GROUP


def test_posix_creationflags_are_zero(monkeypatch, fake_git):
    """The Windows-only constants must not leak into a POSIX call."""
    monkeypatch.setattr(sys, "platform", "linux")

    _, seen = _run(monkeypatch, _FakeProc())

    assert seen["kwargs"]["creationflags"] == 0


# --- failures --------------------------------------------------------------


def test_timeout_kills_the_tree_and_reports_504(monkeypatch, fake_git):
    """A hung git is killed with its children, and says so as a 504.

    Killing the TREE matters: what hangs is usually the ssh or curl git
    started, and it holds the index lock the next request needs.
    """
    proc = _FakeProc(hangs=True)
    stopped: list = []
    monkeypatch.setattr(runner, "stop_process", stopped.append)
    _fake_popen(monkeypatch, proc)

    with pytest.raises(GitError) as excinfo:
        runner.run_git(["status"], cwd=Path.cwd(), timeout=10)

    assert excinfo.value.code == "timeout"
    assert excinfo.value.status == 504
    assert stopped == [proc]
    # The pipes are collected afterwards, so no zombie and no warning.
    assert len(proc.communicate_calls) == 2


def test_missing_git_never_spawns(monkeypatch):
    """``git_missing`` is decided before a process is started."""
    monkeypatch.setattr(runner, "git_executable", lambda: None)

    def _popen(argv, **kwargs):
        raise AssertionError("a process was started without a git")

    monkeypatch.setattr(subprocess, "Popen", _popen)

    with pytest.raises(GitError) as excinfo:
        runner.run_git(["status"], cwd=Path.cwd(), timeout=10)

    assert excinfo.value.code == "git_missing"
    assert excinfo.value.status == 503
    assert excinfo.value.hint


def test_git_that_vanished_is_git_missing(monkeypatch, fake_git):
    """git was on PATH when we looked and is not there now (an uninstall,
    a PATH edit): a reportable outcome, not a traceback."""
    def _popen(argv, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", _GIT)

    monkeypatch.setattr(subprocess, "Popen", _popen)

    with pytest.raises(GitError) as excinfo:
        runner.run_git(["status"], cwd=Path.cwd(), timeout=10)

    assert excinfo.value.code == "git_missing"


def test_git_that_cannot_be_executed_is_structured(monkeypatch, fake_git):
    """A permission bit on the executable is a 500 with a reason, not an
    ``OSError`` escaping into the response."""
    def _popen(argv, **kwargs):
        raise PermissionError(13, "Permission denied", _GIT)

    monkeypatch.setattr(subprocess, "Popen", _popen)

    with pytest.raises(GitError) as excinfo:
        runner.run_git(["status"], cwd=Path.cwd(), timeout=10)

    assert excinfo.value.code == "git_failed"
    assert excinfo.value.status == 500


def test_unexpected_return_code_is_classified(monkeypatch, fake_git):
    """A failing command comes back with a code the frontend can translate."""
    proc = _FakeProc(returncode=128,
                     stderr=b"fatal: Authentication failed for 'https://x/y'\n")
    _fake_popen(monkeypatch, proc)

    with pytest.raises(GitError) as excinfo:
        runner.run_git(["push"], cwd=Path.cwd(), timeout=10)

    assert excinfo.value.code == "auth_required"


def test_ok_codes_are_not_failures(monkeypatch, fake_git):
    """``diff`` exits 1 to mean "there are differences"; only the caller
    knows that, so only the caller can say so."""
    proc = _FakeProc(returncode=1, stdout=b"@@ -1 +1 @@\n")

    result, _ = _run(monkeypatch, proc, ("diff",), ok_codes=(0, 1))

    assert result.returncode == 1
    assert result.out.startswith("@@")


# --- what comes back -------------------------------------------------------


def test_crlf_survives_the_round_trip(monkeypatch, fake_git):
    """A diff of a CRLF file describes CRLF lines. Text mode would eat them."""
    proc = _FakeProc(stdout=b"+first\r\n+second\r\n")

    result, _ = _run(monkeypatch, proc)

    assert result.stdout == b"+first\r\n+second\r\n"
    assert result.out == "+first\r\n+second\r\n"


def test_undecodable_bytes_are_replaced_not_raised(monkeypatch, fake_git):
    """A latin-1 filename in a repository is not a 500."""
    proc = _FakeProc(stdout=b"caf\xe9.txt", stderr=b"\xff")

    result, _ = _run(monkeypatch, proc)

    assert result.out == "caf\ufffd.txt"
    assert result.err == "\ufffd"


def test_result_keeps_the_argv_it_ran(monkeypatch, fake_git, tmp_path):
    """What ran is part of the result -- a log line nobody has to reconstruct."""
    result, seen = _run(monkeypatch, _FakeProc(), cwd=tmp_path)

    assert result.argv == seen["argv"]


# --- discovering git -------------------------------------------------------


def test_git_executable_is_found_once(monkeypatch):
    """git does not move; PATH is scanned once per process."""
    calls: list[str] = []
    monkeypatch.setattr(runner, "shutil",
                        types.SimpleNamespace(
                            which=lambda name: calls.append(name) or _GIT))

    assert runner.git_executable() == _GIT
    assert runner.git_executable() == _GIT
    assert calls == ["git"]


def test_a_missing_git_is_rechecked_after_thirty_seconds(monkeypatch):
    """A machine without git must not scan PATH on every status poll -- and
    must still notice an install without a server restart."""
    clock = [1000.0]
    found: list[str | None] = [None]
    calls: list[str] = []
    # Patched on the runner's own references, not on the modules: a fake
    # clock installed globally would be inherited by pytest itself.
    monkeypatch.setattr(runner, "time",
                        types.SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(runner, "shutil",
                        types.SimpleNamespace(
                            which=lambda name: calls.append(name) or found[0]))

    assert runner.git_executable() is None
    clock[0] += runner.MISSING_RECHECK_S - 1
    assert runner.git_executable() is None
    assert calls == ["git"], "PATH was scanned again inside the window"

    clock[0] += 2
    found[0] = _GIT
    assert runner.git_executable() == _GIT
    assert len(calls) == 2


def test_git_version_ignores_the_vendor_suffix(monkeypatch, fake_git):
    """``git version 2.53.0.windows.2`` is 2.53.0: the vendor's own counter
    is not comparable with anyone else's."""
    _fake_popen(monkeypatch, _FakeProc(stdout=b"git version 2.53.0.windows.2\n"))

    assert runner.git_version() == (2, 53, 0)


def test_git_version_defaults_a_missing_patch(monkeypatch, fake_git):
    """Some builds report two components only."""
    _fake_popen(monkeypatch, _FakeProc(stdout=b"git version 2.39\n"))

    assert runner.git_version() == (2, 39, 0)


def test_git_version_is_read_once(monkeypatch, fake_git):
    """The answer cannot change while the process runs."""
    calls: list[list[str]] = []

    def _popen(argv, **kwargs):
        calls.append(argv)
        return _FakeProc(stdout=b"git version 2.53.0\n")

    monkeypatch.setattr(subprocess, "Popen", _popen)

    assert runner.git_version() == (2, 53, 0)
    assert runner.git_version() == (2, 53, 0)
    assert len(calls) == 1
    assert calls[0][-1] == "--version"


def test_git_version_without_git_is_none(monkeypatch):
    """No git, no version -- and no traceback out of a status read."""
    monkeypatch.setattr(runner, "git_executable", lambda: None)

    assert runner.git_version() is None


def test_git_version_survives_unreadable_output(monkeypatch, fake_git):
    """Something on PATH called ``git`` that is not git."""
    _fake_popen(monkeypatch, _FakeProc(stdout=b"this is not git\n"))

    assert runner.git_version() is None
