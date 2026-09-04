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

import os
import signal
import subprocess
import sys
import tempfile
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
                 returncode: int = 0, hangs: bool = False,
                 ignores_sigterm: bool = False):
        self._stdout = stdout
        self._stderr = stderr
        self._returncode = returncode
        self._hangs = hangs
        self._ignores_sigterm = ignores_sigterm
        self.pid = 4242
        self.returncode: int | None = None
        self.communicate_calls: list[tuple] = []
        self.wait_calls: list[float | None] = []

    def communicate(self, data=None, timeout=None):
        self.communicate_calls.append((data, timeout))
        if self._hangs and len(self.communicate_calls) == 1:
            raise subprocess.TimeoutExpired(cmd="git", timeout=timeout or 0)
        self.returncode = self._returncode
        return self._stdout, self._stderr

    def wait(self, timeout=None):
        """What the POSIX kill path waits on between its two signals."""
        self.wait_calls.append(timeout)
        if self._ignores_sigterm:
            raise subprocess.TimeoutExpired(cmd="git", timeout=timeout or 0)
        self.returncode = -15
        return self.returncode


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


@pytest.fixture(autouse=True)
def _no_inherited_ssh_setting(monkeypatch):
    """No test here inherits the developer's own ssh configuration.

    BOTH variables, because either one answers the question ``git_env``
    asks: a machine that exports ``GIT_SSH`` -- which a corporate setup
    really does -- would otherwise skip the probe and turn every assertion
    below about batch mode into a pass for the wrong reason. The tests that
    are ABOUT one of them set it back.
    """
    for name in runner.SSH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


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

    ``--literal-pathspecs`` is the one that is easy to lose and DANGEROUS:
    without it the paths after ``--`` are patterns, so a request naming
    ``[.]env`` or ``*`` gets a file nobody validated.
    """
    _, seen = _run(monkeypatch, _FakeProc(),
                   ("add", "-A", "--", "src/a.txt", "b.txt"), cwd=tmp_path)

    assert seen["argv"] == [
        _GIT, "-C", str(tmp_path.resolve()),
        "--literal-pathspecs",
        "-c", "core.quotepath=false",
        "-c", "core.askPass=",
        "-c", "color.ui=never",
        "add", "-A", "--", "src/a.txt", "b.txt",
    ]


def test_dropping_literal_pathspecs_drops_only_that_one(
        monkeypatch, fake_git, tmp_path):
    """The exemption, pinned at the argv rather than through a symptom.

    Two callers ask for it and neither can be tested for it behaviourally
    in a way that survives git changing its mind: ``check-ignore`` REFUSES
    the option outright, and ``stash push`` accepts it and then leaves the
    untracked file it just stashed sitting in the working tree. A git that
    fixed the second would make its behavioural test pass either way, and
    the exemption could then be deleted in silence.

    The other three options are not optional, and this says so: dropping
    ``core.askPass=`` by accident here would hang a request on a machine
    with a credential helper.
    """
    _, seen = _run(monkeypatch, _FakeProc(), ("stash", "push"), cwd=tmp_path,
                   literal_pathspecs=False)

    assert seen["argv"] == [
        _GIT, "-C", str(tmp_path.resolve()),
        "-c", "core.quotepath=false",
        "-c", "core.askPass=",
        "-c", "color.ui=never",
        "stash", "push",
    ]
    assert runner.LITERAL_PATHSPECS not in seen["argv"]


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
    assert env["GIT_ASKPASS"] == ""
    assert env["SSH_ASKPASS_REQUIRE"] == "never"
    assert env["GIT_EDITOR"] == ":"
    assert env["GIT_SEQUENCE_EDITOR"] == ":"
    assert env["LC_ALL"] == "C"
    assert env["GIT_ALLOW_PROTOCOL"] == "https:ssh:file"


def test_an_inherited_askpass_helper_is_emptied(monkeypatch, fake_git):
    """The variable beats ``-c core.askPass=``, so it has to be overwritten.

    VS Code exports a ``GIT_ASKPASS`` into every terminal it opens, which is
    where a developer starts this server; git reads the variable before the
    config and never looks at the config when it is set. Inherited, it would
    open a dialog on the server's own screen and hold the request until
    somebody clicked it. Empty is git's own spelling of "no helper", and it
    ends the chain rather than passing the question to ``SSH_ASKPASS``.
    """
    monkeypatch.setenv("GIT_ASKPASS", "C:/Program Files/VSCode/askpass.sh")

    assert runner.git_env()["GIT_ASKPASS"] == ""


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


def test_env_keeps_the_legacy_git_ssh_variable(monkeypatch):
    """``GIT_SSH`` is a setting too, and ours would REPLACE it silently.

    It is the old spelling -- it names an ssh BINARY rather than a command
    line -- and git reads ``GIT_SSH_COMMAND`` in preference to it. So a
    default written into the modern variable does not sit beside a
    ``GIT_SSH``, it overrides it: the wrapper script a corporate setup puts
    there (a key, a jump host, a smartcard) would never run, and the user
    would have no way to tell from the failure.

    The probe must not run either: the environment has already answered.
    """
    monkeypatch.setattr(runner, "git_executable", lambda: _GIT)
    monkeypatch.setenv("GIT_SSH", "C:/corp/ssh-wrapper.exe")
    probes: list[int] = []
    monkeypatch.setattr(runner, "_ssh_command_configured",
                        lambda: probes.append(1) or False)

    env = runner.git_env()

    assert "GIT_SSH_COMMAND" not in env
    assert env["GIT_SSH"] == "C:/corp/ssh-wrapper.exe"
    assert probes == []


def test_env_adds_batch_mode_when_nothing_is_configured(monkeypatch):
    """No env var and no ``core.sshCommand``: ssh must fail fast, not ask.

    Without batch mode, ssh prompts for a passphrase or a host-key
    confirmation on a server that has no terminal to answer it.
    """
    monkeypatch.setattr(runner, "git_executable", lambda: _GIT)
    for name in runner.SSH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # ``git config --get`` exits 1 for "unset", which is an ok_code here.
    _fake_popen(monkeypatch, _FakeProc(returncode=1))

    assert runner.git_env()["GIT_SSH_COMMAND"] == "ssh -oBatchMode=yes"


def test_env_leaves_ssh_alone_when_core_sshcommand_is_set(monkeypatch):
    """A user who configured ssh in git config gets their configuration."""
    monkeypatch.setattr(runner, "git_executable", lambda: _GIT)
    _fake_popen(monkeypatch, _FakeProc(stdout=b"ssh -i C:/keys/id_ed25519\n"))

    assert "GIT_SSH_COMMAND" not in runner.git_env()


def test_ssh_probe_runs_once_per_process(monkeypatch):
    """The answer cannot change often enough to be worth a process a poll."""
    monkeypatch.setattr(runner, "git_executable", lambda: _GIT)
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


def test_the_probe_never_asks_the_servers_own_repository(monkeypatch, tmp_path):
    """The probe runs from the temp directory, not from ``Path.cwd()``.

    The server's working directory is its own checkout -- a repository with
    its own config, and one the Source Control tab must never consult. A
    ``core.sshCommand`` set THERE would otherwise decide how the user's
    project reaches its remotes.
    """
    monkeypatch.setattr(runner, "git_executable", lambda: _GIT)
    # A repository, standing in for the checkout the server runs from.
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    seen = _fake_popen(monkeypatch, _FakeProc(returncode=1))

    runner.git_env()

    assert seen["argv"][1:3] == [
        "-C", str(Path(tempfile.gettempdir()).resolve())]
    assert str(tmp_path) not in seen["argv"]


def test_a_probe_that_failed_is_not_repeated_on_every_command(monkeypatch):
    """A host whose config cannot be read pays one probe per interval.

    ``git_env`` is called once per git command and the tab polls the status
    endpoint while it is open, so a probe that is retried on failure is a
    whole process per request, forever. The failure is remembered exactly as
    long as a missing binary is.
    """
    clock = [1000.0]
    calls: list[list[str]] = []
    monkeypatch.setattr(runner, "git_executable", lambda: _GIT)
    monkeypatch.setattr(runner, "time",
                        types.SimpleNamespace(monotonic=lambda: clock[0]))

    def _popen(argv, **kwargs):
        calls.append(argv)
        # git is there and answering; reading the config is what fails.
        return _FakeProc(returncode=128, stderr=b"fatal: bad config line 1\n")

    monkeypatch.setattr(subprocess, "Popen", _popen)

    for _ in range(4):
        assert runner.git_env()["GIT_SSH_COMMAND"] == "ssh -oBatchMode=yes"
    assert len(calls) == 1, "a broken host paid a process per command"

    clock[0] += runner.MISSING_RECHECK_S + 1
    assert runner.git_env()["GIT_SSH_COMMAND"] == "ssh -oBatchMode=yes"
    assert len(calls) == 2, "the failure was never rechecked"


def test_a_probe_that_could_not_run_is_not_remembered(monkeypatch):
    """A guess must not outlive the thing that caused it.

    If git was missing when the question was first asked, the fallback is
    right for that moment -- and wrong forever after, because it would go on
    overriding a ``core.sshCommand`` the user really has.
    """
    monkeypatch.setattr(runner, "git_executable", lambda: None)
    assert runner.git_env()["GIT_SSH_COMMAND"] == "ssh -oBatchMode=yes"

    monkeypatch.setattr(runner, "git_executable", lambda: _GIT)
    _fake_popen(monkeypatch, _FakeProc(stdout=b"ssh -i C:/keys/id_ed25519\n"))

    assert "GIT_SSH_COMMAND" not in runner.git_env()


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


def test_a_posix_child_gets_a_session_of_its_own(monkeypatch, fake_git):
    """``start_new_session`` is the whole POSIX kill story in one kwarg.

    Without it, git and the ``ssh`` it starts sit in the SERVER's process
    group, and the only thing a timeout can reach is git itself -- so the
    ssh that is actually blocked on the network survives, holding the index
    lock the next request needs. With it, the child is its own session
    leader and its pid IS the group id.
    """
    monkeypatch.setattr(sys, "platform", "linux")

    _, seen = _run(monkeypatch, _FakeProc())

    assert seen["kwargs"]["start_new_session"] is True


def test_a_windows_child_is_started_exactly_as_before(monkeypatch, fake_git):
    """Windows takes its group from ``creationflags`` and ignores the kwarg.

    Passing it anyway would be a no-op that reads like a promise, and the
    kill there is ``taskkill /F /T`` -- which walks the tree from the pid
    and needs nothing from the way the child was started.
    """
    monkeypatch.setattr(sys, "platform", "win32")

    _, seen = _run(monkeypatch, _FakeProc())

    assert "start_new_session" not in seen["kwargs"]


# --- failures --------------------------------------------------------------


def _posix_kill(monkeypatch) -> list[tuple[int, int]]:
    """Fake the POSIX group-kill calls; returns what was signalled.

    ``os.killpg``, ``os.getpgid`` and ``signal.SIGKILL`` do not exist on
    Windows, which is where this suite usually runs -- so all three are
    installed with ``raising=False``. On Linux, where they do exist, the
    fakes replace the real ones for the length of the test, which is the
    point: no test in this file may signal anything.
    """
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(os, "getpgid", lambda pid: pid, raising=False)
    monkeypatch.setattr(
        os, "killpg", lambda pgid, sig: sent.append((pgid, sig)),
        raising=False)
    return sent


def test_timeout_kills_the_tree_and_reports_504(monkeypatch, fake_git):
    """A hung git is killed with its children, and says so as a 504.

    Killing the TREE matters: what hangs is usually the ssh or curl git
    started, and it holds the index lock the next request needs. On Windows
    that is ``taskkill /F /T`` from the pid, which is ``stop_process`` --
    the platform is stated because the POSIX path below is a different kill
    entirely.
    """
    monkeypatch.setattr(sys, "platform", "win32")
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


def test_a_timeout_on_posix_signals_the_whole_group(monkeypatch, fake_git):
    """``SIGTERM`` to the GROUP, not ``terminate()`` to the child alone.

    ``stop_process`` on POSIX is ``proc.terminate()``, which reaches git and
    nothing it started -- so the ssh blocked on a network read survives its
    parent. The signal goes to the negative-pid group instead, and a child
    that stops on the first one is never sent the second.
    """
    sent = _posix_kill(monkeypatch)
    proc = _FakeProc(hangs=True)
    monkeypatch.setattr(runner, "stop_process",
                        lambda _proc: pytest.fail("the Windows kill ran"))
    _fake_popen(monkeypatch, proc)

    with pytest.raises(GitError) as excinfo:
        runner.run_git(["fetch"], cwd=Path.cwd(), timeout=10)

    assert excinfo.value.code == "timeout"
    assert sent == [(proc.pid, signal.SIGTERM)]
    assert proc.wait_calls == [runner.DRAIN_TIMEOUT_S]
    assert len(proc.communicate_calls) == 2


def test_a_posix_child_that_ignores_sigterm_is_killed(monkeypatch, fake_git):
    """A credential helper that will not stop is not allowed to hold on.

    It holds the pipes, and the index lock behind them, for as long as it
    likes -- so the second signal is the one that cannot be ignored, sent
    after the same grace ``_drain`` gives.
    """
    sent = _posix_kill(monkeypatch)
    proc = _FakeProc(hangs=True, ignores_sigterm=True)
    _fake_popen(monkeypatch, proc)

    with pytest.raises(GitError):
        runner.run_git(["fetch"], cwd=Path.cwd(), timeout=10)

    assert sent == [(proc.pid, signal.SIGTERM), (proc.pid, signal.SIGKILL)]


def test_a_posix_group_that_is_already_gone_is_not_an_error(monkeypatch,
                                                            fake_git):
    """A child that exited between the timeout and the signal is the
    outcome being asked for, not a second failure on top of the 504."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)

    def _gone(pid):
        raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr(os, "getpgid", _gone, raising=False)
    monkeypatch.setattr(
        os, "killpg",
        lambda pgid, sig: pytest.fail("signalled a group that is gone"),
        raising=False)
    _fake_popen(monkeypatch, _FakeProc(hangs=True))

    with pytest.raises(GitError) as excinfo:
        runner.run_git(["fetch"], cwd=Path.cwd(), timeout=10)

    assert excinfo.value.code == "timeout"


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
