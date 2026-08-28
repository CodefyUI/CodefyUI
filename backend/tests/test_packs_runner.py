"""Running ``uv pip install`` on behalf of a learner who is watching.

The runner is the only place in the Package Center that starts a process, so
it is the only place that can get the four things below wrong, and each of
them fails in a way nobody would connect back to it:

* no shell, ever. The specs come from the catalog today, but a shell in this
  path would make the day one arrives from a request body catastrophic
  instead of merely wrong;
* on Windows, no console window flashing up over the editor, and a process
  GROUP so that cancelling kills the resolver's children too;
* a sanitised environment -- ``PYTHONPATH`` from a dev shell would put the
  repo's own ``app`` package on the child's path;
* cancellation that actually stops the process, rather than orphaning a
  multi-gigabyte download that keeps running after the UI says "cancelled".

No test here starts a real process: ``subprocess.Popen`` is replaced with a
fake that hands back canned output.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from app.core.packs import runner
from app.core.packs.errors import PackCancelled


class _FakePipe:
    """A stdout pipe that can be iterated and, like a real one, CLOSED."""

    def __init__(self, lines: list[str]):
        self._lines = iter(lines)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self) -> str:
        if self.closed:
            raise StopIteration
        return next(self._lines)

    def close(self) -> None:
        self.closed = True


class _FakeProc:
    """A finished (or never-finishing) ``uv`` process, without the process."""

    def __init__(self, lines: list[str], *, returncode: int | None = 0):
        self.stdout = _FakePipe(lines)
        self._returncode = returncode
        self.pid = 4242
        self.terminated = False
        self.killed = False
        self.wait_timeouts: list[float | None] = []

    def poll(self):
        return self._returncode

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if self._returncode is None:
            raise subprocess.TimeoutExpired(cmd="uv", timeout=timeout or 0)
        return self._returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def _fake_popen(monkeypatch, proc: _FakeProc) -> dict:
    """Install *proc* as the next ``Popen`` result; returns the recorded call."""
    seen: dict = {}

    def _popen(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return proc

    monkeypatch.setattr(subprocess, "Popen", _popen)
    return seen


def _run(proc, monkeypatch, *, cancel=False, specs=("demo-pkg>=1",),
         constraints_path=None, tail=None):
    """Call ``run_pip`` against *proc*; returns ``(rc, events, recorded call)``."""
    events: list[dict] = []
    seen = _fake_popen(monkeypatch, proc)
    rc = runner.run_pip(
        specs,
        constraints_path=constraints_path,
        emit=events.append,
        cancel_check=lambda: cancel,
        cwd=Path.cwd(),
        tail=tail,
    )
    return rc, events, seen


def test_pip_install_argv_shape(monkeypatch, tmp_path):
    """The exact command line, including the two flags that are easy to lose."""
    monkeypatch.setattr(runner, "find_uv", lambda: "C:/tools/uv.exe")
    constraints = tmp_path / "constraints.txt"

    argv = runner.pip_install_argv(("a>=1,<2", "b"),
                                   constraints_path=constraints,
                                   python="/opt/venv/bin/python")

    assert argv == [
        "C:/tools/uv.exe", "pip", "install",
        "--python", "/opt/venv/bin/python",
        "--no-progress",
        "-c", str(constraints),
        "a>=1,<2", "b",
    ]


def test_pip_install_argv_defaults_to_this_interpreter(monkeypatch):
    """No constraints file, no uv on PATH: still a plain argv aimed at THIS
    interpreter, because that is the one the constraints file describes."""
    monkeypatch.setattr(runner, "find_uv", lambda: None)

    argv = runner.pip_install_argv(("a",), constraints_path=None)

    assert argv == ["uv", "pip", "install", "--python", sys.executable,
                    "--no-progress", "a"]


def test_pip_install_argv_dry_run_is_opt_in(monkeypatch):
    """``--dry-run`` appears only when asked for, and BEFORE the specs.

    Nothing in the app passes it -- ``tests/test_packs_network.py`` does, to
    ask a live index whether a pack would resolve here without installing
    anything into the interpreter running the question. A flag that leaked
    into the default would turn every real install into a no-op that reported
    success, which is the one failure this test exists to keep impossible.
    """
    monkeypatch.setattr(runner, "find_uv", lambda: None)

    assert "--dry-run" not in runner.pip_install_argv(
        ("a",), constraints_path=None)
    assert runner.pip_install_argv(("a",), constraints_path=None,
                                   dry_run=True) == [
        "uv", "pip", "install", "--python", sys.executable,
        "--no-progress", "--dry-run", "a"]


def test_run_pip_streams_log_lines(monkeypatch):
    """Every output line becomes a log event; blank lines and newlines do not."""
    proc = _FakeProc(["Resolved 3 packages\n", "\n", "Installed 3 packages\n"])

    rc, events, seen = _run(proc, monkeypatch)

    assert rc == 0
    lines = [event["line"] for event in events if event["type"] == "log"]
    assert lines[0] == " ".join(seen["argv"]), "the command itself was not logged"
    assert lines[1:] == ["Resolved 3 packages", "Installed 3 packages"]


def test_run_pip_keeps_the_last_output_lines_for_the_caller(monkeypatch):
    """The tail is what a failure message is built from, so it is capped."""
    proc = _FakeProc([f"line {n}\n" for n in range(200)], returncode=1)
    tail: list[str] = []

    rc, _, _ = _run(proc, monkeypatch, tail=tail)

    assert rc == 1
    assert len(tail) == runner.TAIL_LINES
    assert tail[-1] == "line 199"


def test_run_pip_never_uses_shell(monkeypatch):
    """A shell here would turn a spec into a command."""
    _, _, seen = _run(_FakeProc([]), monkeypatch)

    assert "shell" not in seen["kwargs"]
    assert isinstance(seen["argv"], list)


def test_run_pip_windows_creationflags(monkeypatch):
    """No console window over the editor, and a killable process GROUP."""
    monkeypatch.setattr(sys, "platform", "win32")

    _, _, seen = _run(_FakeProc([]), monkeypatch)

    flags = seen["kwargs"]["creationflags"]
    assert flags & runner.CREATE_NO_WINDOW
    assert flags & runner.CREATE_NEW_PROCESS_GROUP


def test_run_pip_posix_creationflags_are_zero(monkeypatch):
    """The Windows-only constants must not leak into a POSIX call."""
    monkeypatch.setattr(sys, "platform", "linux")

    _, _, seen = _run(_FakeProc([]), monkeypatch)

    assert seen["kwargs"]["creationflags"] == 0


def test_run_pip_env_sanitised(monkeypatch):
    """UTF-8 in, ``PYTHONPATH`` out; the rest of the environment survives."""
    monkeypatch.setenv("PYTHONPATH", "D:/Github/CodefyUI/backend")
    monkeypatch.setenv("CODEFYUI_MARKER", "kept")

    _, _, seen = _run(_FakeProc([]), monkeypatch)

    env = seen["kwargs"]["env"]
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert "PYTHONPATH" not in env
    assert env["CODEFYUI_MARKER"] == "kept"


def test_run_pip_cancel_terminates_the_process_tree_on_windows(monkeypatch):
    """Windows: kill the whole tree, because uv's children hold the download.

    And kill it the way everything else here starts a process: with a
    deadline, so a wedged ``taskkill`` cannot hold the caller's thread
    forever, and with no console window of its own.
    """
    monkeypatch.setattr(sys, "platform", "win32")
    killed: list[tuple] = []
    monkeypatch.setattr(subprocess, "run",
                        lambda argv, **kwargs: killed.append((argv, kwargs)))
    proc = _FakeProc([], returncode=None)

    with pytest.raises(PackCancelled):
        _run(proc, monkeypatch, cancel=True)

    argv, kwargs = killed[0]
    assert argv == ["taskkill", "/F", "/T", "/PID", "4242"]
    assert kwargs["timeout"] == runner.KILL_TIMEOUT_S
    assert kwargs["creationflags"] & runner.CREATE_NO_WINDOW


def test_run_pip_survives_a_taskkill_that_hangs(monkeypatch):
    """A kill that never returns must not become a job that never ends."""
    monkeypatch.setattr(sys, "platform", "win32")

    def _hangs(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=10)

    monkeypatch.setattr(subprocess, "run", _hangs)

    with pytest.raises(PackCancelled):
        _run(_FakeProc([], returncode=None), monkeypatch, cancel=True)


def test_run_pip_closes_the_pipe_when_cancelled(monkeypatch):
    """The reader is a daemon and ``join`` has a timeout, so it can outlive
    the call. Closing the pipe ends its read -- otherwise a thread nobody is
    waiting for goes on emitting log lines into a cancelled job."""
    monkeypatch.setattr(sys, "platform", "linux")
    proc = _FakeProc(["still going\n"], returncode=None)

    with pytest.raises(PackCancelled):
        _run(proc, monkeypatch, cancel=True)

    assert proc.stdout.closed


def test_run_pip_closes_the_pipe_after_a_normal_exit(monkeypatch):
    proc = _FakeProc(["Installed 3 packages\n"])

    _run(proc, monkeypatch)

    assert proc.stdout.closed


def test_run_pip_cancel_terminates_then_kills_on_posix(monkeypatch):
    """POSIX: ask nicely, then insist after three seconds."""
    monkeypatch.setattr(sys, "platform", "linux")
    proc = _FakeProc([], returncode=None)

    with pytest.raises(PackCancelled):
        _run(proc, monkeypatch, cancel=True)

    assert proc.terminated
    assert proc.wait_timeouts == [3]
    assert proc.killed, "a process that ignored terminate was left running"


def test_run_pip_uv_missing_returns_127(monkeypatch):
    """uv absent is a reportable outcome, not a traceback."""
    def _popen(argv, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "uv")

    monkeypatch.setattr(subprocess, "Popen", _popen)
    events: list[dict] = []

    rc = runner.run_pip(("a",), constraints_path=None, emit=events.append,
                        cancel_check=lambda: False, cwd=Path.cwd())

    assert rc == 127
    assert any("uv" in event.get("line", "") for event in events)


@pytest.mark.parametrize("line", [
    "  x No solution found when resolving dependencies:",
    "  \u00d7 No solution found when resolving dependencies:",
    "Because torch==2.11.0+cu128 depends on ...",
    "  \u2570\u2500\u25b6 Because torch==2.11.0+cu128 depends on ...",
    "constraint torch==2.11.0 is not satisfiable",
])
def test_resolver_conflict_is_recognised(line):
    """uv's ways of saying "your pins and this pack disagree", each of them
    first on the line behind whatever gutter that version draws."""
    assert runner.looks_like_resolver_conflict(["Resolved 0 packages", line])


@pytest.mark.parametrize("lines", [
    pytest.param(["error: Failed to fetch: https://pypi.org/simple/demo/",
                  "  Caused by: Connection reset by peer"], id="network"),
    pytest.param(["error: Failed to build `sentencepiece==0.2.0`",
                  "  The build backend returned an error because the "
                  "compiler crashed"], id="because-mid-sentence"),
    pytest.param(["error: the wheel violates a constraint of this platform"],
                 id="constraint-mid-sentence"),
])
def test_ordinary_failure_is_not_a_resolver_conflict(lines):
    """A failure that merely CONTAINS one of the words is not a version
    conflict, and must not tell the user to restart the server: the words
    are anchored to the start of the line, where uv puts them."""
    assert not runner.looks_like_resolver_conflict(lines)
