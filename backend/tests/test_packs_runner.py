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


class _FakeProc:
    """A finished (or never-finishing) ``uv`` process, without the process."""

    def __init__(self, lines: list[str], *, returncode: int | None = 0):
        self.stdout = iter(lines)
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
    """Windows: kill the whole tree, because uv's children hold the download."""
    monkeypatch.setattr(sys, "platform", "win32")
    killed: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run",
                        lambda argv, **kwargs: killed.append(argv))
    proc = _FakeProc([], returncode=None)

    with pytest.raises(PackCancelled):
        _run(proc, monkeypatch, cancel=True)

    assert killed == [["taskkill", "/F", "/T", "/PID", "4242"]]


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
    "Because torch==2.11.0+cu128 depends on ...",
    "constraint torch==2.11.0 is not satisfiable",
])
def test_resolver_conflict_is_recognised(line):
    """uv's three ways of saying "your pins and this pack disagree"."""
    assert runner.looks_like_resolver_conflict(["Resolved 0 packages", line])


def test_ordinary_failure_is_not_a_resolver_conflict():
    """A network failure must not be reported as a version conflict."""
    assert not runner.looks_like_resolver_conflict(
        ["error: Failed to fetch: https://pypi.org/simple/demo/",
         "  Caused by: Connection reset by peer"])
