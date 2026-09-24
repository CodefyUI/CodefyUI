"""Ctrl+C through the launcher's Windows hop into the venv (#488).

On Windows `_reexec` cannot replace the process, so it runs the venv's
python as a child and forwards its exit code. A Ctrl+C at the console reaches
BOTH processes. The child handles it (`cdui run`, `cdui packs` and
`cdui cache` document exit code 130 for it). The launcher, waiting on the
child, used to get its own KeyboardInterrupt once the child had exited: a
traceback after the child's clean message, and exit status 0xC000013A
instead of the child's code.

No real signal is sent here: each `wait()` of a fake child follows a script,
and a KeyboardInterrupt in that script is the Ctrl+C reaching the launcher
while it waits. The fake also has what `subprocess.run` uses on top of
`wait()`, so an implementation that goes back to `run` fails this file on the
interrupt it lets escape, not on a missing method.
"""

from __future__ import annotations

import sys

import pytest

import dev  # scripts/dev.py -- conftest puts scripts/ on sys.path

#: What Windows reports for a process that Ctrl+C ended (STATUS_CONTROL_C_EXIT).
CTRL_C_EXIT = 0xC000013A


class _Child:
    """A `subprocess.Popen` whose waits follow a script.

    Each `wait()` takes the next outcome: an exception is raised, a number is
    the child's exit code.
    """

    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.args: list | None = None
        self.waits = 0
        self.killed = False
        self.returncode = None

    def wait(self, timeout=None):
        self.waits += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        self.returncode = outcome
        return outcome

    # What `subprocess.run` needs besides `wait()`.
    def communicate(self, input=None, timeout=None):
        self.wait()
        return None, None

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _reexec_on_windows(monkeypatch, outcomes: list):
    """Run `_reexec`'s Windows branch against a child whose waits follow
    *outcomes*. Returns the child and what ended `_reexec`: the exit code it
    passed to `sys.exit`, or the name of the exception that escaped it."""
    child = _Child(outcomes)

    def _popen(cmd, **kwargs):
        child.args = list(cmd)
        return child

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(dev.subprocess, "Popen", _popen)
    monkeypatch.setattr(dev, "_has_console_window", lambda: True)

    try:
        dev._reexec("C:/py/python.exe", ["dev.py", "run", "graph.json"])
    except SystemExit as exc:
        ended = exc.code
    except KeyboardInterrupt:
        ended = "KeyboardInterrupt"
    else:
        pytest.fail("_reexec returned; it has to end this process")
    assert child.args == ["C:/py/python.exe", "dev.py", "run", "graph.json"]
    return child, ended


@pytest.mark.parametrize(
    ("outcomes", "code"),
    [
        ([KeyboardInterrupt(), 130], 130),
        ([KeyboardInterrupt(), CTRL_C_EXIT], 130),
        ([KeyboardInterrupt(), 0], 0),
        ([KeyboardInterrupt(), KeyboardInterrupt(), 1], 1),
    ],
    ids=["child-exits-130", "child-killed-by-ctrl-c", "child-exits-0",
         "second-ctrl-c"],
)
def test_ctrl_c_ends_the_launcher_with_the_childs_exit_code(monkeypatch,
                                                            outcomes, code):
    """The child got the same Ctrl+C and decides what it means, so the
    launcher waits for it and exits with its code -- or with 130 when the
    Ctrl+C killed the child outright, the code the help text documents. The
    child is never killed from here: it may still be cleaning up."""
    child, ended = _reexec_on_windows(monkeypatch, outcomes)

    assert ended == code
    assert child.waits == len(outcomes), "stopped waiting before the child exited"
    assert not child.killed


@pytest.mark.parametrize(
    ("code", "passed"),
    [(3, 3), (0xC0000005, 0xC0000005 - (1 << 32))],
    ids=["failed", "crashed"],
)
def test_without_a_ctrl_c_the_childs_exit_code_is_forwarded(monkeypatch, code,
                                                            passed):
    child, ended = _reexec_on_windows(monkeypatch, [code])

    assert ended == passed
    assert child.waits == 1


@pytest.mark.parametrize(
    ("code", "passed"),
    [
        (0, 0),
        (1, 1),
        (130, 130),
        (CTRL_C_EXIT, 130),
        (0x7FFFFFFF, 0x7FFFFFFF),
        (0xC0000005, 0xC0000005 - (1 << 32)),
        (0xFFFFFFFF, -1),
    ],
    ids=["ok", "failed", "exited-130", "killed-by-ctrl-c", "largest-positive",
         "access-violation", "all-bits-set"],
)
def test_the_childs_status_is_passed_on_in_a_form_sys_exit_keeps(code, passed):
    """`sys.exit` on Windows squeezes its argument into a 32-bit signed C
    long. A child that died with an NTSTATUS (0xC0000005 is an access
    violation) reports more than 0x7FFFFFFF, which overflows: the launcher
    exited 0xFFFFFFFF and hid what the child died of. The same 32 bits as a
    negative number exit with exactly the child's status."""
    assert dev._reexec_exit_code(code) == passed
