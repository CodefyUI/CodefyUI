"""Run ``uv pip install`` for a pack, and report it line by line.

One subprocess, watched from two threads: a daemon reader that turns the
child's output into ``log`` events as it arrives, and the caller's thread
polling for "has it finished" and "did the user press Stop". The split is
what makes the install cancellable at all -- reading a pipe blocks, and a
blocked thread cannot check anything.

Four decisions here are not stylistic:

* **``--python sys.executable``**. Bare ``uv pip install`` looks for a
  ``.venv`` next to the CURRENT WORKING DIRECTORY, which for a server
  started from the repo root is the wrong place (or nowhere). The
  constraints file this runs under describes THIS interpreter, so the
  install has to land in THIS interpreter or the pins describe a different
  machine than the one being changed. ``scripts/plugins.py`` pins it for
  the same reason.
* **never a shell**. ``uv`` is happy to fetch ``git+`` URLs; a shell would
  add every other program on the box to that list. The specs come from the
  catalog allowlist today, and this stays safe on the day one does not.
* **``CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP`` on Windows**. The first
  keeps a console from flashing over the editor on every install. The
  second gives the child its own group, so cancelling can kill uv AND the
  downloader it spawned -- terminating uv alone leaves a process holding
  the network and the cache lock.
* **a sanitised environment**. ``PYTHONPATH`` is dropped: a developer shell
  that puts ``backend/`` on it would put this repo's ``app`` package inside
  the installer's interpreter. So are ``PYTHONHOME`` and the two variables
  that move ``sys.executable`` -- see :data:`_STDLIB_POINTER_VARS`, which
  exists because one of them really did make a child load another
  interpreter's standard library. ``PYTHONUTF8``/``PYTHONIOENCODING`` are set
  because on a cp950 or cp1252 console uv's output is otherwise undecodable
  and the log events arrive as mojibake.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .errors import PackCancelled

#: How many output lines to keep for a failure message. Enough to hold uv's
#: whole "No solution found" explanation, short enough to put in a toast.
TAIL_LINES = 40

#: How often the caller's thread asks "finished? cancelled?". A quarter of a
#: second is under the threshold where a Stop button feels broken, and costs
#: four wakeups a second next to a process that is saturating a disk.
POLL_INTERVAL_S = 0.25

#: How long a terminated POSIX process gets to exit before it is killed.
TERMINATE_GRACE_S = 3

#: Windows process-creation flags, looked up rather than referenced: the
#: constants do not exist on POSIX, and ``test_run_pip_windows_creationflags``
#: monkeypatches ``sys.platform`` on a Linux CI runner.
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)

#: How long ``taskkill`` gets before we stop waiting for it.
KILL_TIMEOUT_S = 10

#: What uv says when the pack and this interpreter's pins cannot both be
#: satisfied. Its resolver explains itself with "No solution found" followed
#: by a chain of "Because ..." lines; "constraint" catches the wording used
#: when the constraints file itself is the blocker.
#:
#: Anchored to the START of a line, and that is the whole point: "because" is
#: an ordinary English word that turns up in perfectly ordinary failures
#: ("failed to build because the compiler crashed"), and matching it anywhere
#: would answer a broken toolchain with "restart the server and try again".
#: uv puts its markers first on the line, behind nothing but indentation and
#: a bullet -- ``x``, ``\u00d7`` or a box-drawing gutter -- which is what the
#: leading group skips. Matched case-insensitively, on the joined output.
_CONFLICT_PATTERN = re.compile(
    r"^[^\w\n]*(?:x[^\w\n]+)?"
    r"(?:no solution found|because\b|constraint\b)",
    re.MULTILINE)


def find_uv() -> str | None:
    """The ``uv`` executable, or None when it is not on PATH."""
    return shutil.which("uv")


def pip_install_argv(
    specs: Sequence[str],
    *,
    constraints_path: Path | None,
    python: str = sys.executable,
    dry_run: bool = False,
) -> list[str]:
    """The full command line for installing *specs*.

    Built as a separate function so a test can assert the shape of a command
    nobody should have to run to find out what it is. ``uv`` falls back to
    the bare name when it is not on PATH: the resulting ``FileNotFoundError``
    is what :func:`run_pip` turns into a reportable 127.

    ``dry_run=True`` adds uv's ``--dry-run``: it resolves the specs against
    the constraints and prints what it WOULD do without writing anything to
    the interpreter. Nothing in the app passes it -- it exists so
    ``tests/test_packs_network.py`` can ask "would this pack install on this
    machine?" without mutating the venv the question is about, which is the
    only honest way to run that check against a live index.
    """
    return [
        find_uv() or "uv",
        "pip", "install",
        "--python", python,
        "--no-progress",
        *(["--dry-run"] if dry_run else []),
        *(["-c", str(constraints_path)] if constraints_path is not None else []),
        *specs,
    ]


def looks_like_resolver_conflict(lines: Sequence[str]) -> bool:
    """Did uv fail because it could not satisfy the pins, rather than the net?

    The distinction decides what the user is told to do next: a resolver
    conflict means "this cannot be done inside the running server, restart
    and try again", while a failed download means "try again".
    """
    return _CONFLICT_PATTERN.search("\n".join(lines).lower()) is not None


#: Variables that tell a Python process where its standard library and its
#: ``sys.executable`` are. Every one of them is a statement about THIS
#: interpreter, and every child here is a different one.
#:
#: ``PYTHONHOME`` is not a theoretical risk: a uv-managed virtualenv's
#: ``python.exe`` is a trampoline that runs the server with ``PYTHONHOME``
#: set to the base interpreter it was built from, so a copied environment
#: carries it. The restart helper -- the OUTER interpreter, a different
#: version -- then loaded that base's stdlib and died on ``import argparse``
#: with "AssertionError: SRE module mismatch", detached, after the server had
#: already scheduled its own shutdown. ``PYTHONEXECUTABLE`` and the macOS
#: framework launcher's ``__PYVENV_LAUNCHER__`` move ``sys.executable`` the
#: same way, and ``site`` derives the rest of the paths from it.
_STDLIB_POINTER_VARS = ("PYTHONHOME", "__PYVENV_LAUNCHER__", "PYTHONEXECUTABLE")


def pip_env() -> dict[str, str]:
    """This process's environment, minus what would confuse the child.

    Public because every subprocess the Package Center starts wants it --
    the install itself, the import probe that checks the install worked, and
    the restart helper.
    """
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("PYTHONPATH", None)
    for name in _STDLIB_POINTER_VARS:
        env.pop(name, None)
    return env


def creation_flags() -> int:
    """Windows creation flags, or 0 everywhere else. Reads ``sys.platform`` at
    CALL time so the platform is a fact a test can state.

    Public for the same reason as :func:`pip_env`: a probe that flashed a
    console window over the editor would be as wrong as an install that did.
    """
    if sys.platform == "win32":
        return CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
    return 0


def _stop_process(proc) -> None:
    """End *proc* and everything it started, as forcefully as the OS requires."""
    if sys.platform == "win32":
        # taskkill /T is the only way to reach uv's children from here:
        # ``Popen.terminate`` on Windows is TerminateProcess, which does not
        # touch the process group the flags above put them in.
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, check=False,
                           timeout=KILL_TIMEOUT_S,
                           creationflags=creation_flags())
        except subprocess.TimeoutExpired:
            # taskkill itself hung. Nothing left to try, and blocking the
            # caller's thread forever is worse than a process we could not
            # reach: the reader below is a daemon and the pipe is closed.
            pass
        try:
            # Reap it, so the Popen object does not warn about a running
            # child when it is collected. taskkill /F has already ended it;
            # the timeout is for the case where it somehow has not.
            proc.wait(timeout=TERMINATE_GRACE_S)
        except subprocess.TimeoutExpired:
            pass
        return
    proc.terminate()
    try:
        proc.wait(timeout=TERMINATE_GRACE_S)
    except subprocess.TimeoutExpired:
        proc.kill()


#: The public name for it: ending a child process the way the OS requires is
#: shared by every subprocess the app starts, not only by pip.
stop_process = _stop_process


def _close_output(proc) -> None:
    """Shut the pipe the reader thread is holding.

    The reader is a daemon and ``join`` has a timeout, so it can outlive this
    function -- and a thread still blocked on ``readline`` would go on
    emitting ``log`` events into a job that has already reported itself
    cancelled. Closing the pipe ends its read instead.
    """
    stdout = getattr(proc, "stdout", None)
    if stdout is None:
        return
    try:
        stdout.close()
    except (OSError, ValueError):
        pass


def run_pip(
    specs: Sequence[str],
    *,
    constraints_path: Path | None,
    emit: Callable[[dict], None],
    cancel_check: Callable[[], bool],
    cwd: Path,
    tail: list[str] | None = None,
) -> int:
    """Install *specs* with uv, streaming output as ``log`` events.

    Returns uv's exit code -- 127 when uv itself is missing. Raises
    :class:`PackCancelled` when *cancel_check* goes true, after stopping the
    process; it does NOT decide what a non-zero code means, because that
    depends on the pack (see ``flows.install_pack_live``).

    *tail* -- when given -- collects the last :data:`TAIL_LINES` lines of
    output, which is what a caller builds a failure message from. It is an
    argument rather than a return value because the exit code is the thing
    every caller needs and the output is the thing only a failing one does.
    """
    argv = pip_install_argv(specs, constraints_path=constraints_path)
    emit({"type": "log", "line": " ".join(argv)})

    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=pip_env(),
            creationflags=creation_flags(),
        )
    except FileNotFoundError:
        emit({"type": "log",
              "line": "uv was not found on PATH; install uv and try again"})
        return 127

    def _pump() -> None:
        # Every failure here is the pipe going away under us, which is what
        # cancelling does on purpose. The exit code is the real answer.
        try:
            for raw in proc.stdout:
                line = raw.rstrip("\r\n")
                if not line:
                    continue
                if tail is not None:
                    tail.append(line)
                    del tail[:-TAIL_LINES]
                emit({"type": "log", "line": line})
        except (OSError, ValueError):
            return

    reader = threading.Thread(target=_pump, name="packs-pip-reader", daemon=True)
    reader.start()

    while True:
        if cancel_check():
            _stop_process(proc)
            reader.join(timeout=TERMINATE_GRACE_S)
            _close_output(proc)
            raise PackCancelled("install cancelled")
        returncode = proc.poll()
        if returncode is not None:
            break
        time.sleep(POLL_INTERVAL_S)

    # The child is gone but its output may not be drained yet; the events
    # have to be out before the caller decides what the exit code meant.
    reader.join(timeout=TERMINATE_GRACE_S)
    _close_output(proc)
    return returncode
