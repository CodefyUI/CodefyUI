"""One CodefyUI server per data store.

uvicorn runs an application's startup BEFORE it binds the port, and this
server's startup writes: it replaces ``session.token`` and marks every
``running`` and ``queued`` row of the run database ``interrupted``
(``RunService.recover_interrupted``). So a second start against stores a live
server is using -- ``cdui start`` beside a ``cdui start -f`` or a
``cdui dev``, a hand-run uvicorn, on the same port or another one -- used to
lock every CLI out of the first server (the token file now held the second
one's token) and retire its runs, and only then fail to bind, or on another
port carry on as a second server over the same database. A run retired that
way can never be marked finished: ``mark_finished`` will not overwrite a
terminal row.

So the lifespan takes :func:`hold` before it writes anything, and a start that
cannot have it stops there. Two locks, one per store the startup writes to:

* ``<user data dir>/server.lock``, beside ``session.token``
  (``CODEFYUI_USER_DATA_DIR``; ``cdui`` points it at
  ``<install>/.codefyui_dev``).
* ``<database>-server.lock``, beside the run database (``CODEFYUI_DB_PATH``).
  The default database is ``backend/data/codefyui.db``, in the install and not
  in the user data dir, so two servers with separate data dirs still share it
  -- and that pair is exactly the one whose runs got retired. The name fits
  ``.gitignore``'s ``backend/data/*.db-*``, so a running server leaves nothing
  for ``git status`` to show.

**The lock is the operating system's**, the same primitive as
:mod:`app.core.plugins.lockfile_lock`'s: ``msvcrt.locking`` on a byte far past
the end of the file on Windows (locks are mandatory there, and the holder
record at offset 0 has to stay readable) and ``fcntl.flock`` elsewhere. It is
written here rather than borrowed because only a lock that another holder has
may read as "held" -- any other failure to lock is raised, not taken for a
second server -- and that module's age-based stale rule is not borrowed
either: a server holds its lock for weeks, not for one edit. Its holder-record
reader and its liveness probe are.

The OS lets go of a lock with the last descriptor that holds it: when the
server exits, however it exits -- or, on POSIX, when the last child it forked
while holding the lock exits too, since a forked child shares the descriptor.
So a crashed or killed server does not block the next start. On Windows the
release comes a few milliseconds after the kill rather than with it
(measured: about 5 ms after ``TerminateProcess``), which is why a lock whose
recorded holder has exited is retried for :data:`DEAD_HOLDER_WAIT_S` before it
is reported. A live holder is refused at once.

**Re-entrant within one process.** The test suite runs the lifespan many times
in one interpreter, sometimes nested (a ``TestClient`` inside another), and
both halves of a nested pair are one server as far as the stores go. A count
per lock file does that; the OS lock itself refuses a second descriptor even
in the same process, on both platforms.

A refusal is logged once, as one line, and raised as
:class:`ServerAlreadyRunning`: uvicorn then logs "Application startup failed"
and exits with status 3, having bound nothing.
"""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import os
import platform
import sys
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import settings
from .auth import token_file_path
from .plugins.lockfile_lock import _holder_pid, _pid_is_alive, _read_holder

logger = logging.getLogger(__name__)

#: The byte the lock is taken on: far past the end of the file, so the holder
#: record at offset 0 stays readable under Windows' mandatory locks. (flock is
#: whole-file, and ignores it.)
_LOCK_BYTE_OFFSET = 1 << 30

if sys.platform == "win32":  # pragma: no cover - CI's Linux takes the other
    import msvcrt

    #: What ``msvcrt.locking`` reports when another handle holds the byte:
    #: EACCES for the non-blocking mode used here (measured), EDEADLOCK for
    #: the retrying ones.
    _HELD_ERRNOS = frozenset({errno.EACCES, errno.EDEADLOCK})

    def _lock_byte(fd: int) -> None:
        os.lseek(fd, _LOCK_BYTE_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _unlock_byte(fd: int) -> None:
        os.lseek(fd, _LOCK_BYTE_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:  # pragma: no cover - Windows takes the other
    import fcntl

    #: What a non-blocking flock reports when another descriptor holds it.
    _HELD_ERRNOS = frozenset({errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES})

    def _lock_byte(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock_byte(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _take_os_lock(fd: int) -> bool:
    """Take the lock without waiting. ``False`` means another holder has it.

    Any other failure -- a descriptor gone bad, a filesystem with no locks --
    is raised: it says nothing about a second server, and a start that took
    it for one would refuse with a message that sends the user looking for a
    server that is not there.
    """
    try:
        _lock_byte(fd)
    except OSError as exc:
        if exc.errno in _HELD_ERRNOS:
            return False
        raise
    return True


def _release_descriptor(fd: int) -> None:
    """Empty the record, drop the lock, close -- in that order, so that a lock
    file with a record in it means a holder that did not get this far."""
    with contextlib.suppress(OSError):
        os.ftruncate(fd, 0)
    with contextlib.suppress(OSError):
        _unlock_byte(fd)
    with contextlib.suppress(OSError):
        os.close(fd)


#: The lock's file name in the user data directory.
DATA_DIR_LOCK_NAME = "server.lock"

#: Appended to the database's file name to name its lock.
DB_LOCK_SUFFIX = "-server.lock"

#: Seconds a start keeps asking for a lock whose recorded holder has exited
#: before it reports it. Far above the few milliseconds Windows takes to drop
#: a killed holder's lock, and short enough that a lock which is not coming
#: free is reported while somebody is still looking at the terminal.
DEAD_HOLDER_WAIT_S = 5.0

#: How often that wait asks again.
POLL_INTERVAL_S = 0.05


class ServerAlreadyRunning(RuntimeError):
    """Another process holds one of this server's locks. Nothing was written.

    ``holder`` is the record that process wrote into the lock file (``pid``,
    ``machine``, ``host``, ``port``, ``started_at``), or ``None`` when there
    is none that can be read.
    """

    def __init__(self, *, what: str, lock_path: Path,
                 holder: dict[str, Any] | None, holder_alive: bool) -> None:
        self.what = what
        self.lock_path = lock_path
        self.holder = holder
        self.holder_pid = _holder_pid(holder)
        super().__init__(self._describe(holder_alive))

    def _describe(self, holder_alive: bool) -> str:
        pid = self.holder_pid
        if pid is None:
            return (f"Refusing to start: another process holds the lock on "
                    f"{self.what} ({self.lock_path}) and has not recorded "
                    f"which process it is. Try again in a moment.")
        if not holder_alive:
            return (f"Refusing to start: {self.what} is still locked "
                    f"({self.lock_path}), although the server recorded there "
                    f"(pid {pid}) has exited; a process it started may still "
                    f"hold it. Try again in a moment.")
        who = f"pid {pid}"
        machine = self.holder.get("machine")
        if isinstance(machine, str) and machine and machine != platform.node():
            who += f" on {machine}"
        address = _address(self.holder)
        if address:
            who += f", {address}"
        return (f"Refusing to start: another CodefyUI server ({who}) is using "
                f"{self.what}. Stop that server first, or give this one its "
                f"own data directory (CODEFYUI_USER_DATA_DIR) and run "
                f"database (CODEFYUI_DB_PATH).")


def server_locks(data_dir: Path | None = None,
                 db_path: Path | None = None) -> list[tuple[Path, str]]:
    """The locks a server takes, as ``(lock file, what it stands for)``.

    Read from the configuration when called, not at import: ``cdui`` exports
    ``CODEFYUI_USER_DATA_DIR`` after this module may already be loaded, and
    tests point both stores at temporary directories.
    """
    data_dir = Path(data_dir) if data_dir is not None else token_file_path().parent
    db_path = Path(db_path) if db_path is not None else Path(settings.DB_PATH)
    return [
        (data_dir / DATA_DIR_LOCK_NAME, f"the data directory {data_dir}"),
        (db_path.with_name(db_path.name + DB_LOCK_SUFFIX),
         f"the run database {db_path}"),
    ]


@contextlib.contextmanager
def hold(locks: Iterable[tuple[Path, str]], *, host: str,
         port: int) -> Iterator[None]:
    """Hold every lock in *locks* for the length of the ``with`` block.

    Taken in order and released in reverse. All or nothing: when one of them
    belongs to another process, the ones already taken are let go again, the
    refusal is logged as ONE line, and :class:`ServerAlreadyRunning` is
    raised. *host* and *port* are recorded in the lock files for that line.

    Blocking, and it can wait up to :data:`DEAD_HOLDER_WAIT_S`: the lifespan
    calls it at startup, before the server serves anything.
    """
    taken: list[str] = []
    try:
        for path, what in locks:
            try:
                taken.append(_acquire(Path(path), what, host=host, port=port))
            except ServerAlreadyRunning as refusal:
                _report(refusal)
                raise
        yield
    finally:
        for key in reversed(taken):
            _release(key)


# -- the in-process count ---------------------------------------------------

@dataclass
class _Held:
    fd: int
    depth: int
    #: Who took it. A child forked while it is held inherits this table --
    #: and on POSIX the locked descriptor with it, which keeps the flock held
    #: for as long as the child lives -- but it is not the server: its own
    #: acquire must go to the OS and be refused, and its release must never
    #: unlock the descriptor it shares with the parent.
    pid: int


_held: dict[str, _Held] = {}
_held_guard = threading.Lock()


def _acquire(path: Path, what: str, *, host: str, port: int) -> str:
    """Take *path*'s lock, or count one more hold of it. Returns its key."""
    path.parent.mkdir(parents=True, exist_ok=True)
    key = os.path.normcase(str(path.resolve()))
    # Held across the wait below: a second thread of this process asking for
    # the same file must count against this hold rather than open a second
    # descriptor, which the OS would refuse as if it were another server.
    with _held_guard:
        held = _held.get(key)
        if held is not None and held.pid == os.getpid():
            held.depth += 1
            return key
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            _wait_for(fd, path, what)
        except BaseException:
            os.close(fd)
            raise
        _write_holder(fd, host=host, port=port)
        _held[key] = _Held(fd=fd, depth=1, pid=os.getpid())
        return key


def _release(key: str) -> None:
    with _held_guard:
        held = _held.get(key)
        if held is None or held.pid != os.getpid():
            return
        held.depth -= 1
        if held.depth > 0:
            return
        del _held[key]
        # Empties the record first, then drops the lock and closes: a lock
        # file with a record in it therefore means a holder that crashed.
        _release_descriptor(held.fd)


def _wait_for(fd: int, path: Path, what: str) -> None:
    """Take the OS lock on *fd*, or raise :class:`ServerAlreadyRunning`.

    A holder that is alive is refused at once. One that is not -- a record
    naming an exited pid, or no record at all -- is waited out for at most
    :data:`DEAD_HOLDER_WAIT_S`: the lock of a process that has just died is
    still refused for a moment on Windows, and a holder that has just taken
    the lock has not written its record yet.

    A failure to lock that is not another holder is raised as an ``OSError``
    naming the lock file, and the server does not start.
    """
    deadline = time.monotonic() + DEAD_HOLDER_WAIT_S
    while True:
        try:
            if _take_os_lock(fd):
                return
        except OSError as exc:
            raise OSError(
                exc.errno,
                f"cannot take the instance lock {path}: "
                f"{exc.strerror or exc}") from exc
        holder = _read_holder(path)
        alive = _holder_is_alive(holder)
        if alive or time.monotonic() >= deadline:
            raise ServerAlreadyRunning(what=what, lock_path=path,
                                       holder=holder, holder_alive=alive)
        time.sleep(POLL_INTERVAL_S)


def _holder_is_alive(holder: dict[str, Any] | None) -> bool:
    pid = _holder_pid(holder)
    if pid is None:
        return False
    machine = holder.get("machine")
    if isinstance(machine, str) and machine and machine != platform.node():
        # A data directory on shared storage, locked from another machine:
        # its pid means nothing here, so it is believed rather than checked.
        return True
    return _pid_is_alive(pid)


def _write_holder(fd: int, *, host: str, port: int) -> None:
    """Record who holds the lock, at offset 0 where a refused start reads it."""
    raw = json.dumps({
        "pid": os.getpid(),
        "machine": platform.node(),
        "host": host,
        "port": port,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }).encode("utf-8")
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, raw)
        os.ftruncate(fd, len(raw))
    except OSError:  # pragma: no cover - the lock is held either way
        # The record only feeds a refusal's message; the lock is what
        # excludes. A full disk must not turn a lock we hold into a failure.
        pass


def _address(holder: dict[str, Any] | None) -> str:
    """``host:port`` from a holder record, or ``""`` when it does not say."""
    if not isinstance(holder, dict):
        return ""
    host, port = holder.get("host"), holder.get("port")
    if not isinstance(host, str) or not host or not isinstance(port, int):
        return ""
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


# -- how a refusal reads -----------------------------------------------------

def _report(refusal: ServerAlreadyRunning) -> None:
    logger.error("%s", refusal)
    logging.getLogger("uvicorn.error").addFilter(_DROP_REFUSAL_TRACEBACK)


class _DropRefusalTraceback(logging.Filter):
    """Keep uvicorn from printing a refusal a second time, as a traceback.

    Starlette hands uvicorn the text of a failed startup's whole traceback,
    and uvicorn logs it on ``uvicorn.error``. For a refusal that is some
    twenty lines of framework frames under a sentence :func:`hold` has
    already logged -- and ``cdui start`` shows the last twenty lines of the
    log when a server dies at once, which that traceback alone would fill.
    Only a traceback that ENDS in :class:`ServerAlreadyRunning` is dropped;
    every other startup failure keeps its traceback.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            text = record.getMessage()
        except Exception:  # noqa: BLE001 - a filter must never raise
            return True
        if not text.startswith("Traceback"):
            return True
        last = text.rstrip().rsplit("\n", 1)[-1]
        return not last.startswith(f"{_REFUSAL_TYPE}:")


_REFUSAL_TYPE = f"{ServerAlreadyRunning.__module__}.{ServerAlreadyRunning.__qualname__}"
_DROP_REFUSAL_TRACEBACK = _DropRefusalTraceback()
