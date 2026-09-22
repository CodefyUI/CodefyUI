"""One writer at a time for ``installed.json`` (#412).

``<USER_DATA>/plugins/installed.json`` is the only record of which plugins
this install has, which capabilities were granted for each, and which
built-in packs were thrown away on purpose. Seven places used to edit it the
same way -- ``load_lockfile()``, change the dict, ``save_lockfile()`` -- with
nothing between the read and the write to stop a second writer. Two writers
that overlapped never corrupted the file (``save_lockfile`` renames a
complete document into place), but the loser's whole EDIT disappeared: an
install, an uninstall, a granted capability, a ``removed`` tombstone,
overwritten by a document read before it happened, with nothing anywhere
saying so. The Plugin Center made that reachable from the browser while
``cdui plugin install`` was running in a terminal beside it.

So every writer now goes through :func:`locked_lockfile`, which holds a
cross-process lock across the read AND the write:

    with locked_lockfile() as lockfile:
        lockfile["plugins"][plugin_id] = record
        lockfile.save()

Three things about the shape are deliberate.

**The lock is the operating system's**, ``msvcrt.locking`` here and
``fcntl.flock`` there, because the two writers are two INTERPRETERS -- the
CLI in a terminal and the server in its own process -- and no object in
either of them can be seen by the other. It is also why nothing third-party
is used: this repository's CI resolves ``pyproject.toml`` from scratch and
ignores ``uv.lock``, so a lock library would be a new dependency in every
user's install, and the standard library already has the primitive on both
platforms.

**Saving is explicit.** ``save()`` rather than "the context manager writes
what you leave behind", because several of these writers decide INSIDE the
lock that there is nothing to write -- a disable of something already
disabled, an uninstall whose files would not delete -- and rewriting the file
for a no-op is a change a backup tool, a file watcher and a project diff all
see.

**Readers take no lock at all.** ``load_lockfile`` is untouched, so
``GET /api/plugins``, ``/catalog`` and ``cdui plugin list`` never wait for a
writer. A listing that blocked behind an install would be a worse bug than
the one this fixes.

Nothing here is re-entrant: a second ``locked_lockfile`` inside the first
waits and then refuses, in the same process as in any other. That is the
correct answer -- the inner one would be editing a document the outer one is
about to overwrite -- and it is what makes the exclusion testable without
threads.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import platform
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from app.core import plugin_loader

logger = logging.getLogger(__name__)

#: Seconds a writer waits for the lock before refusing. The default is the
#: CLI's: a terminal can afford to wait out anything a normal edit does, and
#: the person is looking at it. Callers on the event loop pass far less, and
#: the install flow's lock step passes far more -- see the constants beside
#: those call sites, which is where the reason for each figure lives.
DEFAULT_TIMEOUT = 5.0

#: How often the wait re-tries. Short enough that a normal hand-off is not
#: noticeable, long enough that waiting is not a spin.
POLL_INTERVAL_S = 0.02

#: A claim older than this is stale whoever recorded it. Sized against what
#: the writers actually do rather than against a clock: the longest of them
#: is an uninstall's ``shutil.rmtree``, and a lockfile edit that has been
#: open for five minutes is a process that is not coming back.
STALE_AFTER_S = 300.0

#: The byte the lock is taken on. Far past the end of the file ON PURPOSE.
#: Windows byte-range locks are MANDATORY -- a locked region cannot even be
#: read by anybody else -- so locking byte 0 would make the holder record
#: below unreadable to the very waiters it exists to inform. Locking beyond
#: the end of the file is legal on both platforms, costs no disk, and leaves
#: the whole of the actual file readable. (``fcntl.flock`` is whole-file and
#: advisory, so the offset is simply ignored there.)
_LOCK_BYTE_OFFSET = 1 << 30

#: How many times a single acquire will re-open the lock file after finding
#: that somebody replaced it. A bound rather than a loop, because "the file
#: keeps changing under me" is a state to report, not one to spin in.
_MAX_REOPENS = 10


if sys.platform == "win32":  # pragma: no cover - the other branch is CI's
    import msvcrt

    #: Which standard-library primitive this platform locks with. Read by a
    #: test, so that swapping in a third-party library is a decision somebody
    #: has to make on purpose rather than one that slips through.
    _BACKEND = "msvcrt"

    def _take_os_lock(fd: int) -> bool:
        """Try to take the lock without waiting. ``False`` means held."""
        try:
            os.lseek(fd, _LOCK_BYTE_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _drop_os_lock(fd: int) -> None:
        try:
            os.lseek(fd, _LOCK_BYTE_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass

else:  # pragma: no cover - the other branch is this developer's machine
    import fcntl

    _BACKEND = "fcntl"

    def _take_os_lock(fd: int) -> bool:
        """Try to take the lock without waiting. ``False`` means held."""
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def _drop_os_lock(fd: int) -> None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass


class LockfileBusy(RuntimeError):
    """Another writer holds ``installed.json``. Nothing was written.

    Carries the holder's pid when the lock file names one, because that is
    the only thing a message can put in front of a person that helps: "wait"
    is advice, "pid 4812 is editing it" is something they can go and look at.
    ``waited`` is how long this call actually spent, for a log line that has
    to distinguish "refused immediately" from "waited the whole timeout".
    """

    def __init__(self, *, holder_pid: int | None = None,
                 waited: float = 0.0) -> None:
        held_by = f" (held by pid {holder_pid})" if holder_pid else ""
        super().__init__(
            f"{plugin_loader.lockfile_path()} is being edited by another "
            f"writer{held_by}"
        )
        self.holder_pid = holder_pid
        self.waited = waited


class LockedLockfile(dict):
    """The lockfile document, read under the lock, with a ``save()``.

    A ``dict`` subclass rather than a wrapper so that every call site reads
    exactly as it did before -- ``lockfile["plugins"][plugin_id]`` -- and the
    only change at each of them is where the write happens.

    ``save()`` writes immediately, while the lock is still held, and refuses
    once it is not: an edit saved after the ``with`` block would be the
    original bug wearing the fix's clothes.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__(data)
        self._held = True

    def save(self) -> None:
        """Write this document back, atomically, under the lock."""
        if not self._held:
            raise RuntimeError(
                "the plugin lockfile was released before this save -- move "
                "the save inside the `with locked_lockfile()` block"
            )
        plugin_loader.save_lockfile(self)

    def _release(self) -> None:
        self._held = False


def lock_path() -> Path:
    """``installed.json.lock``, beside the document it guards."""
    p = plugin_loader.lockfile_path()
    return p.with_name(p.name + ".lock")


@contextlib.contextmanager
def locked_lockfile(
    *,
    timeout: float | None = None,
    stale_after: float | None = None,
) -> Iterator[LockedLockfile]:
    """Hold the writer's lock, hand over the document, let it be saved.

    The read happens INSIDE the lock, which is the whole point: a document
    read before the lock was taken is a document that may already have been
    overwritten by the writer that held it.

    :param timeout: seconds to wait for the lock. ``0`` means "take it or
        refuse", which is what a test wants and what a caller on the event
        loop is close to. ``None`` reads :data:`DEFAULT_TIMEOUT` HERE rather
        than freezing it as a default argument, so the module constant is a
        knob that can actually be turned.
    :param stale_after: seconds after which a recorded claim is disbelieved;
        ``None`` reads :data:`STALE_AFTER_S` the same way.
    :raises LockfileBusy: the lock could not be taken in *timeout* seconds.
        Nothing was read and nothing was written.
    """
    fd = _acquire(
        DEFAULT_TIMEOUT if timeout is None else timeout,
        STALE_AFTER_S if stale_after is None else stale_after,
    )
    held = LockedLockfile(plugin_loader.load_lockfile())
    try:
        yield held
    finally:
        held._release()
        _release(fd)


# -- acquiring -------------------------------------------------------------

def _acquire(timeout: float, stale_after: float) -> int:
    """A descriptor holding the lock, or :class:`LockfileBusy`.

    Breaking a stale lock buys ONE more round of *timeout*, and only one
    (``broken_already``): the worst case is twice the wait the caller asked
    for, and a file that keeps producing stale claims is reported rather than
    broken over and over.
    """
    path = lock_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - a user data dir we cannot make
        raise LockfileBusy() from exc

    started = time.monotonic()
    deadline = started + max(timeout, 0.0)
    fd = _open_lock_file(path)
    broken_already = False
    reopens = 0
    try:
        while True:
            if _take_os_lock(fd):
                if _holds_the_current_file(fd, path):
                    _write_holder(fd)
                    return fd
                # Somebody broke what they took for a stale lock while we
                # were opening or locking this one: what we are holding is
                # now an orphaned file that excludes nobody. Drop it and
                # take the one that is actually at the path. Without this,
                # two waiters that break at the same moment end up holding a
                # lock each, on two different files.
                _drop_os_lock(fd)
                _close(fd)
                fd = -1
                reopens += 1
                if reopens > _MAX_REOPENS:  # pragma: no cover - pathological
                    raise LockfileBusy(waited=time.monotonic() - started)
                fd = _open_lock_file(path)
                continue
            if time.monotonic() < deadline:
                _wait_a_moment(deadline)
                continue

            holder = _read_holder(path)
            reason = (
                None if broken_already
                else _stale_reason(holder, stale_after)
            )
            if reason is None:
                raise LockfileBusy(holder_pid=_holder_pid(holder),
                                   waited=time.monotonic() - started)
            # A lock the operating system is still refusing, held by
            # something that is demonstrably gone: a holder Windows has not
            # got round to unlocking yet (see :func:`_replace_lock_file`), a
            # descriptor inherited by a process that outlived the one that
            # took it, or a filesystem whose locks do not behave. Breaking
            # it is the only way out, and it is a decision that leaves a
            # line behind rather than happening quietly.
            logger.warning(
                "plugin lockfile: breaking a stale lock on %s -- %s. If a "
                "plugin command is still running, stop it and check %s.",
                path, reason, plugin_loader.lockfile_path(),
            )
            broken_already = True
            # Closed and forgotten BEFORE the replace, because the replace
            # opens descriptors of its own: leaving the number around to be
            # closed by the handler below could close whatever took it.
            _close(fd)
            fd = -1
            fd = _replace_lock_file(path)
            deadline = time.monotonic() + max(timeout, 0.0)
    except BaseException:
        _close(fd)
        raise


def _wait_a_moment(deadline: float) -> None:
    """Sleep until the next retry, never past the deadline."""
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(min(POLL_INTERVAL_S, remaining))


def _open_lock_file(path: Path) -> int:
    """A read/write descriptor on the lock file, creating it if need be.

    A failure to open is answered as "busy" rather than raised onward: the
    usual cause on Windows is another program holding the file, and the
    caller's move is the same as for any other busy lockfile. What it must
    never do is continue unlocked.
    """
    try:
        return os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as exc:
        raise LockfileBusy() from exc


def _replace_lock_file(path: Path) -> int:
    """Put a fresh lock file at *path* and return a descriptor on it.

    The old one is left to whatever still holds it: its descriptor now names
    a file nothing will ever open again, while every new arrival opens this
    one. Nothing of ours is open across the rename, which matters on Windows
    -- a file somebody has open cannot be replaced there, so this can only
    succeed when nobody is holding the file, which is exactly the case it is
    for.

    On Windows this path is LOAD-BEARING rather than defensive. Byte-range
    locks are released when a process dies, but the documentation is explicit
    that how long that takes "depends upon available system resources", and
    measured here it is not immediate: a holder killed mid-edit leaves the
    lock in place for long enough that the next writer meets it. Without the
    break, a crashed ``cdui plugin install`` really would wedge the lockfile.
    """
    replacement = path.with_name(f"{path.name}.new-{os.getpid()}")
    try:
        with open(replacement, "wb"):
            pass
        os.replace(replacement, path)
    except OSError as exc:
        with contextlib.suppress(OSError):
            os.unlink(replacement)
        logger.warning("plugin lockfile: could not replace %s (%s)", path, exc)
        raise LockfileBusy() from exc
    return _open_lock_file(path)


# -- the holder record -----------------------------------------------------

def _holds_the_current_file(fd: int, path: Path) -> bool:
    """Whether *fd* still names the file that is at *path* right now.

    A stale lock is broken by REPLACING the lock file, so without this two
    waiters that break at the same moment would hold one lock each, on two
    different files, excluding nobody. Both platforms answer ``st_ino`` and
    ``st_dev`` for a real file -- on Windows they come from the file index,
    which is what makes this portable.
    """
    try:
        held = os.fstat(fd)
        current = os.stat(path)
    except OSError:
        # The path is mid-rename, or gone. Either way this descriptor is not
        # the one to hold, and the caller re-opens.
        return False
    return (held.st_ino, held.st_dev) == (current.st_ino, current.st_dev)


def _write_holder(fd: int) -> None:
    """Record who holds the lock, at offset 0 where waiters can read it."""
    record = json.dumps({
        "pid": os.getpid(),
        "acquired_at": time.time(),
        "host": platform.node(),
    })
    raw = record.encode("utf-8")
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, raw)
        os.ftruncate(fd, len(raw))
    except OSError:  # pragma: no cover - the lock is held either way
        # The record is for diagnosis; the LOCK is what excludes. A full
        # disk must not turn a lock we hold into a failure.
        pass


def _read_holder(path: Path) -> dict[str, Any] | None:
    """Whoever wrote the lock file last, or ``None`` if it does not say.

    ``None`` is the answer for an empty file, unreadable bytes and JSON that
    is not an object alike, and all three mean the same thing here: somebody
    may well hold the lock and has not said who. There is a real window for
    it -- a writer takes the lock before it writes its record.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if not raw.strip():
        return None
    try:
        holder = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return holder if isinstance(holder, dict) else None


def _holder_pid(holder: dict[str, Any] | None) -> int | None:
    if not isinstance(holder, dict):
        return None
    pid = holder.get("pid")
    return pid if isinstance(pid, int) and pid > 0 else None


def _stale_reason(holder: dict[str, Any] | None,
                  stale_after: float) -> str | None:
    """Why this claim should be disbelieved, or ``None`` to believe it.

    Age is asked first, because a pid is not a unique name: both platforms
    reuse them, so "the recorded pid is alive" can be true of a completely
    unrelated program that inherited the number. A claim older than the
    window is stale whoever answers to it now.

    A record that says nothing (:func:`_read_holder` returned ``None``) is
    never stale. The safe reading of "somebody is there and has not said who"
    is to wait for them.
    """
    if not isinstance(holder, dict):
        return None
    pid = _holder_pid(holder)
    if pid is None:
        return None
    acquired = holder.get("acquired_at")
    if isinstance(acquired, (int, float)):
        age = time.time() - float(acquired)
        if age > stale_after:
            return f"the claim by pid {pid} is {age:.0f}s old"
    if not _pid_is_alive(pid):
        return f"pid {pid} is gone"
    return None


def _pid_is_alive(pid: int) -> bool:
    """Whether a process with that id exists. Errs towards ``True``.

    Every unknown answers yes, because the only thing this decides is
    whether to BREAK somebody's lock, and breaking one that is really held is
    the failure this whole module exists to prevent.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":  # pragma: no cover - POSIX takes the other
        return _windows_pid_is_alive(pid)
    try:  # pragma: no cover - Windows is this developer's machine
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        # PermissionError: it exists and belongs to somebody else.
        return True
    return True


def _windows_pid_is_alive(pid: int) -> bool:  # pragma: no cover - platform
    """``OpenProcess`` + ``GetExitCodeProcess``, through ``ctypes``.

    NOT ``os.kill(pid, 0)``. On Windows CPython implements ``os.kill`` as
    ``TerminateProcess`` for any signal that is not a console control event,
    so the POSIX idiom for "does this process exist" would kill it.
    """
    import ctypes
    from ctypes import wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    ERROR_INVALID_PARAMETER = 87
    STILL_ACTIVE = 259

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL,
                                         wintypes.DWORD)
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE,
                                                ctypes.POINTER(wintypes.DWORD))
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            # 87 is "no process with that id". Anything else -- access
            # denied, most likely -- is a process that exists.
            return ctypes.get_last_error() != ERROR_INVALID_PARAMETER
        try:
            code = wintypes.DWORD()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == STILL_ACTIVE
            return True
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        # ctypes is unavailable, or something about this build is not what
        # it looks like. Assume alive; see :func:`_pid_is_alive`.
        return True


# -- releasing -------------------------------------------------------------

def _release(fd: int) -> None:
    """Clear the record, drop the lock, close the descriptor -- in that order.

    The record goes FIRST and is emptied rather than rewritten, so that a
    lock file with anything in it means "a holder did not release cleanly".
    Clearing it after dropping the lock would race the next holder's own
    record straight out of the file.
    """
    try:
        os.ftruncate(fd, 0)
    except OSError:
        pass
    _drop_os_lock(fd)
    _close(fd)


def _close(fd: int) -> None:
    """Close a descriptor, tolerating ``-1`` for "there is none any more"."""
    if fd < 0:
        return
    try:
        os.close(fd)
    except OSError:
        pass
