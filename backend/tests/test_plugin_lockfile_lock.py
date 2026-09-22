"""Two writers, one ``installed.json``, and the edit that used to vanish (#412).

The bug was never corruption. ``save_lockfile`` writes a pid-named temp file
and renames it, so a reader always sees one whole document; what went missing
was a whole EDIT -- an install, an uninstall, a tombstone -- overwritten by a
document that another writer had read BEFORE it happened, with nothing
anywhere saying so. Seven call sites did the same read-modify-write, and one
of them (:func:`~app.core.plugins.inspect._record_moved_repository`) does it
from a plain GET.

So the centrepiece here is an INTERLEAVE, driven rather than raced. Writer A
is the real ``uninstall_plugin``, whose window is the widest in the system --
it reads the lockfile, deletes a directory, and only then writes; writer B is
let in at exactly that point, through a hook on the delete. No threads, no
sleeps, no "run it a thousand times and hope": the second writer enters
between the first's read and the first's write because the test puts it
there. This repository has already had two flaky timing tests (#370) and
neither of them proved anything.

The rest is the lock's own contract: a live holder is answered rather than
waited on, a record left behind by a process that is gone wedges nothing, and
the READERS never take the lock at all -- which is what keeps a listing from
blocking on an install.

Nothing here touches the developer's real plugin directory: ``user_root`` is
autouse and points both plugin roots at ``tmp_path``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import routes_plugins
from app.config import settings
from app.core import plugin_loader
from app.core.auth import TOKEN_HEADER, session_token
from app.core.plugins import flows, lifecycle
from app.core.plugins import inspect as plugin_inspect
from app.core.plugins import lockfile_lock
from app.core.plugins.lockfile_lock import LockfileBusy, locked_lockfile
from app.main import app

import plugins as plugin_cli  # scripts/plugins.py -- conftest puts it on the path

REPO_ROOT = Path(__file__).resolve().parents[2]


# -- fixtures --------------------------------------------------------------

@pytest.fixture(autouse=True)
def user_root(tmp_path, monkeypatch) -> Path:
    """Both plugin roots, in a directory of our own.

    Autouse for the reason ``test_plugins_flows.py`` gives: every test in
    here writes a lockfile, and one that forgot to ask would write the
    developer's.
    """
    target = tmp_path / "user" / "plugins"
    target.mkdir(parents=True)
    monkeypatch.setattr(plugin_loader, "plugins_user_root", lambda: target)
    monkeypatch.setattr(plugin_loader, "plugins_builtin_root",
                        lambda: tmp_path / "builtin")
    return target


def _entry(source_kind: str, **extra) -> dict:
    record = {
        "source_kind": source_kind,
        "source": "x",
        "installed_at": "2026-09-01T00:00:00+00:00",
        "manifest": {},
        "trusted_modules": [],
        "capabilities": [],
        "enabled": True,
    }
    record.update(extra)
    return record


def _write_lockfile(plugins: dict, **top) -> None:
    plugin_loader.save_lockfile({"schema": 1, "plugins": plugins, **top})


def _installed() -> dict:
    return plugin_loader.load_lockfile().get("plugins", {})


# -- the interleave --------------------------------------------------------

def test_an_edit_made_inside_the_uninstall_window_is_never_erased(
        user_root, monkeypatch):
    """A reads, B does a whole edit, A writes. B's edit must survive.

    This is #412 reproduced exactly, with the timing driven instead of
    raced. ``uninstall_plugin`` reads the lockfile, deletes the pack's
    directory and only then saves -- the delete is the window, and the hook
    below lets the second writer in while it is open.

    Two endings are acceptable and they are the same promise from two sides.
    Either B got in, in which case its entry must still be there once A has
    written (nothing may be silently overwritten), or B was told the lockfile
    was busy, in which case it never wrote anything to lose and retries the
    moment A is done. What must never happen -- and is what happened before
    the lock -- is B reporting success and its edit being gone.
    """
    _write_lockfile({"deep": _entry("github_url"), "stats": _entry("builtin")})
    (user_root / "deep").mkdir()

    ending: list[str] = []
    real_remove = lifecycle._remove_downloaded_files

    def remove_then_let_b_in(plugin_id: str):
        # A is now past its read and has just deleted the files; its save is
        # the next thing that will happen. This is the gap.
        result = real_remove(plugin_id)
        try:
            with locked_lockfile(timeout=0) as lockfile:
                lockfile.setdefault("plugins", {})["foundations"] = _entry(
                    "builtin")
                lockfile.save()
            ending.append("written")
        except LockfileBusy:
            ending.append("refused")
        return result

    monkeypatch.setattr(lifecycle, "_remove_downloaded_files",
                        remove_then_let_b_in)

    outcome = lifecycle.uninstall_plugin("deep", builtin_ids=set())
    assert outcome is not None and outcome.removed
    assert ending, "the second writer never ran -- the hook missed the window"

    if ending == ["refused"]:
        # Refused means nothing of B's was written, so B still has its edit
        # and makes it now that the window has closed.
        with locked_lockfile() as lockfile:
            lockfile.setdefault("plugins", {})["foundations"] = _entry("builtin")
            lockfile.save()

    final = _installed()
    assert "deep" not in final, "writer A's uninstall was lost"
    assert "foundations" in final, (
        "writer B's install was erased by writer A's save -- the read-modify-"
        "write is still unguarded"
    )


def test_the_uninstall_holds_the_lock_across_its_whole_window(
        user_root, monkeypatch):
    """And says so: inside the window the lockfile IS busy.

    The test above accepts either ending so that it fails for the right
    reason before the fix. This one pins the ending down: while
    ``uninstall_plugin`` is between its read and its write, a second writer
    is refused rather than admitted.
    """
    _write_lockfile({"deep": _entry("github_url")})
    (user_root / "deep").mkdir()

    seen: list[str] = []
    real_remove = lifecycle._remove_downloaded_files

    def remove_then_probe(plugin_id: str):
        result = real_remove(plugin_id)
        try:
            with locked_lockfile(timeout=0):
                seen.append("admitted")
        except LockfileBusy:
            seen.append("refused")
        return result

    monkeypatch.setattr(lifecycle, "_remove_downloaded_files", remove_then_probe)
    lifecycle.uninstall_plugin("deep", builtin_ids=set())

    assert seen == ["refused"]


# -- the lock's own contract ----------------------------------------------

def test_a_second_holder_is_refused_rather_than_left_waiting():
    """Two holders at once is the one thing the lock exists to prevent.

    Nested in ONE process on purpose. The real pair is the CLI and the
    server, two interpreters -- and the lock has to be an operating system
    lock rather than a ``threading.Lock`` for that to mean anything. Windows
    refuses a second byte-range lock taken through a different handle even
    inside the same process, and POSIX ``flock`` refuses a second open file
    description, so this nesting is the same refusal the two processes get.
    """
    with locked_lockfile():
        with pytest.raises(LockfileBusy):
            with locked_lockfile(timeout=0):
                pytest.fail("two writers held the lockfile at once")


def test_a_refusal_names_who_is_holding_it():
    """``holder_pid`` is what a message can put in front of a person."""
    with locked_lockfile():
        with pytest.raises(LockfileBusy) as refusal:
            with locked_lockfile(timeout=0):
                pass
    assert refusal.value.holder_pid == os.getpid()


def test_the_lock_is_released_when_the_edit_raises(user_root):
    """A writer that blows up must not take the lockfile with it."""
    _write_lockfile({})

    with pytest.raises(ZeroDivisionError):
        with locked_lockfile() as lockfile:
            lockfile["plugins"]["broken"] = _entry("builtin")
            1 / 0

    # Nothing was written -- ``save()`` was never reached ...
    assert _installed() == {}
    # ... and the next writer gets in.
    with locked_lockfile() as lockfile:
        lockfile["plugins"]["fine"] = _entry("builtin")
        lockfile.save()
    assert "fine" in _installed()


def test_saving_after_the_lock_is_released_is_refused(user_root):
    """The document is only writable while the lock is actually held."""
    _write_lockfile({})
    with locked_lockfile() as lockfile:
        pass
    lockfile["plugins"]["sneaky"] = _entry("builtin")
    with pytest.raises(RuntimeError):
        lockfile.save()
    assert _installed() == {}


def test_readers_never_wait_for_a_writer(user_root):
    """``load_lockfile`` takes no lock -- that is what keeps listings fast."""
    _write_lockfile({"foundations": _entry("builtin")})
    with locked_lockfile():
        assert "foundations" in plugin_loader.load_lockfile()["plugins"]


def test_a_no_op_does_not_rewrite_the_file(user_root):
    """Holding the lock is not the same as writing. ``set_enabled`` relies
    on this: pressing disable twice must not touch the file."""
    _write_lockfile({"foundations": _entry("builtin")})
    path = plugin_loader.lockfile_path()
    before = path.read_bytes()

    assert lifecycle.set_enabled("foundations", True) is False

    assert path.read_bytes() == before


# -- stale locks -----------------------------------------------------------

def test_a_record_left_by_a_dead_process_wedges_nothing(user_root, caplog):
    """A killed ``cdui plugin install`` must not lock the file forever.

    The operating system drops a dead process's lock when the process goes,
    so the file it leaves behind is a RECORD and nothing more. This asserts
    the record is never mistaken for a holder: a lock file naming a pid that
    is gone is acquired immediately, not waited on and not refused.
    """
    _write_lockfile({})
    # A fresh timestamp, so the ONLY thing wrong with this claim is that
    # nobody answers to the pid any more.
    lockfile_lock.lock_path().write_text(
        json.dumps({"pid": _a_pid_that_is_gone(), "acquired_at": _now(),
                    "host": "somewhere"}),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        with locked_lockfile(timeout=0) as lockfile:
            lockfile["plugins"]["foundations"] = _entry("builtin")
            lockfile.save()

    assert "foundations" in _installed()


def test_a_lock_that_outlived_its_holder_is_broken_and_logged(
        user_root, monkeypatch, caplog):
    """The belt under the braces: a HELD lock whose holder is gone.

    Reachable when the lock descriptor was inherited by a process that
    outlived the one that took it, or on a filesystem whose locks do not
    behave. The operating system is still refusing the lock, so the only way
    out is to break it -- and breaking it is a decision that has to leave a
    line in the log rather than happening quietly.

    Driven by call count rather than by a clock: the platform lock refuses
    until the lock file has been replaced, which is exactly what a break
    does.
    """
    _write_lockfile({})
    lockfile_lock.lock_path().write_text(
        json.dumps({"pid": _a_pid_that_is_gone(),
                    "acquired_at": _now(), "host": "somewhere"}),
        encoding="utf-8",
    )

    refused_for: dict[str, int] = {"before_break": 1}
    real_take = lockfile_lock._take_os_lock

    def refuse_until_the_file_is_replaced(fd: int) -> bool:
        if refused_for["before_break"] > 0:
            refused_for["before_break"] -= 1
            return False
        return real_take(fd)

    monkeypatch.setattr(lockfile_lock, "_take_os_lock",
                        refuse_until_the_file_is_replaced)

    with caplog.at_level(logging.WARNING):
        with locked_lockfile(timeout=0) as lockfile:
            lockfile["plugins"]["foundations"] = _entry("builtin")
            lockfile.save()

    assert "foundations" in _installed()
    assert any("stale" in record.getMessage() for record in caplog.records), (
        "breaking a stale lock has to say so in the log"
    )


def test_a_live_holder_is_answered_rather_than_broken(user_root, monkeypatch):
    """The other half: a holder that is still running is never broken.

    Same shape as the test above and the opposite ending, because the ONLY
    difference is whether the recorded pid is alive. Our own pid is the one
    pid this test can be sure of.
    """
    _write_lockfile({})
    lockfile_lock.lock_path().write_text(
        json.dumps({"pid": os.getpid(), "acquired_at": _now(),
                    "host": "here"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(lockfile_lock, "_take_os_lock", lambda fd: False)

    with pytest.raises(LockfileBusy) as refusal:
        with locked_lockfile(timeout=0):
            pass
    assert refusal.value.holder_pid == os.getpid()


def test_a_holder_older_than_the_stale_window_is_broken(
        user_root, monkeypatch, caplog):
    """Age is the second stale rule, for a pid that has been recycled.

    A pid is not a unique name -- both platforms reuse them -- so "the
    recorded pid is alive" can be true of a completely different program. An
    hour-old claim on a file whose writers hold it for milliseconds is stale
    whoever answers to that number now.
    """
    _write_lockfile({})
    lockfile_lock.lock_path().write_text(
        json.dumps({"pid": os.getpid(),
                    "acquired_at": _now() - lockfile_lock.STALE_AFTER_S - 60,
                    "host": "here"}),
        encoding="utf-8",
    )

    refused_once = {"n": 1}
    real_take = lockfile_lock._take_os_lock

    def refuse_once(fd: int) -> bool:
        if refused_once["n"] > 0:
            refused_once["n"] -= 1
            return False
        return real_take(fd)

    monkeypatch.setattr(lockfile_lock, "_take_os_lock", refuse_once)

    with caplog.at_level(logging.WARNING):
        with locked_lockfile(timeout=0) as lockfile:
            lockfile["plugins"]["foundations"] = _entry("builtin")
            lockfile.save()

    assert "foundations" in _installed()


def test_an_unreadable_holder_record_is_not_a_reason_to_break(
        user_root, monkeypatch):
    """Junk in the lock file means "somebody is there and has not said who".

    The window is real: a writer takes the operating system lock before it
    has written its record, so a waiter can find an empty or half-written
    file. Refusing is the safe reading of that; breaking would be the
    dangerous one.
    """
    _write_lockfile({})
    lockfile_lock.lock_path().write_text("not json at all", encoding="utf-8")
    monkeypatch.setattr(lockfile_lock, "_take_os_lock", lambda fd: False)

    with pytest.raises(LockfileBusy) as refusal:
        with locked_lockfile(timeout=0):
            pass
    assert refusal.value.holder_pid is None


def test_the_lock_file_is_left_empty_so_a_leftover_record_means_a_crash(
        user_root):
    """A clean release truncates its record; only a crash leaves one."""
    _write_lockfile({})
    with locked_lockfile():
        assert lockfile_lock.lock_path().read_bytes().strip(), (
            "a holder must record who it is while it holds the lock"
        )
    assert lockfile_lock.lock_path().read_bytes() == b""


def test_a_descriptor_on_some_other_file_is_not_the_current_lock(user_root):
    """What a lock file replaced under us looks like from the inside.

    Breaking a stale lock replaces the file, so a descriptor can end up
    naming a file that is no longer at the path -- an orphan that excludes
    nobody. Two different files is exactly that state, and the only portable
    way to build it: Windows refuses to replace a file somebody has open,
    which is the very thing being simulated.
    """
    path = lockfile_lock.lock_path()
    path.write_bytes(b"")
    orphan = path.with_name("orphaned.lock")
    orphan.write_bytes(b"")

    current_fd = os.open(path, os.O_RDWR)
    orphan_fd = os.open(orphan, os.O_RDWR)
    try:
        assert lockfile_lock._holds_the_current_file(current_fd, path)
        assert not lockfile_lock._holds_the_current_file(orphan_fd, path)
    finally:
        os.close(current_fd)
        os.close(orphan_fd)


def test_a_lock_file_replaced_under_us_is_dropped_and_retaken(
        user_root, monkeypatch):
    """And the acquire heals rather than holding the orphan.

    Driven by making the identity check answer ``False`` once, because the
    real cause of it -- another waiter breaking a stale lock at the same
    moment -- cannot be staged on Windows at all.
    """
    _write_lockfile({})
    opens: list[int] = []
    real_open = lockfile_lock._open_lock_file
    real_check = lockfile_lock._holds_the_current_file
    first_answer_is_no = {"n": 1}

    def counting_open(path):
        fd = real_open(path)
        opens.append(fd)
        return fd

    def no_then_yes(fd, path):
        if first_answer_is_no["n"] > 0:
            first_answer_is_no["n"] -= 1
            return False
        return real_check(fd, path)

    monkeypatch.setattr(lockfile_lock, "_open_lock_file", counting_open)
    monkeypatch.setattr(lockfile_lock, "_holds_the_current_file", no_then_yes)

    with locked_lockfile(timeout=0) as lockfile:
        opens_to_acquire = len(opens)
        lockfile["plugins"]["foundations"] = _entry("builtin")
        lockfile.save()
        # The point of healing: what it ends up holding really does exclude.
        with pytest.raises(LockfileBusy):
            with locked_lockfile(timeout=0):
                pass

    assert opens_to_acquire == 2, "the orphaned descriptor was kept"
    assert "foundations" in _installed()


# -- the platform seam -----------------------------------------------------

def test_the_platform_lock_is_the_standard_library_one():
    """``msvcrt`` here, ``fcntl`` there -- and no third-party lock library.

    Pinned because the choice is a dependency decision: this repository's CI
    resolves ``pyproject.toml`` afresh and ignores ``uv.lock``, so a new
    direct dependency is a change to the install of every user.
    """
    assert lockfile_lock._BACKEND in {"msvcrt", "fcntl"}
    assert lockfile_lock._BACKEND == (
        "msvcrt" if sys.platform == "win32" else "fcntl"
    )


def test_a_lock_file_in_a_directory_that_does_not_exist_yet(tmp_path,
                                                            monkeypatch):
    """First run on a fresh machine: nothing under the user data dir yet."""
    fresh = tmp_path / "never" / "existed" / "plugins"
    monkeypatch.setattr(plugin_loader, "plugins_user_root", lambda: fresh)

    with locked_lockfile() as lockfile:
        lockfile["plugins"]["foundations"] = _entry("builtin")
        lockfile.save()

    assert "foundations" in _installed()


# -- every writer, and no others ------------------------------------------

#: The files that edit ``installed.json``. If a new one appears it belongs in
#: this list AND behind the lock; the test below is what makes the second
#: part of that sentence enforceable rather than a note in a review.
WRITER_SOURCES = [
    REPO_ROOT / "backend" / "app" / "core" / "plugins" / "flows.py",
    REPO_ROOT / "backend" / "app" / "core" / "plugins" / "inspect.py",
    REPO_ROOT / "backend" / "app" / "core" / "plugins" / "lifecycle.py",
    REPO_ROOT / "scripts" / "plugins.py",
]


def test_no_writer_saves_the_lockfile_outside_the_lock():
    """``save_lockfile`` is called from one place, and that place locks.

    A structural test because the failure it guards against is an OMISSION:
    the next person to add a lockfile edit will copy the nearest existing one,
    and before #412 every one of those was an unguarded read-modify-write.
    Nothing catches a missing lock at runtime -- the edit is lost silently,
    which is the entire bug -- so it is caught here, in the source.
    """
    offenders = []
    for path in WRITER_SOURCES:
        for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            if "save_lockfile(" in line:
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, (
        "these edit installed.json without the writer's lock; go through "
        "lockfile_lock.locked_lockfile() instead:\n" + "\n".join(offenders)
    )


def test_the_only_production_caller_of_save_lockfile_is_the_lock():
    """And the one place that does call it is the lock module itself."""
    source = (REPO_ROOT / "backend" / "app" / "core" / "plugins"
              / "lockfile_lock.py").read_text(encoding="utf-8")
    assert "save_lockfile(" in source


# -- the writers, under the lock ------------------------------------------

def test_set_enabled_refuses_while_another_writer_holds_the_lockfile(
        user_root):
    _write_lockfile({"foundations": _entry("builtin")})
    with locked_lockfile():
        with pytest.raises(LockfileBusy):
            lifecycle.set_enabled("foundations", False, lock_timeout=0)
    assert plugin_loader.is_enabled(_installed()["foundations"])


def test_uninstall_refuses_before_it_deletes_anything(user_root, monkeypatch):
    """The refusal comes first, so the files are still there to try again."""
    _write_lockfile({"deep": _entry("github_url")})
    (user_root / "deep").mkdir()

    def never(plugin_id: str):  # pragma: no cover - only runs on a bug
        raise AssertionError("the uninstall deleted files without the lock")

    monkeypatch.setattr(lifecycle, "_remove_downloaded_files", never)

    with locked_lockfile():
        with pytest.raises(LockfileBusy):
            lifecycle.uninstall_plugin("deep", builtin_ids=set(),
                                       lock_timeout=0)
    assert (user_root / "deep").exists()
    assert "deep" in _installed()


def test_the_install_step_reports_a_busy_lockfile_as_an_install_failure(
        user_root, monkeypatch):
    """The one writer that must not refuse quietly.

    By the time an install writes its entry the plugin's files are already on
    disk, so "could not record it" is an install FAILURE with the directory
    named -- not a shrug that leaves a pack nothing will ever load.
    """
    _write_lockfile({})
    monkeypatch.setattr(flows, "INSTALL_LOCK_TIMEOUT", 0)
    plan = _a_plan("extras")

    with locked_lockfile():
        with pytest.raises(flows.PluginInstallError) as failure:
            flows._write_lockfile_entry(plan, _entry("github_url"))

    assert "extras" in str(failure.value)
    assert "installed.json" in (failure.value.hint or "")


def test_a_read_that_corrects_a_moved_repository_gives_up_when_busy(
        user_root, monkeypatch):
    """The seventh writer, and the only one reached from a GET.

    ``_record_moved_repository`` runs inside ``inspect_installed``, which the
    Plugin Center fires on an ordinary panel refresh. Before the lock it could
    erase a concurrent CLI install; with it, a busy lockfile means the badge
    stays stale for one more look. It must never raise, and it must never
    wait -- the caller is a read.
    """
    _write_lockfile({"extras": _entry(
        "github_url", url="https://github.com/alice/extras",
        source="alice/extras")})

    found = _an_inspection(owner="bob", repo="extras")
    with locked_lockfile():
        plugin_inspect._record_moved_repository(
            "extras", "alice/extras", found)

    # Gave up silently: the record is untouched, and nothing was lost.
    assert _installed()["extras"]["url"] == "https://github.com/alice/extras"


def test_a_read_that_corrects_a_moved_repository_writes_when_it_can(
        user_root):
    """And the same call with the lockfile free really does correct it."""
    _write_lockfile({"extras": _entry(
        "github_url", url="https://github.com/alice/extras",
        source="alice/extras")})

    plugin_inspect._record_moved_repository(
        "extras", "alice/extras", _an_inspection(owner="bob", repo="extras"))

    assert _installed()["extras"]["url"] == "https://github.com/bob/extras"


# -- the CLI's answer ------------------------------------------------------

@pytest.fixture
def impatient_cli(monkeypatch):
    """Take the CLI's wait down to nothing for the refusal tests.

    The CLI calls ``locked_lockfile()`` with no timeout of its own, so it
    gets :data:`~app.core.plugins.lockfile_lock.DEFAULT_TIMEOUT` -- five real
    seconds, three times over, for tests whose subject is what it SAYS when
    the wait runs out. Patching the constant works because the wait is read
    at the call rather than frozen as a default argument.
    """
    monkeypatch.setattr(lockfile_lock, "DEFAULT_TIMEOUT", 0.0)


def test_unlink_refuses_and_drops_nothing(user_root, impatient_cli, capsys):
    """``cdui plugin unlink`` ends in a sentence, not a traceback."""
    _write_lockfile({"devplug": _entry("local", path=str(user_root))})

    with locked_lockfile():
        rc = plugin_cli.cmd_unlink(argparse.Namespace(plugin_id="devplug"))

    assert rc == 1
    assert "devplug" in _installed()
    printed = capsys.readouterr()
    assert "lockfile" in (printed.out + printed.err).lower()


def test_link_refuses_and_records_nothing(user_root, tmp_path, impatient_cli,
                                          capsys):
    """And so does ``cdui plugin link``, after the manifest but before the
    entry -- so the directory it was pointed at is left exactly alone."""
    source = tmp_path / "my-plugin"
    source.mkdir()
    (source / "cdui.plugin.toml").write_text(
        '[plugin]\nid = "devplug"\nname = "Dev"\nversion = "0.1.0"\n'
        'schema_version = 1\n',
        encoding="utf-8",
    )
    _write_lockfile({})

    with locked_lockfile():
        rc = plugin_cli._link_local(source, force=False)

    assert rc == 1
    assert _installed() == {}
    assert (source / "cdui.plugin.toml").exists()


def test_sync_prune_refuses_rather_than_pruning_half(user_root, impatient_cli,
                                                     capsys):
    """``--prune`` is a read-modify-write like any other."""
    _write_lockfile({"gone-from-the-catalog": _entry("builtin")})

    with locked_lockfile():
        rc = plugin_cli.cmd_sync(argparse.Namespace(
            prune=True, dry_run=False, yes=True))

    assert rc == 1
    assert "gone-from-the-catalog" in _installed()


def test_sync_dry_run_never_asks_for_the_lock(user_root, capsys):
    """A preview changes nothing, so a busy lockfile must not stop it.

    ``--dry-run`` is what somebody runs to find out what ``--prune`` WOULD
    do, and refusing it because an install is in progress would be refusing
    to answer a question.
    """
    _write_lockfile({"gone-from-the-catalog": _entry("builtin")})

    with locked_lockfile():
        rc = plugin_cli.cmd_sync(argparse.Namespace(
            prune=True, dry_run=True, yes=True))

    assert rc == 0
    assert "would prune" in capsys.readouterr().out.lower()
    assert "gone-from-the-catalog" in _installed()


# -- the server's answer ---------------------------------------------------

@pytest.fixture
async def http(monkeypatch):
    """A client carrying the session token, over the real app.

    No ``PluginService`` on ``app.state``, on purpose: ``_refuse_while_busy``
    asks for one rather than requiring it, so a server whose installer never
    started still answers these two routes -- which is the state this test
    wants, because the refusal under test is about the LOCKFILE and not about
    a job.
    """
    monkeypatch.delattr(app.state, "plugin_service", raising=False)
    monkeypatch.setattr(settings, "HOST", "127.0.0.1")
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url=f"http://127.0.0.1:{settings.PORT}",
        headers={TOKEN_HEADER: session_token()},
    ) as client:
        yield client


async def test_disable_answers_409_busy_rather_than_blocking_the_loop(
        user_root, http):
    """The handler is ``async def``, so waiting would stall every request.

    ``busy`` is the code the panel already knows -- and deliberately with no
    ``message``, following the rest of this router: prose about a coded
    refusal belongs to whoever is talking to the user, in their language.
    Deliberately with no ``job_id`` either, which is the difference the
    client can act on: there is no job to follow here, only a moment to wait.
    """
    _write_lockfile({"foundations": _entry("builtin")})

    with locked_lockfile():
        response = await http.post("/api/plugins/foundations/disable")

    assert response.status_code == 409, response.text
    assert response.json()["detail"] == {"code": "busy"}
    # And nothing was flipped, which is what makes retrying safe.
    assert plugin_loader.is_enabled(_installed()["foundations"])


async def test_delete_answers_409_busy_and_deletes_nothing(
        user_root, http, monkeypatch):
    _write_lockfile({"demo": _entry("github_url")})
    (user_root / "demo").mkdir()

    def never(plugin_id: str):  # pragma: no cover - only runs on a bug
        raise AssertionError("the route deleted files without the lock")

    monkeypatch.setattr(lifecycle, "_remove_downloaded_files", never)

    with locked_lockfile():
        response = await http.delete("/api/plugins/demo")

    assert response.status_code == 409, response.text
    assert response.json()["detail"] == {"code": "busy"}
    assert (user_root / "demo").exists()
    assert "demo" in _installed()


async def test_a_free_lockfile_still_lets_the_route_through(user_root, http,
                                                            monkeypatch):
    """The refusal must be about the lock and nothing else."""
    _write_lockfile({"foundations": _entry("builtin")})
    monkeypatch.setattr(routes_plugins, "rediscover_now", lambda: {})

    response = await http.post("/api/plugins/foundations/disable")

    assert response.status_code == 200, response.text
    assert response.json() == {"id": "foundations", "enabled": False}
    assert not plugin_loader.is_enabled(_installed()["foundations"])


# -- helpers ---------------------------------------------------------------

def _now() -> float:
    import time
    return time.time()


def _a_pid_that_is_gone() -> int:
    """A pid no process on this machine answers to.

    Grown from a high number rather than picked at random so the search ends:
    both platforms allocate pids from the low end, and the answer only has to
    be a pid that is free right now.
    """
    for candidate in range(999_000, 1_000_000):
        if not lockfile_lock._pid_is_alive(candidate):
            return candidate
    raise AssertionError("no free pid to test with")  # pragma: no cover


def _a_plan(plugin_id: str):
    """An :class:`~app.core.plugins.flows.InstallPlan`. The lock step reads
    the id off it and nothing else; the rest is filled in so the dataclass
    is the real one rather than a stand-in that could drift from it."""
    return flows.InstallPlan(
        kind="github",
        plugin_id=plugin_id,
        catalog_id=None,
        owner="alice",
        repo=plugin_id,
        ref="",
        sha="0" * 40,
        manifest={},
        granted_capabilities=(),
        trust_author=True,
        force=False,
        mode="install",
        prior=None,
    )


def _an_inspection(*, owner: str, repo: str):
    """An :class:`~app.core.plugins.inspect.Inspection`.

    ``_record_moved_repository`` reads four fields off it -- ``url``,
    ``source``, ``owner`` and ``repo`` -- and the rest are the dataclass's
    own requirements.
    """
    return plugin_inspect.Inspection(
        kind="github",
        mode="update",
        plugin_id=repo,
        catalog_id=None,
        official=False,
        source=f"{owner}/{repo}",
        owner=owner,
        repo=repo,
        url=f"https://github.com/{owner}/{repo}",
        ref="",
        sha="0" * 40,
        name=repo,
        version="1.0.0",
        description="",
        homepage="",
        manifest={},
        capabilities=(),
        allowed_modules=(),
        python_deps={},
        has_frontend=False,
        chapters=(),
        lessons=(),
        consent_required=False,
        installed=None,
        up_to_date=False,
        capabilities_added=(),
        allowed_modules_added=(),
        warnings=(),
    )
