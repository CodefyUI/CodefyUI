"""One server per data store: the instance lock the lifespan takes first.

uvicorn runs the application's startup BEFORE it binds the port, and the
startup writes: it replaces ``session.token`` and marks every ``running`` and
``queued`` row of the run database ``interrupted``. A second start against a
store a live server is using -- ``cdui start`` twice, ``cdui dev`` beside
``cdui start -f``, a hand-run uvicorn -- therefore locked every CLI out of the
first server and retired its runs, and only then failed to bind (or, on
another port, carried on as a second server over the same database).

What is pinned here:

* the lock refuses a second PROCESS -- real subprocesses, because a thread
  shares the one thing (the process) that decides the answer;
* it is released on shutdown, and re-acquired after a holder that died
  without releasing it;
* the same process may take it again (the suite nests ``TestClient``s);
* both stores are covered, each on its own: the user data dir (the token)
  and the run database, which by default lives in the install and not in the
  data dir;
* the lifespan takes it before anything is written, and a refused start ends
  with a non-zero exit, the token and the run rows exactly as they were.

Nothing here touches the developer's real data: every lock, token and
database lives under ``tmp_path``.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import fnmatch
import json
import logging
import os
import platform
import signal
import socket
import sqlite3
import subprocess
import sys
import textwrap
import time
import traceback
from pathlib import Path

import pytest

import app.main as main_mod
from app.config import settings
from app.core import instance_lock
from app.core.auth import init_allowed_hosts
from app.core.db import Database
from app.core.run_store import RunProvenance, RunStore

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent

#: A second process that takes the locks it is handed on argv. Prints
#: ``held <pid>`` and holds until its stdin says anything (or closes), then
#: ``released``; a refusal prints ``refused <json>`` and exits 3.
_HOLDER = textwrap.dedent("""
    import json, os, sys
    from pathlib import Path
    from app.core import instance_lock

    locks = [(Path(path), what) for path, what in json.loads(sys.argv[1])]
    try:
        with instance_lock.hold(locks, host="127.0.0.1", port=int(sys.argv[2])):
            print("held", os.getpid(), flush=True)
            sys.stdin.readline()
    except instance_lock.ServerAlreadyRunning as refusal:
        print("refused", json.dumps({"pid": refusal.holder_pid,
                                     "message": str(refusal)}), flush=True)
        sys.exit(3)
    print("released", flush=True)
""")


def _argv(locks, port: int) -> list[str]:
    return [sys.executable, "-c", _HOLDER,
            json.dumps([[str(path), what] for path, what in locks]), str(port)]


def _second_process(locks, port: int = 8472) -> tuple[int, str, dict | None]:
    """Try the locks from another process; ``(exit code, first word, refusal)``.

    Its stdin is empty, so a process that gets the locks lets them go again
    at once and exits 0.
    """
    done = subprocess.run(_argv(locks, port), input="", capture_output=True,
                          text=True, cwd=BACKEND_DIR, timeout=60)
    words = done.stdout.split(maxsplit=1)
    first = words[0] if words else ""
    refusal = json.loads(words[1]) if first == "refused" else None
    assert first in ("held", "refused"), (done.stdout, done.stderr)
    return done.returncode, first, refusal


@contextlib.contextmanager
def _holding_process(locks, port: int = 8471):
    """A live second process holding *locks*; yields ``(proc, its pid)``.

    The pid is the one the holder reports, not ``proc.pid``: on Windows the
    venv's ``python.exe`` is a launcher and the interpreter holding the lock
    is its child.
    """
    proc = subprocess.Popen(_argv(locks, port), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd=BACKEND_DIR)
    try:
        line = proc.stdout.readline().split()
        if not line or line[0] != "held":
            proc.kill()
            pytest.fail(f"the holder did not start: {line} "
                        f"{proc.communicate(timeout=30)[1]}")
        yield proc, int(line[1])
    finally:
        # Closing stdin is what tells a live holder to release and exit.
        with contextlib.suppress(OSError):
            proc.stdin.close()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - a hang
            proc.kill()
            proc.wait()
        proc.stdout.close()
        proc.stderr.close()


def _kill_hard(pid: int) -> None:
    """End *pid* the way a crash or ``kill -9`` does: no cleanup runs.

    Windows' ``os.kill`` with anything but a console event is
    ``TerminateProcess``, which is exactly that; POSIX gets SIGKILL.
    """
    os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))


@pytest.fixture
def stores(tmp_path) -> tuple[Path, Path]:
    """A user data dir and a run database of this test's own."""
    return tmp_path / "data", tmp_path / "db" / "codefyui.db"


@pytest.fixture
def locks(stores):
    data_dir, db_path = stores
    return instance_lock.server_locks(data_dir=data_dir, db_path=db_path)


# -- what is locked ---------------------------------------------------------

def test_both_stores_get_a_lock_of_their_own(stores):
    data_dir, db_path = stores
    paths = [path for path, _ in instance_lock.server_locks(
        data_dir=data_dir, db_path=db_path)]
    assert paths == [data_dir / "server.lock",
                     db_path.with_name("codefyui.db-server.lock")]


def test_the_database_lock_is_a_name_git_already_ignores():
    """A server started from a checkout must leave ``git status`` clean.

    The default database is ``backend/data/codefyui.db``, inside the repo,
    and the lock file sits beside it. ``.gitignore`` covers that directory
    with ``backend/data/*.db`` and ``backend/data/*.db-*`` and nothing wider,
    so the lock's name has to be one of those shapes.
    """
    default_db = BACKEND_DIR / "data" / "codefyui.db"
    lock = instance_lock.server_locks(data_dir=Path("unused"),
                                      db_path=default_db)[1][0]
    relative = lock.relative_to(REPO_ROOT).as_posix()
    patterns = [line.strip() for line in
                (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
                if line.strip().startswith("backend/data/")]
    assert any(fnmatch.fnmatchcase(relative, pattern) for pattern in patterns), (
        relative, patterns)


# -- the lock's own contract -------------------------------------------------

def test_a_second_process_is_refused(locks, stores):
    data_dir, _ = stores
    with instance_lock.hold(locks, host="127.0.0.1", port=8471):
        code, word, refusal = _second_process(locks)

    assert (code, word) == (3, "refused")
    assert refusal["pid"] == os.getpid(), "the refusal must name who holds it"
    assert "127.0.0.1:8471" in refusal["message"]
    assert str(data_dir) in refusal["message"]


def test_the_lock_is_released_when_the_holder_shuts_down(locks):
    with instance_lock.hold(locks, host="127.0.0.1", port=8471):
        pass
    assert _second_process(locks)[:2] == (0, "held")


def test_the_lock_is_released_when_the_with_block_raises(locks):
    with pytest.raises(ZeroDivisionError):
        with instance_lock.hold(locks, host="127.0.0.1", port=8471):
            1 / 0
    assert _second_process(locks)[:2] == (0, "held")


def test_a_live_holder_is_refused_at_once(locks, monkeypatch, caplog):
    """No waiting on a server that is running: the wait is for dead ones."""
    monkeypatch.setattr(instance_lock, "DEAD_HOLDER_WAIT_S", 60.0)
    with _holding_process(locks, port=8471) as (_, holder_pid):
        started = time.monotonic()
        with caplog.at_level(logging.ERROR, logger="app.core.instance_lock"):
            with pytest.raises(instance_lock.ServerAlreadyRunning) as refused:
                with instance_lock.hold(locks, host="127.0.0.1", port=8472):
                    pytest.fail("two servers held one data store")
        waited = time.monotonic() - started

    assert waited < 10, f"waited {waited:.1f}s on a holder that is alive"
    assert refused.value.holder_pid == holder_pid
    assert "127.0.0.1:8471" in str(refused.value)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1, "one clear line, not one per retry"
    assert errors[0].getMessage() == str(refused.value)


def test_a_holder_that_died_does_not_block_the_next_start(locks):
    """Killed with no cleanup at all -- its record is still in the file."""
    with _holding_process(locks) as (proc, holder_pid):
        _kill_hard(holder_pid)
        proc.wait(timeout=20)

    started = time.monotonic()
    with instance_lock.hold(locks, host="127.0.0.1", port=8471):
        record = json.loads(locks[0][0].read_text(encoding="utf-8"))
    assert record["pid"] == os.getpid(), "the dead holder's record was kept"
    assert time.monotonic() - started < instance_lock.DEAD_HOLDER_WAIT_S


def test_a_lingering_lock_of_a_dead_holder_is_waited_out(locks, monkeypatch):
    """Windows releases a killed holder's lock a moment AFTER the kill.

    Driven by call count rather than by a clock: the operating system refuses
    three times while the file names a pid that no longer exists, then
    grants it. The start must wait that out rather than refuse.
    """
    data_lock = locks[0][0]
    data_lock.parent.mkdir(parents=True)
    data_lock.write_text(json.dumps({"pid": _a_pid_that_is_gone(),
                                     "machine": _this_machine()}),
                         encoding="utf-8")
    refusals = {"left": 3}
    real_take = instance_lock._take_os_lock

    def refuse_three_times(fd: int) -> bool:
        if refusals["left"]:
            refusals["left"] -= 1
            return False
        return real_take(fd)

    monkeypatch.setattr(instance_lock, "_take_os_lock", refuse_three_times)
    monkeypatch.setattr(instance_lock, "POLL_INTERVAL_S", 0.01)

    with instance_lock.hold(locks, host="127.0.0.1", port=8471):
        assert refusals["left"] == 0


def test_a_lock_that_stays_held_after_its_holder_died_is_refused_in_the_end(
        locks, monkeypatch):
    """Bounded: a lock that never comes free is reported, not waited on."""
    data_lock = locks[0][0]
    data_lock.parent.mkdir(parents=True)
    gone = _a_pid_that_is_gone()
    data_lock.write_text(json.dumps({"pid": gone, "machine": _this_machine()}),
                         encoding="utf-8")
    monkeypatch.setattr(instance_lock, "_take_os_lock", lambda fd: False)
    monkeypatch.setattr(instance_lock, "DEAD_HOLDER_WAIT_S", 0.2)
    monkeypatch.setattr(instance_lock, "POLL_INTERVAL_S", 0.01)

    with pytest.raises(instance_lock.ServerAlreadyRunning) as refused:
        with instance_lock.hold(locks, host="127.0.0.1", port=8471):
            pass
    assert refused.value.holder_pid == gone
    assert "has exited" in str(refused.value)


def test_a_holder_on_another_machine_is_believed(locks, monkeypatch):
    """A pid recorded by another machine (a data dir on shared storage)
    cannot be checked from here, so it is not taken for dead."""
    data_lock = locks[0][0]
    data_lock.parent.mkdir(parents=True)
    data_lock.write_text(json.dumps({"pid": _a_pid_that_is_gone(),
                                     "machine": "some-other-box",
                                     "host": "0.0.0.0", "port": 8000}),
                         encoding="utf-8")
    monkeypatch.setattr(instance_lock, "_take_os_lock", lambda fd: False)
    monkeypatch.setattr(instance_lock, "DEAD_HOLDER_WAIT_S", 60.0)

    started = time.monotonic()
    with pytest.raises(instance_lock.ServerAlreadyRunning) as refused:
        with instance_lock.hold(locks, host="127.0.0.1", port=8471):
            pass
    assert time.monotonic() - started < 10
    assert "some-other-box" in str(refused.value)


def test_a_failure_to_lock_that_is_not_another_holder_is_raised(
        locks, monkeypatch):
    """Only a lock another holder has reads as "held". Anything else -- here
    a filesystem with no locks -- is not a second server, and a refusal
    would send the user looking for one."""
    def no_locks(fd):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(instance_lock, "_lock_byte", no_locks)
    with pytest.raises(OSError) as failure:
        with instance_lock.hold(locks, host="127.0.0.1", port=8471):
            pytest.fail("held a lock the OS never granted")
    assert failure.value.errno == errno.ENOLCK
    assert str(locks[0][0]) in str(failure.value)


@pytest.mark.parametrize("code", sorted(instance_lock._HELD_ERRNOS))
def test_the_platforms_held_errors_read_as_held(locks, monkeypatch, code):
    data_lock = locks[0][0]
    data_lock.parent.mkdir(parents=True)
    data_lock.write_text(json.dumps({"pid": os.getpid(),
                                     "machine": _this_machine()}),
                         encoding="utf-8")

    def held(fd):
        raise OSError(code, "held by another handle")

    monkeypatch.setattr(instance_lock, "_lock_byte", held)
    with pytest.raises(instance_lock.ServerAlreadyRunning):
        with instance_lock.hold(locks, host="127.0.0.1", port=8471):
            pass


def test_the_same_process_may_take_it_again(locks):
    """The suite nests TestClients: both halves are one server."""
    with instance_lock.hold(locks, host="127.0.0.1", port=8471):
        with instance_lock.hold(locks, host="127.0.0.1", port=8471):
            pass
        # Leaving the inner block must not release what the outer one holds.
        assert _second_process(locks)[:2] == (3, "refused")
    assert _second_process(locks)[:2] == (0, "held")


def test_each_store_is_covered_on_its_own(tmp_path):
    """Sharing either store is enough to be refused.

    The run database is the reason there are two locks: by default it lives
    in the install, so two servers with separate data dirs still share it --
    and the second one's startup used to retire the first one's runs.
    """
    data_a, data_b = tmp_path / "data-a", tmp_path / "data-b"
    db_a, db_b = tmp_path / "db-a" / "codefyui.db", tmp_path / "db-b" / "x.db"
    with _holding_process(instance_lock.server_locks(
            data_dir=data_a, db_path=db_a)):
        for data_dir, db_path, shared in ((data_a, db_b, data_a),
                                          (data_b, db_a, db_a)):
            with pytest.raises(instance_lock.ServerAlreadyRunning) as refused:
                with instance_lock.hold(instance_lock.server_locks(
                        data_dir=data_dir, db_path=db_path),
                        host="127.0.0.1", port=8472):
                    pytest.fail("two servers shared a store")
            assert str(shared) in str(refused.value)

        # All or nothing: the data-b lock taken before db-a refused was let
        # go again. A clean release empties the record.
        assert (data_b / "server.lock").read_bytes() == b""

        with instance_lock.hold(instance_lock.server_locks(
                data_dir=data_b, db_path=db_b), host="127.0.0.1", port=8472):
            pass


def test_a_refused_start_leaves_the_holders_record_alone(locks):
    with _holding_process(locks):
        before = [path.read_bytes() for path, _ in locks]
        with pytest.raises(instance_lock.ServerAlreadyRunning):
            with instance_lock.hold(locks, host="127.0.0.1", port=8472):
                pass
        assert [path.read_bytes() for path, _ in locks] == before


def test_the_record_names_this_server_and_is_emptied_on_release(locks):
    with instance_lock.hold(locks, host="127.0.0.1", port=8471):
        for path, _ in locks:
            record = json.loads(path.read_text(encoding="utf-8"))
            assert (record["pid"], record["host"], record["port"]) == (
                os.getpid(), "127.0.0.1", 8471)
    # A clean release leaves an empty file, so a record means a crash.
    assert [path.read_bytes() for path, _ in locks] == [b"", b""]


# -- how a refusal reads in uvicorn's log ------------------------------------

def _traceback_of(exc: BaseException) -> str:
    try:
        raise exc
    except BaseException:
        return traceback.format_exc()


def test_uvicorn_does_not_repeat_a_refusal_as_a_traceback(locks, monkeypatch):
    """Starlette hands uvicorn the whole traceback of a failed startup, and
    uvicorn logs it on ``uvicorn.error``. A refusal has already been logged
    as one sentence, so its traceback is dropped -- and nothing else is.

    The refusal is staged in-process: a record naming a live pid (this
    one) over a lock the operating system says is taken.
    """
    data_lock = locks[0][0]
    data_lock.parent.mkdir(parents=True)
    data_lock.write_text(json.dumps({"pid": os.getpid(),
                                     "machine": _this_machine()}),
                         encoding="utf-8")
    monkeypatch.setattr(instance_lock, "_take_os_lock", lambda fd: False)
    with pytest.raises(instance_lock.ServerAlreadyRunning) as refused:
        with instance_lock.hold(locks, host="127.0.0.1", port=8472):
            pass

    uvicorn_error = logging.getLogger("uvicorn.error")

    def kept(text: str) -> bool:
        record = uvicorn_error.makeRecord(
            "uvicorn.error", logging.ERROR, __file__, 0, text, (), None)
        return bool(uvicorn_error.filter(record))

    assert not kept(_traceback_of(refused.value))
    assert kept(_traceback_of(ValueError("some other startup failure")))
    assert kept("Application startup failed. Exiting.")


# -- the lifespan ------------------------------------------------------------

@pytest.fixture
def lifespan_stores(stores, monkeypatch):
    """Point the real lifespan at this test's stores, as test_main_lifespan
    does: the token file resolves under the data dir at call time, the
    database is a settings field, logging setup is left alone and no project
    .env is read."""
    data_dir, db_path = stores
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(data_dir))
    monkeypatch.setattr(settings, "DB_PATH", db_path)
    monkeypatch.setattr(settings, "PROJECT_DIR", None)
    monkeypatch.setattr(main_mod, "setup_logging", lambda **kwargs: None)
    yield stores
    # The lifespan re-ran init_allowed_hosts; put back the conftest one.
    init_allowed_hosts(settings.HOST, settings.PORT)


async def test_the_lifespan_is_refused_before_it_writes_anything(
        lifespan_stores, monkeypatch):
    data_dir, db_path = lifespan_stores

    def never(*args, **kwargs):  # pragma: no cover - only runs on a bug
        raise AssertionError("a refused start wrote to a store")

    monkeypatch.setattr(main_mod, "write_token_file", never)
    monkeypatch.setattr(main_mod, "Database", never)

    with _holding_process(instance_lock.server_locks(
            data_dir=data_dir, db_path=db_path)):
        with pytest.raises(instance_lock.ServerAlreadyRunning):
            async with main_mod.lifespan(main_mod.app):
                pytest.fail("the lifespan started under another server")


async def test_the_lifespan_releases_the_lock_when_it_stops(lifespan_stores):
    data_dir, db_path = lifespan_stores
    locks = instance_lock.server_locks(data_dir=data_dir, db_path=db_path)

    async with main_mod.lifespan(main_mod.app):
        assert (await asyncio.to_thread(_second_process, locks))[1] == "refused"
    assert (await asyncio.to_thread(_second_process, locks))[:2] == (0, "held")


async def test_the_lifespan_releases_the_lock_when_startup_fails(
        lifespan_stores, monkeypatch):
    data_dir, db_path = lifespan_stores

    def disk_full():
        raise OSError("No space left on device")

    monkeypatch.setattr(main_mod, "write_token_file", disk_full)
    with pytest.raises(OSError):
        async with main_mod.lifespan(main_mod.app):
            pass  # pragma: no cover - startup fails before this

    locks = instance_lock.server_locks(data_dir=data_dir, db_path=db_path)
    assert (await asyncio.to_thread(_second_process, locks))[:2] == (0, "held")


async def test_nested_lifespans_in_one_process_both_start(lifespan_stores):
    """``TestClient`` inside ``TestClient`` (test_llm_routes does it)."""
    async with main_mod.lifespan(main_mod.app):
        async with main_mod.lifespan(main_mod.app):
            pass


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _seed_a_running_run(db_path: Path) -> str:
    async def seed() -> str:
        db = Database(db_path)
        db.connect()
        try:
            record = await RunStore(db).create_run(
                graph_snapshot={"nodes": [], "edges": []}, status="running",
                provenance=RunProvenance())
        finally:
            db.close()
        return record.id

    return asyncio.run(seed())


def _run_status(db_path: Path, run_id: str) -> str:
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        return conn.execute("SELECT status FROM exec_runs WHERE id = ?",
                            (run_id,)).fetchone()[0]


def _snapshot(directory: Path) -> dict[str, tuple[int, int, bytes]]:
    return {path.name: (path.stat().st_size, path.stat().st_mtime_ns,
                        path.read_bytes())
            for path in sorted(directory.iterdir())}


def test_a_second_real_server_touches_nothing_and_exits_non_zero(tmp_path):
    """The whole bug, end to end, with a real uvicorn.

    A first server holds the stores (a process holding the same locks the
    lifespan takes), its token is on disk and it has a run in flight. A second
    server is started on a FREE port, so nothing but the lock stands in its
    way: before the fix it rewrote the token, marked the run interrupted and
    went on serving. Now it must exit non-zero with one clear line, and leave
    both stores byte for byte as they were.
    """
    data_dir = tmp_path / "data"
    db_path = tmp_path / "db" / "codefyui.db"
    data_dir.mkdir()
    token = data_dir / "session.token"
    token.write_text("the-first-servers-token", encoding="ascii")
    run_id = _seed_a_running_run(db_path)

    port = _free_port()
    env = {key: value for key, value in os.environ.items()
           if key != "CODEFYUI_PROJECT_DIR"}
    env.update({
        "CODEFYUI_USER_DATA_DIR": str(data_dir),
        "CODEFYUI_DB_PATH": str(db_path),
        "CODEFYUI_GRAPHS_DIR": str(tmp_path / "graphs"),
        "CODEFYUI_MODELS_DIR": str(tmp_path / "models"),
        "CODEFYUI_IMAGES_DIR": str(tmp_path / "images"),
        "CODEFYUI_DATA_FILES_DIR": str(tmp_path / "files"),
        "CODEFYUI_MEDIA_DIR": str(tmp_path / "media"),
        "CODEFYUI_HOST": "127.0.0.1",
        "CODEFYUI_PORT": str(port),
        "PYTHONIOENCODING": "utf-8",
    })

    # A file, not a pipe: nothing reads a pipe while the server runs, and on
    # Windows an undrained pipe blocks the writer after about 4 KB -- the
    # size of the traceback this test exists to keep out of the log.
    log_path = tmp_path / "second-server.log"
    with _holding_process(instance_lock.server_locks(
            data_dir=data_dir, db_path=db_path), port=8471):
        before = (_snapshot(data_dir), _snapshot(db_path.parent))
        with open(log_path, "wb") as log:
            server = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app",
                 "--host", "127.0.0.1", "--port", str(port)],
                cwd=BACKEND_DIR, env=env, stdout=log,
                stderr=subprocess.STDOUT)
            try:
                # Fail fast rather than time out: before the fix the token
                # is rewritten within seconds and the server keeps serving.
                deadline = time.monotonic() + 90
                while server.poll() is None and time.monotonic() < deadline:
                    if token.read_bytes() != b"the-first-servers-token":
                        break
                    time.sleep(0.1)
            finally:
                if server.poll() is None:
                    server.kill()
                server.wait(timeout=30)
        output = log_path.read_text(encoding="utf-8", errors="replace")

        assert token.read_bytes() == b"the-first-servers-token", (
            "the second server rewrote the live server's token\n" + output)
        assert _run_status(db_path, run_id) == "running", (
            "the second server retired the live server's run\n" + output)
        assert server.returncode not in (0, None), output
        assert (_snapshot(data_dir), _snapshot(db_path.parent)) == before, (
            "a refused start changed a file in a store\n" + output)

    assert "Refusing to start" in output and str(data_dir) in output, output
    assert "Traceback" not in output, (
        "the refusal should read as one line, not a traceback\n" + output)


# -- helpers -------------------------------------------------------------------

def _this_machine() -> str:
    return platform.node()


def _a_pid_that_is_gone() -> int:
    """A pid no process on this machine answers to (see the lockfile tests)."""
    for candidate in range(999_000, 1_000_000):
        if not instance_lock._pid_is_alive(candidate):
            return candidate
    raise AssertionError("no free pid to test with")  # pragma: no cover
