"""The restart-mode core: what the server writes down before it goes away.

A restart-mode install is the one case where finishing the job means ending
the process that started it. Four artefacts carry the decision across that
gap, and each of them is pinned here:

* ``restart_available()`` -- may this server offer it at all? Only a process
  ``cdui start`` launched knows how to come back, and only while the
  launcher it was started from is still on disk.
* the PENDING file -- what to install, into which interpreter, and whose
  exit to wait for. Written atomically, and REFUSED rather than overwritten
  while another server's claim is still live: two helpers installing into
  one site-packages is the corruption this whole feature exists to avoid.
* the detached HELPER -- the process that outlives this one. Its argv, its
  detach flags and its log file are the entire contract with dev.py's
  ``packs-run-pending`` (R3), which cannot import any of this.
* the OUTCOME file -- ``status`` and ``message``, which is what the SPA reads
  after the reload to say whether it worked.

Nothing here touches the network or shuts anything down, and the only
process any test starts is the ``python -c pass`` that ``_pid_alive`` is
then asked about. The helper's ``subprocess.Popen`` is faked, ``_pid_alive``
is monkeypatched wherever a verdict about somebody else's process is needed,
and the event loop is a stub that RECORDS what it was asked to do -- a test
that actually delivered the SIGINT it schedules would kill the pytest run.
"""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import subprocess
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.core.packs import restart, runner
from app.core.packs.catalog import get_pack
from app.core.packs.errors import PackInstallError, PendingExists
from app.core.packs.paths import job_log_dir, last_restart_file, pending_restart_file

#: Every key the pending file carries. Key-set equality on purpose: dev.py's
#: helper reads this file with no access to the dataclass, so an added or
#: renamed key is a broken handshake and should break a test once, here.
PENDING_KEYS = {"schema", "job_id", "pack_id", "kind", "index_url", "packages",
                "specs", "venv_python", "server_pid", "launcher",
                "relaunch_argv", "created_at"}


@pytest.fixture(autouse=True)
def isolated_control_dir(tmp_path, monkeypatch):
    """A throwaway user-data root, and none of the launch environment.

    Every function under test reads ``os.environ`` at call time, so a box
    that happens to have run ``cdui start`` in this shell must not change
    what the suite asserts.
    """
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    for name in ("CODEFYUI_MANAGED", restart.ENABLE_ENV,
                 restart.LAUNCHER_ENV, restart.RELAUNCH_ARGV_ENV):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


@pytest.fixture
def launcher(tmp_path, monkeypatch) -> list[str]:
    """A launcher that exists on disk, exported the way ``cdui start`` will."""
    python = tmp_path / "outer-python.exe"
    python.write_text("", encoding="utf-8")
    dev_py = tmp_path / "dev.py"
    dev_py.write_text("", encoding="utf-8")
    argv = [str(python), str(dev_py)]
    monkeypatch.setenv(restart.LAUNCHER_ENV, json.dumps(argv))
    return argv


@pytest.fixture
def cu128(monkeypatch):
    """Pin what this machine "should" have, so no test shells out to a GPU."""
    monkeypatch.setattr(restart, "gpu_info",
                        lambda: {"recommended_variant": "cu128"})


def _pending_on_disk() -> restart.PendingRestart:
    return restart.PendingRestart.from_json(
        pending_restart_file().read_text(encoding="utf-8"))


def _iso_ago(seconds: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


# -- may this server restart itself? ---------------------------------------

def test_restart_available_needs_start_mode_and_launcher(tmp_path, monkeypatch):
    """The decision is three facts wide, and every one of them can say no."""
    real = tmp_path / "outer-python.exe"
    real.write_text("", encoding="utf-8")
    ok = json.dumps([str(real), str(tmp_path / "dev.py")])

    cases = [
        # managed, launcher, kill switch, expected, why
        (None, None, None, False,
         "an unmanaged process has nobody to bring it back"),
        (None, ok, None, False, "a launcher alone is not a supervisor"),
        ("dev", ok, None, False,
         "cdui dev reloads itself in place; it does not relaunch"),
        ("start", None, None, False, "cdui start exported no launcher"),
        ("start", ok, None, True, "the shipping case"),
        ("start", ok, "1", True, "the kill switch is off"),
        ("start", ok, "0", False, "the kill switch is on"),
        ("start", json.dumps([str(tmp_path / "gone.exe")]), None, False,
         "the launcher is not on disk any more"),
        ("start", json.dumps([str(tmp_path)]), None, False,
         "a directory is not an interpreter"),
        ("start", "[]", None, False, "an empty launcher launches nothing"),
        ("start", "not json", None, False, "unparseable launcher"),
        ("start", json.dumps({"python": str(real)}), None, False,
         "an object is not an argv"),
        ("start", json.dumps([1, 2]), None, False, "numbers are not an argv"),
        ("start", json.dumps([str(real), 7]), None, False,
         "one bad element spoils the argv"),
    ]

    for managed, launcher_json, enable, expected, why in cases:
        for name, value in (("CODEFYUI_MANAGED", managed),
                            (restart.LAUNCHER_ENV, launcher_json),
                            (restart.ENABLE_ENV, enable)):
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        assert restart.restart_available() is expected, why


def test_runs_active_sees_running_and_queued_runs():
    """A restart that kills a forty-minute training run is not a feature.

    ``RunService`` answers "running" and "waiting for a device" with two
    different calls, and both count: a queued run belongs to a user who is
    watching for it to start, and this process is the only one that knows it
    exists (the queue is in memory).
    """
    from app.core.run_service import RunService

    # The names this reads through. Asserted rather than assumed so that
    # renaming either one fails here instead of silently answering "no runs".
    assert callable(RunService.active_run_ids)
    assert callable(RunService.queue_snapshot)

    class _Stub:
        def __init__(self, active=(), queued=None, boom=False):
            self._active, self._queued, self._boom = list(active), queued or {}, boom

        def active_run_ids(self):
            if self._boom:
                raise RuntimeError("the service is mid-shutdown")
            return list(self._active)

        def queue_snapshot(self):
            return dict(self._queued)

    def _app(service):
        return types.SimpleNamespace(state=types.SimpleNamespace(
            run_service=service))

    assert restart.runs_active(_app(_Stub())) is False
    assert restart.runs_active(_app(_Stub(active=["run-1"]))) is True
    assert restart.runs_active(_app(_Stub(queued={"cuda": ["run-2"]}))) is True
    # An empty queue for a device is not a queued run.
    assert restart.runs_active(_app(_Stub(queued={"cuda": []}))) is False
    # No service at all: nothing is running, because nothing can be.
    assert restart.runs_active(types.SimpleNamespace(
        state=types.SimpleNamespace())) is False
    # An unreadable service is treated as busy -- the only safe direction.
    assert restart.runs_active(_app(_Stub(boom=True))) is True


# -- which wheel, and from where -------------------------------------------

def test_resolve_gpu_torch_maps_variants_and_rejects_mps(cu128):
    for variant in restart.VARIANTS:
        if variant == "mps":
            continue
        assert restart.resolve_gpu_torch(variant) == (
            variant, restart.TORCH_INDEX_URLS[variant])

    # None and "auto" are the same request: "decide for me".
    recommended = ("cu128", restart.TORCH_INDEX_URLS["cu128"])
    assert restart.resolve_gpu_torch(None) == recommended
    assert restart.resolve_gpu_torch("auto") == recommended

    # The refusal has to SAY why, not just fail: "no index URL" reads like a
    # bug in the panel, and the user would try again.
    with pytest.raises(ValueError, match="Apple Silicon"):
        restart.resolve_gpu_torch("mps")


def test_resolve_gpu_torch_rejects_an_auto_that_lands_on_mps(monkeypatch):
    """Apple Silicon ships its acceleration in the default wheel.

    There is no index to switch to, so "install the recommended build" has
    to fail here rather than write a pending file with ``index_url: null``
    that the helper would hand to ``--index-url``.
    """
    monkeypatch.setattr(restart, "gpu_info",
                        lambda: {"recommended_variant": "mps"})
    with pytest.raises(ValueError, match="Apple Silicon"):
        restart.resolve_gpu_torch(None)


def test_resolve_gpu_torch_names_the_variants_it_would_accept(cu128):
    for bogus in ("skip", "cu999", "", "CPU", "cuda"):
        with pytest.raises(ValueError) as exc:
            restart.resolve_gpu_torch(bogus)
        message = str(exc.value)
        assert "cu128" in message and "rocm6.2" in message, (
            f"the refusal for {bogus!r} does not name the real variants")


def test_gpu_map_parity_with_dev_py():
    """The RESOLVER, not only the map, has to agree with the installer CLI.

    ``test_api_packs.test_gpu_info_never_raises_and_mirrors_dev_py`` already
    pins ``restart.TORCH_INDEX_URLS == dev.TORCH_INDEX_URLS``, and that is
    deliberately not repeated. What is new in this PR is a SECOND reader of
    that map: a restart-mode install writes the index URL into the pending
    file and dev.py's helper installs from it, so what matters here is that
    ``resolve_gpu_torch`` hands back exactly the URL ``cdui install --gpu``
    would have used -- for every variant a user can pick.
    """
    import dev  # scripts/dev.py -- conftest puts scripts/ on sys.path

    for variant in restart.VARIANTS:
        if variant == "mps":
            assert dev.TORCH_INDEX_URLS[variant] is None, (
                "mps grew an index URL; resolve_gpu_torch must stop refusing it")
            continue
        assert restart.resolve_gpu_torch(variant) == (
            variant, dev.TORCH_INDEX_URLS[variant]), variant


# -- the pending file ------------------------------------------------------

def test_pending_round_trip_and_bad_shapes(monkeypatch, launcher, cu128):
    monkeypatch.setenv(restart.RELAUNCH_ARGV_ENV,
                       json.dumps(["--host", "127.0.0.1", "--port", "8000"]))

    torch_pending = restart.build_pending(
        get_pack("gpu-torch"), job_id="job-1", kind="torch")
    assert restart.PENDING_SCHEMA == 1
    assert torch_pending.schema == restart.PENDING_SCHEMA
    assert torch_pending.pack_id == "gpu-torch"
    assert torch_pending.kind == "torch"
    assert torch_pending.packages == ("torch", "torchvision")
    assert torch_pending.specs == ()
    assert torch_pending.index_url == restart.TORCH_INDEX_URLS["cu128"]
    assert torch_pending.venv_python == sys.executable
    assert torch_pending.server_pid == os.getpid()
    assert torch_pending.launcher == tuple(launcher)
    assert torch_pending.relaunch_argv == ("--host", "127.0.0.1",
                                           "--port", "8000")
    assert torch_pending.created_at.endswith("+00:00"), "created_at is not UTC"

    # An explicit variant wins over what the machine recommends.
    pinned = restart.build_pending(get_pack("gpu-torch"), job_id="job-1",
                                   kind="torch", variant="cu121")
    assert pinned.index_url == restart.TORCH_INDEX_URLS["cu121"]

    pack = get_pack("sentence-embeddings")
    pip_pending = restart.build_pending(pack, job_id="job-2", kind="pip")
    assert pip_pending.kind == "pip"
    assert pip_pending.specs == tuple(pack.pip)
    assert pip_pending.packages == ()
    assert pip_pending.index_url is None

    with pytest.raises(ValueError, match="conda"):
        restart.build_pending(pack, job_id="job-3", kind="conda")

    for original in (torch_pending, pip_pending):
        assert restart.PendingRestart.from_json(original.to_json()) == original

    raw = json.loads(torch_pending.to_json())
    assert set(raw) == PENDING_KEYS
    assert raw["packages"] == ["torch", "torchvision"], "tuples must serialise"
    assert raw["launcher"] == launcher

    bad = {
        "not json at all": "not json at all",
        "a list, not an object": json.dumps([1, 2, 3]),
        "a schema from the future": json.dumps({**raw, "schema": 2}),
        "a schema that is not a number": json.dumps({**raw, "schema": "1"}),
        "a missing key": json.dumps({k: v for k, v in raw.items()
                                     if k != "venv_python"}),
        "an unknown kind": json.dumps({**raw, "kind": "wheel"}),
        "a pid that is text": json.dumps({**raw, "server_pid": "4242"}),
        "a pid that is a bool": json.dumps({**raw, "server_pid": True}),
        "a launcher that is a string": json.dumps({**raw,
                                                   "launcher": "python dev.py"}),
        "a package that is a number": json.dumps({**raw,
                                                  "packages": ["torch", 7]}),
        "an index_url that is a number": json.dumps({**raw, "index_url": 7}),
        "an empty job id": json.dumps({**raw, "job_id": ""}),
    }
    for why, text in bad.items():
        try:
            restart.PendingRestart.from_json(text)
        except ValueError:
            continue
        pytest.fail(f"from_json accepted {why}")


def test_from_json_ignores_keys_it_does_not_know(launcher, cu128):
    """Forward compatibility in one direction, deliberately.

    This module and dev.py's helper are two implementations of one file,
    versioned together by ``schema`` -- but they are edited in different
    commits, and a helper from a newer install may add a field this reader
    has never heard of. Dropping it is what lets the older reader keep
    working; every field it DOES know is still checked, so nothing is
    silently mis-read.
    """
    original = restart.build_pending(get_pack("gpu-torch"), job_id="job-1",
                                     kind="torch")
    raw = json.loads(original.to_json())
    raw["variant"] = "cu128"                 # a field a later schema adds
    raw["notes"] = {"written_by": "the helper"}

    parsed = restart.PendingRestart.from_json(json.dumps(raw))

    assert parsed == original, "an unknown key changed how the rest was read"
    assert set(json.loads(parsed.to_json())) == PENDING_KEYS


def test_build_pending_refuses_a_pip_pack_with_nothing_to_install(launcher,
                                                                  cu128):
    """A pip restart for a pack with no specs restarts the server for nothing.

    The GPU pack is exactly that pack -- its ``pip`` is empty because its
    install is a wheel swap, ``kind="torch"`` -- so asking for it as a pip
    restart is a caller that has confused the two. The refusal names the
    pack, because that is what tells a reader which call was wrong.
    """
    with pytest.raises(ValueError, match="gpu-torch"):
        restart.build_pending(get_pack("gpu-torch"), job_id="job-1",
                              kind="pip")


def test_write_pending_refuses_a_fresh_live_one_and_overwrites_a_stale_one(
        monkeypatch, launcher, cu128):
    pack = get_pack("gpu-torch")
    first = restart.build_pending(pack, job_id="first", kind="torch")

    path = restart.write_pending(first)
    assert path == pending_restart_file()
    assert _pending_on_disk() == first
    assert not list(path.parent.glob("*.tmp")), "a temp file was left behind"

    # Another server, while the first is alive and its claim is fresh.
    monkeypatch.setattr(restart, "_pid_alive", lambda pid: True)
    second = restart.build_pending(pack, job_id="second", kind="torch")
    with pytest.raises(PendingExists) as exc:
        restart.write_pending(second)
    assert isinstance(exc.value, PackInstallError), (
        "the route's error mapping is written against PackInstallError")
    assert _pending_on_disk().job_id == "first", "the live claim was clobbered"

    # The same claim, once the server that made it is gone.
    monkeypatch.setattr(restart, "_pid_alive", lambda pid: False)
    restart.write_pending(second)
    assert _pending_on_disk().job_id == "second"

    # Alive, but the claim is older than a restart could possibly take.
    monkeypatch.setattr(restart, "_pid_alive", lambda pid: True)
    ancient = dataclasses.replace(
        second, job_id="ancient",
        created_at=_iso_ago(restart.STALE_PENDING_S + 60))
    path.write_text(ancient.to_json(), encoding="utf-8")
    third = restart.build_pending(pack, job_id="third", kind="torch")
    restart.write_pending(third)
    assert _pending_on_disk().job_id == "third"

    # A file nobody can read is not a claim anybody has to honour.
    path.write_text("{half written", encoding="utf-8")
    fourth = restart.build_pending(pack, job_id="fourth", kind="torch")
    restart.write_pending(fourth)
    assert _pending_on_disk().job_id == "fourth"


def test_clear_stale_pending(monkeypatch, launcher, cu128):
    pack = get_pack("gpu-torch")
    path = pending_restart_file()

    assert restart.clear_stale_pending() is False, "there was nothing to clear"

    monkeypatch.setattr(restart, "_pid_alive", lambda pid: True)
    restart.write_pending(restart.build_pending(pack, job_id="live",
                                                kind="torch"))
    assert restart.clear_stale_pending() is False
    assert path.exists(), "a live claim was deleted"

    monkeypatch.setattr(restart, "_pid_alive", lambda pid: False)
    assert restart.clear_stale_pending() is True
    assert not path.exists()
    assert restart.clear_stale_pending() is False, "deleting twice reported twice"

    # Old enough that whoever wrote it is not coming back, pid or no pid.
    monkeypatch.setattr(restart, "_pid_alive", lambda pid: True)
    aged = dataclasses.replace(
        restart.build_pending(pack, job_id="aged", kind="torch"),
        created_at=_iso_ago(restart.STALE_PENDING_S + 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(aged.to_json(), encoding="utf-8")
    assert restart.clear_stale_pending() is True
    assert not path.exists()

    path.write_text("{half written", encoding="utf-8")
    assert restart.clear_stale_pending() is True
    assert not path.exists()


def test_a_pending_file_that_is_not_text_self_heals(monkeypatch, launcher,
                                                    cu128):
    """Bytes that are not UTF-8 are not a claim, and must not be an exception.

    ``Path.read_text`` answers those with ``UnicodeDecodeError``, which is a
    ``ValueError`` -- so a reader that only catches ``OSError`` lets it out
    of BOTH :func:`write_pending` and :func:`clear_stale_pending`. The file
    would then refuse every future restart-mode install with a 500, and
    nothing in the product could delete it: the user would have to be told
    to go and find it by hand.
    """
    path = pending_restart_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    garbage = b"\xff\xfe{\x00\x00 not text \x80\x81"

    path.write_bytes(garbage)
    assert restart.clear_stale_pending() is True
    assert not path.exists()

    # And the next install writes over it rather than tripping over it.
    path.write_bytes(garbage)
    restart.write_pending(restart.build_pending(
        get_pack("gpu-torch"), job_id="after", kind="torch"))
    assert _pending_on_disk().job_id == "after"


def test_pid_alive_answers_for_a_real_process():
    """The one function here that asks the OS instead of a fixture.

    Every other test monkeypatches it, so without this the real
    implementation for whichever platform the suite is on -- ``os.kill(pid,
    0)`` on POSIX, ``OpenProcess`` plus ``GetExitCodeProcess`` on Windows --
    would never run at all, and it is what decides whether another server's
    pending file may be overwritten.
    """
    assert restart._pid_alive(os.getpid()) is True

    proc = subprocess.Popen([sys.executable, "-c", "pass"],
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    proc.wait(timeout=60)
    assert restart._pid_alive(proc.pid) is False


def test_pid_alive_never_signals_a_process_group(monkeypatch):
    """0 and -1 are GROUPS to ``os.kill``, and one of those groups is ours.

    Asserted on the POSIX path from any OS, because the mistake it guards
    against only bites there: ``os.kill(0, 0)`` succeeds (it signals this
    process's whole group) and would report an impossible pid as alive,
    while ``os.kill(-1, 0)`` reaches every process the user can signal.
    """
    monkeypatch.setattr(sys, "platform", "linux")

    def _never(pid, sig):
        pytest.fail(f"os.kill was called with pid {pid}")

    monkeypatch.setattr(os, "kill", _never)

    assert restart._pid_alive(0) is False
    assert restart._pid_alive(-1) is False


class _FakeExport:
    """One kernel32 entry point: callable, and it accepts the ``argtypes`` /
    ``restype`` declarations ``_pid_alive_windows`` writes onto it."""

    def __init__(self, impl):
        self._impl = impl
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self._impl(*args)


class _FakeKernel32:
    """As much of kernel32 as the pid check touches, and nothing else."""

    def __init__(self, *, handle: int, exit_code: int | None = None,
                 get_exit_ok: bool = True):
        self.closed: list = []
        self._exit_code = exit_code
        self._get_exit_ok = get_exit_ok
        self.OpenProcess = _FakeExport(
            lambda access, inherit, pid: handle)
        self.GetExitCodeProcess = _FakeExport(self._get_exit)
        self.CloseHandle = _FakeExport(self.closed.append)

    def _get_exit(self, handle, out) -> int:
        if not self._get_exit_ok:
            return 0
        # ``out`` is what ``ctypes.byref(c_ulong())`` produced; ``_obj`` is
        # the c_ulong the caller will read the value back out of.
        out._obj.value = self._exit_code
        return 1


def _install_kernel32(monkeypatch, kernel, *, last_error: int = 0) -> None:
    import ctypes

    # ``WinDLL`` does not exist off Windows, hence raising=False -- which is
    # the point: this pins the Windows branch from any runner.
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, **kwargs: kernel,
                        raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: last_error)


def test_pid_alive_windows_reads_the_exit_code_not_just_the_handle(monkeypatch):
    """``OpenProcess`` succeeding is not the answer, and that is the bug here.

    A handle opens fine for a process that has already exited but whose
    object has not been released, so a pid would read as alive forever and
    the pending file it wrote could never be replaced.
    """
    running = _FakeKernel32(handle=0x1234, exit_code=restart._STILL_ACTIVE)
    _install_kernel32(monkeypatch, running)
    assert restart._pid_alive_windows(4242) is True
    assert running.closed == [0x1234], "the process handle was leaked"

    exited = _FakeKernel32(handle=0x1234, exit_code=0)
    _install_kernel32(monkeypatch, exited)
    assert restart._pid_alive_windows(4242) is False
    assert exited.closed == [0x1234]


def test_pid_alive_windows_reads_a_refused_handle_by_its_error(monkeypatch):
    denied = _FakeKernel32(handle=0)
    _install_kernel32(monkeypatch, denied,
                      last_error=restart._ERROR_ACCESS_DENIED)
    assert restart._pid_alive_windows(4242) is True, (
        "ACCESS_DENIED means the process exists and belongs to someone else")
    assert denied.closed == [], "there was no handle to close"

    missing = _FakeKernel32(handle=0)
    _install_kernel32(monkeypatch, missing, last_error=87)  # INVALID_PARAMETER
    assert restart._pid_alive_windows(4242) is False


def test_pid_alive_windows_assumes_alive_when_it_cannot_tell(monkeypatch):
    """Every unknown answers "alive": the caller acts on False by DELETING
    another server's pending file."""
    unreadable = _FakeKernel32(handle=0x99, get_exit_ok=False)
    _install_kernel32(monkeypatch, unreadable)
    assert restart._pid_alive_windows(4242) is True
    assert unreadable.closed == [0x99], "the handle was leaked on the way out"

    import ctypes

    def _no_kernel32(name, **kwargs):
        raise OSError("kernel32 is not loadable here")

    monkeypatch.setattr(ctypes, "WinDLL", _no_kernel32, raising=False)
    assert restart._pid_alive_windows(4242) is True


def test_pid_alive_dispatches_to_the_windows_path_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(restart, "_pid_alive_windows", lambda pid: True)

    def _never(pid, sig):
        pytest.fail("os.kill was used on Windows")

    monkeypatch.setattr(os, "kill", _never)

    assert restart._pid_alive(4242) is True


# -- the detached helper ---------------------------------------------------

def _fake_popen(monkeypatch, pid: int = 4242) -> dict:
    """Record the next ``Popen`` call instead of making it."""
    seen: dict = {}

    class _FakeProc:
        def __init__(self):
            self.pid = pid

    def _popen(argv, **kwargs):
        seen["argv"] = list(argv)
        seen["kwargs"] = kwargs
        seen["stdout_name"] = getattr(kwargs.get("stdout"), "name", None)
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", _popen)
    return seen


def test_spawn_helper_argv_flags_and_log_file(monkeypatch, tmp_path, launcher,
                                              cu128):
    monkeypatch.setenv("PYTHONPATH", "D:/Github/CodefyUI/backend")
    monkeypatch.setenv("PYTHONHOME", "C:/uv/python/cpython-3.11-windows")
    monkeypatch.setenv("CODEFYUI_MARKER", "kept")
    path = restart.write_pending(restart.build_pending(
        get_pack("gpu-torch"), job_id="abc123", kind="torch"))
    seen = _fake_popen(monkeypatch)

    monkeypatch.setattr(sys, "platform", "win32")
    assert restart.spawn_helper(path) == 4242

    log_path = job_log_dir() / "restart-abc123.log"
    assert seen["argv"] == [*launcher, "packs-run-pending", str(path)]
    assert seen["stdout_name"] == str(log_path)
    assert log_path.exists(), "the helper has nowhere to write"

    kwargs = seen["kwargs"]
    assert kwargs["stderr"] is subprocess.STDOUT
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["creationflags"] & restart.DETACHED_PROCESS
    assert kwargs["creationflags"] & runner.CREATE_NEW_PROCESS_GROUP
    assert "start_new_session" not in kwargs, (
        "start_new_session is a POSIX-only kwarg")
    assert "shell" not in kwargs
    assert kwargs["env"]["PYTHONUTF8"] == "1"
    assert "PYTHONPATH" not in kwargs["env"], (
        "the helper would import this repo's app package")
    # The helper is the OUTER interpreter, and a uv-managed venv's python.exe
    # runs this server with PYTHONHOME pointing at the base interpreter it was
    # built from. Passing that down made the helper load the wrong stdlib and
    # die on `import argparse` -- after the server had already scheduled its
    # own shutdown, which is a box with no server and no helper on it.
    assert "PYTHONHOME" not in kwargs["env"], (
        "the helper would load this venv's base interpreter's stdlib")
    assert kwargs["env"]["CODEFYUI_MARKER"] == "kept", (
        "the environment is sanitised, not rebuilt")
    # The one variable the handshake actually depends on: the helper writes
    # the outcome record where the NEXT server will look for it, and both
    # find that directory through this variable. Filtering it would leave a
    # restart that worked and a panel that never hears about it.
    assert kwargs["env"]["CODEFYUI_USER_DATA_DIR"] == str(tmp_path)

    # POSIX detaches with a session of its own, and must not carry a Windows
    # constant into a call that would reject it.
    monkeypatch.setattr(sys, "platform", "linux")
    restart.spawn_helper(path)
    assert seen["kwargs"]["start_new_session"] is True
    assert "creationflags" not in seen["kwargs"]


def test_spawn_helper_log_name_stays_inside_the_log_directory(
        monkeypatch, launcher, cu128):
    """The job id is read back off disk and then joined into a path.

    Nothing writes anything but a uuid4 hex there, and the log is still not
    allowed to land wherever that id says: a name is built, not a path.
    """
    pending = restart.build_pending(get_pack("gpu-torch"),
                                    job_id="../../../evil job", kind="torch")
    path = restart.write_pending(pending)
    seen = _fake_popen(monkeypatch)

    restart.spawn_helper(path)

    written = Path(seen["stdout_name"])
    assert written.parent == job_log_dir()
    assert written.name == "restart-.._.._.._evil_job.log"


def test_spawn_helper_refuses_a_pending_it_cannot_act_on(monkeypatch, launcher,
                                                         cu128):
    """The file is read BEFORE the server schedules its own shutdown.

    A pending whose launcher is empty spawns nothing, and finding that out
    after the shutdown is scheduled leaves a stopped server and an install
    that never runs.
    """
    pending = restart.build_pending(get_pack("gpu-torch"), job_id="j",
                                    kind="torch")
    path = restart.write_pending(pending)
    _fake_popen(monkeypatch)

    path.write_text(dataclasses.replace(pending, launcher=()).to_json(),
                    encoding="utf-8")
    with pytest.raises(PackInstallError):
        restart.spawn_helper(path)

    path.write_text("{half written", encoding="utf-8")
    with pytest.raises(ValueError):
        restart.spawn_helper(path)


def test_spawn_helper_refuses_a_launcher_that_is_gone(monkeypatch, launcher,
                                                      cu128):
    """``restart_available`` checked this when the panel was DRAWN.

    Minutes may have passed since, and the pending file may have been
    written by an older server whose checkout has been moved or deleted
    since. Handing that path to ``Popen`` is a ``FileNotFoundError`` -- or,
    worse, whatever now sits at that path -- after which the server would
    shut down and nothing would bring it back.
    """
    path = restart.write_pending(restart.build_pending(
        get_pack("gpu-torch"), job_id="j", kind="torch"))
    seen = _fake_popen(monkeypatch)
    Path(launcher[0]).unlink()

    with pytest.raises(PackInstallError, match="launcher"):
        restart.spawn_helper(path)
    assert not seen, "a helper was started from an interpreter that is gone"


# -- going away ------------------------------------------------------------

def test_schedule_self_shutdown_raises_sigint_later():
    """Scheduled, never delivered here: a real SIGINT would end the run."""

    class _FakeLoop:
        def __init__(self):
            self.calls = []

        def call_later(self, delay, callback, *args):
            self.calls.append((delay, callback, args))
            return "handle"

    loop = _FakeLoop()
    assert restart.schedule_self_shutdown(loop) is None
    assert loop.calls == [(0.5, signal.raise_signal, (signal.SIGINT,))]

    restart.schedule_self_shutdown(loop, delay=2.5)
    assert loop.calls[-1] == (2.5, signal.raise_signal, (signal.SIGINT,))


# -- the outcome file ------------------------------------------------------

def test_write_and_read_last_restart():
    record = {
        "schema": restart.OUTCOME_SCHEMA,
        "job_id": "job-1",
        "pack_id": "gpu-torch",
        "kind": "torch",
        "status": "ok",
        "returncode": 0,
        "message": "GPU PyTorch (cu128) installed",
        "log_tail": "Installed 2 packages",
        "finished_at": "2026-08-29T09:00:00+00:00",
    }
    path = last_restart_file()
    assert not path.exists()

    restart.write_last_restart(record)

    assert json.loads(path.read_text(encoding="utf-8")) == record
    assert restart.read_last_restart() == record
    assert not list(path.parent.glob("*.tmp")), "a temp file was left behind"

    # The two keys the SPA is written against (packStore.checkInProgress).
    failed = {**record, "status": "failed", "returncode": 1,
              "message": "uv exited 1"}
    restart.write_last_restart(failed)
    read_back = restart.read_last_restart()
    assert read_back["status"] == "failed"
    assert read_back["message"] == "uv exited 1"
    assert read_back == failed, "the second record did not replace the first"
