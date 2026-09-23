"""`cdui start` and `cdui dev` do not launch onto an address already in use.

`start()` used to recognise a running server only through the pidfile, which
only its own background path writes. A foreground `cdui start -f` (the
systemd deployment), a `cdui dev`, or another install's server was invisible
to it, so a second start launched a process that ran its whole startup and
then died at bind time. Now the launcher looks at the address first: when it
is taken it says what holds it -- a CodefyUI server, or some other program --
and exits 1 with nothing launched and nothing written.

The probe is the part with platform traps, so it is tested against real
sockets on loopback addresses:

* a FREE port must be recognised without a network round trip -- a connect
  to a free localhost port on Windows waits out its whole timeout, which
  every normal `cdui start` would then pay;
* only the bind address itself can stop the start as "in use by another
  program". 0.0.0.0 and 127.0.0.1 are looked at as well, because on Windows
  a listener on one does not collide with a bind on the other -- but there
  they count only when a CodefyUI server answers.

When the server is launched and does not come up (the instance lock, past
this check), `server.addr` is put back and `-f` passes on the server's exit
code.

Repo gotcha this file obeys: `ROOT`, `VENV`, `DIST_DIR`, `FRONTEND_DIR` and
the three server state files are redirected at `tmp_path` in every test that
calls `start()` or `dev()`, `run()` and `Popen` are recorders, and an autouse
fixture asserts the real `frontend/` tree is untouched.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import dev  # scripts/dev.py -- conftest puts scripts/ on sys.path

REAL_FRONTEND = Path(dev.__file__).resolve().parent.parent / "frontend"


@pytest.fixture(autouse=True)
def _real_frontend_survives():
    before = REAL_FRONTEND.exists() and sorted(p.name for p in REAL_FRONTEND.iterdir())
    yield
    after = REAL_FRONTEND.exists() and sorted(p.name for p in REAL_FRONTEND.iterdir())
    assert before == after, "a launcher test touched the real frontend/ tree"


@pytest.fixture(autouse=True)
def _restore_launch_env():
    """`start()` and `dev()` write raw os.environ, which monkeypatch cannot
    undo; put back every key they touch."""
    keys = ("CODEFYUI_HOST", "CODEFYUI_PORT", "CODEFYUI_MANAGED",
            "CODEFYUI_LAUNCHER", "CODEFYUI_RELAUNCH_ARGV",
            "CODEFYUI_OUTER_PYTHON", "CODEFYUI_PROJECT_DIR",
            "WEB_CONCURRENCY")
    saved = {key: os.environ.get(key) for key in keys}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


class _ReachedLaunch(Exception):
    """Raised by a stubbed Popen: proof the command got as far as launching."""


class _FakeProc:
    """Enough of Popen for start()'s health poll: alive, and healthy."""

    pid = 4242

    def poll(self):
        return None


@pytest.fixture
def launcher(tmp_path, monkeypatch) -> list:
    """Sandbox `start()` / `dev()`; returns the list of launched commands."""
    root = tmp_path / "codefyui"
    frontend = root / "frontend"
    dist = frontend / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html>")
    backend = root / "backend"
    backend.mkdir(parents=True)

    monkeypatch.setattr(dev, "ROOT", root)
    monkeypatch.setattr(dev, "BACKEND_DIR", backend)
    monkeypatch.setattr(dev, "FRONTEND_DIR", frontend)
    monkeypatch.setattr(dev, "DIST_DIR", dist)
    monkeypatch.setattr(dev, "DIST_INDEX", dist / "index.html")
    monkeypatch.setattr(dev, "VENV", root / "venv")
    monkeypatch.setattr(dev, "SERVER_PIDFILE", tmp_path / "state" / "server.pid")
    monkeypatch.setattr(dev, "SERVER_ADDRFILE", tmp_path / "state" / "server.addr")
    monkeypatch.setattr(dev, "SERVER_LOG", tmp_path / "state" / "server.log")
    monkeypatch.setattr(dev, "DEV_LOCKFILE", tmp_path / "state" / "installed.json")
    # `_restart_preflight` reads -- and deletes -- a claim under this dir.
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path / "data"))

    monkeypatch.setattr(dev, "_require_venv_tool", lambda name: f"/fake/{name}")
    monkeypatch.setattr(dev, "_running_server_pid", lambda: None)
    monkeypatch.setattr(dev, "_warn_if_dist_stale", lambda: None)
    monkeypatch.setattr(dev, "_apply_dev_env", lambda: None)
    monkeypatch.setattr(dev, "_print_uninstalled_builtin_packs", lambda: None)
    monkeypatch.setattr(dev, "_install_frontend_deps_if_needed", lambda: None)
    monkeypatch.setattr(dev, "_local_ips", lambda: ["10.0.0.9"])
    monkeypatch.setattr(dev.shutil, "which", lambda name: f"/fake/{name}")
    # The poll AFTER a background launch. The check before it is the real one.
    monkeypatch.setattr(dev, "_server_healthy", lambda *a, **kw: True)
    # Pinned: `t()` reads it at call time, and it follows the machine's locale.
    monkeypatch.setattr(dev, "LANG", "en")
    # A taken port that never answers HTTP costs this much per test.
    monkeypatch.setattr(dev, "_ALREADY_RUNNING_HEALTH_TIMEOUT_S", 0.5)

    launched: list = []

    def _popen(cmd, **kw):
        launched.append(list(cmd))
        return _FakeProc()

    monkeypatch.setattr(dev.subprocess, "Popen", _popen)
    monkeypatch.setattr(dev, "run", lambda cmd, **kw: launched.append(list(cmd)))
    return launched


# -- stand-ins for what might already hold a port ---------------------------

@contextlib.contextmanager
def _codefyui_server(body: dict):
    """A tiny HTTP server on 127.0.0.1 answering ``/api/health`` with *body*.

    Bound without SO_REUSEADDR: the option means "may share this port" on
    Windows, and uvicorn does not set it there.
    """
    class _Health(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - the stdlib's name
            payload = json.dumps(body).encode("utf-8")
            self.send_response(200 if self.path == "/api/health" else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Health,
                                 bind_and_activate=False)
    server.allow_reuse_address = False
    server.server_bind()
    server.server_activate()
    # A short poll interval: shutdown() waits for the loop to notice.
    thread = threading.Thread(target=server.serve_forever,
                              kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


@contextlib.contextmanager
def _taken_port(address: str = "127.0.0.1"):
    """A port held by a program that is not CodefyUI -- it listens, and
    never answers HTTP. An address this machine cannot bind (127.0.0.2
    outside Linux and Windows) skips the test."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        try:
            sock.bind((address, 0))
        except OSError as exc:
            pytest.skip(f"cannot bind {address} here: {exc}")
        sock.listen()
        yield sock.getsockname()[1]
    finally:
        sock.close()


@contextlib.contextmanager
def _banner_service(banner: bytes):
    """A service on 127.0.0.1 that greets each connection with *banner* --
    as SSH does -- and never speaks HTTP."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(0.1)
    stop = threading.Event()

    def serve() -> None:
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except OSError:
                continue
            with conn:
                conn.sendall(banner)
                # Read until the client hangs up: closing with its request
                # unread would be a reset, which can race the banner away.
                conn.settimeout(2)
                with contextlib.suppress(OSError):
                    while conn.recv(4096):
                        pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield listener.getsockname()[1]
    finally:
        stop.set()
        thread.join(timeout=10)
        listener.close()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


#: What CodefyUI's /api/health answers, minus what no test reads. The probe
#: recognises it by `status` and `nodes_loaded`.
_HEALTH = {"status": "ok", "version": "9.9.9", "boot_id": "b" * 32,
           "nodes_loaded": 3}


# -- the probe ------------------------------------------------------------------

def test_a_free_port_is_found_free_without_a_network_round_trip(monkeypatch):
    """Every normal start takes this path, so it must cost nothing: on
    Windows, asking a free localhost port for /api/health waits out the whole
    timeout before the refusal arrives."""
    monkeypatch.setattr(dev, "urlopen", lambda *a, **kw: pytest.fail(
        "asked a free port for /api/health"))
    assert dev._server_already_running("127.0.0.1", _free_port()) is None


def test_a_codefyui_server_on_the_port_is_recognised():
    with _codefyui_server(_HEALTH) as port:
        assert dev._server_already_running("127.0.0.1", port) == (
            "codefyui", _HEALTH)


def test_anything_else_on_the_port_is_reported_as_another_program(
        monkeypatch):
    monkeypatch.setattr(dev, "_ALREADY_RUNNING_HEALTH_TIMEOUT_S", 0.5)
    with _taken_port() as port:
        assert dev._server_already_running("127.0.0.1", port) == ("other", None)


def test_a_json_health_answer_that_is_not_codefyuis_is_another_program():
    """Plenty of programs answer /api/health. CodefyUI's carries
    `nodes_loaded`, and has since its first release."""
    with _codefyui_server({"status": "ok"}) as port:
        assert dev._server_already_running("127.0.0.1", port) == ("other", None)


def test_a_service_that_does_not_speak_http_is_another_program():
    """An SSH banner comes back from urllib as BadStatusLine, which is not an
    OSError -- it must not escape the probe as a traceback."""
    with _banner_service(b"SSH-2.0-OpenSSH_9.6\r\n") as port:
        assert dev._server_already_running("127.0.0.1", port) == ("other", None)


def test_a_codefyui_server_on_loopback_is_found_from_a_wildcard_start():
    """On Windows, binding 0.0.0.0 beside a 127.0.0.1 listener succeeds, and
    http://localhost:<port> would still reach the server already there."""
    with _codefyui_server(_HEALTH) as port:
        assert dev._server_already_running("0.0.0.0", port) == (
            "codefyui", _HEALTH)


@pytest.mark.parametrize("taken, start_host", [
    ("127.0.0.2", "127.0.0.1"),     # stopped a default start on Linux
    ("127.0.0.1", "127.0.0.2"),     # stopped a concrete --host everywhere
])
def test_another_program_on_another_address_is_not_in_the_way(
        monkeypatch, taken, start_host):
    """Only the bind address can stop the bind. The other addresses are
    looked at for a CodefyUI server, and another program there is none of
    this start's business."""
    monkeypatch.setattr(dev, "_ALREADY_RUNNING_HEALTH_TIMEOUT_S", 0.5)
    with _taken_port(taken) as port:
        assert dev._server_already_running(start_host, port) is None


@pytest.mark.skipif(sys.platform not in ("win32", "linux"),
                    reason="BSD sockets settle this bind their own way")
def test_another_program_on_loopback_stops_a_wildcard_start_only_where_the_bind_fails(
        monkeypatch):
    """Windows binds 0.0.0.0 beside a 127.0.0.1 listener, so that start goes
    ahead; Linux refuses the same bind, so there the start is stopped."""
    monkeypatch.setattr(dev, "_ALREADY_RUNNING_HEALTH_TIMEOUT_S", 0.5)
    with _taken_port("127.0.0.1") as port:
        expected = None if sys.platform == "win32" else ("other", None)
        assert dev._server_already_running("0.0.0.0", port) == expected


# -- `cdui start` ----------------------------------------------------------------

@pytest.mark.parametrize("foreground", [False, True],
                         ids=["background", "foreground"])
def test_start_refuses_when_a_codefyui_server_already_answers(
        launcher, monkeypatch, capsys, foreground):
    with _codefyui_server(_HEALTH) as port:
        monkeypatch.setattr(sys, "argv", ["cdui", "start", "--port", str(port),
                                          *(["-f"] if foreground else [])])
        with pytest.raises(SystemExit) as leaving:
            dev.start()

    assert leaving.value.code == 1
    assert launcher == [], "a server was launched onto a taken port"
    assert not dev.SERVER_ADDRFILE.exists(), (
        "server.addr now names an address this start never served on")
    err = capsys.readouterr().err
    assert f"http://localhost:{port}" in err
    assert "already running" in err and "9.9.9" in err
    assert "cdui stop" in err and "--port" in err


def test_start_refuses_when_another_program_holds_the_port(launcher,
                                                           monkeypatch, capsys):
    with _taken_port() as port:
        monkeypatch.setattr(sys, "argv", ["cdui", "start", "--port", str(port)])
        with pytest.raises(SystemExit) as leaving:
            dev.start()

    assert leaving.value.code == 1
    assert launcher == []
    err = capsys.readouterr().err
    assert f"Port {port}" in err and "another program" in err
    assert "--port" in err


def test_start_says_nothing_and_launches_when_the_port_is_free(launcher,
                                                               monkeypatch,
                                                               capsys):
    port = _free_port()
    monkeypatch.setattr(sys, "argv", ["cdui", "start", "--port", str(port)])
    dev.start()
    assert launcher and launcher[0][:2] == ["/fake/uvicorn", "app.main:app"]
    assert "already" not in capsys.readouterr().err


def test_its_own_background_server_is_still_answered_by_the_pidfile(
        launcher, monkeypatch, capsys):
    """Unchanged: `cdui start` beside the background server this install
    started says so and exits 0 -- the port is never probed for it."""
    monkeypatch.setattr(dev, "_running_server_pid", lambda: 4321)
    monkeypatch.setattr(dev, "_server_already_running", lambda *a: pytest.fail(
        "probed the port of a server the pidfile already names"))
    monkeypatch.setattr(sys, "argv", ["cdui", "start"])
    dev.start()
    assert launcher == []
    assert "4321" in capsys.readouterr().out


def test_the_refusal_speaks_the_cli_language(launcher, monkeypatch, capsys):
    monkeypatch.setattr(dev, "LANG", "zh")
    with _codefyui_server(_HEALTH) as port:
        monkeypatch.setattr(sys, "argv", ["cdui", "start", "--port", str(port)])
        with pytest.raises(SystemExit):
            dev.start()
    err = capsys.readouterr().err
    assert "已有 CodefyUI 伺服器在執行" in err
    assert "already running" not in err


@pytest.mark.parametrize("extra", [["--workers", "2"], ["--workers=2"]])
def test_workers_cannot_be_forwarded(extra, monkeypatch, capsys):
    """Every worker would be a second server on the same data: the instance
    lock refuses all but the first, and uvicorn restarts them without end."""
    monkeypatch.setattr(dev, "LANG", "en")
    with pytest.raises(SystemExit) as leaving:
        dev._reject_owned_uvicorn_flags(extra)
    assert leaving.value.code == 2
    err = capsys.readouterr().err
    assert "--workers" in err and "single process" in err
    assert "server.addr" not in err, "that is the bind flags' explanation"


# -- a start whose server does not come up ----------------------------------------
#
# Past the port check the usual cause is the instance lock: another server on
# the same data, on another port. server.addr is written before the launch, so
# it has to be put back -- `cdui run` and `cdui status` read it.

_PREVIOUS = pytest.mark.parametrize("previous", ["127.0.0.1:8000", None],
                                    ids=["an-earlier-address", "none"])


def _with_server_addr(previous: "str | None") -> None:
    dev.SERVER_ADDRFILE.parent.mkdir(parents=True, exist_ok=True)
    if previous is not None:
        dev.SERVER_ADDRFILE.write_text(previous)


def _server_addr_now() -> "str | None":
    return (dev.SERVER_ADDRFILE.read_text()
            if dev.SERVER_ADDRFILE.exists() else None)


@_PREVIOUS
def test_a_refused_foreground_start_exits_with_the_servers_code(
        launcher, monkeypatch, capsys, previous):
    """`cdui start -f` is what systemd runs: the server's own exit code (3,
    a startup failure) comes back out, with no launcher traceback."""
    _with_server_addr(previous)
    monkeypatch.setattr(dev, "_server_already_running", lambda h, p: None)

    def refused(cmd, **kw):
        raise dev.subprocess.CalledProcessError(3, cmd)

    monkeypatch.setattr(dev, "run", refused)
    monkeypatch.setattr(sys, "argv", ["cdui", "start", "-f", "--port", "8471"])

    with pytest.raises(SystemExit) as leaving:
        dev.start()

    assert leaving.value.code == 3
    assert _server_addr_now() == previous


@_PREVIOUS
def test_a_background_server_that_dies_at_once_leaves_server_addr_alone(
        launcher, monkeypatch, capsys, previous):
    _with_server_addr(previous)
    monkeypatch.setattr(dev, "_server_already_running", lambda h, p: None)

    class _Dead:
        pid = 4242

        def poll(self):
            return 3

    monkeypatch.setattr(dev.subprocess, "Popen", lambda cmd, **kw: _Dead())
    monkeypatch.setattr(sys, "argv", ["cdui", "start", "--port", "8471"])

    with pytest.raises(SystemExit) as leaving:
        dev.start()

    assert leaving.value.code == 1
    assert _server_addr_now() == previous
    assert not dev.SERVER_PIDFILE.exists()


def test_a_server_that_comes_up_keeps_its_address(launcher, monkeypatch):
    """The other half: a start that succeeds records where it serves."""
    _with_server_addr("127.0.0.1:8000")
    monkeypatch.setattr(dev, "_server_already_running", lambda h, p: None)
    monkeypatch.setattr(sys, "argv", ["cdui", "start", "--port", "8471"])
    dev.start()
    assert _server_addr_now() == "127.0.0.1:8471"


@pytest.mark.parametrize("meanwhile", [None, "127.0.0.1:9999"],
                         ids=["deleted-by-cdui-stop", "rewritten-by-a-start"])
def test_a_foreground_server_stopped_later_leaves_server_addr_to_its_new_owner(
        launcher, monkeypatch, meanwhile):
    """It came up and served; then `cdui stop` deleted server.addr and killed
    it (exit 1 under taskkill /F), or another start wrote its own address.
    Either way the file is not this start's to put back any more."""
    _with_server_addr("127.0.0.1:8000")
    monkeypatch.setattr(dev, "_server_already_running", lambda h, p: None)

    def came_up_then_stopped(cmd, **kw):
        if meanwhile is None:
            dev.SERVER_ADDRFILE.unlink()
        else:
            dev.SERVER_ADDRFILE.write_text(meanwhile)
        raise dev.subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(dev, "run", came_up_then_stopped)
    monkeypatch.setattr(sys, "argv", ["cdui", "start", "-f", "--port", "8471"])

    with pytest.raises(SystemExit) as leaving:
        dev.start()

    assert leaving.value.code == 1
    assert _server_addr_now() == meanwhile


@pytest.mark.parametrize("code, expected", [
    (3, 3),                 # uvicorn's startup failure: the lock's refusal
    (-9, 137),              # POSIX, killed by SIGKILL: 128 + 9, not 247
    (-15, 143),             # POSIX, SIGTERM
    (0xC0000005, 1),        # Windows access violation: too big for sys.exit
])
def test_a_foreground_servers_exit_status_becomes_the_launchers(
        launcher, monkeypatch, code, expected):
    monkeypatch.setattr(dev, "_server_already_running", lambda h, p: None)

    def exited(cmd, **kw):
        raise dev.subprocess.CalledProcessError(code, cmd)

    monkeypatch.setattr(dev, "run", exited)
    monkeypatch.setattr(sys, "argv", ["cdui", "start", "-f", "--port", "8471"])

    with pytest.raises(SystemExit) as leaving:
        dev.start()

    assert leaving.value.code == expected


@pytest.mark.parametrize("foreground", [False, True],
                         ids=["background", "foreground"])
def test_web_concurrency_does_not_reach_the_server(launcher, monkeypatch,
                                                   capsys, foreground):
    """uvicorn takes WEB_CONCURRENCY as --workers when the flag is absent --
    the refusal of `-- --workers` would otherwise have a way round it."""
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    monkeypatch.setattr(dev, "_server_already_running", lambda h, p: None)
    seen: list = []

    def _spawn(cmd, **kw):
        seen.append(os.environ.get("WEB_CONCURRENCY", "<unset>"))
        return _FakeProc()

    monkeypatch.setattr(dev.subprocess, "Popen", _spawn)
    monkeypatch.setattr(dev, "run", _spawn)
    monkeypatch.setattr(sys, "argv", ["cdui", "start", "--port", "8471",
                                      *(["-f"] if foreground else [])])
    dev.start()

    assert seen == ["<unset>"]
    assert "WEB_CONCURRENCY=4" in capsys.readouterr().err


# -- `cdui dev` ------------------------------------------------------------------

def test_dev_refuses_before_launching_anything(launcher, monkeypatch, capsys):
    """`cdui dev` has no --port: its backend is uvicorn's default address."""
    asked: list = []

    def _taken(host, port):
        asked.append((host, port))
        return "codefyui", {"status": "ok", "version": "9.9.9"}

    monkeypatch.setattr(dev, "_server_already_running", _taken)
    monkeypatch.setattr(dev.subprocess, "Popen",
                        lambda cmd, **kw: pytest.fail(f"launched {cmd}"))
    monkeypatch.setattr(sys, "argv", ["cdui", "dev"])

    with pytest.raises(SystemExit) as leaving:
        dev.dev()

    assert leaving.value.code == 1
    assert asked == [("127.0.0.1", 8000)]
    err = capsys.readouterr().err
    assert "http://localhost:8000" in err and "already running" in err
    assert "--port" not in err, "cdui dev has no --port to suggest"


def test_dev_launches_when_its_port_is_free(launcher, monkeypatch):
    monkeypatch.setattr(dev, "_server_already_running", lambda host, port: None)

    def _reached(cmd, **kw):
        raise _ReachedLaunch

    monkeypatch.setattr(dev.subprocess, "Popen", _reached)
    monkeypatch.setattr(sys, "argv", ["cdui", "dev"])
    with pytest.raises(_ReachedLaunch):
        dev.dev()
