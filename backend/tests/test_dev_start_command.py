"""`cdui start` argument forwarding (core#249).

`cdui start -- <uvicorn args>` exists so that a reverse-proxy deployment can
reach `--proxy-headers`, `--root-path` and `--forwarded-allow-ips` WITHOUT
abandoning the daemon -- invoking uvicorn by hand gives up the pidfile,
`cdui status` and `cdui stop`.

Two properties are load-bearing and are what these tests pin:

1. The tail after `--` reaches uvicorn's command line verbatim, on BOTH the
   foreground and the background/daemon path (they share one `cmd` list).
2. The separator cuts in both directions. cdui's own flag scanners
   (`_parse_host_port`, `_parse_project`, the `-f` scan) are positional and
   would otherwise read a forwarded `--host` / `--project` / `-f` as their
   own. After the split they only ever see the head.

Repo gotcha this file obeys: stubbing `run` alone is NOT enough when testing
`dev.py`. `ROOT`, `VENV`, `DIST_DIR` and `FRONTEND_DIR` are read at call time
and any `shutil.rmtree` in the command under test executes inline, against the
developer's real `frontend/dist`. Every test below redirects those four plus
the three server state files at `tmp_path`, and an autouse fixture asserts the
real `frontend/` tree is untouched.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import dev  # scripts/dev.py — conftest puts scripts/ on sys.path

REAL_FRONTEND = Path(dev.__file__).resolve().parent.parent / "frontend"


@pytest.fixture(autouse=True)
def _real_frontend_survives():
    before = REAL_FRONTEND.exists() and sorted(p.name for p in REAL_FRONTEND.iterdir())
    yield
    after = REAL_FRONTEND.exists() and sorted(p.name for p in REAL_FRONTEND.iterdir())
    assert before == after, "cdui start touched the real frontend/ tree"


@pytest.fixture(autouse=True)
def _restore_bind_env():
    """`start()` writes raw os.environ (monkeypatch cannot undo a write it
    never saw) -- snapshot/restore the two keys it touches."""
    saved = {k: os.environ.get(k) for k in ("CODEFYUI_HOST", "CODEFYUI_PORT")}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class _FakeProc:
    """Just enough of Popen for start()'s health-poll loop: alive forever."""

    pid = 4242

    def poll(self):
        return None


@pytest.fixture
def started(tmp_path, monkeypatch):
    """Sandbox `start()` and return a recorder of what it would have launched.

    Returns a dict with `popen` (background path) and `run` (foreground path)
    command lists, plus `popen_kwargs`.
    """
    root = tmp_path / "codefyui"
    frontend = root / "frontend"
    dist = frontend / "dist"
    dist.mkdir(parents=True)
    index = dist / "index.html"
    index.write_text("<!doctype html>")
    backend = root / "backend"
    backend.mkdir(parents=True)

    monkeypatch.setattr(dev, "ROOT", root)
    monkeypatch.setattr(dev, "BACKEND_DIR", backend)
    monkeypatch.setattr(dev, "FRONTEND_DIR", frontend)
    monkeypatch.setattr(dev, "DIST_DIR", dist)
    monkeypatch.setattr(dev, "DIST_INDEX", index)
    monkeypatch.setattr(dev, "VENV", root / "venv")
    monkeypatch.setattr(dev, "SERVER_PIDFILE", tmp_path / "state" / "server.pid")
    monkeypatch.setattr(dev, "SERVER_ADDRFILE", tmp_path / "state" / "server.addr")
    monkeypatch.setattr(dev, "SERVER_LOG", tmp_path / "state" / "server.log")
    monkeypatch.setattr(dev, "DEV_LOCKFILE", tmp_path / "state" / "installed.json")

    monkeypatch.setattr(dev, "_require_venv_tool", lambda name: f"/fake/{name}")
    monkeypatch.setattr(dev, "_running_server_pid", lambda: None)
    monkeypatch.setattr(dev, "_warn_if_dist_stale", lambda: None)
    monkeypatch.setattr(dev, "_apply_dev_env", lambda: None)
    monkeypatch.setattr(dev, "_print_uninstalled_builtin_packs", lambda: None)
    monkeypatch.setattr(dev, "_local_ips", lambda: ["10.0.0.9"])
    monkeypatch.setattr(dev, "_server_healthy", lambda *a, **kw: True)
    # `t()` reads this global at call time and it is derived from the
    # developer's locale. Pin it so assertions are about behaviour, not about
    # which machine ran them.
    monkeypatch.setattr(dev, "LANG", "en")

    rec: dict = {"popen": None, "popen_kwargs": None, "run": None}

    def _fake_popen(cmd, **kw):
        rec["popen"] = list(cmd)
        rec["popen_kwargs"] = kw
        return _FakeProc()

    monkeypatch.setattr(dev.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(dev, "run", lambda cmd, **kw: rec.__setitem__("run", list(cmd)))
    return rec


def _argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["cdui", "start", *args])


# ── _split_forwarded_args ─────────────────────────────────────────────────


def test_split_without_separator_forwards_nothing():
    assert dev._split_forwarded_args(["--host", "0.0.0.0", "-f"]) == (
        ["--host", "0.0.0.0", "-f"],
        [],
    )


def test_split_partitions_at_the_separator():
    head, tail = dev._split_forwarded_args(
        ["--port", "9000", "--", "--proxy-headers", "--root-path", "/x"]
    )
    assert head == ["--port", "9000"]
    assert tail == ["--proxy-headers", "--root-path", "/x"]


def test_split_consumes_only_the_first_separator():
    """A later `--` is a uvicorn argument, not a second cut."""
    head, tail = dev._split_forwarded_args(["--", "--a", "--", "--b"])
    assert head == []
    assert tail == ["--a", "--", "--b"]


def test_split_with_empty_tail_is_a_noop():
    assert dev._split_forwarded_args(["-f", "--"]) == (["-f"], [])


def test_split_does_not_alias_the_input_list():
    argv = ["--host", "1.2.3.4"]
    head, tail = dev._split_forwarded_args(argv)
    head.append("mutated")
    assert argv == ["--host", "1.2.3.4"]
    assert tail == []


# ── _reject_owned_uvicorn_flags ───────────────────────────────────────────


def test_forwarding_proxy_flags_is_allowed():
    dev._reject_owned_uvicorn_flags(
        ["--proxy-headers", "--forwarded-allow-ips", "127.0.0.1", "--root-path=/x"]
    )


@pytest.mark.parametrize(
    "extra",
    [
        ["--host", "0.0.0.0"],
        ["--host=0.0.0.0"],
        ["--port", "9000"],
        ["--port=9000"],
        ["--proxy-headers", "--port", "9000"],
    ],
)
def test_forwarding_a_bind_flag_cdui_owns_is_refused(extra, capsys):
    with pytest.raises(SystemExit) as exc:
        dev._reject_owned_uvicorn_flags(extra)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "cdui start" in err


def test_a_value_that_merely_looks_like_a_bind_flag_is_not_refused():
    """Only the flag NAME is inspected, never a value."""
    dev._reject_owned_uvicorn_flags(["--forwarded-allow-ips=--host", "--header", "x: --port"])


# ── start(): the tail reaches uvicorn ─────────────────────────────────────


def test_daemon_path_appends_the_forwarded_args(started, monkeypatch):
    _argv(monkeypatch, "--", "--proxy-headers", "--forwarded-allow-ips", "127.0.0.1")
    dev.start()
    assert started["popen"] == [
        "/fake/uvicorn",
        "app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--proxy-headers",
        "--forwarded-allow-ips",
        "127.0.0.1",
    ]
    # The daemon is still a daemon: detached, and the pidfile was written.
    assert dev.SERVER_PIDFILE.read_text() == "4242"
    assert dev.SERVER_ADDRFILE.read_text() == "127.0.0.1:8000"


def test_foreground_path_appends_the_forwarded_args(started, monkeypatch):
    _argv(monkeypatch, "-f", "--", "--root-path", "/codefyui")
    dev.start()
    assert started["popen"] is None
    assert started["run"] == [
        "/fake/uvicorn",
        "app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--root-path",
        "/codefyui",
    ]


def test_without_a_separator_the_command_is_unchanged(started, monkeypatch):
    _argv(monkeypatch, "--host", "0.0.0.0", "--port", "8080")
    dev.start()
    assert started["popen"] == [
        "/fake/uvicorn",
        "app.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
    ]


def test_app_target_stays_at_index_one(started, monkeypatch):
    """`cdui stop`'s process matchers key on `app.main:app`; extras go last."""
    _argv(monkeypatch, "--", "--timeout-keep-alive", "75")
    dev.start()
    assert started["popen"][1] == "app.main:app"
    assert started["popen"][-2:] == ["--timeout-keep-alive", "75"]


def test_cdui_own_flags_still_parse_alongside_a_forwarded_tail(started, monkeypatch):
    _argv(monkeypatch, "--host", "127.0.0.1", "--port", "9100", "--", "--proxy-headers")
    dev.start()
    assert started["popen"][:6] == [
        "/fake/uvicorn",
        "app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "9100",
    ]
    assert dev.SERVER_ADDRFILE.read_text() == "127.0.0.1:9100"
    assert os.environ["CODEFYUI_PORT"] == "9100"


# ── the separator cuts in BOTH directions ─────────────────────────────────


def test_a_forwarded_host_does_not_move_cduis_own_bind(started, monkeypatch):
    """Refused rather than silently applied -- but the point is that
    `_parse_host_port` never sees it."""
    _argv(monkeypatch, "--", "--host", "0.0.0.0")
    with pytest.raises(SystemExit) as exc:
        dev.start()
    assert exc.value.code == 2
    assert started["popen"] is None


def test_the_separator_never_becomes_a_flag_value(started, monkeypatch):
    """A dangling `--host` must not swallow the separator.

    `_parse_host_port` reads the NEXT token after `--host`. Handed the raw
    argv it would bind to the literal string `--`; handed the head it sees a
    value-less flag and keeps the default, exactly as `cdui start --host`
    with nothing after it does today.
    """
    _argv(monkeypatch, "--host", "--", "--proxy-headers")
    dev.start()
    assert dev.SERVER_ADDRFILE.read_text() == "127.0.0.1:8000"
    assert started["popen"] == [
        "/fake/uvicorn",
        "app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--proxy-headers",
    ]


def test_a_forwarded_foreground_flag_does_not_daemonize_differently(started, monkeypatch):
    """`-f` after `--` is uvicorn's problem, not a cdui foreground request."""
    _argv(monkeypatch, "--", "-f")
    dev.start()
    assert started["run"] is None, "`-f` after `--` must not select the foreground path"
    assert started["popen"] == [
        "/fake/uvicorn",
        "app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "-f",
    ]


def test_a_forwarded_project_flag_is_not_activated(started, monkeypatch):
    """`_activate_project` exits(1) on a missing manifest. If the tail were
    scanned, this would die instead of forwarding."""
    activated: list = []
    monkeypatch.setattr(dev, "_activate_project", lambda raw: activated.append(raw))
    _argv(monkeypatch, "--", "--project", "/definitely/not/a/project")
    dev.start()
    assert activated == []
    assert started["popen"][-2:] == ["--project", "/definitely/not/a/project"]


def test_cduis_own_project_flag_still_works_with_a_tail(started, monkeypatch):
    activated: list = []
    monkeypatch.setattr(dev, "_activate_project", lambda raw: activated.append(raw))
    _argv(monkeypatch, "--project", "/some/proj", "--", "--proxy-headers")
    dev.start()
    assert activated == ["/some/proj"]
    assert started["popen"][-1] == "--proxy-headers"
