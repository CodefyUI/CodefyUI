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

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
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
    never saw) -- snapshot/restore the keys it touches.

    The three restart keys are load-bearing beyond this file: a leaked
    CODEFYUI_LAUNCHER makes `restart.restart_available()` answer True for
    every test that runs afterwards, and a whole suite would then take a
    branch nobody asked for.
    """
    saved = {k: os.environ.get(k)
             for k in ("CODEFYUI_HOST", "CODEFYUI_PORT", "CODEFYUI_MANAGED",
                       "CODEFYUI_LAUNCHER", "CODEFYUI_RELAUNCH_ARGV",
                       "CODEFYUI_OUTER_PYTHON")}
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
    command lists, plus `popen_kwargs` and `managed` -- the value of
    CODEFYUI_MANAGED as the child saw it, read at spawn time because the
    child inherits the environment as it stands then.
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
    # The sixth state file, and the only one not reached through a module
    # constant: `_restart_preflight` derives the claim's path from this
    # variable, and it DELETES what it finds there. Without this the tests
    # below that do not take `control` run against the developer's own
    # `<repo>/.codefyui_dev/packs/pending_restart.json`. The same directory
    # `control` uses, so the two fixtures cannot disagree in either order.
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path / "data"))

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

    rec: dict = {"popen": None, "popen_kwargs": None, "run": None,
                 "managed": None, "launcher": None, "relaunch": None}

    def _snapshot() -> None:
        """The environment AS THE CHILD SEES IT -- read at spawn time,
        because that is when it is copied into the server."""
        rec["managed"] = os.environ.get("CODEFYUI_MANAGED")
        rec["launcher"] = os.environ.get("CODEFYUI_LAUNCHER")
        rec["relaunch"] = os.environ.get("CODEFYUI_RELAUNCH_ARGV")

    def _fake_popen(cmd, **kw):
        rec["popen"] = list(cmd)
        rec["popen_kwargs"] = kw
        _snapshot()
        return _FakeProc()

    def _fake_run(cmd, **kw):
        rec["run"] = list(cmd)
        _snapshot()

    monkeypatch.setattr(dev.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(dev, "run", _fake_run)
    return rec


def _argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["cdui", "start", *args])


def _base_cmd(host: str = "127.0.0.1", port: str = "8000") -> list:
    """The uvicorn command every start path begins with, before the tail.

    ``--ws-max-size`` is spelled from the launcher's own helper rather than
    pinned to a literal here (core#274): the value is derived from the
    environment, and `test_ws_max_size.py` is what asserts the helper agrees
    with `settings.WS_MAX_MESSAGE_BYTES`. Pinning the number in seven places
    would only test that a constant is a constant.
    """
    return [
        "/fake/uvicorn",
        "app.main:app",
        "--host",
        host,
        "--port",
        port,
        "--ws-max-size",
        str(dev._ws_max_size()),
    ]


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
    assert started["popen"] == _base_cmd() + [
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
    assert started["run"] == _base_cmd() + ["--root-path", "/codefyui"]


def test_without_a_separator_the_command_is_unchanged(started, monkeypatch):
    _argv(monkeypatch, "--host", "0.0.0.0", "--port", "8080")
    dev.start()
    assert started["popen"] == _base_cmd("0.0.0.0", "8080")


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
    assert started["popen"] == _base_cmd() + ["--proxy-headers"]


def test_a_forwarded_foreground_flag_does_not_daemonize_differently(started, monkeypatch):
    """`-f` after `--` is uvicorn's problem, not a cdui foreground request."""
    _argv(monkeypatch, "--", "-f")
    dev.start()
    assert started["run"] is None, "`-f` after `--` must not select the foreground path"
    assert started["popen"] == _base_cmd() + ["-f"]


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


# ── the launch marker ─────────────────────────────────────────────────────
#
# Only dev.py knows whether the server it is starting is one it supervises.
# The Package Center reads CODEFYUI_MANAGED back as `launch_mode` to decide
# whether it may offer a restart-mode install, so a server started by hand
# (`uvicorn app.main:app`) has to stay "unknown" rather than inherit a claim
# nobody can honour.


def test_start_records_managed_env(started, monkeypatch):
    _argv(monkeypatch)
    dev.start()
    assert started["managed"] == "start"


def test_start_records_managed_env_on_the_foreground_path(started, monkeypatch):
    _argv(monkeypatch, "-f")
    dev.start()
    assert started["run"] is not None
    assert started["managed"] == "start"


# ── the restart handshake `cdui start` opens ──────────────────────────────
#
# A pack whose install replaces something the server has already imported can
# only be installed by going away and coming back, and NOTHING in the server
# knows how it was started. `cdui start` is the only process that does, so it
# writes both halves of the answer into the child's environment: what to run
# (CODEFYUI_LAUNCHER) and what to pass it (CODEFYUI_RELAUNCH_ARGV). Without
# them `restart.restart_available()` is False and the Package Center refuses
# the install with a command to type instead — which is the correct answer
# for a `uvicorn app.main:app` somebody started by hand.


def test_start_exports_launcher_and_relaunch_argv(started, monkeypatch):
    """Both halves reach the child, as JSON, round-tripping exactly."""
    _argv(monkeypatch, "--host", "0.0.0.0", "--port", "9100",
          "--", "--proxy-headers")
    dev.start()

    launcher = json.loads(started["launcher"])
    assert launcher == [dev._outer_python(), str(Path(dev.__file__).resolve())]
    # An interpreter and a script, not a shell line: the helper is started
    # with `Popen(argv)` and no shell anywhere.
    assert Path(launcher[1]).name == "dev.py"

    # The tail goes back behind its `--`, so the relaunched start splits it
    # the same way this one did.
    assert json.loads(started["relaunch"]) == [
        "--host", "0.0.0.0", "--port", "9100", "--", "--proxy-headers",
    ]


def test_start_exports_json_so_a_path_with_spaces_survives(started, monkeypatch,
                                                           tmp_path):
    """The whole reason these are JSON and not space-joined strings."""
    _argv(monkeypatch, "--project", str(tmp_path / "my proj"))
    monkeypatch.setattr(dev, "_activate_project", lambda raw: None)
    dev.start()
    assert json.loads(started["relaunch"]) == [
        "--project", str(tmp_path / "my proj")]


@pytest.mark.parametrize("flag", ["--project {}", "--project={}"])
def test_a_relative_project_is_absolutised_for_the_relaunch(
        started, monkeypatch, tmp_path, flag):
    """`cdui start --project ./lab`, restarted, must find the same lab.

    The helper relaunches with `cwd=ROOT` and `_activate_project` resolves
    against the current directory, so a relative path typed anywhere else
    names `<repo>/lab` on the way back: either nothing at all (exit 1, and
    the server never comes back from a restart-mode install) or, on a box
    that happens to have one, somebody else's project. What goes into
    `CODEFYUI_RELAUNCH_ARGV` is the directory `_activate_project` actually
    opened.
    """
    proj = tmp_path / "lab"
    proj.mkdir()
    (proj / "codefyui.project.toml").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _argv(monkeypatch, *flag.format("./lab").split(" "))

    dev.start()

    argv = json.loads(started["relaunch"])
    value = (argv[1] if argv[0] == "--project"
             else argv[0].split("=", 1)[1])
    assert Path(value).is_absolute(), "a relative path relaunches somewhere else"
    assert Path(value) == proj.resolve()
    assert len(argv) == (2 if argv[0] == "--project" else 1), (
        "the flag was duplicated rather than rewritten")


def test_an_absolute_project_reaches_the_relaunch_unchanged(started,
                                                            monkeypatch,
                                                            tmp_path):
    """The overwhelmingly common case, and the one that must not be mangled
    into a different directory by the rewrite above."""
    proj = tmp_path / "lab"
    proj.mkdir()
    (proj / "codefyui.project.toml").write_text("", encoding="utf-8")
    _argv(monkeypatch, "--project", str(proj))

    dev.start()

    assert Path(json.loads(started["relaunch"])[1]) == proj.resolve()


@pytest.mark.parametrize("flag", ["-f", "--foreground"])
def test_the_relaunch_is_always_a_daemon(started, monkeypatch, flag):
    """`--foreground` is stripped: the helper that relaunches has no console
    to hand over, and a foreground server parented by it would die with it."""
    _argv(monkeypatch, flag, "--port", "9100")
    dev.start()
    assert started["run"] is not None, "this is still the foreground path"
    assert json.loads(started["relaunch"]) == ["--port", "9100"]


def test_the_foreground_path_still_opens_the_handshake(started, monkeypatch):
    """A foreground server can be restarted too — it just comes back as a
    daemon, which the stripped flag above is what makes true."""
    _argv(monkeypatch, "-f")
    dev.start()
    assert started["launcher"] is not None
    assert json.loads(started["relaunch"]) == []


def test_start_with_no_arguments_relaunches_with_no_arguments(started,
                                                              monkeypatch):
    _argv(monkeypatch)
    dev.start()
    assert json.loads(started["relaunch"]) == []


# ── the launcher is the OUTER interpreter ─────────────────────────────────
#
# `start` is not in _SKIP_VENV_EXEC, so by the time it runs this process has
# already been re-exec'd as backend/.venv's python and `sys.executable` is
# that. A restart-mode install REWRITES that venv, so the helper must be the
# interpreter from outside it — which only exists if it was written down on
# the way through.


def test_the_venv_hop_records_the_outer_interpreter(monkeypatch, tmp_path):
    venv = tmp_path / ".venv"
    venv_py = venv / "python"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("")
    monkeypatch.setattr(dev, "VENV", venv)
    monkeypatch.setattr(dev, "VENV_PY", venv_py)
    monkeypatch.delenv(dev.OUTER_PYTHON_ENV, raising=False)

    seen: dict = {}
    monkeypatch.setattr(dev, "_reexec", lambda exe, argv: seen.update(
        exe=exe, outer=os.environ.get(dev.OUTER_PYTHON_ENV)))

    dev._exec_into_venv_if_available()

    assert seen["exe"] == str(venv_py), "the hop still happens"
    assert seen["outer"] == sys.executable, "recorded BEFORE the hop, not after"


def test_the_re_execd_child_does_not_overwrite_the_recording(monkeypatch,
                                                             tmp_path):
    """The child runs this same function and IS inside the venv. Overwriting
    there would replace the answer with the one value it must never be.

    The recorded interpreter is deliberately NOT this test's own: an
    assignment and a `setdefault` are indistinguishable when the value being
    written happens to equal the value already there.
    """
    venv = tmp_path / ".venv"
    venv_py = venv / "python"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("")
    outer = tmp_path / "outer-python"                      # the parent's
    outer.write_text("")
    monkeypatch.setattr(dev, "VENV", venv)
    monkeypatch.setattr(dev, "VENV_PY", venv_py)
    monkeypatch.setattr(sys, "prefix", str(venv))          # we are inside it
    monkeypatch.setenv(dev.OUTER_PYTHON_ENV, str(outer))
    monkeypatch.setattr(dev, "_reexec", lambda exe, argv: pytest.fail(
        "already inside the venv; there is nothing to hop into"))

    dev._exec_into_venv_if_available()

    assert os.environ[dev.OUTER_PYTHON_ENV] == str(outer)
    assert dev._outer_python() == str(outer), "and it is what the launcher uses"


def _reexeced(monkeypatch, *, console: bool) -> dict:
    """Run `_reexec`'s Windows branch with the console probe pinned."""
    seen: dict = {}

    class _Done:
        returncode = 0

    def _run(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        seen["kwargs"] = kwargs
        return _Done()

    monkeypatch.setattr(sys, "platform", "win32")
    # POSIX has no such constant, and this test states the branch rather than
    # inheriting it from the machine it runs on.
    monkeypatch.setattr(dev.subprocess, "CREATE_NO_WINDOW", 0x08000000,
                        raising=False)
    monkeypatch.setattr(dev.subprocess, "run", _run)
    monkeypatch.setattr(dev, "_has_console_window", lambda: console)

    with pytest.raises(SystemExit) as exc:
        dev._reexec("C:/py/python.exe", ["dev.py", "start"])

    assert exc.value.code == 0
    assert seen["cmd"] == ["C:/py/python.exe", "dev.py", "start"]
    return seen


def test_the_windows_hop_hands_the_child_the_parents_own_stdio(monkeypatch):
    """Windows has no real exec, so `_reexec` runs the child and forwards its
    exit code -- and a child given no stdio arguments does NOT inherit a
    console-less parent's handles: it attaches to a new console of its own.

    From the restart helper (detached, stdout redirected into the job log)
    that means every line the relaunched `start()` prints -- the reach lines,
    "the server exited right after start", the log tail it prints to explain
    why -- goes to a window that closes with the process. The log ends up
    holding the helper's own lines and nothing else, which is exactly the
    moment somebody is reading it.
    """
    seen = _reexeced(monkeypatch, console=True)

    kwargs = seen["kwargs"]
    assert kwargs["stdin"] is sys.stdin
    assert kwargs["stdout"] is sys.stdout
    assert kwargs["stderr"] is sys.stderr


def test_the_windows_hop_opens_no_console_when_the_parent_has_none(monkeypatch):
    """...and while it is at it, does not open a WINDOW either.

    Only when there is no console to keep: an interactive `cdui` run owns
    one, and taking it away would break the `[y/N]` prompt in
    `cdui plugin install` all over again (this function's whole docstring).
    """
    seen = _reexeced(monkeypatch, console=False)
    assert seen["kwargs"]["creationflags"] & dev.subprocess.CREATE_NO_WINDOW

    seen = _reexeced(monkeypatch, console=True)
    assert "creationflags" not in seen["kwargs"], (
        "an interactive parent keeps its console, and its input()")


def test_the_console_probe_answers_rather_than_raising():
    """It decides whether a user's terminal keeps its console. Anything it
    cannot answer must read as "there is one" -- the behaviour every `cdui`
    run had before this flag existed."""
    assert isinstance(dev._has_console_window(), bool)


def test_outer_python_falls_back_when_the_recorded_one_is_gone(monkeypatch,
                                                               tmp_path):
    """A stale value inherited from a shell, or an interpreter since removed.
    A launcher that is not on disk is refused outright by
    `restart.restart_available`, so answering with one is worse than
    answering with the venv's python."""
    monkeypatch.setenv(dev.OUTER_PYTHON_ENV, str(tmp_path / "not-a-python"))
    assert dev._outer_python() == sys.executable


def test_outer_python_prefers_the_recording_over_this_interpreter(monkeypatch,
                                                                  tmp_path):
    recorded = tmp_path / "outer-python"
    recorded.write_text("")
    monkeypatch.setenv(dev.OUTER_PYTHON_ENV, str(recorded))
    assert dev._outer_python() == str(recorded)


# ── the pending-restart preflight ─────────────────────────────────────────
#
# One file, two failure modes, and they pull in opposite directions.
#
# A claim left behind by a server that crashed between writing the file and
# exiting refuses every future restart-mode install with "one is already
# pending", so it has to be cleared. But a claim belonging to a restart that
# is STILL RUNNING must not be — starting a second server on the port the
# helper is about to relaunch onto puts a process back on the very files it
# is replacing, which is the whole failure a restart-mode install exists to
# avoid.
#
# `_pending_state` is what tells them apart, and `cdui status` renders the
# same answer, so the two commands can never describe one file differently.


@pytest.fixture
def control(tmp_path, monkeypatch) -> Path:
    """The packs control directory `cdui start` reads its claim out of."""
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path / "data"))
    path = tmp_path / "data" / "packs"
    path.mkdir(parents=True)
    return path


def _write_pending(control: Path, *, pid: int = 4242, age_s: float = 0.0,
                   helper_pid: "int | None" = None) -> Path:
    path = control / "pending_restart.json"
    created = datetime.now(timezone.utc) - timedelta(seconds=age_s)
    claim = {
        "schema": 1, "job_id": "j1", "pack_id": "gpu-torch", "kind": "torch",
        "index_url": "https://download.pytorch.org/whl/cu128",
        "packages": ["torch", "torchvision"], "specs": [],
        "venv_python": "/nowhere/python", "server_pid": pid,
        "launcher": ["/py", "/dev.py"], "relaunch_argv": [],
        "created_at": created.isoformat(),
    }
    if helper_pid is not None:
        claim["helper_pid"] = helper_pid
    path.write_text(json.dumps(claim), encoding="utf-8")
    return path


def test_the_sandbox_keeps_the_preflight_out_of_the_real_user_data(started,
                                                                   tmp_path):
    """Every `start()` runs `_restart_preflight`, and that reads -- and
    DELETES -- `<user data>/packs/pending_restart.json`. Asserted once, here,
    because without it every test in this file that does not take `control`
    judges the developer's own claim in `<repo>/.codefyui_dev`, and an
    abandoned real one gets deleted by running the suite."""
    assert dev._pending_restart_file().is_relative_to(tmp_path)


def test_start_stands_down_while_a_helper_is_still_running(
        started, monkeypatch, control, capsys):
    """(a) A live `helper_pid`. Not an error: the user asked for a running
    server and one is on its way — from the process that is mid-install."""
    path = _write_pending(control, helper_pid=777)
    monkeypatch.setattr(dev, "_pid_alive", lambda pid: pid == 777)
    _argv(monkeypatch)

    dev.start()          # no SystemExit: standing down is not a failure

    assert started["popen"] is None, "a second server would fight the helper"
    assert path.exists(), "the helper still needs its own claim"
    out = capsys.readouterr().out
    assert "cdui status" in out, "and the user is told where to watch it"
    assert str(path) in out, (
        "a pid reads as alive whenever the OS has recycled that number, and "
        "this branch has no deadline — the claim's path is the way out")


def test_start_clears_a_claim_whose_helper_died(started, monkeypatch, control,
                                                capsys):
    """(b) The helper wrote its pid and then died — nothing will ever finish
    this install, and nothing else may start until the claim is gone."""
    path = _write_pending(control, helper_pid=777)
    monkeypatch.setattr(dev, "_pid_alive", lambda pid: False)
    _argv(monkeypatch)

    dev.start()

    assert not path.exists()
    assert started["popen"] is not None, "and the server starts"
    assert "restart" in capsys.readouterr().out.lower(), (
        "a deletion the user cannot see is a mystery")


def test_start_stands_down_for_a_live_helper_past_the_fifteen_minutes(
        started, monkeypatch, control):
    """(b') A LIVE helper outranks the clock. Fifteen minutes is nothing to a
    torch download over a slow line, and an install still running at minute
    sixteen is finishing, not abandoned — deleting its claim and starting a
    second server would put two writers in one site-packages."""
    path = _write_pending(control, helper_pid=777,
                          age_s=dev.STALE_PENDING_S + 300)
    monkeypatch.setattr(dev, "_pid_alive", lambda pid: pid == 777)
    _argv(monkeypatch)

    dev.start()

    assert started["popen"] is None
    assert path.exists()


def test_start_clears_an_old_claim_whose_helper_is_gone(started, monkeypatch,
                                                        control):
    """(b'') The other half of the same rule: the age cap never rescues a
    claim, it only condemns one. A pid that is no longer alive is a dead
    install at any age, and the user gets their server back."""
    path = _write_pending(control, helper_pid=777,
                          age_s=dev.STALE_PENDING_S + 300)
    monkeypatch.setattr(dev, "_pid_alive", lambda pid: False)
    _argv(monkeypatch)

    dev.start()

    assert not path.exists()
    assert started["popen"] is not None


def test_start_stands_down_for_a_claim_whose_helper_has_not_written_its_pid(
        started, monkeypatch, control):
    """(c) The gap this rule exists for: the server has spawned the helper
    and exited, and the helper is a detached process that has not reached its
    first statement yet. Deleting here would race a restart that is fine."""
    path = _write_pending(control, age_s=5)
    monkeypatch.setattr(dev, "_pid_alive", lambda pid: False)
    _argv(monkeypatch)

    dev.start()

    assert started["popen"] is None
    assert path.exists()


def test_the_grace_for_a_helper_that_has_not_stamped_its_pid_is_a_minute():
    """The number, not just the rule. Every other test here is written against
    the constant, so it would follow the constant anywhere — and this window
    is the exact length of time a user whose helper never started sits in
    front of a launcher that refuses to give them a server. The docs quote it
    (`optional-packs.md`, `cli-commands.md`), so it is part of the contract."""
    assert dev.HELPER_START_GRACE_S == 60


def test_start_clears_a_claim_whose_helper_never_arrived(started, monkeypatch,
                                                         control):
    """(d) Past the grace window with still no `helper_pid`: the server wrote
    the claim and died before its helper ever ran, so nobody is coming."""
    path = _write_pending(control, age_s=dev.HELPER_START_GRACE_S + 30)
    monkeypatch.setattr(dev, "_pid_alive", lambda pid: True)
    _argv(monkeypatch)

    dev.start()

    assert not path.exists()
    assert started["popen"] is not None


def test_start_falls_back_to_the_files_own_age(started, monkeypatch, control):
    """`created_at` is written by another process and may be missing or
    nonsense in exactly the file this has to judge. The mtime is the second
    clock, and a claim with neither readable age is treated as old — an age
    nobody can establish must not be the reason a user has no server."""
    path = control / "pending_restart.json"
    path.write_text(json.dumps({"schema": 1, "pack_id": "gpu-torch",
                                "server_pid": 4242}), encoding="utf-8")
    old = time.time() - (dev.HELPER_START_GRACE_S + 60)
    os.utime(path, (old, old))
    _argv(monkeypatch)

    dev.start()

    assert not path.exists()


def test_start_deletes_a_pending_file_nothing_can_read(started, monkeypatch,
                                                       control):
    """Writes are atomic on both sides, so an unreadable file is not a
    half-written claim — it is not a claim, and leaving it there would refuse
    every install forever."""
    path = control / "pending_restart.json"
    path.write_bytes(b"\xff\xfe not json at all")
    _argv(monkeypatch)

    dev.start()

    assert not path.exists()
    assert started["popen"] is not None


def test_start_says_nothing_when_there_is_no_pending_file(started, monkeypatch,
                                                          tmp_path, capsys):
    """The overwhelmingly common case. A launcher that reports on a file
    that is not there teaches people to ignore it."""
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path / "data"))
    _argv(monkeypatch)
    dev.start()
    assert "restart" not in capsys.readouterr().out.lower()


def test_a_failed_delete_does_not_stop_the_server_starting(started, monkeypatch,
                                                           control):
    """The claim is dead either way, and the server the user asked for
    matters more than the tidying."""
    _write_pending(control, age_s=dev.STALE_PENDING_S + 60)

    def _refuse(self, *a, **kw):
        raise OSError("held open by something else")

    monkeypatch.setattr(Path, "unlink", _refuse)
    _argv(monkeypatch)

    dev.start()

    assert started["popen"] is not None


# ── the other two commands that run out of the venv a helper is rewriting ──
#
# `start` stands down at exit 0 -- the user asked for a running server and
# one is on its way, so their request is being satisfied by somebody else.
# Nobody else is going to run the user's `update`, so these two REFUSE, at
# exit 1, matching `update`'s own "a server is running" refusal that scripts
# already gate on.


class _ReachedLaunch(Exception):
    """Raised by a stubbed Popen: proof `dev()` got as far as launching."""


def _update_argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["cdui", "update", *args])


def _updatable(monkeypatch, started) -> list:
    """Stub everything `update()` does AFTER the preflight, and record the
    call to `install` that ends it. `dev.run` is already a recorder, so the
    git commands land in ``started["run"]``."""
    (dev.ROOT / ".git").mkdir(parents=True, exist_ok=True)
    installed: list = []
    monkeypatch.setattr(dev, "_resolve_update_options",
                        lambda argv: (None, False))
    monkeypatch.setattr(dev, "install", lambda **kw: installed.append(kw))
    monkeypatch.setattr(dev.shutil, "which", lambda name: "/fake/pnpm")
    return installed


def test_update_refuses_while_a_restart_is_finishing(started, monkeypatch,
                                                     control, tmp_path,
                                                     capsys):
    """`update` hard-realigns the checkout, deletes frontend/dist and
    rewrites the venv -- while a detached helper may be mid-`uv pip install`
    into that same venv. Two writers in one site-packages, and the loser is
    whichever finishes second."""
    installed = _updatable(monkeypatch, started)
    path = _write_pending(control, helper_pid=777)
    monkeypatch.setattr(dev, "_pid_alive", lambda pid: pid == 777)
    _update_argv(monkeypatch)

    with pytest.raises(SystemExit) as leaving:
        dev.update()

    assert leaving.value.code == 1
    assert started["run"] is None, "update reached git under a live install"
    assert installed == []
    assert path.exists(), "the helper still needs its own claim"
    assert "cdui status" in capsys.readouterr().out
    # The same sandbox guard as `test_the_sandbox_keeps_the_preflight_out_of
    # _the_real_user_data`: the preflight DELETES what it finds, and a test
    # that read the developer's own claim would delete that.
    assert dev._pending_restart_file().is_relative_to(tmp_path)


def test_update_proceeds_once_an_abandoned_claim_is_cleared(started,
                                                            monkeypatch,
                                                            control, tmp_path):
    """The other half, or the fix would trade one wedge for another: a claim
    nobody is acting on is cleared and the update runs."""
    installed = _updatable(monkeypatch, started)
    path = _write_pending(control, helper_pid=777,
                          age_s=dev.STALE_PENDING_S + 60)
    monkeypatch.setattr(dev, "_pid_alive", lambda pid: False)
    _update_argv(monkeypatch)

    dev.update()

    assert not path.exists()
    assert started["run"] is not None, "update never reached git"
    assert installed == [{"gpu": None, "dev": False}]
    assert dev._pending_restart_file().is_relative_to(tmp_path)


def _devable(monkeypatch) -> None:
    monkeypatch.setattr(dev.shutil, "which", lambda name: "/fake/pnpm")
    monkeypatch.setattr(dev, "_install_frontend_deps_if_needed", lambda: None)
    monkeypatch.setattr(sys, "argv", ["cdui", "dev"])


def test_dev_refuses_while_a_restart_is_finishing(started, monkeypatch,
                                                 control, tmp_path):
    """`cdui dev` starts a `--reload` uvicorn out of the venv the helper is
    replacing. It checked nothing at all before this."""
    _devable(monkeypatch)
    monkeypatch.setattr(dev.subprocess, "Popen",
                        lambda cmd, **kw: pytest.fail(f"launched {cmd}"))
    path = _write_pending(control, helper_pid=777)
    monkeypatch.setattr(dev, "_pid_alive", lambda pid: pid == 777)

    with pytest.raises(SystemExit) as leaving:
        dev.dev()

    assert leaving.value.code == 1
    assert path.exists()
    assert dev._pending_restart_file().is_relative_to(tmp_path)


def test_dev_proceeds_once_an_abandoned_claim_is_cleared(started, monkeypatch,
                                                         control, tmp_path):
    """A dead claim is cleared and `cdui dev` launches."""
    _devable(monkeypatch)
    launched: list = []

    def _reached(cmd, **kw):
        launched.append(list(cmd))
        raise _ReachedLaunch

    monkeypatch.setattr(dev.subprocess, "Popen", _reached)
    path = _write_pending(control, helper_pid=777,
                          age_s=dev.STALE_PENDING_S + 60)
    monkeypatch.setattr(dev, "_pid_alive", lambda pid: False)

    with pytest.raises(_ReachedLaunch):
        dev.dev()

    assert launched and launched[0][0] == "/fake/uvicorn", launched
    assert not path.exists()
    assert dev._pending_restart_file().is_relative_to(tmp_path)
