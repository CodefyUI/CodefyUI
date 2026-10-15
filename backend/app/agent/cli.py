"""``cdui agent start|stop``: the worker agent's command line (#771).

``cdui agent ...`` reaches :func:`main` through ``scripts/agent.py`` inside
the backend venv; ``python -m app.agent ...`` is the same entry point.

Same-host defaults, so ``cdui agent start`` next to ``cdui start`` needs no
flags:

- ``--control``: the address the last ``cdui start`` recorded
  (``server.addr``), re-read at every reconnect, so a server restarted on
  another port is found again;
- ``--token``: the running server's session token file, also re-read at
  every reconnect (the token rotates per server process);
- ``--name``: this machine's hostname, lower-cased to a worker name.

When a default cannot be found and the terminal is interactive, the agent
asks for that value (and only for ``--control`` / ``--token``).

Background by default, like ``cdui start``: the agent is re-launched as a
detached ``--foreground`` process, its pid in ``<user-data>/agent.pid`` and
its log in ``<user-data>/agent.log``. ``--foreground`` keeps it attached
(systemd, a terminal you watch). The agent itself never imports torch.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from ..core import worker_protocol as _wp

#: The ``target.worker`` alphabet and the two names no worker may take,
#: from the protocol module both ends share.
WORKER_NAME_PATTERN = _wp.WORKER_NAME_PATTERN
RESERVED_NAMES = _wp.RESERVED_WORKER_NAMES

#: Env var carrying a token into the background process (never argv, which
#: any local user can read with ``ps``).
TOKEN_ENV = "CODEFYUI_AGENT_TOKEN"
#: Marks the re-launched background process.
DAEMON_FLAG = "--daemonized"

PIDFILE_NAME = "agent.pid"
LOGFILE_NAME = "agent.log"
RUNS_DIR_NAME = "agent-runs"

#: How long ``start`` watches a fresh background agent before reporting it.
START_SETTLE_S = 1.5
#: How long ``stop`` waits for the agent to wind its runs down.
STOP_WAIT_S = 30.0

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_REPO_ROOT = _BACKEND_DIR.parent

logger = logging.getLogger("app.agent")


# ── language ──────────────────────────────────────────────────────────────


def _lang() -> str:
    forced = (os.environ.get("CODEFYUI_LANG") or "").lower()
    if forced:
        return "zh" if forced.startswith("zh") or forced == "chinese" else "en"
    locale = (os.environ.get("LC_ALL") or os.environ.get("LANG") or "").lower()
    return "zh" if locale.startswith("zh") else "en"


def t(zh: str, en: str) -> str:
    return zh if _lang() == "zh" else en


def _say(zh: str, en: str) -> None:
    print(t(zh, en))


def _err(zh: str, en: str) -> None:
    print(t(zh, en), file=sys.stderr)


# ── paths and same-host defaults ──────────────────────────────────────────


def user_data_dir() -> Path:
    from ..config import _user_data_root

    return _user_data_root()


def pidfile() -> Path:
    return user_data_dir() / PIDFILE_NAME


def logfile() -> Path:
    return user_data_dir() / LOGFILE_NAME


def runs_dir() -> Path:
    return user_data_dir() / RUNS_DIR_NAME


def _server_addr_files() -> list[Path]:
    """Where ``cdui start`` may have written ``server.addr``: the user-data
    dir ``scripts/dev.py`` points ``CODEFYUI_USER_DATA_DIR`` at, then the
    repo's ``.codefyui_dev`` (``dev.SERVER_ADDRFILE``) when an outer
    caller pointed the user-data dir elsewhere."""
    return [user_data_dir() / "server.addr",
            _REPO_ROOT / ".codefyui_dev" / "server.addr"]


def last_server_url() -> str | None:
    """``http://host:port`` of the last ``cdui start``, or None."""
    for path in _server_addr_files():
        try:
            raw = path.read_text(encoding="utf-8").strip()
            host, _, port = raw.rpartition(":")
            port_number = int(port)
        except (OSError, ValueError):
            continue
        host = host.strip("[]") or "127.0.0.1"
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"
        if ":" in host:
            host = f"[{host}]"
        return f"http://{host}:{port_number}"
    return None


def session_token() -> str | None:
    """The running server's session token, or None when it cannot be read."""
    try:
        return token_file().read_text(
            encoding="ascii").strip() or None
    except (OSError, UnicodeDecodeError):
        return None


def default_name() -> str:
    """This host's name as a worker name (``[a-z0-9][a-z0-9-]*``)."""
    raw = socket.gethostname().split(".")[0].lower()
    name = re.sub(r"[^a-z0-9-]+", "-", raw).strip("-")[:63]
    if not name or name in RESERVED_NAMES:
        name = f"worker-{name}" if name else "worker"
    return name


# ── argument parsing ──────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cdui agent",
        description=t("CodefyUI worker agent：連到控制面，每個 run 一個子程序執行。",
                      "CodefyUI worker agent: connects to the control plane "
                      "and runs each assigned run in its own subprocess."))
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser(
        "start", help=t("啟動 agent（預設在背景）", "start the agent "
                        "(in the background by default)"))
    start.add_argument("--control", default=None, help=t(
        "控制面網址，例如 http://127.0.0.1:8000（預設：上次 cdui start 的位址）",
        "control-plane URL, e.g. http://127.0.0.1:8000 (default: the "
        "address of the last cdui start)"))
    start.add_argument("--token", default=None, help=t(
        "連線 token（預設：伺服器的 session token 檔）",
        "connection token (default: the server's session token file)"))
    start.add_argument("--name", default=None, help=t(
        "worker 名稱（預設：主機名稱）", "worker name (default: hostname)"))
    start.add_argument("--devices", default=None, help=t(
        "提供給控制面的裝置，逗號分隔，例如 cuda:0,cuda:1（預設：全部）",
        "devices to offer, comma-separated, e.g. cuda:0,cuda:1 "
        "(default: all)"))
    start.add_argument("--foreground", action="store_true", help=t(
        "在前景執行（systemd 或想看日誌時）",
        "stay in the foreground (systemd, or to watch the log)"))
    start.add_argument(DAEMON_FLAG, action="store_true",
                       help=argparse.SUPPRESS)
    sub.add_parser("stop", help=t("停止背景 agent", "stop the background agent"))
    return parser


def validate_name(name: str) -> str:
    name = name.strip().lower()
    if WORKER_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(t(
            f"worker 名稱 {name!r} 不合法：只能用小寫英數字與 -，最長 63 字",
            f"invalid worker name {name!r}: lower-case letters, digits and "
            "'-', at most 63 characters"))
    if name in RESERVED_NAMES:
        raise ValueError(t(f"worker 名稱 {name!r} 是保留字",
                           f"worker name {name!r} is reserved"))
    return name


def parse_devices(raw: str | None) -> list[str] | None:
    """``"cuda:0, CUDA:1"`` -> ``["cuda:0", "cuda:1"]``; None when not given."""
    if raw is None:
        return None
    devices = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not devices:
        raise ValueError(t("--devices 是空的", "--devices is empty"))
    return devices


def offered_reports(facts: list[Any], wanted: list[str] | None) -> list[Any]:
    """The probe's devices, narrowed to *wanted* (CPU is always offered)."""
    if wanted is None:
        return list(facts)
    known = {fact.value for fact in facts}
    unknown = [device for device in wanted if device not in known]
    if unknown:
        raise ValueError(t(
            f"這台機器沒有這些裝置：{', '.join(unknown)}（有：{', '.join(sorted(known))}）",
            f"device(s) not found on this machine: {', '.join(unknown)} "
            f"(found: {', '.join(sorted(known))})"))
    return [fact for fact in facts
            if fact.value in wanted or fact.kind == "cpu"]


def _interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def resolve_control(explicit: str | None, *, interactive: bool,
                    ask: Callable[[str], str] = input) -> str | Callable[[], str]:
    """``--control``, else a reader of ``server.addr``, else a prompt."""
    if explicit:
        return explicit
    if last_server_url() is not None:
        return lambda: last_server_url() or "http://127.0.0.1:8000"
    fallback = "http://127.0.0.1:8000"
    if interactive:
        answer = ask(t(f"控制面網址 [{fallback}]：",
                       f"Control-plane URL [{fallback}]: ")).strip()
        return answer or fallback
    return fallback


def resolve_token(explicit: str | None, *, interactive: bool,
                  ask: Callable[[str], str] = getpass.getpass,
                  ) -> str | None | Callable[[], str | None]:
    """``--token`` / ``$CODEFYUI_AGENT_TOKEN``, else the session token file
    (re-read per dial), else a prompt."""
    if explicit:
        return explicit
    env_token = os.environ.get(TOKEN_ENV)
    if env_token:
        return env_token
    if session_token() is not None or not interactive:
        return session_token
    answer = ask(t("連線 token：", "Connection token: ")).strip()
    return answer or session_token


def token_file() -> Path:
    return user_data_dir() / "session.token"


class MissingToken(ValueError):
    """No token was given and the session token file does not exist."""


def require_token(token: Any) -> None:
    """Refuse to start without a token: a reader that finds no file now.

    Only for a start in this terminal; the background process it launches
    re-reads the file at every dial and logs an error while it is missing.
    """
    if callable(token) and token() is None:
        raise MissingToken(t(
            f"找不到連線 token：沒有 --token、${TOKEN_ENV}，也沒有 "
            f"{token_file()}（先執行 cdui start，或用 --token 指定）",
            f"no connection token: no --token, no ${TOKEN_ENV}, and no "
            f"{token_file()} (run cdui start first, or pass --token)"))


# ── process helpers ───────────────────────────────────────────────────────


def _read_pid() -> int | None:
    try:
        return int(pidfile().read_text().strip())
    except (OSError, ValueError):
        return None


def _agent_process(pid: int) -> Any:
    """The ``psutil.Process`` for *pid* if it is a live agent, else None.

    The command line is checked so a pid reused by another program after
    the agent died is never signalled.
    """
    import psutil

    try:
        proc = psutil.Process(pid)
        cmdline = " ".join(proc.cmdline())
    except (psutil.Error, OSError):
        return None
    return proc if "app.agent" in cmdline else None


def running_agent_pid() -> int | None:
    """The background agent's pid, clearing a stale pidfile."""
    pid = _read_pid()
    if pid is None:
        return None
    if _agent_process(pid) is not None:
        return pid
    pidfile().unlink(missing_ok=True)
    return None


def _log_tail(lines: int = 20) -> str:
    try:
        return "\n".join(logfile().read_text(
            encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


# ── commands ──────────────────────────────────────────────────────────────


def cmd_stop(_args: argparse.Namespace) -> int:
    pid = running_agent_pid()
    if pid is None:
        _say("沒有執行中的背景 agent。", "No background agent is running.")
        return 0
    import psutil

    proc = _agent_process(pid)
    if proc is None:  # pragma: no cover - raced with its own exit
        pidfile().unlink(missing_ok=True)
        return 0
    # Windows has no SIGTERM: terminate() is TerminateProcess, which gives
    # the agent no chance to stop its runs, so take its run subprocesses
    # down with it rather than orphaning them.
    children = proc.children(recursive=True) if sys.platform == "win32" else []
    _say(f"正在停止 agent（PID {pid}）…", f"Stopping the agent (PID {pid})...")
    try:
        proc.terminate()
        for child in children:
            child.terminate()
        _, alive = psutil.wait_procs([proc, *children], timeout=STOP_WAIT_S)
        for straggler in alive:
            straggler.kill()
    except psutil.NoSuchProcess:
        pass
    pidfile().unlink(missing_ok=True)
    _say("agent 已停止。", "Agent stopped.")
    return 0


def _start_background(args: argparse.Namespace, control: Any,
                      token: Any) -> int:
    existing = running_agent_pid()
    if existing is not None:
        _say(f"agent 已在執行（PID {existing}）。用 cdui agent stop 停止它。",
             f"The agent is already running (PID {existing}). Stop it with "
             "cdui agent stop.")
        return 0
    argv = [sys.executable, "-m", "app.agent", "start", "--foreground",
            DAEMON_FLAG]
    env = dict(os.environ)
    if isinstance(control, str):
        argv += ["--control", control]
    if isinstance(token, str):
        env[TOKEN_ENV] = token
    if args.name:
        argv += ["--name", args.name]
    if args.devices:
        argv += ["--devices", args.devices]
    log_path = logfile()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    popen_kw: dict[str, Any] = {}
    if sys.platform == "win32":
        popen_kw["creationflags"] = (subprocess.CREATE_NO_WINDOW
                                     | subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        popen_kw["start_new_session"] = True
    with open(log_path, "a", buffering=1, encoding="utf-8") as log:
        proc = subprocess.Popen(argv, cwd=_BACKEND_DIR, env=env, stdout=log,
                                stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, **popen_kw)
    pidfile().write_text(str(proc.pid))
    deadline = time.monotonic() + START_SETTLE_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pidfile().unlink(missing_ok=True)
            _err("錯誤：agent 啟動後隨即結束。最後的日誌：",
                 "Error: the agent exited right after starting. Last log "
                 "lines:")
            print(_log_tail(), file=sys.stderr)
            return 1
        time.sleep(0.1)
    _say(f"=== agent 已在背景啟動（PID {proc.pid}）===",
         f"=== Agent started in the background (PID {proc.pid}) ===")
    _say(f"    日誌 → {log_path}", f"    log  → {log_path}")
    _say("    停止 → cdui agent stop", "    stop → cdui agent stop")
    return 0


def _setup_logging(daemonized: bool) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if not daemonized:  # a daemon's stderr IS agent.log already
        log_path = logfile()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO, handlers=handlers, force=True,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _install_stop_handlers(loop: asyncio.AbstractEventLoop,
                           stop: asyncio.Event) -> None:
    def request_stop(*_: Any) -> None:
        loop.call_soon_threadsafe(stop.set)

    for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, request_stop)
        except (NotImplementedError, RuntimeError, ValueError):
            signal.signal(sig, request_stop)  # Windows: no loop handlers


def run_agent(args: argparse.Namespace, control: Any, token: Any,
              devices: list[Any]) -> int:
    """The agent itself: connect and serve until stopped."""
    from ..config import settings
    from ..core import worker_protocol as wp
    from ..core.version import get_version
    from .connection import AgentConnection
    from .runner import Runner

    name = args.name
    wanted = parse_devices(args.devices)
    offered = {report.value for report in devices}
    runner = Runner(runs_dir=runs_dir(), offered_devices=offered,
                    keep_last=settings.RUN_RETENTION_KEEP_LAST)

    def probe() -> list[wp.DeviceReport]:
        return [_report(fact) for fact in offered_reports(
            _probe_facts(), wanted)]

    async def serve() -> int:
        stop = asyncio.Event()
        _install_stop_handlers(asyncio.get_running_loop(), stop)
        connection = AgentConnection(
            runner=runner, name=name, version=get_version(),
            control_url=control, token=token, probe_devices=probe,
            max_message_bytes=settings.WS_MAX_MESSAGE_BYTES,
            token_source=(f"the session token file {token_file()}"
                          if callable(token) else "--token / "
                          f"${TOKEN_ENV}"))
        return await connection.run(stop)

    logger.info("agent %s starting (pid %s); offering %s", name, os.getpid(),
                ", ".join(sorted(offered)))
    try:
        return asyncio.run(serve())
    finally:
        if args.daemonized and _read_pid() == os.getpid():
            pidfile().unlink(missing_ok=True)
        logger.info("agent stopped")


def _probe_facts() -> list[Any]:
    """This machine's devices without torch, plus MPS on Apple silicon.

    ``probe_devices(allow_torch=False)`` cannot ask torch whether MPS
    works, so it reports none; a macOS arm64 host has it (every Apple
    silicon Mac does), offered here with unknown memory.
    """
    from ..core.resources import DeviceFact, probe_devices

    facts = probe_devices(allow_torch=False)
    if apple_silicon() and not any(f.kind == "mps" for f in facts):
        facts.append(DeviceFact("mps", "mps", None, "Apple MPS", None, None,
                                "none"))
    return facts


def apple_silicon() -> bool:
    import platform

    return sys.platform == "darwin" and platform.machine() == "arm64"


def _report(fact: Any) -> Any:
    from ..core import worker_protocol as wp

    return wp.DeviceReport(
        value=fact.value, kind=fact.kind, index=fact.index, name=fact.name,
        total_bytes=fact.total_bytes, free_bytes=fact.free_bytes,
        source=fact.source, uuid=getattr(fact, "uuid", None))


def cmd_start(args: argparse.Namespace) -> int:
    # NVML numbers cards in PCI bus order; with this, CUDA does too, in this
    # process and in every run subprocess (``core.resources``).
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    try:
        args.name = validate_name(args.name or default_name())
        wanted = parse_devices(args.devices)
        facts = offered_reports(_probe_facts(), wanted)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    interactive = _interactive() and not args.daemonized
    control = resolve_control(args.control, interactive=interactive)
    token = resolve_token(args.token, interactive=interactive)
    if not args.daemonized:
        try:
            require_token(token)
        except MissingToken as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if not args.foreground:
        return _start_background(args, control, token)
    _setup_logging(args.daemonized)
    return run_agent(args, control, token, facts)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass
    args = build_parser().parse_args(argv)
    if args.command == "stop":
        return cmd_stop(args)
    return cmd_start(args)
