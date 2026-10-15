"""``cdui agent start|stop|forget``: the worker agent's command line (#771).

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

Another machine (#780): ``--control URL --token T`` with an enrollment
token from the Workers panel (``abcdef.0123456789abcdef``) enrolls once and
stores the worker credential it gets (``app.agent.enroll``,
``<user-data>/agent/<host>-<port>.token``, 0600). Every later
``cdui agent start --control URL`` finds that credential and uses it, with
or without ``--token``, and asks for nothing; ``cdui agent forget URL``
deletes it so the next start enrolls again.

``--mount LABEL=PATH`` (repeatable, #782) declares a shared directory, e.g.
``--mount data=/nas/lab/codefyui/data``: the control plane sends runs that
need files only to workers holding its label, and the run subprocesses
derive their data directories from it. Mounts are saved in
``<user-data>/agent.json`` (``agent_config``), so a later ``cdui agent
start`` keeps them; ``--mount LABEL=`` removes one.

Trust (#786): over ``https`` the agent verifies the control plane's
certificate (``app.agent.trust``) for the enrollment request and every
``wss`` dial. ``--ca-fingerprint sha256:<hex>`` pins the SPKI hash of the
server's leaf certificate (recorded in the credential file, so later starts
keep it; a mismatch is refused); without it the system CAs decide and
nothing is recorded. Over plain ``http`` to another machine one warning is
printed at start.

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
import threading
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
    start.add_argument("--mount", action="append", default=None,
                       metavar="LABEL=PATH", help=t(
        "共用儲存：LABEL=PATH，例如 data=/nas/lab/codefyui/data（可重複；"
        "會保存，下次啟動沿用；LABEL= 移除）",
        "shared storage: LABEL=PATH, e.g. data=/nas/lab/codefyui/data "
        "(repeatable; saved and reused by later starts; LABEL= removes "
        "one)"))
    start.add_argument("--ca-fingerprint", default=None, help=t(
        "控制面伺服器憑證（leaf）公鑰的指紋 sha256:<hex>；釘選後不檢查 CA，"
        "不符就拒絕連線",
        "fingerprint of the control plane's server (leaf) certificate "
        "public key, sha256:<hex>; pins it instead of checking CAs, and "
        "refuses to connect when it does not match"))
    start.add_argument("--uds", default=None, metavar="PATH", help=t(
        "經由這個 Unix socket 連到控制面（SSH worker 的通道）；--control "
        "仍是控制面的網址",
        "reach the control plane through this Unix socket (an SSH "
        "worker's tunnel); --control still names the control plane"))
    start.add_argument("--foreground", action="store_true", help=t(
        "在前景執行（systemd 或想看日誌時）",
        "stay in the foreground (systemd, or to watch the log)"))
    start.add_argument(DAEMON_FLAG, action="store_true",
                       help=argparse.SUPPRESS)
    sub.add_parser("stop", help=t("停止背景 agent", "stop the background agent"))
    forget = sub.add_parser("forget", help=t(
        "刪掉這台機器存下的某個控制面的 worker 憑證",
        "delete the worker credential stored for a control plane"))
    forget.add_argument("control", help=t(
        "控制面網址，例如 http://192.168.1.5:8000",
        "control-plane URL, e.g. http://192.168.1.5:8000"))
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
    """``"cuda:0, CUDA:01, mps"`` -> ``["cuda:0", "cuda:1", "mps:0"]``, in
    ``worker_protocol.canonical_device``'s spelling (the one the agent and
    the control plane compare devices in); None when not given."""
    from ..core import worker_protocol as wp

    if raw is None:
        return None
    devices = []
    for part in raw.split(","):
        if not part.strip():
            continue
        try:
            device = wp.canonical_device(part)
            if device == "cuda":  # which card: the agent offers cards
                raise ValueError(device)
            devices.append(device)
        except ValueError:
            raise ValueError(t(
                f"--devices 看不懂 {part.strip()!r}（請用 cpu、cuda:N 或 mps）",
                f"--devices: unknown device {part.strip()!r} (use cpu, "
                "cuda:N or mps)")) from None
    if not devices:
        raise ValueError(t("--devices 是空的", "--devices is empty"))
    return devices


def _canonical(value: str) -> str:
    from ..core import worker_protocol as wp

    try:
        return wp.canonical_device(value)
    except ValueError:
        return value


def offered_reports(facts: list[Any], wanted: list[str] | None) -> list[Any]:
    """The probe's devices, narrowed to *wanted* (CPU is always offered).
    Compared canonically: the probe reports MPS as ``mps``."""
    if wanted is None:
        return list(facts)
    wanted = [_canonical(device) for device in wanted]
    known = {_canonical(fact.value) for fact in facts}
    unknown = [device for device in wanted if device not in known]
    if unknown:
        raise ValueError(t(
            f"這台機器沒有這些裝置：{', '.join(unknown)}（有：{', '.join(sorted(known))}）",
            f"device(s) not found on this machine: {', '.join(unknown)} "
            f"(found: {', '.join(sorted(known))})"))
    return [fact for fact in facts
            if _canonical(fact.value) in wanted or fact.kind == "cpu"]


def resolve_mounts(flags: list[str] | None) -> dict[str, str]:
    """The agent's mounts: the saved ones updated by *flags*, saved back when
    they changed. ``ValueError`` (bilingual) for a malformed flag, a path a
    flag sets that is not a directory, or a configuration that cannot be
    saved. A SAVED path that is not a directory (a share not mounted yet)
    is only warned about: the agent starts without offering that label and
    offers it once the directory appears (``Runner.available_mounts``)."""
    from . import agent_config

    config = agent_config.load(user_data_dir())
    saved = agent_config.saved_mounts(config)
    try:
        mounts = agent_config.merge_mounts(saved, flags)
    except ValueError as exc:
        raise ValueError(t(f"--mount 格式錯誤：{exc}", str(exc))) from None
    flagged = {label: path for label, path in mounts.items()
               if saved.get(label) != path}
    missing = agent_config.missing_dirs(flagged)
    if missing:
        raise ValueError(t(
            f"掛載路徑不是資料夾：{', '.join(missing)}（先掛上共用儲存，或用 "
            "--mount LABEL= 移除）",
            f"mount path is not a directory: {', '.join(missing)} (mount "
            "the shared storage first, or remove it with --mount LABEL=)"))
    waiting = agent_config.missing_dirs(mounts)
    if waiting:
        _err(f"警告：已儲存的掛載路徑目前不是資料夾：{', '.join(waiting)}；"
             "agent 會在它出現後才提供這個 label",
             f"warning: saved mount path is not a directory now: "
             f"{', '.join(waiting)}; the agent offers that label once it "
             "appears")
    if mounts != saved:
        try:
            agent_config.save(user_data_dir(), {**config, "mounts": mounts})
        except OSError as exc:
            raise ValueError(t(
                f"無法儲存 agent 設定 {agent_config.config_path(user_data_dir())}：{exc}",
                f"could not save the agent configuration "
                f"{agent_config.config_path(user_data_dir())}: {exc}")) from None
    return mounts


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


def _report_running(pid: int) -> int:
    _say(f"agent 已在執行（PID {pid}）。用 cdui agent stop 停止它。",
         f"The agent is already running (PID {pid}). Stop it with "
         "cdui agent stop.")
    return 0


def _start_background(args: argparse.Namespace, control: Any,
                      token: Any) -> int:
    existing = running_agent_pid()
    if existing is not None:
        return _report_running(existing)
    argv = [sys.executable, "-m", "app.agent", "start", "--foreground",
            DAEMON_FLAG]
    env = dict(os.environ)
    if isinstance(control, str):
        argv += ["--control", control]
    if getattr(args, "uds", None):
        argv += ["--uds", args.uds]
    if getattr(args, "ca_fingerprint", None):
        argv += ["--ca-fingerprint", args.ca_fingerprint]
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
              devices: list[Any], token_source: str | None = None,
              mounts: dict[str, str] | None = None,
              trust: Any = None) -> int:
    """The agent itself: connect and serve until stopped."""
    from ..config import settings
    from ..core import worker_protocol as wp
    from ..core.version import get_version
    from .connection import AgentConnection
    from .facts import gather as gather_facts
    from .runner import Runner

    name = args.name
    wanted = parse_devices(args.devices)
    offered = {report.value for report in devices}
    runner = Runner(runs_dir=runs_dir(), offered_devices=offered,
                    mounts=mounts,
                    keep_last=settings.RUN_RETENTION_KEEP_LAST)

    def probe() -> list[wp.DeviceReport]:
        return [_report(fact) for fact in offered_reports(
            _probe_facts(), wanted)]

    async def serve() -> int:
        stop = asyncio.Event()
        _install_stop_handlers(asyncio.get_running_loop(), stop)
        # Once per agent: a helper subprocess imports torch and discovers
        # the nodes, so this process never does (#781). A stop meanwhile
        # ends the helper and the agent at once.
        logger.info("collecting this machine's environment fingerprint and "
                    "node catalog")
        cancel = threading.Event()
        gathering = asyncio.ensure_future(
            asyncio.to_thread(gather_facts, cancel=cancel))
        stopping = asyncio.ensure_future(stop.wait())
        await asyncio.wait({gathering, stopping},
                           return_when=asyncio.FIRST_COMPLETED)
        stopping.cancel()
        if stop.is_set():
            cancel.set()
            await gathering
            return 0
        facts = gathering.result()
        connection = AgentConnection(
            runner=runner, name=name, version=get_version(),
            control_url=control, token=token, probe_devices=probe,
            max_message_bytes=settings.WS_MAX_MESSAGE_BYTES, facts=facts,
            token_source=token_source or (
                f"the session token file {token_file()}"
                if callable(token) else f"--token / ${TOKEN_ENV}"),
            trust=trust,
            on_trusted=(fingerprint_recorder(control)
                        if isinstance(control, str) else None),
            uds=getattr(args, "uds", None))
        return await connection.run(stop)

    logger.info("agent %s starting (pid %s); offering %s; mounts %s", name,
                os.getpid(), ", ".join(sorted(offered)),
                ", ".join(f"{k}={v}" for k, v in sorted(
                    (mounts or {}).items())) or "none")
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


# ── trust (#786) ──────────────────────────────────────────────────────────


def plain_http_to_another_machine(control: Any) -> bool:
    """*control* is an ``http://`` URL whose host is not this machine (a
    loopback URL is the same host, or an SSH tunnel's local end)."""
    from urllib.parse import urlsplit

    from ..core.auth import is_loopback_peer

    if not isinstance(control, str):
        return False
    parts = urlsplit(control.strip())
    host = (parts.hostname or "").lower()
    return (parts.scheme.lower() == "http" and host != "localhost"
            and not is_loopback_peer(host))


def warn_plain_http(control: str) -> None:
    _err(f"警告：{control} 是未加密的 HTTP：worker token 與每個指派中的 secret "
         "參數都以明文傳輸。請改用 https:// 控制面網址（TLS 反向代理）或 SSH "
         "通道。",
         f"Warning: {control} is plain HTTP: the worker token and the "
         "secrets in every assignment travel unencrypted. Use an https:// "
         "control URL (a TLS proxy) or an SSH tunnel.")


def retrust_hint(control: str) -> str:
    return t(
        f"若憑證是刻意更換的，先確認新指紋（Add worker 對話框，或在代理主機上用 "
        f"openssl 計算），再執行 cdui agent start --control {control} "
        f"--ca-fingerprint <新指紋>；或執行 cdui agent forget {control} "
        "刪掉已存憑證後重新加入",
        "If the certificate was replaced on purpose, check its new "
        "fingerprint (the Add worker dialog, or openssl on the proxy host) "
        f"and run cdui agent start --control {control} --ca-fingerprint "
        f"<new fingerprint>; or run cdui agent forget {control} to drop the "
        "stored credential and enroll again")


def build_trust(control: str, pin: str | None, stored: Any) -> Any:
    """The verifier for an ``https`` *control*: the user's pin
    (``--ca-fingerprint`` now, else the one an earlier start recorded),
    else the system CAs."""
    from .trust import ControlPlaneTrust

    source = "--ca-fingerprint"
    if pin is None and stored is not None and stored.ca_fingerprint:
        pin = stored.ca_fingerprint
        source = t(f"{stored.path} 記錄的 --ca-fingerprint",
                   f"the --ca-fingerprint recorded in {stored.path}")
    return ControlPlaneTrust(pin, source=source, hint=retrust_hint(control))


def fingerprint_recorder(control: str) -> Callable[[str], None]:
    """Stores the user's pin in *control*'s credential file when it differs
    from the one there (``record_fingerprint`` skips the write otherwise).
    Runs in a thread, once per verified connection."""
    from .enroll import record_fingerprint

    def record(fingerprint: str) -> None:
        path = record_fingerprint(user_data_dir(), control, fingerprint)
        if path is not None:
            logger.info("recorded the pinned certificate fingerprint %s in "
                        "%s", fingerprint, path)

    return record


def cmd_forget(args: argparse.Namespace) -> int:
    from .enroll import forget_credential

    try:
        removed = forget_credential(user_data_dir(), args.control)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if removed is None:
        _say(f"沒有 {args.control} 的已存憑證。",
             f"No credential is stored for {args.control}.")
        return 0
    _say(f"已刪除 {removed}。下次 cdui agent start --control {args.control} "
         "需要新的 enrollment token。",
         f"Deleted {removed}. The next cdui agent start --control "
         f"{args.control} needs a new enrollment token.")
    return 0


def use_stored_credential(args: argparse.Namespace, control: str,
                          explicit_name: str | None) -> Any:
    """The stored credential for *control*, applied to *args*, or None.

    Returns the credential; ``args.name`` becomes the name it was enrolled
    under. Raises ``ValueError`` when ``--name`` asks for another name.
    """
    from ..core.worker_store import is_enrollment_token
    from .enroll import load_credential

    stored = load_credential(user_data_dir(), control)
    if stored is None:
        return None
    if explicit_name and validate_name(explicit_name) != stored.worker_id:
        raise ValueError(t(
            f"{stored.path} 存的是 worker {stored.worker_id!r} 的憑證，不是 "
            f"{explicit_name!r}；要換名稱，先執行 cdui agent forget {control}"
            "，再用新的 enrollment token 加入",
            f"the credential in {stored.path} is for worker "
            f"{stored.worker_id!r}, not {explicit_name!r}; to change the name, "
            f"run cdui agent forget {control} and enroll with a new token"))
    args.name = stored.worker_id
    if not args.daemonized:
        _say(f"使用已存的 worker 憑證（{stored.path}）。",
             f"Using the stored worker credential ({stored.path}).")
        if args.token and is_enrollment_token(args.token):
            _say(f"未使用 enrollment token；要重新加入，先執行 cdui agent "
                 f"forget {control}。",
                 "The enrollment token was not used; to enroll again, run "
                 f"cdui agent forget {control} first.")
    return stored


def enroll_now(args: argparse.Namespace, control: str, token: str,
               facts: list[Any], trust: Any = None) -> str:
    """Trade the enrollment *token* for a worker token and store it, with
    the user's pin when *trust* has one (``https``). Returns the worker
    token. Raises ``EnrollError``."""
    from .connection import platform_facts
    from .enroll import enroll, save_credential

    _say(f"正在以 worker {args.name!r} 加入 {control}…",
         f"Enrolling as worker {args.name!r} with {control}...")
    worker_id, worker_token = enroll(
        control, token, name=args.name, platform=platform_facts(),
        devices=[_report(fact).model_dump() for fact in facts], trust=trust,
        uds=getattr(args, "uds", None))
    pin = trust.expected if trust is not None else None
    path = save_credential(user_data_dir(), control, worker_id, worker_token,
                           ca_fingerprint=pin)
    args.name = worker_id
    _say(f"已加入；worker 憑證存於 {path}（之後重啟會沿用）。",
         f"Enrolled; the worker credential is stored in {path} (restarts "
         "reuse it).")
    return worker_token


def cmd_start(args: argparse.Namespace) -> int:
    # NVML numbers cards in PCI bus order; with this, CUDA does too, in this
    # process and in every run subprocess (``core.resources``).
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # A running agent read its mounts when it started: changing them now
    # would save a configuration it does not use. Refused, nothing saved.
    if args.mount and not args.daemonized \
            and running_agent_pid() is not None:
        print(t("錯誤：agent 執行中時不能變更 --mount；請先執行 cdui agent stop。",
                "Error: --mount cannot change while the agent is running; "
                "stop the agent first (cdui agent stop)."), file=sys.stderr)
        return 1
    explicit_name = args.name
    from ..core import tls_fingerprint
    from .trust import is_https

    try:
        pin = (tls_fingerprint.normalize(args.ca_fingerprint)
               if args.ca_fingerprint else None)
        args.name = validate_name(args.name or default_name())
        wanted = parse_devices(args.devices)
        facts = offered_reports(_probe_facts(), wanted)
        # Saved here, so the background process started below reads them.
        mounts = resolve_mounts(args.mount)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not args.foreground and not args.daemonized:
        # Before anything is asked or enrolled: a single-use token spent on
        # a start that then finds an agent running would be lost.
        existing = running_agent_pid()
        if existing is not None:
            return _report_running(existing)
    uds = getattr(args, "uds", None)
    if uds and not (args.control and args.control.strip().lower()
                    .startswith("http://")):
        _err("--uds 需要 http:// 的 --control 網址。",
             "--uds needs an http:// --control URL.")
        return 2
    interactive = _interactive() and not args.daemonized
    control = resolve_control(args.control, interactive=interactive)
    stored = None
    if isinstance(control, str):
        try:
            stored = use_stored_credential(args, control, explicit_name)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if pin is not None and not (isinstance(control, str)
                                and is_https(control)):
        _err("--ca-fingerprint 需要 https:// 控制面網址。",
             "--ca-fingerprint needs an https:// control URL.")
        return 2
    if plain_http_to_another_machine(control) and not args.daemonized \
            and not uds:  # ssh carries the socket across the network
        warn_plain_http(control)
    trust = (build_trust(control, pin, stored)
             if isinstance(control, str) and is_https(control) else None)
    token_source = None
    if stored is not None:
        token: Any = stored.worker_token
        token_source = (f"the stored credential {stored.path} (cdui agent "
                        f"forget {control} to enroll again)")
    else:
        token = resolve_token(args.token, interactive=interactive)
    from ..core.worker_store import is_enrollment_token

    if isinstance(token, str) and is_enrollment_token(token):
        if args.daemonized:
            print(t("背景 agent 收到 enrollment token；請在前景重新執行 "
                    "cdui agent start。",
                    "The background agent was given an enrollment token; run "
                    "cdui agent start again in a terminal."), file=sys.stderr)
            return 2
        from .enroll import EnrollError

        url = control() if callable(control) else control
        try:
            token = enroll_now(args, url, token, facts, trust)
        except (EnrollError, ValueError) as exc:
            _err(f"加入失敗：{exc}", f"Enrollment failed: {exc}")
            return 1
        control = url
        token_source = (f"the stored credential (cdui agent forget {url} to "
                        "enroll again)")
    if not args.daemonized:
        try:
            require_token(token)
        except MissingToken as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if not args.foreground:
        return _start_background(args, control, token)
    _setup_logging(args.daemonized)
    return run_agent(args, control, token, facts, token_source=token_source,
                     mounts=mounts, trust=trust)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass
    args = build_parser().parse_args(argv)
    if args.command == "stop":
        return cmd_stop(args)
    if args.command == "forget":
        return cmd_forget(args)
    return cmd_start(args)
