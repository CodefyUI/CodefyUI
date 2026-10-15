"""SSH workers (#784): the control plane starts a worker agent on
``user@host`` and the agent connects back through ``ssh -R``.

For a laptop behind NAT and a GPU server that only opens SSH: neither end
opens a port. This process runs, per ``ssh_workers`` row::

    ssh -T -o BatchMode=yes -o ExitOnForwardFailure=yes
        -o ServerAliveInterval=15 -o ServerAliveCountMax=3
        -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20
        [-p <port>] [-i <identity>]
        -R 127.0.0.1:<remote port>:127.0.0.1:<this server's port>
        -- <target> "sh -c '...cdui agent start --foreground
                      --control http://127.0.0.1:<remote port> --name <id>'"

and the agent on the remote host dials its own loopback, which the
reverse forward carries here. Facts that shape it:

- **The system ``ssh``, with the server account's own config**, so a
  ``Host`` alias, a ``ProxyJump`` and the account's keys and agent work as
  they do in a terminal. ``BatchMode=yes``: ssh never prompts (a key that
  needs a passphrase must be in an agent). ``StrictHostKeyChecking`` is
  ``accept-new`` unless ``CODEFYUI_SSH_STRICT_HOST_KEY_CHECKING=yes``.
- **The forward binds the remote host's loopback only** (``-R
  127.0.0.1:...``). Every user of that host can still reach the port, so a
  request arriving with that ``Host`` may reach ``/ws/worker`` and
  ``/api/workers/enroll`` only (``auth.tunnel_host_allows``).
- **No token on any command line.** Each launch mints a one-time
  enrollment token (10 minutes, one use; bound to the worker once it is
  enrolled, so it can only re-key that name) and writes it to ssh's
  **stdin**; the remote shell reads it into ``$CODEFYUI_AGENT_TOKEN``. The
  agent uses the worker credential it stored on the remote host when it
  has one for that address (``app.agent.enroll``, keyed by the tunnel's
  ``127.0.0.1:<remote port>``) and enrolls with the token otherwise. The
  remote port is stored with the row, so a restart reuses the address and
  the credential stored for it. A worker token never leaves either host.
- **Non-interactive shells have a minimal PATH** (#809): the launcher is
  called by path (``remote_cdui``, default ``~/CodefyUI/cdui``, where
  ``install.sh`` puts it) and ``~/.local/bin`` / ``~/.cargo/bin`` are put
  on PATH first.
- **The agent outlives a dropped session, for the resume grace.** ssh's
  stdin stays open; a watcher on the remote host reads it. A ``stop``
  line (Disable, Remove, a shutdown, a restart for a credential) sends
  the agent SIGTERM at once. End of input without one is a dropped
  session (a laptop asleep, a network gone): the agent keeps running, its
  runs keep going and spool (#783), and only after
  ``CODEFYUI_WORKER_RESUME_GRACE_S`` with no newer session attached does
  it get SIGTERM. The next launch re-creates the tunnel on the **same**
  remote port and the remote script finds the agent alive (its pid in
  ``~/.codefyui-ssh/<port>/pid``): it starts no second one, it re-attaches,
  and the agent reconnects through the new tunnel by itself. When the
  remote sshd has not yet noticed the old session is gone and still holds
  the port, the launch ends that stale session first
  (:func:`remote_reclaim_command`). Disable with no session attached
  stops the agent with a one-shot ssh (:func:`remote_stop_command`).
- **Retries with backoff** (2 s doubling to 60 s, reset after a minute
  up). A forward failure (the remote port is taken) moves to the next port
  of ``CODEFYUI_SSH_REMOTE_PORT_RANGE``; a server that refuses forwarding
  itself (``AllowTcpForwarding`` / ``PermitListen``: an explicit refusal,
  or :data:`FORWARD_GIVE_UP` ports in a row) is a final failure with that
  reason, until Enable. The last lines ssh and the agent wrote, and the
  remote agent's pid, are kept and shown in the Workers panel.
- **Stored credentials that do not fit.** The first launch on a port this
  entry has not used before runs ``cdui agent forget`` first, so a
  credential another worker (or another control plane) left for that
  address is never presented. When it happens anyway -- this server
  refuses the credential on that tunnel (``note_refused``, called by
  ``ws_worker``) or the agent says its credential is for another worker --
  the session is restarted once with ``cdui agent forget`` first.
- **One action at a time per entry**: add, enable, disable and delete hold
  the entry's lock, so an enable that waits behind a disable ends with the
  tunnel running and a delete is never undone by an enable racing it.

Same-host detection is unchanged: the tunnel makes the agent a loopback
peer, but ``same_host`` also needs the agent's machine id to equal this
server's (``ws_worker``), and a remote host's does not.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..config import settings
from . import auth
from .worker_store import SshWorker

logger = logging.getLogger(__name__)

DEFAULT_REMOTE_CDUI = "~/CodefyUI/cdui"
DEFAULT_PORT_RANGE = (18000, 18100)

STATE_CONNECTING = "connecting"
STATE_UP = "up"
STATE_FAILED = "failed"
STATE_STOPPED = "stopped"

#: Lines of ssh / agent output kept per entry.
TAIL_LINES = 20
#: Lines of that tail that make up ``last_error``.
ERROR_LINES = 6
BACKOFF_MIN_S = 2.0
BACKOFF_MAX_S = 60.0
#: Up this long, and the next failure starts the backoff over.
STABLE_S = 60.0
#: How often a running launch checks whether its worker is online.
POLL_S = 1.0
#: The one-time enrollment token each launch sends.
ENROLL_TTL_S = 600
#: A stop waits this long for the agent to wind down after its stdin
#: closes, then terminates ssh, then kills it.
STOP_GRACE_S = 10.0
KILL_GRACE_S = 5.0
SERVER_ALIVE_INTERVAL_S = 15
CONNECT_TIMEOUT_S = 20

#: ssh's words when ``ExitOnForwardFailure`` ends a session.
FORWARD_FAILED = "remote port forwarding failed"
#: An sshd that does not allow the forward at all (``AllowTcpForwarding``,
#: ``PermitListen``, ``DisableForwarding``) words it one of these ways.
FORWARD_FORBIDDEN = ("administratively prohibited", "forwarding disabled",
                     "not permitted", "permitlisten", "allowtcpforwarding")
#: Forward failures on this many ports in a row: forwarding is not allowed.
FORWARD_GIVE_UP = 3
#: The agent's words when its stored credential names another worker
#: (``app.agent.cli.use_stored_credential``; the remote runs in English).
CREDENTIAL_MISMATCH = "is for worker"
#: The remote script's line naming the agent's pid.
REMOTE_PID = re.compile(r"codefyui-ssh: agent pid (\d+)")
#: The remote shell's exit status when no token arrived on stdin.
NO_TOKEN_EXIT = 97
#: ... and when it cannot create its state directory.
NO_STATE_EXIT = 98
#: The remote script's state, per tunnel port, under the remote $HOME.
REMOTE_STATE_DIR = ".codefyui-ssh"
#: How long a one-shot remote command (stop, reclaim) may take.
ONESHOT_TIMEOUT_S = 40.0

#: ``user@host``, ``host``, or an ssh ``Host`` alias; never an option.
TARGET_PATTERN = re.compile(
    r"(?:[A-Za-z0-9_][A-Za-z0-9._-]{0,63}@)?[A-Za-z0-9_][A-Za-z0-9._-]{0,252}")
#: A path or a command name the remote shell runs unquoted: no space, no
#: quote, no shell syntax. ``~/`` at the start means the remote $HOME.
REMOTE_CDUI_PATTERN = re.compile(r"(?:~/)?[A-Za-z0-9_.+/-]{1,255}")


class SshConfigError(ValueError):
    """An SSH worker setting that cannot be used; the message says why."""


# ── validation ────────────────────────────────────────────────────────────


def validate_target(target: str) -> str:
    target = (target or "").strip()
    if TARGET_PATTERN.fullmatch(target) is None:
        raise SshConfigError(
            f"SSH target {target!r} must be user@host, host, or a Host alias "
            "from your ssh config (letters, digits, '.', '_', '-')")
    return target


def validate_remote_cdui(path: str | None) -> str:
    path = (path or "").strip() or DEFAULT_REMOTE_CDUI
    if (REMOTE_CDUI_PATTERN.fullmatch(path) is None or path.startswith("-")
            or ".." in path.split("/")):
        raise SshConfigError(
            f"remote cdui path {path!r} must be a plain path such as "
            f"{DEFAULT_REMOTE_CDUI} or /opt/CodefyUI/cdui (no spaces, quotes "
            "or '..')")
    return path


def validate_identity(path: str | None) -> str | None:
    """An identity file on this machine: expanded, and it must exist."""
    path = (path or "").strip()
    if not path:
        return None
    resolved = Path(os.path.expanduser(path))
    if not resolved.is_absolute():
        raise SshConfigError(
            f"identity file {path!r} must be an absolute path (or start "
            "with ~/)")
    if not resolved.is_file():
        raise SshConfigError(f"identity file {resolved} does not exist on "
                             "the control-plane machine")
    return str(resolved)


def default_worker_name(target: str) -> str:
    """``alice@gpu1.lab.example`` -> ``gpu1`` (the agent's own default is
    the remote hostname's first label, too)."""
    host = target.rpartition("@")[2].split(".")[0].lower()
    name = re.sub(r"[^a-z0-9-]+", "-", host).strip("-")[:63]
    return name or "ssh-worker"


def parse_port_range(raw: str | None) -> tuple[int, int]:
    """``"18000-18100"`` -> ``(18000, 18100)``; the default when unusable."""
    match = re.fullmatch(r"\s*(\d{1,5})\s*-\s*(\d{1,5})\s*", raw or "")
    if match is not None:
        low, high = int(match.group(1)), int(match.group(2))
        if 1024 <= low <= high <= 65535:
            return low, high
    logger.warning("CODEFYUI_SSH_REMOTE_PORT_RANGE=%r is not low-high within "
                   "1024-65535; using %d-%d", raw, *DEFAULT_PORT_RANGE)
    return DEFAULT_PORT_RANGE


def strict_host_key_checking() -> str:
    value = (settings.SSH_STRICT_HOST_KEY_CHECKING or "").strip().lower()
    return "accept-new" if value == "accept-new" else "yes"


def ssh_executable() -> str | None:
    """The ``ssh`` on PATH (``ssh.exe`` on Windows), or None."""
    return shutil.which("ssh")


def local_forward_host() -> str:
    """Where the forward lands on this machine, which ssh connects to.

    A wildcard IPv4 or 127.x bind: ``127.0.0.1``. ``localhost`` and the
    IPv6 wildcard: ``localhost``, which ssh resolves and tries address by
    address (uvicorn may have bound ``localhost`` to ``::1``). ``::1``:
    ``[::1]``. Any other address is the only one the server listens on, so
    the forward goes there; such a peer is not loopback, and the tunnel
    ``Host`` is what marks it as tunnelled (``auth.tunnel_port``).
    """
    host = settings.HOST.strip().strip("[]").lower()
    if host in ("", "0.0.0.0") or host.startswith("127."):
        return "127.0.0.1"
    if host in ("localhost", "::"):
        return "localhost"
    return f"[{host}]" if ":" in host else host


# ── the command ───────────────────────────────────────────────────────────


def control_url(remote_port: int) -> str:
    return f"http://127.0.0.1:{remote_port}"


def _remote_cdui(entry: SshWorker) -> str:
    cdui = validate_remote_cdui(entry.remote_cdui)
    return '"$HOME"/' + cdui[2:] if cdui.startswith("~/") else cdui


def _state_dir(remote_port: int) -> str:
    """Where the remote script keeps the agent's pid, its log and the
    attached session's pid, per tunnel port (per control-plane address)."""
    return f'"$HOME"/{REMOTE_STATE_DIR}/{remote_port}'


def _sh(script: str) -> str:
    assert "'" not in script and "\\" not in script
    return f"sh -c '{script}'"


def resume_grace_s() -> int:
    """How long a remote agent outlives a dropped session: the control
    plane's own resume grace (#783), so its runs can still be resumed."""
    return max(1, int(settings.WORKER_RESUME_GRACE_S))


def remote_command(entry: SshWorker, remote_port: int, *,
                   forget_first: bool = False,
                   grace_s: int | None = None) -> str:
    """The command ssh runs on the remote host.

    One ``sh -c '...'`` whose script has no single quote and no backslash,
    so it reads the same to bash, zsh, fish and csh login shells. It reads
    the token from stdin, puts ``~/.local/bin`` on PATH and asks for
    English output (which this module reads). Then, in the per-port state
    directory (``~/.codefyui-ssh/<remote port>``):

    - it records itself as the attached session (``session``);
    - if the agent named in ``pid`` is alive, it **re-attaches**: no
      second agent; the agent reconnects through this session's tunnel by
      itself. Otherwise it starts one in the foreground, its output in
      ``log``, its pid in ``pid``. Either way it prints the pid and follows
      ``log`` on its stderr;
    - a watcher reads ssh's stdin: a ``stop`` line sends the agent SIGTERM
      at once (Disable, Remove, shutdown). End of input without ``stop`` is
      a dropped session: after *grace_s* the agent gets SIGTERM, unless a
      newer session has attached by then.
    """
    cdui = _remote_cdui(entry)
    url = control_url(remote_port)
    d = _state_dir(remote_port)
    grace = resume_grace_s() if grace_s is None else max(1, int(grace_s))
    forget = (f"{cdui} agent forget {url} </dev/null >/dev/null 2>&1; "
              if forget_first else "")
    script = (
        'IFS= read -r t; [ -n "$t" ] || exit ' + str(NO_TOKEN_EXIT) + "; "
        "CODEFYUI_AGENT_TOKEN=$t; export CODEFYUI_AGENT_TOKEN; t=; "
        'PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"; export PATH; '
        "CODEFYUI_LANG=en; export CODEFYUI_LANG; "
        f"d={d}; "
        # Alive and not a zombie (an exited child stays one until waited).
        'al() { ps -p "$1" -o stat= 2>/dev/null | grep -q "^[^Z]"; }; '
        'mkdir -p "$d" && chmod 700 "$d" || exit ' + str(NO_STATE_EXIT) + "; "
        'echo $$ > "$d/session"; '
        "exec 3<&0; "
        'a=$(cat "$d/pid" 2>/dev/null); o=; '
        'if [ -n "$a" ] && ps -p "$a" -o args= 2>/dev/null '
        '| grep -q "agent start"; then '
        "echo codefyui-ssh: agent pid $a reattached >&2; "
        'tail -n 5 -f "$d/log" >&2 & l=$!; '
        "else "
        + forget
        + ': > "$d/log"; '
        f"{cdui} agent start --foreground --control {url} --name {entry.id} "
        '</dev/null 3<&- >> "$d/log" 2>&1 & a=$!; o=1; '
        'echo $a > "$d/pid"; '
        "echo codefyui-ssh: agent pid $a >&2; "
        'tail -n +1 -f "$d/log" >&2 & l=$!; '
        "fi; "
        "( while IFS= read -r m; do "
        'if [ "$m" = stop ]; then kill -TERM $a; exit 0; fi; done; '
        f"sleep {grace}; "
        'if [ "$(cat "$d/session" 2>/dev/null)" = "$$" ]; then '
        "kill -TERM $a; exit 0; fi; exit 3 ) <&3 >/dev/null 2>&1 & w=$!; "
        "exec 3<&-; "
        "while al $a && al $w; do sleep 1; done; "
        "if al $w; then kill $w 2>/dev/null; ws=0; "
        "else wait $w; ws=$?; fi; "
        '[ "$ws" = 3 ] && { kill $l 2>/dev/null; exit 0; }; '
        "while al $a; do sleep 1; done; "
        "kill $l 2>/dev/null; s=0; "
        '[ -n "$o" ] && { wait $a; s=$?; }; '
        '[ "$(cat "$d/session" 2>/dev/null)" = "$$" ] '
        '&& rm -f "$d/pid"; '
        "exit $s"
    )
    return _sh(script)


def remote_stop_command(remote_port: int) -> str:
    """Stop the agent of *remote_port*'s state directory, if it is alive
    (Disable or Remove while no session is attached)."""
    d = _state_dir(remote_port)
    return _sh(
        f"d={d}; "
        'a=$(cat "$d/pid" 2>/dev/null); '
        'if [ -n "$a" ] && ps -p "$a" -o args= 2>/dev/null '
        '| grep -q "agent start"; then kill -TERM $a; '
        "echo codefyui-ssh: stopped agent pid $a; fi; exit 0")


def remote_reclaim_command(remote_port: int) -> str:
    """End the remote sshd of the session last attached for *remote_port*
    (one whose client is gone, but whose sshd has not noticed yet and
    still holds the forward's port), so the next launch can listen there
    again and the surviving agent reconnects through it."""
    d = _state_dir(remote_port)
    return _sh(
        f"d={d}; "
        'q=$(cat "$d/session" 2>/dev/null); '
        'while [ -n "$q" ] && [ "$q" -gt 1 ] 2>/dev/null; do '
        'c=$(ps -o args= -p "$q" 2>/dev/null); '
        'case "$c" in sshd:*) kill "$q"; '
        "echo codefyui-ssh: ended stale session $q; break;; esac; "
        'q=$(ps -o ppid= -p "$q" 2>/dev/null | tr -d " "); done; exit 0')


def ssh_base_argv(ssh: str, entry: SshWorker) -> list[str]:
    argv = [ssh, "-T",
            "-o", "BatchMode=yes",
            "-o", f"ServerAliveInterval={SERVER_ALIVE_INTERVAL_S}",
            "-o", "ServerAliveCountMax=3",
            "-o", f"StrictHostKeyChecking={strict_host_key_checking()}",
            "-o", f"ConnectTimeout={CONNECT_TIMEOUT_S}"]
    if entry.port != 22:
        # Left out at 22, so a Host alias's own ``Port`` still applies.
        argv += ["-p", str(entry.port)]
    if entry.identity:
        argv += ["-i", entry.identity]
    return argv


def ssh_argv(ssh: str, entry: SshWorker, remote_port: int, local_port: int,
             *, forget_first: bool = False,
             grace_s: int | None = None) -> list[str]:
    argv = ssh_base_argv(ssh, entry)
    argv[2:2] = ["-o", "ExitOnForwardFailure=yes"]
    argv += ["-R", f"127.0.0.1:{remote_port}:{local_forward_host()}:"
                   f"{local_port}",
             "--", validate_target(entry.target),
             remote_command(entry, remote_port, forget_first=forget_first,
                            grace_s=grace_s)]
    return argv


def ssh_oneshot_argv(ssh: str, entry: SshWorker, command: str) -> list[str]:
    """ssh without a forward, for one remote command."""
    return [*ssh_base_argv(ssh, entry), "--", validate_target(entry.target),
            command]


def next_backoff(current: float) -> float:
    return min(BACKOFF_MAX_S, max(BACKOFF_MIN_S, current * 2))


# ── one entry ─────────────────────────────────────────────────────────────


class SshLaunch:
    """One managed ``ssh`` subprocess for one ``ssh_workers`` row, with
    its restart loop. Start with :meth:`start`, end with :meth:`stop`."""

    def __init__(self, entry: SshWorker, launcher: SshLauncher) -> None:
        self.entry = entry
        self.launcher = launcher
        self.state = STATE_STOPPED
        self.last_error: str | None = None
        self.exit_code: int | None = None
        self.tail: collections.deque[str] = collections.deque(
            maxlen=TAIL_LINES)
        self.pid: int | None = None
        #: The agent's pid on the remote host, from the remote script.
        self.remote_pid: int | None = None
        self.attempts = 0
        #: Epoch seconds the next attempt starts, while waiting to retry.
        self.retry_at: float | None = None
        #: Epoch seconds the worker came online over this tunnel.
        self.up_since: float | None = None
        #: The loop stopped for good (forwarding is not allowed); Enable
        #: starts it again.
        self.gave_up = False
        self._forget_first = False
        self._refused = False
        self._mismatch = False
        self._forward_failures = 0
        #: The remote port a session last attached on (the remote script
        #: printed the agent pid): an agent may survive there.
        self._attached_port: int | None = None
        self._reclaimed = False
        #: The last session re-attached to an agent that survived a drop.
        self.reattached = False
        self._proc: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._enrollment_id: str | None = None
        self._port: int | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop = asyncio.Event()
        self.state = STATE_CONNECTING
        self.gave_up = False
        self._forward_failures = 0
        self._task = asyncio.get_running_loop().create_task(
            self._loop(), name=f"ssh-worker-{self.entry.id}")

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def stop(self) -> None:
        """End the ssh session: close its stdin (the remote agent winds
        down), then terminate, then kill. The loop does not restart it."""
        self._stop.set()
        proc = self._proc
        if proc is not None and proc.returncode is None:
            await _end_process(proc)
        elif self._attached_port is not None:
            # No session to send ``stop`` through: an agent that survived
            # a drop may still run there, waiting out its grace.
            await self._oneshot(remote_stop_command(self._attached_port),
                                "stop the remote agent")
        self._attached_port = None
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(self._task, KILL_GRACE_S)
            if not self._task.done():  # pragma: no cover - defensive
                self._task.cancel()
        self._task = None
        self._release_port()
        await self._drop_enrollment()
        self.state = STATE_STOPPED
        self.retry_at = None
        self.up_since = None
        self.pid = None
        self.remote_pid = None
        self.reattached = False

    def note_refused(self) -> None:
        """This server refused the credential the agent presented on this
        tunnel: restart it with ``cdui agent forget`` first."""
        self._refused = True

    # ── the loop ──────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        backoff = BACKOFF_MIN_S
        while not self._stop.is_set():
            self.attempts += 1
            self.retry_at = None
            started = time.monotonic()
            was_up = False
            try:
                was_up = await self._attempt()
            except Exception as exc:  # noqa: BLE001 -- shown, then retried
                logger.exception("SSH worker %s: launch failed", self.entry.id)
                self._fail(str(exc) or type(exc).__name__)
            if self._stop.is_set() or self.gave_up:
                break
            if was_up and time.monotonic() - started >= STABLE_S:
                backoff = BACKOFF_MIN_S
            if self._forget_first:
                # A credential to clear: no reason to wait long for that.
                backoff = BACKOFF_MIN_S
            self.retry_at = time.time() + backoff
            logger.info("SSH worker %s: retrying in %.0f s", self.entry.id,
                        backoff)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), backoff)
            backoff = next_backoff(backoff)

    def _fail(self, message: str, exit_code: int | None = None) -> None:
        self.state = STATE_FAILED
        self.exit_code = exit_code
        self.last_error = message
        self.up_since = None

    async def _attempt(self) -> bool:
        """One ssh session, start to exit. Returns whether the worker came
        online during it."""
        ssh = self.launcher.ssh_path()
        if ssh is None:
            self._fail("ssh was not found on this machine's PATH; install "
                       "an OpenSSH client (on Windows: the OpenSSH Client "
                       "optional feature)")
            return False
        port = await self._ensure_port()
        if port is None:
            low, high = self.launcher.port_range
            self._fail(f"no free remote port left in {low}-{high} "
                       "(CODEFYUI_SSH_REMOTE_PORT_RANGE)")
            return False
        token = await self._mint_token()
        if self._stop.is_set():
            return False
        argv = ssh_argv(ssh, self.entry, port, self.launcher.local_port(),
                        forget_first=self._forget_first)
        self.reattached = False
        self._forget_first = False
        self._refused = False
        self._mismatch = False
        self.remote_pid = None
        self.tail.clear()
        self.state = STATE_CONNECTING
        logger.info("SSH worker %s: ssh %s (remote port %d)", self.entry.id,
                    self.entry.target, port)
        kwargs: dict[str, Any] = {}
        if sys.platform != "win32":
            # Not in the server's process group, and no controlling tty: a
            # Ctrl-C at the server's terminal reaches the server, whose
            # shutdown closes this session in order, and nothing ssh starts
            # (a ProxyJump hop) can stop to prompt.
            kwargs["start_new_session"] = True
        proc = await asyncio.create_subprocess_exec(
            *argv, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, **kwargs)
        self._proc = proc
        self.pid = proc.pid
        assert proc.stdin is not None and proc.stdout is not None
        try:
            proc.stdin.write((token + "\n").encode("ascii"))
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass  # ssh is already gone; its output says why
        reader = asyncio.get_running_loop().create_task(
            self._read(proc.stdout))
        was_up = False
        try:
            while proc.returncode is None:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(proc.wait(), POLL_S)
                if proc.returncode is not None:
                    break
                if self._stop.is_set():
                    await _end_process(proc)
                    break
                if self._worker_online():
                    if self.state != STATE_UP:
                        self.state = STATE_UP
                        self.up_since = time.time()
                        self.last_error = None
                        self.exit_code = None
                        self._forward_failures = 0
                        logger.info("SSH worker %s is up", self.entry.id)
                        await self._drop_enrollment()
                    was_up = True
                elif self.state == STATE_UP:
                    self.state = STATE_CONNECTING
                    self.up_since = None
                if self._refused:
                    # The credential the agent stored for this address is
                    # one this server does not accept (another control
                    # plane's, or revoked): start over with a forget.
                    self._forget_first = True
                    logger.warning(
                        "SSH worker %s: this server refused the agent's "
                        "stored credential; restarting it with a fresh "
                        "enrollment", self.entry.id)
                    await _end_process(proc)
        finally:
            with contextlib.suppress(asyncio.TimeoutError, Exception):
                await asyncio.wait_for(reader, KILL_GRACE_S)
            self._proc = None
            self.pid = None
        if self._stop.is_set():
            return was_up
        code = proc.returncode
        tail = [line for line in self.tail if line.strip()]
        if self._mismatch:
            self._forget_first = True
        if code == NO_TOKEN_EXIT:
            message = "the remote shell got no token on stdin"
        else:
            message = "\n".join(tail[-ERROR_LINES:]) or (
                f"ssh exited with status {code}")
        lowered = " ".join(tail).lower()
        if any(word in lowered for word in FORWARD_FORBIDDEN):
            self._give_up(
                f"{self.entry.target} does not allow remote port forwarding "
                "(sshd AllowTcpForwarding / PermitListen); ask its admin to "
                f"allow -R to 127.0.0.1, then press Enable.\n{message}", code)
            return was_up
        if (any(FORWARD_FAILED in line for line in tail) and not was_up
                and port == self._attached_port and not self._reclaimed):
            # Our own last session's sshd may still hold the port (it has
            # not noticed the drop): end it, and listen there again, where
            # the surviving agent dials.
            self._reclaimed = True
            await self._oneshot(remote_reclaim_command(port),
                                "end the stale session")
            self._fail(message, code)
            return was_up
        if any(FORWARD_FAILED in line for line in tail) and not was_up:
            self._forward_failures += 1
            if self._forward_failures >= FORWARD_GIVE_UP:
                self._give_up(
                    f"remote port forwarding failed on {FORWARD_GIVE_UP} "
                    f"ports in a row on {self.entry.target}: its sshd "
                    "probably does not allow it (AllowTcpForwarding / "
                    "PermitListen), or the ports are all taken (change "
                    "CODEFYUI_SSH_REMOTE_PORT_RANGE). Press Enable to try "
                    f"again.\n{message}", code)
                return was_up
            await self._move_port(port)
        else:
            self._forward_failures = 0
        self._fail(message, code)
        logger.warning("SSH worker %s: ssh exited with status %s: %s",
                       self.entry.id, code, message.replace("\n", " | "))
        return was_up

    def _give_up(self, message: str, code: int | None) -> None:
        self.gave_up = True
        self.retry_at = None
        self._fail(message, code)
        logger.warning("SSH worker %s: not retrying: %s", self.entry.id,
                       message.replace("\n", " | "))

    async def _read(self, stream: asyncio.StreamReader) -> None:
        while True:
            try:
                raw = await stream.readline()
            except (ValueError, asyncio.LimitOverrunError):
                continue  # an over-long line; skip it
            if not raw:
                return
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            self.tail.append(line)
            pid = REMOTE_PID.search(line)
            if pid is not None:
                self.remote_pid = int(pid.group(1))
                self.reattached = "reattached" in line
                self._attached_port = self._port
                self._reclaimed = False
            if CREDENTIAL_MISMATCH in line:
                self._mismatch = True
            logger.debug("ssh %s: %s", self.entry.id, line)

    def _worker_online(self) -> bool:
        worker = self.launcher.registry.get(self.entry.id)
        return worker is not None and bool(worker.online)

    # ── ports and tokens ──────────────────────────────────────────────────

    async def _ensure_port(self) -> int | None:
        port = self.entry.remote_port
        if port is None or not self.launcher.port_usable(port, self.entry.id):
            port = self.launcher.free_port(self.entry.id, after=port)
            if port is None:
                return None
            await self._save_port(port)
        self._hold_port(port)
        return port

    async def _move_port(self, failed: int) -> None:
        port = self.launcher.free_port(self.entry.id, after=failed)
        if port is None or port == failed:
            return
        logger.warning("SSH worker %s: remote port %d is taken on %s; "
                       "trying %d", self.entry.id, failed, self.entry.target,
                       port)
        await self._save_port(port)

    async def _save_port(self, port: int) -> None:
        """Take *port* for this entry. A port this entry has not used may
        hold another worker's credential on the remote host (or another
        control plane's): the first launch on it forgets that first."""
        self.entry = replace(self.entry, remote_port=port)
        self.launcher.entries[self.entry.id] = self.entry
        self._forget_first = True
        await self.launcher.store.update_ssh_worker(self.entry.id,
                                                    remote_port=port)

    def _hold_port(self, port: int) -> None:
        if self._port is not None and self._port != port:
            auth.forget_tunnel_port(self._port)
        self._port = port
        auth.allow_tunnel_port(port)

    def _release_port(self) -> None:
        if self._port is not None:
            auth.forget_tunnel_port(self._port)
            self._port = None

    @property
    def tunnel_port(self) -> int | None:
        """The remote port this launch holds, while it holds one."""
        return self._port

    async def _mint_token(self) -> str:
        """A one-time enrollment token for this launch (the previous
        launch's, if unspent, is deleted). Bound to the worker once it is
        enrolled, so it can re-key that name and nothing else."""
        await self._drop_enrollment()
        store = self.launcher.store
        bound = self.entry.id if await store.is_enrolled(
            self.entry.id) else None
        created = await store.create_enrollment(
            ttl_s=ENROLL_TTL_S, uses=1, label=f"ssh {self.entry.target}",
            worker=bound)
        self._enrollment_id = created["id"]
        return created["token"]

    async def _drop_enrollment(self) -> None:
        if self._enrollment_id is None:
            return
        enrollment_id, self._enrollment_id = self._enrollment_id, None
        try:
            await self.launcher.store.delete_enrollment(enrollment_id)
        except Exception:  # noqa: BLE001 -- it expires by itself anyway
            logger.warning("could not delete enrollment token %s",
                           enrollment_id, exc_info=True)

    # ── the panel's view ──────────────────────────────────────────────────

    def describe(self) -> dict[str, Any]:
        entry = self.entry
        return {
            "id": entry.id,
            "target": entry.target,
            "port": entry.port,
            "identity": entry.identity,
            "remote_cdui": entry.remote_cdui,
            "remote_port": entry.remote_port,
            "enabled": entry.enabled,
            "state": self.state if entry.enabled else STATE_STOPPED,
            "last_error": self.last_error,
            "exit_code": self.exit_code,
            "tail": list(self.tail),
            "attempts": self.attempts,
            "retry_at": self.retry_at,
            "up_since": self.up_since,
            "gave_up": self.gave_up,
            "pid": self.pid,
            "remote_pid": self.remote_pid,
            "reattached": self.reattached,
        }

    async def _oneshot(self, command: str, what: str) -> None:
        """Run *command* on the remote host over its own short ssh (no
        forward). Best effort: a failure is logged and kept in the tail."""
        ssh = self.launcher.ssh_path()
        if ssh is None:
            return
        kwargs: dict[str, Any] = {}
        if sys.platform != "win32":
            kwargs["start_new_session"] = True
        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_oneshot_argv(ssh, self.entry, command),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT, **kwargs)
            out, _ = await asyncio.wait_for(proc.communicate(),
                                            ONESHOT_TIMEOUT_S)
        except (OSError, asyncio.TimeoutError) as exc:
            logger.warning("SSH worker %s: could not %s: %s", self.entry.id,
                           what, exc)
            return
        for line in out.decode("utf-8", errors="replace").splitlines():
            if line.strip():
                self.tail.append(line.rstrip())
        logger.info("SSH worker %s: %s (ssh exited with status %s)",
                    self.entry.id, what, proc.returncode)


async def _end_process(proc: asyncio.subprocess.Process) -> None:
    """Send ``stop`` (the remote agent ends now, not after the resume
    grace) and close stdin, wait; terminate, wait; kill."""
    if proc.stdin is not None:
        with contextlib.suppress(Exception):
            proc.stdin.write(b"stop\n")
            await proc.stdin.drain()
        with contextlib.suppress(Exception):
            proc.stdin.close()
    for step, grace in (("eof", STOP_GRACE_S), ("terminate", KILL_GRACE_S),
                        ("kill", KILL_GRACE_S)):
        if step == "terminate":
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
        elif step == "kill":
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        try:
            await asyncio.wait_for(proc.wait(), grace)
            return
        except asyncio.TimeoutError:
            continue


# ── all entries ───────────────────────────────────────────────────────────


class SshLauncher:
    """Every SSH worker of this control plane: the rows (``worker_store``)
    and one :class:`SshLaunch` per enabled row. Add, enable, disable and
    delete of one entry run one at a time (:meth:`_lock`)."""

    def __init__(self, store: Any, registry: Any, *,
                 ssh: str | None = None,
                 port_range: tuple[int, int] | None = None,
                 local_port: int | None = None) -> None:
        self.store = store
        self.registry = registry
        self._ssh = ssh
        self.port_range = port_range or parse_port_range(
            settings.SSH_REMOTE_PORT_RANGE)
        self._local_port = local_port
        self.entries: dict[str, SshWorker] = {}
        self.launches: dict[str, SshLaunch] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def ssh_path(self) -> str | None:
        return self._ssh or ssh_executable()

    def local_port(self) -> int:
        return self._local_port or settings.PORT

    def _lock(self, name: str) -> asyncio.Lock:
        lock = self._locks.get(name)
        if lock is None:
            lock = self._locks[name] = asyncio.Lock()
        return lock

    # ── ports ─────────────────────────────────────────────────────────────

    def _ports_taken(self, exclude: str) -> set[int]:
        return {entry.remote_port for name, entry in self.entries.items()
                if name != exclude and entry.remote_port is not None}

    def port_usable(self, port: int, owner: str) -> bool:
        low, high = self.port_range
        return (low <= port <= high and port != self.local_port()
                and port not in self._ports_taken(owner))

    def free_port(self, owner: str, after: int | None = None) -> int | None:
        """The first usable port after *after* (wrapping), or None."""
        low, high = self.port_range
        span = list(range(low, high + 1))
        if after is not None and low <= after <= high:
            start = span.index(after) + 1
            span = span[start:] + span[:start]
        for port in span:
            if self.port_usable(port, owner):
                return port
        return None

    # ── events from the socket side ──────────────────────────────────────

    def note_refused(self, tunnel_port: int) -> None:
        """``ws_worker`` refused the credential an agent presented through
        the tunnel on *tunnel_port* (an unknown or revoked worker token)."""
        for launch in self.launches.values():
            if launch.tunnel_port == tunnel_port:
                launch.note_refused()

    def worker_connected(self, name: str) -> None:
        """A worker connected: one whose SSH row is disabled stays
        drained, whoever started it."""
        entry = self.entries.get(name)
        if entry is not None and not entry.enabled:
            self.registry.drain(name, True)

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def start_all(self) -> None:
        """Load the rows and start every enabled one. A disabled row's
        worker is kept draining."""
        for entry in await self.store.list_ssh_workers():
            self.entries[entry.id] = entry
            if entry.enabled:
                self._launch(entry)
            else:
                self.registry.drain(entry.id, True)
        if self.launches:
            logger.info("starting %d SSH worker(s): %s", len(self.launches),
                        ", ".join(sorted(self.launches)))

    def _launch(self, entry: SshWorker) -> SshLaunch:
        launch = self.launches.get(entry.id)
        if launch is None:
            launch = SshLaunch(entry, self)
            self.launches[entry.id] = launch
        launch.entry = entry
        launch.start()
        return launch

    async def add(self, entry: SshWorker) -> dict[str, Any]:
        """Store *entry* and start it. Raises ``worker_store.NameTaken``."""
        async with self._lock(entry.id):
            stored = await self.store.add_ssh_worker(entry)
            self.entries[stored.id] = stored
            if stored.enabled:
                self._launch(stored)
            return self.describe(stored.id)

    async def set_enabled(self, name: str,
                          enabled: bool) -> dict[str, Any] | None:
        """Enable (undrain, start; also restarts one that gave up) or
        disable (drain, then end the ssh session) *name*. None when there
        is no such row, also when it was deleted while this waited."""
        async with self._lock(name):
            entry = self.entries.get(name)
            if entry is None:
                return None
            if entry.enabled != enabled:
                await self.store.update_ssh_worker(name, enabled=enabled)
                entry = replace(entry, enabled=enabled)
                self.entries[name] = entry
            if enabled:
                self.registry.drain(name, False)
                self._launch(entry)
            else:
                # Drain first, so nothing new is placed on it while it goes.
                self.registry.drain(name, True)
                launch = self.launches.get(name)
                if launch is not None:
                    launch.entry = entry
                    await launch.stop()
            return self.describe(name)

    async def remove(self, name: str) -> bool:
        """Stop *name* and delete its row. The worker's enrollment is left
        to ``DELETE /api/workers/{id}``."""
        async with self._lock(name):
            launch = self.launches.pop(name, None)
            if launch is not None:
                await launch.stop()
            self.entries.pop(name, None)
            return await self.store.delete_ssh_worker(name)

    async def shutdown(self) -> None:
        launches = list(self.launches.values())
        if launches:
            logger.info("closing %d SSH worker session(s)", len(launches))
        await asyncio.gather(*(launch.stop() for launch in launches),
                             return_exceptions=True)

    # ── the panel's view ──────────────────────────────────────────────────

    def describe(self, name: str) -> dict[str, Any]:
        launch = self.launches.get(name)
        entry = self.entries[name]
        if launch is None:
            launch = SshLaunch(entry, self)
        else:
            launch.entry = entry
        return launch.describe()

    def describe_all(self) -> list[dict[str, Any]]:
        return [self.describe(name) for name in sorted(self.entries)]
