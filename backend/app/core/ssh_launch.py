"""SSH workers (#784): the control plane starts a worker agent on
``user@host`` and the agent connects back through ``ssh -R``.

For a laptop behind NAT and a GPU server that only opens SSH: neither end
opens a port. This process runs, per ``ssh_workers`` row::

    ssh -T -o BatchMode=yes -o ExitOnForwardFailure=yes
        -o ServerAliveInterval=15 -o ServerAliveCountMax=3
        -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20
        [-p <port>] [-i <identity>]
        -R <remote $HOME>/.codefyui-ssh/<cp>-<slot>/sock:<listener socket>
        -- <target> "sh -c '...cdui agent start --uds <that sock>
                      --foreground --control http://cp-<cp>.invalid:<slot>
                      --name <id>'"

``<cp>`` is this control plane's instance id (:func:`load_instance_id`,
eight hex digits kept in its user-data directory), so two control planes that
use the same account on one GPU host never share a directory, a socket,
a stored credential or an agent.

and the agent on the remote host dials that Unix socket, which the reverse
forward carries to this server. Facts that shape it:

- **The system ``ssh``, with the server account's own config**, so a
  ``Host`` alias, a ``ProxyJump`` and the account's keys and agent work as
  they do in a terminal. ``BatchMode=yes``: ssh never prompts (a key that
  needs a passphrase must be in an agent). ``StrictHostKeyChecking`` is
  ``accept-new`` unless ``CODEFYUI_SSH_STRICT_HOST_KEY_CHECKING=yes``.
- **Only this account on the remote host can reach the forward.** It is
  a Unix-domain socket (OpenSSH >= 6.7, sshd ``AllowStreamLocalForwarding``,
  on by default) in ``~/.codefyui-ssh/<cp>-<slot>/``, a directory of mode 0700:
  another user of a shared GPU host can neither connect to it nor bind it
  first and catch the agent's token, as a loopback TCP port would allow.
  Before each session a one-shot ssh (:func:`remote_prepare_command`)
  creates that directory, removes a socket a dropped session left behind
  (sshd's ``StreamLocalBindUnlink`` is off by default, and the client's
  option of that name does not reach a ``-R`` forward) and reports the
  absolute path, which ``-R`` needs.
- **The forward lands on a listener of its own** (``core.tunnel_listener``),
  opened per SSH worker, never on the server's port: a Unix socket in a
  private (0700) temporary directory on POSIX, so other users of this
  machine cannot reach it either (``-R`` forwards socket to socket), and an
  ephemeral port on 127.0.0.1 on Windows. It serves the agent routes only,
  each with a worker (or enrollment) token, and never the session token;
  nothing a client sends, ``Host`` included, makes a request on the main
  port count as tunnelled.
- **No token on any command line.** Each launch mints a one-time
  enrollment token (10 minutes, one use; bound to the worker once it is
  enrolled, so it can only re-key that name) and writes it to ssh's
  **stdin**; the remote shell reads it into ``$CODEFYUI_AGENT_TOKEN``. The
  agent uses the worker credential it stored on the remote host when it
  has one for that address (``app.agent.enroll``, keyed by the control URL
  ``http://cp-<cp>.invalid:<slot>``, which names the tunnel and resolves
  nowhere; the agent dials the socket) and enrolls with the token
  otherwise. The
  slot (``remote_port``, from ``CODEFYUI_SSH_REMOTE_PORT_RANGE``) is stored
  with the row, so a restart reuses the state directory and the credential
  stored for it. A worker token never leaves either host.
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
  it get SIGTERM. The next launch re-creates the forward at the **same**
  socket path and the remote script finds the agent alive (its pid in
  ``~/.codefyui-ssh/<cp>-<slot>/pid``, checked against the agent's own
  command line so a reused pid is never taken for it): it starts no second
  one, it re-attaches,
  and the agent reconnects through the new tunnel by itself. Disable with
  no session attached stops the agent with a one-shot ssh
  (:func:`remote_stop_command`).
- **An intentional stop lets the agent finish.** Disable, Remove and a
  server shutdown send ``stop``; the agent stops its runs (their grace),
  uploads what they wrote (a transfer run's interrupt checkpoint) and
  reports each ``run_finished`` before it exits, so ssh is given
  :data:`STOP_GRACE_S` before it is terminated (:data:`SHUTDOWN_GRACE_S`
  at a server shutdown, which stops the SSH workers before the run service
  so their reports are still recorded). A run still attached after that
  ends at once (``WorkerRegistry.stopped``), not after the resume grace:
  nothing is coming back for it.
- **Delete cleans up on the remote host; Disable does not.** Once the
  session has ended and the row is deleted, Delete answers and, in the
  background, runs one more one-shot ssh (:func:`remote_remove_command`)
  that forgets the credential the agent stored for this control plane's
  address and removes the slot's state directory, best effort and within
  :data:`REMOVE_TIMEOUT_S`; the slot stays reserved until it has finished,
  so no other entry takes it meanwhile. An unreachable host does not stop
  the Delete; what was left behind is logged. Disable keeps both, so
  Enable reconnects with the same credential.
- **The remote script leaves no process behind.** Whatever ends it (the
  agent exiting, ``stop``, a newer session taking over, a signal) ends its
  ``tail`` of the log and its grace timer too; once its output is in
  place it writes nothing more, so a session gone (its stderr a broken
  pipe) cannot kill it before it has (#829).
- **Retries with backoff** (2 s doubling to 60 s, reset after a minute
  up). A server that refuses the forward (``AllowStreamLocalForwarding`` /
  ``DisableForwarding``: an explicit refusal, or :data:`FORWARD_GIVE_UP`
  failed forwards in a row) is a final failure with that reason, until
  Enable. So is a failure that cannot heal by itself (#840,
  :func:`classify_failure`): the host key does not verify, or the session's
  remote shell cannot run ``cdui`` (exit 127 / 126). A refused login
  (``Permission denied``) or a host name that does not resolve can heal
  by itself (an ssh-agent or keychain unlocked after login, a VPN or Wi-Fi
  coming up), so those are retried with the usual backoff, the reason
  shown, and end the retries only once they have lasted
  :data:`LASTING_GIVE_UP_S` without a break. Retry (Enable on a running
  entry) tries again at once. The last lines ssh and the agent wrote
  (those of the last attempt only), and the remote agent's pid once that
  agent ran, are kept and shown in the Workers panel.
- **Stored credentials that do not fit.** The first launch on a slot this
  entry has not used before runs ``cdui agent forget`` first, so a
  credential another worker (or another control plane) left for that
  address is never presented. When it happens anyway -- the agent itself
  reports that this server refused its token, or that its credential is
  for another worker (both read from its log, which only that agent
  writes) -- the session is restarted once with ``cdui agent forget``
  first. A refusal ``ws_worker`` sees on the listener is not acted on: it
  cannot tell the agent from another local client.
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
from collections.abc import Awaitable, Callable
from typing import Any

from ..config import settings
from .tunnel_listener import TunnelListener
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
#: A stop waits this long for the agent to wind down after ``stop``, then
#: terminates ssh, then kills it. The agent gives each run its own 10 s
#: grace (``Runner.shutdown_grace_s``), terminates it, kills it 5 s later
#: (``KILL_AFTER_S``), then uploads what a transfer run wrote (its interrupt
#: checkpoint) and reports ``run_finished``: all of that must still find
#: the tunnel up. An agent with no run exits within a second or two.
STOP_GRACE_S = 60.0
#: The same, at a server shutdown: every SSH worker is told to stop at once
#: and given this long in all. ``cdui stop`` itself kills the server after
#: about 2 s; then the agents still stop (they got ``stop`` first), ssh
#: exits once they have, and the runs they could not report are settled
#: ``interrupted`` by the next start.
SHUTDOWN_GRACE_S = 25.0
KILL_GRACE_S = 5.0
SERVER_ALIVE_INTERVAL_S = 15
CONNECT_TIMEOUT_S = 20

#: ssh's words when ``ExitOnForwardFailure`` ends a session.
FORWARD_FAILED = "remote port forwarding failed"
#: An sshd that does not allow the forward at all
#: (``AllowStreamLocalForwarding``, ``DisableForwarding``) words it one of
#: these ways.
FORWARD_FORBIDDEN = ("administratively prohibited", "forwarding disabled",
                     "not permitted", "permitlisten", "allowtcpforwarding",
                     "allowstreamlocalforwarding")
#: Forward failures this many times in a row: forwarding is not allowed.
FORWARD_GIVE_UP = 3
#: A failure that may heal by itself (a refused login, a name that does
#: not resolve) ends the retries once it has lasted this long, without a
#: break (#840).
LASTING_GIVE_UP_S = 600.0
#: The remote shell's exit status when it cannot find, or cannot run, the
#: command (``cdui`` at the wrong path).
COMMAND_NOT_FOUND_EXIT = 127
COMMAND_NOT_EXECUTABLE_EXIT = 126
#: How a failure is treated (:func:`classify_failure`): ``final`` ends the
#: retries at once, ``lasting`` once it has lasted :data:`LASTING_GIVE_UP_S`.
FAILURE_FINAL = "final"
FAILURE_LASTING = "lasting"
#: ssh's words for failures a retry does not fix by itself, their kind, and
#: what to do about each.
FAILURE_WORDS = (
    ("host key verification failed", FAILURE_FINAL, "the host key does not "
     "verify (the host is not in known_hosts, or its key changed)"),
    ("permission denied (", FAILURE_LASTING, "the host refused the login "
     "(check the user name and that the key works without a prompt: ssh -o "
     "BatchMode=yes {target} true)"),
    ("could not resolve hostname", FAILURE_LASTING, "the host name does not "
     "resolve (check the target, the Host alias in ~/.ssh/config, or this "
     "machine's network)"),
)
#: The agent's words when its stored credential names another worker
#: (``app.agent.cli.use_stored_credential``; the remote runs in English).
CREDENTIAL_MISMATCH = "is for worker"
#: The agent's words when this server refused its token
#: (``app.agent.connection``, logged on each refused dial).
TOKEN_REFUSED = "the control plane refused the token"
#: The remote script's line naming the agent's pid.
REMOTE_PID = re.compile(r"codefyui-ssh: agent pid (\d+)")
#: The remote shell's exit status when no token arrived on stdin.
NO_TOKEN_EXIT = 97
#: ... and when it cannot create its state directory.
NO_STATE_EXIT = 98
#: The remote script's state, per slot, under the remote $HOME.
REMOTE_STATE_DIR = ".codefyui-ssh"
#: The forward's Unix socket, in that per-slot directory.
SOCKET_NAME = "sock"
#: The prepare command's line naming the (absolute) state directory.
REMOTE_DIR = re.compile(r"codefyui-ssh: dir (.*)$")
#: What a remote state directory may be spelled with: ``-R`` splits on
#: ``:``, and the remote commands are kept free of quoting surprises.
SAFE_REMOTE_DIR = re.compile(r"/[A-Za-z0-9._@+/-]+")
#: The file under this server's user-data directory holding its instance id.
INSTANCE_ID_FILE = "ssh-instance-id"
INSTANCE_ID_PATTERN = re.compile(r"[0-9a-f]{8}")
#: ``sun_path`` holds 108 bytes on Linux, 104 on macOS and the BSDs.
MAX_SOCKET_PATH = 100
#: How long a one-shot remote command (prepare, stop) may take.
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


# ── the command ───────────────────────────────────────────────────────────


def load_instance_id(user_data: Path | None = None) -> str:
    """This control plane's id among the ones that may share an account on
    a GPU host: eight hex digits, made once and kept in the user-data
    directory (``ssh-instance-id``)."""
    import secrets

    from ..config import _user_data_root

    path = (user_data or _user_data_root()) / INSTANCE_ID_FILE
    try:
        found = path.read_text(encoding="ascii").strip()
        if INSTANCE_ID_PATTERN.fullmatch(found):
            return found
    except (OSError, UnicodeDecodeError):
        pass
    made = secrets.token_hex(4)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(made + "\n", encoding="ascii")
    except OSError:
        logger.warning("could not keep the SSH instance id in %s; using %s "
                       "for this run", path, made, exc_info=True)
    return made


def control_url_legacy(remote_port: int) -> str:
    """The address an agent of an earlier build dialled over TCP."""
    return f"http://127.0.0.1:{remote_port}"


def control_url(cp: str, remote_port: int) -> str:
    """The address the agent stores its credential under. ``.invalid``
    never resolves: the agent reaches the server through ``--uds`` only."""
    return f"http://cp-{cp}.invalid:{remote_port}"


def _remote_cdui(entry: SshWorker) -> str:
    cdui = validate_remote_cdui(entry.remote_cdui)
    return '"$HOME"/' + cdui[2:] if cdui.startswith("~/") else cdui


def state_key(cp: str, remote_port: int) -> str:
    return f"{cp}-{remote_port}"


def _state_dir(cp: str, remote_port: int) -> str:
    """Where the remote script keeps the agent's pid, its log, the attached
    session's pid and the forward's socket: one directory per control plane
    and slot."""
    return f'"$HOME"/{REMOTE_STATE_DIR}/{state_key(cp, remote_port)}'


#: Shell functions the remote scripts share. ``al``: alive and not a zombie
#: (an exited child stays one until waited). ``own``: alive and the agent
#: of this state directory -- its command line names this directory's
#: socket (``/proc``), or at least ``agent start`` where there is no
#: ``/proc``. A pid file outlives its process, and the pid may be reused.
_FUNCS = (# ``/proc`` where there is one: BusyBox's ``ps`` (the applet its
          # ``sh`` runs) has no ``-p``. Else ``ps -p`` (macOS, the BSDs).
          'al() { [ -n "$1" ] || return 1; if [ -r "/proc/$1/status" ]; then '
          '! grep -q "^State:[[:space:]]*Z" "/proc/$1/status"; else '
          'ps -p "$1" -o stat= 2>/dev/null | grep -q "^[^Z]"; fi; }; '
          # ``has PID TEXT``: the process's command line contains TEXT.
          # ``/proc`` where there is one (Linux, BusyBox included), its
          # NUL-separated arguments joined with spaces first, as ``ps``
          # shows them (TEXT may span two: ``agent start``), by ``xargs
          # -0`` (``tr`` would need a backslash for NUL, and BusyBox's has
          # no ``[:cntrl:]``); else ``ps -ww`` (no truncation), else plain
          # ``ps`` as a best effort.
          'has() { if [ -r "/proc/$1/cmdline" ]; then '
          'xargs -0 < "/proc/$1/cmdline" 2>/dev/null | grep -qF -- "$2"; '
          "else "
          '{ ps -ww -p "$1" -o args= 2>/dev/null '
          '|| ps -p "$1" -o args= 2>/dev/null; } | grep -qF -- "$2"; fi; }; '
          'own() { [ -n "$1" ] && al "$1" && has "$1" '
          f'"$d/{SOCKET_NAME}"; '
          "}; ")


#: What each remote script that runs ``cdui`` starts with: a
#: non-interactive ssh shell reads no profile, so ``~/.local/bin`` (uv,
#: #809) is put on PATH first, and English output (which this module
#: reads) is asked for.
_PROLOGUE = ('PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"; export PATH; '
             "CODEFYUI_LANG=en; export CODEFYUI_LANG; ")


def _sh(script: str) -> str:
    assert "'" not in script and "\\" not in script
    return f"sh -c '{script}'"


def resume_grace_s() -> int:
    """How long a remote agent outlives a dropped session: the control
    plane's own resume grace (#783), so its runs can still be resumed."""
    return max(1, int(settings.WORKER_RESUME_GRACE_S))


def remote_command(entry: SshWorker, cp: str, remote_port: int, *,
                   forget_first: bool = False,
                   grace_s: int | None = None) -> str:
    """The command ssh runs on the remote host.

    One ``sh -c '...'`` whose script has no single quote and no backslash,
    so it reads the same to bash, zsh, fish and csh login shells. It reads
    the token from stdin, puts ``~/.local/bin`` on PATH and asks for
    English output (which this module reads). Then, in the per-slot state
    directory (``~/.codefyui-ssh/<cp>-<slot>``, where the forward's socket
    is):

    - it records itself as the attached session (``session``);
    - if the agent named in ``pid`` is alive and is this directory's agent
      (``own``), it **re-attaches**: no
      second agent; the agent reconnects through this session's tunnel by
      itself. Otherwise it starts one in the foreground (``--uds`` names
      the socket), its output in ``log``, its pid in ``pid``. Either way it
      prints the pid and follows ``log`` on its stderr;
    - a watcher reads ssh's stdin: a ``stop`` line sends the agent SIGTERM
      at once (Disable, Remove, shutdown). End of input without ``stop`` is
      a dropped session: after *grace_s* the agent gets SIGTERM, unless a
      newer session has attached by then.
    """
    cdui = _remote_cdui(entry)
    url = control_url(cp, remote_port)
    d = _state_dir(cp, remote_port)
    grace = resume_grace_s() if grace_s is None else max(1, int(grace_s))
    forget = (f"{cdui} agent forget {url} </dev/null >/dev/null 2>&1; "
              if forget_first else "")
    script = (
        'IFS= read -r t; [ -n "$t" ] || exit ' + str(NO_TOKEN_EXIT) + "; "
        "CODEFYUI_AGENT_TOKEN=$t; export CODEFYUI_AGENT_TOKEN; t=; "
        + _PROLOGUE + f"d={d}; " + _FUNCS +
        'mkdir -p "$d" && chmod 700 "$d" || exit ' + str(NO_STATE_EXIT) + "; "
        'echo $$ > "$d/session"; '
        # Whatever ends this shell ends its ``tail`` too (#829): a signal
        # (HUP, TERM, PIPE) runs the EXIT trap. Its children get the
        # default actions back.
        "l=; ct() { [ -z \"$l\" ] || kill $l 2>/dev/null; l=; }; trap ct EXIT; "
        'trap "exit 129" HUP; trap "exit 141" PIPE; trap "exit 143" TERM; '
        "exec 3<&0; "
        'a=$(cat "$d/pid" 2>/dev/null); o=; '
        'if own "$a"; then '
        "echo codefyui-ssh: agent pid $a reattached >&2; "
        # Only what the agent writes from now on: an old "refused the
        # token" or "is for worker" line is not this session's news.
        'tail -n 0 -f "$d/log" >&2 & l=$!; '
        "else "
        + forget
        + ': > "$d/log"; '
        f'{cdui} agent start --uds "$d/{SOCKET_NAME}" --foreground '
        f"--control {url} --name {entry.id} "
        '</dev/null 3<&- >> "$d/log" 2>&1 & a=$!; o=1; '
        'echo $a > "$d/pid"; '
        "echo codefyui-ssh: agent pid $a >&2; "
        'tail -n +1 -f "$d/log" >&2 & l=$!; '
        "fi; "
        # From here on this shell writes nothing. Once the session is gone
        # its stderr is a broken pipe, and a shell that reports a job it
        # killed (bash does: "Terminated") would die of SIGPIPE writing
        # that, before it ends its ``tail`` (#829).
        "exec >/dev/null 2>&1; "
        "( while IFS= read -r m; do "
        'if [ "$m" = stop ]; then kill -TERM $a; exit 0; fi; done; '
        # The grace in one-second steps: a watcher killed meanwhile leaves
        # a ``sleep`` behind for a second at most, and there is no window
        # in which a signal could orphan a long one.
        f"i=0; while [ $i -lt {grace} ]; do sleep 1; i=$((i+1)); done; "
        'if [ "$(cat "$d/session" 2>/dev/null)" = "$$" ]; then '
        "kill -TERM $a; exit 0; fi; exit 3 ) <&3 >/dev/null 2>&1 & w=$!; "
        "exec 3<&-; "
        "while al $a && al $w; do sleep 1; done; "
        "if al $w; then kill $w 2>/dev/null; ws=0; "
        "else wait $w; ws=$?; fi; "
        '[ "$ws" = 3 ] && exit 0; '
        "while al $a; do sleep 1; done; "
        "ct; s=0; "
        '[ -n "$o" ] && { wait $a; s=$?; }; '
        '[ "$(cat "$d/session" 2>/dev/null)" = "$$" ] '
        '&& rm -f "$d/pid"; '
        "exit $s"
    )
    return _sh(script)


def remote_stop_command(cp: str, remote_port: int) -> str:
    """Stop the agent of this control plane's *remote_port* state
    directory, if it is alive (Disable or Remove while no session is
    attached)."""
    d = _state_dir(cp, remote_port)
    return _sh(
        f"d={d}; " + _FUNCS +
        'a=$(cat "$d/pid" 2>/dev/null); '
        'if own "$a"; then kill -TERM $a; '
        "echo codefyui-ssh: stopped agent pid $a; fi; exit 0")


#: The remove command's lines: what it cleaned, and what it could not.
REMOVED_DIR = "codefyui-ssh: removed"
FORGOT = "codefyui-ssh: forgot"
#: How long the remove command waits for a still-running agent to exit.
REMOVE_AGENT_WAIT_S = 10
#: Time for ``cdui agent forget`` on the remote host (a ``uv run`` start
#: on a cold cache).
FORGET_ALLOWANCE_S = 30
#: The remove one-shot's whole budget: ssh's ``ConnectTimeout``, the wait
#: for the agent, and the forget. It runs in the background, after Delete
#: has answered.
REMOVE_TIMEOUT_S = float(CONNECT_TIMEOUT_S + REMOVE_AGENT_WAIT_S
                         + FORGET_ALLOWANCE_S)


def remote_remove_command(entry: SshWorker, cp: str, remote_port: int) -> str:
    """Delete (#829), after the session has ended: stop this state
    directory's agent if it still runs, then forget the credential the
    agent stored for this control plane's address (``cdui agent forget``)
    and remove the directory (``~/.codefyui-ssh/<cp>-<slot>``; this control
    plane's alone, never another's), then ``~/.codefyui-ssh`` itself when
    nothing else is left in it. A session still attached there, or an
    agent that does not exit within :data:`REMOVE_AGENT_WAIT_S`, keeps both
    the credential and the directory (and says so). Each step prints what
    it did."""
    cdui = _remote_cdui(entry)
    url = control_url(cp, remote_port)
    d = _state_dir(cp, remote_port)
    key = state_key(cp, remote_port)
    return _sh(
        _PROLOGUE + f"d={d}; " + _FUNCS +
        'a=$(cat "$d/pid" 2>/dev/null); k=1; i=0; '
        'if own "$a"; then kill -TERM $a; '
        f"while al $a && [ $i -lt {REMOVE_AGENT_WAIT_S} ]; do "
        "sleep 1; i=$((i+1)); done; "
        'if al $a; then k=; echo "codefyui-ssh: agent pid $a still runs; '
        'kept $d and its credential"; fi; fi; '
        # A dropped session's script outlives its ssh until its agent has
        # gone (it notices within a second): the same budget for that.
        's=$(cat "$d/session" 2>/dev/null); '
        f'at() {{ [ -n "$s" ] && al "$s" && has "$s" ".codefyui-ssh/{key};"; '
        "}; "
        f"while at && [ $i -lt {REMOVE_AGENT_WAIT_S} ]; do "
        "sleep 1; i=$((i+1)); done; "
        'if at; then k=; echo "codefyui-ssh: session pid $s still attached; '
        'kept $d and its credential"; fi; '
        'if [ -n "$k" ]; then '
        f"if {cdui} agent forget {url} </dev/null >/dev/null 2>&1; then "
        f'echo "{FORGOT} {url}"; else '
        f'echo "codefyui-ssh: could not forget {url}"; fi; '
        f'rm -rf "$d" && echo "{REMOVED_DIR} $d" '
        f'&& rmdir "$HOME"/{REMOTE_STATE_DIR} 2>/dev/null; fi; '
        "exit 0")


def remote_prepare_command(cp: str, remote_port: int) -> str:
    """Before each session: create *remote_port*'s state directory (mode
    0700, as is ``~/.codefyui-ssh``), remove a socket a dropped session
    left there (``-R`` cannot bind over it), and print the directory's
    absolute path (``-R`` takes no ``~``). The directory is this control
    plane's alone (``<cp>-<slot>``): the socket removed is never another's
    live one."""
    d = _state_dir(cp, remote_port)
    legacy = f'"$HOME"/{REMOTE_STATE_DIR}/{remote_port}'
    tcp = control_url_legacy(remote_port)
    return _sh(
        f"umask 077; d={d}; L={legacy}; " + _FUNCS +
        f'mkdir -p "$d" && chmod 700 "$HOME"/{REMOTE_STATE_DIR} "$d" '
        f"|| exit {NO_STATE_EXIT}; "
        f'rm -f "$d/{SOCKET_NAME}"; '
        # An agent of an earlier build, in the old ``<slot>/`` layout,
        # dialling a TCP forward: stopped once its session is gone (it
        # can never connect again), reported while one is attached.
        'la=$(cat "$L/pid" 2>/dev/null); ls=$(cat "$L/session" 2>/dev/null); '
        'if [ -n "$la" ] && al "$la" && has "$la" "agent start" '
        f'&& has "$la" "{tcp}" && ! has "$la" "--uds"; then '
        'if [ -n "$ls" ] && al "$ls"; then '
        'echo "codefyui-ssh: legacy agent pid $la in $L is still attached; '
        'left running"; '
        'else kill -TERM "$la"; '
        'echo "codefyui-ssh: stopped legacy agent pid $la in $L"; fi; fi; '
        'echo "codefyui-ssh: dir $d"')


def socket_path(remote_dir: str) -> str:
    return f"{remote_dir.rstrip('/')}/{SOCKET_NAME}"


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


def ssh_argv(ssh: str, entry: SshWorker, cp: str, remote_port: int,
             listener: str, *, remote_dir: str, forget_first: bool = False,
             grace_s: int | None = None) -> list[str]:
    """The session: *remote_dir*'s socket forwarded to *listener* on this
    machine (the SSH worker's own listener: a socket path, or
    ``127.0.0.1:<port>``)."""
    argv = ssh_base_argv(ssh, entry)
    argv[2:2] = ["-o", "ExitOnForwardFailure=yes"]
    argv += ["-R", f"{socket_path(remote_dir)}:{listener}",
             "--", validate_target(entry.target),
             remote_command(entry, cp, remote_port,
                            forget_first=forget_first, grace_s=grace_s)]
    return argv


def ssh_oneshot_argv(ssh: str, entry: SshWorker, command: str) -> list[str]:
    """ssh without a forward, for one remote command."""
    return [*ssh_base_argv(ssh, entry), "--", validate_target(entry.target),
            command]


#: The one-shot's answer when there is no ssh to run.
NO_SSH = "ssh was not found on this machine's PATH"


async def run_oneshot(launcher: SshLauncher, entry: SshWorker, command: str,
                      what: str, *, timeout_s: float | None = None,
                      ) -> tuple[int | None, list[str]]:
    """Run *command* on *entry*'s host over a short ssh of its own (no
    forward). Returns ssh's exit status and its output lines. The status is
    None when ssh did not run (the last line says why) or did not finish
    within *timeout_s* (:data:`ONESHOT_TIMEOUT_S`): it is killed then, and
    reaped, as it is when this task is cancelled, and the last line says
    whether the host answered at all."""
    timeout_s = ONESHOT_TIMEOUT_S if timeout_s is None else timeout_s
    ssh = launcher.ssh_path()
    if ssh is None:
        return None, [NO_SSH]
    kwargs: dict[str, Any] = {}
    if sys.platform != "win32":
        kwargs["start_new_session"] = True
    try:
        proc = await asyncio.create_subprocess_exec(
            *ssh_oneshot_argv(ssh, entry, command),
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, **kwargs)
    except OSError as exc:
        logger.warning("SSH worker %s: could not %s: %s", entry.id, what, exc)
        return None, [f"could not start ssh: {exc}"]
    lines: list[str] = []

    async def collect() -> None:
        assert proc.stdout is not None
        while True:
            try:
                raw = await proc.stdout.readline()
            except (ValueError, asyncio.LimitOverrunError):
                continue  # an over-long line; skip it
            if not raw:
                return
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line.strip():
                lines.append(line)

    try:
        await asyncio.wait_for(asyncio.gather(collect(), proc.wait()),
                               timeout_s)
    except asyncio.TimeoutError:
        if lines:
            message = (f"ssh was cut off after {timeout_s:.0f} s trying to "
                       f"{what}; its last line: {lines[-1]}")
        else:
            message = (f"ssh did not finish within {timeout_s:.0f} s "
                       f"trying to {what} (no answer)")
        logger.warning("SSH worker %s: %s", entry.id, message)
        lines.append(message)
        return None, lines
    finally:
        if proc.returncode is None:
            # Timed out, or this task was cancelled: never left behind.
            # A cancellation is re-raised after this ``finally``.
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(proc.wait()),
                                       KILL_GRACE_S)
    logger.info("SSH worker %s: %s (ssh exited with status %s)", entry.id,
                what, proc.returncode)
    return proc.returncode, lines


async def _first_of(stop: asyncio.Event, wake: asyncio.Event,
                    timeout: float) -> bool:
    """Wait up to *timeout* for *stop* or *wake*; True when *wake* came."""
    waiters = [asyncio.ensure_future(stop.wait()),
               asyncio.ensure_future(wake.wait())]
    try:
        await asyncio.wait(waiters, timeout=timeout,
                           return_when=asyncio.FIRST_COMPLETED)
    finally:
        for waiter in waiters:
            waiter.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)
    return wake.is_set() and not stop.is_set()


def next_backoff(current: float) -> float:
    return min(BACKOFF_MAX_S, max(BACKOFF_MIN_S, current * 2))


def classify_failure(code: int | None, lines: list[str], entry: SshWorker,
                     *, session: bool) -> tuple[str, str] | None:
    """``(kind, reason)`` for a failed ssh (exit *code*, output *lines*)
    that a plain retry will not fix, or None for one worth retrying
    (#840). ``final``: the host key does not verify, or, for the *session*
    (not a one-shot, which never runs ``cdui``), the remote shell cannot
    run ``cdui`` (exit 127 / 126). ``lasting``: a refused login or a host
    name that does not resolve, which can heal by themselves."""
    lowered = " ".join(lines).lower()
    for words, kind, reason in FAILURE_WORDS:
        if words in lowered:
            return kind, reason.format(target=entry.target)
    if session and code in (COMMAND_NOT_FOUND_EXIT,
                            COMMAND_NOT_EXECUTABLE_EXIT):
        what = "not found" if code == COMMAND_NOT_FOUND_EXIT else (
            "not executable")
        return FAILURE_FINAL, (
            f"the remote host cannot run {entry.remote_cdui} (exit {code}: "
            f"{what}); fix the cdui path of this SSH worker, or install "
            "CodefyUI there with install.sh")
    return None


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
        #: This attempt found a credential to clear: retry without backoff.
        self._retry_soon = False
        self._refused = False
        self._mismatch = False
        self._forward_failures = 0
        #: When (monotonic) the current unbroken run of ``lasting``
        #: failures began, or None (#840, :func:`classify_failure`).
        self._lasting_since: float | None = None
        #: Set by Retry while the loop waits out its backoff: try now.
        self._wake = asyncio.Event()
        #: The next tail line starts a new attempt's output (#840).
        self._fresh_tail = False
        #: The slot a session last attached on (the remote script printed
        #: the agent pid): an agent may survive there.
        self._attached_port: int | None = None
        #: This SSH worker's own listener (``core.tunnel_listener``), where
        #: the forward lands; open from the first attempt to :meth:`stop`.
        self._listener: TunnelListener | None = None
        #: The remote state directory, absolute (the prepare step's answer).
        self.remote_dir: str | None = None
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
            # Retry on an entry still retrying: cut its backoff short.
            self._wake.set()
            return
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()
        self.state = STATE_CONNECTING
        self.gave_up = False
        self._forward_failures = 0
        self._lasting_since = None
        self._task = asyncio.get_running_loop().create_task(
            self._loop(), name=f"ssh-worker-{self.entry.id}")

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def stop(self, grace_s: float = STOP_GRACE_S) -> None:
        """End the ssh session: send ``stop`` and close its stdin (the
        remote agent winds down, its runs report how they ended), wait up
        to *grace_s*, then terminate, then kill. Then close the listener and
        end whatever run of this worker is still waiting for it. The loop
        does not restart it."""
        self._stop.set()
        proc = self._proc
        if proc is not None and proc.returncode is None:
            await _end_process(proc, grace_s)
        elif self._attached_port is not None:
            # No session to send ``stop`` through: an agent that survived
            # a drop may still run there, waiting out its grace.
            # Within *grace_s* too: a cancelled one-shot kills its ssh.
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._oneshot(
                    remote_stop_command(self.launcher.instance_id,
                                        self._attached_port),
                    "stop the remote agent"), grace_s)
        self._attached_port = None
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(self._task, KILL_GRACE_S)
            if not self._task.done():  # pragma: no cover - defensive
                self._task.cancel()
        self._task = None
        await self._close_listener()
        self._port = None
        await self._drop_enrollment()
        # Stopped on purpose: a run the agent did not report the end of
        # (it was killed, or the session was gone) will not come back; it
        # ends now, not after the resume grace.
        self.launcher.worker_stopped(self.entry.id)
        self.state = STATE_STOPPED
        self.retry_at = None
        self.up_since = None
        self.pid = None
        self.remote_pid = None
        self.reattached = False

    # ── the loop ──────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        backoff = BACKOFF_MIN_S
        while not self._stop.is_set():
            self.attempts += 1
            self.retry_at = None
            self._retry_soon = False
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
            if self._retry_soon:
                # A credential to clear: no reason to wait long for that.
                # (Not every pending forget: a new slot's forget waits for
                # a session, and a host that fails before one is backed
                # off as usual, #840.)
                backoff = BACKOFF_MIN_S
            self.retry_at = time.time() + backoff
            logger.info("SSH worker %s: retrying in %.0f s", self.entry.id,
                        backoff)
            woken = await _first_of(self._stop, self._wake, backoff)
            self._wake.clear()
            backoff = BACKOFF_MIN_S if woken else next_backoff(backoff)

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
            self._fail(f"no free slot left in {low}-{high} "
                       "(CODEFYUI_SSH_REMOTE_PORT_RANGE)")
            return False
        listener = await self._listen()
        # Each attempt's output only: a host that keeps refusing would
        # otherwise fill the tail with the same line (#840). The last
        # attempt's stays shown until this one writes something.
        self._fresh_tail = True
        remote_dir = await self._prepare(port)
        if remote_dir is None or self._stop.is_set():
            return False
        token = await self._mint_token()
        if self._stop.is_set():
            return False
        argv = ssh_argv(ssh, self.entry, self.launcher.instance_id, port,
                        listener, remote_dir=remote_dir,
                        forget_first=self._forget_first)
        self.reattached = False
        self._forget_first = False
        self._refused = False
        self._mismatch = False
        self.remote_pid = None
        # The session's output replaces the prepare step's, once it has
        # some: the card never goes blank meanwhile.
        self._fresh_tail = True
        self.state = STATE_CONNECTING
        logger.info("SSH worker %s: ssh %s (slot %d, forward to %s)",
                    self.entry.id, self.entry.target, port, listener)
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
                        self._lasting_since = None
                        logger.info("SSH worker %s is up", self.entry.id)
                        await self._drop_enrollment()
                    was_up = True
                elif self.state == STATE_UP:
                    self.state = STATE_CONNECTING
                    self.up_since = None
                if self._refused:
                    # The agent says this server refused the credential it
                    # stored for this address (another control plane's, or
                    # revoked): start over with a forget.
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
        if self._mismatch or self._refused:
            # The agent's stored credential was refused, or names another
            # worker: the next session forgets it first. (The agent now
            # exits on a refused credential, #840, ending the session.)
            self._forget_first = True
            self._retry_soon = True
        if code in (COMMAND_NOT_FOUND_EXIT, COMMAND_NOT_EXECUTABLE_EXIT) \
                and not self.reattached:
            # The remote script printed a pid for an agent that never ran.
            self.remote_pid = None
            self._attached_port = None
            tail = [line for line in tail if REMOTE_PID.search(line) is None]
        if code == NO_TOKEN_EXIT:
            message = "the remote shell got no token on stdin"
        else:
            message = "\n".join(tail[-ERROR_LINES:]) or (
                f"ssh exited with status {code}")
        lowered = " ".join(tail).lower()
        if any(word in lowered for word in FORWARD_FORBIDDEN):
            self._give_up(
                f"{self.entry.target} does not allow forwarding to a Unix "
                "socket (sshd AllowStreamLocalForwarding / "
                "DisableForwarding); ask its admin to allow it, then press "
                f"Enable.\n{message}", code)
            return was_up
        if any(FORWARD_FAILED in line for line in tail) and not was_up:
            self._forward_failures += 1
            if self._forward_failures >= FORWARD_GIVE_UP:
                self._give_up(
                    f"the forward to a Unix socket failed {FORWARD_GIVE_UP} "
                    f"times in a row on {self.entry.target}: its sshd "
                    "probably does not allow it (AllowStreamLocalForwarding; "
                    "OpenSSH 6.7 or later is needed). Press Enable to try "
                    f"again.\n{message}", code)
                return was_up
        else:
            self._forward_failures = 0
        if self._permanent(code, tail, message, was_up, session=True):
            return was_up
        self._fail(message, code)
        logger.warning("SSH worker %s: ssh exited with status %s: %s",
                       self.entry.id, code, message.replace("\n", " | "))
        return was_up

    def _permanent(self, code: int | None, lines: list[str], message: str,
                   was_up: bool = False, *, session: bool = False) -> bool:
        """Record a failure a plain retry does not fix (#840): a ``final``
        one ends the retries; a ``lasting`` one is shown with its reason and
        retried, and ends them once it has lasted :data:`LASTING_GIVE_UP_S`.
        True when it recorded the failure (the caller records nothing)."""
        found = None if was_up else classify_failure(
            code, lines, self.entry, session=session)
        if found is None:
            self._lasting_since = None
            return False
        kind, reason = found
        if kind == FAILURE_FINAL:
            self._lasting_since = None
            self._give_up(f"{reason}. Not retrying; press Retry once it is "
                          f"fixed.\n{message}", code)
            return True
        now = time.monotonic()
        if self._lasting_since is None:
            self._lasting_since = now
        lasted = now - self._lasting_since
        if lasted >= LASTING_GIVE_UP_S:
            self._give_up(f"{reason}, for {lasted / 60:.0f} min. Not "
                          "retrying; press Retry once it is fixed.\n"
                          f"{message}", code)
            return True
        self._fail(f"{reason}. Retrying; it stops if this lasts "
                   f"{LASTING_GIVE_UP_S / 60:.0f} min. Retry tries again "
                   f"now.\n{message}", code)
        logger.warning("SSH worker %s: %s (retrying)", self.entry.id, reason)
        return True

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
            self._add_tail([line])
            pid = REMOTE_PID.search(line)
            if pid is not None:
                self.remote_pid = int(pid.group(1))
                self.reattached = "reattached" in line
                self._attached_port = self._port
            if CREDENTIAL_MISMATCH in line:
                self._mismatch = True
            if TOKEN_REFUSED in line:
                self._refused = True
            logger.debug("ssh %s: %s", self.entry.id, line)

    def _add_tail(self, lines: list[str]) -> None:
        """Keep *lines*; the first of a new attempt replaces the last
        attempt's output."""
        if lines and self._fresh_tail:
            self._fresh_tail = False
            self.tail.clear()
        self.tail.extend(lines)

    def _worker_online(self) -> bool:
        worker = self.launcher.registry.get(self.entry.id)
        return worker is not None and bool(worker.online)

    # ── ports and tokens ──────────────────────────────────────────────────

    async def _ensure_port(self) -> int | None:
        port = self.entry.remote_port
        if port is None or not self.launcher.port_usable(port, self.entry.id):
            port = self.launcher.free_port(self.entry.id)
            if port is None:
                return None
            await self._save_port(port)
        self._port = port
        return port

    async def _save_port(self, port: int) -> None:
        """Take *port* for this entry. A port this entry has not used may
        hold another worker's credential on the remote host (or another
        control plane's): the first launch on it forgets that first."""
        self.entry = replace(self.entry, remote_port=port)
        self.launcher.entries[self.entry.id] = self.entry
        self._forget_first = True
        await self.launcher.store.update_ssh_worker(self.entry.id,
                                                    remote_port=port)

    # ── the listener and the remote directory ────────────────────────────

    async def _listen(self) -> str:
        """Where the forward goes on this machine (``-R``'s right-hand
        side): this SSH worker's own listener, opened at the first attempt
        and kept until :meth:`stop`, opened again when it has died (a
        launcher without an app, in unit tests, has a fixed port)."""
        if self._listener is not None and not self._listener.alive:
            await self._close_listener()
        if self._listener is None:
            listener = self.launcher.make_listener(self.entry.id)
            if listener is None:
                return f"127.0.0.1:{self.launcher.local_port()}"
            await listener.open()
            self._listener = listener
        assert self._listener.target is not None
        return self._listener.target

    async def _close_listener(self) -> None:
        listener, self._listener = self._listener, None
        if listener is not None:
            await listener.close()

    async def _prepare(self, port: int) -> str | None:
        """Run :func:`remote_prepare_command`: the absolute state directory
        on the remote host, or None (the failure is recorded)."""
        cp = self.launcher.instance_id
        code, lines = await self._oneshot(remote_prepare_command(cp, port),
                                          "prepare the remote directory")
        found = [m.group(1) for m in map(REMOTE_DIR.search, lines)
                 if m is not None]
        if code != 0 or not found:
            text = "\n".join(lines[-ERROR_LINES:])
            if code == NO_STATE_EXIT:
                text = (f"could not create ~/{REMOTE_STATE_DIR}/"
                        f"{state_key(cp, port)} on "
                        f"{self.entry.target}\n{text}")
            text = text or (f"preparing {self.entry.target} failed (ssh "
                            f"exited with status {code})")
            if not self._permanent(code, lines, text):
                self._fail(text, code)
            return None
        remote_dir = found[-1].strip()
        if SAFE_REMOTE_DIR.fullmatch(remote_dir) is None:
            self._give_up(
                f"the remote state directory {remote_dir!r} on "
                f"{self.entry.target} has a character an SSH worker cannot "
                "use (a space, ':' or a quote, from $HOME): only letters, "
                "digits and . _ @ + - / are supported. Use an account whose "
                "home directory is spelled that way.", None)
            return None
        if len(socket_path(remote_dir).encode()) > MAX_SOCKET_PATH:
            self._give_up(
                f"the socket path {socket_path(remote_dir)} on "
                f"{self.entry.target} is longer than {MAX_SOCKET_PATH} bytes, "
                "which a Unix socket cannot have (a very long $HOME)", None)
            return None
        self.remote_dir = remote_dir
        return remote_dir

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

    async def _oneshot(self, command: str,
                       what: str) -> tuple[int | None, list[str]]:
        """:func:`run_oneshot` for this entry; its output also goes to the
        tail."""
        code, lines = await run_oneshot(self.launcher, self.entry, command,
                                        what)
        self._add_tail(lines)
        return code, lines


async def _end_process(proc: asyncio.subprocess.Process,
                       grace_s: float = STOP_GRACE_S) -> None:
    """Send ``stop`` (the remote agent ends now, not after the resume
    grace) and close stdin, wait *grace_s*; terminate, wait; kill."""
    if proc.stdin is not None:
        with contextlib.suppress(Exception):
            proc.stdin.write(b"stop\n")
            await proc.stdin.drain()
        with contextlib.suppress(Exception):
            proc.stdin.close()
    for step, grace in (("eof", grace_s), ("terminate", KILL_GRACE_S),
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
                 local_port: int | None = None,
                 app: Any = None,
                 instance_id: str | None = None) -> None:
        self.store = store
        self.registry = registry
        #: This control plane's id on the remote hosts
        #: (:func:`load_instance_id`).
        self._instance_id = instance_id
        #: The ASGI app each SSH worker's listener serves (its agent routes
        #: only); None in unit tests, which forward to *local_port*.
        self.app = app
        self._ssh = ssh
        self.port_range = port_range or parse_port_range(
            settings.SSH_REMOTE_PORT_RANGE)
        self._local_port = local_port
        self.entries: dict[str, SshWorker] = {}
        self.launches: dict[str, SshLaunch] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        #: Slots of deleted entries whose remote clean-up still runs, and
        #: whose they were. No entry takes one meanwhile: the clean-up would
        #: forget its credential and remove its directory.
        self.reserved: dict[int, str] = {}
        #: Those clean-ups (:meth:`_clean_remote`), each a task.
        self._cleanups: set[asyncio.Task] = set()

    @property
    def instance_id(self) -> str:
        """This control plane's id on the remote hosts, read (or made) the
        first time a launch needs it."""
        if self._instance_id is None:
            self._instance_id = load_instance_id()
        return self._instance_id

    def ssh_path(self) -> str | None:
        return self._ssh or ssh_executable()

    def local_port(self) -> int:
        return self._local_port or settings.PORT

    def make_listener(self, name: str) -> TunnelListener | None:
        """A new listener for SSH worker *name*, or None without an app."""
        return TunnelListener(self.app, name) if self.app is not None else None

    def worker_stopped(self, name: str) -> None:
        """SSH worker *name* was stopped on purpose (Disable, Remove, a
        shutdown): its runs still waiting for it end now."""
        stopped = getattr(self.registry, "stopped", None)
        if stopped is not None:
            stopped(name, f"SSH worker {name} was stopped")

    def _lock(self, name: str) -> asyncio.Lock:
        lock = self._locks.get(name)
        if lock is None:
            lock = self._locks[name] = asyncio.Lock()
        return lock

    # ── ports ─────────────────────────────────────────────────────────────

    def _ports_taken(self, exclude: str) -> set[int]:
        taken = {entry.remote_port for name, entry in self.entries.items()
                 if name != exclude and entry.remote_port is not None}
        # A deleted entry's slot, still being cleaned: nobody's, this
        # entry's (a new one under the same name) included.
        taken.update(self.reserved)
        return taken

    def port_usable(self, port: int, owner: str) -> bool:
        """*port* is a slot in the range no other entry holds (a slot names
        a state directory, not a TCP port)."""
        low, high = self.port_range
        return low <= port <= high and port not in self._ports_taken(owner)

    def free_port(self, owner: str) -> int | None:
        """The first usable slot, or None."""
        low, high = self.port_range
        for port in range(low, high + 1):
            if self.port_usable(port, owner):
                return port
        return None

    # ── events from the socket side ──────────────────────────────────────

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

    async def remove(self, name: str, *,
                     before_cleanup: Callable[[], Awaitable[Any]]
                     | None = None) -> bool:
        """Stop *name* and delete its row (under the entry's lock), then
        await *before_cleanup* (the route's revocation of the worker) and
        start the clean-up on the remote host in the background
        (:meth:`_clean_remote`); the entry's slot stays reserved until that
        has finished. False when there is no such row."""
        async with self._lock(name):
            launch = self.launches.pop(name, None)
            if launch is not None:
                await launch.stop()
            entry = self.entries.pop(name, None)
            deleted = await self.store.delete_ssh_worker(name)
            port = entry.remote_port if entry is not None else None
            if port is not None:
                self.reserved[port] = name
        try:
            if deleted and before_cleanup is not None:
                await before_cleanup()
        finally:
            if entry is not None and port is not None:
                self._start_cleanup(entry, port)
        return deleted

    def _start_cleanup(self, entry: SshWorker, port: int) -> None:
        task = asyncio.get_running_loop().create_task(
            self._clean_remote(entry, port), name=f"ssh-clean-{entry.id}")
        self._cleanups.add(task)

        def done(finished: asyncio.Task) -> None:
            self._cleanups.discard(finished)
            if self.reserved.get(port) == entry.id:
                del self.reserved[port]

        task.add_done_callback(done)

    async def wait_for_cleanups(self) -> None:
        """Until every clean-up started so far has finished."""
        while self._cleanups:
            await asyncio.gather(*list(self._cleanups),
                                 return_exceptions=True)

    async def _clean_remote(self, entry: SshWorker, port: int) -> None:
        """Best effort, within :data:`REMOVE_TIMEOUT_S`: run
        :func:`remote_remove_command` for *entry*'s *port*, so neither its
        state directory nor the credential its agent stored stays on the
        remote host. Disable never does this (Enable reuses both). Whatever
        could not be removed (an unreachable host, an agent that would not
        stop) is logged."""
        cp = self.instance_id
        where = f"~/{REMOTE_STATE_DIR}/{state_key(cp, port)}/"
        url = control_url(cp, port)
        left = [where, f"the credential stored for {url} (cdui agent "
                       f"forget {url})"]
        code: int | None = None
        lines: list[str] = []
        try:
            code, lines = await run_oneshot(
                self, entry, remote_remove_command(entry, cp, port),
                "remove the remote state", timeout_s=REMOVE_TIMEOUT_S)
        except asyncio.CancelledError:
            logger.warning(
                "SSH worker %s deleted; its clean-up on %s was cut short "
                "(server shutdown), so these were left behind: %s",
                entry.id, entry.target, "; ".join(left))
            raise
        except Exception as exc:  # noqa: BLE001 -- logged, Delete is done
            lines = [str(exc) or type(exc).__name__]
        if any(REMOVED_DIR in line for line in lines):
            left.remove(where)
        if any(FORGOT in line for line in lines):
            left.pop()
        if left:
            reason = " | ".join(lines[-ERROR_LINES:]) or (
                f"ssh exited with status {code}")
            logger.warning(
                "SSH worker %s deleted, but on %s these were left behind: "
                "%s (%s)", entry.id, entry.target, "; ".join(left), reason)
        else:
            logger.info("SSH worker %s: removed %s and the credential for "
                        "%s on %s", entry.id, where, url, entry.target)

    async def shutdown(self) -> None:
        """Stop every SSH worker at once: each gets ``stop`` immediately,
        and :data:`SHUTDOWN_GRACE_S` in all to wind down before its ssh is
        terminated, then killed (and reaped)."""
        launches = list(self.launches.values())
        if launches:
            logger.info("closing %d SSH worker session(s)", len(launches))
        await asyncio.gather(*(launch.stop(SHUTDOWN_GRACE_S)
                               for launch in launches),
                             return_exceptions=True)
        # Clean-ups of deleted entries still running: a moment to finish,
        # then cancelled (their ssh is killed and reaped; each logs what it
        # left behind).
        pending = list(self._cleanups)
        if pending:
            _, late = await asyncio.wait(pending, timeout=KILL_GRACE_S)
            for task in late:
                task.cancel()
            await asyncio.gather(*late, return_exceptions=True)

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
