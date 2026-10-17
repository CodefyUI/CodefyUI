"""A stand-in for ``ssh`` in the SSH-worker tests (#784).

``tests/test_ssh_launch.py`` installs it on PATH as ``ssh`` (a shim that
runs this file with the test interpreter). It appends its argv, as one
JSON line, to ``$FAKE_SSH_LOG`` and the first stdin line (the token) to
``$FAKE_SSH_LOG.stdin``, then acts by ``$FAKE_SSH_MODE``:

- ``hold`` (default): print a line, then wait for stdin to end and exit 0
  -- an ssh session whose remote agent ends when its stdin does;
- ``fail``: print an ssh authentication error and exit 255;
- ``nohost``: print ssh's "Could not resolve hostname" and exit 255;
- ``hostkey``: print ssh's "Host key verification failed." and exit 255;
- ``prepare127``: exit 127 from the prepare command (a remote shell that
  lacks a tool the prepare step runs), then hold like ``hold``;
- ``down``: print ssh's "Connection refused" and exit 255 (a failure
  worth retrying);
- ``nocdui``: answer the prepare command, then in the session print the
  remote script's pid line and the shell's "cdui: not found", and exit
  127, as a wrong cdui path does;
- ``refused``: print the agent's "the control plane refused the token"
  error, then hold like ``hold``;
- ``mismatch``: print the agent's "credential is for worker X" error and
  exit 2, as an agent with another worker's stored credential does;
- ``forbidden``: refuse every forward with sshd's "administratively
  prohibited" words;
- ``tunnel``: be the reverse forward for real -- listen on the ``-R``
  remote Unix socket path and relay each connection to the local end --
  and run the remote command with ``sh -c`` (``$HOME`` from
  ``$FAKE_SSH_HOME``), its stdin copied from this process's, as an ssh
  server would: when this process dies (a dropped session), the command
  reads the end of its input. The exit status is the command's.

Without ``-R`` (a one-shot command: prepare, stop) ``tunnel`` mode runs
the command the same way; the other modes answer the prepare command with
a made-up directory (``/home/fake/.codefyui-ssh/<cp>-<slot>``, or
``$FAKE_SSH_PREPARE_DIR``) and exit 0.

A ``-R`` whose slot (``.codefyui-ssh/<cp>-<slot>/sock``) is listed in
``$FAKE_SSH_BUSY_PORTS`` (comma-separated, or ``all``) fails the way
``ExitOnForwardFailure=yes`` does, in every mode but ``fail``. In ``hold``
mode it prints the remote script's pid line.

With ``$FAKE_SSH_HANG_S`` above 0 a one-shot writes its pid to
``$FAKE_SSH_LOG.pid`` and sleeps that long (a remote host that never
answers, or, with ``$FAKE_SSH_HANG_SAY``, one that prints that line
first and then stalls), for the launcher's timeout to kill.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time


def _pipe(source: socket.socket, sink: socket.socket) -> None:
    try:
        while True:
            data = source.recv(65536)
            if not data:
                break
            sink.sendall(data)
    except OSError:
        pass
    finally:
        for sock in (source, sink):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def _dial(local: str) -> socket.socket:
    """``-R``'s local end: a socket path, or ``host:port``."""
    if local.startswith("/"):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(local)
        except OSError:
            sock.close()
            raise
        return sock
    host, _, port = local.rpartition(":")
    return socket.create_connection((host.strip("[]"), int(port)))


def _relay(listener: socket.socket, local: str) -> None:
    while True:
        try:
            client, _ = listener.accept()
        except OSError:
            return
        try:
            upstream = _dial(local)
        except OSError:
            client.close()
            continue
        for a, b in ((client, upstream), (upstream, client)):
            threading.Thread(target=_pipe, args=(a, b), daemon=True).start()


def _remote_env() -> dict[str, str]:
    env = dict(os.environ)
    if os.environ.get("FAKE_SSH_HOME"):
        env["HOME"] = os.environ["FAKE_SSH_HOME"]
    return env


def _copy_stdin(child: subprocess.Popen) -> None:
    """ssh's input to the remote command, line by line, then its end."""
    assert child.stdin is not None
    try:
        for line in sys.stdin.buffer:
            child.stdin.write(line)
            child.stdin.flush()
    except (OSError, ValueError):
        pass
    finally:
        try:
            child.stdin.close()
        except OSError:
            pass


def main() -> int:
    argv = sys.argv[1:]
    log = os.environ["FAKE_SSH_LOG"]
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(argv) + "\n")
    mode = os.environ.get("FAKE_SSH_MODE", "hold")
    if mode == "fail":
        print("alice@gpu-box: Permission denied (publickey).",
              file=sys.stderr, flush=True)
        return 255
    if mode == "nohost":
        print("ssh: Could not resolve hostname gpu-box: Name or service not "
              "known", file=sys.stderr, flush=True)
        return 255
    if mode == "hostkey":
        print("Host key verification failed.", file=sys.stderr, flush=True)
        return 255
    if mode == "prepare127" and "-R" not in argv \
            and "codefyui-ssh: dir" in argv[-1]:
        print("sh: 1: mkdir: not found", file=sys.stderr, flush=True)
        return 127
    if mode == "down":
        print("ssh: connect to host gpu-box port 22: Connection refused",
              file=sys.stderr, flush=True)
        return 255
    if "-R" not in argv:
        hang = float(os.environ.get("FAKE_SSH_HANG_S") or 0)
        if hang > 0:
            # A one-shot that never finishes: the launcher must kill it.
            with open(log + ".pid", "a", encoding="utf-8") as handle:
                handle.write(f"{os.getpid()}\n")
            if os.environ.get("FAKE_SSH_HANG_SAY"):
                print(os.environ["FAKE_SSH_HANG_SAY"], flush=True)
            time.sleep(hang)
            return 0
        if mode != "tunnel":
            key = re.search(r"\.codefyui-ssh/([0-9a-f]+-\d+)", argv[-1])
            if "codefyui-ssh: dir" in argv[-1] and key is not None:
                where = (os.environ.get("FAKE_SSH_PREPARE_DIR")
                         or f"/home/fake/.codefyui-ssh/{key.group(1)}")
                print(f"codefyui-ssh: dir {where}", flush=True)
            return 0
        return subprocess.run(["sh", "-c", argv[-1]], env=_remote_env(),
                              stdin=subprocess.DEVNULL,
                              check=False).returncode
    forward = argv[argv.index("-R") + 1]
    path, rest = forward.split(":", 1)
    slot = re.search(r"\.codefyui-ssh/[0-9a-f]+-(\d+)/", path)
    remote_port = slot.group(1) if slot is not None else ""
    if mode == "forbidden":
        print("Warning: remote port forwarding failed for listen path "
              f"{path}: administratively prohibited",
              file=sys.stderr, flush=True)
        return 255
    busy = os.environ.get("FAKE_SSH_BUSY_PORTS", "").split(",")
    if remote_port in busy or "all" in busy:
        print(f"Error: remote port forwarding failed for listen path "
              f"{path}", file=sys.stderr, flush=True)
        return 255
    if mode == "tunnel":
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            old = os.umask(0o177)
            try:
                listener.bind(path)
            finally:
                os.umask(old)
        except OSError:
            print(f"Error: remote port forwarding failed for listen path "
                  f"{path}", file=sys.stderr, flush=True)
            return 255
        listener.listen(16)
        threading.Thread(target=_relay, daemon=True,
                         args=(listener, rest)).start()
        child = subprocess.Popen(["sh", "-c", argv[-1]], env=_remote_env(),
                                 stdin=subprocess.PIPE)
        threading.Thread(target=_copy_stdin, args=(child,),
                         daemon=True).start()
        return child.wait()
    token = sys.stdin.readline().strip()
    with open(log + ".stdin", "a", encoding="utf-8") as handle:
        handle.write(token + "\n")
    if mode == "nocdui":
        print("codefyui-ssh: agent pid 4343", file=sys.stderr, flush=True)
        print("sh: 1: /home/alice/nowhere/cdui: not found", file=sys.stderr,
              flush=True)
        return 127
    if mode == "mismatch":
        print("the credential in /home/a/agent/127.0.0.1-18000.token is for "
              "worker 'other-box', not 'gpu-box'; to change the name, run "
              "cdui agent forget http://127.0.0.1:18000 and enroll with a new "
              "token", file=sys.stderr, flush=True)
        return 2
    print("codefyui-ssh: agent pid 4242", file=sys.stderr, flush=True)
    if mode == "refused":
        print("2026-01-01 00:00:00,000 ERROR app.agent.connection: the "
              "control plane refused the token (HTTP 401); token from the "
              "stored credential", file=sys.stderr, flush=True)
    for line in sys.stdin:
        with open(log + ".stdin", "a", encoding="utf-8") as handle:
            handle.write(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
