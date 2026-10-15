"""A stand-in for ``ssh`` in the SSH-worker tests (#784).

``tests/test_ssh_launch.py`` installs it on PATH as ``ssh`` (a shim that
runs this file with the test interpreter). It appends its argv, as one
JSON line, to ``$FAKE_SSH_LOG`` and the first stdin line (the token) to
``$FAKE_SSH_LOG.stdin``, then acts by ``$FAKE_SSH_MODE``:

- ``hold`` (default): print a line, then wait for stdin to end and exit 0
  -- an ssh session whose remote agent ends when its stdin does;
- ``fail``: print an ssh authentication error and exit 255;
- ``mismatch``: print the agent's "credential is for worker X" error and
  exit 2, as an agent with another worker's stored credential does;
- ``forbidden``: refuse every forward with sshd's "administratively
  prohibited" words;
- ``tunnel``: be the reverse forward for real -- listen on the ``-R``
  remote port on 127.0.0.1 and relay each connection to the local end --
  and run the remote command with ``sh -c`` (``$HOME`` from
  ``$FAKE_SSH_HOME``), its stdin copied from this process's, as an ssh
  server would: when this process dies (a dropped session), the command
  reads the end of its input. The exit status is the command's.

Without ``-R`` (a one-shot command: stop, reclaim) ``tunnel`` mode runs
the command the same way; the other modes just exit 0.

A ``-R`` port listed in ``$FAKE_SSH_BUSY_PORTS`` (comma-separated, or
``all``) fails the way ``ExitOnForwardFailure=yes`` does, in every mode
but ``fail``. In ``hold`` mode it prints the remote script's pid line.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading


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


def _relay(listener: socket.socket, host: str, port: int) -> None:
    while True:
        try:
            client, _ = listener.accept()
        except OSError:
            return
        try:
            upstream = socket.create_connection((host, port))
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
    if "-R" not in argv:
        if mode != "tunnel":
            return 0
        return subprocess.run(["sh", "-c", argv[-1]], env=_remote_env(),
                              stdin=subprocess.DEVNULL,
                              check=False).returncode
    forward = argv[argv.index("-R") + 1]
    bind, remote_port, rest = forward.split(":", 2)
    local_host, _, local_port = rest.rpartition(":")
    if mode == "forbidden":
        print("Warning: remote port forwarding failed for listen port "
              f"{remote_port}: administratively prohibited",
              file=sys.stderr, flush=True)
        return 255
    busy = os.environ.get("FAKE_SSH_BUSY_PORTS", "").split(",")
    if remote_port in busy or "all" in busy:
        print(f"Error: remote port forwarding failed for listen port "
              f"{remote_port}", file=sys.stderr, flush=True)
        return 255
    if mode == "tunnel":
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((bind, int(remote_port)))
        except OSError:
            print(f"Error: remote port forwarding failed for listen port "
                  f"{remote_port}", file=sys.stderr, flush=True)
            return 255
        listener.listen(16)
        threading.Thread(target=_relay, daemon=True, args=(
            listener, local_host.strip("[]"), int(local_port))).start()
        child = subprocess.Popen(["sh", "-c", argv[-1]], env=_remote_env(),
                                 stdin=subprocess.PIPE)
        threading.Thread(target=_copy_stdin, args=(child,),
                         daemon=True).start()
        return child.wait()
    token = sys.stdin.readline().strip()
    with open(log + ".stdin", "a", encoding="utf-8") as handle:
        handle.write(token + "\n")
    if mode == "mismatch":
        print("the credential in /home/a/agent/127.0.0.1-18000.token is for "
              "worker 'other-box', not 'gpu-box'; to change the name, run "
              "cdui agent forget http://127.0.0.1:18000 and enroll with a new "
              "token", file=sys.stderr, flush=True)
        return 2
    print("codefyui-ssh: agent pid 4242", file=sys.stderr, flush=True)
    for line in sys.stdin:
        with open(log + ".stdin", "a", encoding="utf-8") as handle:
            handle.write(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
