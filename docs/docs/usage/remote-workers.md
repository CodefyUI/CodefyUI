---
sidebar_position: 3.7
title: Remote Workers
description: Run graphs on another machine — enroll a GPU server as a worker agent with a one-time token from the Workers panel.
---

# Remote Workers

A worker agent (`cdui agent start`) runs graphs for a CodefyUI server, the control plane, each run in a subprocess of its own. It can run on the same machine as the server ([CLI Commands, Agent commands](/getting-started/cli-commands#agent-commands)) or on another machine, which this page covers. The agent always connects out to the server; the worker machine opens no port. Runs reach a worker through their target ([Run Queue, Workers](/usage/run-queue#workers)).

## Adding a worker

1. **Make the server reachable.** By default `cdui start` listens on `127.0.0.1`, which no other machine can reach. Start it with `cdui start --host 0.0.0.0`. The server then accepts its own interface addresses (such as `192.168.1.5:8000`) as a `Host`; if the worker dials a hostname instead (`gpu-gw.lan:8000`, a VPN name), add it to `CODEFYUI_EXTRA_ALLOWED_HOSTS`. Anyone who can reach that port controls the server: use a trusted network, or put the server behind a TLS proxy ([Deployment](/usage/deployment)).
2. **Generate a token.** Open **Workers** (Settings, Execution, or the Runs panel header), press **Add worker**, and stay on **Another machine**. The address field starts from the address the page uses; when that is `127.0.0.1` or `localhost`, pick one of the server's other addresses or type the one the worker reaches. Choose how long the token is valid (24 hours by default) and how many agents may use it (1 by default), then press **Generate token**.
3. **Run the command on the worker**, in a CodefyUI install of the same version:

   ```bash
   cdui agent start --control http://192.168.1.5:8000 --token abcdef.0123456789abcdef
   ```

   The agent trades the enrollment token for a worker credential of its own and stores it in `<user-data>/agent/<host>-<port>.token` (readable by that OS user only). The worker appears in the Workers panel within a few seconds. `--name` sets the worker name (default: the hostname).

The enrollment token is shown once; the server keeps only its hash. A used-up, expired or revoked token is refused with a message that says which. The tokens issued are listed in the dialog until you delete them.

**Restarts.** `cdui agent start --control <url>` finds the stored credential and reconnects with it, without a token. An enrolled worker is remembered by the server across its own restarts: it is listed as `offline` until it connects, and runs that target it wait in the queue. `cdui agent forget <url>` deletes the stored credential, so the next start enrolls again.

**Removing a worker.** **Remove** in the Workers panel revokes the worker's credential, disconnects it (its running runs end `interrupted`, its queued runs fail), and drops it from the list; its agent is refused from then on.

**Re-keying a worker.** A name is taken once enrolled: an ordinary token cannot enroll it again (the server answers `409`). To give a worker a new credential, for a reinstalled machine or a lost credential file, press **Re-key** on its card: the token generated there can enroll that worker only, and the command carries its `--name`. Using it invalidates the old credential. If the machine still has the old credential file, run `cdui agent forget <url>` there first.

Without TLS, the tokens, and the inputs of every run sent to the worker, cross the network unencrypted; the server logs a warning when that happens from another machine.

## When the connection drops

A worker that loses its connection keeps running its runs and keeps their events until this server has them. On this server the runs show `disconnected` and keep their slots for `CODEFYUI_WORKER_RESUME_GRACE_S` (5 minutes); a reconnect within that time continues them with no gap in their logs, and after it they end `interrupted`. The outcomes, and what a sleeping laptop means for them, are in [When the connection drops](./run-queue#when-the-connection-drops).
