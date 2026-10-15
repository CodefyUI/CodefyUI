---
sidebar_position: 3.7
title: Remote Workers
description: Run graphs on another machine — enroll a GPU server as a worker agent with a one-time token from the Workers panel, or let the server start it over SSH.
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

Without TLS, the tokens, and the inputs of every run sent to the worker, cross the network unencrypted; the server logs a warning when that happens from another machine. [Trust between agent and control plane](#trust-between-agent-and-control-plane) covers what each end checks.

## Storage: no shared filesystem

A run that reads uploads or writes files can go to a worker that shares no folder with this server. The server then acts as the file store: it sends the run's uploaded inputs to the worker and receives what the run writes. A worker that has the shared folder ([Storage: shared filesystem](./run-queue#storage-shared-filesystem)) is still preferred; this applies when the run targets a worker without one, or when `any` finds none free.

**Inputs.** The files a run can take this way are the uploads picked in a file field: CSVReader and the other data-file fields, ImageReader, ModelLoader, CheckpointLoader. At assignment the server names each one by its SHA-256 hash. The agent keeps every file it fetched in `<user-data>/agent-cache/<sha256>`, so a later run that uses the same file does not download it again; a missing one is downloaded from the server's own download routes with the worker's token, and its hash is checked. While it fetches, the Runs panel's log shows `fetching inputs 2/3: train.csv (1.2 MiB, fetched)` (a `run_warning` of kind `transfer_progress`); `cached` instead of `fetched` means it came from the cache. A Stop during the fetch ends the run `cancelled`.

**Where the run works.** Each run gets a data folder of its own on the worker, `<user-data>/agent-runs/<run id>/data`, laid out like `backend/data` (`files/`, `images/`, `models/`, `media/`), with a copy of each input under the name the graph uses (a node that writes over its input cannot touch the cache). The worker's own data folder is not touched, except that runs share its `cache/` folder (LMTokenizedDataset's token blocks). Datasets downloaded by Dataset or HuggingFaceDataset still go to the worker's own cache, as they do on the server.

**Outputs.** The agent uploads, before the run's end reaches the server:

- every file the run records as an artifact (a periodic checkpoint, a TensorBoard folder file by file), as the run reports it, into `<data folder>/runs/<run id>/<kind>/<name>`;
- after the run ends, every other file it wrote under `models/`, `media/` or `output/` (CheckpointSaver, ModelSaver, VideoWrite, ImageWriter), and any input it wrote over, into this server's own data folders at the same relative path, where the same node writes on a local run (`models/best.pt` in `CODEFYUI_MODELS_DIR`). A different file already there is kept and the new one gets a suffix (`best-1.pt`); the same content is not stored twice. They are listed with kind `output`;
- last, the checkpoint a stopped training loop writes, into the run's folder, so a cancelled run's resume point is on the server.

The uploads run beside the run, so a slow network never holds the run up; its end is reported once they are done. `GET /api/runs/{id}/artifacts` lists them with `backend: "transfer"` and their path on this server, and the Runs panel offers **Download** next to each one (`GET /api/runs/{id}/artifacts/{artifact_id}/download`). A file larger than this server's upload limit (`CODEFYUI_MAX_UPLOAD_SIZE`, 500 MB) stays on the worker, listed at its path there, with a `transfer_failed` warning. A Stop that arrives while the last uploads are going stops the ones not sent yet, with one `transfer_failed` warning saying how many stay on the worker.

**Clean-up.** The run's own folder belongs to the run: when run retention (`CODEFYUI_RUN_RETENTION_KEEP_LAST`) or **Delete** removes the run, its `runs/<run id>` folder goes too. Download or copy what you want to keep. A run that retention removes after it ended `interrupted` keeps its `checkpoint` folder, as a local run keeps its resume point. Files placed in the data folders are yours, as a local run's are: neither removes them. On the worker, the run's folder follows the agent's own run retention; the cache is never cleaned automatically, and deleting `agent-cache` is safe (files are fetched again).

**Limits.**

| Setting (on the server) | Default | Meaning |
| --- | --- | --- |
| `CODEFYUI_TRANSFER_ENABLED` | `1` | `0` turns this off: a run that uses files then needs a worker with the shared folder, as before |
| `CODEFYUI_TRANSFER_MAX_INPUT_BYTES` | 2 GiB | The most a run's inputs may add up to |
| `CODEFYUI_MAX_UPLOAD_SIZE` | 500 MB | The largest single output a worker may upload |

A run this cannot serve is refused at submit when it names a worker without the shared folder, and the message says why: a reader given a plain path (FileReader, ImageFolderDataset, ImageBatchReader, VideoLoad, DocumentLoader reading a folder) rather than an upload, an upload this server does not have, or inputs over the limit. For example: ``worker gpu-box-1 has no `data` mount; add `--mount data=...` to its agent, or make the run fit the transfer backend: its inputs add up to 3.10 GiB, over the transfer limit of 2.00 GiB (CODEFYUI_TRANSFER_MAX_INPUT_BYTES)``.

Inputs and outputs go to the same address as the agent's connection, with its worker token and the same certificate check (a pinned certificate is pinned for them too): without TLS they cross the network unencrypted.

## When the connection drops

A worker that loses its connection keeps running its runs and keeps their events until this server has them. On this server the runs show `disconnected` and keep their slots for `CODEFYUI_WORKER_RESUME_GRACE_S` (5 minutes); a reconnect within that time continues them with no gap in their logs, and after it they end `interrupted`. The outcomes, and what a sleeping laptop means for them, are in [When the connection drops](./run-queue#when-the-connection-drops).
## Trust between agent and control plane

A run sent to a worker carries its graph with the values of its secret parameters (API keys, passwords) filled in, so each end has to know who it is talking to. The control plane knows the agent by its worker credential. The agent knows the control plane by its TLS certificate, checked the same way for the enrollment request and for every `/ws/worker` connection.

CodefyUI serves no TLS itself; put it behind a TLS proxy ([Deployment](/usage/deployment)) and give the worker the `https://` address.

**A certificate the worker's CAs trust** (a public CA, or a company CA installed on the worker): nothing to do. The agent checks it like a browser does (the certificate authority and the host name) on every connection and records nothing, so a renewal, even with a new key, keeps working.

**A self-signed certificate, or a CA the worker does not have: pin its key.** `--ca-fingerprint` takes the SHA-256 of the server certificate's public key (its SubjectPublicKeyInfo, written `sha256:<64 hex digits>`), the way `kubeadm join --discovery-token-ca-cert-hash` does:

```bash
cdui agent start --control https://cp.lab --token abcdef.0123456789abcdef \
  --ca-fingerprint sha256:96d448128bb7dfbceea1a628393cfdd0f1f86716ad81deb4fab0c0fe281a0690
```

The flag keeps its name, but it pins the server's own (leaf) certificate key, never a CA's: a pin matches only the certificate whose key the TLS handshake proved, on every Python version. With a pin the agent checks neither the certificate authority nor the host name. A certificate that does not match is refused before the token is sent, with both fingerprints printed, and the agent exits with code `2`. The pin is recorded in the credential file (`<user-data>/agent/<host>-<port>.token`), so later starts keep it without the flag; a stored value that is not a valid fingerprint is reported in the log, naming the file, and ignored (the worker credential stays usable).

To get the fingerprint, run this on the proxy host, against the certificate file the proxy serves (the leaf comes first in `fullchain.pem`):

```bash
openssl x509 -in /path/to/fullchain.pem -pubkey -noout | openssl pkey -pubin -outform der | openssl dgst -sha256 | awk '{print "sha256:" $NF}'
```

The Add worker dialog shows this command when the address is `https://`. If the server can read the certificate file itself, set `CODEFYUI_TLS_CERT_FILE` to its path: the dialog then shows the fingerprint and the generated command carries `--ca-fingerprint` (the server reads the file again only when it changes). Leave it unset when the worker's CAs trust the certificate, so the command does not pin a key that a renewal will replace.

**A certificate the worker's CAs reject, without a pin**, is not used: the enrollment fails, and a running agent logs the error with its cause (the first time with the fingerprint the server presents, to compare with the dialog's and pass with `--ca-fingerprint`) and retries with its usual backoff, so a proxy being repaired or a CA bundle being updated needs no restart.

**After a pinned certificate changes its key**, check the new fingerprint on the proxy host, then either:

- run `cdui agent start --control <url> --ca-fingerprint <new fingerprint>`, which keeps the worker credential and records the new pin after it connects; or
- run `cdui agent forget <url>`, which deletes the credential with its pin, and enroll again with a new token.

A renewed certificate keeps its fingerprint only if it keeps its key (certbot's `--reuse-key`).

**Plain HTTP.** Over an `http://` address to another machine, `cdui agent start` prints one warning: the worker token and the secrets in every assignment travel unencrypted. The agent still connects. The Workers panel marks such a worker **Plain HTTP**: its connection reached the server unencrypted from another machine. The server decides from the scheme it was reached with (`wss` only when uvicorn reports it, which it does for a proxy listed in `--forwarded-allow-ips`; CodefyUI never reads `X-Forwarded-Proto` itself) and from the peer address: a loopback peer, such as a same-host agent or the local end of an SSH tunnel, carries no badge, since the hop across the network is not that socket. An `http://127.0.0.1` address prints no warning on the agent for the same reason.

Not covered: mutual TLS (the worker credential identifies the agent), a certificate authority managed by CodefyUI, and user identity.

## Keeping workers in sync {/* #keeping-workers-in-sync */}

A worker must run the same CodefyUI version as the server, and it should know the same nodes. The server checks both every time the agent connects.

**Version.** An agent of another version is refused with both versions in the reason, for example `version mismatch: worker 3.1.0, control plane 3.1.2`. The agent stops instead of retrying, and the Workers panel lists the name under **Refused connections** with that reason, so a worker that never appears says why; an agent that tried to take the name of a connected worker is listed there with its address, never on that worker's card. The version is read before anything else in the agent's first message, so even an agent whose message format changed gets this reason. Install the same version on the worker and start the agent again; the refusal is cleared when the name connects. The panel keeps the last refusal per name until the server restarts.

**Node types.** Plugins, Package Center packs and custom nodes are installed per machine. When the agent starts, a short-lived helper process loads every node the way a run does and reports the node types and a hash of their definitions (the node catalog). When the hash differs from the server's, the worker is **Degraded** and its card lists the types the worker lacks and the ones only it has. A degraded worker:

- never receives a run whose target is `any` when it lacks a type the server has (a worker that only has more types than the server stays eligible);
- receives a run only when every node type the graph uses is one it has: after presets and subgraphs are expanded, including the nodes of every installed preset a **Map** node runs. Otherwise the submission is refused with the missing types named, and a queued run is failed the same way before it is sent, in case the worker reconnected with fewer types while it waited.

To clear it, install the same plugins and packs on the worker (`cdui plugin install ...`, the Package Center, the same custom node files) and restart the agent: it reports its nodes only when it starts. A worker whose helper failed reports no catalog: its card warns about it, `any` skips it, and a run that names it is sent without the check (the agent's log says why the helper failed). A command that copies the server's plugin set to a worker (`cdui agent sync`) is planned.

**Environment fingerprint.** The same helper collects the worker's [environment fingerprint](/advanced/api-reference#run-environment) (CodefyUI commit, Python, torch build, accelerators, plugin revisions, numerics-relevant environment variables). The Workers panel shows its digest, and every run the worker executes records that digest (a worker that reconnects without one shows "not reported", never an older digest), so the Runs panel names the environment that actually ran it. The fingerprint describes the agent's environment as its run subprocesses inherit it, before a run's device set narrows `CUDA_VISIBLE_DEVICES`.

## Connecting over SSH

For a GPU server you reach with SSH and nothing else, for example a laptop behind NAT and a server that only opens port 22, the server can start the worker itself. It runs `ssh` to the GPU server, starts `cdui agent start` there, and the agent connects back through a reverse tunnel in the same SSH session (`ssh -R`). Neither machine opens a port, and the server can stay on `127.0.0.1`.

**Before you start:**

- CodefyUI is installed on the GPU server, in the same version, with `install.sh`. The server does not install it for you.
- The server's machine has an OpenSSH client on its PATH (`ssh`). macOS and Linux have one; on Windows, add the **OpenSSH Client** optional feature, and the server must be started from a session that has it on PATH.
- `ssh <target>` works from a terminal on the server's machine, as the OS user that runs `cdui start`, **without any prompt**. ssh runs with `BatchMode=yes`: no password, and a key with a passphrase only through `ssh-agent`. Your `~/.ssh/config` applies, so a `Host` alias, a `ProxyJump` and the keys it names all work.

**Adding one.** Open **Workers**, press **Add worker**, and choose **Over SSH**:

| Field | Meaning |
|---|---|
| SSH target | `user@host`, or a `Host` alias from the server account's `~/.ssh/config`. |
| Port | ssh's `-p`. Left at 22, it is not passed, so the port in your ssh config for that host applies. |
| Worker name | Optional; the target's host name by default (`alice@gpu1.lab` becomes `gpu1`). A name another worker already has is refused. |
| Identity file | Optional; a private key on the server's machine (ssh's `-i`). Empty uses your ssh config and agent. |
| cdui on the remote host | The launcher's path there; empty means `~/CodefyUI/cdui`, where `install.sh` puts it. |

Press **Add and connect**. The worker's card shows an **SSH tunnel** line: **Connecting** while ssh runs and the agent has not connected yet, **Up** once it has, **Failed** with the last lines ssh or the agent wrote (a refused key, an unknown host, a missing `cdui`) while it waits to try again, and **Stopped** when disabled. **Recent ssh output** shows more of that output.

**What runs.** For each enabled SSH worker the server keeps one process like this:

```bash
ssh -T -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
    -o StrictHostKeyChecking=accept-new \
    -R 127.0.0.1:18000:127.0.0.1:8000 -- alice@gpu-box \
    "sh -c '... ~/CodefyUI/cdui agent start --foreground --control http://127.0.0.1:18000 --name gpu-box'"
```

The remote command puts `~/.local/bin` on PATH first: a non-interactive ssh shell does not read your shell profile. The remote port comes from `CODEFYUI_SSH_REMOTE_PORT_RANGE` (`18000-18100`); when it is taken on the remote host, the next one is tried and remembered. When ssh exits, the server starts it again after 2 s, doubling up to a minute, and again after its own restart. If the remote sshd does not allow the forward at all (`AllowTcpForwarding no`, a `PermitListen` that excludes the port; an explicit refusal, or three ports failing in a row), the tunnel stays **Failed** with that reason and stops retrying; **Retry** starts it again once the server's admin has allowed `-R` to `127.0.0.1`. The tunnel line also shows the remote agent's pid: on the remote host the agent's command line is `python .../scripts/dev.py agent start --foreground ...`, so `pgrep -af 'agent start'` finds it.

**Credentials.** Each start sends a one-time enrollment token (valid 10 minutes) on ssh's standard input, never on a command line, so `ps` on either machine does not show it. The first start enrolls the worker; the agent stores its worker credential on the remote host for the address `http://127.0.0.1:<remote port>`, and every later start reconnects with it and leaves the token unused. Once the worker is enrolled, the token is bound to it, so it can only re-key that worker. The first start on a port this worker has not used before runs `cdui agent forget` for that address first, so a credential another worker or another control plane left there is never presented. If one is presented anyway (this server refuses it on that tunnel, or the agent reports a credential for another worker name), the server starts the agent again with `cdui agent forget` first, which enrolls it anew. Every tunnel arrives from this machine's own address, so enrollment attempts through a tunnel are rate-limited per tunnel (its remote port), apart from each other and from agents on this machine.

**A dropped session.** When the session ends without Disable (the laptop sleeps, the network goes), the remote agent keeps running and so do its runs; they wait `disconnected` on this server and keep their events, as in [When the connection drops](#when-the-connection-drops). The server starts ssh again with the same remote port, and the remote script, finding the agent alive, attaches to it instead of starting another (the tunnel line shows the pid it re-attached to). The agent reconnects through the new tunnel by itself and its runs continue. If the remote sshd has not noticed that the old session is gone and still holds the port, the server ends that stale session over a short separate ssh first. With no session back within `CODEFYUI_WORKER_RESUME_GRACE_S` (5 minutes), the agent stops. The remote script keeps the agent's pid and log in `~/.codefyui-ssh/<remote port>/` on the remote host.

**Disable, Enable, Delete.** **Disable** drains the worker, then ends its ssh session; a disabled worker stays drained even if an agent connects under its name later. Disable tells the remote agent to stop through the session (or, when no session is up, with a short separate ssh), so a run still on it ends `interrupted`; drain first and wait for it when that matters. **Enable** starts the session again. On a worker that has not connected yet, **Delete** ends the session and forgets the entry; on a connected one, **Remove** revokes the worker and deletes its SSH entry too.

**Security.**

- The tunnel listens on the remote host's loopback only (`-R 127.0.0.1:...`). Every user of that host can still connect to that port, so requests through it reach `/ws/worker` and `POST /api/workers/enroll`, and, with a valid worker token, the routes the transfer backend uses ([Storage: no shared filesystem](#storage-no-shared-filesystem)): the three input downloads (`GET /api/files/download/...`, `/api/images/download/...`, `/api/models/download/...`) and `POST /api/runs/{id}/artifacts/upload`. Those downloads are open reads on the LAN, but through a tunnel one without a worker token answers `403`. Every other path answers `421`.
- `StrictHostKeyChecking=accept-new` trusts a host the first time and refuses a changed key. Set `CODEFYUI_SSH_STRICT_HOST_KEY_CHECKING=yes` to require that the host is already in the server account's `known_hosts`.
- The server uses the SSH access of the OS user that runs it, as the Source Control tab does with git. Anyone who controls this server can start an SSH worker on every host that user reaches.
- The agent behind the tunnel connects from `127.0.0.1`, but it counts as a worker on this machine only when its machine id equals the server's, so a GPU server is never mistaken for the server's own machine. It authenticates with its worker token.

| Setting | Default | Meaning |
|---|---|---|
| `CODEFYUI_SSH_REMOTE_PORT_RANGE` | `18000-18100` | Ports the tunnels may take on the remote hosts' loopback |
| `CODEFYUI_SSH_STRICT_HOST_KEY_CHECKING` | `accept-new` | `yes` requires a known host |
