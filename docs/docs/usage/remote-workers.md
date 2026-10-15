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

Without TLS, the tokens, and the inputs of every run sent to the worker, cross the network unencrypted; the server logs a warning when that happens from another machine. [Trust between agent and control plane](#trust-between-agent-and-control-plane) covers what each end checks.

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
