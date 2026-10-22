---
sidebar_position: 6.6
title: Execution Design Notes
description: Why runs execute the way they do when a server can hand them to worker agents — the goals, the six invariants and the reason for each, the decisions with the alternatives they ruled out, the dated decision log, and the questions still open.
---

# Execution Design Notes

[Execution Architecture](./execution-architecture) describes how a run executes in the server process or on a worker agent. This page records why it was built that way: the goals of the control-plane / worker split ([#764](https://github.com/CodefyUI/CodefyUI/issues/764)), the invariants every change keeps, the decisions and the alternatives they ruled out, and the questions that are still open. Each decision links to the page that documents its behaviour; the how-to lives in [Multi-GPU Execution](/usage/multi-gpu), [Remote Workers](/usage/remote-workers), [Run Queue](/usage/run-queue) and [Studies](/usage/studies).

## Goals {/* #goals */}

Three situations drive the design:

| Situation | Need |
| --- | --- |
| Remote execution | `cdui start` on a laptop or a server is the control plane; each GPU server runs `cdui agent start`; a run picks its target in the web UI. |
| One machine with several GPUs | A run picks a set of cards; runs may share a card; a run that cannot fit is refused with a reason before it starts; one run never breaks another. |
| One machine, one process | The default install, the canvas and the exported Python script behave exactly as before the split. |

Out of scope: user accounts and multi-tenancy (the backend has no user concept, [#247](https://github.com/CodefyUI/CodefyUI/issues/247)), training across machines, and sharded (FSDP) or model-parallel training.

## Invariants {/* #invariants */}

Every change in this area keeps these. A change that cannot is stopped and reviewed on its own.

1. **With no worker enrolled and no `target` option, submission behaves exactly as 3.0.2**: the same queue key, the same `InProcessExecutor`, the same event sequence. *Why:* the single-machine path is what most installs and every classroom use, and it is the path the December exam runs on. The `single-node-parity` CI job compares the event sequence of default submissions with a stored snapshot, and the snapshot changes only in a commit that means to change it.
2. **The export surface is frozen**: `backend/app/core/codegen.py`, `backend/run_graph.py`, node `execute()` semantics, and the runtime symbols exported scripts import (`graph_engine.invoke_node`, `prepare_executable_graph`, `topological_sort`; the `ExecutionContext` constructor with `derive_seed` and `current_node_id`; `device_utils.resolve_device`; `seeding.*`, `runtime.*`, `api_contract.*`). *Why:* an exported script imports these from the installed backend at run time, and an online judge checks its output, so a change to any of them changes graded results. The run subprocess constructs an `ExecutionContext` without changing its interface, and device sets are made with `CUDA_VISIBLE_DEVICES` so that `resolve_device` stays as it is.
3. **The interactive (canvas) lane always executes in the control-plane process.** *Why:* a canvas run uses state that lives only there: the socket's execution cache, persistent module weights and the outputs the Inspector reads ([Decisions, The canvas stays local](#interactive-lane-local)).
4. **A worker never opens an inbound port and never writes SQLite.** *Why:* GPU servers sit behind NAT and firewalls, and a listening worker would need its own authentication; the database has one writer, the control plane ([Decisions, Workers dial out](#workers-dial-out)).
5. **New behaviour is off by default or sits behind an explicit option.** *Why:* invariant 1 holds only if nothing new switches itself on. In the code the switches are the `target` option, the `provenance` run option, a card's **Shared by memory** policy (cards are **Exclusive** by default), `TrainingLoop`'s `strategy` (`single` by default), `CODEFYUI_S3_ENABLED` (off) and the agent's own flags (`--one-shot`, `--mount`).
6. **Web UI first.** Workers are managed in the browser; the CLI gains only `cdui agent start`, `stop` and `forget`, and no `cdui status` or TUI feature is added. *Why:* the browser works the same on every platform the server runs on, and terminal UIs are hard to keep compatible across them.

## Decisions {/* #decisions */}

### The control plane stays in the server process {/* #control-plane-in-the-server */}

**Decision.** The `cdui start` process keeps the queue, the scheduler, the run rows, the event log and the worker registry. Only the execution of a queued run moves out, through one seam: `RunService._execute` hands an `ExecutionRequest` to an `Executor` (`InProcessExecutor` or `WorkerExecutor`), and both report through the same sinks into the same event log.

**Why.** The run service already owned every run (submit, durable events, cancel, restart recovery), and its event vocabulary already was a wire format, so a worker's events can take the same path an in-process run's do. The browser and `cdui run` needed no change.

**Rejected.** A separate scheduler service with a broker (Redis, as n8n's queue mode uses), or Ray or Celery as the execution backend: each adds a service to install and run, and Ray couples to torch versions and supports Windows poorly. Several uvicorn workers (`cdui start -- --workers N`): each would be a second server on the same database, which the instance lock refuses ([#529](https://github.com/CodefyUI/CodefyUI/issues/529)). The scale is a few GPU servers and tens of runs, which one SQLite writer and one event loop serve.

**Documented in** [Execution Architecture, Roles](./execution-architecture#roles) and [Event path](./execution-architecture#event-path).

### The canvas stays local {/* #interactive-lane-local */}

**Decision.** **Run** on the canvas executes in the control-plane process and is never placed on a worker; a submission that combines the interactive lane with a worker target is refused. **Queue on...** is the way to send the tab's graph to a worker.

**Why.** The canvas reuses the socket's execution cache and persistent weights between clicks and records outputs for the Inspector, all held in that process. A queued run uses none of them, so it moves to a worker without losing anything. Recorded outputs of a worker run came later ([#789](https://github.com/CodefyUI/CodefyUI/issues/789)): they stay on the worker and are read over its connection.

**Rejected.** Sending canvas runs to workers: it would need the cache and the weights store to live on the worker and be addressed from the canvas, for no gain on the short runs the canvas is for.

**Documented in** [Execution Architecture, Lanes](./execution-architecture#lanes-and-where-they-execute) and [Multi-GPU Execution, What the canvas does](/usage/multi-gpu#the-canvas).

### Workers dial out {/* #workers-dial-out */}

**Decision.** The agent opens `/ws/worker` to the control plane and keeps it open; assignments, cancels, drains and output reads travel back over that connection. The control plane never connects to a worker. A machine the control plane cannot be reached from directly is served by an SSH session the control plane starts (`ssh -R` to a Unix socket on the remote host). A run subprocess never receives `CODEFYUI_DB_PATH`, and only the control plane writes the database.

**Why.** Lab GPU servers are often behind NAT or a firewall with only SSH open, while the control plane needs only one reachable URL. The worker reuses the control plane's Host check and authentication, and the same code serves a worker on the same machine and one elsewhere.

**Rejected.** Push, where the control plane calls an HTTP API on each worker (ComfyUI-Distributed, ComfyUI_NetDist): every worker would open a port and authenticate callers, and the control plane would need to know every address. A shared secret in an environment variable on every worker: enrollment issues a credential per machine instead, so one can be revoked alone.

**Documented in** [Execution Architecture, Security](./execution-architecture#security), [Remote Workers, Reaching the control plane](/usage/remote-workers#reaching-the-control-plane) and [Adding a worker](/usage/remote-workers#adding-a-worker).

### Same version on both ends {/* #same-version */}

**Decision.** A worker of another CodefyUI version is refused at the handshake, with both versions in the reason. A worker whose node catalog differs is **Degraded**: it takes only graphs whose node types it has, and `any` skips it when it lacks a type the control plane has.

**Why.** The two ends share one protocol module and one event vocabulary, and a run's graph must mean the same thing on both. Refusing a version is simpler to reason about than negotiating one; the protocol still grows by addition, with capabilities, so optional features can be checked per agent.

**Rejected.** Running mixed versions behind protocol negotiation. A command that copies the control plane's plugins to a worker is not built yet ([Open questions](#open-questions)).

**Documented in** [Remote Workers, Keeping workers in sync](/usage/remote-workers#keeping-workers-in-sync) and [Execution Architecture, Security](./execution-architecture#security).

### One subprocess per run {/* #one-subprocess-per-run */}

**Decision.** The agent starts each run in a fresh interpreter from a `multiprocessing` *spawn* context. The subprocess is the only place on a worker that imports PyTorch; its cards are set with `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES`, renumbered from `cuda:0` in the order the run lists them. The agent itself reads device facts through NVML without torch.

**Why.** In the server process every run shares one Python process and one CUDA context, so a segfault, an out-of-memory kill or a corrupted context ends every run and the UI with it. A subprocess confines that to one run, can be terminated after a grace period (a thread cannot be killed without risking the device), gives each run its own random state, so seeded runs on different cards run at the same time, and lets a memory cap apply to one run. *spawn* is required on Windows and after CUDA has been initialised. The agent never holds a CUDA context of its own.

**Rejected.** Threads in one process, which is what the server still does for its own runs and why its runs share one seed gate and report peak memory as `exact: false`. Containers per run: stronger isolation, at the cost of GPU passthrough setup and no Windows story. A pool of pre-started interpreters is not built: each run pays its own torch import.

**Documented in** [Execution Architecture, Isolation](./execution-architecture#isolation).

### Admission: a count per slot, a memory budget on a shared card {/* #memory-admission */}

**Decision.** A slot is one device on one machine, `(worker, device)`, and admits a fixed number of runs (1 on an accelerator, 2 on the CPU, overridable per slot). A worker's CUDA card set to **Shared by memory** admits by budget instead: a run declares `target.memory_gib` (or gets the last measured peak of the same graph on the same number of cards, plus 20%), it fits while its budget plus `CODEFYUI_EXECUTION_CUDA_CONTEXT_MIB` is at most `min(free - reserved, total - reserved - running)`, and its subprocess caps its allocator to the budget (`torch.cuda.set_per_process_memory_fraction`) before it reports `run_started`. A run for a named worker that does not fit is refused at once with the numbers, unless `target.wait` is true; an `any` run waits. Slots that are one physical device (matched by NVML UUID, or the CPU and `mps:0` of one machine) count as one budget under the smallest limit among their offerers ([#842](https://github.com/CodefyUI/CodefyUI/issues/842)).

**Why.** "Runs may share a card; a run that cannot fit is refused" needs a decision before the run starts, and the cap is what makes sharing safe: a run that outgrows its budget runs out of memory alone. The ledger counts budgets because per-run use is not visible in a heartbeat. The second term of the headroom keeps promises within the card while runs that started a moment ago hold little yet. Counting stays the default because it is the 3.0 behaviour and needs no budget from the user.

**Rejected.** Memory admission in the server process: its runs share one process, so nothing can hold one of them to a budget, and `target.memory_gib` with `local` is refused. A static estimate from the graph's parameters as the server's default: the Queue on... dialog offers it, marked as an estimate, and the server applies only measured peaks on its own. Admitting by `free - reserved` alone, the form first specified in [#788](https://github.com/CodefyUI/CodefyUI/issues/788): it admits a third run beside two that have not allocated yet.

**Documented in** [Multi-GPU Execution, Slots and limits](/usage/multi-gpu#slots-and-limits), [Sharing a card](/usage/multi-gpu#sharing-a-card) and [Execution Architecture, Scheduling](./execution-architecture#scheduling).

### Storage: mounts first, then transfer {/* #storage */}

**Decision.** A run that uses files in the data folder goes to a worker that holds the same mount label (`data`), with paths translated per machine; failing that, when every file it uses is an upload within the transfer limit, the control plane acts as the file store (the transfer backend), or an S3 / MinIO bucket does when it is configured. A run that fits neither is refused for a named worker and skipped by `any`.

**Why.** Labs usually have a NAS, and a shared folder moves no bytes. Machines without one still need to work. The model is Kubernetes volumes (a pod is placed where its volume is) and Argo's artifact repository (steps exchange files through a store, often MinIO).

**Rejected.** Moving every file through the control plane for every worker, which would make the control plane's link the bottleneck even where a NAS exists. Treating artifact rows as plain paths, which would name files on another machine: rows record a backend, a mount label and a relative path.

**Documented in** [Remote Workers, Storage](/usage/remote-workers#storage) and [Execution Architecture, Storage](./execution-architecture#storage).

### A disconnect keeps the run {/* #disconnects */}

**Decision.** A closed connection, or three missed heartbeats, marks the worker and its runs `disconnected`. The agent spools every frame until the control plane acknowledges it, a reconnect within `CODEFYUI_WORKER_RESUME_GRACE_S` (300 s) replays the rest, and a control plane that restarted without stopping its runs adopts the worker runs it left behind ([#793](https://github.com/CodefyUI/CodefyUI/issues/793)).

**Why.** A laptop as control plane sleeps and changes networks, while the run subprocess on the worker does not depend on the socket. The grace period follows the Kubernetes node controller, which waits minutes before evicting a node that went silent.

**Rejected.** Cancelling a run when its connection drops, which would make every network blip cost the run.

**Documented in** [Remote Workers, When the connection drops](/usage/remote-workers#when-the-connection-drops) and [Execution Architecture, Isolation](./execution-architecture#isolation).

### One-shot workers for batch systems {/* #one-shot-workers */}

**Decision.** `cdui agent start --one-shot` takes exactly one run, reports it and exits with a code that says how the run ended. `any` never places a run on one; a run reaches it by naming it or the one-shot pool (`target.pool: "one-shot"`), and waits for one for up to `CODEFYUI_WORKER_ONE_SHOT_QUEUE_TIMEOUT_S` (24 hours). `--instance` lets several jobs, such as the tasks of a SLURM array, share one credential.

**Why.** On a cluster the batch system decides which machine and which GPU a job gets, and for how long. A one-shot agent inside the job leaves that decision to SLURM, while CodefyUI sends the graph and records the result. This follows the GitHub runner started with `--ephemeral`.

**Rejected.** Submitting batch jobs from the control plane (SLURM as an execution backend): CodefyUI would have to model each cluster's partitions, accounts and limits. Placing `any` runs on one-shot agents: an agent that exits after one run would take runs the queue meant for a long-lived worker, and its recorded outputs could never be read, so `record_outputs` is refused for it.

**Documented in** [Remote Workers, One-shot workers](/usage/remote-workers#one-shot-workers) and [SLURM](/usage/remote-workers#slurm).

### DDP inside the run subprocess {/* #ddp */}

**Decision.** `TrainingLoop` with `strategy: ddp` on a worker run with two or more CUDA cards trains one model across them ([#792](https://github.com/CodefyUI/CodefyUI/issues/792)). The run subprocess starts one rank per card; the ranks meet through a `torch.distributed.FileStore` in the run's directory and talk over the loopback interface only (`GLOO_SOCKET_IFNAME` and `NCCL_SOCKET_IFNAME`). Rank 0 executes the whole graph and owns the pipe to the agent; the other ranks execute only the loop and what feeds it. Everywhere else (the canvas, the server process, one card, an exported script) the loop trains in one process with a warning.

**Why.** To the agent, the protocol and the Runs panel a DDP run stays one run with one event log. A file rendezvous needs no listening socket, so the worker still opens nothing another machine could reach (invariant 4), and no free port has to be found and held. The exported script keeps the single-process path, so the export surface is untouched (invariant 2).

**Rejected.** A TCP rendezvous on a free loopback port, the form first specified in #792: a listening socket and a port race for no benefit on one machine. Training across machines. DDP in the server process, which would put several ranks into the process that serves the UI.

**Documented in** [Multi-GPU Execution, Train one model on several cards](/usage/multi-gpu#ddp) and [Execution Architecture, Isolation](./execution-architecture#isolation).

### Fingerprint and provenance are separate records {/* #fingerprint-and-provenance */}

**Decision.** The environment fingerprint records which code and environment executed a run: CodefyUI version and commit, Python, the torch build, accelerators, plugin revisions and content digests, and a fixed list of environment variables, recorded when the run starts. For a worker run, the run subprocess reports its own fingerprint, which replaces the agent's (`execution_facts.fingerprint_source`). Node provenance, which dataset files and model weights the graph read, is recorded only with the run option `provenance` (`identity` or `content`), one record per reading node ([#344](https://github.com/CodefyUI/CodefyUI/issues/344)).

**Why.** The fingerprint is cheap and the same for every run on one environment, so equal digests mean equal environments. Data identity costs file reads that grow with the dataset, so it is opt-in, and a stat identity (path, size, modification time) is labelled apart from a sha256 content hash so that the two are never compared. The subprocess reports its own fingerprint because a worker's checkout or plugins can change after its agent started.

**Rejected.** Folding data identity into the fingerprint, which would make the environment digest change with every dataset and cost a hash on every run. Recording the fingerprint at submission: a queued run can wait through a server update or execute on a worker, so the environment that queued it may differ from the one that ran it.

**Documented in** [API Reference, Run environment](./api-reference#run-environment) and [Node provenance](./api-reference#node-provenance).

### Studies with budgets the server enforces {/* #studies */}

**Decision.** A study (variants x repetitions on one exact objective) lives in the database and is driven by the server's study coordinator through the ordinary run queue. Its budgets (`max_runs`, `max_parallel_runs`, `max_run_seconds`, `max_active_seconds`) are enforced by the server whether or not anyone watches; a used-up budget halts the study with every result kept, and a resume with a raised budget continues it ([#343](https://github.com/CodefyUI/CodefyUI/issues/343)). `options.target` reaches every child, so children placed on workers run in parallel.

**Why.** An experiment held in a browser tab ended when the tab closed. A budget only the client checked would stop applying the moment the client went away. A variant is ranked only when every scheduled repetition succeeded, a stricter rule than a sweep's, because a study's ranking may promote a variant.

**Rejected.** Extending sweeps with repetitions: a sweep ranks a failed run on the value it reached and keeps no per-repetition attempts, which recovery after a restart needs.

**Documented in** [Studies](/usage/studies) and [Run Queue, Studies](/usage/run-queue#studies).

### Plugin jobs run core code only {/* #plugin-jobs */}

**Decision.** A plugin submits durable work through `/api/plugins/{plugin_id}/jobs`, scoped to its id, with a required idempotency key. Core registers the only kind, `study`, whose spec is a `POST /api/studies` body; a job is a study with the plugin as `owner`. Plugin-supplied server handlers (the Graph Copilot contract's `AgentJobHandler`) are not loaded.

**Why.** What Graph Copilot executes after a plan is approved is a deterministic study, which needs no plugin code on the server. The planning and the model conversation stay in the plugin's page.

**Rejected.** Loading a plugin's own long-lived handler in the control plane, which needs a trust prompt and a sandbox that do not exist yet. The registry (`register_job_kind`) is where a later kind attaches.

**Documented in** [API Reference, Plugin jobs](./api-reference#plugin-jobs) and [Studies, Studies from plugins](/usage/studies#studies-from-plugins).

## Decision log {/* #decisions-log */}

| Date | Decision |
|------|----------|
| 2026-10-15 | Split execution from the control plane; keep the in-process path as the default and the only single-node path. |
| 2026-10-15 | Web UI first: the Workers panel, the Add-worker dialog and the run target selector live in the browser; the CLI keeps only start, join and stop. |
| 2026-10-15 | Topology: any machine runs `cdui start` as the control plane; GPU machines run `cdui agent start`, joining with a one-line command the browser generates. Connections always go worker → control plane. |
| 2026-10-15 | Admission stays count-based per `(worker, device)` slot; memory-based admission is deferred. Superseded on 2026-10-18 by the per-card **Shared by memory** policy ([#788](https://github.com/CodefyUI/CodefyUI/issues/788)). |
| 2026-10-15 | Storage: a NAS (mounts) is the default; a no-NAS transfer backend is required too. |
| 2026-10-15 | The interactive lane stays in the control plane; worker credentials come from enrollment; Linux workers first, Windows best effort; control plane and worker run the same version; DDP later ([#792](https://github.com/CodefyUI/CodefyUI/issues/792)). |
| 2026-10-15 | One spawned subprocess per run, with the device set as `CUDA_VISIBLE_DEVICES` ([#771](https://github.com/CodefyUI/CodefyUI/issues/771)); the agent process never imports torch. |
| 2026-10-15 | Same-host workers are detected by machine id, and shared cards by NVML UUID, never by peer address ([#773](https://github.com/CodefyUI/CodefyUI/issues/773)). |
| 2026-10-15 | Until enrollment exists, `/ws/worker` accepts loopback peers with the session token only ([#772](https://github.com/CodefyUI/CodefyUI/issues/772)). Superseded by enrollment ([#780](https://github.com/CodefyUI/CodefyUI/issues/780)): the session token stays loopback-only, and a worker token is accepted from any peer. |
| 2026-10-15 | Enrollment: one-time tokens traded for per-worker credentials; the session token stays loopback-only; identity is bound to the credential, an enrolled name changes hands only through a re-key token, and same-host needs a loopback peer and a matching machine id ([#780](https://github.com/CodefyUI/CodefyUI/issues/780)). |
| 2026-10-15 | One submit rule for single runs, sweeps and studies: only a worker name never seen since startup is refused; a run for a known worker that is away waits for it ([#777](https://github.com/CodefyUI/CodefyUI/issues/777)). |
| 2026-10-18 | One physical device offered by several slots is one budget: slots grouped by UUID (and a same-host worker's CPU / `mps:0` with the server's) instead of aliased, the smallest limit among the online offerers, one memory ledger, the stricter policy ([#842](https://github.com/CodefyUI/CodefyUI/issues/842)). |
| 2026-10-21 | DDP is a `TrainingLoop` strategy inside the run subprocess: one rank per card of a worker run with two or more CUDA cards, rank 0 owns the pipe to the agent, the other ranks run only what the loop needs; everywhere else (canvas, server process, exported script) the loop trains in one process with a warning. No multi-node training ([#792](https://github.com/CodefyUI/CodefyUI/issues/792)). |
| 2026-10-21 | Studies are server-owned with server-enforced budgets; plugins submit them as plugin-scoped jobs of kind `study`, and plugin server code is not loaded ([#343](https://github.com/CodefyUI/CodefyUI/issues/343)). |
| 2026-10-22 | The environment fingerprint describes the process that executed a run (on a worker, its run subprocess); data and model identity is a separate, opt-in node provenance record ([#344](https://github.com/CodefyUI/CodefyUI/issues/344)). |

## Open questions {/* #open-questions */}

These remain open in the code as it stands:

- **Open: how far Windows workers are supported.** Workers are developed and tested on Linux, macOS works, and Windows is best effort. `cdui agent start --foreground` runs on Windows, but only a systemd unit is documented for running the agent as a service ([Remote Workers, Running the agent as a service](/usage/remote-workers#running-the-agent-as-a-service)).
- **Open: several storage labels per worker.** An agent may declare several labels (`--mount` repeats), but placement checks one: the control plane's own label, `CODEFYUI_MOUNT_LABEL` (`data`). A worker with a NAS and a local SSD cannot yet have a run placed by the second one ([Remote Workers, Shared filesystem](/usage/remote-workers#storage-shared-filesystem)).
- **Open: keeping plugins in sync.** Plugins and packs are installed on each worker by hand, and a degraded worker says which node types it lacks; no command copies the control plane's plugin set to a worker ([Remote Workers, Keeping workers in sync](/usage/remote-workers#keeping-workers-in-sync)).

The other questions the design started with are settled in the code: enrollment tokens last 24 hours and allow one use by default, both set per token in the Add worker dialog; a disconnected worker's runs wait `CODEFYUI_WORKER_RESUME_GRACE_S` (300 s), and a restarted control plane adopts them; and every Wave 12 item (memory admission, recorded outputs on workers, one-shot workers, S3, DDP, adoption, memory estimates) has landed.
