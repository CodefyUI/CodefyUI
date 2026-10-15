---
sidebar_position: 6.5
title: Execution Architecture
description: How a run gets from submission to a result when it can execute in the server process or on a worker agent — the control plane, executors, worker agents and run subprocesses, slots and placement, the event path, isolation, storage, security, and the invariants every change keeps.
---

# Execution Architecture

[Architecture](./architecture) describes the server as a whole. This page covers one part of it in depth: where a run's graph executes. By default every run executes inside the server process, exactly as in 3.0. A queued run can instead be sent to a **worker agent** (`cdui agent start`), which executes it in a subprocess of its own. The design is tracked in [#764](https://github.com/CodefyUI/CodefyUI/issues/764); the usage side is [Run Queue, Workers](/usage/run-queue#workers) and [CLI Commands, Agent commands](/getting-started/cli-commands#agent-commands).

:::note Status
In the code today: the executor seam, placement by `(worker, device)` slot, the `target` option (for single runs, sweeps and studies), worker agents on the **same machine** as the server, device sets, peak memory per run, the Workers panel, and the toolbar's **Queue on...** dialog with the Runs panel's Worker and Devices columns. Planned and not yet in the code: enrollment tokens and remote workers ([#780](https://github.com/CodefyUI/CodefyUI/issues/780)), handshake checks of version and node catalog ([#781](https://github.com/CodefyUI/CodefyUI/issues/781)), storage mounts ([#782](https://github.com/CodefyUI/CodefyUI/issues/782)) and the transfer backend ([#785](https://github.com/CodefyUI/CodefyUI/issues/785)), a spool that lets a run survive a dropped connection ([#783](https://github.com/CodefyUI/CodefyUI/issues/783)), and control-plane fingerprint pinning ([#786](https://github.com/CodefyUI/CodefyUI/issues/786)). Sections below say which parts are planned.
:::

## Vocabulary

| Term | Meaning |
|------|---------|
| **Control plane** | The `cdui start` server process: UI, REST, `/ws/execution`, SQLite, the run service and the worker registry. One process. |
| **Executor** | The object that executes one run's graph and reports how it ended (`core/executors.py`). `InProcessExecutor` runs it in the control plane; `WorkerExecutor` hands it to a worker. |
| **Worker** (worker agent) | A `cdui agent start` process. It dials the control plane, receives assignments and starts one run subprocess per run. It never imports PyTorch itself. |
| **Run subprocess** | A fresh `spawn` interpreter the worker starts for one run. It is the only place on a worker that imports PyTorch and initialises CUDA. |
| **Slot** | One device on one machine, keyed `(worker_id, device)`: `("local", "cuda:0")` is the server's first card, `("gpu-box-1", "cuda:0")` the first card of the worker `gpu-box-1`. A slot admits a fixed number of runs at once. |
| **Placement** | The scheduler's answer for a waiting run: the worker, the devices it executes on, and the slots it holds. |
| **Target** | The `options.target` object a submission carries: `{"worker", "devices"}`. |
| **Device set** | A `target.devices` list of more than one device of one kind. |
| **Mount**, **transfer backend** | Planned storage terms ([Storage](#storage)). |

## Roles

**Control plane.** The `cdui start` server process owns everything durable and everything the canvas needs. `RunService` (`core/run_service.py`) accepts submissions, keeps one FIFO per queue key, asks a scheduler which run may start, writes the run row and its event log, and fans events out to `/ws/execution` and the REST readers. `WorkerRegistry` (`core/workers.py`) holds the connected workers in memory: their devices, heartbeats, status, the runs each one is executing and the socket to reach it. A restarted control plane starts with no workers, and each agent registers again on its next dial.

**Executors.** `RunService._execute` builds one `ExecutionRequest` (graph, options, `ExecutionContext`, placement, sinks) and awaits `Executor.run`, which returns an `ExecutionOutcome` (status, error, peak memory). The placement decides which executor runs it:

- `InProcessExecutor` calls `execute_graph` in the control-plane process. It is the pre-#766 `_execute` body, unchanged, and it executes every run that has no target or targets `local`.
- `WorkerExecutor` sends an `assign` frame to the placed worker and turns the frames that come back into the same sink calls `InProcessExecutor` makes. A stop on the run is forwarded as `cancel`.

**Worker agent.** `cdui agent start` (`backend/app/agent/`) is a small asyncio process. `connection.py` dials `/ws/worker`, says `hello`, sends a heartbeat every `welcome.heartbeat_s` seconds (10 by default) and dispatches `assign`, `cancel` and `drain`. `runner.py` starts and supervises the run subprocesses. The agent reads device facts through NVML without importing torch (`core/resources.py` in torch-free mode), so it cannot hold a CUDA context of its own.

**Run subprocess.** `child.py` is the subprocess entry point. It sets the environment the agent chose (`CUDA_VISIBLE_DEVICES` and the rest, [Isolation](#isolation)) before anything imports torch, builds the `ExecutionContext` the way `RunService._start` does, and runs `InProcessExecutor` with sinks that write to a pipe. The agent reads that pipe on a thread and forwards each message as a protocol frame.

```
Browser / cdui run ── REST, /ws/execution ──► control plane (cdui start)
                                               RunService · scheduler · WorkerRegistry · SQLite
                                               InProcessExecutor (default) · WorkerExecutor
                                                          ▲
                                                          │ WS /ws/worker  (the agent dials out)
                                                          │
                                               cdui agent (same machine, this release)
                                               one spawn subprocess per run
                                               CUDA_VISIBLE_DEVICES = the run's cards
```

## Lanes and where they execute

A run belongs to one of two lanes ([Run Queue, Lanes](/usage/run-queue#lanes)):

- **Interactive** (the canvas). It always executes in the control-plane process, because it uses state that lives there: the socket's execution cache, persistent module weights and captured outputs for the Inspector. A submission that combines the interactive lane with a worker target is refused at submit.
- **Queued** (`POST /api/runs`, `cdui run`, the toolbar's **Queue on...** dialog, sweeps, studies). It executes in the control plane by default. With `options.target.worker` set to a worker name, or to `any` and placed on a worker, it executes in a run subprocess on that worker. A sweep or study passes its `options.target` to every child, so `any` spreads the children over free slots. `record_outputs` is refused for worker runs until [#789](https://github.com/CodefyUI/CodefyUI/issues/789); target `local` to capture outputs.

A run without `target`, or with `"worker": "local"`, never meets the worker code path: the same queue key, the same `InProcessExecutor` and the same event sequence as 3.0.2, which the `single-node-parity` CI job checks against a stored snapshot.

## Event path

Events are built once and stored once, wherever the graph executes.

1. **Engine callbacks.** `execute_graph` reports progress, block-internal status and node signals (metrics, artifacts, warnings) through three callbacks.
2. **Pure builders.** `core/run_events.py` turns callback arguments into event payloads (`node_status`, `inner_node_status`, metric points, signal frames). The functions have no I/O and no reference to `RunService`, so a run subprocess imports them and produces the same bytes.
3. **Sinks.** `RunSinks` carries four callables: `emit` (into `RunService._emit`, which caps the payload with `CODEFYUI_RUN_EVENT_PAYLOAD_CAP_BYTES` and appends it to `exec_run_events`), `record_metrics` (the batcher behind `exec_run_metrics`), `record_artifact` (`exec_run_artifacts`) and `warn` (a `run_warning` event).
4. **Durable log.** The event log is the record. `/ws/execution` and `GET /api/runs/{id}/events` are views over it; a client that reconnects replays from its cursor.

For a run on a worker, steps 1 and 2 happen in the run subprocess. Its sinks write `("event", ...)`, `("metrics", ...)`, `("artifact", ...)`, `("warning", ...)` and finally `("finished", ...)` to a pipe; the agent forwards each as a frame over `/ws/worker`, numbered with a per-run `seq`; `WorkerExecutor` hands each frame to the control plane's real sinks. Only the node-level vocabulary travels: `execution_start` and the terminal event (`execution_complete`, `execution_error`, `execution_stopped`) stay the control plane's, as they are for an in-process run. The run subprocess caps payloads with the control plane's own policy, and the control plane applies it again, which changes nothing.

The frames are defined once, in `core/worker_protocol.py`, imported by both ends:

| Direction | Frames |
|-----------|--------|
| agent → control plane | `hello` (name, version, protocol version, environment fingerprint, node catalog hash, machine id, devices, mounts, platform), `heartbeat` (devices with free memory, running run ids), `run_alive` (after a reconnect), and per run `run_started`, `event`, `metrics`, `artifact`, `warning`, `run_finished` |
| control plane → agent | `welcome` (worker id, `heartbeat_s`, `grace_s`), `reject` (reason, then close), `assign` (run id, graph, options, device, devices, optional `grace_s`), `cancel` (run id, optional `grace_s`), `drain` |

## Scheduling

**Queue keys and slots.** Every waiting run sits in one FIFO keyed by a string. A local run's key is its resolved device in canonical form (`cpu`, `cuda:0`, `mps:0`); a worker run's key is `<worker>/<device as asked>` (`gpu-box-1/cuda`, `any/cpu`). The `/` cannot occur in a device string, so the two families never collide. When a slot frees, `RunService.reschedule` offers the front of each FIFO to the scheduler (`core/scheduling.py`), oldest submission first across all FIFOs, one start at a time; the scheduler answers with a `Placement` or `None` (the run stays first in line). Several FIFOs can share one slot (`cuda:0`, `any/cuda` and a same-host worker's `<worker>/cuda:0`), and serving the oldest front first keeps one long queue from starving the others. Cancelling the run at the front of a FIFO offers the next one at once.

**Admission is a count per slot** ([#768](https://github.com/CodefyUI/CodefyUI/issues/768), [#773](https://github.com/CodefyUI/CodefyUI/issues/773)). A slot admits `CODEFYUI_RUN_QUEUE_MAX_CONCURRENT_GPU` runs (1) for an accelerator and `CODEFYUI_RUN_QUEUE_MAX_CONCURRENT_CPU` (2) for `cpu`. `CODEFYUI_RUN_QUEUE_MAX_CONCURRENT` overrides single slots: a bare `cuda:0=2` is the server's own card, `gpu-box-1/cuda:0=3` is that worker's. No admission decision reads free memory yet; memory-based admission is [#788](https://github.com/CodefyUI/CodefyUI/issues/788).

**Device sets.** A run with a device set needs a slot on every device of the set: it starts only when all of them have room, takes them in one step and releases them in one step. While it waits at the front of its FIFO, its slots are reserved, so steady traffic on one of its cards cannot starve it. Reservations are first come, first served, which rules out two set runs reserving each other into a deadlock. An `any` run with a set reserves nothing, because its machine is unknown until it is placed. On the server itself a set run holds every slot of its set and executes on the first device; on a worker its subprocess sees every device of the set.

**Same-host sharing.** A worker whose `hello.machine_id` equals the server's own (a SHA-256 digest of `/etc/machine-id`, the macOS hardware UUID or the Windows machine GUID) runs on the same machine. Its CPU, `mps:0` and every CUDA card whose NVML UUID matches one of the server's cards are aliased onto the server's slot for that card, under the server's limit. A local run and a worker run therefore cannot both take one physical card under a limit of 1. A peer address cannot decide this: an `ssh -R` tunnel makes a remote agent look like loopback.

**The `target` option** ([#769](https://github.com/CodefyUI/CodefyUI/issues/769)) is `{"worker": "local" | "any" | <name>, "devices": [...]}`, both keys optional. `worker` is a name matching `[a-z0-9][a-z0-9-]{0,62}`; `local` and `any` are reserved. `devices` lists concrete devices of one kind with no duplicates (`cpu`, `cuda:N`, `mps:N`), and its first entry becomes the run's `device`. The scheduler places:

- **`local`** (or no target): `LocalScheduler`, the 3.0 rule.
- **a worker name**: that worker, on the device set, or on the requested device, or on the first offered device of the requested kind with a free slot.
- **`any`**: the first connected, non-draining worker that can take the run, in name order, and the server last. With a CUDA set of more than one card, or of one card other than `cuda:0`, the server never qualifies: a worker's run subprocess sees the set renumbered from `cuda:0` (`CUDA_VISIBLE_DEVICES`), the server's process does not, so a node pinned to `cuda:1` would address a different card. Other sets (`cpu`, `mps`, `cuda:0` alone) qualify when the server has every device of the set.

**Why some refusals are immediate.** A submission is refused with `400` when it can never run as asked: a worker name that has not connected since the server started, a worker that does not offer the requested devices (as it last reported them), an `any` run that neither the server nor any worker it has seen could serve (for a CUDA set that renumbering changes, no worker it has seen), the interactive lane or `record_outputs` with a worker target. Parking such a run would leave a `queued` row that nothing will ever start, and a client could not tell it from a slow queue. A run that can run later waits instead, including a run for a known worker that is away now, and `queue_reason` on its row says why when the reason is more than a busy slot: `waiting for worker gpu-box to reconnect`, `worker gpu-box is draining`, `waiting for a worker that offers cuda:0, cuda:1`. Draining a worker (`POST /api/workers/{id}/drain`) stops new placements on it; its running runs continue.

**Sweeps and studies** ([#777](https://github.com/CodefyUI/CodefyUI/issues/777)). `options.target` reaches every child. Before any row is written, `RunService.check_placeable_batch` applies the submit rules above to each distinct child (children differ only in their graph's `settings`), so a sweep or study is refused whole or accepted whole; a worker that goes away while the children are being submitted makes them wait, never fails the batch. Each sweep variant and study attempt reports `worker_id`, `devices` and `queue_key` while its run exists. A seeded run on this server takes the process-wide seed gate and runs alone; a worker run has random state of its own and skips that gate, so seeded children on different workers or cards run at the same time. The **New sweep** dialog offers the same target picker as **Queue on...**, and the sweep view shows each variant's Worker and Devices; studies take `target` through the API.

## Isolation

**One subprocess per run** ([#771](https://github.com/CodefyUI/CodefyUI/issues/771)). For each `assign` the worker starts `child.main` in a `multiprocessing` *spawn* context, so every run gets a fresh interpreter and its own CUDA context. A segfault, an out-of-memory kill or a forced stop ends that run; the agent, the control plane and the other runs carry on. The agent refuses at once (a `run_finished` with status `failed`) an assignment it cannot honour: `record_outputs`, a device it does not offer, a second copy of a run it is already executing, or any run while draining.

**Device visibility** ([#774](https://github.com/CodefyUI/CodefyUI/issues/774)). The subprocess environment sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so the indices the agent reports through NVML are the indices torch uses, and `CUDA_VISIBLE_DEVICES` to the run's cards in the order the assignment lists them. With `["cuda:2", "cuda:0"]` the run's `cuda:0` is the machine's `cuda:2`. A `cpu` or `mps` run gets an empty `CUDA_VISIBLE_DEVICES`. Node `device` params are never rewritten, and `device_utils.resolve_device` is unchanged; the environment does the renumbering. The subprocess also loses `CODEFYUI_DB_PATH`, the agent's token and the ROCm visibility filters. `run_started` carries `visible_devices` and `device_aliases`, which the control plane merges into the run's `execution_facts`.

**Cancel.** A Stop on a worker run is forwarded as `cancel`. The agent sends `("cancel",)` down the pipe, which the subprocess turns into `ExecutionContext.cancel()`, the same cooperative stop an in-process run gets. If the subprocess has not ended after `grace_s` (30 s by default), the agent terminates it, and kills it 5 s later; the control plane files the run as stopped, the same as an in-process cancel. `cdui agent stop` asks every running run to stop the same way with a 10 s grace.

**Disconnects.** Closing `/ws/worker` never cancels a run by itself. A worker whose socket closes, or that misses three heartbeats in a row, is marked `disconnected`; each run it was executing ends `interrupted` with a `run_warning` of kind `worker_disconnected`, and its slots are freed. A second connection under the same name replaces the first; a run still in flight on the old connection carries on when the new one lists it in `run_alive`, and ends `interrupted` otherwise. Keeping a run alive across a longer outage is [#783](https://github.com/CodefyUI/CodefyUI/issues/783).

**What stays in-process.** Runs executing in the control plane share one process and one CUDA context, as in 3.0. Their `peak_memory` entries are marked `exact: false` for that reason; a worker run's CUDA entries are `exact: true`, because its subprocess holds only that run.

## Storage

**Today.** A worker runs on the same machine as the server, so both see one filesystem. A run subprocess writes where its nodes write, and an `artifact` frame carries the node's `kind`, `path`, `meta` and `node_id`; the control plane stores that row in `exec_run_artifacts` with the path exactly as the worker reported it. Each run's console output goes to `agent-runs/<run id>/output.log` in the agent's user-data directory, and the last 4 KB of it travel with an unreported crash. The run subprocess never opens SQLite.

**Planned.** Remote workers need a way to find a run's input files and to return what it writes. The design follows Kubernetes volumes and Argo's artifact repository:

- **Mounts** ([#782](https://github.com/CodefyUI/CodefyUI/issues/782)): a worker declares labels for shared filesystems (`data` → `/mnt/nas/data`), the control plane translates paths between its own and the worker's view, and the scheduler places a run only on a worker that has the labels it needs. The `mounts` field in `hello` and in `GET /api/workers` is already carried for this.
- **Transfer backend** ([#785](https://github.com/CodefyUI/CodefyUI/issues/785)): with no shared filesystem, the control plane acts as the artifact repository and files move over HTTP, content-addressed.

## Security

**Today: same-host only.** `/ws/worker` accepts a connection when every one of these holds; otherwise it closes before `accept`:

- the `Host` header is on the allowlist, as for `/ws/execution`;
- there is no `Origin` header (an agent sends none, and no web page has a reason to open this socket);
- the peer address is loopback;
- the request carries the server's session token (`?token=` or `X-CodefyUI-Token`), which a same-host agent reads from the token file the server writes.

A refused handshake is closed with **4003** (Host, Origin, non-loopback peer) or **4401** (token). After `accept` the first frame must be `hello` within 10 s; a protocol-version mismatch or an invalid or reserved name gets `reject`. The REST side follows the run routes: `GET /api/workers` is open, `POST /api/workers/{id}/drain` and `/undrain` need the session token ([API Reference](./api-reference#runs-and-sweeps)).

**What travels in an assignment.** `assign` carries the run's graph with its SECRET params restored from the in-memory vault ([#251](https://github.com/CodefyUI/CodefyUI/issues/251)), because the worker has no vault of its own. The database copy of the graph stays scrubbed. Today that frame crosses a loopback socket only.

**The direction rule.** A worker always dials out to the control plane and never listens on a port; the control plane never connects to a worker. A worker never writes SQLite.

**Planned.** Per-worker credentials through enrollment tokens, issued from the Workers panel's Add-worker dialog ([#780](https://github.com/CodefyUI/CodefyUI/issues/780)); handshake checks of version, environment fingerprint and node catalog ([#781](https://github.com/CodefyUI/CodefyUI/issues/781)); the control plane's TLS fingerprint pinned by the agent on first use, and a rule that secrets travel only over a verified channel ([#786](https://github.com/CodefyUI/CodefyUI/issues/786)).

## Invariants

Every change in this area keeps these ([#764](https://github.com/CodefyUI/CodefyUI/issues/764)):

1. With no worker enrolled and no `target` option, submission behaves exactly as 3.0.2: the same queue key, the same `InProcessExecutor`, the same event sequence. The `single_node_parity` snapshot changes only in a commit that means to change it.
2. **The export surface is frozen.** `backend/app/core/codegen.py`, `backend/run_graph.py`, node `execute()` semantics, and the runtime symbols exported scripts import: `graph_engine.invoke_node`, `prepare_executable_graph` and `topological_sort`; the `ExecutionContext` constructor with `derive_seed` and `current_node_id`; `device_utils.resolve_device`; `seeding.*`, `runtime.*` and `api_contract.*`. The run subprocess constructs an `ExecutionContext` and never changes its interface; device sets use `CUDA_VISIBLE_DEVICES`, and `resolve_device` stays as it is.
3. The interactive (canvas) lane always executes in the control-plane process.
4. A worker never opens an inbound port and never writes SQLite.
5. New behaviour sits behind the `target` option or `CODEFYUI_EXECUTION_*` settings and is off by default.
6. Web UI first: workers are managed in the browser, and the CLI gains only `cdui agent start|stop`. No new `cdui status` or TUI features.

## Decisions log

| Date | Decision |
|------|----------|
| 2026-10-15 | Split execution from the control plane; keep the in-process path as the default and the only single-node path. |
| 2026-10-15 | Web UI first: the Workers panel, the Add-worker dialog and the run target selector live in the browser; the CLI keeps only start, join and stop. |
| 2026-10-15 | Topology: any machine runs `cdui start` as the control plane; GPU machines run `cdui agent start`, joining with a one-line command the browser generates. Connections always go worker → control plane. |
| 2026-10-15 | Admission stays count-based per `(worker, device)` slot; memory-based admission is deferred ([#788](https://github.com/CodefyUI/CodefyUI/issues/788)). |
| 2026-10-15 | Storage: a NAS (mounts) is the default; a no-NAS transfer backend is required too. |
| 2026-10-15 | The interactive lane stays in the control plane; worker credentials come from enrollment; Linux workers first, Windows best effort; control plane and worker run the same version; DDP later ([#792](https://github.com/CodefyUI/CodefyUI/issues/792)). |
| 2026-10-15 | One spawned subprocess per run, with the device set as `CUDA_VISIBLE_DEVICES` ([#771](https://github.com/CodefyUI/CodefyUI/issues/771)); the agent process never imports torch. |
| 2026-10-15 | Same-host workers are detected by machine id, and shared cards by NVML UUID, never by peer address ([#773](https://github.com/CodefyUI/CodefyUI/issues/773)). |
| 2026-10-15 | Until enrollment exists, `/ws/worker` accepts loopback peers with the session token only ([#772](https://github.com/CodefyUI/CodefyUI/issues/772)). |
| 2026-10-15 | One submit rule for single runs, sweeps and studies: only a worker name never seen since startup is refused; a run for a known worker that is away waits for it ([#777](https://github.com/CodefyUI/CodefyUI/issues/777)). |

## Entry points

| Area | File |
|------|------|
| Run service (queue, row, event log, cancel) | `backend/app/core/run_service.py` |
| Executor seam, `InProcessExecutor`, `WorkerExecutor` | `backend/app/core/executors.py` |
| Pure event builders | `backend/app/core/run_events.py` |
| Scheduler, slots, placement | `backend/app/core/scheduling.py` |
| Worker registry | `backend/app/core/workers.py` |
| Worker wire protocol | `backend/app/core/worker_protocol.py` |
| Device inventory (NVML, torch-free mode) | `backend/app/core/resources.py` |
| Machine identity | `backend/app/core/machine_identity.py` |
| `/ws/worker` handler | `backend/app/api/ws_worker.py` |
| `/api/workers` routes | `backend/app/api/routes_workers.py` |
| Worker agent (CLI, connection, runner, run subprocess) | `backend/app/agent/` |
| Workers panel | `frontend/src/components/Workers/` |
| Queue on... dialog | `frontend/src/components/Toolbar/QueueRunDialog.tsx`, `frontend/src/components/Toolbar/queueTargets.ts` |
