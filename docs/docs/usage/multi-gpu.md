---
sidebar_position: 3.6
title: Multi-GPU Execution
description: Use a machine with several GPUs from CodefyUI — when to queue a run instead of clicking Run, starting a worker agent on the same machine, choosing a device or a device set, slots and their limits, the Workers panel, what the canvas does, peak memory, and the current limits.
---

# Multi-GPU Execution

The CodefyUI **server** is one process. A **run** executes in that process by default, or in a subprocess of a worker agent when you target one, and it uses the devices of its target. On a machine with several GPUs that gives you two things: several runs at once, one per card (or more, if you allow it), and a single run that sees several cards. This page is the how-to; [Execution Architecture](/advanced/execution-architecture) explains the design, and [Run Queue](./run-queue) is the reference for queue order and options.

## Run or queue {/* #run-or-queue */}

| | **Run** (the canvas) | **Queued run** |
| --- | --- | --- |
| Started from | The toolbar's **Run** | **Queue on...** next to Run, `cdui run`, `POST /api/runs`, a sweep or study |
| Executes in | Always the server process | The server process, or a worker when its `target` says so |
| Waits for a free card | No: starts at once (two at most, one per tab) | Yes: joins the queue of its device |
| Devices | One: the graph's device, else Settings | One device, or a device set on a worker |
| Keeps weights and cache between clicks, captures outputs for the Inspector | Yes | No (outputs only with `record_outputs`, and only on the server) |

Use **Run** while you build and debug a graph. Queue a run for anything long, anything you want on a particular card, and anything that needs more than one card.

### Queue on... {/* #queue-on */}

The caret next to **Run** in the toolbar opens a menu whose **Queue on...** item submits the active tab's graph as a queued run (the same `POST /api/runs` that `cdui run` uses):

- **Target**: **This server**, each connected worker with its status and devices (a draining worker is listed but cannot be chosen), and **Any available** once a worker is connected.
- **Devices**: one device for This server; for a worker, a set of one kind, kept in the order the worker lists its devices, with each CUDA card's in-run name shown as you tick it (`cuda:2 -> cuda:0 in this run`); for Any available, a kind (CPU, CUDA GPU, Apple MPS).
- **Name**, **Random seed**, **Deterministic algorithms** and **Record node outputs** start from the tab's settings. Record node outputs is disabled for a worker or Any available.

**Queue** submits the run, closes the dialog and opens it in the Runs tab. A refusal from the server is shown in the dialog.

## Start a worker on this machine {/* #start-a-same-host-worker */}

A worker agent executes each run it is given in a subprocess of its own, so a crash or an out-of-memory kill ends that run and leaves the server and the other runs alone. In this release the agent runs on the same machine as the server.

```bash
cdui start                 # the server, if it is not running yet
cdui agent start           # a worker named after this machine's hostname
cdui agent stop            # when you are done
```

`cdui agent start` needs no flags next to `cdui start`: it finds the server's address and session token on its own and reconnects when the server restarts. `--name gpu-box` sets the worker name, `--devices cuda:0,cuda:1` limits the cards it offers (the CPU is always offered), and `--foreground` keeps it attached for systemd. All flags: [CLI Commands, Agent commands](/getting-started/cli-commands#agent-commands).

Within a few seconds the worker appears in the [Workers panel](#the-workers-panel) and in `GET /api/workers`.

## Choose devices {/* #choose-devices */}

A queued run names where it executes in the [Queue on...](#queue-on) dialog, with `--target` and `--devices` on `cdui run`, or with `options.target` in `POST /api/runs`:

```bash
# One card on this server, as before
cdui run train.json --device cuda:1

# One card on the worker gpu-box
cdui run train.json --target gpu-box --device cuda:1

# Two cards on the worker: the run sees them as cuda:0 and cuda:1
cdui run train.json --target gpu-box --devices cuda:2,cuda:3

# Whichever has a free CUDA card first: a connected worker, else this server
cdui run train.json --target any --device cuda
```

```json
{"graph": {...}, "options": {"target": {"worker": "gpu-box", "devices": ["cuda:2", "cuda:3"]}}}
```

- **`local`** (the default) is this server. A run here executes on one device: with a device set it holds a slot on every device of the set, so nothing else is placed on them, and executes on the first.
- **A worker name** sends the run to that worker. The worker must have connected since the server started and must offer the devices, or the submit fails with `400`. A worker that is away now is not a refusal: the run waits for it to reconnect.
- **`any`** takes the first connected, non-draining worker with a free slot, in name order, and this server last.

**In-run device names.** A run on a worker sees only its own cards, renumbered from 0 in the order you listed them. With `--devices cuda:2,cuda:0`, the run's `cuda:0` is the machine's `cuda:2` and its `cuda:1` is the machine's `cuda:0`. A node whose **device** parameter is `auto` follows the run's device (the first card); to put a node on another card of the set, set its **device** to that card's in-run name, here `cuda:1`. The run records the mapping in `execution_facts` as `visible_devices` and `device_aliases`. See [Device Backends](/advanced/device-backends#addressing-one-card-out-of-several).

**Sweeps and studies.** `POST /api/sweeps` and `POST /api/studies` pass `options.target` to every child. With `"target": {"worker": "any"}` and no `devices`, each child takes the next free slot of the sweep's device kind, so a four-variant sweep on two agents with one card each runs two variants at once; seeded children on workers run side by side, each in a process of its own. A target a single run would be refused for refuses the whole sweep or study with `400` before anything is queued. Each variant and attempt reports `worker_id`, `devices` and `queue_key`. The **New sweep** dialog has the same Target and Devices picker as **Queue on...**: This server, a connected worker with a device set, or Any available with a device kind. Left at This server and the tab's device, it sends the sweep as before, with no target. The sweep view shows each variant's Worker and Devices. Studies take `target` through the API. See [Run Queue, Sweeps](./run-queue#sweeps).

## Slots and limits {/* #slots-and-limits */}

A **slot** is one device on one machine. Each slot admits one run at a time on an accelerator and two on the CPU, and everything else waits in submit order. A device-set run needs a free slot on every device of its set and takes them all at once; while it waits at the front of its queue it reserves them, so a stream of single-card runs cannot starve it.

`CODEFYUI_RUN_QUEUE_MAX_CONCURRENT` changes one slot's limit. A bare device is this server's; `worker/device=N` is a worker's:

```bash
CODEFYUI_RUN_QUEUE_MAX_CONCURRENT="cuda:0=2,gpu-box/cuda:1=3" cdui start
```

A worker on the same machine shares this server's slot for every card that is physically the same (matched by the card's UUID), so a run on the server and a run on that worker never both take one card under a limit of 1. For such a card the server's limit applies, and a `worker/device` override for it is ignored with a warning. Raising a limit lets runs share a card's memory; nothing checks that they fit. Details: [Run Queue, Configuration](./run-queue#configuration).

## The Workers panel {/* #the-workers-panel */}

Open it with **Workers** in the Runs panel header, or from **Settings → Execution → Workers → Manage**. It refreshes every 5 seconds while open and shows, for each worker:

- its status (**Online**, **Degraded** when a heartbeat is late, **Draining**, **Disconnected**), its version and whether it matches the server's, its environment digest, mount labels and the age of its last heartbeat;
- each device with its free and total memory from the latest heartbeat and its slots taken / limit;
- the runs it is executing, each a link to the run in the Runs panel.

**Drain** stops new runs from being placed on a worker while its running runs finish; **Undrain** reverses it. A queued run whose worker is draining or away waits, and the Runs panel shows why under the run's worker. **Add worker** shows the same-host command above with its options.

## What the canvas does {/* #the-canvas */}

**Run** on the canvas always executes in the server process, on the graph's device (else the one in Settings), and never goes to a worker. It skips the queue, so it can share a card with a queued run even when that card's limit is 1. It keeps the tab's cached outputs and model weights between clicks and records outputs for the Inspector, which a worker run cannot do. To send the tab's graph to a worker, use [Queue on...](#queue-on).

## Peak memory in the Runs panel {/* #peak-memory */}

When a run finishes, its detail header shows **Peak memory** for CUDA (the allocator's high-water mark, every card on hover) or **Memory held at end** for MPS (what the driver still holds, cache included; MPS has no peak counter). A run on a worker executes alone in its subprocess, so its figure is exact. Runs that executed in the server process share one CUDA context, so their figure is approximate when another run used the card at the same time. Use the peak of a finished run to decide whether two such runs fit on one card before you raise its limit.

## Limits {/* #limits */}

- **One run does not train across cards.** A device set makes several cards visible to one run, and nodes can be pinned to them by in-run name, but there is no `DistributedDataParallel` strategy yet ([#792](https://github.com/CodefyUI/CodefyUI/issues/792)).
- **No memory admission.** Runs are admitted by count per slot; a run is not refused because a card lacks memory, and an over-full card fails with an out-of-memory error ([#788](https://github.com/CodefyUI/CodefyUI/issues/788)). See [Training Memory](/advanced/training-memory).
- **Same machine only.** Remote GPU servers need enrollment ([#780](https://github.com/CodefyUI/CodefyUI/issues/780)).
- **No captured outputs on a worker** (`record_outputs`, [#789](https://github.com/CodefyUI/CodefyUI/issues/789)), and no persistent weights: every worker run starts fresh.
- **A disconnect ends the run.** If the agent stops or loses its connection, its runs end `interrupted` ([#783](https://github.com/CodefyUI/CodefyUI/issues/783) adds resume).

## See also

- [Run Queue](./run-queue) — queue order, `cdui run`, the `target` option and workers in detail
- [Execution Architecture](/advanced/execution-architecture) — control plane, executors, worker protocol and invariants
- [Device Backends](/advanced/device-backends) and [Training Memory](/advanced/training-memory) — devices, card addressing and out-of-memory errors
