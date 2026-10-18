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
| Keeps weights and cache between clicks, captures outputs for the Inspector | Yes | No (outputs only with `record_outputs`; a worker keeps them and the Inspector reads them from it) |

Use **Run** while you build and debug a graph. Queue a run for anything long, anything you want on a particular card, and anything that needs more than one card.

### Queue on... {/* #queue-on */}

The caret next to **Run** in the toolbar opens a menu whose **Queue on...** item submits the active tab's graph as a queued run (the same `POST /api/runs` that `cdui run` uses):

- **Target**: **This server**, each connected worker with its status and devices (a draining worker is listed but cannot be chosen), and **Any available** once a worker is connected.
- **Devices**: one device for This server; for a worker, a set of one kind, kept in the order the worker lists its devices, with each CUDA card's in-run name shown as you tick it (`cuda:2 -> cuda:0 in this run`); for Any available, a kind (CPU, CUDA GPU, Apple MPS).
- **Name**, **Random seed**, **Deterministic algorithms** and **Record node outputs** start from the tab's settings. On a worker the recorded outputs stay on the worker, and the dialog says so.

**Queue** submits the run, closes the dialog and opens it in the Runs tab. A refusal from the server is shown in the dialog.

## Start a worker on this machine {/* #start-a-same-host-worker */}

A worker agent executes each run it is given in a subprocess of its own, so a crash or an out-of-memory kill ends that run and leaves the server and the other runs alone. This section starts one on the same machine as the server; for a GPU server elsewhere, see [Remote Workers](/usage/remote-workers#adding-a-worker).

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
- **`any`** takes the first connected, non-draining worker with a free slot, in name order, and this server last. An `any` run with a CUDA set of more than one card, or of one card other than `cuda:0`, runs on a worker only, because only a worker renumbers the set from `cuda:0` for the run (see below); the submit fails with `400` when no worker the server has seen offers the whole set.

**In-run device names.** A run on a worker sees only its own cards, renumbered from 0 in the order you listed them. With `--devices cuda:2,cuda:0`, the run's `cuda:0` is the machine's `cuda:2` and its `cuda:1` is the machine's `cuda:0`. A node whose **device** parameter is `auto` follows the run's device (the first card); to put a node on another card of the set, set its **device** to that card's in-run name, here `cuda:1`. The run records the mapping in `execution_facts` as `visible_devices` and `device_aliases`. See [Device Backends](/advanced/device-backends#addressing-one-card-out-of-several).

**Sweeps and studies.** `POST /api/sweeps` and `POST /api/studies` pass `options.target` to every child. With `"target": {"worker": "any"}` and no `devices`, each child takes the next free slot of the sweep's device kind, so a four-variant sweep on two agents with one card each runs two variants at once; seeded children on workers run side by side, each in a process of its own. A target a single run would be refused for refuses the whole sweep or study with `400` before anything is queued. Each variant and attempt reports `worker_id`, `devices` and `queue_key`. The **New sweep** dialog has the same Target and Devices picker as **Queue on...**: This server, a connected worker with a device set, or Any available with a device kind. Left at This server and the tab's device, it sends the sweep as before, with no target. The sweep view shows each variant's Worker and Devices. Studies take `target` through the API. See [Run Queue, Sweeps](./run-queue#sweeps).

## Slots and limits {/* #slots-and-limits */}

A **slot** is one device on one machine. Each slot admits one run at a time on an accelerator and two on the CPU, and everything else waits in submit order. A device-set run needs a free slot on every device of its set and takes them all at once; while it waits at the front of its queue it reserves them, so a stream of single-card runs cannot starve it.

`CODEFYUI_RUN_QUEUE_MAX_CONCURRENT` changes one slot's limit. A bare device is this server's; `worker/device=N` is a worker's:

```bash
CODEFYUI_RUN_QUEUE_MAX_CONCURRENT="cuda:0=2,gpu-box/cuda:1=3" cdui start
```

A card that several offerers share counts as one card: a worker on the same machine and this server (matched by the card's UUID), or two workers that offer the same card (two agents on one GPU server, such as an SSH worker and an enrolled agent). The runs through every offerer count against it, under the smallest limit among the offerers that are online (a worker on the same machine without a `<worker>/<device>` override of its own uses this server's limit for the card, so `cuda:0=2` raises it), so a run on the server and runs on those workers never take one card past its limit. A card shared by memory keeps one ledger, and admits by memory only while every online worker offering it is set to share it. The Workers panel says "Same physical card as ..." on such a card and names the offerer whose limit applies; an override that the smaller limit lowers is logged as a warning when a worker connects. Raising a limit lets runs share a card's memory; nothing checks that they fit. To share a worker's card by memory instead, see [Sharing a card](#sharing-a-card). Details: [Run Queue, Configuration](./run-queue#configuration).

## Sharing a card {/* #sharing-a-card */}

A worker's card admits one run at a time by default (**Exclusive**). To let several runs share it, set the card to **Shared by memory** in the [Workers panel](#the-workers-panel). The choice is stored per worker and card, and survives a reconnect and a restart of the server. Only CUDA and MPS cards can be shared, and only on a worker: a run on this server shares one process with every other run here, so nothing could hold it to a budget.

A run sent to a shared card declares a **memory budget**, in GiB per card of its device set: `target.memory_gib`, or the **Memory budget** field of [Queue on...](#queue-on). The field is filled in with the CUDA peak of the last succeeded run of the same graph on the same number of cards plus 20%, when there is one (a per-card peak of a run on two cards says nothing about the same graph on one; MPS reports no peak, only what the driver holds at the end, so an MPS run declares its budget). That measured figure is also the default the server applies to a run that declares none. "The same graph" is the graph as it executes: node types, the params that shape its memory or pick its data (a `Dataset` node's `name` among them), and the wiring, topology included. Node ids and positions, display labels, seeds, how long it runs (`epochs`, `max_steps`, `lr`) and the loader's worker processes do not count, so a rerun with another seed or more epochs finds its peak. The params of a node type this server does not know (a plugin installed only on a worker) count too: they are left out of the stored graph, but not out of the key. A shared MPS card reports no memory total, so it cannot admit a run by budget: the dialog says so in place of the field, and the server refuses the run until the card is set back to Exclusive. A CUDA card that has not reported its memory yet keeps the field, with a note that the budget is judged once its worker's next heartbeat arrives.

Without such a run, the field is filled with a **static estimate**, tagged **Estimate** in the dialog and never presented as a peak: every parameter of the graph's models at 4 bytes, plus, for a graph that trains, what each trainable one holds for training (its gradient and optimizer state: 12 more bytes with Adam, 8 with SGD and momentum, 4 with plain SGD; RMSprop with momentum or `centered` one more state each) and, under bf16/fp16 autocast, a 2-byte copy of every weight; plus the activations of one training step at the graph's batch size or of an evaluation pass at its own batch, whichever is larger; plus 20% on all of that for the allocator's reserve; plus 128 MiB for the CUDA libraries' workspaces. The dialog lists each of those terms, so they add up to the figure. The server counts each model by building it on PyTorch's `meta` device (shapes only, no memory, in a thread of its own with a time limit), and the activations by one forward pass on a `meta` input whose shape comes from the model's first layer and the graph's data nodes (`image_size` or `seq_len` set explicitly on a node that provides data, or a known dataset; a default value never decides the size). A build that overruns its time limit is abandoned, and the dialog offers no estimate; the estimate is not built at all when a measured peak exists. On the built-in training examples the estimate came out 1.2 to 1.8 times the measured peak for runs above 1 GiB, and at most 0.25 GiB for the small ones, whose peak is mostly fixed overhead. An estimate that had to skip or guess part of the graph is shown as a **Rough estimate** with the reason, and the field stays empty: a model node it cannot count (such as one from a plugin), no input shape for a model's activations, an input size the graph does not give (a recurrent net or an adaptive-pooling CNN runs at any size, so a default size would be a guess), or no batch size.

The server never applies a static estimate on its own. A measured suggestion left as it is is not sent: the server applies the same default, and the run's Budget column names the run it came from. An estimate left as it is is sent as the run's declared budget, marked as an estimate (`target.memory_source: "estimate"`), and the Budget column marks it `est.`. An estimate submitted after the graph already has a larger measured peak (from the dialog, the API or the CLI) gets that peak plus 20% instead. When a run of the same graph succeeds with a larger peak while such a run is still waiting, the waiting run's budget is raised the same way, and its reason and Budget cell in the Runs panel say so; a raised budget that no card the run may use can hold fails the run with that reason rather than leaving it waiting forever. A budget you typed is never changed. A run that runs out of memory under its budget says where the budget came from (typed, measured, or an estimate) and asks for a larger `target.memory_gib`. With no measured peak and no budget declared, the server refuses the run and asks for one. A sweep or a study is checked whole before any of its runs is written, each run with its own `target.wait`: when one of its runs would be refused (no budget, one larger than every card it may use, or one that does not fit now with `wait: false`), the sweep or study is refused; a sweep of a graph that has no measured peak yet declares its budget.

```json
{"options": {"target": {"worker": "gpu-box", "devices": ["cuda:0"], "memory_gib": 6}}}
```

The server keeps a ledger per card: its total and free memory from the worker's heartbeat (free counts every process on the card, other users' jobs included), the budgets of runs assigned but not started (**reserved**), and the budgets of runs that started (**running**). A run on a CUDA card also holds its CUDA context (cuBLAS handles, loaded kernels), which the cap below does not cover, so the server counts each one as its budget plus `CODEFYUI_EXECUTION_CUDA_CONTEXT_MIB` (768 MiB by default; about 650 MiB was measured on a CUDA 12 card). A run fits when that sum is at most `min(free - reserved, total - reserved - running)`, and a refusal names both: `needs 6.0 GiB (+0.8 GiB CUDA context), ...`. The worker caps each run's allocator to its budget (`torch.cuda.set_per_process_memory_fraction`) before the run starts, so a run that outgrows its budget runs out of memory alone and the other runs on the card go on.

When a budget does not fit:

- a run for a named worker is refused at once, with the reason: `needs 12.0 GiB, cuda:1 on gpu-box has 6.3 GiB free (runs a1b2c3d4, c3d4e5f6)`. With `target.wait: true` (**Wait in the queue when it does not fit now** in the dialog) it waits instead;
- a run for `any` waits until a card has room, unless its budget is larger than every candidate card (refused), or `target.wait` is `false` (refused instead of waiting). A run for the one-shot pool (`target.pool`) is judged the same way, on the one-shot workers' cards;
- a run waiting for memory says so under its worker in the Runs panel.

A run without a budget never goes to a shared card, and a run on a shared card that holds no budget (a canvas run on a same-host worker's card, or one that started before the card was shared) keeps it to itself until it ends. The Runs panel's **Budget** column shows each run's budget, and the Workers panel draws each shared card's memory: in use, reserved, and what a new budget may take (its CUDA context already taken off). A run that waited for a card while it was Exclusive gets its graph's default budget when the card is set to Shared, or later, when a run of the same graph on the same number of cards succeeds; until then it waits, and the Runs panel says so (cancel it and queue it again with a budget to start it sooner). A run admitted by its budget is never started on an agent that cannot cap it: if the worker reconnected with an older agent in between, the run fails with the reason. A run the server adopts after a restart ([Remote Workers](/usage/remote-workers)) counts its budget on its card again from the start, as running.

## The Workers panel {/* #the-workers-panel */}

Open it with **Workers** in the Runs panel header, or from **Settings → Execution → Workers → Manage**. It refreshes every 5 seconds while open and shows, for each worker:

- its status (**Online**, **Degraded** when a heartbeat is late or its node types differ from the server's, **Draining**, **Disconnected**), its version and whether it matches the server's, its environment digest, whether its node types match the server's (and which differ), the reason its last connection was refused, mount labels (see [Remote Workers, Storage](/usage/remote-workers#storage)) and the age of its last heartbeat;
- each device with its free and total memory from the latest heartbeat and its slots taken / limit, and for a CUDA or MPS card how it admits runs (**Exclusive** or **Shared by memory**, [Sharing a card](#sharing-a-card)) with a shared card's memory bar;
- the runs it is executing, each a link to the run in the Runs panel.

Agents the server refused (another version, for example) and that have no card are listed under **Refused connections** with the reason ([Remote Workers, Keeping workers in sync](/usage/remote-workers#keeping-workers-in-sync)).

**Drain** stops new runs from being placed on a worker while its running runs finish; **Undrain** reverses it. A queued run whose worker is draining or away waits, and the Runs panel shows why under the run's worker. **Add worker** generates an enrollment command for another machine, or shows the same-host command above with its options. **Re-key** gives a worker a new credential, and **Remove** revokes it, disconnects the worker and fails the runs queued for it ([Remote Workers](/usage/remote-workers)).

## What the canvas does {/* #the-canvas */}

**Run** on the canvas always executes in the server process, on the graph's device (else the one in Settings), and never goes to a worker. It skips the queue, so it can share a card with a queued run even when that card's limit is 1. It keeps the tab's cached outputs and model weights between clicks and records outputs for the Inspector, which a worker run cannot do. To send the tab's graph to a worker, use [Queue on...](#queue-on).

## Peak memory in the Runs panel {/* #peak-memory */}

When a run finishes, its detail header shows **Peak memory** for CUDA (the allocator's high-water mark, every card on hover) or **Memory held at end** for MPS (what the driver still holds, cache included; MPS has no peak counter). A run on a worker executes alone in its subprocess, so its figure is exact. Runs that executed in the server process share one CUDA context, so their figure is approximate when another run used the card at the same time. Use the peak of a finished run to decide whether two such runs fit on one card before you raise its limit.

## Limits {/* #limits */}

- **One run does not train across cards.** A device set makes several cards visible to one run, and nodes can be pinned to them by in-run name, but there is no `DistributedDataParallel` strategy yet ([#792](https://github.com/CodefyUI/CodefyUI/issues/792)).
- **Memory admission on workers only.** A card on this server is admitted by count per slot: a run is not refused because the card lacks memory, and an over-full card fails with an out-of-memory error. A worker's card can be [shared by memory](#sharing-a-card). See [Training Memory](/advanced/training-memory).
- **The ledger counts budgets, not what runs use.** A running run is counted at its budget, and free memory comes from the worker's heartbeat (every 10 seconds) for the whole card: memory another user's job takes between two heartbeats is not seen until the next one, and per-run use is not measured.
- **Plain HTTP between machines.** A remote worker enrolls and connects over whatever the control plane serves; without a TLS proxy or an SSH tunnel its token and run inputs cross the network unencrypted ([Remote Workers, Trust](/usage/remote-workers#trust-between-agent-and-control-plane)).
- **Captured outputs stay on the worker** (`record_outputs`, [#789](https://github.com/CodefyUI/CodefyUI/issues/789)): the Inspector reads them only while the worker is connected. No persistent weights: every worker run starts fresh.
- **A long disconnect ends the run.** A worker that loses its connection keeps its runs going for `CODEFYUI_WORKER_RESUME_GRACE_S` (5 minutes); past that they end `interrupted` ([Remote Workers, When the connection drops](/usage/remote-workers#when-the-connection-drops)).

## See also

- [Run Queue](./run-queue) — queue order, `cdui run`, the `target` option and workers in detail
- [Execution Architecture](/advanced/execution-architecture) — control plane, executors, worker protocol and invariants
- [Device Backends](/advanced/device-backends) and [Training Memory](/advanced/training-memory) — devices, card addressing and out-of-memory errors
