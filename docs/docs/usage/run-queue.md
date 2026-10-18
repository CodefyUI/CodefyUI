---
sidebar_position: 3.5
title: Run Queue
description: Queue graphs per device and walk away — lanes, FIFO order, concurrency limits, cancelling a waiting run, and the cdui run CLI.
---

# Run Queue

A run belongs to the **server**, not to whatever submitted it. Close the tab, close the terminal, log out — the run keeps going, and you can come back to it later over the API or the CLI.

That is what makes queueing useful. Submit five training jobs, shut the laptop lid, and the server works through them one at a time on the GPU instead of starting all five at once and running out of VRAM forty minutes in.

## One queue per device

Every run without a worker target is scheduled against its **resolved device** — `cpu`, `cuda:0`, `mps` — and executes in the server process. That string is the queue key, so each device has its own independent line and its own concurrency limit. The device is resolved in this order: the run's own `device` option, then the graph file's `settings.device`, then `cpu`; only an explicit `auto` picks the best accelerator present.

| Queue key | Runs at once (default) | Why |
| --- | --- | --- |
| `cuda:0`, `cuda:1`, `mps`, any accelerator | 1 | A second run on the same card competes for the same VRAM. The usual result of guessing otherwise is a CUDA out-of-memory error partway through a long job. |
| `cpu` | 2 | Here the failure mode is contention, not death — and a machine with cores to spare should use them. |

A saturated GPU never delays a CPU run, and two different cards never wait on each other. Runs within one queue start in **submit order**, oldest first.

**Admission.** A limit belongs to a *slot*: one device on one machine. `cuda:0` on this server and `cuda:0` on the worker `gpu-box-1` are two slots, each with its own limit. A run with a device set needs a slot on every device of the set: it starts only when all of them have room, takes them together, and gives them all back when it ends. While it waits at the front of its queue it holds none of them but **reserves** them: no other run may start on a reserved card until the set run has started, the runs already on those cards finish normally, and the set run starts once they have. The cost is idle time: a reserved card that is already free stays idle until the set's other cards free too, at most as long as the longest run already on one of them. Reservations are first come, first served, so two set runs that share a card start one after the other. A device-set run with `"worker": "any"` reserves nothing, because which machine's cards it needs is decided only when it starts.

A [worker](#workers) on the same machine as the server shares this server's slots for the devices that are physically the same: the CPU, the Apple GPU (`mps:0`), and every CUDA card whose UUID matches one of this server's cards, whatever number each side gives it. A run here and a run on that worker therefore never both get one card under a limit of 1. "Same machine" means the agent connected from a loopback address and reported the same machine id as the server (`/etc/machine-id`, the macOS hardware UUID or the Windows machine GUID, sent as a digest). Neither alone counts: an SSH tunnel is loopback too, and the machine id is reported by the agent itself. A CUDA card without a UUID, or with one this server does not have, keeps a slot of its own, and the server logs that once. When a slot several queues share frees, the run submitted earliest among the waiting runs at the front of those queues gets it, so a long line in one queue does not starve another. A waiting run's `queue_position` counts only the queue it waits in (a device-set run waits in its first device's queue), so position 1 means nothing is ahead of it in that line, not that it starts next.

:::note Aliases of one card share one queue
`cuda` and `cuda:0` are the same physical device, so they are canonicalised onto a single queue key (`cuda:0`) rather than becoming two independent FIFOs over one card. Same for `mps` and `mps:0`. On a multi-GPU box `cuda` follows the process's current device.
:::

:::note Where a run's device comes from
A run without an explicit `device` option uses the graph file's `settings.device`, and a graph without one runs on `cpu`; no mechanism picks an accelerator on your behalf. A run that names a device set (`target.devices`, or `--devices` on `cdui run`) uses the set's first device instead, and `settings.device` is not consulted. `--device auto` (or `"device": "auto"` in the request body) is the one exception: it resolves to the best accelerator this server can see, `cuda` before `mps` before `cpu`, and queues on that line. An accelerator that is not present falls back to `cpu` with a warning.
:::

A run can also name a **target**: `"target": {"worker": "local", "devices": ["cuda:0", "cuda:1"]}` in the request's `options`, or `--target` / `--devices` on `cdui run`. `worker` is `local` (this server, also what an omitted `worker` means), the name of a connected worker agent (see [Workers](#workers)), or `any`. `devices` lists concrete devices of one kind (`cpu`, `cuda:N`, `mps:N`; a bare `mps` means `mps:0`, a bare `cuda` is refused; no `auto`, no duplicates, no mixing `cuda` with `cpu`). Its first entry becomes the run's device, and therefore its queue, when `device` is omitted or `auto`, and also when `device` is only the kind (`"device": "cuda"` with `["cuda:1", "cuda:0"]` runs on `cuda:1`); a `device` that names anything else is refused. The run's row reports both as `worker_id` and `devices`. On this server the run queues and executes on its first device and holds a slot on every device of the set; on a worker it holds and uses every device of the set. Three more keys apply to a worker's card that is shared by memory: `memory_gib` (the run's budget per card, a number above 0 and at most 4096; refused with `local`), `memory_source` (`"estimate"`: that `memory_gib` is the static estimate the Queue on... dialog offered, which the Runs panel marks `est.`; `"declared"`, the default, is not stored; refused without `memory_gib`) and `wait` (`true`: wait when the budget does not fit now; `false`: refuse instead), see [Multi-GPU, Sharing a card](./multi-gpu#sharing-a-card). A run without a target behaves exactly as before.

### Workers

A worker agent (`cdui agent start`, see [CLI Commands](/getting-started/cli-commands#agent-commands)) connects to this server and runs each run it is given in a subprocess of its own. It runs on the same machine as the server, or on another machine once enrolled with a token from the Workers panel (see [Remote Workers](/usage/remote-workers#adding-a-worker)); an enrolled worker is listed as `offline` until it connects, and a run can target it meanwhile (also from **Queue on...**) and wait. Removing a worker fails the runs still queued for it. `GET /api/workers` lists the agents with their devices, each with its `slots` (`limit`, `taken`); its `local` field lists this server's devices the same way, and `same_host: true` marks an agent on this machine. A device of such an agent that is one of this server's carries `shared_with: "local"`; its `slots` are this server's slot for that card (the same counter and limit), so count it once when adding up usage. `POST /api/workers/{id}/drain` stops new runs on one (`/undrain` resumes).

How `target.worker` places a run:

- **omitted or `local`**: this server, on the queue of its resolved device, exactly as without workers.
- **a worker name**: that worker. The worker must have connected to this server since it started, and it must offer the device set, the named device, or a device of the named kind (`cuda`, `mps`; `auto` picks CUDA, then MPS, then the CPU), as it last reported its devices; otherwise the submit fails with `400`. A degraded worker, whose node types differ from this server's, takes only a graph whose node types it has; any other is `400` naming the missing types ([Remote Workers, Keeping workers in sync](/usage/remote-workers#keeping-workers-in-sync)). A worker that connected and is away now is not a refusal: the run is queued and waits for it to reconnect. The run waits in the queue `<worker>/<device>` while that device's slots on the worker are taken (one run per accelerator, two on the CPU, as here) and while the worker is draining or away; in the last two cases the run's `queue_reason` says so (`worker gpu-box is draining`, `waiting for worker gpu-box to reconnect`).
- **`any`**: the first connected, non-draining worker with a free slot for the device that does not lack node types this server has (a worker that lacks some, or reported no catalog, is skipped), in name order, and this server last. A run whose CUDA set a worker would renumber (more than one card, or one card other than `cuda:0`) never runs on this server: on a worker the set is renumbered from `cuda:0` for the run, here it is not, so a node pinned to `cuda:1` would address a different card. It waits for a worker that offers the whole set. A `cpu`, `mps` or `["cuda:0"]` set can still run here when this server has it. A submit that neither this server nor any worker it has seen since it started could ever run fails with `400`; a device offered only by a worker that is away queues the run until a worker that offers it connects.

`"target": {"pool": "one-shot"}` (no `worker`, or `any`) places a run on a one-shot worker (`cdui agent start --one-shot`), and nowhere else: `any` never uses one. The run waits for one that offers its device, in the pool's line for its device (`pool:one-shot/<device>`; an agent started with `--run-id` takes its run out of turn), and fails after `CODEFYUI_WORKER_ONE_SHOT_QUEUE_TIMEOUT_S` (24 hours); a run naming a one-shot worker waits the same way. See [Remote Workers, One-shot workers](/usage/remote-workers#one-shot-workers).

A run on a worker cannot use `record_outputs` or the interactive lane (the canvas always runs here); both are refused at submit. Once it starts, its row records `worker_id`, `devices` and a `queue_key` of `<worker>/<device>`, and its events are the same as a local run's. **Stop** reaches the worker, which stops the run cooperatively and terminates it after a grace period.

A worker keeps its runs going through a dropped connection for `CODEFYUI_WORKER_RESUME_GRACE_S` (5 minutes), and a run that reads or writes files needs a folder both machines share or the transfer backend: see [Remote Workers, When the connection drops](/usage/remote-workers#when-the-connection-drops) and [Remote Workers, Storage](/usage/remote-workers#storage).

A seeded run on a worker executes alone **in its own process**: the "a seeded run executes alone" rule of [Reproducible runs](./running-graphs#reproducible-runs-seed) guards this server's process-wide random state, and a worker's subprocess has random state of its own. So seeded runs on different workers or different cards run at the same time, and a seeded run on a worker neither waits for this server's runs nor holds them up. Seeded runs that execute on this server, including `any` runs that landed here, still run one at a time.

For running several jobs on a multi-GPU machine, starting a worker and choosing devices, see [Multi-GPU Execution](./multi-gpu).

This is a limit on **runs per device**, and it is not the same knob as `CODEFYUI_MAX_PARALLEL_NODES`, which bounds how many *nodes inside one run* execute at the same time. The two multiply.

## Lanes

A run's **lane** records where it came from and decides how it is scheduled.

| Lane | Submitted by | Scheduling | Limit |
| --- | --- | --- | --- |
| `queued` (default) | `POST /api/runs`, `cdui run` | Joins its device's FIFO | Per-device, see above |
| `interactive` | The canvas (**Run** on a graph) | Skips the FIFO and starts immediately | 2 at once, and one per editor session |

The canvas bypasses the queue on purpose: a classroom demo must not sit behind a six-hour training job. The trade-off is that an interactive run can push a card past that device's queued-lane limit — the FIFO exists to keep unattended work orderly, not to make a device exclusive.

### The interactive lane's two limits

Because it bypasses the queue, the interactive lane is bounded two other ways:

- **At most two interactive runs at once** (`CODEFYUI_RUN_INTERACTIVE_MAX_CONCURRENT`), so a wall of open tabs cannot exhaust the GPU.
- **One run per editor session.** Each open canvas connection lends its execution cache and its persistent module weights to the runs it starts — that is what makes clicking **Run** twice reuse the first run's weights and cached upstream nodes. Two runs sharing that state *at the same time* would read each other's half-built tensors, so the server refuses the second one.

Both refusals are immediate and explicit (HTTP 503, or an error on the canvas) rather than a silent wait. There is a live user in front of a canvas, and a click that quietly joined an invisible queue is indistinguishable from a hang. In practice you will not hit the per-session rule: the **Run** button is disabled while a run is in flight.

## Watching and cancelling

`GET /api/runs` reports `queue_position` for every waiting run — 1-based, **within its own device queue**, so a CPU run behind two other CPU runs is third in line no matter how many CUDA runs were submitted before it. `cdui run --wait` prints it while it waits, and the **Runs** tab of the results panel shows it as `Queue #N`.

Cancelling a run that has not started yet simply removes it from the line. Nothing executed, no device was touched, and the runs behind it move up. The row is recorded as `cancelled` with no start time, and anything following the run receives a normal stop event.

Cancelling a run that is already going is cooperative — see [Running Graphs](./running-graphs#stopping).

:::note Two runs may briefly both read as running on a cap-1 queue
A run gives its device slot back the moment its graph finishes, before it writes its final status and closing event. The next run therefore starts while the previous one is still filing paperwork, so a poll landing in that window can see two `running` rows on a queue whose limit is 1. Nothing is executing twice — the device really is idle — and it is deliberate: the alternative is holding a GPU idle across a database write, and worse, telling a canvas its run is over while the next click is still refused.
:::

### Runs panel

The **Runs** tab of the results panel lists the newest 50 runs the server owns — started from any tab, **Queue on...**, `cdui run` or the API — newest first, with filter chips **All / Running / Queued / Succeeded / Failed / Cancelled / Interrupted** and the columns **Run, Status, Worker, Devices, Started, Duration, Final loss**. **Worker** reads **This server** for a local run, **Any available** for an `any` run not placed yet, else the worker's name; **Devices** lists the run's device set, or its device, with each card's in-run name on hover for a run on a worker. A run started from a tab is listed under that tab's name, cut to 64 characters. A filter chip applies before the 50-row cut, and the count next to **Refresh** shows how many of the matching runs are listed (`50 of 312`). Older runs are reachable through `GET /api/runs?offset=`. A waiting run shows `Queue #N` beside its devices, and under its worker the reason it waits when there is one (`queue_reason`, such as `waiting for worker gpu-box to reconnect`). **New sweep** starts a [parameter sweep](#in-the-runs-panel). Each row offers up to four actions:

| Action | What it does |
| --- | --- |
| **Stop** | Asks a queued or running run to stop — cooperatively, see [Stopping](./running-graphs#stopping). |
| **Watch** | Queued or running runs only. Streams that run into the active tab's Execution Log, replaying it from the start, and switches the panel to that log. This is how you pick up a run submitted from a terminal or from another tab; the tab stops following whatever it was watching before (it asks first if that run is still going), and the runs themselves are unaffected. |
| **CSV** | Downloads the run's metrics (`GET /api/runs/{id}/metrics?format=csv`). |
| **Delete** | Finished runs only. Removes the run's metrics, event log, artifact records and any captured outputs; checkpoint files on disk are kept. |

Click a row for its detail: the device it ran on (hover it to see the requested device when the scheduler resolved a different one, such as `auto`), the seed and **Deterministic** when they were set, **Env** with the first 12 characters of the environment digest (hover for the whole digest; see [Run environment](../advanced/api-reference#run-environment)), the error if it failed, the metrics chart with its own **Download CSV**, the recorded artifacts each with a **Copy path** button, and the last 200 events of its log, which keeps updating while the run is active. Loading the editor while runs are still in progress shows a toast that counts them and points you here. With no tab open (the [welcome screen](./tabs-persistence#the-welcome-screen)) there is no results panel; open a tab to reach this one.

## `cdui run`

Submit a saved graph file to a running server:

```bash
cdui run mygraph.json
```

The command streams progress to the terminal and exits with the run. It is a **client**: the run is created on the server, so it survives the terminal that started it.

```bash
# Name it, put it on the GPU, seed it
cdui run train.json --name "resnet epoch sweep" --device cuda:0 --seed 42

# Queue five jobs and walk away
for i in 1 2 3 4 5; do cdui run "sweep-$i.json" --device cuda:0 --detach; done

# Follow along, but give up after ten minutes (the run keeps going)
cdui run train.json --device cuda:0 --timeout 600

# Keep node outputs for later inspection
cdui run infer.json --record-outputs
```

```powershell
# PowerShell equivalent of the sweep
1..5 | ForEach-Object { cdui run "sweep-$_.json" --device cuda:0 --detach }
```

| Flag | Meaning |
| --- | --- |
| `--name <text>` | Label stored on the run and shown wherever runs are listed |
| `--device <dev>` | `cpu` \| `auto` \| `cuda` \| `cuda:N` \| `mps` \| `mps:N`. Omitted: the graph's `settings.device`, else `cpu`; with `--devices`, the first device of the set instead (the graph's `settings.device` is then not consulted). `auto`: the best accelerator this server has. The resolved device is the queue it joins. |
| `--target <name>` | Where the run executes: `local` (this server), the name of a connected worker agent, or `any`. See [Workers](#workers). Omitted: `local`. |
| `--devices <list>` | Comma-separated device set of one kind, e.g. `cuda:0,cuda:1`. Its first entry is the run's device when `--device` is omitted, `auto` or only the kind (`cuda`); any other `--device` must equal it. |
| `--seed <n>` | Seed every node from `n`, making the run reproducible. Omitted: the graph's `settings.seed`, if any. A seeded run executes one node at a time — see **[Reproducible runs](./running-graphs#reproducible-runs-seed)**. |
| `--no-seed` | Run unseeded even when the graph has a `settings.seed` (sends `"seed": null`) |
| `--deterministic` | Also ask PyTorch for deterministic kernels (`warn_only`) |
| `--record-outputs` | Capture node outputs for later inspection |
| `--wait` | Stream progress until the run ends (**default**) |
| `--detach` | Print the run id and exit 0 immediately |
| `--timeout <s>` | Stop waiting after N seconds. The run continues on the server. |
| `--host`, `--port` | Server address (defaults to the last server `cdui start` launched) |

While a run is waiting, the CLI reports where it is in line rather than sitting silent:

```text
=== Run submitted ===
  Run ID          3f1c9ab2c04e4d5f8b1a7e6d2c930f45
  Graph           sweep-3.json
  Target          local
  Device          cuda:0
  Status          queued

  ○ queued  position 3 on cuda:0
  ▸ started
  ✓ dataset
    trainer  epoch 3  batch 912/938  loss=0.1187
    trainer  epoch 3/10  loss=0.1235  lr=0.001
  ✓ trainer
  ✓ run complete

  Result          succeeded
```

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | The run succeeded (or `--detach` submitted it successfully) |
| 1 | The run failed, was cancelled or interrupted — or the CLI could not submit it (no server, no graph file, rejected envelope) |
| 2 | Bad command line (argument parsing) |
| 130 | Ctrl+C. Stops **watching** only — the run keeps going on the server, exactly as `--detach` would have left it. |

`--timeout` expiring also exits 1: the command cannot report success for a run it stopped watching.

:::tip No server? Use the offline runner instead
`cdui run` talks to a running server. To execute a graph in-process with no server at all, use the **[CLI Graph Runner](./cli-runner)** — it is unchanged and remains the right tool for a box with no daemon.
:::

## Sweeps

A **sweep** runs one graph many times with different parameter values and ranks the results by a metric — a grid or random search over the queue. `POST /api/sweeps` compiles the search space into one complete copy of the graph per combination, queues every copy as an ordinary run on the queued lane, and answers `201` with `sweep_id`, `total_combinations`, the expanded `params` and one entry per queued variant (`index`, `run_id`, `status`, `seed`, `params`). The three routes are listed in the [API Reference](../advanced/api-reference); the two `POST`s need the session token like every other mutating route, the `GET` is open.

```json
{
  "base_graph": {"nodes": [ ... ], "edges": [ ... ]},
  "name": "lr x batch",
  "sweep_spec": {
    "method": "grid",
    "params": [
      {"node_id": "opt", "param": "lr",
       "range": {"min": 1e-4, "max": 1e-1, "count": 4, "scale": "log", "type": "float"}},
      {"node_id": "loader", "param": "batch_size", "values": [32, 64, 128]}
    ]
  },
  "objective": {"metric": "val_loss", "direction": "minimize"},
  "options": {"device": "cuda:0"}
}
```

Each entry in `params` addresses one parameter of one node by the node's id and carries either an explicit `values` list (no repeats) or a `range` that is expanded for you: `count` points from `min` to `max`, evenly spaced on a `linear` or a base-10 `log` scale (which needs a positive `min`), rounded to whole numbers for `type: int` — a range that collapses onto fewer distinct values simply gets fewer. Every value is checked against the node's definition (type, allowed options, min/max) before anything is queued; a spec that cannot be honoured is refused with a `400` naming the entry, and no partial sweep is left behind. A body that does not fit the request schema is refused earlier, with FastAPI's `422` validation error — for example a missing `objective`, an unknown key at the top level or inside `sweep_spec`, `objective`, a `params` entry or its `range`, a `method`, `scale`, `type` or `direction` outside its listed values, a non-integer `count`, `seed` or `samples`, a `null`, list or object inside `values`, or a blank `objective.metric`. Every other sweep refusal on this page is a `400`, including an unknown key inside `options`.

**Grid or random.** `method: grid` enumerates every combination, the last-listed param varying fastest, and does not accept `samples`. `method: random` draws `samples` distinct combinations and requires both `samples` and a `seed` (0 to 4294967295); the same seed always draws the same combinations. Asking for more samples than the space holds is refused, and so is a sweep that would compile more variants than the cap — it is never silently truncated.

**`objective` is required.** `metric` is the name of a series a node logs (`train_loss`, `val_loss`, `val_accuracy`, `eval_accuracy`, `accuracy`, `mse`, `r2`, `success_rate`, or whatever a plugin node records; the series `TrainingLoop` and the evaluation nodes record are listed in [Running Graphs](./running-graphs#training-loops-and-loss-charts)) and `direction` is `minimize` or `maximize`. The name is not checked at submit time, because no variant has run yet; if no variant ends up recording it, the ranked table comes back empty with an `objective_warning` listing the series the runs did record.

**`objective.node_id` names the node whose series ranks the sweep.** It is optional, and a node inside a block is named by its flattened id (`block/inner`). With it, each variant's objective is the last point that node logged (highest `step`, then the last write), so another node logging the same name cannot change the value. Without it the name alone selects: while exactly one node logs the metric, that node's last point is used. When several nodes log it in a run, that variant gets no objective and no rank, its `ambiguous_producers` lists the nodes, and `objective_warning` says to choose a node. No producer is picked for you. Sweeps created before `node_id` existed follow the same name-only rule: values already harvested are kept, and variants harvested later that several nodes logged stay unranked.

**`options`** go to every variant unchanged (device, `record_outputs`, ...), with three refusals: `options.seed` (the sweep owns seeding), `lane: interactive` (a sweep always queues), and `record_outputs` on a sweep with more variants than the output store keeps (20 by default) — the earliest variants' captures would be evicted before the sweep finished. To seed the training itself, set `sweep_spec.seed` and `"seed_variants": true`: variant *i* then runs with seed `seed + i` (wrapped into the valid seed range). Setting `seed_variants: true` without `sweep_spec.seed` is refused with `400` before any row is created. Each variant is then a seeded run, so it executes one node at a time and nothing in the same process runs alongside it — a seeded sweep on this server is strictly sequential and holds up canvas runs for its whole duration, see [Reproducible runs](./running-graphs#reproducible-runs-seed).

**`options.target`** goes to every variant as well (see [Workers](#workers)). With `"target": {"worker": "any"}` and no `devices`, each variant takes the next free slot of the sweep's device kind on any connected worker, or on this server last, so a sweep of four variants on two agents with one card each runs two variants at once and starts the next one on whichever card frees first. Seeded variants on workers run side by side too, because each executes in a process of its own. A target that a single run would be refused for (a worker this server has never seen, a device nobody offers, `record_outputs` on a worker) refuses the whole sweep with `400` before any row is created. A worker that is away does not fail the sweep, even when it goes away while the variants are being submitted: its variants wait for it to reconnect. Without a target a sweep runs exactly as before: every variant waits in its device's queue on this server.

**What can be swept:** int, float, bool, string and select parameters on registered nodes. Not a preset instance's inner parameters, not a subgraph instance's parameters, not a `SECRET` parameter (the chosen values are stored in the clear on the sweep row), and not a node id that appears twice in the graph.

**Cross-validation is a sweep over a fold.** `KFoldSplit` holds out the fold its `fold` param names, so a grid sweep with `{"node_id": "<KFoldSplit id>", "param": "fold", "values": [0, 1, 2, 3, 4]}` (for `n_folds` 5) runs each fold once, and each variant records its own metric, such as `r2` from `RegressionMetrics`. The mean of the variants' objective values is the cross-validation score; the sweep ranks the folds but does not average them. The folds depend only on `KFoldSplit`'s `n_folds`, `shuffle` and `seed`, never on the run seed, so `seed_variants` does not change which rows each variant holds out.

| Variable | Default | Bounds |
| --- | --- | --- |
| `CODEFYUI_MAX_SWEEP_RUNS` | `32` | Variants in one sweep — the grid size, or `samples` |
| `CODEFYUI_MAX_SWEEP_PARAMS` | `4` | Entries in `params` |
| `CODEFYUI_MAX_SWEEP_DOMAIN` | `32` | Distinct values for one param, after range expansion |

### Reading the results

`GET /api/sweeps/{id}` returns the sweep: its `state` (`running`, `cancelling`, `finished`, or `failed` when the submit loop broke part-way — the children already queued keep running), the objective, per-status `counts`, `params` with each expanded domain, and `variants` **in rank order**, best first. Each variant carries its `index` (submission order), `run_id`, live `status`, the `params` it was given, its `seed`, the `objective` value it reached, `objective_node_id` (the node that value was read from), `ambiguous_producers` (set only for an ambiguous name-only objective), its `rank`, `run_exists` and — while the child run still exists — its `final_metrics`, `worker_id`, `devices` and `queue_key` (where it runs; a waiting `any` variant shows `any` and `any/<device>` until it is placed); `best` names the rank-1 variant. A variant is ranked once its run has ended and recorded the objective, on that series' final value; a run that failed after logging it is still ranked, and unranked variants keep their row with `rank: null` in index order. `?format=csv` downloads the same table as a spreadsheet, with `objective_metric`, `objective_node_id` and `ambiguous_producers` (`;`-separated, with `(run)` for a series logged without a node) columns and one column per swept parameter. `final_metrics` is the Runs table's summary and does not choose the objective: it shows one number per name even where several nodes log it.

Results outlive the children: every `GET /api/sweeps/{id}` and every cancel copies each finished variant's status and objective onto the sweep row, and retention does the same for any unread result before it prunes a run. A child removed by retention therefore keeps its harvested `status`, `objective`, `objective_node_id` and `ambiguous_producers`, with `run_exists: false` and no `final_metrics`. A child deleted with `DELETE /api/runs/{id}` before any read or cancel harvested it shows `status: "missing"` and `objective: null`. The variant's row stays in both cases.

A sweep can carry an `error` while its `state` is still `running` or `finished`. That is retention reporting a harvest it could not finish: the children were deleted on schedule — a delete is never held back so that results can be saved first — but their objectives could not be copied across on the way out, so those numbers are gone for good. The message says how many runs went and what went wrong. Without it an empty comparison table would be indistinguishable from a sweep that simply never recorded anything, and that is the difference the message exists to draw. `state: "failed"` still means only one thing: the submit loop broke part-way.

### Cancelling

`POST /api/sweeps/{id}/cancel` asks every queued or running child to stop — one cooperative cancel each — and reports `cancelled` and `already_finished` counts plus a per-variant list in index order. A variant is in neither count when it never got a run, or when its run still shows `running` but the cancel could not reach it (for example, a run started by another server process using the same database). The sweep's state becomes `cancelling` only if at least one child was still active, and it never becomes `cancelled`: a sweep whose first thirty variants finished and last two were stopped is a finished sweep with two cancelled rows. Once every child is terminal, the next read (the cancel reply itself, if they already were) settles the state to `finished`. Asking twice is harmless (`cancelled: 0`).

A cancel is a request, not a guarantee. Stopping is cooperative: the server raises a flag and the node has to read it. The training, evaluation and sampling nodes check it every batch and stop within one, but a node that never checks runs to completion exactly as if nobody had asked. For a run in the server process there is no force-stop behind the request and no timeout: the only way to impose a deadline there is to kill the run mid-step, which throws away the work it has done and, on a GPU, can leave the device unusable until the server exits. A run on a [worker](#workers) has a process of its own, so the worker terminates it when it has not stopped within the grace period (30 seconds by default), and only that run's process ends.

So `cancelling` has no time bound. It ends when the last child reaches a terminal status, not after any particular wait; restarting the server ends it too, because runs abandoned by the old process are retired on the way back up. To see what a cancel is still waiting on, read `counts.running` plus `counts.queued` from `GET /api/sweeps/{id}`: those are the children that were asked and have not stopped, and `variants` names them. How long the sweep has been showing `cancelling` tells you nothing.

### In the Runs panel

**New sweep** in the [Runs panel](#runs-panel) toolbar starts a sweep from the open graph. The dialog lists the int, float, bool and select parameters of the graph's top-level registered nodes; inside a block it still lists the top-level graph, because that is the graph a sweep runs. Free-text string parameters can be swept through the API only. Each parameter takes a list of values or, for a number, a range, and the dialog shows how many variants that makes. Past one of the three defaults in the table above it warns and names the setting, but still sends the sweep: the server may have been configured higher, and it refuses what it cannot run. The objective defaults to `train_loss`, which `TrainingLoop` records every epoch, and the field suggests the series recorded by the runs in the list. **Objective node** lists the nodes those runs saw log the chosen metric; **Any node** sends the name alone, and the dialog warns when the listed runs show more than one node logging it. The variants get the tab's device and run settings, but not its seed (use **Seed every variant**), and they do not record outputs. **Target** and **Devices** are the [Queue on...](#workers) dialog's: **This server**, a connected worker with a device set, or **Any available** with a device kind, which spreads the variants over every free card (see `options.target` above). The picker starts on **This server** and the tab's device, listing `auto` (the server picks) or a device this server lacks as it is; left there, the sweep is sent with the tab's device and no target, as before. A sweep whose **Name** is left blank is named after the tab, cut to 64 characters.

**Start sweep** opens the sweep: its state, the objective, a count per status, and a table with one row per variant showing its parameter values, objective, rank, final metrics, and the **Worker** and **Devices** it runs on (`any` until an `any` variant is placed), with the objective curves below. The view reads the sweep again every 2 seconds until it has finished. The **Rank**, **Status** and **Objective** headers sort the table; rows with no value stay at the bottom. **Open run** shows that variant's run, and **Open parent sweep** in a child run's detail comes back. **Stop sweep** sends `POST /api/sweeps/{id}/cancel`, and **Download CSV** downloads the `?format=csv` table.

The children are ordinary runs: they appear in the Runs panel list under the sweep's `name` followed by the variant's number as the sweep's table shows it, its `index` plus one (`lr x batch #3`), with the sweep's name cut if the two would pass 64 characters; the children of an unnamed sweep are unnamed. Each row from `GET /api/runs` carries `sweep_id` and `sweep_variant`. Follow one variant with `GET /api/runs/{id}/events` as for any other run. The editor has no list of past sweeps; open one again from a child run's **Open parent sweep**.

## Studies

A **study** is an experiment the server owns from start to finish: a set of graph variants, each run a fixed number of times with a different seed per repetition, ranked on one exact objective. Graph Copilot used to hold this plan in the browser tab that started it, so closing the tab abandoned the study. A study submitted to `POST /api/studies` lives in the server's database: it keeps running after the tab closes, survives a server restart, and resumes without re-running a repetition that already completed. Any client can list studies and follow one by id; nothing ties a study to the tab or plugin that submitted it.

```json
{
  "name": "lr study",
  "objective": {"node_id": "train", "metric": "val_loss", "direction": "minimize"},
  "base_graph": {"nodes": [ ... ], "edges": [ ... ]},
  "variants": [
    {"label": "baseline", "ops": []},
    {"label": "lr 1e-3", "ops": [{"node_id": "opt", "param": "lr", "value": 0.001}]},
    {"label": "wide", "graph": {"nodes": [ ... ], "edges": [ ... ]}}
  ],
  "repetitions": 3,
  "seed": 7,
  "options": {"device": "cuda:0"},
  "budget": {"max_runs": 12, "max_parallel_runs": 1},
  "resume_policy": "explicit"
}
```

Each variant is either a whole `graph` or a list of `ops` applied to `base_graph` (`[]` is the base graph itself); its `label` is required and unique within the study. An op sets one parameter of one node and is checked exactly as a sweep parameter is: the node must exist once, be a registered node that is not a preset or subgraph instance, and the value must fit the parameter's type, options and bounds. `SECRET` parameters cannot be set by an op. Up to `CODEFYUI_MAX_SWEEP_RUNS` (32) variants and 32 `repetitions` are accepted.

The study, its variants and every attempt are written in one transaction, so a refused request (`400`, or `422` for a body that does not fit the schema) leaves nothing behind. The reply is `201` with the study as `GET /api/studies/{id}` returns it, plus `created: true`. The routes are listed in the [API Reference](../advanced/api-reference#runs-and-sweeps); the `POST`s need the session token, the `GET`s are open.

### Submitting a study

**`objective` names an exact producer.** `node_id`, `metric` and `direction` are all required. A repetition's value is the last point that node logged under that series name (highest `step`, then the last write), which is the rule a sweep with `objective.node_id` uses. There is no name-only form: a ranking that may promote a variant needs one unambiguous number per run.

**Seeds.** Repetition `r` of every variant runs with seed `(seed + r) mod 2^32`. All variants of one repetition share its seed, so a comparison within a repetition is paired, and a retried repetition reuses its seed. Leave `seed` out and the server picks one and records it. `options.seed` is refused because the study owns seeding, and so is `lane: interactive`, because a study's children always queue. Every child is a seeded run, so it executes one node at a time and nothing runs alongside it, as for a seeded sweep.

**Idempotency.** Send an `Idempotency-Key` header (or the body field `idempotency_key`; the two must agree when both are sent). A retry of the same request with the same key returns the same study with `200` and `created: false` and creates no runs, so a client that lost the first reply can safely send it again. The same key with a different request is a `409` naming the existing study. Requests are compared after `SECRET` values are blanked, so a retry that differs only in a secret value counts as the same request.

**Budget.** `budget.max_parallel_runs` (default 1, up to 32) is how many of the study's children may be queued or running at once. Every child is seeded, so children that execute on this server still run one at a time, and a higher value only puts the next ones in the queue ahead of time. `options.target` reaches every child as it does a sweep's variants, and the same `400` applies before any row is written (a retry with the same idempotency key returns the existing study, or `409`, before that check); children placed on workers run in processes of their own, so with `"target": {"worker": "any"}` and two free cards, `max_parallel_runs: 2` runs two children at once. `budget.max_runs` (optional) caps the attempts the study may submit in total, retries included; it must be at least `variants x repetitions`. A study that runs out of `max_runs` with repetitions still to run ends `failed`. Limits on per-run time and total active time are not enforced yet.

**Order.** Children are submitted repetition by repetition: every variant's repetition 0, then every variant's repetition 1, and so on. A study that stops part-way has therefore compared every variant on the same seeds. Each child is an ordinary run named `<study name> / <label> / r<n>`, and each row from `GET /api/runs` carries `study_id` and `study_attempt_id`.

### Reading a study

`GET /api/studies/{id}` returns the study's `status`, `revision` (it goes up with every change to the study or its attempts), `counts` of the current attempts by status, and one entry per variant with:

- `repetitions`: for each repetition, its seed and its current attempt.
- `attempts`: every attempt ever made, with `attempt` (1, 2, ...), `run_id`, `status`, `error`, timestamps, `objective_value`, `final_metrics` and `fingerprint_digest`, plus `worker_id`, `devices` and `queue_key` while its child run exists (where it runs, as for a sweep variant). Each repetition's `current` attempt carries the same fields.
- `aggregate`: `count`, `mean` and `std` over the current attempts that recorded the objective, and `values`, the raw value of each repetition (`null` where there is none). `std` is the sample standard deviation and is `null` with fewer than two values.
- `eligible`, `ineligible_reasons`, `rank`, `tie` and `tied_with`.

`rankings` lists the ranked variants best first. A variant is **eligible** only when every scheduled repetition's current attempt succeeded and recorded a finite value for the objective. A repetition that failed is a result and is never retried; it makes its variant ineligible even when it logged a number before failing, and that number is still shown in `values`. This is stricter than the sweep table, which ranks a failed run on the value it reached. Ranks are competition ranks over the eligible means: variants whose means are exactly equal share a rank, are marked `tie: true` and list each other in `tied_with`, and the next rank skips accordingly (1, 1, 3). No run-time tie-break is applied, and the result does not depend on the order variants were listed in.

A study's evidence outlives its children. When a child ends, its objective value, final metrics, start and finish times and environment digest are copied onto its attempt, and retention copies anything not yet copied before it prunes a run. After retention has deleted a study's children, `GET /api/studies/{id}` still returns every value and ranking; only `run_id` points at a run that no longer exists.

`GET /api/studies/{id}/events?cursor=` returns the study's audit log after a cursor: `study_created`, `study_started`, `attempt_submitting`, `attempt_submitted`, `attempt_started`, `attempt_finished`, `attempt_relinked`, `attempt_lost`, `attempt_added`, `study_cancel_requested`, `study_cancelled`, `study_interrupted`, `study_resumed`, `study_failed` and `study_finished` (whose payload carries the final ranking). Cursors are gapless and every payload carries the `revision` the change produced. `?wait=` long-polls up to 60 seconds while the study is active. `GET /api/studies` lists studies newest first; `?status=queued|running|cancel_requested` finds the ones still in flight.

### Cancelling and resuming a study

`POST /api/studies/{id}/cancel` stops scheduling and sends a cooperative cancel to each of the study's queued or running children; `cancelled_runs` lists the runs it reached. The study reads `cancel_requested` until every child has stopped, then `cancelled`. Completed attempts are kept, and attempts that were never submitted stay `pending`. Cancelling a cancelled study again changes nothing; a `succeeded` study cannot be cancelled (`409`).

`POST /api/studies/{id}/resume` continues an `interrupted`, `cancelled` or `failed` study. A repetition whose current attempt succeeded or failed is kept as it is and never run again. A repetition whose current attempt was cancelled, interrupted or lost gets a new attempt record with the next attempt number and the same seed, and the old attempt stays in `attempts` as history. Pending repetitions are then submitted as usual. Resuming any other status is a `409`.

A child cancelled directly with `POST /api/runs/{id}/cancel` makes its study `interrupted`. The study does not retry the run on its own; resume it to run that repetition again.

### Studies across a restart

The run queue does not survive a restart, so the server retires a study's queued and running children as `interrupted` when it starts again (see [When the server stops](#when-the-server-stops)). The study coordinator then reconciles every study that was still active: each child's evidence is copied onto its attempt, a study whose cancel was in progress becomes `cancelled`, and any other unfinished study becomes `interrupted` and waits. A study with a child the restarted server took over from a worker is not settled: that child counts as running, the study carries on with it, and if the child ends `interrupted` the study becomes `interrupted` and waits for a resume (or resumes itself with `"resume_policy": "auto"`). A study whose cancel was in progress asks its taken-over child to stop again and becomes `cancelled` once it has. Nothing is resubmitted until you call `resume`, unless the study was submitted with `"resume_policy": "auto"`, in which case it resumes itself after the restart.

Each attempt records the run id it is about to use before the child is submitted, and the child carries the attempt's id. If the server stops, or a write fails, between submitting a child and recording that it was submitted, the study finds the child by that stamp and links it back (`attempt_relinked`) instead of submitting a second one. If no child carries the stamp, the submission never created a run; the attempt is marked `lost` and the repetition gets a new attempt.

`SECRET` parameter values are kept only in the memory of the server that received the study, as for any queued run. A study resumed after a restart runs with those values blank.

## When the server stops

Nothing resumes a queue across a restart: the queue lives in the server process's memory (workers keep no queue of their own), and a waiting row would otherwise sit forever waiting on a scheduler that no longer exists.

A graceful shutdown immediately retires every waiting run as `interrupted`, writes its normal stop event, and gives executing runs 5 seconds to stop cooperatively. `systemctl stop` on the unit in [Deployment](./deployment#a-systemd-unit) shuts the server down this way, because it sends SIGTERM to the server process itself. On Windows, so does Ctrl+C on `cdui start --foreground`, pressed once: a second Ctrl+C while uvicorn is waiting for connections to close skips these steps.

On Linux and macOS, Ctrl+C on `cdui start --foreground` is not graceful: the `cdui` process kills the server about 0.25 seconds after the Ctrl+C, so executing runs do not get their 5 seconds and waiting runs may not be retired. `cdui stop` does not wait for these steps either: on Windows it force-kills the server's process tree (`taskkill /F /T`), and on Linux and macOS it sends SIGTERM and then SIGKILL about 2 seconds later. Those stops, any other hard kill, or an executing run that outlasts the 5 seconds can leave rows as `queued` or `running`. At the next startup, recovery changes both statuses to `interrupted` and writes the same stop event a graceful shutdown writes, unless the run's event log already ends with a closing event. An editor tab that reconnects, and anything reading `GET /api/runs/{id}/events`, therefore sees the run end. Neither resumes.

A run executing on a [worker](#workers) gets the same treatment on a graceful shutdown: the stop reaches the worker, a run still going after the 5 seconds is terminated there, and the server then closes its worker connections. The agents keep running and reconnect when the server is back. After a hard stop the worker is not told anything: its run subprocesses keep running, and the next start takes those runs over instead of retiring them; they continue in the same log once their worker reconnects, or end `interrupted` if it does not within `CODEFYUI_WORKER_RESUME_GRACE_S` ([Remote Workers, When the connection drops](./remote-workers#when-the-connection-drops)).

Requeue anything you still want by submitting it again.

A study keeps its plan across a restart and runs the repetitions that did not complete once it is resumed; see [Studies across a restart](#studies-across-a-restart).

## Configuration

All of these are environment variables read at startup (like every other setting — the `CODEFYUI_` prefix).

| Variable | Default | Meaning |
| --- | --- | --- |
| `CODEFYUI_RUN_QUEUE_MAX_CONCURRENT_GPU` | `1` | Concurrent runs per accelerator queue |
| `CODEFYUI_RUN_QUEUE_MAX_CONCURRENT_CPU` | `2` | Concurrent runs on the `cpu` queue |
| `CODEFYUI_RUN_QUEUE_MAX_CONCURRENT` | *(empty)* | Per-slot overrides. A bare device is this server's (`cuda:0=2,cpu=8`); `<worker>/<device>` is that worker's (`gpu-box-1/cuda:0=2`) and never this server's. Beats the two defaults above. |
| `CODEFYUI_RUN_INTERACTIVE_MAX_CONCURRENT` | `2` | Concurrent canvas runs |
| `CODEFYUI_WORKER_RESUME_GRACE_S` | `300` | Seconds a disconnected worker's runs wait for it before they end `interrupted` (see [Remote Workers, When the connection drops](./remote-workers#when-the-connection-drops)). `0` ends them at once. |
| `CODEFYUI_MAX_PARALLEL_NODES` | `4` | Nodes executing concurrently inside one run |
| `CODEFYUI_RUN_RETENTION_KEEP_LAST` | `200` | How many finished runs to keep. `0` keeps none; a negative value disables retention. |
| `CODEFYUI_RUN_EVENT_PAYLOAD_CAP_BYTES` | `131072` (128 KB) | Largest event payload stored and fanned out; an over-cap output entry is replaced by an elision marker that still names its port. `0` or less disables the cap. |
| `CODEFYUI_RUN_EVENTS_RESPONSE_CAP_BYTES` | `4194304` (4 MB) | Byte budget for one `GET /api/runs/{id}/events` page; a page that stops early hands back a cursor to resume from. |
| `CODEFYUI_MOUNT_LABEL` | `data` | The label of this server's data folder for workers that share it ([Remote Workers, Storage](./remote-workers#storage-shared-filesystem)). |

For example, with `gpu-box-1` a worker on another machine, `CODEFYUI_RUN_QUEUE_MAX_CONCURRENT="cuda:0=2,gpu-box-1/cuda:0=3"` lets two runs share this server's `cuda:0` and three share `cuda:0` on `gpu-box-1`, and leaves every other card at 1. A card that a worker on the same machine shares with this server always uses this server's limit; a `<worker>/<device>` override naming such a card is ignored, with a warning when the worker connects. `local/<device>` is accepted as another spelling of a bare device.

A malformed override entry is ignored with a warning rather than failing startup, and a limit below 1 is treated as 1 — a queue that cannot drain is a hang, not a policy. The three sweep caps are listed under [Sweeps](#sweeps).

Retention runs at startup and after every run ends, keeping the newest `KEEP_LAST` finished runs (active runs are never pruned but still count towards the window). Unlike **Delete** in the Runs panel, pruning also removes the pruned runs' auto-written checkpoints and TensorBoard directories — a checkpoint from a run that ended `interrupted` is kept, so a crash never loses its resume point.

## See also

- [Running Graphs](./running-graphs) — what happens inside a single run
- [Multi-GPU Execution](./multi-gpu) — worker agents, device sets and slots on a multi-GPU machine
- [CLI Graph Runner](./cli-runner) — executing a graph with no server
- [API Reference](../advanced/api-reference) — the `/api/runs`, `/api/sweeps` and `/api/studies` endpoints
