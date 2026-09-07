---
sidebar_position: 3.5
title: Run Queue
description: Queue graphs per device and walk away — lanes, FIFO order, concurrency limits, cancelling a waiting run, and the cdui run CLI.
---

# Run Queue

A run belongs to the **server**, not to whatever submitted it. Close the tab, close the terminal, log out — the run keeps going, and you can come back to it later over the API or the CLI.

That is what makes queueing useful. Submit five training jobs, shut the laptop lid, and the server works through them one at a time on the GPU instead of starting all five at once and running out of VRAM forty minutes in.

## One queue per device

Every run is scheduled against its **resolved device** — `cpu`, `cuda:0`, `mps`. That string is the queue key, so each device has its own independent line and its own concurrency limit.

| Queue key | Runs at once (default) | Why |
| --- | --- | --- |
| `cuda:0`, `cuda:1`, `mps`, any accelerator | 1 | A second run on the same card competes for the same VRAM. The usual result of guessing otherwise is a CUDA out-of-memory error partway through a long job. |
| `cpu` | 2 | Here the failure mode is contention, not death — and a machine with cores to spare should use them. |

A saturated GPU never delays a CPU run, and two different cards never wait on each other. Runs within one queue start in **submit order**, oldest first.

:::note Aliases of one card share one queue
`cuda` and `cuda:0` are the same physical device, so they are canonicalised onto a single queue key (`cuda:0`) rather than becoming two independent FIFOs over one card. Same for `mps` and `mps:0`. On a multi-GPU box `cuda` follows the process's current device.
:::

:::note `--device auto` resolves to CPU today
Device resolution maps an unknown or `auto` request to `cpu`, so `auto` currently queues on the CPU line. Ask for `cuda` (or `cuda:0`) explicitly to use a card.
:::

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

The **Runs** tab of the results panel lists every run the server owns — started from any tab, `cdui run` or the API — newest first, with filter chips **All / Running / Queued / Succeeded / Failed / Cancelled / Interrupted** and the columns **Run, Status, Device, Started, Duration, Final loss**. A waiting run shows `Queue #N` beside its device. Each row offers up to four actions:

| Action | What it does |
| --- | --- |
| **Stop** | Asks a queued or running run to stop — cooperatively, see [Stopping](./running-graphs#stopping). |
| **Watch** | Streams that run into the active tab's Execution Log, replaying it from the start. This is how you pick up a run submitted from a terminal or from another tab; the tab stops following whatever it was watching before, and the runs themselves are unaffected. |
| **CSV** | Downloads the run's metrics (`GET /api/runs/{id}/metrics?format=csv`). |
| **Delete** | Finished runs only. Removes the run's metrics, event log, artifact records and any captured outputs; checkpoint files on disk are kept. |

Click a row for its detail: the seed and **Deterministic** when they were set, the error if it failed, the metrics chart with its own **Download CSV**, the recorded artifacts each with a **Copy path** button, and the last 200 events of its log, which keeps updating while the run is active. Loading the editor while runs are still in progress shows a toast that counts them and points you here.

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
| `--device <dev>` | `cpu` \| `auto` \| `cuda` \| `cuda:N` \| `mps` (default `auto`, which resolves to `cpu` today). The resolved device is the queue it joins. |
| `--seed <n>` | Seed every node from `n`, making the run reproducible. A seeded run executes one node at a time — see **[Reproducible runs](./running-graphs#reproducible-runs-seed)**. |
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
  Device          cuda:0
  Status          queued

  ○ queued  position 3 on cuda:0
  ▸ started
  ✓ dataset
    trainer  epoch 3/10  loss=0.1235
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

Each entry in `params` addresses one parameter of one node by the node's id and carries either an explicit `values` list (no repeats) or a `range` that is expanded for you: `count` points from `min` to `max`, evenly spaced on a `linear` or a base-10 `log` scale (which needs a positive `min`), rounded to whole numbers for `type: int` — a range that collapses onto fewer distinct values simply gets fewer. Every value is checked against the node's definition (type, allowed options, min/max) before anything is queued; a spec that cannot be honoured is refused with a `400` naming the entry, and no partial sweep is left behind.

**Grid or random.** `method: grid` enumerates every combination, the last-listed param varying fastest, and does not accept `samples`. `method: random` draws `samples` distinct combinations and requires both `samples` and a `seed` (0 to 4294967295); the same seed always draws the same combinations. Asking for more samples than the space holds is refused, and so is a sweep that would compile more variants than the cap — it is never silently truncated.

**`objective` is required.** `metric` is the name of a series a node logs (`train_loss`, `val_loss`, `eval_accuracy`, or whatever a plugin node records) and `direction` is `minimize` or `maximize`. The name is not checked at submit time, because no variant has run yet; if no variant ends up recording it, the ranked table comes back empty with an `objective_warning` listing the series the runs did record.

**`options`** go to every variant unchanged (device, `record_outputs`, ...), with three refusals: `options.seed` (the sweep owns seeding), `lane: interactive` (a sweep always queues), and `record_outputs` on a sweep with more variants than the output store keeps (20 by default) — the earliest variants' captures would be evicted before the sweep finished. To seed the training itself, set `sweep_spec.seed` and `"seed_variants": true`: variant *i* then runs with seed `seed + i` (wrapped into the valid seed range). Setting `seed_variants: true` without `sweep_spec.seed` is refused with `400` before any row is created. Each variant is then a seeded run, so it executes one node at a time and nothing runs alongside it — a seeded sweep is strictly sequential and holds up canvas runs for its whole duration, see [Reproducible runs](./running-graphs#reproducible-runs-seed).

**What can be swept:** int, float, bool, string and select parameters on registered nodes. Not a preset instance's inner parameters, not a subgraph instance's parameters, not a `SECRET` parameter (the chosen values are stored in the clear on the sweep row), and not a node id that appears twice in the graph.

| Variable | Default | Bounds |
| --- | --- | --- |
| `CODEFYUI_MAX_SWEEP_RUNS` | `32` | Variants in one sweep — the grid size, or `samples` |
| `CODEFYUI_MAX_SWEEP_PARAMS` | `4` | Entries in `params` |
| `CODEFYUI_MAX_SWEEP_DOMAIN` | `32` | Distinct values for one param, after range expansion |

### Reading the results

`GET /api/sweeps/{id}` returns the sweep: its `state` (`running`, `cancelling`, `finished`, or `failed` when the submit loop broke part-way — the children already queued keep running), the objective, per-status `counts`, `params` with each expanded domain, and `variants` **in rank order**, best first. Each variant carries its `index` (submission order), `run_id`, live `status`, the `params` it was given, its `seed`, the `objective` value it reached, its `rank`, `run_exists` and — while the child run still exists — its `final_metrics`; `best` names the rank-1 variant. A variant is ranked once its run has ended and recorded the objective, on that series' final value; a run that failed after logging it is still ranked, and unranked variants keep their row with `rank: null` in index order. `?format=csv` downloads the same table as a spreadsheet with one column per swept parameter.

Results outlive the children: each finished variant's objective is copied onto the sweep row, and retention harvests any unread result before it prunes a run. A pruned or deleted child shows as `status: "missing"` with `run_exists: false`, and its row stays.

### Cancelling

`POST /api/sweeps/{id}/cancel` asks every queued or running child to stop — one cooperative cancel each — and reports `cancelled` and `already_finished` counts plus a per-variant list in index order. The sweep's state becomes `cancelling` only if at least one child was still active, and it never becomes `cancelled`: a sweep whose first thirty variants finished and last two were stopped is a finished sweep with two cancelled rows. Once every child is terminal, the next read (the cancel reply itself, if they already were) settles the state to `finished`. Asking twice is harmless (`cancelled: 0`).

### Where the variants show up

There is no sweep view in the editor yet. The children are ordinary runs: they appear in the [Runs panel](#runs-panel) under the sweep's `name`, and each row from `GET /api/runs` carries `sweep_id` and `sweep_variant`. Follow one variant with `GET /api/runs/{id}/events` as for any other run.

## When the server stops

Nothing resumes a queue across a restart: the schedule lives in the server's memory, and a waiting row would otherwise sit forever waiting on a scheduler that no longer exists.

A graceful stop (`cdui stop`) immediately retires every waiting run as `interrupted`, writes its normal stop event, and asks executing runs to stop cooperatively. A hard kill — or an executing task that outlasts the graceful-shutdown timeout — can leave rows as `queued` or `running`. At the next startup, recovery changes both statuses to `interrupted`; neither resumes.

Requeue anything you still want by submitting it again.

## Configuration

All of these are environment variables read at startup (like every other setting — the `CODEFYUI_` prefix).

| Variable | Default | Meaning |
| --- | --- | --- |
| `CODEFYUI_RUN_QUEUE_MAX_CONCURRENT_GPU` | `1` | Concurrent runs per accelerator queue |
| `CODEFYUI_RUN_QUEUE_MAX_CONCURRENT_CPU` | `2` | Concurrent runs on the `cpu` queue |
| `CODEFYUI_RUN_QUEUE_MAX_CONCURRENT` | *(empty)* | Per-key overrides, e.g. `cuda:0=2,cpu=8`. Beats the two defaults above. |
| `CODEFYUI_RUN_INTERACTIVE_MAX_CONCURRENT` | `2` | Concurrent canvas runs |
| `CODEFYUI_MAX_PARALLEL_NODES` | `4` | Nodes executing concurrently inside one run |
| `CODEFYUI_RUN_RETENTION_KEEP_LAST` | `200` | How many finished runs to keep. `0` keeps none; a negative value disables retention. |
| `CODEFYUI_RUN_EVENT_PAYLOAD_CAP_BYTES` | `131072` (128 KB) | Largest event payload stored and fanned out; an over-cap output entry is replaced by an elision marker that still names its port. `0` or less disables the cap. |
| `CODEFYUI_RUN_EVENTS_RESPONSE_CAP_BYTES` | `4194304` (4 MB) | Byte budget for one `GET /api/runs/{id}/events` page; a page that stops early hands back a cursor to resume from. |

A malformed override entry is ignored with a warning rather than failing startup, and a limit below 1 is treated as 1 — a queue that cannot drain is a hang, not a policy. The three sweep caps are listed under [Sweeps](#sweeps).

Retention runs at startup and after every run ends, keeping the newest `KEEP_LAST` finished runs (active runs are never pruned but still count towards the window). Unlike **Delete** in the Runs panel, pruning also removes the pruned runs' auto-written checkpoints and TensorBoard directories — a checkpoint from a run that ended `interrupted` is kept, so a crash never loses its resume point.

## See also

- [Running Graphs](./running-graphs) — what happens inside a single run
- [CLI Graph Runner](./cli-runner) — executing a graph with no server
- [API Reference](../advanced/api-reference) — the `/api/runs` endpoints
