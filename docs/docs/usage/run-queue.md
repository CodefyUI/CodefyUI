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

`GET /api/runs` reports `queue_position` for every waiting run — 1-based, **within its own device queue**, so a CPU run behind two other CPU runs is third in line no matter how many CUDA runs were submitted before it. `cdui run --wait` prints it while it waits, and a Runs panel will render it once one lands.

Cancelling a run that has not started yet simply removes it from the line. Nothing executed, no device was touched, and the runs behind it move up. The row is recorded as `cancelled` with no start time, and anything following the run receives a normal stop event.

Cancelling a run that is already going is cooperative — see [Running Graphs](./running-graphs#stopping).

:::note Two runs may briefly both read as running on a cap-1 queue
A run gives its device slot back the moment its graph finishes, before it writes its final status and closing event. The next run therefore starts while the previous one is still filing paperwork, so a poll landing in that window can see two `running` rows on a queue whose limit is 1. Nothing is executing twice — the device really is idle — and it is deliberate: the alternative is holding a GPU idle across a database write, and worse, telling a canvas its run is over while the next click is still refused.
:::

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

## When the server stops

Nothing resumes a queue across a restart: the schedule lives in the server's memory, and a waiting row would otherwise sit forever waiting on a scheduler that no longer exists.

So a graceful stop (`cdui stop`) retires every waiting run as `interrupted` on the way out, with a normal stop event. A hard kill leaves those rows `queued`, and the next start retires them the same way. Either path produces the same rows; the graceful one just does it immediately. Runs that were *already executing* stop cooperatively first and record an honest status of their own.

Requeue anything you still want by submitting it again.

## Configuration

All of these are environment variables read at startup (like every other setting — the `CODEFYUI_` prefix).

| Variable | Default | Meaning |
| --- | --- | --- |
| `CODEFYUI_RUN_QUEUE_MAX_CONCURRENT_GPU` | `1` | Concurrent runs per accelerator queue |
| `CODEFYUI_RUN_QUEUE_MAX_CONCURRENT_CPU` | `2` | Concurrent runs on the `cpu` queue |
| `CODEFYUI_RUN_QUEUE_MAX_CONCURRENT` | *(empty)* | Per-key overrides, e.g. `cuda:0=2,cpu=8`. Beats the two defaults above. |
| `CODEFYUI_RUN_INTERACTIVE_MAX_CONCURRENT` | `2` | Concurrent canvas runs |
| `CODEFYUI_RUN_RETENTION_KEEP_LAST` | `200` | How many finished runs to keep |

A malformed override entry is ignored with a warning rather than failing startup, and a limit below 1 is treated as 1 — a queue that cannot drain is a hang, not a policy.

## See also

- [Running Graphs](./running-graphs) — what happens inside a single run
- [CLI Graph Runner](./cli-runner) — executing a graph with no server
- [API Reference](../advanced/api-reference) — the `/api/runs` endpoints
