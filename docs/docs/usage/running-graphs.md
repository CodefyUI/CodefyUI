---
sidebar_position: 3
title: Running Graphs
description: How execution works — WebSocket streaming, the results panel, live loss charts, and partial re-execution.
---

# Running Graphs

When you click **Run**, the frontend sends the graph to the backend over a WebSocket (`ws://host/ws/execution`) and the backend streams results back as each node completes.

## Real-time execution

- The backend validates the graph (DAG check, type safety, at least one [`Start`](./first-graph) node), topologically sorts it (Kahn's algorithm, with cycle detection), and runs independent nodes in parallel.
- Each node reports status as it goes: `running` → `completed` (or `error`), with a small **output summary** embedded inline for quick viewing.
- The **Execution Log** tab shows this per-node progress and any `Print` node output.

## Training loops and loss charts

The `TrainingLoop` node emits progress events during training. The **Training** tab of the results panel plots a **live loss chart** as epochs complete, so you can watch convergence in real time.

## Partial re-execution (dirty tracking)

CodefyUI tracks which nodes are **dirty**. When you change a node's parameters or inputs, only that node and its **downstream dependencies** are marked for re-execution. Unchanged nodes return cached outputs (shown as `cached`), so iterating on a single hyperparameter re-runs just the affected part of the graph — a big speedup during development.

Deterministic nodes are cached automatically; non-deterministic ones (training loops, random ops, or any node with `cacheable = False`) always re-run.

### What is never cached

A cache entry is keyed by a hash of the node's type, its parameters, its upstream nodes' cache keys, and the run device. Anything a node reads from *outside* the graph is invisible to that key — the key records a file **path**, never the bytes at that path. Nodes that reach for external state therefore opt out of caching entirely and re-execute on every run:

| Node | External state it reads |
| --- | --- |
| `CSVReader`, `FileReader` | A file on disk |
| `ImageReader`, `ImageBatchReader` | An image file, or every image in a directory |
| `Dataset`, `HuggingFaceDataset`, `KaggleDataset` | Downloaded dataset files (plus the network, and `KAGGLE_*` credentials for Kaggle) |
| `ModelLoader`, `CheckpointLoader` | A `.pt` / `.pth` weights or checkpoint file |
| `LLMChat` | A remote model API |

One known exception: `GraphInput` with `type=image` stays cacheable. API callers send the image with the request, so it lands in the node's parameters and the key stays complete — but a **canvas** run instead loads the `default` path from disk, and only the path is in the key. Editing that image between canvas runs can therefore serve a stale tensor; a follow-up issue tracks closing the gap.

So editing a CSV on disk and clicking **Run** again gives you the new rows: the reader never replays the tensor it built last time. The same opt-out covers nodes whose output escapes the cache key for other reasons — `GaussianNoise`, `DDPMSampler`, `BackwardOnce`, `DiffusionTrainingLoop`, and every layer that owns weights (`Linear`, `Conv2d`, `LSTM` and the rest), whose parameters drift as training proceeds.

Opting out **propagates downstream**: every node fed by one of these re-executes too, because a cache key records only the *keys* of upstream nodes, not their actual outputs. A cached downstream node would otherwise hand back a stale result computed from the old file.

The trade-off is deliberate — a graph that starts from a file reader re-reads that file on every run. Correctness first: the alternative (hashing file size and modification time into the key) is a possible future optimization, not something you can rely on today.

## Reproducible runs (seed)

By default a run draws its randomness from whatever entropy PyTorch picks, so two runs of the same graph give slightly different weights, a different shuffle order, and therefore a different loss curve. Set a **Random seed** in **Settings → Training** to make a run repeatable.

With a seed set:

- Every node is seeded from a value derived from `(seed, node id)`, so what a node draws depends on the seed and on its own identity — not on how much randomness the rest of the graph consumed first, and not on the order the engine happened to schedule things in.
- `DataLoader` gets its own generator, so the epoch shuffle order is fixed too, and each worker process gets its own independent stream.
- **The run executes one node at a time**, and **it does not overlap another run.** Seeding writes process-global RNG state, so a second node — or a second *run* — drawing from it at the same moment moves the numbers. A seeded run therefore waits for the runs already in flight, then runs alone, and anything submitted behind it waits for it. Unseeded runs are unaffected: they still run in parallel with each other, and their nodes still run in parallel.

Two runs of the same graph with the same seed produce bitwise-identical loss curves on CPU. Different seeds produce genuinely different ones.

"Another run" means every graph this server is executing, including a headless `POST /api/graph/run/{name}` call: it waits while a seeded run is going, and a seeded run waits for it. Headless calls that are not seeded still overlap each other freely.

The cost is worth stating plainly: a seeded run is slower (about 3-4x on a graph of independent branches, near zero on the usual mostly-linear teaching graph), and it can wait behind a long job even when started from the canvas. Reproducibility is opt-in, and a run that asked for it would rather be late than wrong.

**Deterministic algorithms** is the other half. It asks PyTorch for kernels that combine those draws the same way every time (`torch.use_deterministic_algorithms(True, warn_only=True)`). It is `warn_only` on purpose: an operation with no deterministic implementation prints a warning rather than killing the run, so you get "everything reproducible was made reproducible" instead of a stack trace from inside cuDNN.

The seed is stored with the run and shown in the **Runs** panel, so an interesting result can always be traced back to the settings that produced it.

From the CLI:

```bash
cdui run graph.json --seed 1234 --deterministic
```

:::note
A seed fixes the *software's* randomness. Exact bitwise agreement is promised on CPU; across different GPUs, driver versions or PyTorch builds, floating-point reduction order can still differ.
:::

## Stopping

Click **Stop** to cancel an in-flight run. **Stop is the only thing that cancels a run.**

Closing the browser tab, navigating away, or losing the connection no longer stops anything: the run lives on the server, not in the socket. When you come back, the tab reconnects to the run it was watching, replays everything you missed into the results panel, and carries on following it live — so a long training job survives a reload, a laptop lid, or a flaky Wi-Fi hop.

Cancelling is cooperative rather than immediate, because there is no safe way to interrupt arbitrary node code partway through. The long-running nodes check for it every batch, step or item — a training loop stops within one batch and writes an interrupt checkpoint on its way out — and every other node runs to the end of its current call, after which the run stops at the next node boundary. Either way it is recorded as `cancelled`.

## Beyond the browser

You can run any saved graph from the command line without starting the server — see the **[CLI Graph Runner](./cli-runner)**.
