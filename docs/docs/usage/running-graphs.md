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

## Stopping

Click **Stop** to cancel an in-flight run. The WebSocket connection also reconnects automatically if it drops mid-session.

## Beyond the browser

You can run any saved graph from the command line without starting the server — see the **[CLI Graph Runner](./cli-runner)**.
