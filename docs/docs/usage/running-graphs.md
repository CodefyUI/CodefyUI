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

## A node without a trigger can still run

Removing a node's trigger edge does not, by itself, take it out of the run. If a **data** edge still connects its output into something that does run — whether that's a required input or an optional one makes no difference — the node runs too, with or without a trigger of its own. A `Dataset` or the first node of a transform chain typically has no trigger at all and is expected to run this way; the same rule applies to anything else you have wired in.

The practical effect: **disconnecting the trigger edge alone no longer parks a branch.** If you want a node to stay wired for later but not run right now, disconnect its **data** edge(s) instead — that is what actually removes it from the run. **Bypass** (right-click a node, or `Ctrl`/`Cmd`+`B`) is a one-click alternative for a node partway through a chain, skipping it while passing its input straight through to whatever it fed — but it only works when the node has an input of the same type as its output to forward, so it is refused on a source node with no inputs at all (`CSVReader`, `ImageReader`, `Dataset`, and the rest of the file-reading nodes). For one of those, disconnecting the edge is the only way to park it.

One consequence worth knowing: a reader node (`CSVReader`, `ImageReader`, and the like) left wired to an input — even an optional one — but never triggered now executes where it used to be silently skipped. If the file it points at has since been deleted or moved, a graph that previously ran without error can start failing with `FileNotFoundError` on that node.

## Training loops and loss charts

The `TrainingLoop` node emits progress events during training. The **Training** tab of the results panel plots a **live loss chart** as epochs complete, so you can watch convergence in real time.

## Partial re-execution (dirty tracking)

CodefyUI tracks which nodes are **dirty**. When you change a node's parameters or inputs, only that node and its **downstream dependencies** are marked for re-execution. Unchanged nodes return cached outputs (shown as `cached`), so iterating on a single hyperparameter re-runs just the affected part of the graph — a big speedup during development.

Deterministic nodes are cached automatically; non-deterministic ones (training loops, random ops, or any node with `cacheable = False`) always re-run.

### Content-aware caching for file-reading nodes

A cache entry is keyed by a hash of the node's type, its parameters, its upstream nodes' cache keys, and the run device. Anything a node reads from *outside* the graph is invisible to `params` alone — a `path` parameter records *where* to read, never *what* is there. A node that reads external state therefore also folds a content fingerprint into its key: the resolved file's size and modification time, plus (for files up to 8 MB) a content hash, so a same-size edit landing inside one filesystem timestamp tick still changes the key. `CSVReader`, `FileReader`, `ImageReader`, `ImageBatchReader`, `ModelLoader`, `CheckpointLoader`, `Dataset` and `ImageFolderDataset` all do this — editing the file (or, for `Dataset`/`ImageFolderDataset`, anything under the directory) and clicking **Run** again gives you the new content; leaving it untouched gets you the cached result instead of a re-read.

`GraphInput` with `type=image` uses the same mechanism on a **canvas** run: the API path already has the caller's value in `params`, but a canvas run instead loads the `default` path from disk, and the fingerprint is what makes editing that image between canvas runs pick up the new pixels.

### What is never cached

Some nodes still opt out of caching entirely with `cacheable = False`, for two different reasons:

**External state a fingerprint can't describe.** `HuggingFaceDataset` and `KaggleDataset` hit the network and, for Kaggle, `KAGGLE_*` environment credentials — a fingerprint of the local cache directory cannot tell "the remote revision changed" or "the credentials changed" from "nothing changed", so both re-execute on every run. `LLMChat` reaches a remote model API for the same reason.

**The node's purpose is a side effect.** `ImageWriter`, `ModelSaver`, and `CheckpointSaver` exist to write a file; a cache hit would return the recorded `{"path": ...}` without touching disk, which is wrong when the node's whole point is the write (deleting the output and re-running must recreate it). These re-execute every run regardless of what feeds them.

**The node might have a side effect and only its author knows.** `PythonScript` runs code you type on the canvas. `code` is a cache-keyed parameter, so an *edited* script re-runs — but an unedited script over unchanged inputs is exactly where a hit happens, and a script can mutate an input tensor or model in place, change process-global `torch`/`numpy` state, or use whatever an `ANY`-typed input port handed it. None of that is visible to the node's type or to any check on its source, so it opts out unconditionally. See [the PythonScript node](../advanced/python-script-node.md#caching).

The same opt-out covers nodes whose output escapes the cache key for other reasons — `GaussianNoise`, `DDPMSampler`, `BackwardOnce`, `DiffusionTrainingLoop`, and every layer that owns weights (`Linear`, `Conv2d`, `LSTM` and the rest), whose parameters drift as training proceeds.

Opting out **propagates downstream**: every node fed by one of these re-executes too, because a cache key records only the *keys* of upstream nodes, not their actual outputs. A cached downstream node would otherwise hand back a stale result computed from data that has since changed.

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
