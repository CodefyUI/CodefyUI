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

## Stopping

Click **Stop** to cancel an in-flight run. The WebSocket connection also reconnects automatically if it drops mid-session.

## Beyond the browser

You can run any saved graph from the command line without starting the server — see the **[CLI Graph Runner](./cli-runner)**.
