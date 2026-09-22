---
sidebar_position: 6
title: Architecture
description: How CodefyUI is put together — backend-authoritative node definitions, WebSocket execution, topological scheduling, and the registry/plugin system.
---

# Architecture

```
frontend/   React 19 · TypeScript · React Flow 12 · Zustand 5 · Vite 6
backend/    Python 3.10+ · FastAPI · PyTorch
```

A single uvicorn process serves the REST API, the execution WebSocket, and the prebuilt React app.

## Core principles

| Principle | Detail |
|-----------|--------|
| **Backend-authoritative** | `GET /api/nodes` returns every node definition. Adding a backend node makes it appear in the UI automatically — no frontend changes. |
| **Default node renderer** | One component (`BaseNode`) renders any node from its backend definition, so a new backend node needs no frontend code. A frontend allowlist (`VIZ_NODE_TYPES`) gives nine teaching nodes a richer renderer built on the same card, a plugin node can draw its own card body through the [plugin frontend API](./plugin-frontend-extensions) (`api.nodes.registerRenderer`), and presets, subgraph instances, `Start` and notes have components of their own. |
| **WebSocket execution** | `ws://host/ws/execution` is a *view* over a server-owned run: it streams per-node status live and can replay a run's stored event log from any cursor, so a reconnecting tab catches up without gaps. Runs themselves are owned by the run service, not the socket. |
| **Topological execution** | Kahn's algorithm for DAG sort + cycle detection, with parallel execution of independent nodes. |

## Execution flow

1. **Submission** — the canvas, `cdui run`, `POST /api/runs`, or a sweep submits the graph to the run service. The service persists and schedules it. Queued-lane runs enter a per-device FIFO; interactive canvas runs bypass the queue. See [Run Queue](/usage/run-queue).
2. **Expansion** — before anything runs, notes are dropped; subgraph instances are inlined (at most 10 levels deep; a block that contains itself is refused), and their inner nodes run as `<instance>/<node>`; preset nodes are flattened into their internal nodes (at most 10 levels); and bypassed nodes are removed, with each of their outputs forwarded from a matching input. See [Subgraphs](./subgraphs).
3. **Validation** — DAG check, port/type safety, and a required [`Start`](/usage/first-graph) node. A node runs if it is reachable via trigger edges, OR if it feeds a data connection — required or optional port, does not matter — into one that is (directly or transitively). A root with no trigger of its own, like a `Dataset` or the head of a transform chain, is retained rather than pruned out from under a node that consumes its output (core#201).
4. **Topological sort** — Kahn's algorithm with cycle detection.
5. **Parallel execution** — independent nodes run concurrently, at most `CODEFYUI_MAX_PARALLEL_NODES` at a time. A run with a seed executes one node at a time, because per-node seeding sets process-wide random number generators.
6. **Caching / dirty tracking** — deterministic node outputs are cached per WebSocket connection (256 entries and 1 GB by default), keyed by node type, params, one reference per incoming edge (the upstream key plus both port names), the resolved device, and a content fingerprint for nodes that read files; changing a node marks it and its downstream dirty so only the affected subgraph re-runs. Non-deterministic nodes (or `cacheable = False`) always run.
7. **Device resolution** — no device means `cpu`, and `auto` means the best accelerator (`cuda`, then `mps`, then `cpu`). An unavailable `cuda` or `mps` falls back to CPU with a warning; an out-of-range or malformed `cuda:N` on a machine with CUDA uses the current CUDA device instead. A node's own `device` parameter, when it is not `auto`, overrides the run's device. See [Device Backends](./device-backends).

## State, outputs, and gradients

- **Run service** — `RunService` owns each run independently of its WebSocket connection. It appends every engine event to a durable log, batches scalar metrics, and cancels cooperatively through the execution context. At startup, it marks `queued` or `running` rows left by the previous process as `interrupted`. Runs, events, metrics, and artifacts are stored in SQLite (`exec_runs`, `exec_run_events`, `exec_run_metrics`, `exec_run_artifacts`), allowing a tab to reconnect or a terminal to monitor a run started by another client.
- **Execution context** carries per-run options: device, seed and deterministic flag, verbose mode, weight persistence, backward-pass and gradient settings, and the cooperative stop flag.
- **Stateful modules** — a mixin persists `nn.Module` weights between runs via a key-value store keyed by (graph id, node id, structure hash), so a model keeps learning across **Run** clicks when *Persist weights* is on.
- **Run output store** — a server-wide in-memory store retains captured outputs for the [Teaching Inspector](/usage/teaching-inspector) and serves them on demand over REST. By default, it tracks at most 20 runs and 2 GiB, evicting complete oldest runs when either limit is exceeded.
- **Backward pass** — when *Capture gradients* is on, the engine attaches hooks, calls `.backward()`, and stores per-layer gradients alongside outputs.
- **Step traces** — in verbose mode, instrumented nodes emit a `__steps__` trace recorded for the Inspector's **Steps** tab.

## Node registry & extensibility

- The **registry** discovers `BaseNode` subclasses by walking the node packages. Built-in nodes use bare names (`Conv2d`); plugin nodes are namespaced (`foundations:Edu-KNN`) to prevent collisions and self-document graphs.
- **[Custom nodes](./custom-nodes)** — drop a `.py` file in `custom_nodes/` and hot-reload.
- **[Plugin packs](./plugins)** — installed from the editor's **Plugin Center** or with `cdui plugin install` (one install implementation, `backend/app/core/plugins/`, serves both), discovered through a lockfile, and **AST-validated** before third-party code is loaded.
- **[Presets](./presets)** — reusable subgraphs expanded at execution time.

## Entry points

| Area | File |
|------|------|
| FastAPI app, lifespan, routes | `backend/app/main.py` |
| BaseNode ABC | `backend/app/core/node_base.py` |
| Node registry + namespacing | `backend/app/core/node_registry.py` |
| Graph validation + execution | `backend/app/core/graph_engine.py` |
| Run service (scheduling, event log, cancel, recovery) | `backend/app/core/run_service.py` |
| Run store (SQLite rows, retention) | `backend/app/core/run_store.py` |
| Run REST routes | `backend/app/api/routes_runs.py` |
| Sweeps | `backend/app/api/routes_sweeps.py`, `backend/app/core/sweep_compiler.py` |
| WebSocket handler | `backend/app/api/ws_execution.py` |
| Plugin discovery | `backend/app/core/plugin_loader.py` |
| Plugin AST gate | `backend/app/core/plugins/gate.py`, `backend/app/core/plugin_validator.py` |
| Plugin install service (Plugin Center and `cdui plugin`) | `backend/app/core/plugins/`, `backend/app/api/routes_plugins.py` |
| Package Center (optional packs) | `backend/app/core/packs/`, `backend/app/api/routes_packs.py` |
| Install job runner (one pack or plugin job at a time) | `backend/app/core/jobs.py` |
| Source Control (git) | `backend/app/core/git/`, `backend/app/api/routes_git.py` |
| Published apps and API keys | `backend/app/api/routes_apps.py`, `backend/app/api/routes_keys.py`, `backend/app/core/api_keys.py` |
| SQLite database (apps, keys, runs) | `backend/app/core/db.py` |
| Graph as a function | `backend/app/api/routes_graph_run.py`, `backend/app/core/api_contract.py` |
| LLM provider proxy | `backend/app/core/llm_proxy/`, `backend/app/api/routes_llm.py` |
| Project directories | `backend/app/core/project.py` |
| Host allowlist, session token, body cap | `backend/app/core/auth.py`, `backend/app/core/body_limit.py` |
| Python export | `backend/app/core/codegen.py` |
| CLI graph runner | `backend/run_graph.py` |
| Frontend root | `frontend/src/App.tsx` |
| WebSocket client | `frontend/src/api/ws.ts` |

:::tip Contributing
The backend-authoritative design means most "add a feature" work is a single Python node. See [Custom Nodes](./custom-nodes) to get started, then graduate to a [plugin pack](./plugins) to share it.
:::
