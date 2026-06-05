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
| **Single BaseNode component** | One React component renders all node types, parameterized by the backend definitions. |
| **WebSocket execution** | `ws://host/ws/execution` streams per-node status; REST handles graph CRUD and output fetching. |
| **Topological execution** | Kahn's algorithm for DAG sort + cycle detection, with parallel execution of independent nodes. |

## Execution flow

1. **Preset expansion** — preset nodes are flattened into their internal nodes before anything runs.
2. **Validation** — DAG check, port/type safety, and a required [`Start`](/usage/first-graph) node. Only nodes reachable via trigger edges execute.
3. **Topological sort** — Kahn's algorithm with cycle detection.
4. **Parallel execution** — independent nodes run concurrently.
5. **Caching / dirty tracking** — deterministic node outputs are cached keyed by node type, params, and upstream outputs; changing a node marks it and its downstream dirty so only the affected subgraph re-runs. Non-deterministic nodes (or `cacheable = False`) always run.
6. **Device resolution** — the requested device is checked against what's available and falls back to CPU with a warning. See [Device Backends](./device-backends).

## State, outputs, and gradients

- **Execution context** carries per-run options: device, verbose mode, weight persistence, and gradient targets.
- **Stateful modules** — a mixin persists `nn.Module` weights between runs via a key-value store keyed by (graph id, node id, structure hash), so a model keeps learning across **Run** clicks when *Persist weights* is on.
- **Run output store** — an LRU cache (last ~20 runs) holds captured outputs for the [Teaching Inspector](/usage/teaching-inspector), fetched on demand over REST.
- **Backward pass** — when *Capture gradients* is on, the engine attaches hooks, calls `.backward()`, and stores per-layer gradients alongside outputs.
- **Step traces** — in verbose mode, instrumented nodes emit a `__steps__` trace recorded for the Inspector's **Steps** tab.

## Node registry & extensibility

- The **registry** discovers `BaseNode` subclasses by walking the node packages. Built-in nodes use bare names (`Conv2d`); plugin nodes are namespaced (`foundations:Edu-KNN`) to prevent collisions and self-document graphs.
- **[Custom nodes](./custom-nodes)** — drop a `.py` file in `custom_nodes/` and hot-reload.
- **[Plugin packs](./plugins)** — installed via CLI, discovered through a lockfile, and **AST-validated** before third-party code is loaded.
- **[Presets](./presets)** — reusable subgraphs expanded at execution time.

## Entry points

| Area | File |
|------|------|
| FastAPI app, lifespan, routes | `backend/app/main.py` |
| BaseNode ABC | `backend/app/core/node_base.py` |
| Node registry + namespacing | `backend/app/core/node_registry.py` |
| Graph validation + execution | `backend/app/core/graph_engine.py` |
| WebSocket handler | `backend/app/api/ws_execution.py` |
| Plugin discovery + AST gate | `backend/app/core/plugin_loader.py` |
| CLI graph runner | `backend/run_graph.py` |
| Frontend root | `frontend/src/App.tsx` |
| WebSocket client | `frontend/src/api/ws.ts` |

:::tip Contributing
The backend-authoritative design means most "add a feature" work is a single Python node. See [Custom Nodes](./custom-nodes) to get started, then graduate to a [plugin pack](./plugins) to share it.
:::
