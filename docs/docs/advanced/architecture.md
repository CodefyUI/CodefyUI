---
sidebar_position: 6
title: Architecture
description: How CodefyUI is put together — backend-authoritative node definitions, WebSocket execution, topological scheduling, the frontend's stores and run events, and the registry/plugin system.
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
- **[Plugin packs](./plugins)** — namespaced node packs that can also ship presets, examples and a frontend; see [Plugins](#plugins) below.
- **[Presets](./presets)** — reusable subgraphs expanded at execution time.

## Frontend data flow

The editor keeps its state in Zustand stores under `frontend/src/store/` and talks to the server two ways: REST calls, mostly from the modules in `frontend/src/api/` (`rest.ts`, plus `git.ts` for Source Control and `executionOutputs.ts` for captured outputs), and one execution WebSocket per canvas tab. Mutating REST calls go through `apiFetch` in `api/_auth.ts`, which adds the session token that `GET /api/auth/bootstrap` hands out; the socket carries the same token in its URL ([Authentication](./api-reference#authentication)).

| Store | Holds | Filled from |
|-------|-------|-------------|
| `nodeDefStore` | Node and preset definitions | `GET /api/nodes` and `GET /api/presets`, at startup, and again whenever the catalog changes: the toolbar's **Reload Nodes**, the Custom Node Manager, **Export as Subgraph** and a change in the Plugin Center |
| `tabStore` | Every open tab: its graph, undo stack, dirty nodes, run status, Execution Log, output summaries, `lastRunId` and socket | IndexedDB at startup, then edits on the canvas and in the panels, and run events |
| `runStore` | The Runs panel: the run list and the selected run | `GET /api/runs` and `GET /api/runs/{id}/events` |

The other stores each back one part of the editor, for example `pluginStore` the Plugin Center, `packStore` the Package Center, `gitStore` Source Control, and `uiStore` the layout and editor settings, including the device chosen in Settings.

### From an edit to a run

1. **Edit** — dropping a node from the sidebar builds it from its definition in `nodeDefStore` and copies the definition and its default params into the node's data, which the card and the config panel render from. `FlowCanvas` draws the active tab with React Flow and writes each move, delete and new connection back through `tabStore`'s `onNodesChange`, `onEdgesChange` and `onConnect`; the config panel writes params through `updateNodeParams`. A param edit marks its node dirty, and a new connection marks the node it feeds.
2. **Autosave** — `tabStore` saves each change to IndexedDB, one record per tab, 250 ms after the last one ([Automatic saving](/usage/tabs-persistence#automatic-saving)). That copy stays in the browser; saving a graph to the server is `POST /api/graph/save`.
3. **Run** — the toolbar's **Run** calls `useGraphExecution`. It checks the canvas on screen for an entry point, then serializes the tab's whole graph without the notes and has `POST /api/graph/validate` check it; a refusal at either step becomes a toast and nothing is sent. It then clears the tab's log, node statuses and output summaries, turns the dirty nodes and everything downstream of them into `changed_nodes`, and sends an `execute` message on the tab's socket with the graph, the device (the graph's own, else the one chosen in Settings) and the tab's run settings, such as the seed and whether to record outputs.
4. **Submit** — `ws_execution.py` hands the graph to the run service on the interactive lane, together with the socket's execution cache and `changed_nodes`, and attaches the socket to the new run. From here the run belongs to the server; see [Execution flow](#execution-flow).

### From run events back to the canvas

1. **Fan-out** — the run service appends every engine event to the run's event log, then passes it to the sockets attached to that run, which send it as `{type, ...payload, run_id, cursor}`. The [WebSocket protocol](./api-reference#websocket-protocol) lists the types.
2. **Per tab** — `useGraphExecution` listens on every tab's socket, so a tab in the background keeps receiving its own run's events.
   - `execution_start` puts the tab into the running state and keeps the run's id as `lastRunId`.
   - `node_status` frames go to `nodeUpdateQueue`, which applies them to `tabStore` once per animation frame, so a run that reports many times a second still re-renders a node card at most once per paint. Text, images, video and charts in the same frames become Execution Log lines, and tensor summaries are what clicking an edge shows after the run.
   - `execution_complete`, `execution_error` and `execution_stopped` set the tab's final status.
3. **Reconnect** — each frame's `run_id` and `cursor` are kept on the tab. After a dropped connection the socket reconnects and re-attaches from the last cursor. After a page reload, a tab that was running asks `GET /api/runs/{id}` whether its run is still going and re-attaches from the start. Closing a tab only detaches it, and the run keeps going; the toolbar's **Stop** sends `cancel` with the run's id.

### Other readers of a run

- The [Inspector](/usage/teaching-inspector#the-inspector-panel) fetches a node's captured outputs for the tab's `lastRunId` from `/api/execution/outputs/`.
- The [Runs panel](/usage/run-queue#runs-panel) does not use the socket: `runStore` polls `GET /api/runs` and long-polls `GET /api/runs/{id}/events` for the selected run. Its **Watch** button attaches the active tab's socket to that run, which replays it into the tab's Execution Log.
- Plugins get these events, renamed into a stable vocabulary, through [`api.events`](./plugin-frontend-extensions#apievents--live-run-events).

## Plugins

A plugin pack is a directory holding a `cdui.plugin.toml` manifest and its content under fixed subdirectory names: `nodes/`, `presets/`, `examples/`, `assets/` and, for a pack with its own UI, `frontend/`. [Plugin Packs](./plugins) and [Plugin Frontend Extensions](./plugin-frontend-extensions) are the references; this is the path a pack takes through the code.

1. **Install** — the Plugin Center and `cdui plugin install` run the same install flow, in `backend/app/core/plugins/`. For a pack from GitHub it reads the manifest at one commit, downloads and unpacks that commit, and runs the AST gate over every file Python could import, then installs the pack's Python dependencies, copies its files and records it in the lockfile, `installed.json`. Built-in packs under `plugins/` and folders linked with `cdui plugin link` are not scanned. See [How an install runs](./plugins#how-an-install-runs) and [Security](./plugins#security--three-tiers).
2. **Load** — at startup and on every reload, `plugin_loader.py` reads the lockfile and exposes each enabled pack as the Python package `cdui_plugins.<id>` (the id in snake_case). The registry walks the pack's `nodes/` the same way it walks the built-in nodes and registers each node as `<id>:<NODE_NAME>`; the pack's presets load the same way. Nothing is scanned here: the gate runs only at install time.
3. **Nodes in the editor** — plugin nodes come from `GET /api/nodes` with every other node, so there is no plugin-specific code to list, configure or run them. On the canvas, a namespaced node without a built-in card (`VIZ_NODE_TYPES`, see [Core principles](#core-principles)) is drawn by `PluginNodeBridge`, which uses the card body the plugin registered for that type, or the default one.
4. **Frontend** — `GET /api/plugins` lists the `frontend_entry` of each enabled pack that declares one, a module served from `/plugins/<id>/frontend/`. `PluginHost` waits for the node catalog, imports each module and calls its default export with the object built in `plugins/api.ts`, which offers panels, toolbar buttons, node renderers, graph reads and writes through `tabStore`, run events and run history. The published type contract is `plugins/contract.ts`: `tsc -b` fails when the host drifts from it, and `scripts/sync_plugin_sdk.py` copies it into the `cdui plugin new` template (a backend test fails while that copy is stale). After a plugin is installed, updated, removed, enabled or disabled from the Plugin Center, the editor reloads the node catalog and activates every plugin frontend again ([The activate contract](./plugin-frontend-extensions#the-activate-contract)).

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
| Tabs and graph state (undo, dirty nodes, autosave) | `frontend/src/store/tabStore.ts`, `frontend/src/store/tabPersistence.ts` |
| Canvas | `frontend/src/components/Canvas/FlowCanvas.tsx` |
| REST client and session token | `frontend/src/api/rest.ts`, `frontend/src/api/_auth.ts` |
| WebSocket client | `frontend/src/api/ws.ts` |
| Run, Stop, and run events on the canvas | `frontend/src/hooks/useGraphExecution.ts`, `frontend/src/store/nodeUpdateQueue.ts` |
| Plugin frontend host and contract | `frontend/src/plugins/PluginHost.tsx`, `frontend/src/plugins/api.ts`, `frontend/src/plugins/contract.ts` |

:::tip Contributing
The backend-authoritative design means most "add a feature" work is a single Python node. See [Custom Nodes](./custom-nodes) to get started, then graduate to a [plugin pack](./plugins) to share it.
:::
