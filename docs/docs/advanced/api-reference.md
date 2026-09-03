---
sidebar_position: 5
title: API Reference
description: The CodefyUI backend REST and WebSocket endpoints — nodes, presets, graphs, plugins, the LLM proxy, models, images, and execution outputs.
---

# API Reference

The backend serves a REST API plus a WebSocket for execution. All endpoints are under the same origin as the app (`http://localhost:8000` by default).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health probe — returns `version`, `boot_id` (which process answered, so a client can tell a restart from a server that never went down), `nodes_loaded`, `presets_loaded`, and `caches` (current bytes held by each in-memory store against its budget; see [Training Memory](./training-memory)). |
| `/api/nodes` | GET | List all node definitions. |
| `/api/nodes/{node_name}` | GET | Get a single node definition. |
| `/api/nodes/reload` | POST | Hot-reload all built-in and custom nodes. |
| `/api/presets` | GET | List preset definitions. |
| `/api/presets/{name}` | GET | Get a single preset definition. |
| `/api/presets/create` | POST | Create a new preset from selected nodes. |
| `/api/graph/validate` | POST | Validate a graph. |
| `/api/graph/save` | POST | Save a graph. |
| `/api/graph/load/{name}` | GET | Load a saved graph. |
| `/api/graph/list` | GET | List saved graphs. |
| `/api/graph/export` | POST | Export a single-file headless Python runner. It embeds the graph and requires a compatible CodefyUI backend environment, but no running server. |
| `/api/examples/list` | GET | List example graphs. |
| `/api/examples/load` | GET | Load an example graph. |
| `/api/custom-nodes` | GET | List custom nodes. |
| `/api/custom-nodes/upload` | POST | Upload a custom node. |
| `/api/custom-nodes/toggle` | POST | Enable/disable a custom node. |
| `/api/custom-nodes/{filename}` | DELETE | Delete a custom node. |
| `/api/plugins` | GET | List installed plugin packs. |
| `/api/plugins/catalog` | GET | Every plugin this build can install by name, merged with what is installed — one row per plugin, each carrying the state it is in (installed, disabled, removed on purpose, files gone). |
| `/api/plugins/generation` | GET | The reload counter the editor polls to notice the palette changed. |
| `/api/plugins/{id}` | GET | Get a plugin's manifest + README. |
| `/api/plugins/reload` | POST | Hot-reload all node and preset sources. |
| `/api/plugins/inspect` | POST | Read one source — a catalog name, `owner/repo`, or a URL — at one resolved commit and say what installing it would cost. Installs nothing; the answer is kept under an `inspection_id`. |
| `/api/plugins/install` | POST | Install the manifest an inspection described, by its `inspection_id` — `202` with a `job_id`. Nothing about what gets installed comes from the request body. |
| `/api/plugins/jobs/{job_id}/events` | GET | An install job's log and progress after `?cursor=`; `?wait=` long-polls for up to 60s, and the job may end `needs_restart` with the command to run with the server stopped. |
| `/api/plugins/jobs/{job_id}/cancel` | POST | Ask the running install to stop, cleanly enough that nothing half-written is left behind. |
| `/api/plugins/{id}/update` | POST | Re-read the plugin's own repository — `202 {"job_id"}`, `200 {"status": "up_to_date", "sha"}`, or `200 {"status": "needs_consent", "inspection", "capabilities_added", "allowed_modules_added"}` when this version asks for more than you granted last time, which the client finishes with `POST /api/plugins/install {"inspection_id", "accept_capabilities", "trust_author"}` and no `force`. The update keeps the plugin's enabled state. A built-in or linked plugin — or a repository whose manifest now declares a different plugin id — answers `400 not_updatable`. |
| `/api/plugins/{id}` | DELETE | Uninstall a plugin and report what that left behind. A built-in pack keeps its files and is remembered as removed; a linked directory is untouched; the plugin's Python packages are never removed and the command that would remove them is returned. |
| `/api/plugins/{id}/enable` | POST | Turn an installed plugin on and re-discover. |
| `/api/plugins/{id}/disable` | POST | Turn it off without uninstalling it. |
| `/api/packs` | GET | List every optional pack with what is installed, what a download would cost, and whether this machine can install it. |
| `/api/packs/{id}/install` | POST | Start an install job — `202` with a `job_id`. One job runs at a time. |
| `/api/packs/jobs/{job_id}/cancel` | POST | Ask the running job to stop. A download is aborted mid-file, not at the end of it — which also interrupts any Hugging Face download a running graph happens to be doing at that moment (a dataset or tokenizer fetch), because the two share one transfer session. |
| `/api/packs/jobs/{job_id}/events` | GET | A job's log and progress events after `?cursor=`; `?wait=` long-polls for up to 60s so the panel follows a job without a retry loop. |
| `/api/packs/{id}/items/{item_id}` | DELETE | Delete one downloaded model and free its bytes. A pack's Python packages are not removable from the running server — see `cdui packs remove`. |
| `/api/llm/chat` | POST | Stream a unified SSE chat completion from the configured provider (OpenAI / OpenRouter / Anthropic / OpenAI-Codex / custom OpenAI-compatible). |
| `/api/llm/models` | POST | List the models available for a provider. |
| `/api/llm/codex/login` | POST | Start the OpenAI-Codex (ChatGPT account) OAuth login flow. |
| `/api/llm/codex/status` | GET | Report OpenAI-Codex OAuth login status. |
| `/api/llm/codex/logout` | POST | Clear stored OpenAI-Codex OAuth tokens. |
| `/api/models` | GET | List uploaded model files. |
| `/api/models/upload` | POST | Upload a model weight file. |
| `/api/models/download/{filename}` | GET | Download a model weight file (supports nested paths). |
| `/api/models/{filename}` | DELETE | Delete a model file. |
| `/api/images` | GET | List uploaded image files. |
| `/api/images/upload` | POST | Upload an image file. |
| `/api/images/download/{filename}` | GET | Download an image file. |
| `/api/images/{filename}` | DELETE | Delete an image file. |
| `/api/execution/outputs/{run_id}` | GET | List ports captured for a run. |
| `/api/execution/outputs/{run_id}` | DELETE | Clear a captured run. |
| `/api/execution/outputs/{run_id}/{node_id}/{port}` | GET | Fetch a captured tensor (supports `?slice=0,:,:`). |
| `/api/execution/outputs/{run_id}/{node_id}/__steps_index` | GET | Step-trace metadata for a node (Inspector → Steps tab). |
| `/api/execution/outputs/{run_id}/{node_id}/__grad_index` | GET | Captured gradient metadata (Inspector → Backward tab). |
| `/api/execution/state/reset` | POST | Reset persisted layer weights (per-node or per-graph). |
| `/api/execution/state/list` | GET | List how many modules are persisted (diagnostic). |
| `/ws/execution` | WebSocket | Attach/subscribe view over a run: `execute` starts one, `attach` replays its event log from a cursor and then follows it live, `detach` unsubscribes, `cancel` stops it. Closing the socket never cancels. |

:::note WebSocket auth
The execution WebSocket takes its session token as a query parameter, since browsers can't set custom headers on a WebSocket handshake. The frontend handles this for you.
:::

:::note Installing a pack is a local-only operation
Every mutating `/api/packs` route is refused unless the server is bound to loopback: starting an install runs a package manager against the interpreter that is serving the request, and "whoever can reach the port" is the wrong audience for that. A classroom or office instance that deliberately serves the LAN opts back in with `CODEFYUI_ALLOW_REMOTE_PACK_INSTALL=1`. What may be asked for is bounded by the catalog either way — no pip spec, repo id or URL from a request body ever reaches a subprocess.
:::

:::note Installing a plugin is a local-only operation too
The routes that install, inspect or remove — `inspect`, `install`, the job's `cancel`, `update` and `DELETE` — take the session token *and* refuse unless the server is bound to loopback: installing a plugin puts a stranger's code where this process will import it, and inspecting reaches out to GitHub on the caller's word. `CODEFYUI_ALLOW_REMOTE_PLUGIN_INSTALL=1` opts a deliberate classroom or lab server back in. `reload` and `enable` / `disable` take the token but not the loopback gate — they act on code this machine already has. Step labels and failure messages come from the shared install path and are English in every client. See **[Plugin Center](/advanced/plugins#plugin-center)**.
:::

:::note Optional packs in the node list
`/api/nodes` carries `requires_pack` on each node (the pack id it needs before it can run, or `null`) and `option_packs` on each SELECT param (option value → pack id, for the options that need one particular download). Both are there so the editor can grey out what is not installed and offer the install; the run itself is gated in the backend regardless.
:::
