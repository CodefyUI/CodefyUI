---
sidebar_position: 5
title: API Reference
description: The CodefyUI backend REST and WebSocket endpoints — nodes, presets, graphs, plugins, the LLM proxy, models, images, and execution outputs.
---

# API Reference

The backend serves a REST API plus a WebSocket for execution. All endpoints are under the same origin as the app (`http://localhost:8000` by default).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health probe — returns `nodes_loaded`, `presets_loaded`. |
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
| `/api/plugins/{id}` | GET | Get a plugin's manifest + README. |
| `/api/plugins/reload` | POST | Hot-reload all node and preset sources. |
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
| `/ws/execution` | WebSocket | Real-time graph execution (accepts `run_id`, `record_outputs`). |

:::note WebSocket auth
The execution WebSocket takes its session token as a query parameter, since browsers can't set custom headers on a WebSocket handshake. The frontend handles this for you.
:::
