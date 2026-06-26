---
sidebar_position: 4
title: Plugin Frontend Extensions
description: Ship a JavaScript bundle with your plugin so it can add UI widgets, inspect graphs, and drive the editor — the foundation for Graph Copilot and similar tools.
---

# Plugin Frontend Extensions

A plugin pack can ship a JavaScript bundle alongside its Python nodes. When the CodefyUI editor loads, it discovers and imports that bundle as an ES module, giving the plugin access to a stable JavaScript API for UI, graph manipulation, and proxied HTTP.

:::note Availability
Frontend extensions are in CodefyUI **1.3.0** and later. Check `cdui --version`; if it reports an older version, run `cdui update`.
:::

## Declaring a frontend entry point

Add a `[frontend]` section to `cdui.plugin.toml`:

```toml
[plugin]
id = "my-plugin"
name = "My Plugin"
version = "0.1.0"
requires_codefyui = ">=1.3.0"

[frontend]
entry = "frontend/index.js"
```

`requires_codefyui` is advisory metadata (it is recorded but not currently enforced at install time); set it to the first CodefyUI release that ships the features your plugin depends on — frontend extensions landed in 1.3.0.

The `entry` path must be **relative to the plugin root** and must live under `frontend/`. The file must be a valid ES module with a default export (see [The activate contract](#the-activate-contract) below).

## How the editor serves and discovers the bundle

When the backend starts, it mounts each installed plugin's `frontend/` directory at:

```
/plugins/<plugin-id>/frontend/<file>
```

The plugin listing endpoint exposes the entry point so the editor can load it:

```
GET /api/plugins
```

Example response excerpt:

```json
{
  "id": "my-plugin",
  "name": "My Plugin",
  "version": "0.1.0",
  "frontend_entry": "/plugins/my-plugin/frontend/index.js"
}
```

If `frontend_entry` is `null`, the plugin has no frontend bundle. The editor only loads the module when `frontend_entry` is non-null.

## The activate contract

Your bundle must export a single default function named `activate`. The editor calls it once at startup, after all plugins are loaded, passing the `CodefyUIPluginAPI` object:

```js
// frontend/index.js
export default function activate(api) {
  // api is a CodefyUIPluginAPI instance
}
```

The editor calls `activate` once per page load and does **not** await its return value — do your setup synchronously (you may still start async work; the editor just won't wait for it). Errors thrown synchronously inside `activate` are caught per-plugin, logged to the browser console, and surfaced as a toast; they cannot crash the editor or other plugins. The import is also bounded by a 10-second timeout. (Only the *default export being a function* is required; the name `activate` is convention.)

## CodefyUIPluginAPI v1 reference

### `api.ui` — editor UI

| Method | Signature | Description |
|--------|-----------|-------------|
| `addFloatingWidget` | `({ id }) => HTMLElement` | Create (or reuse) a container `<div>` in the editor's floating-widget stack and return it. `id` must be unique per plugin. You own the returned element — fill it with your own DOM, or mount a React root into it. |
| `toast` | `(message, level?) => void` | Show a transient notification. `level` is `"info"` (default), `"warning"`, or `"error"`. |

### `api.graph` — graph read and write

| Method | Signature | Description |
|--------|-----------|-------------|
| `getGraph` | `() => GraphSnapshot` | Return a deep copy of the current graph state (nodes, edges, params). |
| `getNodeDefinitions` | `() => NodeDefinition[]` | Return the full node palette: types, port schemas, param schemas. |
| `applyOperations` | `(ops: GraphOp[]) => ApplyResult` | Apply a batch of graph operations **synchronously** (returns the result directly — not a Promise). The whole batch is committed as a **single undo snapshot**. |
| `onGraphChanged` | `(callback: (snapshot: GraphSnapshot) => void) => () => void` | Subscribe to graph changes. Returns an unsubscribe function. |

#### GraphOp table

All seven operation types share the property `op` (the discriminant string). Field names below are exact.

| `op` | Fields | Description |
|------|--------|-------------|
| `"add_node"` | `node_type: string`, `ref?: string`, `params?: Record<string, unknown>`, `position?: { x: number; y: number }` | Add a node of the given type. `ref` is a caller-chosen alias that later ops in the same batch can use in place of the generated node id. `position` defaults to a staggered layout. |
| `"connect"` | `source: string`, `source_handle: string`, `target: string`, `target_handle: string` | Connect an output handle to an input handle. `source`/`target` accept a node id or a `ref` from an earlier `add_node`. Use `source_handle: "trigger"` for a trigger edge. |
| `"set_params"` | `node_id: string`, `params: Record<string, unknown>` | Merge parameter values into a node. |
| `"remove_node"` | `node_id: string` | Remove a node and all edges connected to it. |
| `"remove_edge"` | `source: string`, `target: string`, `source_handle?: string`, `target_handle?: string` | Disconnect matching edge(s) between two nodes. |
| `"clear_graph"` | *(none)* | Remove all nodes and edges. |
| `"auto_layout"` | *(none)* | Re-run the automatic graph layout. |

#### ApplyResult shape

```ts
interface OpResult {
  index: number;      // the op's position in the batch
  ok: boolean;        // whether this op applied
  error?: string;     // failure reason when ok is false
  node_id?: string;   // resolved node id (add_node / set_params)
}

interface ApplyResult {
  results: OpResult[];            // one entry per op, in input order
  refs: Record<string, string>;  // ref alias -> generated node id
  node_count: number;            // node count after the batch
  edge_count: number;            // edge count after the batch
}
```

**Batch semantics:** All ops in a single `applyOperations` call form one undo snapshot — pressing Ctrl+Z after an AI edit undoes the entire batch at once. Ops are applied in order; a failing op is skipped and reported in its `results` entry (`ok: false` plus an `error`), while the remaining ops continue. A `ref` alias created by an earlier `add_node` in the same batch is available to later ops, and is echoed back in `refs`.

### `api.http` — session-aware fetch

| Method | Signature | Description |
|--------|-----------|-------------|
| `fetch` | `(path: string, init?: RequestInit) => Promise<Response>` | Identical to the browser `fetch` API, but automatically attaches the CodefyUI session token header. `path` must be a relative path (e.g., `/api/llm/chat`). Use this for all calls to the CodefyUI backend. |

### `api.storage` — namespaced key-value store

Storage is backed by `localStorage` and automatically namespaced to your plugin id, so different plugins cannot collide.

| Method | Signature | Description |
|--------|-----------|-------------|
| `get` | `(key: string) => string \| null` | Retrieve a stored value. |
| `set` | `(key: string, value: string) => void` | Store a value. |
| `remove` | `(key: string) => void` | Delete a key. |

## Trust model

Plugin JavaScript runs inside the editor page with **full access to the editor DOM, graph state, and session token**. Only install plugins from sources you trust. The `cdui plugin install` CLI prints a warning whenever a plugin declares a frontend entry point.

The backend AST security gate applies to plugin Python; there is no sandbox for plugin JavaScript — it runs with the same trust level as the editor itself.

## Minimal working example

The snippet below uses only the raw API — no build step, no framework: a single button that inserts two nodes and wires them together. (For a real React-based panel, see the Graph Copilot plugin source.)

```js
// frontend/index.js
export default function activate(api) {
  const btn = document.createElement("button");
  btn.textContent = "Insert Linear + ReLU";
  btn.style.cssText =
    "padding:6px 12px;background:#0d9488;color:#fff;border:none;border-radius:4px;cursor:pointer";

  btn.addEventListener("click", () => {
    // applyOperations is synchronous — no await.
    const result = api.graph.applyOperations([
      { op: "add_node", node_type: "Linear", ref: "lin1", position: { x: 200, y: 200 } },
      { op: "add_node", node_type: "ReLU",   ref: "relu1", position: { x: 440, y: 200 } },
      // Handle names ("output"/"input" here) come from each node's port schema —
      // call api.graph.getNodeDefinitions() to discover them.
      { op: "connect",
        source: "lin1", source_handle: "output",
        target: "relu1", target_handle: "input" },
    ]);
    const failed = result.results.filter((r) => !r.ok);
    if (failed.length > 0) {
      api.ui.toast(`Some ops failed: ${failed.map((r) => r.error).join(", ")}`, "warning");
    }
  });

  // addFloatingWidget returns a container <div> you fill yourself.
  const panel = api.ui.addFloatingWidget({ id: "demo-insert-panel" });
  panel.appendChild(btn);
}
```

## See also

- [Plugins](/advanced/plugins) — installing packs, the manifest format, and the `cdui plugin` CLI.
- [Graph Copilot](/advanced/graph-copilot) — the first production consumer of the frontend extension API.
- [API Reference](/advanced/api-reference) — backend REST endpoints, including `/api/llm/chat`.
