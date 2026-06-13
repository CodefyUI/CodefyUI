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

`activate` may be `async`. The editor awaits it before marking the plugin as ready. Errors thrown inside `activate` are caught, logged to the browser console, and do not crash other plugins.

## CodefyUIPluginAPI v1 reference

### `api.ui` — editor UI

| Method | Signature | Description |
|--------|-----------|-------------|
| `addFloatingWidget` | `(id, element, options?) => void` | Mount any DOM element as a floating, draggable panel. `id` must be unique. `options.title` sets the panel header. |
| `toast` | `(message, level?) => void` | Show a transient notification. `level` is `"info"` (default), `"warning"`, or `"error"`. |

### `api.graph` — graph read and write

| Method | Signature | Description |
|--------|-----------|-------------|
| `getGraph` | `() => GraphSnapshot` | Return a deep copy of the current graph state (nodes, edges, params). |
| `getNodeDefinitions` | `() => NodeDefinition[]` | Return the full node palette: types, port schemas, param schemas. |
| `applyOperations` | `(ops: GraphOp[]) => Promise<ApplyResult>` | Apply a batch of graph operations. The entire batch is committed as a **single undo snapshot**. |
| `onGraphChanged` | `(callback: (snapshot: GraphSnapshot) => void) => () => void` | Subscribe to graph changes. Returns an unsubscribe function. |

#### GraphOp table

All seven operation types share the property `op` (the discriminant string).

| `op` | Required fields | Description |
|------|-----------------|-------------|
| `"add_node"` | `type: string`, `id?: string`, `x?: number`, `y?: number` | Add a node of the given type. `id` is auto-generated if omitted. |
| `"remove_node"` | `id: string` | Remove a node and all edges connected to it. |
| `"add_edge"` | `from_node: string`, `from_port: string`, `to_node: string`, `to_port: string` | Connect two compatible ports. |
| `"remove_edge"` | `from_node: string`, `from_port: string`, `to_node: string`, `to_port: string` | Disconnect the specified edge. |
| `"set_param"` | `node_id: string`, `param: string`, `value: unknown` | Set a node parameter value. |
| `"move_node"` | `id: string`, `x: number`, `y: number` | Reposition a node on the canvas. |
| `"clear_graph"` | *(none)* | Remove all nodes and edges. |

#### ApplyResult shape

```ts
interface ApplyResult {
  ok: boolean;           // true if all ops succeeded
  applied: string[];     // ids of ops that were applied
  failed: { op: GraphOp; reason: string }[];  // ops that were skipped
}
```

**Batch semantics:** All ops in a single `applyOperations` call form one undo snapshot — pressing Ctrl+Z after an AI edit undoes the entire batch at once. Ops are applied in order; a failing op is skipped and reported in `failed`, but the remaining ops continue. Node `id` refs created by an earlier `add_node` within the same batch are available to subsequent ops in that batch.

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

The snippet below is the pattern used by the official Graph Copilot demo. It adds a single toolbar button that inserts two compatible nodes and wires them together.

```js
// frontend/index.js
export default function activate(api) {
  const btn = document.createElement("button");
  btn.textContent = "Insert Linear + ReLU";
  btn.style.cssText =
    "padding:6px 12px;background:#0d9488;color:#fff;border:none;border-radius:4px;cursor:pointer";

  btn.addEventListener("click", async () => {
    const result = await api.graph.applyOperations([
      { op: "add_node", type: "Linear", id: "lin1", x: 200, y: 200 },
      { op: "add_node", type: "ReLU",   id: "relu1", x: 440, y: 200 },
      { op: "add_edge",
        from_node: "lin1", from_port: "output",
        to_node: "relu1", to_port: "input" },
    ]);
    if (!result.ok) {
      api.ui.toast(`Some ops failed: ${result.failed.map(f => f.reason).join(", ")}`, "warning");
    }
  });

  api.ui.addFloatingWidget("demo-insert-panel", btn, { title: "Demo" });
}
```

## See also

- [Plugins](/advanced/plugins) — installing packs, the manifest format, and the `cdui plugin` CLI.
- [Graph Copilot](/advanced/graph-copilot) — the first production consumer of the frontend extension API.
- [API Reference](/advanced/api-reference) — backend REST endpoints, including `/api/llm/chat`.
