---
sidebar_position: 4
title: Plugin Frontend Extensions
description: Ship a JavaScript bundle with your plugin so it can add UI widgets, inspect graphs, and drive the editor — the foundation for Graph Copilot and similar tools.
---

# Plugin Frontend Extensions

A plugin pack can ship a JavaScript bundle alongside its Python nodes. When the CodefyUI editor loads, it discovers and imports that bundle as an ES module, giving the plugin access to a stable JavaScript API for UI, graph manipulation, and proxied HTTP.

:::note Availability
Frontend extensions are in CodefyUI **1.3.0** and later. Check `cdui --version`; if it reports an older version, run `cdui update`.

Dock panels, toolbar buttons, execution events and the runs facade need **apiVersion 3** (CodefyUI 1.5.0 and later); `graph.getView` needs **apiVersion 4** (CodefyUI 2.3.0 and later). Feature-check before you use them — see [API versions](#api-versions).
:::

## API versions

`api.apiVersion` is a number that only ever grows, and every release so far has been **purely additive**: nothing that worked at an older version has been removed or changed shape. A plugin written for apiVersion 2 keeps working on an apiVersion 4 editor with no changes at all.

| `apiVersion` | CodefyUI | Added |
|--------------|----------|-------|
| 1 | 1.3.0 | `ui.addFloatingWidget`, `ui.toast`, `graph.*`, `http.fetch`, `storage.*` |
| 2 | 1.3.0 | `nodes.registerRenderer` |
| 3 | 1.5.0 | `ui.addPanel` / `removePanel`, `ui.addToolbarButton` / `removeToolbarButton`, `events.onExecution`, `runs.*` |
| 4 | 2.3.0 | `graph.getView` — which level of the graph the user is looking at |

Check it before reaching for anything newer than the version you require, and degrade rather than throw:

```js
export default function activate(api) {
  if (api.apiVersion >= 3) {
    mountDashboard(api.ui.addPanel({ id: "dash", title: "Dashboard" }));
  } else {
    mountDashboard(api.ui.addFloatingWidget({ id: "dash" }));
  }
}
```

Because the additions are additive, a breaking change would come with an `apiVersion` bump and a migration note — never silently.

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

## CodefyUIPluginAPI reference

### `api.ui` — editor UI

| Method | Signature | Description |
|--------|-----------|-------------|
| `addFloatingWidget` | `({ id }) => HTMLElement` | Create (or reuse) a container `<div>` in the editor's floating-widget stack and return it. `id` must be unique per plugin. You own the returned element — fill it with your own DOM, or mount a React root into it. |
| `toast` | `(message, level?) => void` | Show a transient notification. `level` is `"info"` (default), `"warning"`, or `"error"`. |
| `addPanel` | `(opts) => HTMLElement` | **apiVersion 3.** Register a dock panel and return its container element. |
| `removePanel` | `(id: string) => void` | **apiVersion 3.** Remove one of your panels. |
| `addToolbarButton` | `(opts) => () => void` | **apiVersion 3.** Add a toolbar button; returns a remove function. |
| `removeToolbarButton` | `(id: string) => void` | **apiVersion 3.** Remove one of your buttons by id. |

#### Dock panels

Requires `api.apiVersion >= 3`.

```ts
interface PluginPanelOptions {
  id: string;                 // unique within your plugin
  title: string;              // tab label, or right-hand section heading
  icon?: string;              // short glyph shown before the title
  dock?: "bottom" | "right";  // defaults to "bottom"
  onShow?: () => void;        // the element was attached to the document
  onHide?: () => void;        // ...and detached again
}
```

A `"bottom"` panel becomes a tab in the editor's bottom dock, after Execution Log, Training and Runs. A `"right"` panel becomes a section in the right-hand column, alongside the node config and inspector panels. The host owns the tab chrome, the ordering and the placement; you own everything inside the element.

**The element is yours for the life of the panel.** Its identity never changes, so mount into it exactly once:

```js
const el = api.ui.addPanel({ id: "runs", title: "My Runs", icon: "~" });
createRoot(el).render(<MyPanel />);   // once, not per tab switch
```

The editor mounts only the *active* dock tab, so the container your panel sits in is torn down and rebuilt as the user moves between tabs. Your element is not: the editor detaches it and re-attaches it, with its children and their state intact. Calling `addPanel` again with the same `id` returns the same element and just updates the title, icon and dock.

What that costs you is the one thing to be careful about: your code keeps *running* while the panel is off screen, rendering into an element that is not in the document. If the panel does anything expensive — a chart, a poll, an animation — gate it:

```js
api.ui.addPanel({
  id: "runs", title: "My Runs",
  onShow: () => chart.start(),
  onHide: () => chart.stop(),
});
```

Write both callbacks so that calling them twice in a row is harmless: React's development mode replays mount effects, so a single tab switch can produce an extra `onHide`/`onShow` pair.

Panels are removed automatically when your plugin is unloaded or hot-reloaded; `removePanel` is for panels you want gone earlier.

#### Toolbar buttons

Requires `api.apiVersion >= 3`.

```ts
interface PluginToolbarButtonOptions {
  id: string;        // unique within your plugin
  icon: string;      // short glyph — the toolbar has room for a glyph
  tooltip: string;   // hover and accessible text; an icon is not a label
  onClick: () => void;
}
```

```js
const remove = api.ui.addToolbarButton({
  id: "sweep", icon: "~", tooltip: "Start a sweep",
  onClick: () => startSweep(),
});
```

Buttons land in one group at the right of the toolbar, in registration order. There is no way to ask for a position, and the editor decides how many are shown: on a wide window up to three sit inline, and on a narrow one they collapse into a single overflow menu. That is what keeps five installed plugins from pushing Run off the toolbar, so write the `tooltip` as if it were the label — in the menu, it is.

If `onClick` throws, the editor logs it and carries on; the toolbar is not affected.

Re-adding an id replaces the button. The remove function you get back belongs to that one registration, so if you have since replaced the button, calling the older remove function does nothing rather than taking the replacement down with it. When you mean "remove whatever is under this id", call `removeToolbarButton(id)`. Buttons are removed automatically when your plugin is unloaded or hot-reloaded.

### `api.graph` — graph read and write

| Method | Signature | Description |
|--------|-----------|-------------|
| `getGraph` | `() => GraphSnapshot` | Return a deep copy of the **whole** graph state (nodes, edges, params, plus block definitions under `subgraphs`) — always the top level, whatever the user has open. |
| `getNodeDefinitions` | `() => NodeDefinition[]` | Return the full node palette: types, port schemas, param schemas. |
| `applyOperations` | `(ops: GraphOp[]) => ApplyResult` | Apply a batch of graph operations **synchronously** (returns the result directly — not a Promise). The whole batch is committed as a **single undo snapshot**, and it applies to the canvas the user has open — see [Which level the user is looking at](#which-level-the-user-is-looking-at). |
| `onGraphChanged` | `(callback: () => void) => () => void` | Subscribe to graph changes — including the user stepping into or out of a block. The callback takes no arguments; call `getGraph()` from it. Returns an unsubscribe function. |
| `getView` | `() => GraphView` | **apiVersion 4.** Read-only: which level of the graph the user is looking at. |

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

#### Which level the user is looking at

Requires `api.apiVersion >= 4`.

A CodefyUI graph nests. A **block** (subgraph) has a canvas of its own, and the user can step inside one — the bar above the canvas then reads `Main > Encoder`. There is only ever one canvas: stepping inside *swaps* the block's insides onto it, which is exactly why every editing tool works the same inside a block as outside it.

For a plugin that has one consequence, and it decides where your edits land:

- **`getGraph()` always answers with the whole graph.** The editor folds whatever is open back in before serializing, the same way Save and Run do, so you read the same bytes the user would get by saving the file.
- **`applyOperations()` writes to the canvas the user has open.** Inside a block, `add_node` adds a node *to that block*, and `clear_graph` empties *the block* rather than the graph. Node ids you read from `getGraph()` do not exist there, so ops naming them come back `ok: false` with an error.

So a plugin that reads, reasons, then writes can be right about the graph and still write somewhere the user is not looking. `getView()` is how you tell the two situations apart first:

```ts
interface GraphViewLevel {
  subgraphId: string;  // the block definition's id, as getGraph() refers to it
  name: string;        // the block's name, as the breadcrumb bar shows it
}

interface GraphView {
  depth: number;           // 0 at the top level, 1 inside a block, 2 inside a block inside a block
  path: GraphViewLevel[];  // the open blocks, outermost first; empty at the top level
  atTopLevel: boolean;     // depth === 0, for the check you usually want
}
```

```js
const view = api.graph.getView();
if (!view.atTopLevel) {
  const inside = view.path[view.path.length - 1].name;
  api.ui.toast(`Step out of "${inside}" first — an edit now would land inside that block.`, "warning");
  return;
}
api.graph.applyOperations(ops);
```

Refusing is not the only honest answer — waiting, or scoping the edit to something that makes sense inside a block, are both fine. The point is that the choice is now yours to make instead of a coin flip.

The view is **read-only**, and read live: each call is a fresh answer, and there is deliberately no way to navigate somebody's editor from a plugin. `onGraphChanged` fires when the user steps into or out of a block (the canvas changed, after all), so a panel that displays where it would write can re-read `getView()` from that callback.

Where a write lands is the editor's long-standing behaviour, now written down rather than changed. A later revision may let an op name its target level explicitly; it will do that by adding something, not by quietly redirecting the writes that installed plugins already make.

### `api.nodes` — custom node renderers

Requires `api.apiVersion >= 2`.

| Method | Signature | Description |
|--------|-----------|-------------|
| `registerRenderer` | `(nodeType, renderer) => () => void` | Draw a plugin node type's card body with your own UI. Returns an unregister function. |

`nodeType` is the node's **namespaced** type as it appears in `getNodeDefinitions()`. Note the namespace is the snake_case form of your plugin id — plugin `my-plugin` exposes node type `my_plugin:MyNode`. The renderer is imperative, so the host stays framework-agnostic:

```ts
interface NodeRenderContext {
  node: { id: string; type: string; params: Record<string, unknown> };
}
interface PluginNodeRenderer {
  mount(container: HTMLElement, ctx: NodeRenderContext): void;
  update?(container: HTMLElement, ctx: NodeRenderContext): void; // on param change
  unmount?(container: HTMLElement): void;
}
```

The editor still renders the standard node card (title, ports, param list) and hands your renderer a `<div>` for the **body** — slotted between the ports and the params. A node type with no registered renderer renders exactly like a default node.

```js
api.nodes.registerRenderer('my_plugin:MyNode', {
  mount(el, ctx) { el.textContent = `value: ${ctx.node.params.value}`; },
  update(el, ctx) { el.textContent = `value: ${ctx.node.params.value}`; },
});
```

The [plugin template](https://github.com/CodefyUI/CodefyUI-Plugin-Official)'s SDK wraps this with `createRoot`, so you can write the body as a React component.

### `api.events` — live run events

Requires `api.apiVersion >= 3`.

| Method | Signature | Description |
|--------|-----------|-------------|
| `onExecution` | `(cb: (event: ExecutionEvent) => void) => () => void` | Subscribe to the run event stream. Returns an unsubscribe function. |

```ts
type ExecutionEvent =
  | { type: "run_started";  run_id: string; cursor: number; seq: number }
  | { type: "node_status";  run_id: string; cursor: number; seq: number;
      node_id: string; status: string; error?: string }
  | { type: "metric";       run_id: string; cursor: number; seq: number;
      points: readonly RunMetricPoint[] }
  | { type: "run_finished"; run_id: string; cursor: number; seq: number;
      status: "succeeded" | "failed" | "cancelled" | "interrupted";
      error?: string };
```

```js
const off = api.events.onExecution((event) => {
  if (event.type === "metric") {
    for (const p of event.points) record(p.name, p.step, p.value);
  }
  if (event.type === "run_finished") summarise(event.run_id, event.status);
});
```

Things worth knowing before you build on it:

- **A `metric` event carries the whole batch it was recorded as,** in `points`. Those are [`RunMetricPoint`](#apiruns--run-history-read-only)s — the *same* type `api.runs.metrics()` returns — so one fold function can serve the live tail and the REST back-fill. In particular `value` is `null` for a non-finite number on both sides: a diverged loss is a **gap in the curve, not a zero**, and it is delivered rather than skipped. (The one difference: `ts` is populated on points from `api.runs.metrics()` and absent on live ones.)
- **Events are frozen.** One event object is shared by every subscriber, so it and its `points` are `Object.freeze`d — you cannot mutate what another plugin receives, and you should copy before transforming.
- **Events are batched onto animation frames.** A run pushing hundreds of metrics a second reaches you as one burst of calls per frame, not one call per message — the same batching the editor uses for its own node badges. A backgrounded or occluded editor window is never painted, so nothing is delivered until it comes back; what piles up in the meantime is bounded by what it costs to keep — an event plus each metric point it carries, capped at about twenty thousand of those — and past the cap the oldest metrics and node statuses are dropped (`run_started` and `run_finished` never are). The unit matters because one `metric` event carries a whole batch: a run writing fat batches and one writing single points get the same memory budget, not the same number of events. Re-read metrics from `api.runs.metrics()` if you need every point.
- **It is a tail, not a transcript.** The editor replays a run's whole recorded log whenever it attaches to one — on a page reload with a run in flight, or when the user picks a run to watch in the Runs panel. Those replayed entries pass through the same stream, and the host filters out every one already delivered, so a re-attach can never hand you a duplicate to double-count. The exceptions are spelled out under [when the editor attaches to a run you have not seen](#when-the-editor-attaches-to-a-run-you-have-not-seen).
- **Unsubscribe** when you are done; it takes effect immediately, including part-way through a batch. The editor also unsubscribes you automatically on unload or hot-reload.
- **The stream covers the runs the editor is attached to** — the ones started from a canvas tab, plus any run the user chose to watch from the Runs panel. A run submitted by `cdui run` that nobody is watching is visible through `api.runs`, not here.
- **If your callback throws,** the editor logs it and moves on. No other subscriber is affected, but you lose that event.

#### `cursor` and `seq`

Every event carries two numbers, and mixing them up is the easiest way to build a dashboard that lies to its user.

**`cursor` is where the event sits in the run's durable log** — the same cursor `GET /api/runs/{id}/events` pages by and `api.runs.get(id).last_cursor` reports. Use it to line an event up against the REST side.

It is strictly increasing within a run, but it is **not dense, and a jump in it means nothing**. The log also holds entries this stream does not publish, and each one consumes a cursor:

- `artifact` — every checkpoint a run saves writes one;
- `run_warning`;
- a refused submit, and a cancel that had nothing to cancel;
- a metric entry the server collapsed because its payload was too large.

A perfectly healthy training run that checkpoints every epoch therefore produces a cursor gap every epoch. Do not treat that as data loss.

**`seq` is the stream's own counter, and it is the one that signals loss.** It counts the events delivered for a run, densely: the next event you receive for a run has `seq` exactly one higher than the last one you received — unless the host dropped events under the buffering limit described above, which is the only thing that can put a hole in it.

```js
// Per run: remember the last seq you saw, and react to a hole.
const lastSeq = new Map();
api.events.onExecution((event) => {
  const previous = lastSeq.get(event.run_id);
  lastSeq.set(event.run_id, event.seq);
  if (previous !== undefined && event.seq > previous + 1) {
    // The only cause is buffer overflow. Recover from REST.
    void api.runs.metrics(event.run_id).then(backfill);
  }
  apply(event);
});
```

The first `seq` you see for a run is your baseline, not necessarily `1`: it counts from when the *editor* started streaming that run, which may be before your plugin subscribed.

#### When the editor attaches to a run you have not seen

The de-duplication above is bookkeeping the editor keeps **per run, once, for the whole page** — not per plugin. Two consequences:

- When the editor attaches to a run **nothing has streamed yet** — the user clicking a run in the Runs panel — the server replays that run's recorded log from the start, and you receive it, in cursor order, before the live tail begins. Every entry still arrives exactly once, but the first events you see for that run describe the past.
- When the editor attaches to a run **something has already streamed**, the replay is filtered out for everyone. If your plugin subscribed later than another one, you inherit that filtering, so you may see *nothing at all* from the replay of a run you personally never saw. Do not rely on a replay to populate yourself; use `api.runs` for that, which is what it is for.

If you need to know whether an event describes the past, `api.runs.get(run_id)` reports `last_cursor`. Note it is not a one-liner for a run that is still going: you are reading a moving target *after* the replay has already started, so the honest pattern is to buffer events until the promise resolves and only then classify them.

One bound worth knowing rather than discovering: the editor remembers the last **1024** runs it has streamed. Attaching to more than 1024 distinct runs in a single page session and then returning to one from the beginning of that session will replay it to you a second time. No ordinary session comes close, and the number is here so the limit is a documented condition rather than a surprise.

### `api.runs` — run history (read-only)

Requires `api.apiVersion >= 3`.

| Method | Signature | Description |
|--------|-----------|-------------|
| `list` | `(opts?) => Promise<RunListPage>` | Newest-first page of runs. `opts` is `{ status?, limit?, offset? }`. |
| `get` | `(id: string) => Promise<RunInfo \| null>` | One run, or `null` when the server has never heard of it. |
| `metrics` | `(id: string, name?: string) => Promise<RunMetrics>` | Recorded scalar series, ordered `(name, step)`. |

```js
const page = await api.runs.list({ status: ["running"], limit: 1 });
const active = page.runs[0];
if (active) {
  const recorded = await api.runs.metrics(active.id);
  for (const point of recorded.metrics) {
    if (point.value !== null) record(point.name, point.step, point.value);
  }
}
```

```ts
interface RunListPage { runs: RunSummary[]; total: number; limit: number; offset: number }
interface RunInfo extends RunSummary { last_cursor: number }
interface RunMetrics { run_id: string; names: string[]; metrics: RunMetricPoint[] }
interface RunMetricPoint {
  node_id: string | null; name: string; step: number;
  value: number | null;   // null is a diverged (non-finite) value — a gap, not a zero
  ts?: string;            // ISO-8601 UTC; set here, absent on live metric events
}
```

`RunMetricPoint` is the same type the live `metric` event carries in `points`, so a dashboard can fold both with one function. `ts` is the only field that differs between the two sources: `api.runs.metrics()` records when each point was written, the live stream carries only what a chart plots against `step`. A fold that ignores `ts` works on both unchanged.

`RunSummary` mirrors a row of the run history: `id`, `name`, `status`, `error`, `options`, `queue_key`, `created_at`, `started_at`, `finished_at`, `git_commit`, `git_dirty`, `plugin_pins`, `queue_position`, `final_metrics` and `active`. The full shapes are in the vendored SDK types, and the endpoints behind them are documented in the [API Reference](/advanced/api-reference).

The facade exists so the common case needs no hand-rolled fetching: the editor performs the requests through its own API client, with whatever authentication they need already attached. You never construct a URL, and the token is never passed to your code or returned by anything on `api.runs` — that is a convenience, not a sandbox (see [Trust model](#trust-model)).

It is deliberately **read-only in this version**. There is no `submit` and no `cancel`: starting or stopping work on someone's machine should happen behind a UI they opened, not behind a plugin call. If you need that, drive it from a button the user pressed, through `api.http.fetch`.

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

## A live run-metrics panel

The example below is the apiVersion 3 surface doing the thing it was added for: a dock tab that lists a run's metrics as they arrive. It pairs the two halves — `events.onExecution` for the live tail, `runs` for everything that happened before the panel opened — and subscribes *before* back-filling so nothing falls through the gap between them.

```js
// frontend/index.js
export default function activate(api) {
  if (api.apiVersion < 3) return;

  const series = new Map();   // name -> { last, points }

  // One fold for both halves: `event.points` and `runs.metrics().metrics`
  // are the same RunMetricPoint[].
  const fold = (points) => {
    for (const p of points) {
      const previous = series.get(p.name);
      series.set(p.name, {
        // null is a diverged value — a gap, so keep the last finite one.
        last: p.value ?? previous?.last ?? null,
        points: (previous?.points ?? 0) + 1,
      });
    }
  };

  const render = () => {
    if (!el.isConnected) return;   // the tab is not open; nothing to paint
    el.textContent = [...series.entries()]
      .map(([name, s]) =>
        `${name}  last=${s.last === null ? "--" : s.last.toFixed(4)}  n=${s.points}`)
      .join("\n");
  };

  const el = api.ui.addPanel({
    id: "run-metrics", title: "Run Metrics", icon: "~",
    onShow: render,   // paint on the way in, so the tab is never blank
  });

  // 1. the live tail
  api.events.onExecution((event) => {
    if (event.type === "run_started") series.clear();
    if (event.type === "metric") fold(event.points);
    render();
  });

  // 2. the back-fill, for a run that started before this panel existed
  api.runs.list({ status: ["running"], limit: 1 }).then(async (page) => {
    const active = page.runs[0];
    if (!active) return;
    const recorded = await api.runs.metrics(active.id);
    // Same fold, filtered to the series the live tail has not covered.
    fold(recorded.metrics.filter((p) => !series.has(p.name)));
    render();
  });
}
```

The [plugin scaffold](/advanced/plugins) ships the same example as a React component at `ui/src/examples/run-metrics-panel.tsx`, using the SDK's `mountPanel`, `useExecutionEvents` and `useRuns` bindings.

## See also

- [Plugins](/advanced/plugins) — installing packs, the manifest format, and the `cdui plugin` CLI.
- [Graph Copilot](/advanced/graph-copilot) — the first production consumer of the frontend extension API.
- [API Reference](/advanced/api-reference) — backend REST endpoints, including `/api/llm/chat`.
