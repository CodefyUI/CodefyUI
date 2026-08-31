/**
 * CodefyUI plugin frontend API — the CANONICAL, self-contained type contract.
 *
 * This is the single source of truth for the public plugin surface. It has NO
 * imports, so it drops verbatim into any plugin's `ui/src/sdk/types.ts`:
 *
 *   - `scripts/sync_plugin_sdk.py` copies it into the `cdui plugin new` scaffold
 *     payload (`scripts/templates/plugin/ui/src/sdk/types.ts`); a backend test
 *     (`tests/test_plugin_dx.py`) fails if that copy drifts.
 *   - The clone-and-own template `CodefyUI-Plugin-Official/ui/src/sdk/types.ts`
 *     is refreshed from this file too.
 *
 * `contract.assert.ts` (next to this file) statically asserts that the host's
 * real implementation types (`./api`, `./ops`, `../types`, `./nodeRenderers`)
 * stay structurally compatible with the declarations here, so `tsc -b` fails the
 * moment the host drifts from the published contract. A future hardening step
 * could have the host import these types directly to make drift impossible
 * rather than merely detected.
 */

export type ToastType = 'success' | 'error' | 'info' | 'warning';

export interface PortDefinition {
  name: string;
  data_type: string;
  description: string;
  optional: boolean;
}

export interface ParamDefinition {
  name: string;
  param_type:
    | 'int' | 'float' | 'string' | 'bool'
    | 'select' | 'model_file' | 'image_file' | 'data_file' | 'tensor_grid' | 'secret'
    | 'code';
  default: unknown;
  description: string;
  options: string[];
  min_value: number | null;
  max_value: number | null;
  /** Conditional visibility. A value may be an array, read as "any of". */
  visible_when?: Record<string, unknown> | null;
  /** Collected behind the collapsed "Advanced" section. Absent = basic. */
  advanced?: boolean;
  /** Package Center: maps a SELECT option to the optional pack it needs,
   *  `"<pack_id>"` or `"<pack_id>:<item_id>"`. Absent = every option works. */
  option_packs?: Record<string, string> | null;
}

export interface NodeDefinition {
  node_name: string;
  category: string;
  description: string;
  inputs: PortDefinition[];
  outputs: PortDefinition[];
  params: ParamDefinition[];
  /** Package Center: the optional pack this node needs before it can run at
   *  all. Absent / null = covered by the base install. */
  requires_pack?: string | null;
}

/** A batch operation accepted by `api.graph.applyOperations`. Field names are exact. */
export type GraphOp =
  | { op: 'add_node'; node_type: string; ref?: string;
      params?: Record<string, unknown>; position?: { x: number; y: number } }
  | { op: 'connect'; source: string; source_handle: string;
      target: string; target_handle: string }
  | { op: 'set_params'; node_id: string; params: Record<string, unknown> }
  | { op: 'remove_node'; node_id: string }
  | { op: 'remove_edge'; source: string; target: string;
      source_handle?: string; target_handle?: string }
  | { op: 'clear_graph' }
  | { op: 'auto_layout' }
  /* ── requires apiVersion >= 5 ─────────────────────────────────────────── */
  /** Put one node at an exact position. Notes bound to it move with it. */
  | { op: 'move_node'; node_id: string; position: { x: number; y: number } }
  /**
   * Create or replace a segment overlay -- the orange bubble the canvas draws
   * around every node on a data path from head to tail. Omit `segment_id` to
   * create one; pass an existing id to move it. The result carries the id
   * either way. Head and tail must be joined by data edges.
   */
  | { op: 'set_segment'; segment_id?: string; head_node_id: string; tail_node_id: string }
  | { op: 'remove_segment'; segment_id: string }
  /**
   * Add a text note to the canvas — the sticky the editor already draws.
   * `bind_to` attaches it to a node so it follows when that node moves.
   * `text` is 1..4000 characters; `color` is `#rrggbb`.
   */
  | { op: 'add_note'; ref?: string; text: string;
      position?: { x: number; y: number }; color?: string; bind_to?: string }
  /** Rewrite an existing note's text and/or colour. Text notes only. */
  | { op: 'update_note'; node_id: string; text?: string; color?: string }
  /**
   * Name one node — `data.label`, 1..120 characters on a single line. The
   * label sits beside `params`, never inside it, so naming a node is not a
   * parameter change to anything reading the graph.
   */
  | { op: 'set_node_meta'; node_id: string; label: string };

export interface OpResult {
  index: number;
  ok: boolean;
  error?: string;
  node_id?: string;
  /** Set by `set_segment` — apiVersion 5. */
  segment_id?: string;
}

export interface ApplyResult {
  results: OpResult[];
  refs: Record<string, string>;
  node_count: number;
  edge_count: number;
}

export interface SerializedGraphNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: { params: Record<string, unknown>; [key: string]: unknown };
}

export interface SerializedGraphEdge {
  id: string;
  source: string;
  target: string;
  sourceHandle: string;
  targetHandle: string;
}

export interface SerializedGraph {
  nodes: SerializedGraphNode[];
  edges: SerializedGraphEdge[];
  presets?: unknown[];
  segmentGroups?: unknown[];
  name?: string;
  description?: string;
  [key: string]: unknown;
}

/* ── view context (read-only) — requires apiVersion >= 4 ─────────────────── */

/** One opened block on the path from the graph to the canvas the user sees. */
export interface GraphViewLevel {
  /** The block definition's id — stable, and what `getGraph()` refers to it by. */
  subgraphId: string;
  /** The block's name, exactly as the editor's breadcrumb bar shows it. */
  name: string;
}

/**
 * Where the user is looking right now, from `api.graph.getView()`.
 *
 * A CodefyUI graph nests: a block (subgraph) has its own canvas, and the user
 * can step inside one, then inside another. The editor shows this as the
 * "Main > Encoder > Attention" bar above the canvas; `getView()` is the same
 * information, for a plugin.
 *
 * Read-only, and read live — a fresh answer each call. There is no way to
 * navigate the user somewhere through the plugin API, deliberately: moving
 * somebody's editor under them is not something a plugin should be able to do
 * quietly.
 */
export interface GraphView {
  /** 0 at the top level, 1 inside a block, 2 inside a block inside a block. */
  depth: number;
  /** The opened blocks, outermost first. Empty at the top level. */
  path: GraphViewLevel[];
  /** `depth === 0`, named so the check a plugin usually wants reads plainly. */
  atTopLevel: boolean;
}

export interface NodeRenderContext {
  node: {
    id: string;
    type: string;
    params: Record<string, unknown>;
    definition?: unknown;
  };
}

/**
 * Imperative renderer for a node's card body. The SDK's `defineNodeRenderer`
 * adapts a React component to this shape; the host calls mount once, update on
 * param changes, and unmount when the node is removed.
 */
export interface PluginNodeRenderer {
  mount(container: HTMLElement, ctx: NodeRenderContext): void;
  update?(container: HTMLElement, ctx: NodeRenderContext): void;
  unmount?(container: HTMLElement): void;
}

/* ── panels, toolbar buttons — requires apiVersion >= 3 ─────────────────── */

/** Where a panel lives: a tab in the bottom dock, or a right-hand section. */
export type PluginPanelDock = 'bottom' | 'right';

export interface PluginPanelOptions {
  /** Unique within your plugin. The host namespaces it with your plugin id. */
  id: string;
  /** Tab label (bottom dock) or section heading (right dock). */
  title: string;
  /** Optional short glyph shown before the title. */
  icon?: string;
  /** Defaults to `'bottom'`. */
  dock?: PluginPanelDock;
  /**
   * Called after the panel's element is attached to the document, and after
   * it is detached. The element itself survives either way — these exist so
   * you can stop doing work while nobody can see it.
   */
  onShow?: () => void;
  onHide?: () => void;
}

export interface PluginToolbarButtonOptions {
  /** Unique within your plugin. */
  id: string;
  /** Short glyph — the toolbar has room for a glyph, not a sentence. */
  icon: string;
  /** Hover and accessible text. An icon on its own is not a label. */
  tooltip: string;
  onClick: () => void;
}

/* ── execution events — requires apiVersion >= 3 ────────────────────────── */

/**
 * One recorded scalar point.
 *
 * Declared here because BOTH halves of the run surface use it: the live
 * `metric` event below, and `runs.metrics()` further down. One type for one
 * concept is the whole reason a dashboard can share a fold between them.
 */
export interface RunMetricPoint {
  node_id: string | null;
  name: string;
  step: number;
  /** `null` for a non-finite value (a diverged loss) — a gap, not a zero. */
  value: number | null;
  /**
   * When the point was recorded, ISO-8601 UTC.
   *
   * Present on points from `runs.metrics()`, absent on points from a live
   * `metric` event — the live path carries only what a chart needs to plot
   * against `step`. Optional for exactly that reason; a fold that ignores it
   * works on both halves unchanged.
   */
  ts?: string;
}

/** Terminal state of a run. */
export type ExecutionFinishStatus =
  | 'succeeded' | 'failed' | 'cancelled' | 'interrupted';

/**
 * One run event, as `api.events.onExecution` delivers it.
 *
 * Every event carries TWO numbers, and they answer different questions.
 *
 * `cursor` is the event's position in the run's durable log — the same cursor
 * `GET /api/runs/{id}/events` pages by and `runs.get(id).last_cursor` reports.
 * Use it to line an event up against the REST side. It is strictly increasing
 * within a run but **not dense**: the log also holds entries this stream does
 * not publish (artifacts, run warnings, refused submits, entries the server
 * collapsed because they were too large), and each of those consumes a cursor.
 * A cursor jump is therefore ordinary — a run that saves a checkpoint produces
 * one every time — and says nothing about whether you missed anything.
 *
 * `seq` is this stream's own counter, and it is the one to check for loss. It
 * counts the events delivered to plugin subscribers for a run, densely: the
 * next event you receive for a run always has `seq` exactly one higher than
 * the last, unless the host dropped events under the buffering limit. So
 * `seq` gapless means you have everything; `seq` jumping by N means N events
 * were dropped and `runs.metrics()` is how to recover them.
 *
 * (The first `seq` you see for a run is your baseline, not necessarily 1: it
 * counts from when the editor started streaming that run, which may predate
 * your subscription.)
 *
 * A `metric` entry carries the whole batch it was recorded as, rather than
 * being split into one event per point. That keeps the log entry atomic —
 * `cursor` would otherwise say "you have this entry" when only part of it had
 * been delivered — and makes `points` the SAME type `runs.metrics()` returns,
 * so a dashboard can fold the live tail and the REST back-fill with one
 * function. `value` is `null` for a non-finite number exactly as it is there.
 *
 * Events and their `points` are frozen: they are shared between every
 * subscriber, so one plugin cannot mutate what another receives.
 */
export type ExecutionEvent =
  | { type: 'run_started'; run_id: string; cursor: number; seq: number }
  | {
      type: 'node_status'; run_id: string; cursor: number; seq: number;
      node_id: string; status: string; error?: string;
    }
  | {
      type: 'metric'; run_id: string; cursor: number; seq: number;
      points: readonly RunMetricPoint[];
    }
  | {
      type: 'run_finished'; run_id: string; cursor: number; seq: number;
      status: ExecutionFinishStatus; error?: string;
    };

/* ── runs (read-only) — requires apiVersion >= 3 ────────────────────────── */

export type RunStatus =
  | 'queued' | 'running' | 'succeeded'
  | 'failed' | 'cancelled' | 'interrupted';

/** One row of the run history. Timestamps are ISO-8601 UTC with a `Z`. */
export interface RunSummary {
  id: string;
  name: string | null;
  status: RunStatus;
  error: string | null;
  /** Submit options verbatim — `device`, `seed`, `record_outputs`, ... */
  options: Record<string, unknown>;
  queue_key: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  git_commit: string | null;
  git_dirty: boolean | null;
  plugin_pins: Record<string, unknown> | null;
  /** 1-based place in this run's own device queue; `null` is a real answer. */
  queue_position: number | null;
  /** Last recorded value of every series, e.g. `{ train_loss: 0.31 }`. */
  final_metrics: Record<string, number>;
  /** Whether the server is currently driving this run. */
  active: boolean;
}

export interface RunInfo extends RunSummary {
  /** Highest event cursor issued so far — where a follower should resume. */
  last_cursor: number;
}

export interface RunListPage {
  runs: RunSummary[];
  /** Unpaged count for the active filter, so a table can size itself. */
  total: number;
  limit: number;
  offset: number;
}

export interface RunListOptions {
  status?: readonly RunStatus[];
  limit?: number;
  offset?: number;
}

export interface RunMetrics {
  run_id: string;
  /** Every series name in the run, so a legend needs no scan of the points. */
  names: string[];
  metrics: RunMetricPoint[];
}

/* ── workspace tabs — requires apiVersion >= 5 ───────────────────────────── */

/**
 * Who opened a tab, and why. Opaque to the editor: whatever you put here
 * comes back verbatim on `tabs()` and `snapshot()`, and persists with the tab
 * when you open it with `persist: true`.
 */
export interface WorkspaceSource {
  /** A short kind of your own choosing, e.g. `'agent-variant'`. */
  kind: string;
  pluginId: string;
  jobId?: string;
  variantId?: string;
  [key: string]: unknown;
}

/**
 * The loosest shape `openGraphs` accepts for a graph document.
 *
 * `WorkspaceOpenEntry.graph` is declared as `SerializedGraph` below, because
 * the graph you hand over usually came from `getGraph()`. This type is what
 * the editor really reads: two required lists and a bag it walks defensively,
 * so a document you composed yourself is accepted on the same terms as a
 * file. Unknown top-level keys are ignored.
 */
export interface WorkspaceGraphInput {
  nodes: unknown[];
  edges: unknown[];
  presets?: unknown[];
  segmentGroups?: unknown[];
  subgraphs?: unknown[];
  name?: string;
  description?: string;
  format_version?: unknown;
  [key: string]: unknown;
}

export interface WorkspaceOpenEntry {
  /** Shown on the tab. Stored whole; the tab bar ellipsises what will not fit. */
  title: string;
  /**
   * The document to open, normalised the way a file load normalises one:
   * `subgraphs`, `segmentGroups` and `presets` are honoured, and a
   * `format_version` newer than the editor opens the tab read-only.
   */
  graph: SerializedGraph;
  /** Default false. A read-only tab refuses writes on BOTH write paths. */
  readOnly?: boolean;
  source?: WorkspaceSource;
  /** Default false: the tab is transient and is gone after a reload. */
  persist?: boolean;
}

/**
 * One entry's outcome. Results are POSITIONAL — `result[i]` describes
 * `entries[i]` — so one bad candidate never sinks the others.
 */
export type WorkspaceOpenResult =
  | { tabId: string; revision: number }
  | { error: string; code: 'invalid_graph' | 'too_many_tabs' | 'too_large' };

export interface WorkspaceTabInfo {
  tabId: string;
  title: string;
  /**
   * The tab's compare-and-swap token. It starts at 1 and advances by one
   * whenever the tab's document changes — from your writes, the user's edits,
   * an undo, or a redo. Hand it back as `expectedRevision` to write only if
   * nothing has moved since you read.
   */
  revision: number;
  readOnly: boolean;
  /** True for a tab that will not survive a reload (opened without `persist`). */
  transient: boolean;
  source: WorkspaceSource | null;
  active: boolean;
}

export type WorkspaceSnapshot =
  | (WorkspaceTabInfo & { graph: SerializedGraph })
  | { error: 'unknown_tab' };

export interface WorkspaceApplyRequest {
  /** Defaults to the active tab. */
  tabId?: string;
  /** Omitted: no compare. Present and stale: nothing is written. */
  expectedRevision?: number;
  operations: GraphOp[];
  /** Default false. True: any failing op means nothing is committed. */
  atomic?: boolean;
}

/**
 * Why a write was refused. Each is returned, never thrown, and each leaves
 * the tab exactly as it was.
 *
 * `editing_subgraph` means the user has stepped inside a block, so the canvas
 * arrays are that block's contents rather than the document `snapshot()`
 * describes. Retry once they step back out — `graph.getView().atTopLevel`
 * says when.
 */
export type WorkspaceConflict =
  | 'revision_mismatch' | 'read_only' | 'unknown_tab' | 'editing_subgraph';

export interface WorkspaceApplyResult extends ApplyResult {
  tabId: string;
  /**
   * The tab's revision AFTER this call, and on `revision_mismatch` the
   * CURRENT one — so you can re-arm without a second read that races the
   * same way. Unchanged by a conflict or a failed `atomic` preflight.
   */
  revision: number;
  committed: boolean;
  conflict?: WorkspaceConflict;
}

/**
 * One workspace change, as `workspace.onChanged` delivers it.
 *
 * Delivered synchronously, in the order `tabs` (added), `graph`,
 * `active-tab`, `tabs` (removed), so a burst can be processed in one pass. A
 * `graph` event carries `origin` when the change came from a plugin's
 * `workspace.applyOperations`, which is how you ignore your own writes.
 *
 * If your callback throws, the editor logs it and unsubscribes you: a plugin
 * cannot be allowed to break the editor's own state.
 */
export type WorkspaceEvent =
  | { type: 'graph'; tabId: string; revision: number; origin?: { pluginId: string } }
  | { type: 'tabs'; tabId: string; revision: number; removed: boolean }
  | { type: 'active-tab'; tabId: string; revision: number };

/** The object the editor hands every plugin frontend at activation. */
export interface CodefyUIPluginAPI {
  apiVersion: number;
  pluginId: string;
  ui: {
    /** Create (or reuse) a container `<div>` in the editor's widget stack. */
    addFloatingWidget(opts: { id: string }): HTMLElement;
    toast(message: string, type?: ToastType): void;
    /**
     * Register a dock panel and get its container element — requires
     * apiVersion >= 3.
     *
     * The host owns the tab chrome and the placement; you own what goes
     * inside the element. The element's identity is stable for the life of
     * the panel, so mount into it once: the host attaches and detaches it as
     * its tab becomes active, and never empties or replaces it. Calling
     * `addPanel` again with the same id returns the same element and updates
     * the title/icon/dock.
     */
    addPanel(opts: PluginPanelOptions): HTMLElement;
    /** Remove a panel. Its element is detached from the editor. */
    removePanel(id: string): void;
    /**
     * Add a toolbar button — requires apiVersion >= 3. Returns a remove
     * function. Placement and overflow are the host's: buttons share one
     * group and collapse into a menu when the window is too narrow.
     *
     * Re-adding an id replaces the button, and the returned remove function
     * is scoped to the registration it came from: a disposer whose button has
     * since been replaced does nothing, rather than removing the replacement.
     * Use `removeToolbarButton(id)` to remove whatever currently holds an id.
     */
    addToolbarButton(opts: PluginToolbarButtonOptions): () => void;
    removeToolbarButton(id: string): void;
  };
  /**
   * The graph.
   *
   * Read and write are not aimed at the same place while the user is inside a
   * block, and that difference is documented on each member below. `getView()`
   * — apiVersion 4 — is how you find out where the user is before you write.
   */
  graph: {
    /**
     * The WHOLE graph, always: nodes and edges at the top level, plus the block
     * definitions under `subgraphs`.
     *
     * Never affected by where the user is standing — the editor flattens its
     * open sub-canvases into the answer first, exactly as Save and Run do. So a
     * plugin reading the graph sees the same bytes the user would get by saving
     * the file, whether or not a block is open on screen.
     */
    getGraph(): SerializedGraph;
    getNodeDefinitions(): NodeDefinition[];
    /**
     * Synchronous — returns the result directly, committed as one undo step.
     *
     * **A batch applies to the canvas the user has open, which is not always
     * the top level.** Entering a block replaces the canvas with that block's
     * insides, so ops applied then land inside the block: `add_node` adds a
     * node to the block, and `clear_graph` empties the block rather than the
     * graph. Node ids from `getGraph()` — which always answers with the top
     * level — will not be found there, and those ops fail with an error in
     * their `results` entry.
     *
     * This is the editor's long-standing behaviour and it is not changing
     * silently; it is written down here so you can code against it. If your
     * plugin composes a read with a write, call `getView()` between the two and
     * handle `atTopLevel === false` — refuse with a toast, wait for the user to
     * step out, or scope your edit to what makes sense inside a block. Writing
     * anyway is not corruption, but it lands somewhere the user is not looking.
     */
    applyOperations(ops: GraphOp[]): ApplyResult;
    /**
     * Called after the graph changes — including when the user steps into or
     * out of a block, since that swaps the canvas. Re-read `getView()` from the
     * callback if you track which level the user is on.
     */
    onGraphChanged(cb: () => void): () => void;
    /**
     * Where the user is looking — requires apiVersion >= 4.
     *
     * Read-only. It exists so a plugin can tell "the user is inside a block"
     * from "the user is on the graph" before it writes; see `applyOperations`.
     * On an older editor this member is `undefined`, so feature-check with
     * `api.apiVersion >= 4` (or `typeof api.graph.getView === 'function'`).
     */
    getView(): GraphView;
  };
  /**
   * Tabs, snapshots and compare-and-swap writes — requires apiVersion >= 5.
   *
   * This is the half of the API that does not assume the user is looking at
   * what you are working on. You can open a candidate graph in its own
   * labelled tab without moving anybody, read a tab that is not on screen,
   * write to a tab only if it has not changed since you read it, and be told
   * when any of that happens.
   *
   * On an older editor the whole member is `undefined` rather than a stub
   * that throws, so `typeof api.workspace?.openGraphs === 'function'` is an
   * honest feature check (`api.apiVersion >= 5` works too).
   */
  workspace: {
    /**
     * Open one tab per entry. Nothing is activated when `activate` is
     * `'none'`; `'first'` (the default) and `'last'` activate one of the tabs
     * that actually opened.
     *
     * Limits: 8 MiB of serialized JSON per graph, 32 tabs in the editor.
     * A tab opened here is an ordinary tab afterwards — the user can rename
     * it, close it, or Save As into a real graph file.
     */
    openGraphs(
      entries: WorkspaceOpenEntry[],
      options?: { activate?: 'first' | 'last' | 'none' },
    ): WorkspaceOpenResult[];
    /** Every tab, in tab-bar order. */
    tabs(): WorkspaceTabInfo[];
    /**
     * A tab's info plus its WHOLE graph, flattened the way `getGraph()`
     * flattens the active one. Defaults to the active tab. An unknown id
     * answers `{ error: 'unknown_tab' }` rather than throwing.
     */
    snapshot(tabId?: string): WorkspaceSnapshot;
    /**
     * Apply a batch to a named tab, as one undo step, under an optional
     * compare-and-swap.
     *
     * Checked in a fixed order, each refusal leaving the tab untouched:
     * unknown tab, read-only tab, the user is inside a block, stale
     * `expectedRevision`. Then the batch runs; with `atomic: true` a single
     * failing op means nothing is committed, and `results` still comes back
     * full-length so you can see which op was wrong.
     */
    applyOperations(request: WorkspaceApplyRequest): WorkspaceApplyResult;
    /** Subscribe to workspace changes. Returns an unsubscribe function. */
    onChanged(cb: (event: WorkspaceEvent) => void): () => void;
  };
  /** Custom node renderers — requires apiVersion >= 2. */
  nodes: {
    registerRenderer(nodeType: string, renderer: PluginNodeRenderer): () => void;
  };
  /** Live run events — requires apiVersion >= 3. */
  events: {
    /**
     * Subscribe to the run event stream. Returns an unsubscribe function.
     *
     * Events are batched onto animation frames, so a burst arrives as a burst
     * of calls in one frame rather than one call per socket message, and a
     * backgrounded editor delivers nothing until it is painted again. If your
     * callback throws, the host contains it — no other subscriber is
     * affected, but you will lose that event.
     */
    onExecution(cb: (event: ExecutionEvent) => void): () => void;
  };
  /**
   * Read-only view of run history — requires apiVersion >= 3.
   *
   * The host performs the requests, with editor authentication attached; the
   * session token is never handed to plugin code. Submitting and cancelling
   * runs are deliberately absent: a plugin should not be able to start work
   * on the user's machine behind a UI the user did not open.
   */
  runs: {
    list(opts?: RunListOptions): Promise<RunListPage>;
    /** `null` when the server has never heard of the run. */
    get(id: string): Promise<RunInfo | null>;
    metrics(id: string, name?: string): Promise<RunMetrics>;
  };
  http: {
    /** Browser `fetch`, with the CodefyUI session token attached. */
    fetch(url: string, init?: RequestInit): Promise<Response>;
  };
  storage: {
    get(key: string): string | null;
    set(key: string, value: string): void;
    remove(key: string): void;
  };
}

/** The default-export contract: the editor calls this once with the API. */
export type ActivateFn = (api: CodefyUIPluginAPI) => void;
