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
}

export interface NodeDefinition {
  node_name: string;
  category: string;
  description: string;
  inputs: PortDefinition[];
  outputs: PortDefinition[];
  params: ParamDefinition[];
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
  | { op: 'auto_layout' };

export interface OpResult {
  index: number;
  ok: boolean;
  error?: string;
  node_id?: string;
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
     */
    addToolbarButton(opts: PluginToolbarButtonOptions): () => void;
    removeToolbarButton(id: string): void;
  };
  graph: {
    getGraph(): SerializedGraph;
    getNodeDefinitions(): NodeDefinition[];
    /** Synchronous — returns the result directly, committed as one undo step. */
    applyOperations(ops: GraphOp[]): ApplyResult;
    onGraphChanged(cb: () => void): () => void;
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
