/**
 * CodefyUIPluginAPI — the object handed to every plugin frontend entry.
 *
 * This is a public, versioned surface: changing or removing anything here
 * breaks installed plugins. Add, don't mutate; bump apiVersion on breaking
 * changes.
 */
import {
  useTabStore, lastCommitOrigin,
  type GraphDocument, type TabState,
} from '../store/tabStore';
import { useNodeDefStore } from '../store/nodeDefStore';
import { useToastStore } from '../store/toastStore';
import type { ToastType } from '../store/toastStore';
import { apiFetch } from '../api/_auth';
import {
  getRun, getRunMetrics, listRuns,
  type RunInfo, type RunListPage, type RunMetrics, type RunStatus,
} from '../api/rest';
import type { NodeDefinition, WorkspaceSource } from '../types';
import { resolveExample } from '../utils/openExample';
import { subgraphViewPath } from '../utils/subgraph';
import { applyGraphOps, type ApplyOutcome, type GraphOp, type OpResult } from './ops';
import { registerNodeRenderer, type PluginNodeRenderer } from './nodeRenderers';
import {
  registerPluginPanel, removePluginPanel,
  type PluginPanelOptions,
} from './panels';
import {
  registerPluginToolbarButton, removePluginToolbarButton,
  type PluginToolbarButtonOptions,
} from './toolbarButtons';
import { subscribeExecutionEvents, type ExecutionEvent } from './executionEvents';

export interface ApplyResult {
  results: OpResult[];
  refs: Record<string, string>;
  node_count: number;
  edge_count: number;
}

export type SerializedGraph = ReturnType<
  ReturnType<typeof useTabStore.getState>['getSerializedGraph']
>;

export type { WorkspaceSource };

/**
 * The document a plugin hands `workspace.openGraphs`.
 *
 * Deliberately looser than `SerializedGraph`, which is the store's exact
 * RETURN type with every list non-optional: a plugin composes this object
 * itself, and the honest shape of "whatever a graph file holds" is two
 * required lists and a bag. Everything else is read defensively by the same
 * document reader a file load uses.
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
  graph: WorkspaceGraphInput;
  /** Default false. A read-only tab refuses plugin writes on both paths. */
  readOnly?: boolean;
  source?: WorkspaceSource;
  /** Default false: the tab is transient and is gone after a reload. */
  persist?: boolean;
}

export type WorkspaceOpenResult =
  | { tabId: string; revision: number }
  | { error: string; code: 'invalid_graph' | 'too_many_tabs' | 'too_large' };

export interface WorkspaceTabInfo {
  tabId: string;
  title: string;
  revision: number;
  readOnly: boolean;
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

export type WorkspaceConflict =
  | 'revision_mismatch' | 'read_only' | 'unknown_tab' | 'editing_subgraph';

export interface WorkspaceApplyResult extends ApplyResult {
  tabId: string;
  /** The tab's revision AFTER this call; unchanged on a conflict or a preflight failure. */
  revision: number;
  committed: boolean;
  conflict?: WorkspaceConflict;
}

export type WorkspaceEvent =
  | { type: 'graph'; tabId: string; revision: number; origin?: { pluginId: string } }
  | { type: 'tabs'; tabId: string; revision: number; removed: boolean }
  | { type: 'active-tab'; tabId: string; revision: number };

/** Serialized JSON a single `openGraphs` entry may carry. */
const MAX_WORKSPACE_GRAPH_BYTES = 8 * 1024 * 1024;
/** How many tabs the editor will hold before `openGraphs` starts refusing. */
const MAX_WORKSPACE_TABS = 32;
/**
 * Refusal for a legacy batch that would commit a top-level segment from
 * inside a block (#341 section 4.6).
 *
 * Written once and in the same voice as `'tab is read-only'` -- what is
 * wrong, not what to do about it. The contract does not freeze either
 * literal; both are read by people, not matched on.
 */
const SEGMENT_INSIDE_BLOCK_ERROR =
  'set_segment and remove_segment cannot apply while a block is open';

export interface RunListOptions {
  status?: readonly RunStatus[];
  limit?: number;
  offset?: number;
}

/**
 * One opened block on the path from the graph to the canvas the user sees.
 *
 * Structurally the `SubgraphPathEntry` the shared derivation returns; declared
 * separately because the published contract (`contract.ts`) has to declare it
 * with no imports, and `contract.assert.ts` proves the two stay identical.
 */
export interface GraphViewLevel {
  subgraphId: string;
  /** The block's name, exactly as the breadcrumb bar shows it. */
  name: string;
}

/**
 * Where the user is looking, as `api.graph.getView()` answers (core#200 item 7).
 *
 * READ-ONLY, and read live: nothing here can be set through the plugin API, and
 * the object is a fresh snapshot of the moment it was asked for.
 */
export interface GraphView {
  /** 0 at the top level, 1 inside a block, 2 inside a block inside a block. */
  depth: number;
  /** The opened blocks, outermost first. Empty at the top level. */
  path: GraphViewLevel[];
  /** `depth === 0`, named so the common check reads as a sentence. */
  atTopLevel: boolean;
}

/**
 * The current view context, derived from the active tab's editing stack.
 *
 * Optional-chained through the tab because a plugin may call this at any time,
 * including from an activation that runs before the editor has restored its
 * tabs -- and "no tab" is honestly reported as the top level rather than as a
 * thrown error inside third-party code.
 */
export function currentGraphView(): GraphView {
  // `getTab` rather than `getActiveTab`: the latter is typed as always
  // returning a tab (it asserts the lookup with `!`), so asking it would mean
  // casting the answer back to something that can be missing.
  const { activeTabId, getTab } = useTabStore.getState();
  const tab = getTab(activeTabId);
  const path = subgraphViewPath(tab?.subgraphStack, tab?.subgraphs);
  return { depth: path.length, path, atTopLevel: path.length === 0 };
}

export interface CodefyUIPluginAPI {
  apiVersion: 5;
  pluginId: string;
  ui: {
    addFloatingWidget(opts: { id: string }): HTMLElement;
    toast(message: string, type?: ToastType): void;
    /** Register a dock panel; returns its stable container element. */
    addPanel(opts: PluginPanelOptions): HTMLElement;
    removePanel(id: string): void;
    /** Register a toolbar button; returns a remove fn. */
    addToolbarButton(opts: PluginToolbarButtonOptions): () => void;
    removeToolbarButton(id: string): void;
  };
  graph: {
    /** Always the WHOLE graph, from the top level down. See `getView`. */
    getGraph(): SerializedGraph;
    getNodeDefinitions(): NodeDefinition[];
    /**
     * Applies to the canvas the user has open, which is not always the top
     * level. See `getView` -- and `commitGraphOperations` below for why.
     */
    applyOperations(ops: GraphOp[]): ApplyResult;
    onGraphChanged(cb: () => void): () => void;
    /** Read-only: which level of the graph the user is looking at (#200 item 7). */
    getView(): GraphView;
  };
  /**
   * Tabs, snapshots and compare-and-swap writes -- requires apiVersion >= 5.
   *
   * Absent on an older editor rather than stubbed, so
   * `typeof api.workspace?.openGraphs === 'function'` is an honest check.
   */
  workspace: {
    openGraphs(
      entries: WorkspaceOpenEntry[],
      options?: { activate?: 'first' | 'last' | 'none' },
    ): WorkspaceOpenResult[];
    tabs(): WorkspaceTabInfo[];
    snapshot(tabId?: string): WorkspaceSnapshot;
    applyOperations(request: WorkspaceApplyRequest): WorkspaceApplyResult;
    onChanged(cb: (event: WorkspaceEvent) => void): () => void;
  };
  nodes: {
    /** Register a custom renderer for a node type's card body. Returns an unregister fn. */
    registerRenderer(nodeType: string, renderer: PluginNodeRenderer): () => void;
  };
  events: {
    /** Subscribe to run lifecycle events. Returns an unsubscribe fn. */
    onExecution(cb: (event: ExecutionEvent) => void): () => void;
  };
  runs: {
    list(opts?: RunListOptions): Promise<RunListPage>;
    get(id: string): Promise<RunInfo | null>;
    metrics(id: string, name?: string): Promise<RunMetrics>;
  };
  http: {
    fetch(url: string, init?: RequestInit): Promise<Response>;
  };
  storage: {
    get(key: string): string | null;
    set(key: string, value: string): void;
    remove(key: string): void;
  };
}

/** The two ops that write `segmentGroups`, which is top-level state. */
function isSegmentOp(op: GraphOp): boolean {
  return op.op === 'set_segment' || op.op === 'remove_segment';
}

function tabInfoOf(tab: TabState, activeTabId: string): WorkspaceTabInfo {
  return {
    tabId: tab.id,
    title: tab.name,
    revision: tab.revision,
    readOnly: tab.readOnly,
    transient: tab.transient,
    source: tab.source,
    active: tab.id === activeTabId,
  };
}

/**
 * Apply a plugin's batch to a NAMED tab, under a compare-and-swap (#341 4.4).
 *
 * The order of the refusals is the contract, and each returns without a side
 * effect: an unknown tab, then a read-only tab, then a tab whose canvas is
 * showing a block's insides, then a stale revision. Only after all four does
 * the pure reducer run, and only after `atomic`'s preflight does anything get
 * written.
 *
 * Conflicts are RETURNED, never thrown: the consumer treats a throw out of
 * this call as "the canvas may hold a partial update", which is the one thing
 * that is never true here.
 *
 * `refuseInsideBlock` is the ONE way the two callers differ, and it is not a
 * knob a plugin can reach: the workspace path refuses outright, and the legacy
 * path keeps writing the open canvas because that is what it has always done
 * and moving an installed plugin's writes out from under it is not a change to
 * make silently (see `commitGraphOperations`). Its one carve-out is the two
 * segment ops, which are not canvas writes at all -- see the branch below.
 */
function commitToTab(
  pluginId: string,
  request: WorkspaceApplyRequest,
  options: { refuseInsideBlock: boolean },
): WorkspaceApplyResult {
  const store = useTabStore.getState();
  const tabId = request.tabId ?? store.activeTabId;
  const tab = store.getTab(tabId);
  const empty = { results: [] as OpResult[], refs: {} as Record<string, string> };

  if (!tab) {
    return { ...empty, tabId, revision: 0, committed: false,
             conflict: 'unknown_tab', node_count: 0, edge_count: 0 };
  }
  // The counts on a refusal describe the tab as it stands, so a plugin that
  // logs them is not told the graph is empty when it is not.
  const counts = { node_count: tab.nodes.length, edge_count: tab.edges.length };
  if (tab.readOnly) {
    return { ...empty, ...counts, tabId, revision: tab.revision,
             committed: false, conflict: 'read_only' };
  }
  if (tab.subgraphStack.length > 0) {
    // While a block is open, `tab.nodes` / `tab.edges` are the BLOCK's
    // contents, and `snapshot()` answers with the flushed top level -- so a
    // workspace write here would land somewhere the plugin never read.
    // Refused rather than redirected: the plugin retries once the user steps
    // back out, which is a wait it can see, unlike an edit that silently went
    // into a definition.
    if (options.refuseInsideBlock) {
      return { ...empty, ...counts, tabId, revision: tab.revision,
               committed: false, conflict: 'editing_subgraph' };
    }
    // The legacy path goes on writing the open canvas -- except for the two
    // segment ops, which do not write the canvas at all. `enterSubgraph`
    // CAPTURES `segmentGroups` instead of swapping them, so a segment
    // committed from in here is a TOP-LEVEL overlay naming inner node ids: it
    // survives the exit, reaches the saved file, and draws nothing -- which
    // also means the user cannot remove it, the control being on the bubble.
    // `remove_segment` is the mirror, deleting a real top-level overlay the
    // user cannot even see from where they are standing.
    //
    // Refused whole-batch and shaped like the read-only refusal (#341 section
    // 4.6). Not a behaviour change for anybody: both ops are new in v5, so no
    // installed plugin sends them, and a batch without one is untouched.
    if (request.operations.some(isSegmentOp)) {
      return {
        ...counts, tabId, revision: tab.revision, committed: false,
        refs: {},
        results: request.operations.map((_, index) => ({
          index, ok: false, error: SEGMENT_INSIDE_BLOCK_ERROR,
        })),
      };
    }
  }
  if (
    request.expectedRevision !== undefined
    && request.expectedRevision !== tab.revision
  ) {
    return { ...empty, ...counts, tabId, revision: tab.revision,
             committed: false, conflict: 'revision_mismatch' };
  }

  const definitions = useNodeDefStore.getState().definitions;
  const outcome: ApplyOutcome = applyGraphOps(
    { nodes: tab.nodes, edges: tab.edges, segmentGroups: tab.segmentGroups },
    definitions,
    request.operations,
  );
  const applied = {
    tabId,
    results: outcome.results,
    refs: outcome.refs,
    node_count: outcome.nodes.length,
    edge_count: outcome.edges.length,
  };

  // The preflight `atomic` buys: the reducer already works on copies, so
  // "discard the outcome" is the whole implementation. The results are still
  // returned in full so the model can see WHICH op it got wrong.
  if (request.atomic && outcome.results.some((r) => !r.ok)) {
    return { ...applied, revision: tab.revision, committed: false };
  }
  if (!outcome.mutated) {
    return { ...applied, revision: tab.revision, committed: false };
  }

  // One snapshot, then one write: that is what makes a batch one Ctrl+Z.
  store.pushUndoSnapshotFor(tabId);
  store.commitDocument(tabId, {
    nodes: outcome.nodes,
    edges: outcome.edges,
    segmentGroups: outcome.segmentGroups,
    dirtyIds: outcome.dirtyIds,
    origin: { pluginId },
  });
  const after = useTabStore.getState().getTab(tabId)!;
  return { ...applied, revision: after.revision, committed: true };
}

/** The tab-addressed write path, as `api.workspace.applyOperations` exposes it. */
export function commitWorkspaceOperations(
  pluginId: string,
  request: WorkspaceApplyRequest,
): WorkspaceApplyResult {
  return commitToTab(pluginId, request, { refuseInsideBlock: true });
}

/**
 * Apply a plugin's batch to the canvas the user has open.
 *
 * `tab.nodes` / `tab.edges` are the canvas in FRONT OF THE USER, which while a
 * block is open are the block's insides rather than the graph -- so a batch
 * applied then lands inside the block, and `clear_graph` empties the block
 * instead of the graph (core#200 item 7).
 *
 * That is deliberately left as it stands (maintainer decision, 2026-08-12).
 * Changing where a write lands would silently redirect every installed plugin's
 * edits, and both answers are defensible; what was missing was any way for a
 * plugin to KNOW. So the gradual step is `api.graph.getView()` -- read-only
 * view context, above -- letting a plugin refuse, warn, or ask instead of
 * writing blind. A future revision can add an explicit write target on top of
 * it without having moved anybody's writes in the meantime.
 *
 * Note the asymmetry this leaves, and why `getView` matters: `getGraph()`
 * flushes and answers with the whole graph, so a plugin that reads, reasons and
 * writes can compute node ids that exist at the top level and apply them to a
 * canvas where they do not.
 *
 * Since v5 this is a thin shim over the tab-addressed path above: same target,
 * same undo semantics, same `ApplyResult` shape, and none of the conflict
 * channel -- a v1 plugin reads four keys and would not know what to do with a
 * fifth. It keeps `refuseInsideBlock: false` precisely to preserve the
 * behaviour this comment describes; `workspace.applyOperations` is where a
 * plugin gets the refusal instead. The ONE behaviour change v5 makes to an
 * installed plugin is the other one -- a read-only tab now refuses per op
 * instead of being written through (#341 section 4.6).
 *
 * The segment carve-out above is not a third: `set_segment` and
 * `remove_segment` are new in v5, so no installed plugin can send them, and
 * refusing them inside a block keeps a top-level overlay naming inner node
 * ids out of the saved file.
 */
export function commitGraphOperations(ops: GraphOp[], pluginId = ''): ApplyResult {
  const result = commitToTab(pluginId, { operations: ops }, { refuseInsideBlock: false });
  if (result.conflict === 'read_only') {
    return {
      results: ops.map((_, index) => ({
        index, ok: false, error: 'tab is read-only',
      })),
      refs: {},
      node_count: result.node_count,
      edge_count: result.edge_count,
    };
  }
  return {
    results: result.results,
    refs: result.refs,
    node_count: result.node_count,
    edge_count: result.edge_count,
  };
}

function subscribeGraphChanged(cb: () => void): () => void {
  let prevTabId = useTabStore.getState().activeTabId;
  let prevTab = useTabStore.getState().tabs.find((t) => t.id === prevTabId);
  return useTabStore.subscribe((state) => {
    const tab = state.tabs.find((t) => t.id === state.activeTabId);
    // `subgraphs` belongs here alongside nodes and edges (core#200 item 3):
    // it is part of what `graph.getGraph()` answers with, so renaming a block
    // or editing its insides and stepping back out changes the bytes a plugin
    // would read while telling it nothing happened. Reference comparison, the
    // same as the other two -- every store action that touches the definition
    // list replaces it.
    const changed =
      state.activeTabId !== prevTabId
      || tab?.nodes !== prevTab?.nodes
      || tab?.edges !== prevTab?.edges
      || tab?.subgraphs !== prevTab?.subgraphs;
    prevTabId = state.activeTabId;
    prevTab = tab;
    if (changed) cb();
  });
}

/**
 * Read a plugin's graph the way a file load reads a file, and hand back the
 * document `loadGraphDocumentInto` installs. Throws the reader's own error.
 */
function workspaceDocument(graph: WorkspaceGraphInput): GraphDocument {
  const resolved = resolveExample(graph);
  return {
    nodes: resolved.nodes,
    edges: resolved.edges,
    // A plugin's graph is bound to no file: the first Save has to ask where
    // it should go, exactly as an example does.
    boundFile: null,
    subgraphs: resolved.subgraphs,
    segmentGroups: resolved.segmentGroups,
    // The TAB's label is the entry's `title`, already set by `createTab`. The
    // graph's own `name` is deliberately not allowed to overwrite it.
    name: null,
    description: resolved.description,
    formatVersion: resolved.formatVersion,
  };
}

function openWorkspaceGraphs(
  entries: WorkspaceOpenEntry[],
  options?: { activate?: 'first' | 'last' | 'none' },
): WorkspaceOpenResult[] {
  const results: WorkspaceOpenResult[] = [];
  const opened: string[] = [];

  for (const entry of entries) {
    const title = typeof entry?.title === 'string' ? entry.title.trim() : '';
    if (!title) {
      results.push({ error: 'openGraphs: title must be a non-empty string', code: 'invalid_graph' });
      continue;
    }
    let serialized: string;
    try {
      serialized = JSON.stringify(entry.graph);
    } catch {
      results.push({ error: 'openGraphs: graph is not JSON-serializable', code: 'invalid_graph' });
      continue;
    }
    const bytes = new TextEncoder().encode(serialized).byteLength;
    if (bytes > MAX_WORKSPACE_GRAPH_BYTES) {
      results.push({
        error: `openGraphs: the graph is ${Math.round(bytes / 1024 / 1024)} MiB; the limit is 8 MiB`,
        code: 'too_large',
      });
      continue;
    }
    let doc: GraphDocument;
    try {
      doc = workspaceDocument(entry.graph);
    } catch (error) {
      results.push({
        error: `openGraphs: ${error instanceof Error ? error.message : String(error)}`,
        code: 'invalid_graph',
      });
      continue;
    }
    // Checked LAST, and against the live count, so two entries in one call
    // cannot both slip past a limit only one of them fits under.
    if (useTabStore.getState().tabs.length >= MAX_WORKSPACE_TABS) {
      results.push({
        error: `openGraphs: the editor already has ${MAX_WORKSPACE_TABS} tabs open`,
        code: 'too_many_tabs',
      });
      continue;
    }

    const tabId = useTabStore.getState().createTab({ title, activate: false });
    const tooNew = useTabStore.getState().loadGraphDocumentInto(tabId, doc);
    useTabStore.getState().setTabMeta(tabId, {
      // Either reason is enough: the plugin asked, or the document is from a
      // build this one does not understand.
      readOnly: entry.readOnly === true || tooNew,
      source: entry.source ?? null,
      transient: entry.persist !== true,
    });
    results.push({
      tabId,
      revision: useTabStore.getState().getTab(tabId)!.revision,
    });
    opened.push(tabId);
  }

  const activate = options?.activate ?? 'first';
  if (activate !== 'none' && opened.length > 0) {
    useTabStore.getState().setActiveTab(
      activate === 'last' ? opened[opened.length - 1] : opened[0],
    );
  }
  return results;
}

/**
 * One store subscription, diffed element-wise, fanned out as ordered events
 * (#341 section 4.5).
 *
 * The order -- added, changed, activated, removed -- is what lets a consumer
 * process a batch in one pass: a tab it is told about has already been
 * announced, and a tab it is told is gone was still there for everything
 * before it.
 */
function subscribeWorkspaceChanged(
  cb: (event: WorkspaceEvent) => void,
): () => void {
  let prevTabs = useTabStore.getState().tabs;
  let prevActive = useTabStore.getState().activeTabId;
  let alive = true;
  const unsubscribe = useTabStore.subscribe((state) => {
    if (!alive) return;
    const { tabs, activeTabId } = state;
    const before = new Map(prevTabs.map((t) => [t.id, t] as const));
    const after = new Map(tabs.map((t) => [t.id, t] as const));
    const events: WorkspaceEvent[] = [];

    for (const t of tabs) {
      if (!before.has(t.id)) {
        events.push({ type: 'tabs', tabId: t.id, revision: t.revision, removed: false });
      }
    }
    for (const t of tabs) {
      const was = before.get(t.id);
      if (was && was.revision !== t.revision) {
        // Read inside the notification, which is the only window the store
        // keeps it open for.
        const origin = lastCommitOrigin();
        events.push({
          type: 'graph', tabId: t.id, revision: t.revision,
          ...(origin ? { origin } : {}),
        });
      }
    }
    if (activeTabId !== prevActive) {
      events.push({
        type: 'active-tab', tabId: activeTabId,
        revision: after.get(activeTabId)?.revision ?? 0,
      });
    }
    for (const t of prevTabs) {
      if (!after.has(t.id)) {
        events.push({ type: 'tabs', tabId: t.id, revision: t.revision, removed: true });
      }
    }

    // Advanced BEFORE the fan-out: a callback is allowed to write to the
    // store, and a re-entrant notification must diff against what it sees.
    prevTabs = tabs;
    prevActive = activeTabId;

    for (const event of events) {
      try {
        cb(event);
      } catch (error) {
        // A plugin must never be able to break the store, and a callback that
        // throws once will throw on the next event too -- so it is dropped
        // rather than left to fire into a burst of drag events.
        console.error(
          '[CodefyUI] a plugin workspace.onChanged callback threw; unsubscribing it',
          error,
        );
        alive = false;
        unsubscribe();
        return;
      }
    }
  });
  return () => {
    alive = false;
    unsubscribe();
  };
}

export function buildPluginAPI(
  pluginId: string,
  getWidgetContainer: (id: string) => HTMLElement,
  trackCleanup?: (fn: () => void) => void,
): CodefyUIPluginAPI {
  const ns = (key: string) => `plugin:${pluginId}:${key}`;
  return {
    // Bumped for `workspace` and the six agent canvas ops (#341, #342). The
    // number is the only way a plugin can tell a host that has them from one
    // that does not: on a 2.0-to-2.4 editor `api.workspace` is simply
    // `undefined` -- never a stub whose methods throw -- and the documented
    // feature checks are `api.apiVersion >= 5` and
    // `typeof api.workspace?.openGraphs === 'function'`.
    apiVersion: 5,
    pluginId,
    ui: {
      addFloatingWidget: ({ id }) => getWidgetContainer(id),
      toast: (message, type = 'info') =>
        useToastStore.getState().addToast(message, type),
      addPanel: (opts) => {
        const element = registerPluginPanel(pluginId, opts);
        // Tracked so `teardownPlugins()` (dev hot-reload, and the unload path)
        // removes the panel even if the plugin never calls removePanel.
        trackCleanup?.(() => removePluginPanel(pluginId, opts.id));
        return element;
      },
      removePanel: (id) => removePluginPanel(pluginId, id),
      addToolbarButton: (opts) => {
        const remove = registerPluginToolbarButton(pluginId, opts);
        trackCleanup?.(remove);
        return remove;
      },
      removeToolbarButton: (id) => removePluginToolbarButton(pluginId, id),
    },
    graph: {
      getGraph: () => useTabStore.getState().getSerializedGraph(),
      getNodeDefinitions: () => useNodeDefStore.getState().definitions,
      applyOperations: (ops) => commitGraphOperations(ops, pluginId),
      onGraphChanged: (cb) => {
        // Track the unsubscribe so the host can tear it down on a dev
        // hot-reload — otherwise re-activation would stack subscriptions.
        const unsubscribe = subscribeGraphChanged(cb);
        trackCleanup?.(unsubscribe);
        return unsubscribe;
      },
      getView: () => currentGraphView(),
    },
    workspace: {
      openGraphs: (entries, options) => openWorkspaceGraphs(entries, options),
      tabs: () => {
        const { tabs, activeTabId } = useTabStore.getState();
        return tabs.map((t) => tabInfoOf(t, activeTabId));
      },
      snapshot: (tabId) => {
        const state = useTabStore.getState();
        const tab = state.getTab(tabId ?? state.activeTabId);
        if (!tab) return { error: 'unknown_tab' };
        return {
          ...tabInfoOf(tab, state.activeTabId),
          graph: state.getSerializedGraphOf(tab),
        };
      },
      applyOperations: (request) => commitWorkspaceOperations(pluginId, request),
      onChanged: (cb) => {
        // Tracked like `onGraphChanged`'s, so a dev hot-reload does not stack
        // subscriptions.
        const unsubscribe = subscribeWorkspaceChanged(cb);
        trackCleanup?.(unsubscribe);
        return unsubscribe;
      },
    },
    nodes: {
      registerRenderer: (nodeType, renderer) => {
        const unregister = registerNodeRenderer(nodeType, renderer);
        trackCleanup?.(unregister);
        return unregister;
      },
    },
    events: {
      onExecution: (cb) => {
        const unsubscribe = subscribeExecutionEvents(cb);
        trackCleanup?.(unsubscribe);
        return unsubscribe;
      },
    },
    // Read-only by design (see the contract). Every request goes through the
    // host's own API client, so the session token is attached where the host
    // attaches it and never reaches plugin code — there is nothing on this
    // facade, or on anything it returns, that a plugin could read it from.
    runs: {
      list: (opts = {}) => listRuns(opts),
      get: (id) => getRun(id),
      metrics: (id, name) => getRunMetrics(id, name),
    },
    http: {
      fetch: (url, init) => apiFetch(url, init),
    },
    storage: {
      get: (key) => window.localStorage.getItem(ns(key)),
      set: (key, value) => window.localStorage.setItem(ns(key), value),
      remove: (key) => window.localStorage.removeItem(ns(key)),
    },
  };
}
