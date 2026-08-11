/**
 * CodefyUIPluginAPI — the object handed to every plugin frontend entry.
 *
 * This is a public, versioned surface: changing or removing anything here
 * breaks installed plugins. Add, don't mutate; bump apiVersion on breaking
 * changes.
 */
import { useTabStore } from '../store/tabStore';
import { useNodeDefStore } from '../store/nodeDefStore';
import { useToastStore } from '../store/toastStore';
import type { ToastType } from '../store/toastStore';
import { apiFetch } from '../api/_auth';
import {
  getRun, getRunMetrics, listRuns,
  type RunInfo, type RunListPage, type RunMetrics, type RunStatus,
} from '../api/rest';
import type { NodeDefinition } from '../types';
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

export interface RunListOptions {
  status?: readonly RunStatus[];
  limit?: number;
  offset?: number;
}

/** One opened block on the path from the graph to the canvas the user sees. */
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
  const tab = useTabStore.getState().getActiveTab() as
    | ReturnType<ReturnType<typeof useTabStore.getState>['getActiveTab']>
    | undefined;
  const path = subgraphViewPath(tab?.subgraphStack, tab?.subgraphs);
  return { depth: path.length, path, atTopLevel: path.length === 0 };
}

export interface CodefyUIPluginAPI {
  apiVersion: 4;
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
    /** Applies to the canvas the user has open, which is not always the top
     * level. See `getView` before writing. */
    applyOperations(ops: GraphOp[]): ApplyResult;
    onGraphChanged(cb: () => void): () => void;
    /** Read-only: which level of the graph the user is looking at (#200 item 7). */
    getView(): GraphView;
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
 */
export function commitGraphOperations(ops: GraphOp[]): ApplyResult {
  const store = useTabStore.getState();
  const tab = store.getActiveTab();
  const definitions = useNodeDefStore.getState().definitions;
  const outcome: ApplyOutcome = applyGraphOps(
    { nodes: tab.nodes, edges: tab.edges },
    definitions,
    ops,
  );
  if (outcome.mutated) {
    store.pushUndoSnapshot();
    store.setNodes(outcome.nodes);
    store.setEdges(outcome.edges);
    for (const id of outcome.dirtyIds) {
      useTabStore.getState().markDirty(id);
    }
  }
  return {
    results: outcome.results,
    refs: outcome.refs,
    node_count: outcome.nodes.length,
    edge_count: outcome.edges.length,
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

export function buildPluginAPI(
  pluginId: string,
  getWidgetContainer: (id: string) => HTMLElement,
  trackCleanup?: (fn: () => void) => void,
): CodefyUIPluginAPI {
  const ns = (key: string) => `plugin:${pluginId}:${key}`;
  return {
    // Bumped for `graph.getView` (#200 item 7). The number is the only way a
    // plugin can tell a host that has the new member from one that does not:
    // on a 1.5-era editor `api.graph.getView` is simply `undefined`, and the
    // documented feature check is `api.apiVersion >= 4`.
    apiVersion: 4,
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
      applyOperations: (ops) => commitGraphOperations(ops),
      getView: () => currentGraphView(),
      onGraphChanged: (cb) => {
        // Track the unsubscribe so the host can tear it down on a dev
        // hot-reload — otherwise re-activation would stack subscriptions.
        const unsubscribe = subscribeGraphChanged(cb);
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
