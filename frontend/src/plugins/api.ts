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

export interface CodefyUIPluginAPI {
  apiVersion: 3;
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
    getGraph(): SerializedGraph;
    getNodeDefinitions(): NodeDefinition[];
    applyOperations(ops: GraphOp[]): ApplyResult;
    onGraphChanged(cb: () => void): () => void;
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
    const changed =
      state.activeTabId !== prevTabId
      || tab?.nodes !== prevTab?.nodes
      || tab?.edges !== prevTab?.edges;
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
    apiVersion: 3,
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
