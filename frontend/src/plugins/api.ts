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
import type { NodeDefinition } from '../types';
import { applyGraphOps, type ApplyOutcome, type GraphOp, type OpResult } from './ops';

export interface ApplyResult {
  results: OpResult[];
  refs: Record<string, string>;
  node_count: number;
  edge_count: number;
}

export type SerializedGraph = ReturnType<
  ReturnType<typeof useTabStore.getState>['getSerializedGraph']
>;

export interface CodefyUIPluginAPI {
  apiVersion: 1;
  pluginId: string;
  ui: {
    addFloatingWidget(opts: { id: string }): HTMLElement;
    toast(message: string, type?: ToastType): void;
  };
  graph: {
    getGraph(): SerializedGraph;
    getNodeDefinitions(): NodeDefinition[];
    applyOperations(ops: GraphOp[]): ApplyResult;
    onGraphChanged(cb: () => void): () => void;
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
    apiVersion: 1,
    pluginId,
    ui: {
      addFloatingWidget: ({ id }) => getWidgetContainer(id),
      toast: (message, type = 'info') =>
        useToastStore.getState().addToast(message, type),
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
