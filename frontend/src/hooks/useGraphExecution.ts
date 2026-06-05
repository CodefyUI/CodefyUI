import { useCallback, useEffect } from 'react';
import { useTabStore, type TabState } from '../store/tabStore';
import { useToastStore } from '../store/toastStore';
import { useUIStore } from '../store/uiStore';
import { validateGraph } from '../api/rest';
import { findEntryPoints } from '../utils/findEntryPoints';
import { useI18n } from '../i18n';
import type { ExecutionWebSocket } from '../api/ws';

type WsHandlerEntry = {
  ws: ExecutionWebSocket;
  type: string;
  handler: (data: unknown) => void;
};

export function useGraphExecution() {
  const getActiveTab = useTabStore((s) => s.getActiveTab);
  const getSerializedGraph = useTabStore((s) => s.getSerializedGraph);
  const clearExecutionStatus = useTabStore((s) => s.clearExecutionStatus);
  const setTabStatus = useTabStore((s) => s.setTabStatus);
  const clearOutputSummaries = useTabStore((s) => s.clearOutputSummaries);
  const addTabLog = useTabStore((s) => s.addTabLog);
  const clearLogs = useTabStore((s) => s.clearLogs);

  // Attach per-tab WS listeners. We subscribe to tabStore directly (rather
  // than re-running on activeTabId change) so background tabs keep receiving
  // their own execution events even when not in focus. All registrations are
  // released when this hook unmounts, so react-doctor's effect-needs-cleanup
  // contract is satisfied without breaking that persistence.
  useEffect(() => {
    const attached = new Map<string, WsHandlerEntry[]>();

    const detachTab = (tabId: string) => {
      const entries = attached.get(tabId);
      // detachTab is only ever called with ids drawn from attached.keys()
      /* v8 ignore start */
      if (!entries) return;
      /* v8 ignore stop */
      for (const { ws, type, handler } of entries) ws.off(type, handler);
      attached.delete(tabId);
    };

    const attachTab = (tab: TabState) => {
      if (attached.has(tab.id)) return;
      const tabId = tab.id;
      const ws = tab.ws;

      const onNodeStatus = (raw: unknown) => {
        const data = raw as any;
        const store = useTabStore.getState();

        if (data.status === 'progress' && data.progress) {
          const p = data.progress;
          store.setTabNodeProgress(tabId, data.node_id, p);
          if (p.event === 'epoch' || p.event === 'config') {
            store.addTabLog(tabId, {
              nodeId: data.node_id,
              message: `__PROGRESS__:${JSON.stringify(p)}`,
              type: 'info',
            });
          }
          return;
        }

        store.setTabNodeExecutionStatus(tabId, data.node_id, data.status, data.error);

        // Suppress running/cached chatter — only surface terminal transitions.
        if (data.status !== 'running' && data.status !== 'cached') {
          const currentTab = store.tabs.find((t) => t.id === store.activeTabId);
          const nodeLabel =
            currentTab?.nodes.find((n) => n.id === data.node_id)?.data?.label ??
            String(data.node_id).slice(0, 8);

          store.addTabLog(tabId, {
            nodeId: data.node_id,
            message: `Node ${nodeLabel} ${data.status}${data.error ? ': ' + data.error : ''}`,
            type:
              data.status === 'error'
                ? 'error'
                : data.status === 'completed'
                  ? 'success'
                  : 'info',
          });
        }

        if (data.log) {
          store.addTabLog(tabId, { nodeId: data.node_id, message: data.log, type: 'info' });
        }
        if (data.image) {
          store.addTabLog(tabId, {
            nodeId: data.node_id,
            message: `__IMAGE__:${data.image}`,
            type: 'info',
          });
        }
        if (data.output_summary) {
          store.setTabOutputSummary(tabId, data.node_id, data.output_summary);
        }
      };

      const onExecutionComplete = () => {
        const store = useTabStore.getState();
        store.setTabStatus(tabId, 'completed');
        store.addTabLog(tabId, { message: 'Execution completed successfully', type: 'success' });
      };

      const onExecutionError = (raw: unknown) => {
        const data = raw as { error: string };
        const store = useTabStore.getState();
        store.setTabStatus(tabId, 'error');
        store.addTabLog(tabId, { message: `Execution error: ${data.error}`, type: 'error' });
      };

      const onExecutionStart = (raw: unknown) => {
        const data = raw as { run_id?: string };
        const store = useTabStore.getState();
        store.setTabStatus(tabId, 'running');
        if (typeof data.run_id === 'string') {
          store.setLastRunId(tabId, data.run_id);
        }
        store.addTabLog(tabId, { message: 'Execution started', type: 'info' });
      };

      const onExecutionStopped = () => {
        const store = useTabStore.getState();
        store.setTabStatus(tabId, 'idle');
        store.addTabLog(tabId, { message: 'Execution cancelled', type: 'info' });
      };

      const entries: WsHandlerEntry[] = [
        { ws, type: 'node_status', handler: onNodeStatus },
        { ws, type: 'execution_complete', handler: onExecutionComplete },
        { ws, type: 'execution_error', handler: onExecutionError },
        { ws, type: 'execution_start', handler: onExecutionStart },
        { ws, type: 'execution_stopped', handler: onExecutionStopped },
      ];
      for (const { type, handler } of entries) ws.on(type, handler);
      attached.set(tabId, entries);
    };

    for (const tab of useTabStore.getState().tabs) attachTab(tab);

    const unsubscribe = useTabStore.subscribe((state) => {
      const currentIds = new Set(state.tabs.map((t) => t.id));
      for (const id of Array.from(attached.keys())) {
        if (!currentIds.has(id)) detachTab(id);
      }
      for (const tab of state.tabs) attachTab(tab);
    });

    return () => {
      unsubscribe();
      for (const tabId of Array.from(attached.keys())) detachTab(tabId);
    };
  }, []);

  const execute = useCallback(async () => {
    const tab = getActiveTab();

    // Block execution when the graph has no entry points. This mirrors the
    // backend `find_entry_points` so we fail fast with a toast instead of
    // sending a graph that will be rejected server-side.
    const entryIds = findEntryPoints(tab.nodes, tab.edges);
    if (entryIds.length === 0) {
      useToastStore.getState().addToast(
        useI18n.getState().t('execution.error.noEntryPoints'),
        'error',
      );
      return;
    }

    const ws = tab.ws;

    if (!ws.connected) {
      try {
        await ws.connect();
      } catch {
        addTabLog(tab.id, { message: 'Failed to connect to execution server', type: 'error' });
        return;
      }
    }

    const graph = getSerializedGraph();
    // Filter out note nodes — they are annotations, not computational
    const execNodes = graph.nodes.filter((n: any) => n.type !== 'note');

    // Pre-execution validation
    try {
      const validation = await validateGraph(execNodes, graph.edges);
      if (!validation.valid) {
        const { addToast } = useToastStore.getState();
        validation.errors.forEach((err: string) => addToast(err, 'error'));
        return;
      }
    } catch {
      // If validation endpoint is unreachable, proceed anyway
    }

    clearLogs();
    clearExecutionStatus();
    clearOutputSummaries();
    setTabStatus(tab.id, 'running');

    // Partial re-execution: pass changed_nodes hint to backend
    const { getDirtyWithDownstream, clearDirty } = useTabStore.getState();
    const changedNodes = getDirtyWithDownstream();
    clearDirty();

    // Teaching Inspector: generate a run id so the backend can key captured
    // outputs, and pass the per-tab Record toggle state.
    const runId =
      typeof crypto !== 'undefined' && 'randomUUID' in crypto
        ? crypto.randomUUID()
        : `run-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;

    ws.send({
      action: 'execute',
      nodes: execNodes,
      edges: graph.edges,
      presets: graph.presets,
      run_id: runId,
      record_outputs: tab.recordOutputs,
      // A1: verbose step-trace mode
      verbose_mode: tab.verboseMode,
      // A2: weight persistence — backend NodeStateStore keys modules by graph_id
      graph_id: tab.graphId,
      weights_persistent: tab.weightsPersistent,
      // A3: gradient capture
      backward_mode: tab.backwardMode,
      auto_backward: tab.autoBackward,
      // Global compute device (nodes with device='auto' follow this).
      device: useUIStore.getState().globalDevice,
      ...(changedNodes.length > 0 ? { changed_nodes: changedNodes } : {}),
    });
  }, [getActiveTab, getSerializedGraph, clearLogs, clearExecutionStatus, clearOutputSummaries, setTabStatus, addTabLog]);

  const stop = useCallback(() => {
    const tab = getActiveTab();
    tab.ws.send({ action: 'stop' });
  }, [getActiveTab]);

  return { execute, stop };
}
