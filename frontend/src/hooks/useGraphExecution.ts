import { useCallback, useEffect, useRef } from 'react';
import { useTabStore } from '../store/tabStore';
import { useToastStore } from '../store/toastStore';
import { validateGraph } from '../api/rest';
import { findEntryPoints } from '../utils/findEntryPoints';
import { useI18n } from '../i18n';

export function useGraphExecution() {
  const activeTabId = useTabStore((s) => s.activeTabId);
  const getActiveTab = useTabStore((s) => s.getActiveTab);
  const getSerializedGraph = useTabStore((s) => s.getSerializedGraph);
  const clearExecutionStatus = useTabStore((s) => s.clearExecutionStatus);
  const setTabStatus = useTabStore((s) => s.setTabStatus);
  const setTabNodeExecutionStatus = useTabStore((s) => s.setTabNodeExecutionStatus);
  const setTabNodeProgress = useTabStore((s) => s.setTabNodeProgress);
  const setTabOutputSummary = useTabStore((s) => s.setTabOutputSummary);
  const clearOutputSummaries = useTabStore((s) => s.clearOutputSummaries);
  const addTabLog = useTabStore((s) => s.addTabLog);
  const clearLogs = useTabStore((s) => s.clearLogs);
  const setLastRunId = useTabStore((s) => s.setLastRunId);

  // Track which tabs have had WS listeners attached
  const attachedTabs = useRef(new Set<string>());

  // Attach WS listeners for the active tab (idempotent per tab)
  useEffect(() => {
    const tab = getActiveTab();
    if (attachedTabs.current.has(tab.id)) return;
    attachedTabs.current.add(tab.id);

    const tabId = tab.id;
    const ws = tab.ws;

    ws.on('node_status', (data: any) => {
      // Handle progress updates (mid-execution, e.g. training epochs)
      if (data.status === 'progress' && data.progress) {
        const p = data.progress;
        setTabNodeProgress(tabId, data.node_id, p);
        if (p.event === 'epoch' || p.event === 'config') {
          addTabLog(tabId, {
            nodeId: data.node_id,
            message: `__PROGRESS__:${JSON.stringify(p)}`,
            type: 'info',
          });
        }
        return;
      }

      setTabNodeExecutionStatus(tabId, data.node_id, data.status, data.error);

      // Don't log running/cached status to reduce noise
      if (data.status !== 'running' && data.status !== 'cached') {
        const currentTab = useTabStore.getState().tabs.find(
          (t) => t.id === useTabStore.getState().activeTabId,
        );
        const nodeLabel =
          currentTab?.nodes.find((n) => n.id === data.node_id)?.data?.label ??
          String(data.node_id).slice(0, 8);

        addTabLog(tabId, {
          nodeId: data.node_id,
          message: `Node ${nodeLabel} ${data.status}${data.error ? ': ' + data.error : ''}`,
          type: data.status === 'error' ? 'error' : data.status === 'completed' ? 'success' : 'info',
        });
      }

      // If the node produced log output (Print node), show it
      if (data.log) {
        addTabLog(tabId, {
          nodeId: data.node_id,
          message: data.log,
          type: 'info',
        });
      }
      // If the node produced a base64 image, add it as a separate log entry
      if (data.image) {
        addTabLog(tabId, {
          nodeId: data.node_id,
          message: `__IMAGE__:${data.image}`,
          type: 'info',
        });
      }
      // Store output summaries for edge inspection
      if (data.output_summary) {
        setTabOutputSummary(tabId, data.node_id, data.output_summary);
      }
    });

    ws.on('execution_complete', () => {
      setTabStatus(tabId, 'completed');
      addTabLog(tabId, { message: 'Execution completed successfully', type: 'success' });
    });

    ws.on('execution_error', (data: any) => {
      setTabStatus(tabId, 'error');
      addTabLog(tabId, { message: `Execution error: ${data.error}`, type: 'error' });
    });

    ws.on('execution_start', (data: any) => {
      setTabStatus(tabId, 'running');
      if (data && typeof data.run_id === 'string') {
        setLastRunId(tabId, data.run_id);
      }
      addTabLog(tabId, { message: 'Execution started', type: 'info' });
    });

    ws.on('execution_stopped', () => {
      setTabStatus(tabId, 'idle');
      addTabLog(tabId, { message: 'Execution cancelled', type: 'info' });
    });
  }, [activeTabId, getActiveTab, setTabNodeExecutionStatus, setTabNodeProgress, setTabOutputSummary, setTabStatus, setLastRunId, addTabLog]);

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
      ...(changedNodes.length > 0 ? { changed_nodes: changedNodes } : {}),
    });
  }, [getActiveTab, getSerializedGraph, clearLogs, clearExecutionStatus, clearOutputSummaries, setTabStatus, addTabLog]);

  const stop = useCallback(() => {
    const tab = getActiveTab();
    tab.ws.send({ action: 'stop' });
  }, [getActiveTab]);

  return { execute, stop };
}
