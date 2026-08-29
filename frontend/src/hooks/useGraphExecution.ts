import { useCallback, useEffect } from 'react';
import type { ExecutionStatus } from '../types';
import { useTabStore, type TabState } from '../store/tabStore';
import {
  queueTabNodeProgress,
  queueTabNodeStatus,
  discardTabNodeUpdates,
} from '../store/nodeUpdateQueue';
import { useToastStore } from '../store/toastStore';
import { useUIStore } from '../store/uiStore';
import { usePackStore } from '../store/packStore';
import { getRun, validateGraph } from '../api/rest';
import { findEntryPoints } from '../utils/findEntryPoints';
import { localizedPackTitle } from '../utils/packAvailability';
import { friendlyError, missingPackFromError } from '../utils/errorMessages';
import { useI18n } from '../i18n';
import { RECONNECTED_EVENT, type ExecutionWebSocket } from '../api/ws';

type WsHandlerEntry = {
  ws: ExecutionWebSocket;
  type: string;
  handler: (data: unknown) => void;
};

/**
 * Run statuses that are worth re-attaching to on page load. Everything else
 * has a terminal row and nothing left to stream.
 */
const RESUMABLE = new Set(['running', 'queued']);

/** What a finished run's status means for the tab that was watching it. */
const TERMINAL_TAB_STATUS: Record<string, ExecutionStatus> = {
  succeeded: 'completed',
  failed: 'error',
  cancelled: 'idle',
  interrupted: 'idle',
};

/**
 * (tabId, runId) pairs this page load has already tried to re-attach to.
 *
 * React StrictMode mounts every effect twice in development, and a second
 * `attach` replaces the first subscription on the server — which would
 * replay the whole log a second time and double every line in the panel.
 * Module scope, because "once per page load" is exactly its lifetime.
 *
 * This set is the ONLY idempotency guard on that path, deliberately. The
 * obvious companion — a `cancelled` flag set by the effect's cleanup —
 * makes re-attach never fire at all under StrictMode: pass 1 claims the key
 * before its first `await`, pass 1's cleanup sets the flag, pass 2 sees the
 * claimed key and returns, and pass 1 then bails on the flag. Nobody
 * attaches. Nothing here needs the flag anyway: the socket and the tab are
 * owned by the store and outlive this hook, so a send that lands after
 * unmount is still a send to a live socket about a live run.
 */
const reattached = new Set<string>();

/**
 * A run stopped because an optional pack is not installed: say so, and offer
 * the one place that can fix it.
 *
 * Worth a toast on top of the log line because the fix is not on this screen.
 * The log explains the failure where the user is already looking; the Package
 * Center is behind the toolbar's Settings menu, and a beginner who has just
 * been told "install it from the Package Center" should not then have to find
 * it. The action opens it ON the pack, so the install is one more click.
 *
 * Says it once per pack, by looking at the toasts already up rather than by
 * remembering anything. A fail-fast run re-raises the node's exception, so
 * the SAME message arrives twice — once on `node_status`, once as the run's
 * own `execution_error` — and two identical toasts stacked on each other
 * read as two problems. Missing-pack toasts are errors, so they never time
 * out and the first is still up when the second frame lands. Two DIFFERENT
 * packs still get a toast each, which is right: a continue-mode run can be
 * short two of them.
 */
function toastMissingPack(rawError: unknown, errorType?: unknown): void {
  if (typeof rawError !== 'string' || !rawError) return;
  const packId = missingPackFromError(
    rawError,
    typeof errorType === 'string' ? errorType : undefined,
  );
  if (!packId) return;

  const t = useI18n.getState().t;
  const message = t('packs.toast.missingPack', {
    pack: localizedPackTitle(t, usePackStore.getState().byId, packId),
  });
  const store = useToastStore.getState();
  if (store.toasts.some((toast) => toast.message === message)) return;
  store.addToast(message, 'error', {
    action: {
      label: t('packs.toast.openCenter'),
      onClick: () => useUIStore.getState().openPackCenter(packId),
    },
  });
}

/** Connect if needed; false when the server is unreachable. */
async function ensureConnected(ws: ExecutionWebSocket): Promise<boolean> {
  if (ws.connected) return true;
  try {
    await ws.connect();
    return true;
  } catch {
    return false;
  }
}

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

        // #117: the backend declares every renderable payload in `outputs`
        // as `{ output_kind, [output_kind]: payload }`. Unknown kinds are
        // skipped so a newer backend never breaks this handler.
        const outputs: any[] = Array.isArray(data.outputs) ? data.outputs : [];
        const ofKind = (kind: string) =>
          outputs.filter((o) => o && typeof o === 'object' && o.output_kind === kind);
        const firstOf = (kind: string) => ofKind(kind)[0];

        // DEPRECATED (remove one release after #117): a frontend built from
        // source can meet an older prebuilt backend that still sends these
        // flat fields — and that guessed `image` from the string's length.
        const legacy = outputs.length === 0 ? data : {};

        if (data.status === 'progress') {
          const p = firstOf('progress')?.progress ?? legacy.progress;
          if (p) {
            // #125: buffered, not written straight through. A training run
            // streams these faster than the screen repaints, and every
            // direct write rebuilt the whole nodes array.
            queueTabNodeProgress(tabId, data.node_id, p);
            if (p.event === 'epoch' || p.event === 'config') {
              store.addTabLog(tabId, {
                nodeId: data.node_id,
                message: '',
                kind: 'progress',
                progress: p,
                type: 'info',
              });
            }
            return;
          }
        }

        queueTabNodeStatus(tabId, data.node_id, data.status, data.error);

        // Suppress running/cached chatter — only surface terminal transitions.
        if (data.status !== 'running' && data.status !== 'cached') {
          // This event's OWN tab (captured above), not whichever tab is on
          // screen -- a background tab's run must label its log from its
          // own nodes, never the active tab's (#163).
          const eventTab = store.tabs.find((t) => t.id === tabId);
          const nodeLabel =
            eventTab?.nodes.find((n) => n.id === data.node_id)?.data?.label ??
            String(data.node_id).slice(0, 8);

          // Map the exception here, where `error_type` is still available --
          // the panels only ever see the composed line, and the type cannot be
          // recovered from `str(exc)` afterwards.
          const detail = data.error ? friendlyError(data.error, data.error_type) : '';

          // Only on the terminal `error` frame, so this runs once per failing
          // node rather than on every status this node reports.
          if (data.status === 'error') {
            toastMissingPack(data.error, data.error_type);
          }

          store.addTabLog(tabId, {
            nodeId: data.node_id,
            message: `Node ${nodeLabel} ${data.status}${detail ? ': ' + detail : ''}`,
            type:
              data.status === 'error'
                ? 'error'
                : data.status === 'completed'
                  ? 'success'
                  : 'info',
          });
        }

        // One log line per text output, in the order the backend listed them
        // — a node may emit several (and #130's chart pack will emit text
        // alongside other kinds).
        const texts = ofKind('text').map((o) => o.text);
        if (legacy.log) texts.push(legacy.log);
        for (const text of texts) {
          if (!text) continue;
          store.addTabLog(tabId, {
            nodeId: data.node_id,
            message: String(text),
            kind: 'text',
            type: 'info',
          });
        }

        const imageEntries = ofKind('image');
        // Legacy backends sent a bare base64 string with no container info.
        if (typeof legacy.image === 'string' && legacy.image) {
          imageEntries.push({ image: { format: 'png', encoding: 'base64', data: legacy.image } });
        }
        for (const entry of imageEntries) {
          const payload = entry.image;
          if (!payload?.data) continue;
          store.addTabLog(tabId, {
            nodeId: data.node_id,
            message: '',
            kind: 'image',
            image: {
              format: payload.format ?? 'png',
              encoding: payload.encoding ?? 'base64',
              data: payload.data,
              ...(entry.port ? { port: entry.port } : {}),
            },
            type: 'info',
          });
        }

        // #310: a video entry is a REFERENCE (path/url/format) to a file the
        // backend wrote under its media dir — never bytes. No legacy
        // fallback — the kind postdates the flat-field era entirely.
        for (const entry of ofKind('video')) {
          const payload = entry.video;
          if (!payload?.url || !payload?.format) continue;
          store.addTabLog(tabId, {
            nodeId: data.node_id,
            message: '',
            kind: 'video',
            video: { ...payload, ...(entry.port ? { port: entry.port } : {}) },
            type: 'info',
          });
        }

        // #130: a chart entry carries a spec the client draws itself. No
        // legacy fallback — the kind postdates the flat-field era entirely.
        for (const entry of ofKind('chart')) {
          const payload = entry.chart;
          if (!payload?.kind) continue;
          store.addTabLog(tabId, {
            nodeId: data.node_id,
            message: '',
            kind: 'chart',
            chart: { ...payload, ...(entry.port ? { port: entry.port } : {}) },
            type: 'info',
          });
        }

        const summary = firstOf('tensor_summary')?.tensor_summary ?? legacy.output_summary;
        if (summary) {
          store.setTabOutputSummary(tabId, data.node_id, summary);
        }
      };

      const onExecutionComplete = () => {
        const store = useTabStore.getState();
        store.setTabStatus(tabId, 'completed');
        store.addTabLog(tabId, { message: 'Execution completed successfully', type: 'success' });
      };

      const onExecutionError = (raw: unknown) => {
        const data = raw as { error: string; rejected?: boolean };
        const store = useTabStore.getState();
        // A REFUSED submit is not a run outcome (#123). The server turned a
        // click down — the interactive cap, or the one-run-per-session rule
        // — without starting anything, and the run this tab is already
        // following is still executing. Falling through to `error` here
        // would re-enable Run and disable Stop mid-run, taking away the
        // ability to stop the very run the user is watching.
        if (data.rejected) {
          const message = useI18n.getState().t('execution.rejected');
          useToastStore.getState().addToast(message, 'warning');
          store.addTabLog(tabId, { message: `${message} (${data.error})`, type: 'info' });
          return;
        }
        store.setTabStatus(tabId, 'error');
        store.addTabLog(tabId, { message: `Execution error: ${data.error}`, type: 'error' });
        // A fail-fast run re-raises the node's exception, so a missing pack
        // reaches the client here too — untyped, which is why the message
        // has to identify itself. Deliberately after the `rejected` return
        // above: a refused submit ran nothing, and its message is the
        // server's rather than a node's.
        toastMissingPack(data.error);
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

      // The server acknowledged an attach. Its `status` is the run row's,
      // so this — not an optimistic guess at request time — is what puts the
      // tab into `running`: an attach that fails (the run was pruned between
      // the status check and the request) simply never gets here, instead of
      // leaving the tab stuck on "Running" with Run disabled forever.
      const onAttached = (raw: unknown) => {
        const data = raw as { status?: string };
        if (data.status !== 'running' && data.status !== 'queued') return;
        useTabStore.getState().setTabStatus(tabId, 'running');
      };

      // Protocol-level refusals (unknown run, bad cursor, no run service,
      // an older backend answering "Unknown action"). Surfaced rather than
      // swallowed: silence here is how a Stop the server rejected looks
      // exactly like a Stop that worked.
      const onProtocolError = (raw: unknown) => {
        const data = raw as { error?: string };
        const store = useTabStore.getState();
        store.addTabLog(tabId, {
          message: `Execution server: ${data.error ?? 'unknown error'}`,
          type: 'error',
        });

        // A refused attach leaves the tab on `running` with nothing
        // forwarding to it — no frames, no terminal event, Run disabled
        // forever. The reconnect path is where that bites, because it
        // attaches without a preceding status check. Ask REST what the run
        // is really doing rather than guessing from an error string that
        // may not even be about the attach.
        const tab = store.tabs.find((t) => t.id === tabId);
        if (!tab || tab.status !== 'running') return;
        const runId = tab.lastRunId;
        void (async () => {
          let run = null;
          try {
            run = runId ? await getRun(runId) : null;
          } catch {
            return; // server unreachable: it is not ours to declare over
          }
          if (run && RESUMABLE.has(run.status)) return; // still going
          const current = useTabStore.getState();
          const now = current.tabs.find((t) => t.id === tabId);
          if (!now || now.status !== 'running' || now.lastRunId !== runId) return;
          current.setTabStatus(
            tabId, run ? TERMINAL_TAB_STATUS[run.status] ?? 'idle' : 'idle');
        })();
      };

      // #121: every replayed or live frame carries the run it belongs to and
      // its position in that run's event log. Tracking both here — once, for
      // all types — is what lets a dropped socket resume mid-run instead of
      // replaying history that is already on screen.
      const onAnyFrame = (raw: unknown) => {
        const data = raw as { run_id?: unknown; cursor?: unknown };
        const store = useTabStore.getState();
        if (typeof data.run_id === 'string' && data.run_id) {
          store.setLastRunId(tabId, data.run_id);
        }
        if (typeof data.cursor === 'number') {
          store.setLastRunCursor(tabId, data.cursor);
        }
      };

      // The socket came back after a drop. The run kept going without us
      // (that is the point of #120/#121), but its subscription died with the
      // old socket, so nothing is being forwarded until we say so.
      const onReconnected = () => {
        const store = useTabStore.getState();
        const tab = store.tabs.find((t) => t.id === tabId);
        if (!tab || tab.status !== 'running' || !tab.lastRunId) return;
        tab.ws.send({
          action: 'attach',
          run_id: tab.lastRunId,
          cursor: tab.lastRunCursor,
        });
      };

      const entries: WsHandlerEntry[] = [
        { ws, type: 'node_status', handler: onNodeStatus },
        { ws, type: 'execution_complete', handler: onExecutionComplete },
        { ws, type: 'execution_error', handler: onExecutionError },
        { ws, type: 'execution_start', handler: onExecutionStart },
        { ws, type: 'execution_stopped', handler: onExecutionStopped },
        { ws, type: 'attached', handler: onAttached },
        { ws, type: 'error', handler: onProtocolError },
        { ws, type: RECONNECTED_EVENT, handler: onReconnected },
        { ws, type: '*', handler: onAnyFrame },
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

  // ── Re-attach on load (#121) ────────────────────────────────────────────
  // The headline behaviour of the run-service wave, from the browser's side:
  // a run outlives the tab that started it, so on every page load we ask the
  // server whether the run this tab was watching is still going, and pick it
  // back up from the beginning of its log (the panel is empty after a
  // reload, so there is nothing to avoid duplicating).
  //
  // Restoring `running` also re-disables the Run button, which matters more
  // than it looks: without it a user could reload mid-training and start a
  // SECOND run against the same persistent weights.
  useEffect(() => {
    const resume = async (tab: TabState) => {
      const runId = tab.lastRunId;
      if (!runId) return;
      const key = `${tab.id}:${runId}`;
      if (reattached.has(key)) return;
      reattached.add(key);

      let run;
      try {
        run = await getRun(runId);
      } catch {
        return; // server unreachable — nothing to re-attach to
      }
      if (!run || !RESUMABLE.has(run.status)) {
        // The run ended (or was pruned) while the tab was closed. Drop the
        // handle: a restored `lastRunId` only ever means "in flight when we
        // last saved", so keeping a finished one would point the Inspector
        // at a run whose execution is nowhere on screen — and, once the
        // server restarts, whose captured outputs are gone with it.
        useTabStore.getState().setLastRunId(tab.id, null);
        return;
      }
      if (!(await ensureConnected(tab.ws))) return;

      // Re-read the LIVE tab at the last moment. The user can click Run
      // while we are awaiting the status check or the socket: their run is
      // already submitted and attached, and a late attach to the old run
      // would detach it server-side, leaving the new one training
      // invisibly — and Stop would then cancel the wrong one.
      const live = useTabStore.getState().tabs.find((t) => t.id === tab.id);
      if (!live || live.lastRunId !== runId || live.status === 'running') return;

      // The tab goes back to `running` when the server ACKNOWLEDGES the
      // attach (see onAttached), not here — an attach the server refuses
      // must not leave the tab permanently disabled, and `lastRunId` is
      // persisted precisely while the status is `running`, so a wrong guess
      // would be written back and retried on every future page load.
      tab.ws.send({ action: 'attach', run_id: runId, cursor: 0 });
      useToastStore.getState().addToast(
        useI18n.getState().t('execution.reattached'),
        'info',
      );
    };

    // No cleanup, and no `cancelled` flag it could set — see `reattached`.
    for (const tab of useTabStore.getState().tabs) void resume(tab);
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

    if (!(await ensureConnected(ws))) {
      addTabLog(tab.id, { message: 'Failed to connect to execution server', type: 'error' });
      return;
    }

    const graph = getSerializedGraph();
    // Filter out note nodes — they are annotations, not computational
    const execNodes = graph.nodes.filter((n: any) => n.type !== 'note');

    // Pre-execution validation
    try {
      // Embedded presets ride along so a portable graph whose presets are
      // not in the local registry still validates (#84).
      const validation = await validateGraph(
        execNodes, graph.edges, graph.presets, graph.subgraphs,
      );
      if (!validation.valid) {
        const { addToast } = useToastStore.getState();
        validation.errors.forEach((err: string) => addToast(err, 'error'));
        return;
      }
    } catch {
      // If validation endpoint is unreachable, proceed anyway
    }

    clearLogs();
    // Drop anything the previous run left buffered BEFORE resetting the
    // nodes to idle (#125): a patch that survived the reset would land one
    // frame later and paint the old run's status onto the new one.
    discardTabNodeUpdates(tab.id);
    clearExecutionStatus();
    clearOutputSummaries();
    setTabStatus(tab.id, 'running');

    // Partial re-execution: pass changed_nodes hint to backend
    const { getDirtyWithDownstream, clearDirty } = useTabStore.getState();
    const changedNodes = getDirtyWithDownstream();
    clearDirty();

    // Forget the previous run BEFORE submitting. Stop is enabled the moment
    // the status flips to `running`, but the new run's id only arrives with
    // `attached`, so a Stop clicked in that window would otherwise name the
    // PREVIOUS run — cancelling nothing (or, if retention had pruned it,
    // getting an `execution_stopped` that unsticks the UI while the new run
    // keeps training). With no id, `cancel` falls back to the server's own
    // attachment, which is already correct because the socket handles
    // messages serially: `execute` finishes attaching before `cancel` is
    // read.
    useTabStore.getState().setLastRunId(tab.id, null);

    // No client-minted run id since #121: the RUN is created server-side by
    // RunService, and its `exec_runs.id` is the one id — the execution id,
    // the captured-outputs key and the handle `attach` takes. It comes back
    // on `execution_start` and lands in `lastRunId`, which is what the
    // Inspector reads and what a re-attach after F5 resumes from.
    ws.send({
      action: 'execute',
      nodes: execNodes,
      edges: graph.edges,
      presets: graph.presets,
      subgraphs: graph.subgraphs,
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
      // core#134: reproducibility. Sent only when set — `seed: null` is a
      // valid option value, but omitting it keeps the message byte-identical
      // to the pre-#134 one for everyone who never touches the field.
      // `!= null` covers BOTH null and undefined: a tab persisted before
      // #134 has no `seed` key at all, and `{seed: undefined}` would
      // serialise to a message shape nobody expects.
      ...(tab.seed != null ? { seed: tab.seed } : {}),
      ...(tab.deterministic ? { deterministic: true } : {}),
      ...(changedNodes.length > 0 ? { changed_nodes: changedNodes } : {}),
    });
  }, [getActiveTab, getSerializedGraph, clearLogs, clearExecutionStatus, clearOutputSummaries, setTabStatus, addTabLog]);

  // Explicit cancel, naming the run (#121). Closing the tab, navigating away
  // and losing the connection all leave the run alone now — this is the only
  // thing that stops it, so it has to say WHICH run rather than "whatever
  // this socket happens to be doing".
  const stop = useCallback(() => {
    const tab = getActiveTab();
    tab.ws.send({
      action: 'cancel',
      ...(tab.lastRunId ? { run_id: tab.lastRunId } : {}),
    });
  }, [getActiveTab]);

  return { execute, stop };
}
