import { useCallback, useEffect, useRef } from 'react';
import type { ExecutionStatus } from '../types';
import { flushSubgraphEditing, useTabStore, type TabState } from '../store/tabStore';
import {
  queueTabNodeProgress,
  queueTabNodeStatus,
  discardTabNodeUpdates,
  flushTabNodeUpdates,
} from '../store/nodeUpdateQueue';
import { useToastStore } from '../store/toastStore';
import { useUIStore } from '../store/uiStore';
import { usePackStore } from '../store/packStore';
import { getRun, validateGraph } from '../api/rest';
import { findEntryPoints } from '../utils/findEntryPoints';
import { localizedPackTitle } from '../utils/packAvailability';
import { friendlyError, missingPackFromError } from '../utils/errorMessages';
import { dismissValidationToasts, issuesFromErrors, showValidationError, showValidationIssues } from '../utils/validationToasts';
import { useI18n } from '../i18n';
import {
  MESSAGE_TOO_BIG_EVENT,
  RECONNECTED_EVENT,
  type ExecutionWebSocket,
} from '../api/ws';

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

/** True for a run status that means the run is over. */
function isFinishedRunStatus(status: unknown): status is string {
  return typeof status === 'string'
    && Object.prototype.hasOwnProperty.call(TERMINAL_TAB_STATUS, status);
}

/**
 * Say why a tab left Running when nobody pressed Stop (#552): a toast,
 * because the Execution Log may be closed, and the same words in the log,
 * because the toast is gone after four seconds.
 *
 * One toast for the same words, however many tabs say them: a restart ends
 * every running tab's run at once, and identical toasts stacked on each
 * other read as separate problems. Each tab's log still gets its line.
 */
function sayWhyRunEnded(tabId: string, message: string): void {
  const toasts = useToastStore.getState();
  if (!toasts.toasts.some((toast) => toast.message === message)) {
    toasts.addToast(message, 'warning');
  }
  useTabStore.getState().addTabLog(tabId, { message, type: 'error' });
}

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
 * "Persist weights between runs" as each tab's run in flight was sent with
 * it, for the note that run's completion logs. Read once there and dropped.
 */
const keptWeightsAtSubmit = new Map<string, boolean>();

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

  // Validation's toasts are about one tab's graph and do not say which, so
  // they go when another tab comes to the front, and with this hook, which
  // the toolbar takes down with the last tab (dismissValidationToasts).
  const activeTabId = useTabStore((s) => s.activeTabId);
  useEffect(() => () => dismissValidationToasts(), [activeTabId]);

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

      // Runs this tab has seen end (#552): a closing frame, a terminal
      // `attached`, or REST's answer. Two things read it. A replay of an
      // ended run can open with `execution_start`, which must not put the
      // tab back on Running with nothing left to end it. And a refused Run
      // that names one of these runs names nothing that is still going.
      //
      // Per mount of this effect, which is enough: the hook lives in the
      // Toolbar, which unmounts only when no tab is left (every closed
      // tab's socket is disconnected), and the app's error screen recovers
      // by reloading the page. A remount therefore only ever meets sockets
      // that have carried no run under the previous mount.
      const closedRuns = new Set<string>();
      const noteClosed = (runId: unknown) => {
        if (typeof runId === 'string' && runId) closedRuns.add(runId);
      };
      // The run whose interruption this tab has already reported. The
      // terminal `attached` and the closing frame its replay ends with both
      // say so; the user hears it once.
      let reportedRunId: string | null = null;

      // A card a run left running never gets a frame of its own: one the dead
      // process was running, or a block's card whose other inner nodes a Stop
      // never ran (a run from inside a block too). It would keep its Running
      // border, footer and epoch bar. Flushed first, so a node a replay has
      // just painted running is included.
      const settleRunningNodes = () => {
        flushTabNodeUpdates();
        const live = useTabStore.getState().tabs.find((t) => t.id === tabId);
        // The top-level canvas, where a run's statuses land: stashed in the
        // first frame on a run from inside a block (`applyTabNodeUpdates`).
        const topLevel = live?.subgraphStack?.[0]?.nodes ?? live?.nodes ?? [];
        for (const node of topLevel) {
          if (node.data?.executionStatus === 'running') {
            queueTabNodeStatus(tabId, node.id, 'interrupted');
          }
        }
      };

      // End the tab's run on the server's word. An interruption is the one
      // outcome that also needs words: the tab can only show "Idle", and
      // nothing in the run's own frames says the server went away. Its cards
      // are settled on EVERY call, because the closing frame that ends a
      // replay can follow frames that marked a node running again.
      const endRun = (runId: string | null, runStatus: string) => {
        noteClosed(runId);
        useTabStore.getState().setTabStatus(tabId, TERMINAL_TAB_STATUS[runStatus] ?? 'idle');
        // A Stop the server reports (`cancelled`) leaves the cards the live
        // Stop does, on a run from inside a block too: settled the same.
        if (runStatus === 'cancelled') settleRunningNodes();
        if (runStatus !== 'interrupted') return;
        settleRunningNodes();
        if (runId !== null && runId === reportedRunId) return;
        reportedRunId = runId;
        sayWhyRunEnded(tabId, useI18n.getState().t('status.runInterrupted'));
      };

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
          }
          // A frame over the server's event cap arrives with no payload: the
          // entry elided, or the whole event collapsed to a marker (#486). It
          // is still a progress frame, so it never becomes the node's status
          // or a Log line; the card keeps the last frame that did arrive.
          return;
        }

        queueTabNodeStatus(tabId, data.node_id, data.status, data.error);

        // Suppress running/cached chatter — only surface terminal transitions.
        if (data.status !== 'running' && data.status !== 'cached') {
          // This event's OWN tab (captured above), not whichever tab is on
          // screen -- a background tab's run must label its log from its
          // own nodes, never the active tab's (#163).
          const eventTab = store.tabs.find((t) => t.id === tabId);
          // From the whole graph's top level: the server reports a node inside
          // a block under its outermost card, and on a run from inside a block
          // (or one still running when a block was opened) `nodes` holds only
          // the open block's canvas.
          const topNodes = eventTab ? flushSubgraphEditing(eventTab).nodes : [];
          const nodeLabel = topNodes.find((n) => n.id === data.node_id)?.data?.label
            ?? String(data.node_id).slice(0, 8);

          // Map the exception here, where `error_type` is still available --
          // the panels only ever see the composed line, and the type cannot be
          // recovered from `str(exc)` afterwards.
          const detail = data.error ? friendlyError(data.error, data.error_type) : '';

          // Only on the terminal `error` frame, so this runs once per failing
          // node rather than on every status this node reports.
          if (data.status === 'error') {
            toastMissingPack(data.error, data.error_type);
          }

          // runLog.*: the editor's own line, in the language the UI is in. A
          // status with no line of its own (a newer backend's) keeps its token.
          const runLogKey = (() => {
            switch (data.status) {
              case 'completed':
                return 'runLog.node.completed';
              case 'skipped':
                return 'runLog.node.skipped';
              case 'interrupted':
                return 'runLog.node.interrupted';
              case 'error':
                return detail ? 'runLog.node.error' : 'runLog.node.errorBare';
              default:
                return 'runLog.node.other';
            }
          })();
          store.addTabLog(tabId, {
            nodeId: data.node_id,
            // The runLog.* slots in this order: t() fills one at a time, and an
            // error or a node title can itself contain braces. So every
            // translation must keep {label} before {detail}.
            message: useI18n.getState().t(runLogKey, {
              status: String(data.status),
              detail,
              label: nodeLabel,
            }),
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

      const onExecutionComplete = (raw: unknown) => {
        noteClosed((raw as { run_id?: unknown }).run_id);
        const store = useTabStore.getState();
        store.setTabStatus(tabId, 'completed');
        store.addTabLog(tabId, { message: useI18n.getState().t('runLog.completed'), type: 'success' });
        // With weights kept, this run's numbers may include earlier runs'
        // training, which an exported script never has. The switch as this
        // run was sent with it: flipping it mid-run changes the next run, not
        // this one. A run re-attached after a reload was sent by an earlier
        // page, so the tab's setting stands in. Shown whether or not the
        // graph owns weights: the canvas cannot tell, and only a user who
        // turned the switch on sees it.
        const kept = keptWeightsAtSubmit.get(tabId) ?? store.getTab(tabId)?.weightsPersistent;
        keptWeightsAtSubmit.delete(tabId);
        if (kept) {
          store.addTabLog(tabId, {
            message: useI18n.getState().t('settings.persist.runNote'),
            type: 'info',
          });
        }
      };

      const onExecutionError = (raw: unknown) => {
        const data = raw as { error: string; rejected?: boolean; run_id?: unknown };
        const store = useTabStore.getState();
        // A REFUSED submit is not a run outcome (#123). The server turned a
        // click down — the interactive cap, or the one-run-per-session rule
        // — without starting anything, and the run this tab is already
        // following is still executing. Falling through to `error` here
        // would re-enable Run and disable Stop mid-run, taking away the
        // ability to stop the very run the user is watching.
        //
        // Unless nothing of this tab's is going at all (#552). The refusal
        // names the run its socket is attached to: none, or a run this tab
        // has seen end, means the refused click was the only thing the tab
        // was waiting for -- typically the interactive cap, filled by runs in
        // other tabs -- and staying on Running would wait for a run that
        // does not exist. A run it has NOT seen end is live: a second Run
        // click that landed while the first was still validating is refused
        // for the first click's run, and the tab must keep following it.
        // (Run clears `lastRunId` before it sends, so that alone cannot tell
        // the two apart; the '*' handler re-points it at the named run.)
        if (data.rejected) {
          const tab = store.tabs.find((t) => t.id === tabId);
          const attachedTo =
            typeof data.run_id === 'string' && data.run_id ? data.run_id : null;
          const notStarted = tab?.status === 'running' && !tab.lastRunId
            && (attachedTo === null || closedRuns.has(attachedTo));
          const message = useI18n.getState().t(
            notStarted ? 'status.runNotStarted' : 'execution.rejected');
          useToastStore.getState().addToast(message, 'warning');
          store.addTabLog(tabId, { message: `${message} (${data.error})`, type: 'info' });
          if (notStarted) store.setTabStatus(tabId, 'idle');
          return;
        }
        noteClosed(data.run_id);
        store.setTabStatus(tabId, 'error');
        store.addTabLog(tabId, { message: useI18n.getState().t('runLog.error', { error: data.error }), type: 'error' });
        // A fail-fast run re-raises the node's exception, so a missing pack
        // reaches the client here too — untyped, which is why the message
        // has to identify itself. Deliberately after the `rejected` return
        // above: a refused submit ran nothing, and its message is the
        // server's rather than a node's.
        toastMissingPack(data.error);
      };

      const onExecutionStart = (raw: unknown) => {
        const data = raw as { run_id?: string };
        // Replayed history of a run that has already ended (#552).
        if (typeof data.run_id === 'string' && closedRuns.has(data.run_id)) return;
        const store = useTabStore.getState();
        store.setTabStatus(tabId, 'running');
        if (typeof data.run_id === 'string') {
          store.setLastRunId(tabId, data.run_id);
        }
        store.addTabLog(tabId, { message: useI18n.getState().t('runLog.started'), type: 'info' });
      };

      const onExecutionStopped = (raw: unknown) => {
        const data = raw as { run_id?: unknown; reason?: unknown };
        // The server going away, not a user's Stop (#552): a graceful
        // shutdown sends this live, and startup recovery writes it as the
        // last frame of every run the previous process died under.
        if (data.reason === 'interrupted') {
          endRun(typeof data.run_id === 'string' ? data.run_id : null, 'interrupted');
          return;
        }
        noteClosed(data.run_id);
        // A user's Stop: a block's card whose other inner nodes never ran gets
        // no closing frame, on a run from inside a block or at the top level.
        // Settled as a node that stopped early is: interrupted.
        settleRunningNodes();
        const store = useTabStore.getState();
        store.setTabStatus(tabId, 'idle');
        store.addTabLog(tabId, { message: useI18n.getState().t('runLog.cancelled'), type: 'info' });
      };

      // The server acknowledged an attach. Its `status` is the run row's,
      // so this — not an optimistic guess at request time — is what puts the
      // tab into `running`: an attach that fails (the run was pruned between
      // the status check and the request) simply never gets here, instead of
      // leaving the tab stuck on "Running" with Run disabled forever.
      //
      // A finished status ends the tab's run instead (#552). The usual case
      // is the re-attach after a server restart, whose startup recovery
      // filed the run `interrupted`. The replay that follows cannot be
      // relied on to close it (a run retired by an older server has no
      // closing frame), so this ack ends it, and holds against the replay.
      const onAttached = (raw: unknown) => {
        const data = raw as { run_id?: unknown; status?: unknown };
        if (data.status === 'running' || data.status === 'queued') {
          useTabStore.getState().setTabStatus(tabId, 'running');
          return;
        }
        if (!isFinishedRunStatus(data.status)) return;
        endRun(typeof data.run_id === 'string' ? data.run_id : null, data.status);
      };

      // Ask REST what the run is really doing, and end the tab's run if it
      // is over. For the answers after which no frame will ever end it: a
      // refused attach, and a Stop that found nothing to stop.
      const settleFromServer = (runId: string | null) => {
        void (async () => {
          let run = null;
          try {
            run = runId ? await getRun(runId) : null;
          } catch {
            return; // server unreachable: it is not ours to declare over
          }
          if (run && RESUMABLE.has(run.status)) return; // still going
          const now = useTabStore.getState().tabs.find((t) => t.id === tabId);
          if (!now || now.status !== 'running' || now.lastRunId !== runId) return;
          if (run) endRun(runId, run.status);
          else useTabStore.getState().setTabStatus(tabId, 'idle');
        })();
      };

      // Protocol-level refusals (unknown run, bad cursor, no run service,
      // an older backend answering "Unknown action"). Surfaced rather than
      // swallowed: silence here is how a Stop the server rejected looks
      // exactly like a Stop that worked.
      const onProtocolError = (raw: unknown) => {
        const data = raw as { error?: string };
        const store = useTabStore.getState();
        store.addTabLog(tabId, {
          message: useI18n.getState().t('runLog.serverError', {
            error: data.error ?? useI18n.getState().t('runLog.unknownError'),
          }),
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
        settleFromServer(tab.lastRunId);
      };

      // The answer to Stop (#552). `cancelled: true` means the request was
      // delivered, and the run's closing frame follows on the attachment.
      // `cancelled: false` means there was nothing to stop -- the run is
      // already over, or this server never had it -- and no closing frame is
      // coming, so the tab would sit on Running with Stop doing nothing.
      const onCancelAck = (raw: unknown) => {
        const data = raw as { run_id?: unknown; cancelled?: unknown };
        if (data.cancelled !== false) return;
        const tab = useTabStore.getState().tabs.find((t) => t.id === tabId);
        if (!tab || tab.status !== 'running') return;
        settleFromServer(
          typeof data.run_id === 'string' && data.run_id ? data.run_id : tab.lastRunId);
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
        if (!tab || tab.status !== 'running') return;
        if (!tab.lastRunId) {
          // Running with no run id (#552): Run's execute frame went out on
          // the socket that just dropped, and whatever answered it went
          // with it. Nothing will ever arrive for that click on this
          // socket, and there is no run id to re-attach to.
          store.setTabStatus(tabId, 'idle');
          sayWhyRunEnded(tabId, useI18n.getState().t('status.runUnconfirmed'));
          return;
        }
        tab.ws.send({
          action: 'attach',
          run_id: tab.lastRunId,
          cursor: tab.lastRunCursor,
        });
      };

      // The server closed the socket with 1009 (#552): the last frame sent --
      // in practice a Run's execute message -- was refused unread, so a tab
      // still waiting for that Run's answer will never get one. Leave Running
      // now rather than at the reconnect, where it would read as a dropped
      // connection. The socket layer has already raised the toast; the log
      // gets the same words.
      const onMessageTooBig = () => {
        const store = useTabStore.getState();
        const tab = store.tabs.find((t) => t.id === tabId);
        if (!tab || tab.status !== 'running' || tab.lastRunId) return;
        store.setTabStatus(tabId, 'idle');
        store.addTabLog(tabId, {
          message: useI18n.getState().t('connection.tooLarge'),
          type: 'error',
        });
      };

      const entries: WsHandlerEntry[] = [
        { ws, type: 'node_status', handler: onNodeStatus },
        { ws, type: 'execution_complete', handler: onExecutionComplete },
        { ws, type: 'execution_error', handler: onExecutionError },
        { ws, type: 'execution_start', handler: onExecutionStart },
        { ws, type: 'execution_stopped', handler: onExecutionStopped },
        { ws, type: 'attached', handler: onAttached },
        { ws, type: 'cancel_ack', handler: onCancelAck },
        { ws, type: 'error', handler: onProtocolError },
        { ws, type: RECONNECTED_EVENT, handler: onReconnected },
        { ws, type: MESSAGE_TOO_BIG_EVENT, handler: onMessageTooBig },
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

  const submit = useCallback(async (tab: TabState) => {
    // The last Run's validation toasts go first, whatever this one finds:
    // they never time out, and a stale set reads as this Run's problems.
    dismissValidationToasts();
    // Block execution when the graph has no entry points. This mirrors the
    // backend `find_entry_points` so we fail fast with a toast instead of
    // sending a graph that will be rejected server-side.
    //
    // The whole graph's, also on a run from inside a block: the run sends the
    // whole graph (`getSerializedGraph` closes every open level), and the open
    // block's own canvas never holds a Start, so asking it refused every such
    // run with "no entry point".
    const whole = flushSubgraphEditing(tab);
    const entryIds = findEntryPoints(whole.nodes, whole.edges);
    if (entryIds.length === 0) {
      // In the validation set (utils/validationToasts), so the next Run or a
      // tab switch takes it down.
      showValidationError(useI18n.getState().t('execution.error.noEntryPoints'));
      return;
    }

    const ws = tab.ws;

    // Everything read from the ACTIVE tab is read here, before the first
    // await: the user can switch tabs while the socket connects or the graph
    // is checked, and these getters only reach the active tab (#552).
    //
    // The run needs any SECRET param the user typed (an LLM API key), so its
    // message is built with the keys kept; the server keeps its stored copy
    // of the run scrubbed (#251). Validation has no use for a key and gets
    // the blanked graph. Both are taken before the next await, so they
    // describe the same graph.
    const graph = getSerializedGraph({ keepSecrets: true });
    const checked = getSerializedGraph();
    const dirtyAtClick = useTabStore.getState().getDirtyWithDownstream();

    if (!(await ensureConnected(ws))) {
      addTabLog(tab.id, { message: useI18n.getState().t('runLog.connectFailed'), type: 'error' });
      return;
    }

    // Filter out note nodes — they are annotations, not computational
    const isComputational = (n: any) => n.type !== 'note';
    const execNodes = graph.nodes.filter(isComputational);

    // Pre-execution validation
    // Embedded presets ride along so a portable graph whose presets are
    // not in the local registry still validates (#84). Only the request is
    // caught: an endpoint that cannot be reached lets the run go ahead (null
    // here), while a fault in drawing the toasts (utils/validationToasts)
    // must not be read as that and run a graph the server refused.
    const validation = await validateGraph(
      checked.nodes.filter(isComputational), checked.edges, checked.presets, checked.subgraphs,
    ).catch(() => null);
    if (validation?.valid === false) {
      // Localized, naming each node by its title, and replacing the last set
      // rather than stacking on it. A server older than `issues` sends the
      // sentences alone.
      showValidationIssues(tab.id, validation.issues ?? issuesFromErrors(validation.errors));
      return;
    }

    // The socket can close while validation is in flight -- a server
    // restart is the usual reason -- and `ws.send` drops what it cannot
    // deliver, which left the tab on Running with no run behind it (#552).
    // Checked again after the LAST await: nothing from here to the send
    // yields, so the execute frame leaves on an open socket.
    if (!(await ensureConnected(ws))) {
      addTabLog(tab.id, { message: useI18n.getState().t('runLog.connectFailed'), type: 'error' });
      return;
    }

    // Drop anything the previous run left buffered BEFORE resetting the
    // nodes to idle (#125): a patch that survived the reset would land one
    // frame later and paint the old run's status onto the new one.
    discardTabNodeUpdates(tab.id);
    // Partial re-execution: pass changed_nodes hint to backend
    let changedNodes = dirtyAtClick;
    // The store's run resets only reach the ACTIVE tab, so they run only
    // while that is still this one (#552). After a switch they would wipe
    // the other tab's log, cards and dirty set, and send its dirty set as
    // this run's. A tab switched away from keeps its last run's log and
    // cards until the new run's frames arrive, and keeps its dirty set:
    // those nodes run again next time, which is slower but never stale.
    if (useTabStore.getState().activeTabId === tab.id) {
      clearLogs();
      clearExecutionStatus();
      clearOutputSummaries();
      const { getDirtyWithDownstream, clearDirty } = useTabStore.getState();
      changedNodes = getDirtyWithDownstream();
      clearDirty();
    }
    setTabStatus(tab.id, 'running');

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

    // The note this run's completion logs follows the switch as it is sent
    // here, not as it may stand by then.
    keptWeightsAtSubmit.set(tab.id, tab.weightsPersistent);

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
      // The graph's own `settings` (its device and seed) ride along, and
      // the run's snapshot records the device. Sent only when the graph has
      // one of them, so a graph with neither keeps the message shape it had
      // before. The run itself uses the explicit `device` and `seed` below.
      ...(graph.settings ? { settings: graph.settings } : {}),
      record_outputs: tab.recordOutputs,
      // A1: verbose step-trace mode
      verbose_mode: tab.verboseMode,
      // A2: weight persistence — backend NodeStateStore keys modules by graph_id
      graph_id: tab.graphId,
      weights_persistent: tab.weightsPersistent,
      // A3: gradient capture
      backward_mode: tab.backwardMode,
      auto_backward: tab.autoBackward,
      // Device for this run: the graph's own settings.device, else the
      // browser Settings device. Always sent: the backend treats a missing
      // device as cpu, and only an explicit "auto" resolves to the best
      // device. A node whose own device param is not "auto" wins over this
      // on the backend.
      device: graph.settings?.device ?? useUIStore.getState().globalDevice,
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
  }, [getSerializedGraph, clearLogs, clearExecutionStatus, clearOutputSummaries, setTabStatus, addTabLog]);

  // Tabs whose Run is between the click and its execute frame (#552). Run
  // stays clickable until the tab is on Running, which comes after
  // validation's round trip -- under a busy server long enough for a second
  // click to land. That click would send a second execute frame the server
  // refuses, and wipe the first run's log and cards on its way; it is
  // ignored instead. Released however `submit` ends (sent, gave up, threw).
  const submitting = useRef(new Set<string>());

  const execute = useCallback(async () => {
    const tab = getActiveTab();
    if (submitting.current.has(tab.id)) return;
    submitting.current.add(tab.id);
    try {
      await submit(tab);
    } finally {
      submitting.current.delete(tab.id);
    }
  }, [getActiveTab, submit]);

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
