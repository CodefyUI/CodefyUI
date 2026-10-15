import { create } from 'zustand';
import {
  drainWorker,
  fetchWorkerList,
  removeWorker,
  undrainWorker,
  type WorkerInfo,
  type WorkerRejection,
} from '../api/workers';
import { cachedAppVersion } from '../utils/appVersion';
import { useI18n } from '../i18n';
import { useToastStore } from './toastStore';

/**
 * The Workers panel's state (#776): the worker list, its poll, and whether
 * the panel is open.
 *
 * The poll is reference-counted the way `runStore.watch()` is: the first
 * `watch()` starts it, the last release stops it, so the list is read every
 * {@link WORKER_POLL_MS} only while something is showing it.
 *
 * `panelOpen` lives here, not in the panel's own state, because two places
 * open it (Settings -> Execution and the Runs panel header) and the modal
 * registry (`modalState.ts`) has to see it.
 *
 * Responses can land out of order: a list read that started before a drain
 * may answer after it, carrying the worker as it was. Every request takes a
 * sequence number when it is SENT, and a response older than the newest one
 * already applied is dropped, so the panel never steps back in time.
 */

export const WORKER_POLL_MS = 5000;

interface WorkerState {
  workers: WorkerInfo[];
  /** Each worker name's last refused handshake, newest first (#781). */
  rejections: WorkerRejection[];
  /** False until the first read answered (or failed). */
  loaded: boolean;
  error: string | null;
  /**
   * This server's version, for the "matches this server" check: the one the
   * app's last `/api/health` read reported (`cachedAppVersion`), taken on
   * every poll rather than asked for again.
   */
  serverVersion: string | null;
  /** Worker ids with a drain / undrain / remove in flight. */
  busy: Record<string, boolean>;
  panelOpen: boolean;
  /**
   * Bumped when a run link in the panel asks the dock to show the Runs tab;
   * `ResultsPanel` follows it.
   */
  runsRevealSeq: number;
  openPanel: () => void;
  closePanel: () => void;
  refresh: () => Promise<void>;
  setDraining: (workerId: string, draining: boolean) => Promise<void>;
  /** Revoke the worker's credential and drop it from the list (#780). */
  remove: (workerId: string) => Promise<void>;
  /**
   * Close the panel and ask the dock to show its Runs tab. The caller
   * selects the run (`runStore.select`); this store stays out of the run
   * store's import graph, which the modal registry would otherwise pull in.
   */
  revealRuns: () => void;
  /** Start polling; call the returned function to release this watcher. */
  watch: () => () => void;
}

let watchers = 0;
let pollTimer: ReturnType<typeof setTimeout> | null = null;
/** The number the next request takes. */
let nextSeq = 1;
/** The newest request whose response was applied. */
let appliedSeq = 0;

/** True when the response to request *seq* may be applied; records it if so. */
function claim(seq: number): boolean {
  if (seq < appliedSeq) return false;
  appliedSeq = seq;
  return true;
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

async function tick() {
  // Called by the first watch() and by the timer, which the last release
  // clears, so there is always a watcher here.
  pollTimer = null;
  await useWorkerStore.getState().refresh();
  if (watchers > 0 && pollTimer === null) pollTimer = setTimeout(tick, WORKER_POLL_MS);
}

export const useWorkerStore = create<WorkerState>((set, get) => ({
  workers: [],
  rejections: [],
  loaded: false,
  error: null,
  serverVersion: null,
  busy: {},
  panelOpen: false,
  runsRevealSeq: 0,

  openPanel: () => set({ panelOpen: true }),
  closePanel: () => set({ panelOpen: false }),

  refresh: async () => {
    const seq = nextSeq++;
    try {
      const { workers, rejections } = await fetchWorkerList();
      if (!claim(seq)) return;
      set({ workers, rejections, loaded: true, error: null, serverVersion: cachedAppVersion() });
    } catch (err) {
      if (!claim(seq)) return;
      set({ loaded: true, error: errorMessage(err), serverVersion: cachedAppVersion() });
    }
  },

  setDraining: async (workerId, draining) => {
    if (get().busy[workerId]) return;
    set((s) => ({ busy: { ...s.busy, [workerId]: true } }));
    const seq = nextSeq++;
    try {
      const updated = await (draining ? drainWorker(workerId) : undrainWorker(workerId));
      if (claim(seq)) {
        set((s) => ({ workers: s.workers.map((w) => (w.id === workerId ? updated : w)) }));
      }
    } catch (err) {
      const { t } = useI18n.getState();
      const name = get().workers.find((w) => w.id === workerId)?.name ?? workerId;
      useToastStore.getState().addToast(
        t(draining ? 'workers.drain.failed' : 'workers.undrain.failed', {
          name,
          error: errorMessage(err),
        }),
        'error',
      );
    } finally {
      set((s) => {
        const busy = { ...s.busy };
        delete busy[workerId];
        return { busy };
      });
    }
  },

  remove: async (workerId) => {
    if (get().busy[workerId]) return;
    set((s) => ({ busy: { ...s.busy, [workerId]: true } }));
    const seq = nextSeq++;
    try {
      await removeWorker(workerId);
      if (claim(seq)) set((s) => ({ workers: s.workers.filter((w) => w.id !== workerId) }));
    } catch (err) {
      const { t } = useI18n.getState();
      const name = get().workers.find((w) => w.id === workerId)?.name ?? workerId;
      useToastStore.getState().addToast(
        t('workers.remove.failed', { name, error: errorMessage(err) }),
        'error',
      );
    } finally {
      set((s) => {
        const busy = { ...s.busy };
        delete busy[workerId];
        return { busy };
      });
    }
  },

  revealRuns: () => set((s) => ({ panelOpen: false, runsRevealSeq: s.runsRevealSeq + 1 })),

  watch: () => {
    watchers += 1;
    if (watchers === 1) void tick();
    let released = false;
    return () => {
      if (released) return;
      released = true;
      watchers -= 1;
      if (watchers > 0) return;
      if (pollTimer !== null) {
        clearTimeout(pollTimer);
        pollTimer = null;
      }
    };
  },
}));

/** Test-only: reset the poll and the store between cases. */
export function _resetWorkerStoreForTesting(): void {
  watchers = 0;
  nextSeq = 1;
  appliedSeq = 0;
  if (pollTimer !== null) clearTimeout(pollTimer);
  pollTimer = null;
  useWorkerStore.setState({
    workers: [],
    rejections: [],
    loaded: false,
    error: null,
    serverVersion: null,
    busy: {},
    panelOpen: false,
    runsRevealSeq: 0,
  });
}
