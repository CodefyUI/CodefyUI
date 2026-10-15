import { create } from 'zustand';
import { drainWorker, fetchWorkers, undrainWorker, type WorkerInfo } from '../api/workers';
import { fetchHealth } from '../api/rest';
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
 */

export const WORKER_POLL_MS = 5000;

interface WorkerState {
  workers: WorkerInfo[];
  /** False until the first read answered (or failed). */
  loaded: boolean;
  error: string | null;
  /** This server's version, for the "matches this server" check. */
  serverVersion: string | null;
  /** Worker ids with a drain / undrain in flight. */
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

async function readServerVersion() {
  try {
    const health = await fetchHealth();
    useWorkerStore.setState({ serverVersion: health.version });
  } catch {
    // The check is simply not shown; the list's own error reports an
    // unreachable server.
  }
}

export const useWorkerStore = create<WorkerState>((set, get) => ({
  workers: [],
  loaded: false,
  error: null,
  serverVersion: null,
  busy: {},
  panelOpen: false,
  runsRevealSeq: 0,

  openPanel: () => set({ panelOpen: true }),
  closePanel: () => set({ panelOpen: false }),

  refresh: async () => {
    try {
      const workers = await fetchWorkers();
      set({ workers, loaded: true, error: null });
    } catch (err) {
      set({ loaded: true, error: errorMessage(err) });
    }
  },

  setDraining: async (workerId, draining) => {
    if (get().busy[workerId]) return;
    set((s) => ({ busy: { ...s.busy, [workerId]: true } }));
    try {
      const updated = await (draining ? drainWorker(workerId) : undrainWorker(workerId));
      set((s) => ({ workers: s.workers.map((w) => (w.id === workerId ? updated : w)) }));
    } catch (err) {
      const { t } = useI18n.getState();
      useToastStore.getState().addToast(
        t(draining ? 'workers.drain.failed' : 'workers.undrain.failed', {
          name: workerId,
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

  revealRuns: () => set((s) => ({ panelOpen: false, runsRevealSeq: s.runsRevealSeq + 1 })),

  watch: () => {
    watchers += 1;
    if (watchers === 1) {
      if (get().serverVersion === null) void readServerVersion();
      void tick();
    }
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
  if (pollTimer !== null) clearTimeout(pollTimer);
  pollTimer = null;
  useWorkerStore.setState({
    workers: [],
    loaded: false,
    error: null,
    serverVersion: null,
    busy: {},
    panelOpen: false,
    runsRevealSeq: 0,
  });
}
