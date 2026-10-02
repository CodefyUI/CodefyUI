import { create } from 'zustand';
import {
  ApiError,
  cancelSweep as cancelSweepRequest,
  createSweep as createSweepRequest,
  getRunMetrics,
  getSweep,
  type CreateSweepRequest,
  type SweepDetail,
} from '../api/rest';
import {
  loadObjectiveCurves,
  type SweepCurve,
  type SweepCurveCache,
} from '../components/ResultsPanel/sweepCurves';

export const SWEEP_POLL_MS = 2000;

export type SweepCreateState = 'idle' | 'creating';

interface SweepState {
  selectedSweepId: string | null;
  detail: SweepDetail | null;
  createState: SweepCreateState;
  /** Why the last create was refused. The dialog shows it. */
  createError: string | null;
  /** Why the last read of the selected sweep failed; the next good read clears it. */
  error: string | null;
  /** The server answered 404 for the selected sweep, so polling has stopped. */
  notFound: boolean;
  /** A Stop request is in flight; a second one would only report 0 stopped. */
  cancelPending: boolean;
  /** Why the last Stop failed. Kept apart from `error` so a poll cannot wipe it. */
  cancelError: string | null;
  curves: SweepCurve[];
  /** Number acknowledged by the last cancel request, for truthful UI copy. */
  cancelledRequested: number | null;
  /**
   * True while the New Sweep dialog is mounted. The dialog sets it itself, so
   * the flag cannot outlive it however the Runs panel goes away; `modalState`
   * reads it.
   */
  newSweepOpen: boolean;
  /**
   * The canvas tab each sweep was created from in this session, by sweep id.
   * The detail names its columns from that graph only: another graph can
   * reuse the same short node ids.
   */
  origins: Record<string, string>;

  /** Resolves true when the server created the sweep. */
  createSweep: (request: CreateSweepRequest, originTabId?: string) => Promise<boolean>;
  openSweep: (sweepId: string) => Promise<void>;
  closeSweep: () => void;
  refreshSweep: () => Promise<void>;
  cancelSweep: () => Promise<void>;
  /** The Runs panel's mount: attach, re-read the selected sweep, poll it again. */
  resumePolling: () => Promise<void>;
  startPolling: () => void;
  /** The Runs panel's unmount: detach, so nothing polls until it is back. */
  stopPolling: () => void;
  loadCurves: () => Promise<void>;
  setNewSweepOpen: (open: boolean) => void;
  reset: () => void;
}

const initialState = {
  selectedSweepId: null,
  detail: null,
  createState: 'idle' as SweepCreateState,
  createError: null,
  error: null,
  notFound: false,
  cancelPending: false,
  cancelError: null,
  curves: [] as SweepCurve[],
  cancelledRequested: null,
  newSweepOpen: false,
  origins: {} as Record<string, string>,
};

/** What a change of the selected sweep resets. */
const unselected = {
  detail: null,
  error: null,
  notFound: false,
  cancelPending: false,
  cancelError: null,
  curves: [] as SweepCurve[],
  cancelledRequested: null,
};

let pollTimer: ReturnType<typeof setTimeout> | null = null;
/**
 * Bumped by every stop. A poll, open or cancel that awaited a request re-arms
 * only if no stop happened meanwhile: an aborted read returns quietly, so
 * nothing else would tell it the panel went away.
 */
let pollToken = 0;
/**
 * Whether the (one) Runs panel is mounted to show the sweep. Polling needs
 * it: a create still in flight when the panel went opens its sweep under a
 * fresh token, and nothing on screen would ever stop that poller.
 */
let attached = false;
/** Bumped when the selected sweep changes, so a late reply cannot land on another. */
let generation = 0;
let requestSequence = 0;
let detailController: AbortController | null = null;
let curvesController: AbortController | null = null;
/** A curve load was asked for while one ran; it runs again on the newest detail. */
let curvesStale = false;
let curveCache: SweepCurveCache = new Map();

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function hasActiveChildren(detail: SweepDetail | null): boolean {
  return detail !== null
    && ((detail.counts.running ?? 0) + (detail.counts.queued ?? 0)) > 0;
}

function settled(detail: SweepDetail | null): boolean {
  return detail?.state === 'finished'
    || (detail?.state === 'failed' && !hasActiveChildren(detail));
}

function shouldPoll(state: SweepState): boolean {
  return attached
    && state.selectedSweepId !== null && !state.notFound && !settled(state.detail);
}

function clearPollTimer(): void {
  if (pollTimer !== null) clearTimeout(pollTimer);
  pollTimer = null;
}

function haltPolling(): void {
  pollToken += 1;
  clearPollTimer();
}

function abortReaders(): void {
  detailController?.abort();
  detailController = null;
  curvesController?.abort();
  curvesController = null;
}

function schedulePoll(get: () => SweepState): void {
  clearPollTimer();
  if (!shouldPoll(get())) return;
  const token = pollToken;
  pollTimer = setTimeout(async () => {
    pollTimer = null;
    if (token !== pollToken) return;
    await get().refreshSweep();
    if (token === pollToken) schedulePoll(get);
  }, SWEEP_POLL_MS);
}

export const useSweepStore = create<SweepState>()((set, get) => ({
  ...initialState,

  createSweep: async (request, originTabId) => {
    const createGeneration = generation;
    set({ createState: 'creating', createError: null });
    try {
      const created = await createSweepRequest(request);
      if (originTabId) {
        set((state) => ({ origins: { ...state.origins, [created.sweep_id]: originTabId } }));
      }
      // Opened only if the user has not picked another sweep meanwhile.
      if (generation === createGeneration) await get().openSweep(created.sweep_id);
      return true;
    } catch (error) {
      set({ createError: messageOf(error) });
      return false;
    } finally {
      set({ createState: 'idle' });
    }
  },

  openSweep: async (sweepId) => {
    generation += 1;
    haltPolling();
    abortReaders();
    curveCache = new Map();
    set({ selectedSweepId: sweepId, ...unselected });
    const token = pollToken;
    await get().refreshSweep();
    if (token === pollToken) get().startPolling();
  },

  closeSweep: () => {
    generation += 1;
    haltPolling();
    abortReaders();
    curveCache = new Map();
    set({ selectedSweepId: null, ...unselected });
  },

  refreshSweep: async () => {
    const sweepId = get().selectedSweepId;
    if (!sweepId) return;
    const expectedGeneration = generation;
    const sequence = ++requestSequence;
    detailController?.abort();
    const controller = new AbortController();
    detailController = controller;
    const current = () => !controller.signal.aborted
      && generation === expectedGeneration
      && sequence === requestSequence
      && get().selectedSweepId === sweepId;
    try {
      const detail = await getSweep(sweepId, controller.signal);
      if (!current()) return;
      set({ detail, error: null, notFound: false });
      void get().loadCurves();
      if (settled(detail)) clearPollTimer();
    } catch (error) {
      if (!current()) return;
      // A 404 will not change on its own; anything else may be a restart.
      if (error instanceof ApiError && error.status === 404) {
        clearPollTimer();
        set({ notFound: true, error: messageOf(error) });
      } else {
        set({ error: messageOf(error) });
      }
    } finally {
      if (detailController === controller) detailController = null;
    }
  },

  cancelSweep: async () => {
    const sweepId = get().selectedSweepId;
    if (!sweepId || get().cancelPending) return;
    const expectedGeneration = generation;
    const sameSweep = () => generation === expectedGeneration
      && get().selectedSweepId === sweepId;
    // Taken before the POST, so a panel that went while it was in flight is
    // seen here too.
    const token = pollToken;
    set({ cancelPending: true, cancelError: null });
    try {
      const outcome = await cancelSweepRequest(sweepId);
      if (!sameSweep()) return;
      set((state) => ({
        detail: state.detail
          ? { ...state.detail, state: outcome.state }
          : state.detail,
        cancelledRequested: outcome.cancelled,
      }));
      // Gone meanwhile: the panel re-reads the sweep when it is back.
      if (token !== pollToken) return;
      await get().refreshSweep();
      if (token === pollToken) get().startPolling();
    } catch (error) {
      if (sameSweep()) set({ cancelError: messageOf(error) });
    } finally {
      if (sameSweep()) set({ cancelPending: false });
    }
  },

  resumePolling: async () => {
    attached = true;
    if (!get().selectedSweepId) return;
    const token = pollToken;
    await get().refreshSweep();
    if (token === pollToken) get().startPolling();
  },

  startPolling: () => {
    if (attached && pollTimer === null) schedulePoll(get);
  },

  stopPolling: () => {
    attached = false;
    haltPolling();
    abortReaders();
  },

  loadCurves: async () => {
    // One load at a time. A poll that lands mid-load marks it stale rather
    // than aborting it: at a few hundred ms a request, 32 children take
    // longer than one poll, and restarting meant no curve ever arrived.
    if (curvesController !== null) {
      curvesStale = true;
      return;
    }
    const controller = new AbortController();
    curvesController = controller;
    const expectedGeneration = generation;
    const cache = curveCache;
    try {
      do {
        curvesStale = false;
        const detail = get().detail;
        if (!detail || get().selectedSweepId !== detail.sweep_id) return;
        const curves = await loadObjectiveCurves(
          detail,
          (runId) => getRunMetrics(runId, detail.objective.metric, controller.signal),
          controller.signal,
          cache,
        );
        if (controller.signal.aborted
            || generation !== expectedGeneration
            || get().selectedSweepId !== detail.sweep_id) return;
        set({ curves });
      } while (curvesStale);
    } finally {
      if (curvesController === controller) curvesController = null;
    }
  },

  // A dialog that opens starts clean: the refusal it shows is its own.
  setNewSweepOpen: (open) => set(open ? { newSweepOpen: true, createError: null } : { newSweepOpen: false }),

  reset: () => {
    generation += 1;
    requestSequence += 1;
    haltPolling();
    abortReaders();
    curveCache = new Map();
    curvesStale = false;
    set({ ...initialState, curves: [] });
  },
}));

/** Test isolation for module-level timers and abort controllers. */
export function _resetSweepStoreForTesting(): void {
  generation += 1;
  requestSequence += 1;
  attached = false;
  haltPolling();
  abortReaders();
  curveCache = new Map();
  curvesStale = false;
  useSweepStore.setState({ ...initialState, curves: [] });
}
