import { create } from 'zustand';
import {
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
} from '../components/ResultsPanel/sweepCurves';

export const SWEEP_POLL_MS = 2000;

export type SweepCreateState = 'idle' | 'creating';

interface SweepState {
  selectedSweepId: string | null;
  detail: SweepDetail | null;
  createState: SweepCreateState;
  error: string | null;
  curves: SweepCurve[];
  /** Number acknowledged by the last cancel request, for truthful UI copy. */
  cancelledRequested: number | null;

  createSweep: (request: CreateSweepRequest) => Promise<void>;
  openSweep: (sweepId: string) => Promise<void>;
  closeSweep: () => void;
  refreshSweep: () => Promise<void>;
  cancelSweep: () => Promise<void>;
  startPolling: () => void;
  stopPolling: () => void;
  loadCurves: () => Promise<void>;
  reset: () => void;
}

const initialState = {
  selectedSweepId: null,
  detail: null,
  createState: 'idle' as SweepCreateState,
  error: null,
  curves: [] as SweepCurve[],
  cancelledRequested: null,
};

let pollTimer: ReturnType<typeof setTimeout> | null = null;
let generation = 0;
let requestSequence = 0;
let detailController: AbortController | null = null;
let curvesController: AbortController | null = null;

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

function clearPollTimer(): void {
  if (pollTimer !== null) clearTimeout(pollTimer);
  pollTimer = null;
}

function abortReaders(): void {
  detailController?.abort();
  detailController = null;
  curvesController?.abort();
  curvesController = null;
}

function schedulePoll(get: () => SweepState): void {
  clearPollTimer();
  const expectedGeneration = generation;
  if (!get().selectedSweepId || settled(get().detail)) return;
  pollTimer = setTimeout(async () => {
    pollTimer = null;
    if (generation !== expectedGeneration) return;
    await get().refreshSweep();
    if (generation === expectedGeneration
        && get().selectedSweepId
        && !settled(get().detail)) {
      schedulePoll(get);
    }
  }, SWEEP_POLL_MS);
}

export const useSweepStore = create<SweepState>()((set, get) => ({
  ...initialState,

  createSweep: async (request) => {
    const createGeneration = generation;
    set({ createState: 'creating', error: null });
    try {
      const created = await createSweepRequest(request);
      if (generation !== createGeneration) return;
      set({ createState: 'idle' });
      await get().openSweep(created.sweep_id);
    } catch (error) {
      if (generation === createGeneration) {
        set({ createState: 'idle', error: messageOf(error) });
      }
    }
  },

  openSweep: async (sweepId) => {
    generation += 1;
    clearPollTimer();
    abortReaders();
    set({
      selectedSweepId: sweepId,
      detail: null,
      error: null,
      curves: [],
      cancelledRequested: null,
    });
    await get().refreshSweep();
    get().startPolling();
  },

  closeSweep: () => {
    generation += 1;
    clearPollTimer();
    abortReaders();
    set({
      selectedSweepId: null,
      detail: null,
      error: null,
      curves: [],
      cancelledRequested: null,
    });
  },

  refreshSweep: async () => {
    const sweepId = get().selectedSweepId;
    if (!sweepId) return;
    const expectedGeneration = generation;
    const sequence = ++requestSequence;
    detailController?.abort();
    const controller = new AbortController();
    detailController = controller;
    try {
      const detail = await getSweep(sweepId, controller.signal);
      if (controller.signal.aborted
          || generation !== expectedGeneration
          || sequence !== requestSequence
          || get().selectedSweepId !== sweepId) return;
      set({ detail, error: null });
      void get().loadCurves();
      if (settled(detail)) clearPollTimer();
    } catch (error) {
      if (!controller.signal.aborted
          && generation === expectedGeneration
          && sequence === requestSequence
          && get().selectedSweepId === sweepId) {
        set({ error: messageOf(error) });
      }
    } finally {
      if (detailController === controller) detailController = null;
    }
  },

  cancelSweep: async () => {
    const sweepId = get().selectedSweepId;
    if (!sweepId) return;
    const expectedGeneration = generation;
    try {
      const outcome = await cancelSweepRequest(sweepId);
      if (generation !== expectedGeneration || get().selectedSweepId !== sweepId) return;
      set((state) => ({
        detail: state.detail
          ? { ...state.detail, state: outcome.state }
          : state.detail,
        cancelledRequested: outcome.cancelled,
        error: null,
      }));
      await get().refreshSweep();
      get().startPolling();
    } catch (error) {
      if (generation === expectedGeneration && get().selectedSweepId === sweepId) {
        set({ error: messageOf(error) });
      }
    }
  },

  startPolling: () => {
    if (pollTimer === null) schedulePoll(get);
  },

  stopPolling: () => {
    clearPollTimer();
    detailController?.abort();
    detailController = null;
    curvesController?.abort();
    curvesController = null;
  },

  loadCurves: async () => {
    const detail = get().detail;
    if (!detail || get().selectedSweepId !== detail.sweep_id) return;
    const expectedGeneration = generation;
    curvesController?.abort();
    const controller = new AbortController();
    curvesController = controller;
    const curves = await loadObjectiveCurves(
      detail,
      (runId) => getRunMetrics(runId, detail.objective.metric, controller.signal),
      controller.signal,
    );
    if (!controller.signal.aborted
        && generation === expectedGeneration
        && get().selectedSweepId === detail.sweep_id) {
      set({ curves });
    }
    if (curvesController === controller) curvesController = null;
  },

  reset: () => {
    generation += 1;
    requestSequence += 1;
    clearPollTimer();
    abortReaders();
    set({ ...initialState, curves: [] });
  },
}));

/** Test isolation for module-level timers and abort controllers. */
export function _resetSweepStoreForTesting(): void {
  generation += 1;
  requestSequence += 1;
  clearPollTimer();
  abortReaders();
  useSweepStore.setState({ ...initialState, curves: [] });
}
