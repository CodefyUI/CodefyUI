import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as rest from '../api/rest';
import { ApiError } from '../api/rest';
import type {
  CancelSweepResponse,
  CreateSweepRequest,
  CreateSweepResponse,
  RunMetrics,
  SweepDetail,
  SweepState,
} from '../api/rest';
import {
  SWEEP_POLL_MS,
  _resetSweepStoreForTesting,
  useSweepStore,
} from './sweepStore';

vi.mock('../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof rest>();
  return {
    ...actual,
    createSweep: vi.fn(),
    getSweep: vi.fn(),
    cancelSweep: vi.fn(),
    getRunMetrics: vi.fn(),
  };
});

const api = vi.mocked(rest);

const request: CreateSweepRequest = {
  base_graph: { nodes: [], edges: [] },
  sweep_spec: {
    method: 'grid', seed: null, samples: null,
    params: [{ node_id: 'n', param: 'x', values: [1, 2] }],
  },
  objective: { metric: 'loss', direction: 'minimize' },
  options: null,
  name: 'Search',
  seed_variants: false,
};

function detail(state: SweepState = 'running', id = 's1'): SweepDetail {
  return {
    sweep_id: id, name: 'Search', state, method: 'grid', seed: null,
    seed_variants: false,
    objective: { metric: 'loss', direction: 'minimize' },
    created_at: '2026-10-01T00:00:00Z', finished_at: null, error: null,
    counts: { queued: 1, running: 1, succeeded: 0, failed: 0, cancelled: 0, interrupted: 0, missing: 0 },
    params: [], variants: [], best: null,
  };
}

function created(): CreateSweepResponse {
  return {
    sweep_id: 's1', state: 'running', method: 'grid', seed: null,
    seed_variants: false,
    objective: { metric: 'loss', direction: 'minimize' },
    total_combinations: 2, params: [], variants: [],
  };
}

/** A sweep whose children have the given statuses, one run each. */
function withChildren(statuses: Array<'running' | 'succeeded' | 'queued'>): SweepDetail {
  const sweep = detail('running');
  sweep.variants = statuses.map((status, index) => ({
    index, domain_index: index, run_id: `r${index}`, status, params: [],
    seed: null, objective: null, rank: null, run_exists: true,
  }));
  return sweep;
}

function series(runId: string): RunMetrics {
  return {
    run_id: runId, names: ['loss'],
    metrics: [{ node_id: 'n', name: 'loss', step: 1, value: 1 }],
  };
}

function cancelled(count: number): CancelSweepResponse {
  return { sweep_id: 's1', state: 'cancelling', cancelled: count, already_finished: 0, variants: [] };
}

/** A request that only ends when its reader gives up on it. */
function hangUntilAborted<T>(signal?: AbortSignal): Promise<T> {
  return new Promise((_resolve, reject) => {
    signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
  });
}

beforeEach(() => {
  _resetSweepStoreForTesting();
  // As the Runs panel does on mount: polling runs only while a panel shows it.
  void useSweepStore.getState().resumePolling();
  api.createSweep.mockResolvedValue(created());
  api.getSweep.mockResolvedValue(detail());
  api.cancelSweep.mockResolvedValue({
    sweep_id: 's1', state: 'cancelling', cancelled: 2,
    already_finished: 0, variants: [],
  });
  api.getRunMetrics.mockResolvedValue({ run_id: 'r1', names: [], metrics: [] });
});

afterEach(() => {
  _resetSweepStoreForTesting();
  vi.useRealTimers();
  vi.resetAllMocks();
});

describe('sweepStore', () => {
  it('creates a sweep and opens the server detail', async () => {
    await expect(useSweepStore.getState().createSweep(request)).resolves.toBe(true);
    expect(api.createSweep).toHaveBeenCalledWith(request);
    expect(api.getSweep).toHaveBeenCalledWith('s1', expect.any(AbortSignal));
    expect(useSweepStore.getState()).toMatchObject({
      selectedSweepId: 's1', detail: { sweep_id: 's1' }, createState: 'idle',
      createError: null, error: null,
    });
  });

  it('keeps a refused create to the dialog and reports it as not created', async () => {
    api.createSweep.mockRejectedValueOnce(new Error('server cap is 1'));
    await expect(useSweepStore.getState().createSweep(request)).resolves.toBe(false);
    expect(useSweepStore.getState()).toMatchObject({
      createState: 'idle', createError: 'server cap is 1', error: null, selectedSweepId: null,
    });
  });

  it('returns to idle when another sweep was opened while the create was in flight', async () => {
    // The dialog can be closed mid-request and a child's parent sweep opened
    // from the Runs list; the late reply must neither hijack that view nor
    // leave the next dialog stuck on "Starting".
    let release: (value: CreateSweepResponse) => void = () => {};
    api.createSweep.mockImplementationOnce(() => new Promise((resolve) => { release = resolve; }));
    api.getSweep.mockResolvedValue(detail('running', 'other'));

    const creating = useSweepStore.getState().createSweep(request);
    expect(useSweepStore.getState().createState).toBe('creating');
    await useSweepStore.getState().openSweep('other');
    release(created());

    await expect(creating).resolves.toBe(true);
    expect(useSweepStore.getState()).toMatchObject({ createState: 'idle', selectedSweepId: 'other' });
  });

  it('stops a poll that is already in flight when polling stops', async () => {
    vi.useFakeTimers();
    await useSweepStore.getState().openSweep('s1');
    api.getSweep.mockImplementationOnce((_id, signal) => hangUntilAborted(signal));
    await vi.advanceTimersByTimeAsync(SWEEP_POLL_MS);
    expect(api.getSweep).toHaveBeenCalledTimes(2);

    useSweepStore.getState().stopPolling();
    await vi.advanceTimersByTimeAsync(SWEEP_POLL_MS * 5);
    expect(api.getSweep).toHaveBeenCalledTimes(2);
  });

  it('does not start polling when the panel detaches during the opening read', async () => {
    vi.useFakeTimers();
    let release: (value: SweepDetail) => void = () => {};
    api.getSweep.mockImplementationOnce(() => new Promise((resolve) => { release = resolve; }));

    const opening = useSweepStore.getState().openSweep('s1');
    useSweepStore.getState().stopPolling();
    release(detail('running'));
    await opening;
    await vi.advanceTimersByTimeAsync(SWEEP_POLL_MS * 3);

    expect(api.getSweep).toHaveBeenCalledTimes(1);
  });

  it('does not poll after a Stop that was still in flight when the panel went', async () => {
    vi.useFakeTimers();
    await useSweepStore.getState().openSweep('s1');
    let release: (value: CancelSweepResponse) => void = () => {};
    api.cancelSweep.mockImplementationOnce(() => new Promise((resolve) => { release = resolve; }));

    const cancelling = useSweepStore.getState().cancelSweep();
    useSweepStore.getState().stopPolling();
    const before = api.getSweep.mock.calls.length;
    release(cancelled(2));
    await cancelling;
    await vi.advanceTimersByTimeAsync(SWEEP_POLL_MS * 5);

    // The acknowledgement is kept for when the panel returns; nothing is read.
    expect(useSweepStore.getState().cancelledRequested).toBe(2);
    expect(api.getSweep.mock.calls.length - before).toBe(0);
  });

  it('does not poll a sweep that a create opened after the panel went', async () => {
    vi.useFakeTimers();
    let release: (value: CreateSweepResponse) => void = () => {};
    api.createSweep.mockImplementationOnce(() => new Promise((resolve) => { release = resolve; }));

    const creating = useSweepStore.getState().createSweep(request);
    useSweepStore.getState().stopPolling();
    release(created());
    await creating;
    const before = api.getSweep.mock.calls.length;
    await vi.advanceTimersByTimeAsync(SWEEP_POLL_MS * 5);
    expect(useSweepStore.getState().selectedSweepId).toBe('s1');
    expect(api.getSweep.mock.calls.length - before).toBe(0);

    // Back on screen, the panel reads it and polls it.
    await useSweepStore.getState().resumePolling();
    await vi.advanceTimersByTimeAsync(SWEEP_POLL_MS);
    expect(api.getSweep.mock.calls.length - before).toBe(2);
  });

  it('remembers which tab a sweep was created from', async () => {
    await useSweepStore.getState().createSweep(request, 'tab-1');
    expect(useSweepStore.getState().origins).toEqual({ s1: 'tab-1' });
  });

  it('re-reads the selected sweep and polls it again when the panel comes back', async () => {
    vi.useFakeTimers();
    await useSweepStore.getState().openSweep('s1');
    useSweepStore.getState().stopPolling();
    await vi.advanceTimersByTimeAsync(SWEEP_POLL_MS * 3);
    expect(api.getSweep).toHaveBeenCalledTimes(1);

    api.getSweep.mockResolvedValue(detail('cancelling'));
    await useSweepStore.getState().resumePolling();
    expect(api.getSweep).toHaveBeenCalledTimes(2);
    expect(useSweepStore.getState().detail?.state).toBe('cancelling');
    await vi.advanceTimersByTimeAsync(SWEEP_POLL_MS);
    expect(api.getSweep).toHaveBeenCalledTimes(3);
  });

  it('stops polling a sweep the server no longer has, and says so', async () => {
    vi.useFakeTimers();
    api.getSweep.mockRejectedValue(new ApiError(404, "sweep 'gone' not found"));
    await useSweepStore.getState().openSweep('gone');
    await vi.advanceTimersByTimeAsync(SWEEP_POLL_MS * 10);

    expect(api.getSweep).toHaveBeenCalledTimes(1);
    expect(useSweepStore.getState()).toMatchObject({ notFound: true, detail: null });
  });

  it('keeps polling through a read that failed for any other reason', async () => {
    vi.useFakeTimers();
    api.getSweep.mockRejectedValueOnce(new ApiError(503, 'run service not initialised'));
    await useSweepStore.getState().openSweep('s1');
    expect(useSweepStore.getState().error).toBe('run service not initialised');
    await vi.advanceTimersByTimeAsync(SWEEP_POLL_MS);

    expect(api.getSweep).toHaveBeenCalledTimes(2);
    expect(useSweepStore.getState()).toMatchObject({ error: null, notFound: false, detail: { state: 'running' } });
  });

  it('sends one Stop however often it is pressed while the first is in flight', async () => {
    let release: (value: CancelSweepResponse) => void = () => {};
    api.cancelSweep.mockImplementationOnce(() => new Promise((resolve) => { release = resolve; }));
    useSweepStore.setState({ selectedSweepId: 's1', detail: detail('running') });

    const first = useSweepStore.getState().cancelSweep();
    expect(useSweepStore.getState().cancelPending).toBe(true);
    const second = useSweepStore.getState().cancelSweep();
    release(cancelled(2));
    await Promise.all([first, second]);

    expect(api.cancelSweep).toHaveBeenCalledTimes(1);
    expect(useSweepStore.getState()).toMatchObject({ cancelledRequested: 2, cancelPending: false });
  });

  it('keeps a failed Stop on screen through the next poll', async () => {
    vi.useFakeTimers();
    await useSweepStore.getState().openSweep('s1');
    api.cancelSweep.mockRejectedValueOnce(new Error('session token rejected'));
    await useSweepStore.getState().cancelSweep();
    expect(useSweepStore.getState().cancelError).toBe('session token rejected');

    await vi.advanceTimersByTimeAsync(SWEEP_POLL_MS);
    expect(api.getSweep).toHaveBeenCalledTimes(2);
    expect(useSweepStore.getState()).toMatchObject({ cancelError: 'session token rejected', error: null });
  });

  it('reads a finished child once and keeps re-reading the live one', async () => {
    vi.useFakeTimers();
    api.getSweep.mockResolvedValue(withChildren(['succeeded', 'running', 'queued']));
    api.getRunMetrics.mockImplementation(async (runId) => series(runId));

    await useSweepStore.getState().openSweep('s1');
    await vi.advanceTimersByTimeAsync(SWEEP_POLL_MS * 3);

    const reads = (runId: string) => api.getRunMetrics.mock.calls.filter(([id]) => id === runId).length;
    expect(api.getSweep).toHaveBeenCalledTimes(4);
    expect(reads('r0')).toBe(1);
    expect(reads('r1')).toBe(4);
    expect(reads('r2')).toBe(0);
    expect(useSweepStore.getState().curves.map((curve) => curve.runId)).toEqual(['r0', 'r1']);
  });

  it('lets a slow curve load finish instead of restarting it on every poll', async () => {
    // 32 live children at 300 ms a request, four at a time, take 2.4 s: longer
    // than a poll. Aborting the load on each poll meant no curve ever showed.
    vi.useFakeTimers();
    api.getSweep.mockResolvedValue(withChildren(Array.from({ length: 32 }, () => 'running' as const)));
    api.getRunMetrics.mockImplementation((runId, _name, signal) => new Promise((resolve, reject) => {
      const timer = setTimeout(() => resolve(series(runId)), 300);
      signal?.addEventListener('abort', () => {
        clearTimeout(timer);
        reject(new DOMException('aborted', 'AbortError'));
      });
    }));

    await useSweepStore.getState().openSweep('s1');
    await vi.advanceTimersByTimeAsync(SWEEP_POLL_MS * 3);

    expect(useSweepStore.getState().curves).toHaveLength(32);
  });

  it('polls running and cancelling sweeps, then stops on finished', async () => {
    vi.useFakeTimers();
    useSweepStore.setState({ selectedSweepId: 's1', detail: detail('running') });
    api.getSweep
      .mockResolvedValueOnce(detail('cancelling'))
      .mockResolvedValueOnce(detail('finished'));

    useSweepStore.getState().startPolling();
    await vi.advanceTimersByTimeAsync(SWEEP_POLL_MS);
    expect(useSweepStore.getState().detail?.state).toBe('cancelling');
    await vi.advanceTimersByTimeAsync(SWEEP_POLL_MS);
    expect(useSweepStore.getState().detail?.state).toBe('finished');
    await vi.advanceTimersByTimeAsync(SWEEP_POLL_MS * 2);
    expect(api.getSweep).toHaveBeenCalledTimes(2);
  });

  it('reloads objective curves after every accepted live same-state refresh', async () => {
    const withChild = detail('running');
    withChild.variants = [{
      index: 0, domain_index: 0, run_id: 'r1', status: 'running', params: [],
      seed: null, objective: null, rank: null, run_exists: true,
    }];
    api.getSweep
      .mockResolvedValueOnce(withChild)
      .mockResolvedValueOnce({ ...withChild, state: 'cancelling' });
    api.getRunMetrics
      .mockResolvedValueOnce({
        run_id: 'r1', names: ['loss'],
        metrics: [{ node_id: 'n', name: 'loss', step: 1, value: 1 }],
      })
      .mockResolvedValueOnce({
        run_id: 'r1', names: ['loss'],
        metrics: [
          { node_id: 'n', name: 'loss', step: 1, value: 1 },
          { node_id: 'n', name: 'loss', step: 2, value: 0.5 },
        ],
      });
    useSweepStore.setState({ selectedSweepId: 's1', detail: withChild });

    await useSweepStore.getState().refreshSweep();
    await vi.waitFor(() => {
      expect(api.getRunMetrics).toHaveBeenCalledTimes(1);
      expect(useSweepStore.getState().curves[0]?.points).toHaveLength(1);
    });

    await useSweepStore.getState().refreshSweep();
    await vi.waitFor(() => {
      expect(api.getRunMetrics).toHaveBeenCalledTimes(2);
      expect(useSweepStore.getState().curves[0]?.points).toHaveLength(2);
    });
  });

  it('does not let a late request replace a newly selected sweep', async () => {
    let releaseA: (value: SweepDetail) => void = () => {};
    api.getSweep
      .mockImplementationOnce(() => new Promise((resolve) => { releaseA = resolve; }))
      .mockResolvedValueOnce(detail('running', 'b'));

    const openingA = useSweepStore.getState().openSweep('a');
    await useSweepStore.getState().openSweep('b');
    releaseA(detail('finished', 'a'));
    await openingA;

    expect(useSweepStore.getState().selectedSweepId).toBe('b');
    expect(useSweepStore.getState().detail?.sweep_id).toBe('b');
  });

  it('shows a truthful cancelling state before the follow-up refresh settles', async () => {
    useSweepStore.setState({ selectedSweepId: 's1', detail: detail('running') });
    let releaseRefresh: (value: SweepDetail) => void = () => {};
    api.getSweep.mockImplementationOnce(
      () => new Promise((resolve) => { releaseRefresh = resolve; }),
    );

    const cancelling = useSweepStore.getState().cancelSweep();
    await vi.waitFor(() => {
      expect(useSweepStore.getState().detail?.state).toBe('cancelling');
      expect(useSweepStore.getState().cancelledRequested).toBe(2);
    });
    releaseRefresh(detail('finished'));
    await cancelling;
    expect(useSweepStore.getState().detail?.state).toBe('finished');
  });

  it('stopPolling aborts an in-flight curve load without clearing selection', async () => {
    const withChild = detail();
    withChild.variants = [{
      index: 0, domain_index: 0, run_id: 'r1', status: 'running', params: [],
      seed: null, objective: null, rank: null, run_exists: true,
    }];
    useSweepStore.setState({ selectedSweepId: 's1', detail: withChild });
    let seenSignal: AbortSignal | undefined;
    api.getRunMetrics.mockImplementation((_id, _name, signal) => {
      seenSignal = signal;
      return new Promise((_resolve, reject) => {
        signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
      });
    });

    const loading = useSweepStore.getState().loadCurves();
    useSweepStore.getState().stopPolling();
    await loading;

    expect(seenSignal?.aborted).toBe(true);
    expect(useSweepStore.getState().selectedSweepId).toBe('s1');
  });

  it('keeps polling a failed sweep until its surviving children settle', async () => {
    vi.useFakeTimers();
    const failedActive = detail('failed');
    failedActive.counts = { ...failedActive.counts, queued: 1, running: 1 };
    const failedSettled = detail('failed');
    failedSettled.counts = { ...failedSettled.counts, queued: 0, running: 0 };
    useSweepStore.setState({ selectedSweepId: 's1', detail: failedActive });
    api.getSweep.mockResolvedValueOnce(failedSettled);

    useSweepStore.getState().startPolling();
    await vi.advanceTimersByTimeAsync(SWEEP_POLL_MS);
    expect(api.getSweep).toHaveBeenCalledTimes(1);
    expect(useSweepStore.getState().detail?.state).toBe('failed');
    await vi.advanceTimersByTimeAsync(SWEEP_POLL_MS * 2);
    expect(api.getSweep).toHaveBeenCalledTimes(1);
  });

  it('loads the final objective point before polling stops on a same-state failed sweep', async () => {
    vi.useFakeTimers();
    const failedActive = detail('failed');
    failedActive.counts = { ...failedActive.counts, queued: 0, running: 1 };
    failedActive.variants = [{
      index: 0, domain_index: 0, run_id: 'r1', status: 'running', params: [],
      seed: null, objective: null, rank: null, run_exists: true,
    }];
    const failedSettled: SweepDetail = {
      ...failedActive,
      counts: { ...failedActive.counts, running: 0, succeeded: 1 },
      variants: [{ ...failedActive.variants[0], status: 'succeeded', objective: 0.5 }],
    };
    api.getSweep
      .mockResolvedValueOnce(failedActive)
      .mockResolvedValueOnce(failedSettled);
    api.getRunMetrics
      .mockResolvedValueOnce({
        run_id: 'r1', names: ['loss'],
        metrics: [{ node_id: 'n', name: 'loss', step: 1, value: 1 }],
      })
      .mockResolvedValueOnce({
        run_id: 'r1', names: ['loss'],
        metrics: [
          { node_id: 'n', name: 'loss', step: 1, value: 1 },
          { node_id: 'n', name: 'loss', step: 2, value: 0.5 },
        ],
      });

    await useSweepStore.getState().openSweep('s1');
    await vi.waitFor(() => {
      expect(useSweepStore.getState().curves[0]?.points).toEqual([{ x: 1, y: 1 }]);
    });

    await vi.advanceTimersByTimeAsync(SWEEP_POLL_MS);
    expect(useSweepStore.getState().detail).toMatchObject({
      sweep_id: 's1', state: 'failed', counts: { queued: 0, running: 0 },
    });
    await vi.waitFor(() => {
      expect(useSweepStore.getState().curves[0]?.points).toEqual([
        { x: 1, y: 1 }, { x: 2, y: 0.5 },
      ]);
    });
    expect(api.getRunMetrics).toHaveBeenCalledTimes(2);
    expect(api.getRunMetrics).toHaveBeenLastCalledWith('r1', 'loss', expect.any(AbortSignal));

    await vi.advanceTimersByTimeAsync(SWEEP_POLL_MS * 2);
    expect(api.getSweep).toHaveBeenCalledTimes(2);
    expect(api.getRunMetrics).toHaveBeenCalledTimes(2);
  });

  it('reset clears polling and aborts an in-flight curve load', async () => {
    vi.useFakeTimers();
    const withChild = detail();
    withChild.variants = [{
      index: 0, domain_index: 0, run_id: 'r1', status: 'running', params: [],
      seed: null, objective: null, rank: null, run_exists: true,
    }];
    useSweepStore.setState({ selectedSweepId: 's1', detail: withChild });
    let seenSignal: AbortSignal | undefined;
    api.getRunMetrics.mockImplementation((_id, _name, signal) => {
      seenSignal = signal;
      return new Promise((_resolve, reject) => {
        signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
      });
    });

    useSweepStore.getState().startPolling();
    const loading = useSweepStore.getState().loadCurves();
    useSweepStore.getState().reset();
    await loading;
    await vi.advanceTimersByTimeAsync(SWEEP_POLL_MS * 2);

    expect(seenSignal?.aborted).toBe(true);
    expect(api.getSweep).not.toHaveBeenCalled();
    expect(useSweepStore.getState()).toMatchObject({
      selectedSweepId: null, detail: null, curves: [], error: null,
    });
  });
});
