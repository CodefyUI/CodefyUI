import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as rest from '../api/rest';
import type {
  CreateSweepRequest,
  CreateSweepResponse,
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

beforeEach(() => {
  _resetSweepStoreForTesting();
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
    await useSweepStore.getState().createSweep(request);
    expect(api.createSweep).toHaveBeenCalledWith(request);
    expect(api.getSweep).toHaveBeenCalledWith('s1', expect.any(AbortSignal));
    expect(useSweepStore.getState()).toMatchObject({
      selectedSweepId: 's1', detail: { sweep_id: 's1' }, createState: 'idle', error: null,
    });
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
