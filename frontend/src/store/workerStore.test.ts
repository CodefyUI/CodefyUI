import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { drainWorker, fetchWorkers, undrainWorker, type WorkerInfo } from '../api/workers';
import { fetchHealth } from '../api/rest';
import { useToastStore } from './toastStore';
import { WORKER_POLL_MS, _resetWorkerStoreForTesting, useWorkerStore } from './workerStore';

vi.mock('../api/workers', () => ({
  fetchWorkers: vi.fn(),
  drainWorker: vi.fn(),
  undrainWorker: vi.fn(),
}));

vi.mock('../api/rest', () => ({
  fetchHealth: vi.fn(),
}));

function worker(id: string, extra: Partial<WorkerInfo> = {}): WorkerInfo {
  return {
    id,
    name: id,
    status: 'online',
    draining: false,
    version: '3.0.2',
    fingerprint_digest: null,
    platform: {},
    mounts: {},
    devices: [],
    running: [],
    connected_at: 0,
    last_heartbeat_at: 0,
    ...extra,
  };
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

beforeEach(() => {
  _resetWorkerStoreForTesting();
  useToastStore.setState({ toasts: [] });
  vi.mocked(fetchWorkers).mockReset().mockResolvedValue([worker('w1')]);
  vi.mocked(fetchHealth).mockReset().mockResolvedValue({ version: '3.0.2' } as never);
  vi.mocked(drainWorker).mockReset();
  vi.mocked(undrainWorker).mockReset();
});

afterEach(() => {
  vi.useRealTimers();
  _resetWorkerStoreForTesting();
});

describe('workerStore', () => {
  it('opens and closes the panel', () => {
    useWorkerStore.getState().openPanel();
    expect(useWorkerStore.getState().panelOpen).toBe(true);
    useWorkerStore.getState().closePanel();
    expect(useWorkerStore.getState().panelOpen).toBe(false);
  });

  it('refresh stores the list, and the error when the read fails', async () => {
    await useWorkerStore.getState().refresh();
    expect(useWorkerStore.getState()).toMatchObject({ loaded: true, error: null });
    expect(useWorkerStore.getState().workers.map((w) => w.id)).toEqual(['w1']);

    vi.mocked(fetchWorkers).mockRejectedValueOnce(new Error('down'));
    await useWorkerStore.getState().refresh();
    expect(useWorkerStore.getState().error).toBe('down');
    // The last list stays on screen.
    expect(useWorkerStore.getState().workers).toHaveLength(1);

    vi.mocked(fetchWorkers).mockRejectedValueOnce('plain');
    await useWorkerStore.getState().refresh();
    expect(useWorkerStore.getState().error).toBe('plain');
  });

  it('polls every 5 s while watched, once for several watchers, and stops on the last release', async () => {
    vi.useFakeTimers();
    const releaseA = useWorkerStore.getState().watch();
    const releaseB = useWorkerStore.getState().watch();
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchWorkers).toHaveBeenCalledTimes(1);
    expect(fetchHealth).toHaveBeenCalledTimes(1);
    expect(useWorkerStore.getState().serverVersion).toBe('3.0.2');

    await vi.advanceTimersByTimeAsync(WORKER_POLL_MS);
    expect(fetchWorkers).toHaveBeenCalledTimes(2);

    releaseA();
    releaseA(); // a second call is a no-op
    await vi.advanceTimersByTimeAsync(WORKER_POLL_MS);
    expect(fetchWorkers).toHaveBeenCalledTimes(3);

    releaseB();
    await vi.advanceTimersByTimeAsync(WORKER_POLL_MS * 3);
    expect(fetchWorkers).toHaveBeenCalledTimes(3);
  });

  it('does not reread the server version once it is known', async () => {
    useWorkerStore.setState({ serverVersion: '3.0.2' });
    const release = useWorkerStore.getState().watch();
    await flush();
    expect(fetchHealth).not.toHaveBeenCalled();
    release();
  });

  it('leaves the version unknown when health fails', async () => {
    vi.mocked(fetchHealth).mockRejectedValueOnce(new Error('down'));
    const release = useWorkerStore.getState().watch();
    await flush();
    expect(useWorkerStore.getState().serverVersion).toBeNull();
    release();
  });

  it('stops after a read that answers once the last watcher left', async () => {
    let answer: (value: WorkerInfo[]) => void = () => {};
    vi.mocked(fetchWorkers).mockImplementationOnce(
      () => new Promise((resolve) => { answer = resolve; }),
    );
    vi.useFakeTimers();
    const release = useWorkerStore.getState().watch();
    release();
    answer([]);
    await vi.advanceTimersByTimeAsync(WORKER_POLL_MS * 2);
    expect(fetchWorkers).toHaveBeenCalledTimes(1);
  });

  it('keeps one poll when a re-watch overlaps a read still in flight', async () => {
    let answer: (value: WorkerInfo[]) => void = () => {};
    vi.mocked(fetchWorkers).mockImplementationOnce(
      () => new Promise((resolve) => { answer = resolve; }),
    );
    vi.useFakeTimers();
    const first = useWorkerStore.getState().watch();
    first();
    const second = useWorkerStore.getState().watch();
    // The second watch's read answers and schedules the next tick.
    await vi.advanceTimersByTimeAsync(0);
    // Then the first one's: a timer is already set, so it adds none.
    answer([]);
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchWorkers).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(WORKER_POLL_MS);
    expect(fetchWorkers).toHaveBeenCalledTimes(3);
    second();
  });

  it('drains and undrains, replacing the worker with the answer', async () => {
    useWorkerStore.setState({ workers: [worker('w1'), worker('w2')] });
    vi.mocked(drainWorker).mockResolvedValueOnce(worker('w1', { draining: true, status: 'draining' }));
    await useWorkerStore.getState().setDraining('w1', true);
    expect(drainWorker).toHaveBeenCalledWith('w1');
    expect(useWorkerStore.getState().workers[0].status).toBe('draining');
    expect(useWorkerStore.getState().workers[1].status).toBe('online');
    expect(useWorkerStore.getState().busy).toEqual({});

    vi.mocked(undrainWorker).mockResolvedValueOnce(worker('w1'));
    await useWorkerStore.getState().setDraining('w1', false);
    expect(undrainWorker).toHaveBeenCalledWith('w1');
    expect(useWorkerStore.getState().workers[0].status).toBe('online');
  });

  it('ignores a second press while one is in flight', async () => {
    useWorkerStore.setState({ busy: { w1: true } });
    await useWorkerStore.getState().setDraining('w1', true);
    expect(drainWorker).not.toHaveBeenCalled();
  });

  it('toasts a failed drain and undrain', async () => {
    vi.mocked(drainWorker).mockRejectedValueOnce(new Error('gone'));
    await useWorkerStore.getState().setDraining('w1', true);
    vi.mocked(undrainWorker).mockRejectedValueOnce(new Error('gone'));
    await useWorkerStore.getState().setDraining('w1', false);
    const messages = useToastStore.getState().toasts.map((t) => t.message);
    expect(messages).toEqual(['Could not drain w1: gone', 'Could not undrain w1: gone']);
    expect(useWorkerStore.getState().busy).toEqual({});
  });

  it('revealRuns closes the panel and bumps the reveal counter', () => {
    useWorkerStore.setState({ panelOpen: true });
    useWorkerStore.getState().revealRuns();
    expect(useWorkerStore.getState()).toMatchObject({ panelOpen: false, runsRevealSeq: 1 });
  });

  it('the test reset clears a pending poll', async () => {
    vi.useFakeTimers();
    useWorkerStore.getState().watch();
    await vi.advanceTimersByTimeAsync(0);
    _resetWorkerStoreForTesting();
    await vi.advanceTimersByTimeAsync(WORKER_POLL_MS * 2);
    expect(fetchWorkers).toHaveBeenCalledTimes(1);
  });
});
