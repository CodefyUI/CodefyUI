import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  deleteSshWorker,
  drainWorker,
  fetchSshWorkers,
  fetchWorkerList,
  fetchWorkers,
  removeWorker,
  setSshWorkerEnabled,
  undrainWorker,
  WorkersApiError,
  type SshWorker,
  type WorkerInfo,
} from '../api/workers';
import { rememberAppVersion } from '../utils/appVersion';
import { useToastStore } from './toastStore';
import { WORKER_POLL_MS, _resetWorkerStoreForTesting, useWorkerStore } from './workerStore';

vi.mock('../api/workers', async (importOriginal) => ({
  WorkersApiError: (await importOriginal<typeof import('../api/workers')>()).WorkersApiError,
  fetchWorkers: vi.fn(),
  fetchWorkerList: vi.fn(),
  drainWorker: vi.fn(),
  undrainWorker: vi.fn(),
  removeWorker: vi.fn(),
  fetchSshWorkers: vi.fn(),
  setSshWorkerEnabled: vi.fn(),
  deleteSshWorker: vi.fn(),
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


beforeEach(() => {
  _resetWorkerStoreForTesting();
  useToastStore.setState({ toasts: [] });
  vi.mocked(fetchWorkers).mockReset().mockResolvedValue([worker('w1')]);
  vi.mocked(fetchWorkerList).mockReset().mockImplementation(async () => ({ workers: await fetchWorkers(), rejections: [] }));
  rememberAppVersion('3.0.2');
  vi.mocked(drainWorker).mockReset();
  vi.mocked(undrainWorker).mockReset();
  vi.mocked(removeWorker).mockReset().mockResolvedValue(undefined);
  vi.mocked(fetchSshWorkers).mockReset().mockResolvedValue({
    ssh_workers: [], ssh_available: true, port_range: [18000, 18100], default_remote_cdui: '~/CodefyUI/cdui',
  });
  vi.mocked(setSshWorkerEnabled).mockReset();
  vi.mocked(deleteSshWorker).mockReset().mockResolvedValue(undefined);
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

  it('takes the server version the app last read on every poll, without asking again', async () => {
    rememberAppVersion(null);
    await useWorkerStore.getState().refresh();
    expect(useWorkerStore.getState().serverVersion).toBeNull();
    rememberAppVersion('3.1.0');
    vi.mocked(fetchWorkers).mockRejectedValueOnce(new Error('down'));
    await useWorkerStore.getState().refresh();
    expect(useWorkerStore.getState().serverVersion).toBe('3.1.0');
  });

  it('drops a list read that answers after a newer one', async () => {
    const answers: Array<(value: WorkerInfo[]) => void> = [];
    vi.mocked(fetchWorkers).mockImplementation(
      () => new Promise((resolve) => { answers.push(resolve); }),
    );
    const older = useWorkerStore.getState().refresh();
    const newer = useWorkerStore.getState().refresh();
    answers[1]([worker('new')]);
    await newer;
    answers[0]([worker('old')]);
    await older;
    expect(useWorkerStore.getState().workers.map((w) => w.id)).toEqual(['new']);
  });

  it('drops a failed list read that answers after a newer one', async () => {
    let fail: (err: Error) => void = () => {};
    vi.mocked(fetchWorkers).mockImplementationOnce(
      () => new Promise((_resolve, reject) => { fail = reject; }),
    );
    const older = useWorkerStore.getState().refresh();
    await useWorkerStore.getState().refresh();
    fail(new Error('late'));
    await older;
    expect(useWorkerStore.getState().error).toBeNull();
  });

  it('drops a list read sent before a drain that answers after it', async () => {
    useWorkerStore.setState({ workers: [worker('w1')] });
    let answer: (value: WorkerInfo[]) => void = () => {};
    vi.mocked(fetchWorkers).mockImplementationOnce(
      () => new Promise((resolve) => { answer = resolve; }),
    );
    const read = useWorkerStore.getState().refresh();
    vi.mocked(drainWorker).mockResolvedValueOnce(worker('w1', { draining: true, status: 'draining' }));
    await useWorkerStore.getState().setDraining('w1', true);
    answer([worker('w1')]);
    await read;
    expect(useWorkerStore.getState().workers[0].status).toBe('draining');
  });

  it('drops a drain answer older than a list already applied', async () => {
    useWorkerStore.setState({ workers: [worker('w1')] });
    let answer: (value: WorkerInfo) => void = () => {};
    vi.mocked(drainWorker).mockImplementationOnce(
      () => new Promise((resolve) => { answer = resolve; }),
    );
    const drain = useWorkerStore.getState().setDraining('w1', true);
    vi.mocked(fetchWorkers).mockResolvedValueOnce([worker('w1', { status: 'degraded' })]);
    await useWorkerStore.getState().refresh();
    answer(worker('w1', { draining: true, status: 'draining' }));
    await drain;
    expect(useWorkerStore.getState().workers[0].status).toBe('degraded');
    expect(useWorkerStore.getState().busy).toEqual({});
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

  it('toasts a failed drain and undrain under the worker\'s name', async () => {
    useWorkerStore.setState({ workers: [worker('w1', { name: 'GPU box' })] });
    vi.mocked(drainWorker).mockRejectedValueOnce(new Error('gone'));
    await useWorkerStore.getState().setDraining('w1', true);
    vi.mocked(undrainWorker).mockRejectedValueOnce(new Error('gone'));
    await useWorkerStore.getState().setDraining('w2', false);
    const messages = useToastStore.getState().toasts.map((t) => t.message);
    expect(messages).toEqual(['Could not drain GPU box: gone', 'Could not undrain w2: gone']);
    expect(useWorkerStore.getState().busy).toEqual({});
  });

  it('removes a worker from the list once the server revoked it', async () => {
    useWorkerStore.setState({ workers: [worker('w1'), worker('w2')] });
    await useWorkerStore.getState().remove('w1');
    expect(removeWorker).toHaveBeenCalledWith('w1');
    expect(useWorkerStore.getState().workers.map((w) => w.id)).toEqual(['w2']);
    expect(useWorkerStore.getState().busy).toEqual({});
  });

  it('ignores a remove while another action on the worker is in flight', async () => {
    useWorkerStore.setState({ busy: { w1: true } });
    await useWorkerStore.getState().remove('w1');
    expect(removeWorker).not.toHaveBeenCalled();
  });

  it('keeps a list read newer than the remove', async () => {
    useWorkerStore.setState({ workers: [worker('w1')] });
    let answer: () => void = () => {};
    vi.mocked(removeWorker).mockImplementationOnce(() => new Promise<void>((resolve) => { answer = resolve; }));
    const removing = useWorkerStore.getState().remove('w1');
    vi.mocked(fetchWorkers).mockResolvedValueOnce([worker('w1'), worker('w3')]);
    await useWorkerStore.getState().refresh();
    answer();
    await removing;
    expect(useWorkerStore.getState().workers.map((w) => w.id)).toEqual(['w1', 'w3']);
  });

  it('toasts a failed remove under the worker\'s name', async () => {
    useWorkerStore.setState({ workers: [worker('w1', { name: 'GPU box' })] });
    vi.mocked(removeWorker).mockRejectedValueOnce(new Error('gone'));
    await useWorkerStore.getState().remove('w1');
    vi.mocked(removeWorker).mockRejectedValueOnce('plain');
    await useWorkerStore.getState().remove('w9');
    const messages = useToastStore.getState().toasts.map((t) => t.message);
    expect(messages).toEqual(['Could not remove GPU box: gone', 'Could not remove w9: plain']);
    expect(useWorkerStore.getState().workers).toHaveLength(1);
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

  // ── SSH workers (#784) ──

  it('reads the SSH workers with the list; a 404 means none, another failure keeps the list', async () => {
    vi.mocked(fetchSshWorkers).mockResolvedValueOnce({ ssh_workers: [sshEntry('w1')], ssh_available: true, port_range: [18000, 18100], default_remote_cdui: '~/CodefyUI/cdui' });
    await useWorkerStore.getState().refresh();
    expect(useWorkerStore.getState().ssh.map((e) => e.id)).toEqual(['w1']);
    vi.mocked(fetchSshWorkers).mockRejectedValueOnce(new WorkersApiError('Missing or invalid token', 403));
    await useWorkerStore.getState().refresh();
    expect(useWorkerStore.getState()).toMatchObject({ error: null, sshError: 'Missing or invalid token' });
    expect(useWorkerStore.getState().ssh.map((e) => e.id)).toEqual(['w1']);
    vi.mocked(fetchSshWorkers).mockRejectedValueOnce(new Error('offline'));
    await useWorkerStore.getState().refresh();
    expect(useWorkerStore.getState()).toMatchObject({ sshError: 'offline' });
    vi.mocked(fetchSshWorkers).mockRejectedValueOnce(new WorkersApiError('Not Found', 404));
    await useWorkerStore.getState().refresh();
    expect(useWorkerStore.getState()).toMatchObject({ ssh: [], sshError: null, error: null });
  });

  it('enables and disables an SSH worker, replacing it with the answer', async () => {
    useWorkerStore.setState({ ssh: [sshEntry('s1'), sshEntry('s2')] });
    const stopped = sshEntry('s1', { enabled: false, state: 'stopped' });
    vi.mocked(setSshWorkerEnabled).mockResolvedValueOnce(stopped);
    vi.mocked(fetchSshWorkers).mockResolvedValueOnce({ ssh_workers: [stopped, sshEntry('s2')], ssh_available: true, port_range: [18000, 18100], default_remote_cdui: '~/CodefyUI/cdui' });
    const reads = vi.mocked(fetchWorkers).mock.calls.length;
    await useWorkerStore.getState().setSshEnabled('s1', false);
    expect(setSshWorkerEnabled).toHaveBeenCalledWith('s1', false);
    // The list is read again at once: the worker's own status moved too.
    expect(vi.mocked(fetchWorkers).mock.calls.length).toBe(reads + 1);
    expect(useWorkerStore.getState().ssh.map((e) => e.state)).toEqual(['stopped', 'up']);
    expect(useWorkerStore.getState().busy).toEqual({});
  });

  it('drops an SSH answer older than a list already applied', async () => {
    useWorkerStore.setState({ ssh: [sshEntry('s1')] });
    let answer: (entry: SshWorker) => void = () => {};
    vi.mocked(setSshWorkerEnabled).mockImplementationOnce(() => new Promise((resolve) => { answer = resolve; }));
    const pending = useWorkerStore.getState().setSshEnabled('s1', false);
    vi.mocked(fetchSshWorkers).mockResolvedValue({
      ssh_workers: [sshEntry('s1', { state: 'connecting' })], ssh_available: true, port_range: [1, 2], default_remote_cdui: 'x',
    });
    await useWorkerStore.getState().refresh();
    answer(sshEntry('s1', { state: 'stopped' }));
    await pending;
    expect(useWorkerStore.getState().ssh[0].state).toBe('connecting');
  });

  it('keeps a list read newer than an SSH delete', async () => {
    useWorkerStore.setState({ ssh: [sshEntry('s1')] });
    let answer: () => void = () => {};
    vi.mocked(deleteSshWorker).mockImplementationOnce(() => new Promise<void>((resolve) => { answer = resolve; }));
    const deleting = useWorkerStore.getState().deleteSsh('s1');
    vi.mocked(fetchSshWorkers).mockResolvedValueOnce({
      ssh_workers: [sshEntry('s1'), sshEntry('s3')], ssh_available: true, port_range: [1, 2], default_remote_cdui: 'x',
    });
    await useWorkerStore.getState().refresh();
    answer();
    await deleting;
    expect(useWorkerStore.getState().ssh.map((e) => e.id)).toEqual(['s1', 's3']);
  });

  it('deletes an SSH worker, and a remove drops its SSH entry too', async () => {
    useWorkerStore.setState({ workers: [worker('w1')], ssh: [sshEntry('w1'), sshEntry('s2')] });
    await useWorkerStore.getState().deleteSsh('s2');
    expect(deleteSshWorker).toHaveBeenCalledWith('s2');
    expect(useWorkerStore.getState().ssh.map((e) => e.id)).toEqual(['w1']);
    await useWorkerStore.getState().remove('w1');
    expect(useWorkerStore.getState()).toMatchObject({ workers: [], ssh: [] });
  });

  it('ignores an SSH action while one is in flight, and toasts a failure', async () => {
    useWorkerStore.setState({ busy: { s1: true }, ssh: [sshEntry('s1')] });
    await useWorkerStore.getState().setSshEnabled('s1', true);
    await useWorkerStore.getState().deleteSsh('s1');
    expect(setSshWorkerEnabled).not.toHaveBeenCalled();
    expect(deleteSshWorker).not.toHaveBeenCalled();

    useWorkerStore.setState({ busy: {}, workers: [worker('w1', { name: 'GPU box' })] });
    vi.mocked(setSshWorkerEnabled).mockRejectedValueOnce(new Error('nope'));
    await useWorkerStore.getState().setSshEnabled('s1', true);
    vi.mocked(setSshWorkerEnabled).mockRejectedValueOnce(new Error('nope'));
    await useWorkerStore.getState().setSshEnabled('w1', false);
    vi.mocked(deleteSshWorker).mockRejectedValueOnce('plain');
    await useWorkerStore.getState().deleteSsh('zz');
    expect(useToastStore.getState().toasts.map((t) => t.message)).toEqual([
      'Could not enable s1: nope',
      'Could not disable GPU box: nope',
      'Could not delete zz: plain',
    ]);
    expect(useWorkerStore.getState().ssh).toHaveLength(1);
  });
});

function sshEntry(id: string, extra: Partial<SshWorker> = {}): SshWorker {
  return {
    id,
    target: `alice@${id}`,
    port: 22,
    identity: null,
    remote_cdui: '~/CodefyUI/cdui',
    remote_port: 18000,
    enabled: true,
    state: 'up',
    last_error: null,
    exit_code: null,
    tail: [],
    attempts: 1,
    retry_at: null,
    up_since: 1,
    ...extra,
  };
}
