import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as rest from '../api/rest';
import type { RunEventsPage, RunStatus, RunSummary } from '../api/rest';
import {
  ACTIVE_POLL_MS,
  FOLLOW_IDLE_MS,
  IDLE_POLL_MS,
  LOG_TAIL_EVENTS,
  MAX_LOG_LINES,
  _resetRunStoreForTesting,
  isActiveRun,
  reduceRunEvents,
  seriesNames,
  toChartSeries,
  useRunStore,
  type RunDetail,
} from './runStore';
import { useToastStore } from './toastStore';
import { useI18n } from '../i18n';

vi.mock('../api/rest', async (importOriginal) => {
  // ACTIVE_RUN_STATUSES is a real constant the store folds into a Set at
  // module load — keep the originals and stub only the network calls.
  const actual = await importOriginal<typeof rest>();
  return {
    ...actual,
    listRuns: vi.fn(),
    getRun: vi.fn(),
    getRunEvents: vi.fn(),
    getRunMetrics: vi.fn(),
    getRunArtifacts: vi.fn(),
    cancelRun: vi.fn(),
    deleteRun: vi.fn(),
    downloadRunMetricsCsv: vi.fn(),
  };
});

const api = vi.mocked(rest);

/** The most recent toast. `Array.prototype.at` is outside the project lib. */
function lastToast() {
  const { toasts } = useToastStore.getState();
  return toasts[toasts.length - 1];
}

function makeRun(partial: Partial<RunSummary> & { id: string }): RunSummary {
  return {
    name: null,
    status: 'succeeded' as RunStatus,
    error: null,
    options: {},
    queue_key: 'cpu',
    created_at: '2026-08-01T10:00:00.000000Z',
    started_at: '2026-08-01T10:00:01.000000Z',
    finished_at: '2026-08-01T10:00:09.000000Z',
    git_commit: null,
    git_dirty: null,
    plugin_pins: null,
    queue_position: null,
    final_metrics: {},
    active: false,
    ...partial,
  };
}

function page(runs: RunSummary[], total = runs.length) {
  return { runs, total, limit: 50, offset: 0 };
}

function eventsPage(partial: Partial<RunEventsPage> = {}): RunEventsPage {
  return {
    run_id: 'r1',
    status: 'running',
    active: true,
    events: [],
    cursor: 0,
    ...partial,
  };
}

function emptyDetail(overrides: Partial<RunDetail> = {}): RunDetail {
  return {
    runId: 'r1',
    status: 'running',
    series: {},
    log: [],
    artifacts: [],
    cursor: 0,
    loading: false,
    error: null,
    ...overrides,
  };
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useToastStore.setState({ toasts: [] });
  _resetRunStoreForTesting();
  api.listRuns.mockResolvedValue(page([]));
  api.getRun.mockResolvedValue(null);
  api.getRunEvents.mockResolvedValue(eventsPage({ status: 'succeeded', active: false }));
  api.getRunMetrics.mockResolvedValue({ run_id: 'r1', names: [], metrics: [] });
  api.getRunArtifacts.mockResolvedValue({ run_id: 'r1', artifacts: [] });
  api.cancelRun.mockResolvedValue({ run_id: 'r1', status: 'running', cancelled: true });
  api.deleteRun.mockResolvedValue({ run_id: 'r1', deleted: true });
  api.downloadRunMetricsCsv.mockResolvedValue(undefined);
});

afterEach(() => {
  _resetRunStoreForTesting();
  vi.useRealTimers();
  vi.clearAllMocks();
});

// ── list ──────────────────────────────────────────────────────────────────

describe('runStore — the list', () => {
  it('loads runs and the unpaged total', async () => {
    api.listRuns.mockResolvedValue(page([makeRun({ id: 'a' })], 214));
    await useRunStore.getState().refresh();
    expect(useRunStore.getState().runs.map((r) => r.id)).toEqual(['a']);
    expect(useRunStore.getState().total).toBe(214);
    expect(useRunStore.getState().error).toBeNull();
    expect(useRunStore.getState().loading).toBe(false);
  });

  it('sends no status param for the "all" filter and one for a real status', async () => {
    await useRunStore.getState().refresh();
    expect(api.listRuns.mock.calls[0][0]).toMatchObject({ status: undefined });

    useRunStore.getState().setFilter('failed');
    await vi.waitFor(() =>
      expect(api.listRuns).toHaveBeenCalledWith(expect.objectContaining({ status: ['failed'] })),
    );
    expect(useRunStore.getState().filter).toBe('failed');
  });

  it('ignores a re-selection of the filter already in effect', async () => {
    await useRunStore.getState().refresh();
    api.listRuns.mockClear();
    useRunStore.getState().setFilter('all');
    expect(api.listRuns).not.toHaveBeenCalled();
  });

  it('drops a late page belonging to a filter the user already left', async () => {
    let releaseSlow: (v: ReturnType<typeof page>) => void = () => {};
    api.listRuns.mockImplementationOnce(
      () => new Promise((resolve) => { releaseSlow = resolve; }),
    );
    const slow = useRunStore.getState().refresh();
    // The user switches filters while the first request is still open.
    useRunStore.setState({ filter: 'failed' });
    releaseSlow(page([makeRun({ id: 'stale' })]));
    await slow;
    expect(useRunStore.getState().runs).toEqual([]);
  });

  it('surfaces a list failure without wiping the rows already on screen', async () => {
    api.listRuns.mockResolvedValueOnce(page([makeRun({ id: 'a' })]));
    await useRunStore.getState().refresh();
    api.listRuns.mockRejectedValueOnce(new Error('server down'));
    await useRunStore.getState().refresh();
    expect(useRunStore.getState().error).toBe('server down');
    expect(useRunStore.getState().runs.map((r) => r.id)).toEqual(['a']);
  });
});

// ── polling ───────────────────────────────────────────────────────────────

describe('runStore — polling', () => {
  it('polls once for any number of watchers and stops when the last releases', async () => {
    vi.useFakeTimers();
    const releaseA = useRunStore.getState().watch();
    const releaseB = useRunStore.getState().watch();
    await vi.waitFor(() => expect(api.listRuns).toHaveBeenCalledTimes(1));

    await vi.advanceTimersByTimeAsync(IDLE_POLL_MS);
    expect(api.listRuns).toHaveBeenCalledTimes(2);

    releaseA();
    await vi.advanceTimersByTimeAsync(IDLE_POLL_MS);
    // Still one watcher left, so the loop keeps going.
    expect(api.listRuns).toHaveBeenCalledTimes(3);

    releaseB();
    await vi.advanceTimersByTimeAsync(IDLE_POLL_MS * 3);
    expect(api.listRuns).toHaveBeenCalledTimes(3);
  });

  it('polls faster while something is queued or running', async () => {
    vi.useFakeTimers();
    api.listRuns.mockResolvedValue(page([makeRun({ id: 'a', status: 'running' })]));
    const release = useRunStore.getState().watch();
    await vi.waitFor(() => expect(api.listRuns).toHaveBeenCalledTimes(1));

    await vi.advanceTimersByTimeAsync(ACTIVE_POLL_MS);
    expect(api.listRuns).toHaveBeenCalledTimes(2);
    release();
  });

  it('releasing the same watcher twice does not unbalance the count', async () => {
    vi.useFakeTimers();
    const release = useRunStore.getState().watch();
    const other = useRunStore.getState().watch();
    await vi.waitFor(() => expect(api.listRuns).toHaveBeenCalledTimes(1));
    release();
    release();
    await vi.advanceTimersByTimeAsync(IDLE_POLL_MS);
    // `other` is still watching — a double release must not have cancelled it.
    expect(api.listRuns).toHaveBeenCalledTimes(2);
    other();
  });
});

// ── detail selection ──────────────────────────────────────────────────────

describe('runStore — selecting a run', () => {
  it('seeds the chart from /metrics and the artifacts from /artifacts', async () => {
    api.getRun.mockResolvedValue({ ...makeRun({ id: 'r1' }), last_cursor: 12 });
    api.getRunMetrics.mockResolvedValue({
      run_id: 'r1',
      names: ['train_loss'],
      metrics: [
        { node_id: 'loop', name: 'train_loss', step: 1, value: 2 },
        { node_id: 'loop', name: 'train_loss', step: 2, value: 1 },
        // A diverged point is a gap, not a zero.
        { node_id: 'loop', name: 'train_loss', step: 3, value: null },
      ],
    });
    api.getRunArtifacts.mockResolvedValue({
      run_id: 'r1',
      artifacts: [{ id: 4, kind: 'checkpoint', path: 'runs/r1/e1.pt', meta: null, created_at: 'x' }],
    });

    await useRunStore.getState().select('r1');
    const detail = useRunStore.getState().detail!;
    expect(useRunStore.getState().selectedRunId).toBe('r1');
    expect(detail.series).toEqual({ train_loss: { 1: 2, 2: 1 } });
    expect(detail.artifacts.map((a) => a.id)).toEqual([4]);
    expect(detail.loading).toBe(false);
  });

  it('starts the event follower a tail-length behind the head, not from zero', async () => {
    api.getRun.mockResolvedValue({ ...makeRun({ id: 'r1' }), last_cursor: 5000 });
    await useRunStore.getState().select('r1');
    await vi.waitFor(() => expect(api.getRunEvents).toHaveBeenCalled());
    expect(api.getRunEvents.mock.calls[0][1]).toMatchObject({
      cursor: 5000 - LOG_TAIL_EVENTS,
    });
    expect(useRunStore.getState().detail!.cursor).toBe(5000 - LOG_TAIL_EVENTS);
  });

  it('replays a short run from the very beginning', async () => {
    api.getRun.mockResolvedValue({ ...makeRun({ id: 'r1' }), last_cursor: 9 });
    await useRunStore.getState().select('r1');
    await vi.waitFor(() => expect(api.getRunEvents).toHaveBeenCalled());
    expect(api.getRunEvents.mock.calls[0][1]).toMatchObject({ cursor: 0 });
  });

  it('reports a run that vanished between the poll and the click', async () => {
    api.getRun.mockResolvedValue(null);
    await useRunStore.getState().select('r1');
    expect(useRunStore.getState().detail!.error).toBe(
      useI18n.getState().t('runs.error.gone'),
    );
    expect(api.getRunEvents).not.toHaveBeenCalled();
  });

  it('reports a failed detail fetch instead of spinning forever', async () => {
    api.getRun.mockRejectedValue(new Error('boom'));
    await useRunStore.getState().select('r1');
    expect(useRunStore.getState().detail).toMatchObject({ loading: false, error: 'boom' });
  });

  it('survives metrics or artifacts being unavailable', async () => {
    api.getRun.mockResolvedValue({ ...makeRun({ id: 'r1' }), last_cursor: 1 });
    api.getRunMetrics.mockRejectedValue(new Error('nope'));
    api.getRunArtifacts.mockRejectedValue(new Error('nope'));
    await useRunStore.getState().select('r1');
    expect(useRunStore.getState().detail).toMatchObject({
      loading: false,
      series: {},
      artifacts: [],
    });
  });

  it('clears the detail when nothing is selected', async () => {
    api.getRun.mockResolvedValue({ ...makeRun({ id: 'r1' }), last_cursor: 1 });
    await useRunStore.getState().select('r1');
    await useRunStore.getState().select(null);
    expect(useRunStore.getState().detail).toBeNull();
    expect(useRunStore.getState().selectedRunId).toBeNull();
  });

  it('abandons a superseded selection rather than painting it over the new one', async () => {
    let releaseFirst: (v: rest.RunInfo) => void = () => {};
    api.getRun
      .mockImplementationOnce(() => new Promise((resolve) => { releaseFirst = resolve; }))
      .mockResolvedValue({ ...makeRun({ id: 'r2' }), last_cursor: 1 });

    const first = useRunStore.getState().select('r1');
    await useRunStore.getState().select('r2');
    releaseFirst({ ...makeRun({ id: 'r1' }), last_cursor: 999 });
    await first;

    expect(useRunStore.getState().selectedRunId).toBe('r2');
    expect(useRunStore.getState().detail!.runId).toBe('r2');
  });
});

// ── the event follower ────────────────────────────────────────────────────

describe('runStore — following events', () => {
  it('folds a live page into the open detail and stops when the run is over', async () => {
    api.getRun.mockResolvedValue({ ...makeRun({ id: 'r1', status: 'running' }), last_cursor: 0 });
    api.getRunEvents
      .mockResolvedValueOnce(eventsPage({
        cursor: 2,
        events: [
          { cursor: 1, type: 'metric', ts: 't', payload: { points: [{ name: 'train_loss', step: 1, value: 0.9 }] } },
          { cursor: 2, type: 'execution_complete', ts: 't', payload: null },
        ],
      }))
      .mockResolvedValueOnce(eventsPage({ status: 'succeeded', active: false, cursor: 2 }));

    await useRunStore.getState().select('r1');
    await vi.waitFor(() => {
      expect(useRunStore.getState().detail!.series).toEqual({ train_loss: { 1: 0.9 } });
    });
    await vi.waitFor(() => expect(api.getRunEvents).toHaveBeenCalledTimes(2));
    // Terminal status + an empty page: nothing more can arrive, so it stops.
    const calls = api.getRunEvents.mock.calls.length;
    await new Promise((r) => setTimeout(r, 10));
    expect(api.getRunEvents).toHaveBeenCalledTimes(calls);
  });

  it('stops rather than busy-waits when a terminal page stops advancing', async () => {
    // The same page over and over: the first one advances the cursor, the
    // second cannot, and a terminal run that made no progress is over.
    api.getRun.mockResolvedValue({ ...makeRun({ id: 'r1' }), last_cursor: 4 });
    api.getRunEvents.mockResolvedValue(eventsPage({
      status: 'succeeded',
      active: false,
      cursor: 4,
      events: [{ cursor: 4, type: 'execution_complete', ts: 't', payload: null }],
    }));
    await useRunStore.getState().select('r1');
    await vi.waitFor(() => expect(api.getRunEvents).toHaveBeenCalledTimes(2));
    await new Promise((r) => setTimeout(r, 40));
    expect(api.getRunEvents).toHaveBeenCalledTimes(2);
    // And the replayed event was applied exactly once.
    expect(useRunStore.getState().detail!.log).toHaveLength(1);
  });

  it('backs off instead of hammering when an ACTIVE run returns no progress', async () => {
    vi.useFakeTimers();
    api.getRun.mockResolvedValue({ ...makeRun({ id: 'r1', status: 'running' }), last_cursor: 0 });
    api.getRunEvents.mockResolvedValue(eventsPage({ status: 'running', cursor: 0 }));

    await useRunStore.getState().select('r1');
    await vi.advanceTimersByTimeAsync(0);
    expect(api.getRunEvents).toHaveBeenCalledTimes(1);
    // Without the backoff this loop would have run thousands of times by now.
    await vi.advanceTimersByTimeAsync(FOLLOW_IDLE_MS - 1);
    expect(api.getRunEvents).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(2);
    expect(api.getRunEvents).toHaveBeenCalledTimes(2);
  });

  it('stops following when the follower is aborted by a new selection', async () => {
    api.getRun.mockResolvedValue({ ...makeRun({ id: 'r1' }), last_cursor: 0 });
    api.getRunEvents.mockRejectedValue(
      Object.assign(new Error('aborted'), { name: 'AbortError' }),
    );
    await useRunStore.getState().select('r1');
    await vi.waitFor(() => expect(api.getRunEvents).toHaveBeenCalledTimes(1));
    await new Promise((r) => setTimeout(r, 10));
    expect(api.getRunEvents).toHaveBeenCalledTimes(1);
  });
});

// ── the pure reducer ──────────────────────────────────────────────────────

describe('reduceRunEvents', () => {
  it('charts metric points and overwrites a replayed step instead of duplicating it', () => {
    const seeded = emptyDetail({ series: { train_loss: { 1: 9 } } });
    const next = reduceRunEvents(seeded, eventsPage({
      cursor: 3,
      events: [{
        cursor: 3,
        type: 'metric',
        ts: 't',
        payload: {
          points: [
            { name: 'train_loss', step: 1, value: 0.5 },
            { name: 'val_loss', step: 1, value: 0.7 },
          ],
        },
      }],
    }));
    expect(next.series).toEqual({ train_loss: { 1: 0.5 }, val_loss: { 1: 0.7 } });
    expect(seeded.series).toEqual({ train_loss: { 1: 9 } }); // input untouched
  });

  it('skips null, non-finite and malformed metric points', () => {
    const next = reduceRunEvents(emptyDetail(), eventsPage({
      events: [{
        cursor: 1,
        type: 'metric',
        ts: 't',
        payload: {
          points: [
            { name: 'a', step: 1, value: null },
            { name: 'a', step: 2, value: Number.NaN },
            { name: 'a', step: 3 },
            { step: 4, value: 1 },
            { name: 'a', step: 'five', value: 1 },
            { name: 'a', step: 6, value: 0.25 },
          ],
        },
      }],
    }));
    expect(next.series).toEqual({ a: { 6: 0.25 } });
  });

  it('tolerates a metric event with no points array at all', () => {
    const next = reduceRunEvents(emptyDetail(), eventsPage({
      events: [{ cursor: 1, type: 'metric', ts: 't', payload: {} }],
    }));
    expect(next.series).toEqual({});
  });

  it('collects artifacts and never lists the same row twice', () => {
    const first = reduceRunEvents(emptyDetail(), eventsPage({
      events: [{
        cursor: 1,
        type: 'artifact',
        ts: 'ts1',
        payload: { artifact_id: 7, kind: 'checkpoint', path: 'runs/a.pt', meta: { epoch: 2 } },
      }],
    }));
    expect(first.artifacts).toEqual([
      { id: 7, kind: 'checkpoint', path: 'runs/a.pt', meta: { epoch: 2 }, created_at: 'ts1' },
    ]);
    // The same row, replayed after a reconnect.
    const again = reduceRunEvents(first, eventsPage({
      events: [{
        cursor: 1,
        type: 'artifact',
        ts: 'ts1',
        payload: { artifact_id: 7, kind: 'checkpoint', path: 'runs/a.pt', meta: null },
      }],
    }));
    expect(again.artifacts).toHaveLength(1);
    // A payload with no path is not an artifact we can show.
    const noPath = reduceRunEvents(first, eventsPage({
      events: [{ cursor: 2, type: 'artifact', ts: 't', payload: { kind: 'checkpoint' } }],
    }));
    expect(noPath.artifacts).toHaveLength(1);
  });

  it('falls back to the cursor when an artifact event carries no id', () => {
    const next = reduceRunEvents(emptyDetail(), eventsPage({
      events: [{ cursor: 42, type: 'artifact', ts: 't', payload: { path: 'a.pt' } }],
    }));
    expect(next.artifacts[0]).toMatchObject({ id: 42, kind: 'artifact' });
  });

  it('turns lifecycle events into structured log lines with the right tone', () => {
    const next = reduceRunEvents(emptyDetail(), eventsPage({
      status: 'failed',
      events: [
        { cursor: 1, type: 'execution_start', ts: 't', payload: { run_id: 'r1' } },
        { cursor: 2, type: 'node_status', ts: 't', payload: { node_id: 'n1', status: 'completed' } },
        { cursor: 3, type: 'node_status', ts: 't', payload: { node_id: 'n2', status: 'error', error: 'bad shape' } },
        { cursor: 4, type: 'execution_error', ts: 't', payload: { error: 'boom' } },
        { cursor: 5, type: 'execution_stopped', ts: 't', payload: { reason: 'cancelled' } },
        { cursor: 6, type: 'run_warning', ts: 't', payload: { kind: 'dropped_signals', detail: 'gaps' } },
        { cursor: 7, type: 'execution_complete', ts: 't', payload: null },
      ],
    }));
    expect(next.log.map((l) => [l.kind, l.tone])).toEqual([
      ['started', 'info'],
      ['node', 'success'],
      ['node', 'error'],
      ['failed', 'error'],
      ['stopped', 'warning'],
      ['warning', 'warning'],
      ['completed', 'success'],
    ]);
    expect(next.log[2]).toMatchObject({ nodeId: 'n2', detail: 'bad shape' });
    expect(next.status).toBe('failed');
  });

  it('keeps per-epoch progress frames out of the log', () => {
    const next = reduceRunEvents(emptyDetail(), eventsPage({
      events: [
        { cursor: 1, type: 'node_status', ts: 't', payload: { node_id: 'n', status: 'progress' } },
        { cursor: 2, type: 'node_status', ts: 't', payload: { node_id: 'n', status: 'running' } },
      ],
    }));
    expect(next.log.map((l) => l.status)).toEqual(['running']);
  });

  it('ignores an event type it does not know', () => {
    const next = reduceRunEvents(emptyDetail(), eventsPage({
      events: [{ cursor: 1, type: 'from_the_future', ts: 't', payload: { x: 1 } }],
    }));
    expect(next.log).toEqual([]);
  });

  it('caps the log at MAX_LOG_LINES, keeping the newest', () => {
    const events = Array.from({ length: MAX_LOG_LINES + 20 }, (_, i) => ({
      cursor: i + 1,
      type: 'node_status',
      ts: 't',
      payload: { node_id: `n${i}`, status: 'running' },
    }));
    const next = reduceRunEvents(emptyDetail(), eventsPage({ events, cursor: events.length }));
    expect(next.log).toHaveLength(MAX_LOG_LINES);
    expect(next.log[next.log.length - 1].cursor).toBe(events.length);
  });

  it('never lets the cursor move backwards on an empty page', () => {
    const next = reduceRunEvents(emptyDetail({ cursor: 50 }), eventsPage({ cursor: 10 }));
    expect(next.cursor).toBe(50);
  });

  it('ignores events at or below the cursor already applied', () => {
    // A replay must not duplicate a log line (or its React key).
    const seeded = reduceRunEvents(emptyDetail(), eventsPage({
      cursor: 2,
      events: [
        { cursor: 1, type: 'execution_start', ts: 't', payload: null },
        { cursor: 2, type: 'execution_complete', ts: 't', payload: null },
      ],
    }));
    expect(seeded.log).toHaveLength(2);

    const replayed = reduceRunEvents(seeded, eventsPage({
      cursor: 3,
      events: [
        { cursor: 1, type: 'execution_start', ts: 't', payload: null },
        { cursor: 2, type: 'execution_complete', ts: 't', payload: null },
        { cursor: 3, type: 'run_warning', ts: 't', payload: { detail: 'new' } },
      ],
    }));
    expect(replayed.log.map((l) => l.cursor)).toEqual([1, 2, 3]);
  });
});

// ── chart shaping ─────────────────────────────────────────────────────────

describe('toChartSeries', () => {
  it('sorts points by numeric step, not by string key order', () => {
    const [series] = toChartSeries({ loss: { 10: 1, 2: 3, 1: 4 } }, ['loss']);
    expect(series.points.map((p) => p.x)).toEqual([1, 2, 10]);
    expect(series.points.map((p) => p.y)).toEqual([4, 3, 1]);
  });

  it('downsamples a long series but always keeps the newest point', () => {
    const points: Record<number, number> = {};
    for (let i = 1; i <= 1000; i++) points[i] = i;
    const [series] = toChartSeries({ loss: points }, ['loss'], 100);
    expect(series.points.length).toBeLessThanOrEqual(101);
    expect(series.points[0].x).toBe(1);
    expect(series.points[series.points.length - 1].x).toBe(1000);
  });

  it('leaves a short series alone', () => {
    const [series] = toChartSeries({ loss: { 1: 1, 2: 2 } }, ['loss'], 100);
    expect(series.points).toEqual([{ x: 1, y: 1 }, { x: 2, y: 2 }]);
  });

  it('returns an empty series for a name with no points', () => {
    expect(toChartSeries({}, ['loss'])).toEqual([{ name: 'loss', points: [] }]);
  });

  it('names series in a stable sorted order so the legend cannot shuffle', () => {
    expect(seriesNames({ val_loss: {}, lr: {}, train_loss: {} })).toEqual([
      'lr', 'train_loss', 'val_loss',
    ]);
  });
});

// ── row actions ───────────────────────────────────────────────────────────

describe('runStore — row actions', () => {
  it('cancels, toasts, and refreshes the list', async () => {
    await useRunStore.getState().cancel('r1');
    expect(api.cancelRun).toHaveBeenCalledWith('r1');
    expect(useToastStore.getState().toasts[0].message).toBe(
      useI18n.getState().t('runs.toast.cancelling'),
    );
    expect(api.listRuns).toHaveBeenCalled();
    expect(useRunStore.getState().busy).toEqual({});
  });

  it('says so when the cancel raced a completion', async () => {
    api.cancelRun.mockResolvedValue({ run_id: 'r1', status: 'succeeded', cancelled: false });
    await useRunStore.getState().cancel('r1');
    expect(useToastStore.getState().toasts[0]).toMatchObject({
      message: useI18n.getState().t('runs.toast.alreadyDone'),
      type: 'warning',
    });
  });

  it('reports a cancel that failed and still clears the busy flag', async () => {
    api.cancelRun.mockRejectedValue(new Error('403'));
    await useRunStore.getState().cancel('r1');
    expect(useToastStore.getState().toasts[0].type).toBe('error');
    expect(useRunStore.getState().busy).toEqual({});
  });

  it('deletes a run and closes its detail when it was the open one', async () => {
    api.getRun.mockResolvedValue({ ...makeRun({ id: 'r1' }), last_cursor: 1 });
    await useRunStore.getState().select('r1');
    expect(useRunStore.getState().detail).not.toBeNull();

    await useRunStore.getState().remove('r1');
    expect(api.deleteRun).toHaveBeenCalledWith('r1');
    expect(useRunStore.getState().detail).toBeNull();
    expect(useRunStore.getState().selectedRunId).toBeNull();
    expect(lastToast()).toMatchObject({ type: 'success' });
  });

  it('leaves the open detail of a different run alone when deleting', async () => {
    api.getRun.mockResolvedValue({ ...makeRun({ id: 'r1' }), last_cursor: 1 });
    await useRunStore.getState().select('r1');
    await useRunStore.getState().remove('other');
    expect(useRunStore.getState().selectedRunId).toBe('r1');
  });

  it('reports a refused delete (a run that started again, say) as an error', async () => {
    api.deleteRun.mockRejectedValue(new Error('queued or running'));
    await useRunStore.getState().remove('r1');
    expect(useToastStore.getState().toasts[0]).toMatchObject({ type: 'error' });
    expect(useRunStore.getState().busy).toEqual({});
  });

  it('exports CSV through the download helper', async () => {
    await useRunStore.getState().exportCsv('r1');
    expect(api.downloadRunMetricsCsv).toHaveBeenCalledWith('r1');
    expect(useToastStore.getState().toasts).toEqual([]);
  });

  it('toasts when the CSV export fails', async () => {
    api.downloadRunMetricsCsv.mockRejectedValue(new Error('404'));
    await useRunStore.getState().exportCsv('r1');
    expect(useToastStore.getState().toasts[0]).toMatchObject({ type: 'error' });
  });
});

// ── mount-time in-progress check ──────────────────────────────────────────

describe('runStore — checkInProgress', () => {
  it('toasts once per page load when runs are still going', async () => {
    api.listRuns.mockResolvedValue(page([], 2));
    expect(await useRunStore.getState().checkInProgress()).toBe(2);
    expect(useToastStore.getState().toasts[0].message).toContain('2');
    expect(api.listRuns).toHaveBeenCalledWith({
      status: rest.ACTIVE_RUN_STATUSES,
      limit: 1,
    });

    // StrictMode mounts effects twice; the second call must be a no-op.
    api.listRuns.mockClear();
    expect(await useRunStore.getState().checkInProgress()).toBe(0);
    expect(api.listRuns).not.toHaveBeenCalled();
    expect(useToastStore.getState().toasts).toHaveLength(1);
  });

  it('stays quiet when nothing is in progress', async () => {
    api.listRuns.mockResolvedValue(page([], 0));
    expect(await useRunStore.getState().checkInProgress()).toBe(0);
    expect(useToastStore.getState().toasts).toEqual([]);
  });

  it('stays quiet when the server is unreachable at boot', async () => {
    api.listRuns.mockRejectedValue(new Error('ECONNREFUSED'));
    expect(await useRunStore.getState().checkInProgress()).toBe(0);
    expect(useToastStore.getState().toasts).toEqual([]);
  });
});

describe('isActiveRun', () => {
  it('matches the backend ACTIVE_STATUSES exactly', () => {
    expect(isActiveRun('queued')).toBe(true);
    expect(isActiveRun('running')).toBe(true);
    for (const status of ['succeeded', 'failed', 'cancelled', 'interrupted'] as RunStatus[]) {
      expect(isActiveRun(status)).toBe(false);
    }
  });
});
