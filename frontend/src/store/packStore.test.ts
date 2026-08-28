import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as rest from '../api/rest';
import type {
  HealthInfo,
  PackCatalog,
  PackJobEventsPage,
  PackSummary,
} from '../api/rest';
import { PackApiError } from '../api/rest';

// Partial mock: `PackApiError` is a real class the store narrows on with
// `instanceof`, so only the network calls are stubbed.
vi.mock('../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof rest>();
  return {
    ...actual,
    listPacks: vi.fn(),
    installPack: vi.fn(),
    cancelPackJob: vi.fn(),
    getPackJobEvents: vi.fn(),
    removePackItem: vi.fn(),
    fetchHealth: vi.fn(),
  };
});

// The remove confirmation is an in-app modal driven by a promise; mocking the
// helper keeps these tests about the STORE's decisions rather than about the
// dialog component.
vi.mock('../utils/dialog', () => ({
  confirm: vi.fn(async () => true),
  prompt: vi.fn(async () => null),
}));

import {
  EVENT_WAIT_S,
  FOLLOW_IDLE_MS,
  FOLLOW_RETRY_MS,
  MAX_FOLLOW_FAILURES,
  MAX_PACK_LOG_LINES,
  RESTART_GRACE_MS,
  RESTART_PENDING_KEY,
  RESTART_POLL_MS,
  RESTART_TIMEOUT_MS,
  _resetPackStoreForTesting,
  emptyPackJob,
  reducePackEvents,
  usePackStore,
  type PackJob,
} from './packStore';
import { useToastStore } from './toastStore';
import { useI18n } from '../i18n';
import { confirm } from '../utils/dialog';

const api = vi.mocked(rest);
const confirmMock = vi.mocked(confirm);

/** The most recent toast. `Array.prototype.at` is outside the project lib. */
function lastToast() {
  const { toasts } = useToastStore.getState();
  return toasts[toasts.length - 1];
}

function makePack(partial: Partial<PackSummary> & { id: string }): PackSummary {
  return {
    title: partial.id,
    description: '',
    install_mode: 'live',
    status: 'not_installed',
    pip_ready: false,
    usable: false,
    depends_on: [],
    blocked_by: [],
    pip: [],
    items: [],
    size_bytes_total: 0,
    install_command: null,
    ...partial,
  };
}

function catalog(partial: Partial<PackCatalog> = {}): PackCatalog {
  return {
    packs: [],
    active_job: null,
    last_restart_job: null,
    remote_install_allowed: true,
    launch_mode: 'start',
    gpu: null,
    ...partial,
  };
}

function eventsPage(partial: Partial<PackJobEventsPage> = {}): PackJobEventsPage {
  return { job_id: 'j1', status: 'running', events: [], cursor: 0, ...partial };
}

function health(partial: Partial<HealthInfo> = {}): HealthInfo {
  return {
    status: 'ok',
    version: '1.4.2',
    nodes_loaded: 0,
    presets_loaded: 0,
    caches: {},
    project: null,
    ...partial,
  };
}

function seededJob(partial: Partial<PackJob> = {}): PackJob {
  return { ...emptyPackJob('j1', 'word-vectors'), ...partial };
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useToastStore.setState({ toasts: [] });
  _resetPackStoreForTesting();
  sessionStorage.clear();
  api.listPacks.mockResolvedValue(catalog());
  api.installPack.mockResolvedValue({ job_id: 'j1' });
  api.cancelPackJob.mockResolvedValue({ job_id: 'j1', cancelled: true });
  // Running-and-idle by default: an accidental follower parks instead of
  // settling and firing toasts into an unrelated test.
  api.getPackJobEvents.mockResolvedValue(eventsPage({ status: 'running', cursor: 0 }));
  api.removePackItem.mockResolvedValue({
    pack_id: 'word-vectors', item_id: 'glove', removed: true,
  });
  api.fetchHealth.mockResolvedValue(health({ boot_id: 'boot-a' }));
  confirmMock.mockResolvedValue(true);
});

afterEach(() => {
  _resetPackStoreForTesting();
  vi.useRealTimers();
  vi.clearAllMocks();
});

// ── the pure reducer ──────────────────────────────────────────────────────

describe('reducePackEvents', () => {
  it('records steps in order and marks the previous one done when the next starts', () => {
    const next = reducePackEvents(seededJob(), eventsPage({
      cursor: 2,
      events: [
        { type: 'step_started', cursor: 1, ts: 't1', step: 'pip', label: 'Installing packages: x' },
        { type: 'step_started', cursor: 2, ts: 't2', step: 'download:glove', label: 'Downloading glove' },
      ],
    }));

    expect(next.steps).toEqual([
      { step: 'pip', label: 'Installing packages: x', state: 'done' },
      { step: 'download:glove', label: 'Downloading glove', state: 'running' },
    ]);
    // Each started step is announced in the log, in arrival order.
    expect(next.log.map((line) => [line.kind, line.text])).toEqual([
      ['step', 'Installing packages: x'],
      ['step', 'Downloading glove'],
    ]);
  });

  it('marks a step done on its own step_done event', () => {
    const started = reducePackEvents(seededJob(), eventsPage({
      cursor: 1,
      events: [{ type: 'step_started', cursor: 1, ts: 't', step: 'pip', label: 'pip' }],
    }));
    const next = reducePackEvents(started, eventsPage({
      cursor: 2,
      events: [{ type: 'step_done', cursor: 2, ts: 't', step: 'pip' }],
    }));
    expect(next.steps).toEqual([{ step: 'pip', label: 'pip', state: 'done' }]);
  });

  it('keeps per-item byte progress and derives the percent when the server sends none', () => {
    const next = reducePackEvents(seededJob(), eventsPage({
      cursor: 2,
      events: [
        {
          type: 'progress', cursor: 1, ts: 't', item: 'glove',
          bytes_done: 50, bytes_total: 200, percent: 25,
        },
        {
          type: 'progress', cursor: 2, ts: 't', item: 'model',
          bytes_done: 3, bytes_total: 4, percent: null,
        },
      ],
    }));

    expect(next.items.glove).toEqual({ bytesDone: 50, bytesTotal: 200, percent: 25 });
    expect(next.items.model).toEqual({ bytesDone: 3, bytesTotal: 4, percent: 75 });
  });

  it('leaves the percent unknown when there is no total to divide by', () => {
    const next = reducePackEvents(seededJob(), eventsPage({
      cursor: 1,
      events: [{
        type: 'progress', cursor: 1, ts: 't', item: 'glove',
        bytes_done: 9, bytes_total: null, percent: null,
      }],
    }));
    expect(next.items.glove).toEqual({ bytesDone: 9, bytesTotal: null, percent: null });
  });

  it('never turns a progress event into a log line', () => {
    // A 2 GB download emits thousands of these. One line each would bury
    // every message that actually says something.
    const next = reducePackEvents(seededJob(), eventsPage({
      cursor: 3,
      events: [
        { type: 'progress', cursor: 1, ts: 't', item: 'glove', bytes_done: 1, bytes_total: 10, percent: 10 },
        { type: 'progress', cursor: 2, ts: 't', item: 'glove', bytes_done: 5, bytes_total: 10, percent: 50 },
        { type: 'log', cursor: 3, ts: 't', line: 'uv pip install' },
      ],
    }));
    expect(next.log.map((line) => line.text)).toEqual(['uv pip install']);
  });

  it('captures the job_failed message and hint', () => {
    const next = reducePackEvents(seededJob(), eventsPage({
      status: 'failed',
      cursor: 1,
      events: [{
        type: 'job_failed', cursor: 1, ts: 't',
        message: 'installing GloVe failed (uv exited 1)',
        hint: 'no matching distribution',
      }],
    }));

    expect(next.error).toEqual({
      message: 'installing GloVe failed (uv exited 1)',
      hint: 'no matching distribution',
    });
    expect(next.log[0]).toMatchObject({
      kind: 'error', text: 'installing GloVe failed (uv exited 1)',
    });
  });

  it('stores the restart command so the panel can print it', () => {
    const next = reducePackEvents(seededJob(), eventsPage({
      status: 'needs_restart',
      cursor: 1,
      events: [{
        type: 'needs_restart', cursor: 1, ts: 't',
        command: 'cdui install --gpu cu128',
      }],
    }));
    expect(next.restartCommand).toBe('cdui install --gpu cu128');
  });

  it('skips an unknown event type rather than rendering a mystery line', () => {
    const next = reducePackEvents(seededJob(), eventsPage({
      cursor: 2,
      events: [
        { type: 'telemetry_from_a_newer_backend', cursor: 1, ts: 't', whatever: 1 },
        { type: 'log', cursor: 2, ts: 't', line: 'kept' },
      ],
    }));
    expect(next.log.map((line) => line.text)).toEqual(['kept']);
  });

  it('skips events at or below the cursor already applied', () => {
    // The follower resumes from a cursor, so a replayed page is normal; a
    // replayed log line would be a duplicate React key as well as a lie.
    const next = reducePackEvents(seededJob({ cursor: 5 }), eventsPage({
      cursor: 6,
      events: [
        { type: 'log', cursor: 4, ts: 't', line: 'old' },
        { type: 'log', cursor: 5, ts: 't', line: 'boundary' },
        { type: 'log', cursor: 6, ts: 't', line: 'new' },
      ],
    }));
    expect(next.log.map((line) => line.text)).toEqual(['new']);
  });

  it('drops an empty log line instead of printing a blank row', () => {
    const next = reducePackEvents(seededJob(), eventsPage({
      cursor: 2,
      events: [
        { type: 'log', cursor: 1, ts: 't', line: '   ' },
        { type: 'log', cursor: 2, ts: 't', line: 'real' },
      ],
    }));
    expect(next.log.map((line) => line.text)).toEqual(['real']);
  });

  it('caps the log at MAX_PACK_LOG_LINES, keeping the newest', () => {
    const events = Array.from({ length: MAX_PACK_LOG_LINES + 20 }, (_, i) => ({
      type: 'log', cursor: i + 1, ts: 't', line: `line ${i}`,
    }));
    const next = reducePackEvents(seededJob(), eventsPage({
      cursor: events.length, events,
    }));

    expect(next.log).toHaveLength(MAX_PACK_LOG_LINES);
    expect(next.log[next.log.length - 1].text).toBe(`line ${events.length - 1}`);
    expect(next.log[0].text).toBe('line 20');
  });

  it('takes the status from the page and never moves the cursor backwards', () => {
    const next = reducePackEvents(seededJob({ cursor: 9 }), eventsPage({
      status: 'done', cursor: 9,
    }));
    expect(next.status).toBe('done');
    expect(next.cursor).toBe(9);

    // An empty page answers with the cursor it was sent.
    const later = reducePackEvents(next, eventsPage({ status: 'done', cursor: 0 }));
    expect(later.cursor).toBe(9);
  });

  it('gives every log line a distinct seq so React keys never collide', () => {
    const next = reducePackEvents(seededJob(), eventsPage({
      cursor: 3,
      events: [
        { type: 'log', cursor: 1, ts: 't', line: 'a' },
        { type: 'step_started', cursor: 2, ts: 't', step: 'pip', label: 'b' },
        { type: 'log', ts: 't', line: 'c' },        // a frame with no cursor
      ],
    }));
    const seqs = next.log.map((line) => line.seq);
    expect(new Set(seqs).size).toBe(seqs.length);
  });
});

// ── the catalog ───────────────────────────────────────────────────────────

describe('packStore — refresh', () => {
  it('loads the catalog and indexes it by id', async () => {
    api.listPacks.mockResolvedValue(catalog({
      packs: [makePack({ id: 'word-vectors', title: 'Word vectors' })],
      remote_install_allowed: false,
      launch_mode: 'dev',
    }));

    await usePackStore.getState().refresh();

    const state = usePackStore.getState();
    expect(state.packs.map((pack) => pack.id)).toEqual(['word-vectors']);
    expect(state.byId['word-vectors'].title).toBe('Word vectors');
    expect(state.remoteInstallAllowed).toBe(false);
    expect(state.launchMode).toBe('dev');
    expect(state.loaded).toBe(true);
    expect(state.loading).toBe(false);
    expect(state.unsupported).toBe(false);
    expect(state.error).toBeNull();
  });

  it('marks an older backend unsupported on a 404, without a toast', async () => {
    // The Package Center simply does not exist there. That is a UI the user
    // never asked for going quiet, not an error worth interrupting them with.
    api.listPacks.mockRejectedValue(new PackApiError(404, 'Not Found'));

    await usePackStore.getState().refresh();

    const state = usePackStore.getState();
    expect(state.unsupported).toBe(true);
    expect(state.loaded).toBe(true);
    expect(state.packs).toEqual([]);
    expect(state.error).toBeNull();
    expect(useToastStore.getState().toasts).toHaveLength(0);
  });

  it('keeps the rows it already has when the network fails', async () => {
    api.listPacks.mockResolvedValue(catalog({ packs: [makePack({ id: 'rag' })] }));
    await usePackStore.getState().refresh();

    api.listPacks.mockRejectedValue(new Error('Failed to fetch'));
    await usePackStore.getState().refresh();

    const state = usePackStore.getState();
    expect(state.packs.map((pack) => pack.id)).toEqual(['rag']);
    expect(state.error).toBe('Failed to fetch');
    expect(state.unsupported).toBe(false);
  });

  it('adopts a job started elsewhere, following it from cursor 0', async () => {
    api.listPacks.mockResolvedValue(catalog({
      packs: [makePack({ id: 'word-vectors' })],
      active_job: { job_id: 'j9', pack_id: 'word-vectors' },
    }));

    await usePackStore.getState().refresh();

    expect(usePackStore.getState().job).toMatchObject({
      jobId: 'j9', packId: 'word-vectors', status: 'running',
    });
    await vi.waitFor(() => expect(api.getPackJobEvents).toHaveBeenCalled());
    expect(api.getPackJobEvents.mock.calls[0][0]).toBe('j9');
    expect(api.getPackJobEvents.mock.calls[0][1]).toMatchObject({
      cursor: 0, wait: EVENT_WAIT_S,
    });
  });

  it('adopts the same job idempotently, never wiping what it has already shown', async () => {
    api.listPacks.mockResolvedValue(catalog({
      active_job: { job_id: 'j9', pack_id: 'word-vectors' },
    }));
    await usePackStore.getState().refresh();
    usePackStore.setState((state) => ({
      job: { ...state.job!, log: [{ seq: 1, ts: null, kind: 'log', text: 'already shown' }] },
    }));

    await usePackStore.getState().refresh();

    expect(usePackStore.getState().job!.log.map((line) => line.text))
      .toEqual(['already shown']);
  });

  it('marks a locally running job lost when the server no longer reports it', async () => {
    usePackStore.setState({ job: seededJob({ jobId: 'j9' }) });

    await usePackStore.getState().refresh();

    expect(usePackStore.getState().job!.status).toBe('lost');
  });

  it('does not mark a job that was started while the request was in flight', async () => {
    // The answer describes the server BEFORE the install was posted, so its
    // silence about the new job says nothing about it.
    let release: (value: PackCatalog) => void = () => {};
    api.listPacks.mockReturnValue(new Promise<PackCatalog>((resolve) => { release = resolve; }));
    const inFlight = usePackStore.getState().refresh();

    usePackStore.setState({ job: seededJob({ jobId: 'brand-new' }) });
    release(catalog());
    await inFlight;

    expect(usePackStore.getState().job!.status).toBe('running');
  });
});

// ── starting an install ───────────────────────────────────────────────────

describe('packStore — install', () => {
  beforeEach(() => {
    usePackStore.setState({
      packs: [makePack({ id: 'word-vectors', title: 'Word vectors' })],
      byId: { 'word-vectors': makePack({ id: 'word-vectors', title: 'Word vectors' }) },
      launchMode: 'start',
      loaded: true,
    });
  });

  it('posts the selected items and seeds their progress at zero', async () => {
    api.installPack.mockResolvedValue({ job_id: 'j7' });

    await usePackStore.getState().install('word-vectors', { items: ['glove'] });

    expect(api.installPack).toHaveBeenCalledWith(
      'word-vectors', expect.objectContaining({ items: ['glove'] }),
    );
    const state = usePackStore.getState();
    expect(state.job).toMatchObject({ jobId: 'j7', packId: 'word-vectors', status: 'running' });
    expect(state.job!.items).toEqual({
      glove: { bytesDone: 0, bytesTotal: null, percent: 0 },
    });
    // The row says "Installing" before the next catalog poll comes back.
    expect(state.packs[0].status).toBe('installing');
    expect(state.byId['word-vectors'].status).toBe('installing');
    expect(state.busy['word-vectors']).toBeFalsy();
  });

  it('refuses a restart-mode install outside start mode', async () => {
    usePackStore.setState({ launchMode: 'dev' });

    await usePackStore.getState().install('word-vectors', { mode: 'restart' });

    expect(api.installPack).not.toHaveBeenCalled();
    expect(lastToast()).toMatchObject({ type: 'warning' });
    expect(lastToast().message).toContain('cdui dev');
  });

  it('refuses when a job is already running, without asking the server', async () => {
    usePackStore.setState({ job: seededJob({ status: 'running' }) });

    await usePackStore.getState().install('word-vectors');

    expect(api.installPack).not.toHaveBeenCalled();
    expect(lastToast()).toMatchObject({
      type: 'warning', message: 'Another install is already running.',
    });
  });

  it('adopts the server\'s job when it answers 409 busy', async () => {
    api.installPack.mockRejectedValue(new PackApiError(409, 'an install is already running'));
    api.listPacks.mockResolvedValue(catalog({
      active_job: { job_id: 'j-elsewhere', pack_id: 'rag' },
    }));

    await usePackStore.getState().install('word-vectors');

    expect(lastToast()).toMatchObject({ type: 'warning' });
    expect(usePackStore.getState().job).toMatchObject({ jobId: 'j-elsewhere', packId: 'rag' });
  });

  it('says installing is local-only on a 403', async () => {
    api.installPack.mockRejectedValue(new PackApiError(403, 'remote install refused'));

    await usePackStore.getState().install('word-vectors');

    expect(lastToast()).toMatchObject({ type: 'error' });
    expect(lastToast().message).toBe(
      'Installing is only allowed on the computer that runs the server.',
    );
  });

  it('names the blocking pack on a 400 with blocked_by', async () => {
    usePackStore.setState({
      byId: {
        rag: makePack({ id: 'rag' }),
        'sentence-embeddings': makePack({
          id: 'sentence-embeddings', title: 'Sentence embeddings',
        }),
      },
    });
    const err = new PackApiError(400, 'blocked');
    err.body = { blocked_by: ['sentence-embeddings'] };
    api.installPack.mockRejectedValue(err);

    await usePackStore.getState().install('rag');

    expect(lastToast().message).toBe('Install Sentence embeddings first.');
  });

  it('reports any other refusal as an install failure', async () => {
    api.installPack.mockRejectedValue(new PackApiError(507, 'not enough disk space'));

    await usePackStore.getState().install('word-vectors');

    expect(lastToast()).toMatchObject({ type: 'error' });
    expect(lastToast().message).toContain('not enough disk space');
  });

  it('ignores a second click while the first request is in flight', async () => {
    let release: (value: { job_id: string }) => void = () => {};
    api.installPack.mockReturnValue(
      new Promise<{ job_id: string }>((resolve) => { release = resolve; }),
    );

    const first = usePackStore.getState().install('word-vectors');
    expect(usePackStore.getState().busy['word-vectors']).toBe(true);
    await usePackStore.getState().install('word-vectors');
    expect(api.installPack).toHaveBeenCalledTimes(1);

    release({ job_id: 'j7' });
    await first;
    expect(usePackStore.getState().busy['word-vectors']).toBeFalsy();
  });
});

// ── the long-poll follower ────────────────────────────────────────────────

describe('packStore — the follower', () => {
  /**
   * Timers are faked throughout: the loop's only real waits are its idle and
   * retry sleeps, and asserting "it did NOT poll again" by sleeping 10 ms of
   * wall clock proves nothing when the next turn was 500 ms away regardless.
   */
  beforeEach(() => {
    vi.useFakeTimers();
  });

  /** Let queued microtasks (a resolved fetch, a `set`) run to completion. */
  const settle = () => vi.advanceTimersByTimeAsync(0);

  it('folds pages into the job and stops once terminal and the cursor stops moving', async () => {
    api.getPackJobEvents
      .mockResolvedValueOnce(eventsPage({
        cursor: 2,
        events: [
          { type: 'step_started', cursor: 1, ts: 't', step: 'pip', label: 'Installing packages' },
          { type: 'log', cursor: 2, ts: 't', line: 'uv pip install' },
        ],
      }))
      .mockResolvedValue(eventsPage({
        status: 'done',
        cursor: 3,
        events: [{ type: 'job_done', cursor: 3, ts: 't' }],
      }));

    usePackStore.getState().followJob('j1', 'word-vectors', 0);
    await settle();

    const job = usePackStore.getState().job!;
    expect(job.status).toBe('done');
    expect(job.cursor).toBe(3);
    // The terminal page is replayed once; its event is applied exactly once.
    expect(job.log.filter((line) => line.text === 'done')).toHaveLength(1);
    expect(api.getPackJobEvents).toHaveBeenCalledTimes(3);

    await vi.advanceTimersByTimeAsync(60_000);
    expect(api.getPackJobEvents).toHaveBeenCalledTimes(3);
  });

  it('idles for FOLLOW_IDLE_MS when a running page returned nothing', async () => {
    api.getPackJobEvents.mockResolvedValue(eventsPage({ status: 'running', cursor: 0 }));

    usePackStore.getState().followJob('j1', 'word-vectors', 0);
    await settle();
    expect(api.getPackJobEvents).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(FOLLOW_IDLE_MS - 1);
    expect(api.getPackJobEvents).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(2);
    expect(api.getPackJobEvents).toHaveBeenCalledTimes(2);
  });

  it('retries a failed fetch and gives up after MAX_FOLLOW_FAILURES', async () => {
    api.getPackJobEvents.mockRejectedValue(new Error('Failed to fetch'));

    usePackStore.getState().followJob('j1', 'word-vectors', 0);
    await settle();
    expect(api.getPackJobEvents).toHaveBeenCalledTimes(1);
    expect(usePackStore.getState().job!.status).toBe('running');

    await vi.advanceTimersByTimeAsync(FOLLOW_RETRY_MS * MAX_FOLLOW_FAILURES);
    expect(api.getPackJobEvents).toHaveBeenCalledTimes(MAX_FOLLOW_FAILURES);
    expect(usePackStore.getState().job!.status).toBe('lost');

    // Given up means given up: no further polling.
    await vi.advanceTimersByTimeAsync(60_000);
    expect(api.getPackJobEvents).toHaveBeenCalledTimes(MAX_FOLLOW_FAILURES);
  });

  it('recovers from a transient failure without losing the job', async () => {
    api.getPackJobEvents
      .mockRejectedValueOnce(new Error('Failed to fetch'))
      .mockResolvedValue(eventsPage({ status: 'running', cursor: 0 }));

    usePackStore.getState().followJob('j1', 'word-vectors', 0);
    await settle();
    await vi.advanceTimersByTimeAsync(FOLLOW_RETRY_MS + 10);

    expect(api.getPackJobEvents).toHaveBeenCalledTimes(2);
    expect(usePackStore.getState().job!.status).toBe('running');
  });

  it('leaves the parked request alone when the same job is adopted again', async () => {
    // `refresh()` re-adopts `active_job` on EVERY catalog poll. Restarting
    // the follower each time would abort a long poll that was about to
    // answer and re-ask from the same cursor — a request per poll, which is
    // the entire cost the long poll exists to avoid.
    const signals: AbortSignal[] = [];
    api.getPackJobEvents.mockImplementation(
      (_jobId: string, opts?: { signal?: AbortSignal }) => {
        if (opts?.signal) signals.push(opts.signal);
        return new Promise<PackJobEventsPage>(() => {});   // parked, like the server
      },
    );

    usePackStore.getState().followJob('j1', 'word-vectors', 0);
    await settle();
    expect(signals).toHaveLength(1);

    usePackStore.getState().followJob('j1', 'word-vectors', 0);
    await settle();

    expect(signals).toHaveLength(1);
    expect(signals[0].aborted).toBe(false);
  });

  it('runs exactly one follower per job', async () => {
    api.getPackJobEvents.mockResolvedValue(eventsPage({ status: 'running', cursor: 0 }));

    usePackStore.getState().followJob('j1', 'word-vectors', 0);
    await settle();
    usePackStore.getState().followJob('j1', 'word-vectors', 0);
    await settle();
    api.getPackJobEvents.mockClear();

    // Two turns of the idle backoff. One follower means exactly two requests.
    await vi.advanceTimersByTimeAsync(FOLLOW_IDLE_MS * 2 + 10);
    expect(api.getPackJobEvents).toHaveBeenCalledTimes(2);
  });

  it('aborts the parked request when following stops', async () => {
    // A 25 s long poll left dangling holds a connection open on a server with
    // a small pool, so closing must cancel it rather than wait it out.
    const signals: AbortSignal[] = [];
    api.getPackJobEvents.mockImplementation(
      (_jobId: string, opts?: { signal?: AbortSignal }) => {
        if (opts?.signal) signals.push(opts.signal);
        return new Promise<PackJobEventsPage>(() => {});   // parked, like the server
      },
    );

    usePackStore.getState().followJob('j1', 'word-vectors', 0);
    await settle();
    expect(signals).toHaveLength(1);
    expect(signals[0].aborted).toBe(false);

    usePackStore.getState().stopFollowing();
    expect(signals[0].aborted).toBe(true);

    await vi.advanceTimersByTimeAsync(60_000);
    expect(api.getPackJobEvents).toHaveBeenCalledTimes(1);
  });

  it('keeps following with nothing mounted', async () => {
    // The whole point of putting the follower in the store: closing the modal
    // must not abandon a 2 GB download.
    api.getPackJobEvents.mockResolvedValue(eventsPage({ status: 'running', cursor: 0 }));
    usePackStore.getState().followJob('j1', 'word-vectors', 0);
    await settle();

    await vi.advanceTimersByTimeAsync(FOLLOW_IDLE_MS * 4 + 10);

    expect(api.getPackJobEvents).toHaveBeenCalledTimes(5);
    expect(usePackStore.getState().job!.status).toBe('running');
  });

  it('drops a page for a job that is no longer the open one', async () => {
    api.getPackJobEvents.mockResolvedValue(eventsPage({
      cursor: 1, events: [{ type: 'log', cursor: 1, ts: 't', line: 'stale' }],
    }));
    usePackStore.getState().followJob('j1', 'word-vectors', 0);
    usePackStore.setState({ job: seededJob({ jobId: 'other' }) });
    await settle();

    expect(usePackStore.getState().job!.log).toEqual([]);
  });
});

// ── what happens when a job ends ──────────────────────────────────────────

describe('packStore — a job settling', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    usePackStore.setState({
      byId: { 'word-vectors': makePack({ id: 'word-vectors', title: 'Word vectors' }) },
      launchMode: 'start',
    });
  });

  const settle = () => vi.advanceTimersByTimeAsync(0);

  /** A terminal page that advances once, then repeats: the stop condition. */
  function terminal(page: Partial<PackJobEventsPage>) {
    api.getPackJobEvents.mockResolvedValue(eventsPage({ cursor: 1, ...page }));
  }

  it('toasts success and re-reads the catalog when the install is done', async () => {
    terminal({ status: 'done', events: [{ type: 'job_done', cursor: 1, ts: 't' }] });

    usePackStore.getState().followJob('j1', 'word-vectors', 0);
    await settle();

    expect(lastToast()).toMatchObject({
      type: 'success', message: 'Word vectors installed.',
    });
    expect(api.listPacks).toHaveBeenCalled();
  });

  it('toasts the failure message when the install fails', async () => {
    terminal({
      status: 'failed',
      events: [{
        type: 'job_failed', cursor: 1, ts: 't',
        message: 'uv exited 1', hint: null,
      }],
    });

    usePackStore.getState().followJob('j1', 'word-vectors', 0);
    await settle();

    expect(lastToast()).toMatchObject({ type: 'error' });
    expect(lastToast().message).toBe('Install failed: uv exited 1');
    expect(api.listPacks).toHaveBeenCalled();
  });

  it('toasts and refreshes when the install was cancelled', async () => {
    terminal({ status: 'cancelled', events: [{ type: 'job_cancelled', cursor: 1, ts: 't' }] });

    usePackStore.getState().followJob('j1', 'word-vectors', 0);
    await settle();

    expect(lastToast()).toMatchObject({ type: 'info', message: 'Install cancelled.' });
    expect(api.listPacks).toHaveBeenCalled();
  });

  it('starts the restart flow and remembers the pack across the reload', async () => {
    terminal({
      status: 'needs_restart',
      events: [{
        type: 'needs_restart', cursor: 1, ts: 't',
        command: 'cdui install --gpu cu128',
      }],
    });

    usePackStore.getState().followJob('j1', 'word-vectors', 0);
    await settle();

    expect(sessionStorage.getItem(RESTART_PENDING_KEY)).toBe('word-vectors');
    expect(usePackStore.getState().restart).toMatchObject({
      phase: 'waiting', packId: 'word-vectors', command: 'cdui install --gpu cu128',
    });
  });

  it('shows the command instead of restarting under cdui dev', async () => {
    usePackStore.setState({ launchMode: 'dev' });
    terminal({
      status: 'needs_restart',
      events: [{ type: 'needs_restart', cursor: 1, ts: 't', command: 'cdui install --gpu cu128' }],
    });

    usePackStore.getState().followJob('j1', 'word-vectors', 0);
    await settle();

    expect(sessionStorage.getItem(RESTART_PENDING_KEY)).toBeNull();
    expect(usePackStore.getState().restart.phase).toBe('idle');
    expect(lastToast()).toMatchObject({ type: 'warning' });
    expect(lastToast().message).toContain('cdui dev');
    // The job stays on screen so the command block has something to describe.
    expect(usePackStore.getState().job!.restartCommand).toBe('cdui install --gpu cu128');
  });
});

// ── cancel, remove, dismiss ───────────────────────────────────────────────

describe('packStore — cancel, remove and dismiss', () => {
  it('asks the server to stop and clears the cancelling flag', async () => {
    usePackStore.setState({ job: seededJob({ jobId: 'j5', status: 'running' }) });

    await usePackStore.getState().cancel();

    expect(api.cancelPackJob).toHaveBeenCalledWith('j5');
    expect(usePackStore.getState().cancelling).toBe(false);
    // The follower is what records the outcome; cancel does not fake it.
    expect(usePackStore.getState().job!.status).toBe('running');
  });

  it('reports a cancel that could not be delivered', async () => {
    usePackStore.setState({ job: seededJob({ jobId: 'j5', status: 'running' }) });
    api.cancelPackJob.mockRejectedValue(new Error('gone'));

    await usePackStore.getState().cancel();

    expect(lastToast()).toMatchObject({ type: 'error' });
    expect(lastToast().message).toContain('Could not cancel the install');
    expect(usePackStore.getState().cancelling).toBe(false);
  });

  it('confirms, removes, toasts and re-reads the catalog', async () => {
    await usePackStore.getState().removeItem('word-vectors', 'glove');

    expect(confirmMock).toHaveBeenCalled();
    expect(api.removePackItem).toHaveBeenCalledWith('word-vectors', 'glove');
    expect(lastToast()).toMatchObject({ type: 'success', message: 'Removed glove.' });
    expect(api.listPacks).toHaveBeenCalled();
  });

  it('does nothing when the confirmation is declined', async () => {
    confirmMock.mockResolvedValue(false);

    await usePackStore.getState().removeItem('word-vectors', 'glove');

    expect(api.removePackItem).not.toHaveBeenCalled();
    expect(api.listPacks).not.toHaveBeenCalled();
    expect(useToastStore.getState().toasts).toHaveLength(0);
  });

  it('warns rather than promising the space back when removed is false', async () => {
    api.removePackItem.mockResolvedValue({
      pack_id: 'word-vectors', item_id: 'glove', removed: false,
    });

    await usePackStore.getState().removeItem('word-vectors', 'glove');

    expect(lastToast()).toMatchObject({
      type: 'warning', message: 'Could not remove glove',
    });
  });

  it('dismisses a finished job but not a running one', () => {
    usePackStore.setState({ job: seededJob({ status: 'running' }) });
    usePackStore.getState().dismissJob();
    expect(usePackStore.getState().job).not.toBeNull();

    usePackStore.setState({ job: seededJob({ status: 'failed' }) });
    usePackStore.getState().dismissJob();
    expect(usePackStore.getState().job).toBeNull();
  });
});

// ── the restart handshake ─────────────────────────────────────────────────

describe('packStore — restartFlow', () => {
  let originalLocation: Location;
  let reload: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.useFakeTimers();
    originalLocation = window.location;
    reload = vi.fn();
    Object.defineProperty(window, 'location', {
      value: { ...originalLocation, reload },
      configurable: true,
    });
  });

  afterEach(() => {
    Object.defineProperty(window, 'location', {
      value: originalLocation,
      configurable: true,
    });
  });

  const settle = () => vi.advanceTimersByTimeAsync(0);

  it('reloads the page once health has failed and then answered again', async () => {
    api.fetchHealth
      .mockResolvedValueOnce(health({ boot_id: 'boot-a' }))    // before the restart
      .mockRejectedValueOnce(new Error('connection refused'))  // going down
      .mockResolvedValue(health({ boot_id: 'boot-a' }));       // back up

    void usePackStore.getState().restartFlow('gpu-torch', 'cdui install --gpu cu128');
    await settle();
    expect(usePackStore.getState().restart.phase).toBe('waiting');
    expect(reload).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(RESTART_POLL_MS);
    expect(reload).toHaveBeenCalledTimes(1);

    // The loop is over: no second reload however long the page sits there.
    await vi.advanceTimersByTimeAsync(RESTART_TIMEOUT_MS);
    expect(reload).toHaveBeenCalledTimes(1);
  });

  it('reloads as soon as boot_id changes, even if it never saw the server down', async () => {
    // A restart between two polls answers both times; only the process
    // identity proves the old server is gone.
    api.fetchHealth
      .mockResolvedValueOnce(health({ boot_id: 'boot-a' }))
      .mockResolvedValue(health({ boot_id: 'boot-b' }));

    void usePackStore.getState().restartFlow('gpu-torch', 'cmd');
    await settle();

    expect(reload).toHaveBeenCalledTimes(1);
  });

  it('does not reload while a server with no boot_id keeps answering', async () => {
    api.fetchHealth.mockResolvedValue(health());

    void usePackStore.getState().restartFlow('gpu-torch', 'cmd');
    await settle();
    await vi.advanceTimersByTimeAsync(RESTART_POLL_MS * 3);

    expect(reload).not.toHaveBeenCalled();
  });

  it('gives up as notStarted when nothing ever went down within the grace', async () => {
    api.fetchHealth.mockResolvedValue(health({ boot_id: 'boot-a' }));

    void usePackStore.getState().restartFlow('gpu-torch', 'cmd');
    await settle();
    expect(usePackStore.getState().restart.phase).toBe('waiting');

    await vi.advanceTimersByTimeAsync(RESTART_GRACE_MS + RESTART_POLL_MS);

    expect(usePackStore.getState().restart.phase).toBe('notStarted');
    expect(usePackStore.getState().restart.command).toBe('cmd');
    expect(reload).not.toHaveBeenCalled();
  });

  it('times out when the server goes down and never comes back', async () => {
    api.fetchHealth.mockRejectedValue(new Error('connection refused'));

    void usePackStore.getState().restartFlow('gpu-torch', 'cmd');
    await settle();
    await vi.advanceTimersByTimeAsync(RESTART_TIMEOUT_MS + RESTART_POLL_MS);

    expect(usePackStore.getState().restart.phase).toBe('timeout');
    expect(reload).not.toHaveBeenCalled();
  });

  it('measures its deadlines against the wall clock, not the number of polls', async () => {
    // A sleeping laptop fires no timers. Counting turns would leave the
    // overlay up forever; reading the clock ends it on the very next poll.
    api.fetchHealth.mockRejectedValue(new Error('connection refused'));

    void usePackStore.getState().restartFlow('gpu-torch', 'cmd');
    await settle();
    expect(usePackStore.getState().restart.phase).toBe('waiting');

    vi.setSystemTime(Date.now() + RESTART_TIMEOUT_MS + 1000);
    await vi.advanceTimersByTimeAsync(RESTART_POLL_MS);

    expect(usePackStore.getState().restart.phase).toBe('timeout');
    // Two health calls, not four hundred: the clock did the work.
    expect(api.fetchHealth).toHaveBeenCalledTimes(2);
  });
});

// ── the once-per-page-load check ──────────────────────────────────────────

describe('packStore — checkInProgress', () => {
  it('reads the catalog once per page load', async () => {
    await usePackStore.getState().checkInProgress();
    await usePackStore.getState().checkInProgress();

    expect(api.listPacks).toHaveBeenCalledTimes(1);
  });

  it('says a pack is still installing when it adopts a job', async () => {
    api.listPacks.mockResolvedValue(catalog({
      active_job: { job_id: 'j9', pack_id: 'word-vectors' },
    }));

    await usePackStore.getState().checkInProgress();

    expect(lastToast()).toMatchObject({
      type: 'info',
      message: 'A pack is still installing. Open the Package Center to watch it.',
    });
  });

  it('reports the restart-mode install that finished while the page was gone', async () => {
    sessionStorage.setItem(RESTART_PENDING_KEY, 'gpu-torch');
    api.listPacks.mockResolvedValue(catalog({
      packs: [makePack({ id: 'gpu-torch', title: 'GPU PyTorch' })],
      last_restart_job: { status: 'ok', returncode: 0 },
    }));

    await usePackStore.getState().checkInProgress();

    expect(lastToast()).toMatchObject({
      type: 'success', message: 'Server restarted. GPU PyTorch is ready.',
    });
    // Cleared, so a later reload does not report the same install twice.
    expect(sessionStorage.getItem(RESTART_PENDING_KEY)).toBeNull();
  });

  it('reports a restart-mode install that failed', async () => {
    sessionStorage.setItem(RESTART_PENDING_KEY, 'gpu-torch');
    api.listPacks.mockResolvedValue(catalog({
      packs: [makePack({ id: 'gpu-torch', title: 'GPU PyTorch' })],
      last_restart_job: { status: 'failed', message: 'uv exited 1' },
    }));

    await usePackStore.getState().checkInProgress();

    expect(lastToast()).toMatchObject({ type: 'error' });
    expect(lastToast().message).toBe(
      'The server restarted, but installing GPU PyTorch failed: uv exited 1',
    );
    expect(sessionStorage.getItem(RESTART_PENDING_KEY)).toBeNull();
  });

  it('says nothing when no restart was pending', async () => {
    api.listPacks.mockResolvedValue(catalog({
      last_restart_job: { status: 'ok' },
    }));

    await usePackStore.getState().checkInProgress();

    expect(useToastStore.getState().toasts).toHaveLength(0);
  });
});
