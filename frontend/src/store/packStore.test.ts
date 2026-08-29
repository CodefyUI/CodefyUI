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
  LOG_TAIL_TOAST_CHARS,
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
import { useUIStore } from './uiStore';
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
    // The default every case starts from: a server that has NOT said it can
    // restart itself. The cases about the handshake say so themselves.
    restart_available: false,
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
  // The panel is not this file's store, but one toast action opens it — and
  // an open panel inherited by the next case is a state nothing here set.
  useUIStore.setState({ packCenterOpen: false, packCenterFocusPackId: null });
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

  it('clamps a percent the server reported outside 0..100, both ways', () => {
    // A converter that reports 100.4 % on the last chunk, or a negative
    // remainder mid-stream, must not push a bar past its own track.
    const next = reducePackEvents(seededJob(), eventsPage({
      cursor: 2,
      events: [
        {
          type: 'progress', cursor: 1, ts: 't', item: 'high',
          bytes_done: 10, bytes_total: 10, percent: 140,
        },
        {
          type: 'progress', cursor: 2, ts: 't', item: 'low',
          bytes_done: 0, bytes_total: 10, percent: -8,
        },
      ],
    }));

    expect(next.items.high.percent).toBe(100);
    expect(next.items.low.percent).toBe(0);
  });

  it('closes a still-running step when the job finishes', () => {
    // Every step the backend finishes gets its own `step_done`; a step that
    // ended by raising does not, and a spinner on a job that is over is the
    // one thing the panel must never show.
    const started = reducePackEvents(seededJob(), eventsPage({
      cursor: 1,
      events: [{ type: 'step_started', cursor: 1, ts: 't', step: 'verify', label: 'Verifying' }],
    }));
    expect(started.steps[0].state).toBe('running');

    const next = reducePackEvents(started, eventsPage({
      status: 'done',
      cursor: 2,
      events: [{ type: 'job_done', cursor: 2, ts: 't' }],
    }));

    expect(next.steps).toEqual([{ step: 'verify', label: 'Verifying', state: 'done' }]);
    expect(next.log[next.log.length - 1]).toMatchObject({ kind: 'step', text: 'done' });
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
    // No retry offered: nothing in the event said one was possible.
    expect(next.retryMode).toBeNull();
  });

  it('records the retry mode a stopped live install came back with', () => {
    const next = reducePackEvents(seededJob(), eventsPage({
      status: 'needs_restart',
      cursor: 1,
      events: [{
        type: 'needs_restart', cursor: 1, ts: 't',
        command: 'cdui packs install rag --restart',
        kind: 'pip',
        retry_mode: 'restart',
      }],
    }));
    expect(next.retryMode).toBe('restart');
    expect(next.restartCommand).toBe('cdui packs install rag --restart');
  });

  it('ignores a retry mode this build has no button for', () => {
    // The key names what the SERVER can do. A mode from a newer backend has
    // no handler here, and a button that posts it would end in a 400.
    const next = reducePackEvents(seededJob(), eventsPage({
      status: 'needs_restart',
      cursor: 1,
      events: [{
        type: 'needs_restart', cursor: 1, ts: 't',
        command: 'cdui packs install rag --somehow',
        retry_mode: 'reboot-the-planet',
      }],
    }));
    expect(next.retryMode).toBeNull();
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
  // Faked because adoption starts a follower: its `FOLLOW_IDLE_MS` sleep is a
  // real timer that would otherwise outlive the test that created it.
  beforeEach(() => {
    vi.useFakeTimers();
  });

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
    expect(state.restartAvailable).toBe(false);
    expect(state.loaded).toBe(true);
    expect(state.loading).toBe(false);
    expect(state.unsupported).toBe(false);
    expect(state.error).toBeNull();
  });

  it("takes the server's own word for whether it can restart itself", async () => {
    // Not inferred from `launch_mode`: the server also wants its launcher
    // still on disk and its kill switch off before it says yes, and this
    // flag is what every restart in the UI is gated on.
    api.listPacks.mockResolvedValue(catalog({ restart_available: true }));

    await usePackStore.getState().refresh();

    expect(usePackStore.getState().restartAvailable).toBe(true);
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

  it('does not treat a 503 as a server without a Package Center', async () => {
    // The route EXISTS and refused — the Package Center is switched off, or
    // wedged. Answering that by silently enabling every pack-gated node (as
    // the 404 path does) would hide a state the user can act on.
    api.listPacks.mockRejectedValue(
      new PackApiError(503, 'Package Center is not available'),
    );

    await usePackStore.getState().refresh();

    const state = usePackStore.getState();
    expect(state.unsupported).toBe(false);
    expect(state.error).toBe('Package Center is not available');
    expect(state.loading).toBe(false);
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
    // The follower is started inside `refresh`, so the request is already out.
    expect(api.getPackJobEvents).toHaveBeenCalledTimes(1);
    expect(api.getPackJobEvents.mock.calls[0][0]).toBe('j9');
    expect(api.getPackJobEvents.mock.calls[0][1]).toMatchObject({
      cursor: 0, wait: EVENT_WAIT_S,
    });
    // The server rejects `wait > 60` with a 422, and the follower has no
    // recovery for a request that is malformed on every single turn.
    expect(EVENT_WAIT_S).toBeLessThanOrEqual(60);
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

  it('does not mark a job lost while its follower is still watching it', async () => {
    // `active_job` goes null the moment a job finishes, and the follower's
    // next page carries the real ending. Racing it from here turned a
    // successful install into "lost contact with the server".
    vi.useFakeTimers();
    try {
      // The follower is PARKED on its long poll while the catalog is read,
      // which is the whole point: the answer arrives after it.
      let release: (page: PackJobEventsPage) => void = () => {};
      const parked = new Promise<PackJobEventsPage>((resolve) => { release = resolve; });
      const finished = eventsPage({
        job_id: 'j9', status: 'done', cursor: 1,
        events: [{ type: 'job_done', cursor: 1, ts: 't' }],
      });
      api.getPackJobEvents.mockReturnValueOnce(parked).mockResolvedValue(finished);
      usePackStore.getState().followJob('j9', 'word-vectors', 0);

      await usePackStore.getState().refresh();
      expect(usePackStore.getState().job!.status).toBe('running');

      // ...and the page the follower was waiting for still settles it.
      release(finished);
      await vi.advanceTimersByTimeAsync(0);
      expect(usePackStore.getState().job!.status).toBe('done');
      expect(lastToast()).toMatchObject({ type: 'success' });
    } finally {
      vi.useRealTimers();
    }
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
  // A successful install starts a follower too, and the 409 case adopts one
  // through `refresh`; faked so neither leaves a live idle timer behind.
  beforeEach(() => {
    vi.useFakeTimers();
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

  it('refuses before the request when the PACK is the restart-mode one', async () => {
    // The pre-flight resolves the mode exactly as the server would — the
    // pack's own `install_mode` counts, not just an explicit `opts.mode`. A
    // click on the GPU card under `cdui dev` sends no `mode` at all, and
    // without this it would reach a server that answers 409 after the fact.
    const gpu = makePack({ id: 'gpu-torch', title: 'GPU PyTorch' });
    gpu.install_mode = 'restart';
    usePackStore.setState({
      packs: [gpu], byId: { 'gpu-torch': gpu }, launchMode: 'dev',
    });

    await usePackStore.getState().install('gpu-torch');

    expect(api.installPack).not.toHaveBeenCalled();
    expect(lastToast().message).toBe(
      'This pack needs a server restart, which cdui dev cannot do by itself. '
      + 'Use the command shown in the Package Center.',
    );
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

  it('prints the command when a 409 refuses the install and hands one back', async () => {
    // `RestartUnavailable` — every restart-mode install until the supervisor
    // lands. It answers 409 like a collision does, but with `command` instead
    // of `job_id`, and "another install is already running" would be a plain
    // lie about a server sitting idle.
    const err = new PackApiError(409, 'gpu-torch cannot be installed while the server runs');
    err.body = { detail: 'refused', command: 'cdui install --gpu cu128' };
    api.installPack.mockRejectedValue(err);

    await usePackStore.getState().install('word-vectors', { mode: 'restart' });

    expect(lastToast()).toMatchObject({ type: 'warning' });
    expect(lastToast().message).toBe(
      'This pack cannot be installed from inside the app yet. Run: cdui install --gpu cu128',
    );
    // Not a collision: nothing to adopt, so no catalog re-read is triggered.
    expect(api.listPacks).not.toHaveBeenCalled();
  });

  it('says what to wait for when a graph is what blocks the restart', async () => {
    // Same 409, same `command`, and `needsCli` — "cannot be installed from
    // inside the app yet" — would be false: it can, once the run finishes.
    // `reason` is the only thing that separates the two.
    const err = new PackApiError(409, 'gpu-torch cannot be installed while a graph is running');
    err.body = {
      detail: 'refused',
      reason: 'a graph is running',
      command: 'cdui install --gpu cu128',
    };
    api.installPack.mockRejectedValue(err);

    await usePackStore.getState().install('word-vectors', { mode: 'restart' });

    expect(lastToast()).toMatchObject({ type: 'warning' });
    expect(lastToast().message).toBe('A graph is running. Stop it, then install.');
  });

  it.each([
    'a restart-mode install is already pending',
    'a restart is already pending',
  ])('tells the user to wait when the reason is %s', async (reason) => {
    // Two server-side spellings of one situation: another restart-mode submit
    // colliding with a claim on disk, and a live install arriving while this
    // server is already on its way out. The user's move is the same.
    const err = new PackApiError(409, 'refused');
    err.body = { detail: 'refused', reason, command: 'cdui install --gpu cu128' };
    api.installPack.mockRejectedValue(err);

    await usePackStore.getState().install('word-vectors', { mode: 'restart' });

    expect(lastToast().message).toBe(
      'A restart is already pending. Wait for the server to come back.',
    );
  });

  it.each(['the moon is in the wrong phase', 'toString'])(
    'falls back to the command for the unknown reason %s', async (reason) => {
      // A newer server with a refusal this build has no wording for. The
      // command in the body is always a true way through, so the fallback is
      // wrong-ish rather than misleading — and the user is never stuck.
      // `toString` is the same case wearing a hat: the reason arrives off the
      // wire, and an object-literal lookup would answer it with a function.
      const err = new PackApiError(409, 'refused');
      err.body = { detail: 'refused', reason, command: 'cdui install --gpu cu128' };
      api.installPack.mockRejectedValue(err);

      await usePackStore.getState().install('word-vectors', { mode: 'restart' });

      expect(lastToast().message).toBe(
        'This pack cannot be installed from inside the app yet. Run: cdui install --gpu cu128',
      );
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

  it('settles a restart-mode job that lost its poll into the restart handshake', async () => {
    // A restart-mode install ends by the server closing its own listener
    // half a second after the 202, so the events endpoint is SUPPOSED to
    // stop answering — a LAN client or a tab the browser had throttled can
    // miss that window entirely and see nothing but failures. `lost` would
    // strand the user in front of a banner about a server that is doing
    // exactly what it was asked to, with no overlay and no breadcrumb, so
    // the reloaded page could not even report how the install went.
    usePackStore.setState({
      byId: {
        'word-vectors': makePack({
          id: 'word-vectors', title: 'Word vectors', install_mode: 'restart',
          install_command: 'cdui packs install word-vectors --restart',
        }),
      },
      restartAvailable: true,
    });
    api.getPackJobEvents.mockRejectedValue(new Error('Failed to fetch'));

    usePackStore.getState().followJob('j1', 'word-vectors', 0);
    await settle();
    await vi.advanceTimersByTimeAsync(FOLLOW_RETRY_MS * MAX_FOLLOW_FAILURES);

    expect(api.getPackJobEvents).toHaveBeenCalledTimes(MAX_FOLLOW_FAILURES);
    expect(usePackStore.getState().job!.status).toBe('needs_restart');
    expect(usePackStore.getState().restart.phase).toBe('waiting');
    expect(sessionStorage.getItem(RESTART_PENDING_KEY)).toBe('word-vectors j1');
    // No `needs_restart` event ever arrived, so the command comes off the
    // catalog instead. Both give-up screens promise the user one — the
    // `notStarted` heading is literally "Run this command, then reload:".
    expect(usePackStore.getState().job!.restartCommand)
      .toBe('cdui packs install word-vectors --restart');
    expect(usePackStore.getState().restart.command)
      .toBe('cdui packs install word-vectors --restart');
  });

  it('still loses a restart-mode job whose polls came back as HTTP answers', async () => {
    // A status code is the server ANSWERING. A 404 means "I do not have that
    // job" — it aged out, or the helper never spawned and this server never
    // went anywhere — and settling on it put a blocking, focus-trapping
    // "Server restarting" overlay over a live server for thirty seconds,
    // ending on a screen that promises a command it has no way to know.
    usePackStore.setState({
      byId: {
        'word-vectors': makePack({
          id: 'word-vectors', title: 'Word vectors', install_mode: 'restart',
        }),
      },
      restartAvailable: true,
    });
    api.getPackJobEvents.mockRejectedValue(new PackApiError(404, 'no such job'));

    usePackStore.getState().followJob('j1', 'word-vectors', 0);
    await settle();
    await vi.advanceTimersByTimeAsync(FOLLOW_RETRY_MS * MAX_FOLLOW_FAILURES);

    expect(usePackStore.getState().job!.status).toBe('lost');
    expect(usePackStore.getState().restart.phase).toBe('idle');
    expect(sessionStorage.getItem(RESTART_PENDING_KEY)).toBeNull();
  });

  it('names the command when the lost poll settles on a server that cannot restart',
     async () => {
       // The same settle, on a server that says it cannot relaunch itself.
       // It falls through to `needsCli` — "Run: " — and without a command
       // off the catalog that sentence ends at the colon.
       usePackStore.setState({
         byId: {
           'word-vectors': makePack({
             id: 'word-vectors', title: 'Word vectors', install_mode: 'restart',
             install_command: 'cdui packs install word-vectors --restart',
           }),
         },
         launchMode: 'start',
         restartAvailable: false,
       });
       api.getPackJobEvents.mockRejectedValue(new Error('Failed to fetch'));

       usePackStore.getState().followJob('j1', 'word-vectors', 0);
       await settle();
       await vi.advanceTimersByTimeAsync(FOLLOW_RETRY_MS * MAX_FOLLOW_FAILURES);

       expect(usePackStore.getState().restart.phase).toBe('idle');
       expect(lastToast()).toMatchObject({ type: 'warning' });
       expect(lastToast().message)
         .toContain('cdui packs install word-vectors --restart');
     });

  it('still loses a LIVE job whose poll never came back', async () => {
    // The mode is the whole difference: nothing is restarting here, so
    // "we no longer know what this is doing" remains the honest answer.
    usePackStore.setState({ restartAvailable: true });
    api.getPackJobEvents.mockRejectedValue(new Error('Failed to fetch'));

    usePackStore.getState().followJob('j1', 'word-vectors', 0);
    await settle();
    await vi.advanceTimersByTimeAsync(FOLLOW_RETRY_MS * MAX_FOLLOW_FAILURES);

    expect(usePackStore.getState().job!.status).toBe('lost');
    expect(usePackStore.getState().restart.phase).toBe('idle');
    expect(sessionStorage.getItem(RESTART_PENDING_KEY)).toBeNull();
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

  it('resumes from the cursor already applied when a stopped job is re-adopted', async () => {
    // Re-adoption is not a fresh start: replaying from 0 would re-fetch every
    // event of an install that is already minutes in, and only the reducer's
    // cursor guard would stop the log from doubling.
    api.getPackJobEvents.mockResolvedValue(eventsPage({
      job_id: 'j9',
      status: 'running',
      cursor: 7,
      events: [{ type: 'log', cursor: 7, ts: 't', line: 'seven' }],
    }));
    usePackStore.getState().followJob('j9', 'word-vectors', 0);
    await settle();
    expect(usePackStore.getState().job!.cursor).toBe(7);

    usePackStore.getState().stopFollowing();
    api.getPackJobEvents.mockClear();
    api.listPacks.mockResolvedValue(catalog({
      active_job: { job_id: 'j9', pack_id: 'word-vectors' },
    }));

    await usePackStore.getState().refresh();

    expect(api.getPackJobEvents.mock.calls[0][1]).toMatchObject({ cursor: 7 });
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

    // This build's own copy for the pack, not the server's English title:
    // one pack has ONE name on the node, on the card and in this toast.
    expect(lastToast()).toMatchObject({
      type: 'success', message: 'Word vectors (GloVe) installed.',
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
    // A RESTART-mode job on a server that says it can restart: the two halves
    // of the gate. The mode is what makes `needs_restart` an ending somebody
    // asked for; the flag is the server promising to come back.
    usePackStore.setState({
      byId: {
        'word-vectors': makePack({
          id: 'word-vectors', title: 'Word vectors', install_mode: 'restart',
        }),
      },
      restartAvailable: true,
    });
    terminal({
      status: 'needs_restart',
      events: [{
        type: 'needs_restart', cursor: 1, ts: 't',
        command: 'cdui install --gpu cu128',
      }],
    });

    usePackStore.getState().followJob('j1', 'word-vectors', 0);
    await settle();

    // The job beside the pack: the page that comes back reads
    // `last_restart_job`, which is not bounded by age, so it needs to know
    // WHICH install it is being told about.
    expect(sessionStorage.getItem(RESTART_PENDING_KEY)).toBe('word-vectors j1');
    expect(usePackStore.getState().restart).toMatchObject({
      phase: 'waiting', packId: 'word-vectors', command: 'cdui install --gpu cu128',
    });
  });

  it('does not start a handshake for a LIVE job that ends needs_restart', async () => {
    // The only way a live install lands here: its resolver conflicted with
    // the constraints file, so the server it was talking to is not going
    // anywhere. Under `cdui start` this used to raise a blocking "Server
    // restarting" overlay for thirty seconds — over a running server, and
    // on top of the command the user actually needs.
    //
    // The server CAN restart here, deliberately: the job's mode is the half
    // of the gate this case is about, and a restart it can perform is not a
    // restart anybody asked for.
    usePackStore.setState({ restartAvailable: true });
    terminal({
      status: 'needs_restart',
      events: [{
        type: 'needs_restart', cursor: 1, ts: 't',
        command: 'cdui packs install word-vectors --restart',
      }],
    });

    usePackStore.getState().followJob('j1', 'word-vectors', 0);
    await settle();

    expect(usePackStore.getState().job!.mode).toBe('live');
    expect(usePackStore.getState().restart.phase).toBe('idle');
    expect(sessionStorage.getItem(RESTART_PENDING_KEY)).toBeNull();
    expect(lastToast()).toMatchObject({ type: 'warning' });
    expect(lastToast().message).toContain('cdui packs install word-vectors --restart');
    // The job stays on screen: its banner is what renders the command.
    expect(usePackStore.getState().job!.status).toBe('needs_restart');
  });

  it('points a stopped live install at the Package Center when a restart can finish it', async () => {
    // The other half of the case above. The live install stopped at the
    // constraints file, and the server it was talking to CAN restart itself,
    // which is exactly what would finish it: `needsCli` — "cannot be
    // installed from inside the app" — is a flat lie about a server two
    // clicks away from doing it. The retry mode is the server's own offer,
    // and the toast carries the click that reaches it.
    useUIStore.setState({ packCenterOpen: false, packCenterFocusPackId: null });
    usePackStore.setState({ restartAvailable: true });
    terminal({
      status: 'needs_restart',
      events: [{
        type: 'needs_restart', cursor: 1, ts: 't',
        command: 'cdui packs install word-vectors --restart',
        kind: 'pip',
        retry_mode: 'restart',
      }],
    });

    usePackStore.getState().followJob('j1', 'word-vectors', 0);
    await settle();

    expect(usePackStore.getState().job!.mode).toBe('live');
    expect(usePackStore.getState().job!.retryMode).toBe('restart');
    // Still no handshake and no breadcrumb: this server is not going
    // anywhere until the user asks it to.
    expect(usePackStore.getState().restart.phase).toBe('idle');
    expect(sessionStorage.getItem(RESTART_PENDING_KEY)).toBeNull();
    expect(lastToast()).toMatchObject({
      type: 'warning',
      message:
        'The install stopped at a package the server has loaded. Open the Package Center to restart the server and finish it.',
    });
    // ...and the button that gets there, on the pack the toast is about.
    expect(lastToast().action?.label).toBe('Open Package Center');
    lastToast().action!.onClick();
    expect(useUIStore.getState().packCenterOpen).toBe(true);
    expect(useUIStore.getState().packCenterFocusPackId).toBe('word-vectors');
  });

  /** A restart-mode job for `word-vectors`, so the mode half of the gate holds. */
  function restartModePack(): void {
    usePackStore.setState({
      byId: {
        'word-vectors': makePack({
          id: 'word-vectors', title: 'Word vectors', install_mode: 'restart',
        }),
      },
    });
  }

  it('shows the command instead of restarting when the server cannot', async () => {
    // Started with `cdui start` — so the launch mode alone would have said
    // yes — and the server still says no: its launcher has been moved off
    // disk, or the kill switch is thrown. This is the half the launch mode
    // used to stand in for, and the server is the one that knows.
    restartModePack();
    usePackStore.setState({ launchMode: 'start', restartAvailable: false });
    terminal({
      status: 'needs_restart',
      events: [{ type: 'needs_restart', cursor: 1, ts: 't', command: 'cdui install --gpu cu128' }],
    });

    usePackStore.getState().followJob('j1', 'word-vectors', 0);
    await settle();

    expect(usePackStore.getState().job!.mode).toBe('restart');
    expect(sessionStorage.getItem(RESTART_PENDING_KEY)).toBeNull();
    expect(usePackStore.getState().restart.phase).toBe('idle');
    expect(lastToast()).toMatchObject({ type: 'warning' });
    // The command, because nothing else here can finish this install.
    expect(lastToast().message).toContain('cdui install --gpu cu128');
    // The job stays on screen so the command block has something to describe.
    expect(usePackStore.getState().job!.restartCommand).toBe('cdui install --gpu cu128');
  });

  it('names cdui dev when a restart-mode job ends on a server that reloads in place', async () => {
    // `cdui dev` has no supervisor to relaunch it, so no catalog it serves
    // will ever report `restart_available`. Its own sentence rather than the
    // generic one: "not from inside the app yet" reads as a missing feature,
    // when the answer is "not the way you started this server". The command
    // is not lost — the sentence points at the banner, which is rendering it.
    restartModePack();
    usePackStore.setState({ launchMode: 'dev', restartAvailable: false });
    terminal({
      status: 'needs_restart',
      events: [{ type: 'needs_restart', cursor: 1, ts: 't', command: 'cdui install --gpu cu128' }],
    });

    usePackStore.getState().followJob('j1', 'word-vectors', 0);
    await settle();

    expect(sessionStorage.getItem(RESTART_PENDING_KEY)).toBeNull();
    expect(usePackStore.getState().restart.phase).toBe('idle');
    expect(lastToast()).toMatchObject({
      type: 'warning',
      message:
        'This pack needs a server restart, which cdui dev cannot do by itself. Use the command shown in the Package Center.',
    });
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

  it('reports the reason when the remove request itself fails', async () => {
    api.removePackItem.mockRejectedValue(new PackApiError(500, 'permission denied'));

    await usePackStore.getState().removeItem('word-vectors', 'glove');

    // The separator lives in the locale string, not in a template literal:
    // zh-TW wants a full-width colon here.
    expect(lastToast()).toMatchObject({
      type: 'error', message: 'Could not remove glove: permission denied',
    });
    expect(api.listPacks).toHaveBeenCalled();
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

  it('counts an outage it saw on the very first read', async () => {
    // The supervisor can win the race: by the time the flow reads health for
    // the boot id, the old process is already gone. Waiting for a SECOND
    // outage that has no reason to happen would call a restart that worked
    // a no-show thirty seconds later.
    api.fetchHealth
      .mockRejectedValueOnce(new Error('connection refused'))
      .mockResolvedValue(health({ boot_id: 'boot-b' }));

    void usePackStore.getState().restartFlow('gpu-torch', 'cmd');
    await settle();

    expect(reload).toHaveBeenCalledTimes(1);
    expect(usePackStore.getState().restart.phase).toBe('waiting');
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
    sessionStorage.setItem(RESTART_PENDING_KEY, 'gpu-torch');

    void usePackStore.getState().restartFlow('gpu-torch', 'cmd');
    await settle();
    expect(usePackStore.getState().restart.phase).toBe('waiting');

    await vi.advanceTimersByTimeAsync(RESTART_GRACE_MS + RESTART_POLL_MS);

    expect(usePackStore.getState().restart.phase).toBe('notStarted');
    expect(usePackStore.getState().restart.command).toBe('cmd');
    expect(reload).not.toHaveBeenCalled();
    // The breadcrumb is for a page that comes back from an AUTOMATIC
    // reload. Left behind, it would toast an outcome read off some other
    // attempt's record the next time the user reloads by hand.
    expect(sessionStorage.getItem(RESTART_PENDING_KEY)).toBeNull();
  });

  it('times out when the server goes down and never comes back', async () => {
    api.fetchHealth.mockRejectedValue(new Error('connection refused'));
    sessionStorage.setItem(RESTART_PENDING_KEY, 'gpu-torch');

    void usePackStore.getState().restartFlow('gpu-torch', 'cmd');
    await settle();
    await vi.advanceTimersByTimeAsync(RESTART_TIMEOUT_MS + RESTART_POLL_MS);

    expect(usePackStore.getState().restart.phase).toBe('timeout');
    expect(reload).not.toHaveBeenCalled();
    // See the notStarted branch: a stale breadcrumb outlives the handshake
    // that wrote it and reports on the wrong attempt.
    expect(sessionStorage.getItem(RESTART_PENDING_KEY)).toBeNull();
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
  // Faked for the same reason as the refresh block: adopting a job starts a
  // follower whose idle sleep must not outlive the test.
  beforeEach(() => {
    vi.useFakeTimers();
  });

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

  it('reports the outcome only when it belongs to the job the breadcrumb named',
     async () => {
       // `last_restart_job` has no age bound — the record of an install that
       // finished an hour ago is still the one this reads. A lost poll during
       // a network blip can end in a reload with no restart behind it at all,
       // and reporting that stale record would tell the user an install
       // succeeded that never ran.
       sessionStorage.setItem(RESTART_PENDING_KEY, 'gpu-torch abc');
       api.listPacks.mockResolvedValue(catalog({
         packs: [makePack({ id: 'gpu-torch', title: 'GPU PyTorch' })],
         last_restart_job: { job_id: 'xyz', status: 'ok', returncode: 0 },
       }));

       await usePackStore.getState().checkInProgress();

       expect(useToastStore.getState().toasts).toHaveLength(0);
       // Still cleared: this breadcrumb has no outcome coming, and leaving it
       // would make the next hand reload read the same wrong record.
       expect(sessionStorage.getItem(RESTART_PENDING_KEY)).toBeNull();
     });

  it('reports the outcome when the ids agree', async () => {
    sessionStorage.setItem(RESTART_PENDING_KEY, 'gpu-torch abc');
    api.listPacks.mockResolvedValue(catalog({
      packs: [makePack({ id: 'gpu-torch', title: 'GPU PyTorch' })],
      last_restart_job: { job_id: 'abc', status: 'ok', returncode: 0 },
    }));

    await usePackStore.getState().checkInProgress();

    expect(lastToast()).toMatchObject({
      type: 'success', message: 'Server restarted. GPU PyTorch is ready.',
    });
  });

  it('reports a restart-mode install that failed', async () => {
    sessionStorage.setItem(RESTART_PENDING_KEY, 'gpu-torch');
    api.listPacks.mockResolvedValue(catalog({
      packs: [makePack({ id: 'gpu-torch', title: 'GPU PyTorch' })],
      last_restart_job: {
        status: 'failed',
        message: 'uv exited 1',
        // Present, and deliberately not shown: the message is the story.
        log_tail: 'Resolved 41 packages\nERROR: uv exited 1\n',
      },
    }));

    await usePackStore.getState().checkInProgress();

    expect(lastToast()).toMatchObject({ type: 'error' });
    expect(lastToast().message).toBe(
      'The server restarted, but installing GPU PyTorch failed: uv exited 1',
    );
    // One toast: the message is the whole story, and the log tail behind it
    // would only repeat it at ten times the length.
    expect(useToastStore.getState().toasts).toHaveLength(1);
    expect(sessionStorage.getItem(RESTART_PENDING_KEY)).toBeNull();
  });

  it('falls back to the log tail when the failure came back with no message', async () => {
    // The helper died with the old server and took its job log with it, so
    // this record is all that survived. A bare "failed:" with nothing after
    // the colon is the one report worth spending a second toast on — and an
    // `error` toast, which this app never auto-dismisses, is still there when
    // the user comes back to the tab.
    sessionStorage.setItem(RESTART_PENDING_KEY, 'gpu-torch');
    api.listPacks.mockResolvedValue(catalog({
      packs: [makePack({ id: 'gpu-torch', title: 'GPU PyTorch' })],
      last_restart_job: {
        status: 'failed',
        message: '',
        log_tail: 'ERROR: ResolutionImpossible\n',
      },
    }));

    await usePackStore.getState().checkInProgress();

    const toasts = useToastStore.getState().toasts;
    expect(toasts).toHaveLength(2);
    expect(toasts[1]).toMatchObject({
      type: 'error',
      message: 'Last output from the installer: ERROR: ResolutionImpossible',
    });
  });

  it('says nothing extra when a failure has neither a message nor a log', async () => {
    sessionStorage.setItem(RESTART_PENDING_KEY, 'gpu-torch');
    api.listPacks.mockResolvedValue(catalog({
      packs: [makePack({ id: 'gpu-torch', title: 'GPU PyTorch' })],
      last_restart_job: { status: 'failed', log_tail: '   ' },
    }));

    await usePackStore.getState().checkInProgress();

    expect(useToastStore.getState().toasts).toHaveLength(1);
  });

  it('bounds the log tail it puts in a toast, keeping the end', async () => {
    // The end is where an installer says how it went; the beginning is where
    // it lists what it downloaded. A toast has no scroll of its own, so an
    // unbounded tail would grow one card until it covered the canvas.
    sessionStorage.setItem(RESTART_PENDING_KEY, 'gpu-torch');
    api.listPacks.mockResolvedValue(catalog({
      packs: [makePack({ id: 'gpu-torch', title: 'GPU PyTorch' })],
      last_restart_job: {
        status: 'failed',
        message: '',
        log_tail: `${'x'.repeat(5000)}the last line`,
      },
    }));

    await usePackStore.getState().checkInProgress();

    const shown = lastToast().message;
    expect(shown.endsWith('the last line')).toBe(true);
    expect(shown.length).toBeLessThanOrEqual(
      'Last output from the installer: '.length + LOG_TAIL_TOAST_CHARS,
    );
  });

  it('tries again after a boot read that never got an answer', async () => {
    // The once-per-load flag exists to stop two same-tick mounts both
    // fetching, not to make one dropped packet permanent: `refresh` leaves
    // `loaded` false on a network error precisely so a later mount retries.
    api.listPacks.mockRejectedValueOnce(new Error('Failed to fetch'));

    await usePackStore.getState().checkInProgress();
    expect(usePackStore.getState().loaded).toBe(false);

    api.listPacks.mockResolvedValue(catalog({ packs: [makePack({ id: 'rag' })] }));
    await usePackStore.getState().checkInProgress();

    expect(api.listPacks).toHaveBeenCalledTimes(2);
    expect(usePackStore.getState().loaded).toBe(true);
    expect(usePackStore.getState().packs.map((pack) => pack.id)).toEqual(['rag']);
  });

  it('keeps the restart breadcrumb when the boot read fails, and reports it on the retry', async () => {
    // The outcome is read off `last_restart_job`, which only a catalog that
    // ARRIVED can carry. Consuming the key against no catalog would swallow
    // the one report a user who just sat through a restart is waiting for.
    sessionStorage.setItem(RESTART_PENDING_KEY, 'gpu-torch');
    api.listPacks.mockRejectedValueOnce(new Error('Failed to fetch'));

    await usePackStore.getState().checkInProgress();
    expect(sessionStorage.getItem(RESTART_PENDING_KEY)).toBe('gpu-torch');
    expect(useToastStore.getState().toasts).toHaveLength(0);

    api.listPacks.mockResolvedValue(catalog({
      packs: [makePack({ id: 'gpu-torch', title: 'GPU PyTorch' })],
      last_restart_job: { status: 'ok' },
    }));
    await usePackStore.getState().checkInProgress();

    expect(lastToast()).toMatchObject({
      type: 'success', message: 'Server restarted. GPU PyTorch is ready.',
    });
    expect(sessionStorage.getItem(RESTART_PENDING_KEY)).toBeNull();
  });

  it('says nothing when no restart was pending', async () => {
    api.listPacks.mockResolvedValue(catalog({
      last_restart_job: { status: 'ok' },
    }));

    await usePackStore.getState().checkInProgress();

    expect(useToastStore.getState().toasts).toHaveLength(0);
  });

  it('keeps the restart breadcrumb for a server with no Package Center', async () => {
    // A 404 is a definite answer about the catalog and no answer at all about
    // the restart: the outcome is read off `last_restart_job`, which rides on
    // a catalog this build does not serve. Consuming the key here would report
    // nothing and then guarantee that nothing is ever reported.
    sessionStorage.setItem(RESTART_PENDING_KEY, 'gpu-torch');
    api.listPacks.mockRejectedValue(new PackApiError(404, 'Not Found'));

    await usePackStore.getState().checkInProgress();

    expect(usePackStore.getState().unsupported).toBe(true);
    expect(sessionStorage.getItem(RESTART_PENDING_KEY)).toBe('gpu-torch');
    expect(useToastStore.getState().toasts).toHaveLength(0);

    // And it is still there for the first server that CAN answer.
    _resetPackStoreForTesting();
    api.listPacks.mockResolvedValue(catalog({
      packs: [makePack({ id: 'gpu-torch', title: 'GPU PyTorch' })],
      last_restart_job: { status: 'ok' },
    }));
    await usePackStore.getState().checkInProgress();

    expect(lastToast()).toMatchObject({
      type: 'success', message: 'Server restarted. GPU PyTorch is ready.',
    });
    expect(sessionStorage.getItem(RESTART_PENDING_KEY)).toBeNull();
  });
});
