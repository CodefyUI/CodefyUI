import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import type { JobEvent, JobEventsPage } from '../api/rest';
import {
  FOLLOW_IDLE_MS,
  FOLLOW_RETRY_MS,
  EVENT_WAIT_S,
  MAX_FOLLOW_FAILURES,
  MAX_LOG_LINES,
  createJobFollower,
  emptyJob,
  reduceJobEvents,
  type Job,
  type JobFollowerOptions,
} from './jobFollower';

/**
 * A domain job, the way `PackJob` and `PluginJob` are: the generic shape plus
 * a field this module knows nothing about. It is here so the tests can prove
 * that a fold preserves the extra field and that `onExtra` can write it.
 */
interface TestJob extends Job {
  tag: string | null;
}

function makeJob(partial: Partial<TestJob> = {}): TestJob {
  return { ...emptyJob('j1'), tag: null, ...partial };
}

function page(partial: Partial<JobEventsPage> = {}): JobEventsPage {
  return { job_id: 'j1', status: 'running', events: [], cursor: 0, ...partial };
}

// ── the pure reducer ──────────────────────────────────────────────────────

describe('reduceJobEvents', () => {
  it('records steps in order and marks the previous one done when the next starts', () => {
    const next = reduceJobEvents(makeJob(), page({
      cursor: 2,
      events: [
        { type: 'step_started', cursor: 1, ts: 't', step: 'download', label: 'Downloading' },
        { type: 'step_started', cursor: 2, ts: 't', step: 'verify', label: 'Verifying' },
      ],
    }));

    expect(next.steps).toEqual([
      { step: 'download', label: 'Downloading', state: 'done' },
      { step: 'verify', label: 'Verifying', state: 'running' },
    ]);
    expect(next.log.map((entry) => entry.text)).toEqual(['Downloading', 'Verifying']);
  });

  it('falls back to the step id when the server sent no label', () => {
    const next = reduceJobEvents(makeJob(), page({
      cursor: 1,
      events: [{ type: 'step_started', cursor: 1, ts: 't', step: 'deps' }],
    }));

    expect(next.steps).toEqual([{ step: 'deps', label: 'deps', state: 'running' }]);
  });

  it('marks a step done on its own step_done event', () => {
    const started = reduceJobEvents(makeJob(), page({
      cursor: 1,
      events: [{ type: 'step_started', cursor: 1, ts: 't', step: 'deps', label: 'Deps' }],
    }));
    const next = reduceJobEvents(started, page({
      cursor: 2,
      events: [{ type: 'step_done', cursor: 2, ts: 't', step: 'deps' }],
    }));

    expect(next.steps).toEqual([{ step: 'deps', label: 'Deps', state: 'done' }]);
    // `step_done` is bookkeeping, not something to print: the line that named
    // the step was already written when it started.
    expect(next.log).toHaveLength(1);
  });

  it('drops an empty log line instead of printing a blank row', () => {
    const next = reduceJobEvents(makeJob(), page({
      cursor: 3,
      events: [
        { type: 'log', cursor: 1, ts: 't', line: '' },
        { type: 'log', cursor: 2, ts: 't', line: '   ' },
        { type: 'log', cursor: 3, ts: 't', line: 'real' },
      ],
    }));

    expect(next.log.map((entry) => entry.text)).toEqual(['real']);
  });

  it('keeps per-item byte progress and never turns it into a log line', () => {
    const next = reduceJobEvents(makeJob(), page({
      cursor: 2,
      events: [
        {
          type: 'progress', cursor: 1, ts: 't',
          item: 'wheel', bytes_done: 25, bytes_total: 100,
        },
        {
          type: 'progress', cursor: 2, ts: 't',
          item: 'wheel', bytes_done: 50, bytes_total: 100,
        },
      ],
    }));

    expect(next.items).toEqual({
      wheel: { bytesDone: 50, bytesTotal: 100, percent: 50 },
    });
    expect(next.log).toEqual([]);
  });

  it('leaves the percent unknown when there is no total to divide by', () => {
    const next = reduceJobEvents(makeJob(), page({
      cursor: 1,
      events: [{ type: 'progress', cursor: 1, ts: 't', item: 'wheel', bytes_done: 9 }],
    }));

    expect(next.items.wheel).toEqual({ bytesDone: 9, bytesTotal: null, percent: null });
  });

  it('clamps a percent the server reported outside 0..100, both ways', () => {
    const next = reduceJobEvents(makeJob(), page({
      cursor: 2,
      events: [
        { type: 'progress', cursor: 1, ts: 't', item: 'a', bytes_done: 0, percent: 140 },
        { type: 'progress', cursor: 2, ts: 't', item: 'b', bytes_done: 0, percent: -3 },
      ],
    }));

    expect(next.items.a.percent).toBe(100);
    expect(next.items.b.percent).toBe(0);
  });

  it('closes a still-running step when the job finishes', () => {
    const started = reduceJobEvents(makeJob(), page({
      cursor: 1,
      events: [{ type: 'step_started', cursor: 1, ts: 't', step: 'deps', label: 'Deps' }],
    }));
    const next = reduceJobEvents(started, page({
      status: 'done',
      cursor: 2,
      events: [{ type: 'job_done', cursor: 2, ts: 't' }],
    }));

    expect(next.status).toBe('done');
    expect(next.steps.every((step) => step.state === 'done')).toBe(true);
    expect(next.log[next.log.length - 1]).toMatchObject({ kind: 'step', text: 'done' });
  });

  it('captures the job_failed message and hint', () => {
    const next = reduceJobEvents(makeJob(), page({
      status: 'failed',
      cursor: 1,
      events: [{
        type: 'job_failed', cursor: 1, ts: 't',
        message: 'uv exited 1', hint: 'check the network',
      }],
    }));

    expect(next.status).toBe('failed');
    expect(next.error).toEqual({ message: 'uv exited 1', hint: 'check the network' });
    expect(next.log[0]).toMatchObject({ kind: 'error', text: 'uv exited 1' });
  });

  it('stores the restart command so a banner can print it', () => {
    const next = reduceJobEvents(makeJob(), page({
      status: 'needs_restart',
      cursor: 1,
      events: [{
        type: 'needs_restart', cursor: 1, ts: 't', command: 'cdui install --gpu cu128',
      }],
    }));

    expect(next.status).toBe('needs_restart');
    expect(next.restartCommand).toBe('cdui install --gpu cu128');
    // A `needs_restart` event is not a transcript line either: the command
    // has a block of its own to live in.
    expect(next.log).toEqual([]);
  });

  it('skips an unknown event type rather than rendering a mystery line', () => {
    const next = reduceJobEvents(makeJob(), page({
      cursor: 2,
      events: [
        { type: 'quantum_entangled', cursor: 1, ts: 't', payload: 42 },
        { type: 'log', cursor: 2, ts: 't', line: 'after' },
      ],
    }));

    expect(next.log.map((entry) => entry.text)).toEqual(['after']);
  });

  it('skips events at or below the cursor already applied', () => {
    const first = reduceJobEvents(makeJob(), page({
      cursor: 2,
      events: [
        { type: 'log', cursor: 1, ts: 't', line: 'one' },
        { type: 'log', cursor: 2, ts: 't', line: 'two' },
      ],
    }));
    // The same page again — a reconnect that resumed from the wrong cursor.
    const replayed = reduceJobEvents(first, page({
      cursor: 3,
      events: [
        { type: 'log', cursor: 1, ts: 't', line: 'one' },
        { type: 'log', cursor: 2, ts: 't', line: 'two' },
        { type: 'log', cursor: 3, ts: 't', line: 'three' },
      ],
    }));

    expect(replayed.log.map((entry) => entry.text)).toEqual(['one', 'two', 'three']);
  });

  it('caps the log at MAX_LOG_LINES, keeping the newest', () => {
    const events: JobEvent[] = [];
    for (let index = 1; index <= MAX_LOG_LINES + 20; index += 1) {
      events.push({ type: 'log', cursor: index, ts: 't', line: `line ${index}` });
    }
    const next = reduceJobEvents(makeJob(), page({ cursor: events.length, events }));

    expect(next.log).toHaveLength(MAX_LOG_LINES);
    expect(next.log[0].text).toBe('line 21');
    expect(next.log[next.log.length - 1].text).toBe(`line ${MAX_LOG_LINES + 20}`);
  });

  it('takes the status from the page and never moves the cursor backwards', () => {
    const applied = reduceJobEvents(makeJob({ cursor: 9 }), page({ status: 'done', cursor: 4 }));

    expect(applied.status).toBe('done');
    expect(applied.cursor).toBe(9);
  });

  it('gives every log line a distinct seq so React keys never collide', () => {
    const next = reduceJobEvents(makeJob(), page({
      cursor: 2,
      events: [
        // No cursor at all: the line still needs a key nothing else holds.
        { type: 'log', ts: 't', line: 'first' },
        { type: 'log', cursor: 1, ts: 't', line: 'second' },
        { type: 'log', cursor: 2, ts: 't', line: 'third' },
      ],
    }));

    const seqs = next.log.map((entry) => entry.seq);
    expect(new Set(seqs).size).toBe(seqs.length);
  });

  it('carries a domain field through untouched and lets onExtra write it', () => {
    const next = reduceJobEvents<TestJob>(
      makeJob({ tag: 'kept' }),
      page({
        cursor: 2,
        events: [
          { type: 'log', cursor: 1, ts: 't', line: 'noise' },
          { type: 'needs_restart', cursor: 2, ts: 't', command: 'cdui start', flavour: 'gpu' },
        ],
      }),
      (event, draft) => {
        if (event.type === 'needs_restart') draft.tag = String(event.flavour);
      },
    );

    expect(next.tag).toBe('gpu');
    expect(next.restartCommand).toBe('cdui start');
    expect(next.log.map((entry) => entry.text)).toEqual(['noise']);
  });

  it('does not let onExtra clobber the fields the reducer owns', () => {
    const job = makeJob({ tag: null });
    const next = reduceJobEvents<TestJob>(
      job,
      page({ cursor: 1, events: [{ type: 'log', cursor: 1, ts: 't', line: 'kept' }] }),
      (_event, draft) => {
        draft.log = [];
        draft.cursor = -5;
        // The identity fields too: the follower matches every page against
        // `jobId`, so a fold that renamed the job would strand it — nothing
        // would patch it again and nothing would settle it.
        draft.jobId = 'hijacked';
        draft.startedAt = 0;
      },
    );

    expect(next.log.map((entry) => entry.text)).toEqual(['kept']);
    expect(next.cursor).toBe(1);
    expect(next.jobId).toBe('j1');
    expect(next.startedAt).toBe(job.startedAt);
  });

  it('skips an onExtra call for an event the cursor guard dropped', () => {
    const seen: string[] = [];
    reduceJobEvents<TestJob>(
      makeJob({ cursor: 2 }),
      page({
        cursor: 3,
        events: [
          { type: 'log', cursor: 1, ts: 't', line: 'old' },
          { type: 'log', cursor: 3, ts: 't', line: 'new' },
        ],
      }),
      (event) => { seen.push(String(event.line)); },
    );

    expect(seen).toEqual(['new']);
  });

  it('returns the job untouched when the page carried nothing', () => {
    const job = makeJob({ cursor: 4 });
    const next = reduceJobEvents(job, page({ cursor: 4 }));

    // Same arrays, not copies: an empty poll must not re-render every
    // subscriber that memoises on the log.
    expect(next.log).toBe(job.log);
    expect(next.steps).toBe(job.steps);
    expect(next.items).toBe(job.items);
  });
});

// ── the long-poll follower ────────────────────────────────────────────────

describe('createJobFollower', () => {
  /**
   * Timers are faked throughout: the loop's only real waits are its idle and
   * retry sleeps, and asserting "it did NOT poll again" by sleeping 10 ms of
   * wall clock proves nothing when the next turn was 500 ms away regardless.
   */
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  /** Let queued microtasks (a resolved fetch, a patch) run to completion. */
  const settle = () => vi.advanceTimersByTimeAsync(0);

  /**
   * One follower over a one-field store, wired the way a real store wires it:
   * `patchJob` writes only when the id still matches the open job.
   */
  function bench(options: Partial<JobFollowerOptions<TestJob>> = {}) {
    let open: TestJob | null = makeJob();

    // The default answer is a running, empty page, so a test that forgets to
    // arrange one parks on the idle timer instead of settling.
    const fetchPage = vi.fn(async (
      _jobId: string, _cursor: number, _signal: AbortSignal, _waitS: number,
    ): Promise<JobEventsPage> => page());
    const patchJob = vi.fn((jobId: string, next: TestJob) => {
      if (open !== null && open.jobId === jobId) open = next;
    });
    const onSettled = vi.fn();

    const follower = createJobFollower<TestJob>({
      fetchPage, getOpenJob: () => open, patchJob, onSettled, ...options,
    });

    // Read off the recorded calls rather than from inside the implementation:
    // a test that installs its own `mockImplementation` would otherwise
    // silently stop recording.
    const args = () => fetchPage.mock.calls;
    return {
      follower, fetchPage, patchJob, onSettled,
      cursors: () => args().map((call) => call[1]),
      signals: () => args().map((call) => call[2]),
      waits: () => args().map((call) => call[3]),
      job: () => open,
      setJob: (next: TestJob | null) => { open = next; },
    };
  }

  it('folds pages into the job and stops once terminal and the cursor stops moving', async () => {
    const app = bench();
    app.fetchPage
      .mockResolvedValueOnce(page({
        cursor: 2,
        events: [
          { type: 'step_started', cursor: 1, ts: 't', step: 'deps', label: 'Deps' },
          { type: 'log', cursor: 2, ts: 't', line: 'uv pip install' },
        ],
      }))
      .mockResolvedValue(page({
        status: 'done',
        cursor: 3,
        events: [{ type: 'job_done', cursor: 3, ts: 't' }],
      }));

    app.follower.start('j1', 0);
    await settle();

    const job = app.job()!;
    expect(job.status).toBe('done');
    expect(job.cursor).toBe(3);
    // The terminal page is replayed once; its event is applied exactly once.
    expect(job.log.filter((line) => line.text === 'done')).toHaveLength(1);
    expect(app.fetchPage).toHaveBeenCalledTimes(3);
    expect(app.onSettled).toHaveBeenCalledTimes(1);
    expect(app.onSettled).toHaveBeenCalledWith('j1', 'done', job);
    expect(app.follower.followingJobId()).toBeNull();

    await vi.advanceTimersByTimeAsync(60_000);
    expect(app.fetchPage).toHaveBeenCalledTimes(3);
  });

  it('folds with the reducer the domain passed instead of the default', async () => {
    // How a center that reads keys of its own gets them read on a LIVE page
    // and not only in its unit tests: one composed reducer, passed once.
    const reduce = vi.fn((job: TestJob, folded: JobEventsPage): TestJob => ({
      ...reduceJobEvents(job, folded), tag: 'domain',
    }));
    const app = bench({ reduce });
    app.fetchPage.mockResolvedValue(page({
      status: 'done', cursor: 1,
      events: [{ type: 'log', cursor: 1, ts: 't', line: 'one' }],
    }));

    app.follower.start('j1', 0);
    await settle();

    expect(reduce).toHaveBeenCalled();
    expect(app.job()!.tag).toBe('domain');
    expect(app.job()!.log.map((entry) => entry.text)).toEqual(['one']);
  });

  it('resumes each request from the cursor already applied, parking for waitS', async () => {
    const app = bench();
    app.fetchPage.mockResolvedValue(page({
      cursor: 5, events: [{ type: 'log', cursor: 5, ts: 't', line: 'five' }],
    }));

    app.follower.start('j1', 2);
    await settle();
    await vi.advanceTimersByTimeAsync(FOLLOW_IDLE_MS + 10);

    expect(app.cursors().slice(0, 2)).toEqual([2, 5]);
    expect(app.waits()[0]).toBe(EVENT_WAIT_S);
  });

  it('idles for FOLLOW_IDLE_MS when a running page returned nothing', async () => {
    const app = bench();

    app.follower.start('j1', 0);
    await settle();
    expect(app.fetchPage).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(FOLLOW_IDLE_MS - 1);
    expect(app.fetchPage).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(2);
    expect(app.fetchPage).toHaveBeenCalledTimes(2);
  });

  it('asks again immediately while the cursor keeps moving', async () => {
    const app = bench();
    let cursor = 0;
    app.fetchPage.mockImplementation(async () => {
      if (cursor < 3) cursor += 1;
      return page({ cursor });
    });

    app.follower.start('j1', 0);
    await settle();

    // Three advancing pages back to back with no idle sleep between them —
    // the long poll is the wait — and a fourth that stood still, which is
    // where the loop parked on the timer.
    expect(app.fetchPage).toHaveBeenCalledTimes(4);
    expect(app.cursors()).toEqual([0, 1, 2, 3]);
  });

  it('retries a failed fetch and marks the job lost after MAX_FOLLOW_FAILURES', async () => {
    const app = bench();
    app.fetchPage.mockRejectedValue(new Error('Failed to fetch'));

    app.follower.start('j1', 0);
    await settle();
    expect(app.fetchPage).toHaveBeenCalledTimes(1);
    expect(app.job()!.status).toBe('running');

    await vi.advanceTimersByTimeAsync(FOLLOW_RETRY_MS * MAX_FOLLOW_FAILURES);
    expect(app.fetchPage).toHaveBeenCalledTimes(MAX_FOLLOW_FAILURES);
    expect(app.job()!.status).toBe('lost');
    expect(app.follower.followingJobId()).toBeNull();

    // Given up means given up: no further polling.
    await vi.advanceTimersByTimeAsync(60_000);
    expect(app.fetchPage).toHaveBeenCalledTimes(MAX_FOLLOW_FAILURES);
  });

  it('recovers from a transient failure without losing the job', async () => {
    const app = bench();
    app.fetchPage
      .mockRejectedValueOnce(new Error('Failed to fetch'))
      .mockResolvedValue(page());

    app.follower.start('j1', 0);
    await settle();
    await vi.advanceTimersByTimeAsync(FOLLOW_RETRY_MS + 10);

    expect(app.fetchPage).toHaveBeenCalledTimes(2);
    expect(app.job()!.status).toBe('running');
  });

  it('hands the last error to onGiveUp and lets it settle the job itself', async () => {
    const boom = new Error('Failed to fetch');
    const onGiveUp = vi.fn(() => true);
    const app = bench({ onGiveUp });
    app.fetchPage.mockRejectedValue(boom);

    app.follower.start('j1', 0);
    await settle();
    await vi.advanceTimersByTimeAsync(FOLLOW_RETRY_MS * MAX_FOLLOW_FAILURES);

    expect(onGiveUp).toHaveBeenCalledTimes(1);
    expect(onGiveUp).toHaveBeenCalledWith('j1', boom, app.job());
    // It said it handled the ending, so nothing overwrote its verdict.
    expect(app.job()!.status).toBe('running');
  });

  it('falls back to lost when onGiveUp declines the ending', async () => {
    const onGiveUp = vi.fn(() => false);
    const app = bench({ onGiveUp });
    app.fetchPage.mockRejectedValue(new Error('Failed to fetch'));

    app.follower.start('j1', 0);
    await settle();
    await vi.advanceTimersByTimeAsync(FOLLOW_RETRY_MS * MAX_FOLLOW_FAILURES);

    expect(app.job()!.status).toBe('lost');
  });

  it('keeps what a declining onGiveUp wrote to the job', async () => {
    // Declining is about the ENDING, not about the job: a callback is
    // entitled to record what it learned (a hint, a command it recovered
    // elsewhere) and still leave `lost` to the follower. Building the patch
    // from the snapshot taken before the callback ran would undo that write.
    const onGiveUp = vi.fn(() => false);
    const app = bench({ onGiveUp });
    onGiveUp.mockImplementation(() => {
      app.setJob({ ...app.job()!, tag: 'noted' });
      return false;
    });
    app.fetchPage.mockRejectedValue(new Error('Failed to fetch'));

    app.follower.start('j1', 0);
    await settle();
    await vi.advanceTimersByTimeAsync(FOLLOW_RETRY_MS * MAX_FOLLOW_FAILURES);

    expect(app.job()).toMatchObject({ status: 'lost', tag: 'noted' });
  });

  it('honours the timing overrides instead of the defaults', async () => {
    const onGiveUp = vi.fn(() => true);
    const app = bench({ waitS: 3, idleMs: 50, retryMs: 10, maxFailures: 2, onGiveUp });
    app.fetchPage.mockRejectedValue(new Error('Failed to fetch'));

    app.follower.start('j1', 0);
    await settle();
    await vi.advanceTimersByTimeAsync(10 + 5);

    expect(app.fetchPage).toHaveBeenCalledTimes(2);
    expect(app.waits()[0]).toBe(3);
    expect(onGiveUp).toHaveBeenCalledTimes(1);
  });

  it('aborts the parked request when following stops', async () => {
    // A 25 s long poll left dangling holds a connection open on a server with
    // a small pool, so stopping must cancel it rather than wait it out.
    const app = bench();
    // Parked, the way the server parks a long poll that has nothing yet.
    app.fetchPage.mockImplementation(() => new Promise<JobEventsPage>(() => {}));

    app.follower.start('j1', 0);
    await settle();
    expect(app.signals()).toHaveLength(1);
    expect(app.signals()[0].aborted).toBe(false);

    app.follower.stop();
    expect(app.signals()[0].aborted).toBe(true);
    expect(app.follower.followingJobId()).toBeNull();

    await vi.advanceTimersByTimeAsync(60_000);
    expect(app.fetchPage).toHaveBeenCalledTimes(1);
  });

  it('ends a stale loop whose page arrives after the generation moved on', async () => {
    // The abort only rejects the request in flight; on a fetch that ignores
    // the signal (or answers in the same tick it was aborted) the generation
    // is the whole guard.
    const app = bench();
    let deliver: (value: JobEventsPage) => void = () => {};
    app.fetchPage.mockImplementation(() => new Promise<JobEventsPage>((resolve) => {
      deliver = resolve;
    }));

    app.follower.start('j1', 0);
    await settle();

    app.follower.stop();
    deliver(page({ cursor: 9, events: [{ type: 'log', cursor: 9, ts: 't', line: 'late' }] }));
    await settle();

    expect(app.patchJob).not.toHaveBeenCalled();
    expect(app.job()!.log).toEqual([]);
    await vi.advanceTimersByTimeAsync(60_000);
    expect(app.fetchPage).toHaveBeenCalledTimes(1);
  });

  it('leaves the parked request alone when the same job is started again', async () => {
    // Adoption runs on every catalog poll. Restarting the follower each time
    // would abort a long poll that was about to answer and re-ask from the
    // same cursor — a request per poll, which is the entire cost the long
    // poll exists to avoid.
    const app = bench();
    app.fetchPage.mockImplementation(() => new Promise<JobEventsPage>(() => {}));

    app.follower.start('j1', 0);
    await settle();
    expect(app.signals()).toHaveLength(1);

    app.follower.start('j1', 0);
    await settle();

    expect(app.signals()).toHaveLength(1);
    expect(app.signals()[0].aborted).toBe(false);
    expect(app.follower.followingJobId()).toBe('j1');
  });

  it('switches to another job, abandoning the first', async () => {
    const app = bench();
    app.fetchPage.mockImplementation(() => new Promise<JobEventsPage>(() => {}));

    app.follower.start('j1', 0);
    await settle();
    app.follower.start('j2', 4);
    await settle();

    expect(app.signals()[0].aborted).toBe(true);
    expect(app.follower.followingJobId()).toBe('j2');
    expect(app.cursors()).toEqual([0, 4]);
  });

  it('drops a page for a job that is no longer the open one', async () => {
    const app = bench();
    app.fetchPage.mockResolvedValue(page({
      cursor: 1, events: [{ type: 'log', cursor: 1, ts: 't', line: 'stale' }],
    }));

    app.follower.start('j1', 0);
    app.setJob(makeJob({ jobId: 'other' }));
    await settle();

    expect(app.patchJob).not.toHaveBeenCalled();
    expect(app.job()!.log).toEqual([]);
  });

  it('settles with a null job when the open one has moved on', async () => {
    const app = bench();
    app.fetchPage.mockResolvedValue(page({ status: 'failed', cursor: 0 }));

    app.follower.start('j1', 0);
    app.setJob(null);
    await settle();

    expect(app.onSettled).toHaveBeenCalledWith('j1', 'failed', null);
  });

  it('starts from the beginning when the caller passes no cursor', async () => {
    const app = bench();

    app.follower.start('j1');
    await settle();

    expect(app.cursors()[0]).toBe(0);
  });

  it('does nothing when stopped before it was ever started', () => {
    // `_reset...ForTesting` and an unmount both call `stop()` unconditionally.
    const app = bench();

    expect(() => app.follower.stop()).not.toThrow();
    expect(app.follower.followingJobId()).toBeNull();
    expect(app.fetchPage).not.toHaveBeenCalled();
  });

  it('counts a fetchPage that throws synchronously as a failed turn', async () => {
    // A client that rejects before it ever awaits — a URL that will not
    // build, a mock a test forgot to make async — must not escape the retry
    // budget as an unhandled throw out of the loop.
    const app = bench({ retryMs: 10, maxFailures: 2 });
    app.fetchPage.mockImplementation(() => { throw new Error('boom'); });

    app.follower.start('j1', 0);
    await settle();
    await vi.advanceTimersByTimeAsync(15);

    expect(app.fetchPage).toHaveBeenCalledTimes(2);
    expect(app.job()!.status).toBe('lost');
  });

  it('keeps polling a terminal job while its backlog is still arriving', async () => {
    // A job that finished with a hundred queued lines still has pages to hand
    // over; stopping on `status` alone would truncate its log.
    const app = bench();
    app.fetchPage
      .mockResolvedValueOnce(page({ status: 'done', cursor: 1 }))
      .mockResolvedValueOnce(page({ status: 'done', cursor: 2 }))
      .mockResolvedValue(page({ status: 'done', cursor: 2 }));

    app.follower.start('j1', 0);
    await settle();

    expect(app.fetchPage).toHaveBeenCalledTimes(3);
    expect(app.onSettled).toHaveBeenCalledTimes(1);
  });
});
