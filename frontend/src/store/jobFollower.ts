import type { JobEvent, JobEventsPage, JobStatus } from '../api/rest';

/**
 * The long-poll job model, shared by every center that watches one.
 *
 * A pack install and a plugin install are the same shape of thing: a server
 * -side job that outlives the panel it was started from, reports itself as a
 * stream of cursor-numbered events, and has to survive the modal closing, a
 * second browser tab, and a page reload. This module owns the two halves of
 * that which have no domain in them — the pure fold of a page of events into
 * a job, and the loop that keeps asking for the next page — so the stores
 * above it are left with only the part that IS about packs or plugins.
 *
 * Deliberately domain-free: it imports the three wire types and nothing else.
 * A follower's in-flight request and generation counter live in the closure
 * `createJobFollower` returns, never at module scope, so two centers can
 * follow two jobs at once without either one's `stop()` reaching the other.
 */

/** Long-poll parking time for a job follower, in SECONDS (server caps it). */
export const EVENT_WAIT_S = 25;

/**
 * Floor between follower turns that made no progress.
 *
 * The long poll is supposed to park server-side, so an unadvanced page
 * normally means its deadline passed and 500 ms of extra latency is noise. It
 * is also the thing standing between this loop and a busy-wait if a server
 * ever answers instantly without moving the cursor.
 */
export const FOLLOW_IDLE_MS = 500;

/** Wait between retries after a failed events request. */
export const FOLLOW_RETRY_MS = 2000;

/**
 * Consecutive failures before the follower gives up on a job.
 *
 * Five retries across ten seconds outlast a restarting dev server and a
 * flaky Wi-Fi hop, which are the two things that interrupt an install on a
 * developer's machine. Beyond that the honest thing to say is that we no
 * longer know what the job is doing.
 */
export const MAX_FOLLOW_FAILURES = 5;

/** Ring-buffer bound on a rendered job log. */
export const MAX_LOG_LINES = 400;

/**
 * One line of a job's log.
 *
 * `text` is the server's own message (English, and often a pip line), kept
 * verbatim rather than translated: it is a transcript of what ran, and the
 * step LABELS the UI translates come from `JobStep.step` ids instead.
 */
export interface LogLine {
  /** Unique, ascending, and stable — the React key for the line. */
  seq: number;
  ts: string | null;
  kind: 'step' | 'log' | 'error';
  text: string;
}

/** How far one item has downloaded. */
export interface ItemProgress {
  bytesDone: number;
  /** null when the server never learned the size (a chunked response). */
  bytesTotal: number | null;
  /** 0..100, or null when there is no total to divide by. */
  percent: number | null;
}

export interface JobStep {
  /** The server's step id (`pip`, `download:<item>`, `verify`). */
  step: string;
  /** The server's English label — a fallback for a step the UI cannot name. */
  label: string;
  state: 'running' | 'done';
}

/**
 * A job's status as the UI knows it.
 *
 * `lost` is ours, not the server's: after enough failed polls we no longer
 * know what the job is doing, and saying "running" about a job nobody is
 * watching is the one answer that is definitely wrong.
 */
export type JobPhase = JobStatus | 'lost';

/**
 * What every followed job has. A domain adds its own fields on top —
 * `PackJob` its pack id and install mode, `PluginJob` its plugin id and kind
 * — and passes itself as `J` to the two functions below.
 */
export interface Job {
  jobId: string;
  status: JobPhase;
  steps: JobStep[];
  /** Keyed by item id, so a replayed frame overwrites instead of appending. */
  items: Record<string, ItemProgress>;
  log: LogLine[];
  /** Highest event cursor applied — where the follower resumes. */
  cursor: number;
  error: { message: string; hint: string | null } | null;
  /** Set by a `needs_restart` event: what to run when we cannot restart. */
  restartCommand: string | null;
  startedAt: number;
}

/** A job with nothing in it yet — what an adopted or just-started job starts as. */
export function emptyJob(jobId: string): Job {
  return {
    jobId,
    status: 'running',
    steps: [],
    items: {},
    log: [],
    cursor: 0,
    error: null,
    restartCommand: null,
    startedAt: Date.now(),
  };
}

function str(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function clampPercent(value: number): number {
  return Math.min(100, Math.max(0, value));
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Fold one page of events into a job.
 *
 * Pure and exported so the interesting part — what each event type does to
 * the steps, the per-item bars and the log — is testable without a server, a
 * timer or a React tree.
 *
 * A `progress` event NEVER becomes a log line. A 2 GB download emits
 * thousands of them; one line each would bury every message that actually
 * says something, and the bytes already have a bar of their own.
 *
 * *onExtra* is how a domain reads the keys this reducer has no opinion about
 * (a pack's `retry_mode`, a plugin's `sha`): it runs once per applied event,
 * after the switch, on a shallow copy of the job whose DOMAIN fields it may
 * write. The generic fields are written back afterwards from this function's
 * own locals, so a careless callback cannot corrupt the log or the cursor.
 */
export function reduceJobEvents<J extends Job>(
  job: J,
  page: JobEventsPage,
  onExtra?: (event: JobEvent, draft: J) => void,
): J {
  let steps = job.steps;
  let items = job.items;
  let error = job.error;
  let restartCommand = job.restartCommand;
  const lines: LogLine[] = [];
  const draft: J = { ...job };

  // Log keys have to be unique for React and ascending for the reader. Event
  // cursors are both, so they are used as-is; a frame that arrived without
  // one (a hand-written double, an older backend) gets the next number after
  // everything seen so far rather than a colliding zero.
  let lastSeq = job.log.length > 0 ? job.log[job.log.length - 1].seq : 0;
  if (job.cursor > lastSeq) lastSeq = job.cursor;

  const line = (
    seq: number, ts: string | null, kind: LogLine['kind'], text: string,
  ) => lines.push({ seq, ts, kind, text });

  const markStepDone = (predicate: (step: JobStep) => boolean) => {
    if (!steps.some((step) => step.state === 'running' && predicate(step))) return;
    steps = steps.map((step) => (
      step.state === 'running' && predicate(step) ? { ...step, state: 'done' } : step
    ));
  };

  for (const event of page.events) {
    const cursor = num(event.cursor);
    // Idempotent by cursor. `/events` returns events strictly AFTER the
    // cursor we sent, so this only fires on a replay — but a replayed log
    // line would be a duplicate React key as well as a duplicate line, and
    // the reducer is cheaper to make safe than every caller is.
    if (cursor !== null && cursor <= job.cursor) continue;
    const seq = cursor !== null ? Math.max(cursor, lastSeq + 1) : lastSeq + 1;
    lastSeq = seq;
    const ts = str(event.ts);

    switch (event.type) {
      case 'step_started': {
        const step = str(event.step);
        if (step === null) break;
        const label = str(event.label) ?? step;
        // The server emits `step_done` for every step it finishes, but a step
        // that ended by raising never gets one; closing the previous step
        // here keeps the list from showing two spinners at once.
        markStepDone(() => true);
        steps = [...steps, { step, label, state: 'running' }];
        line(seq, ts, 'step', label);
        break;
      }
      case 'step_done': {
        const step = str(event.step);
        if (step === null) break;
        markStepDone((entry) => entry.step === step);
        break;
      }
      case 'log': {
        const text = str(event.line);
        // Blank lines are real in pip output and are nothing to render.
        if (text === null || text.trim() === '') break;
        line(seq, ts, 'log', text);
        break;
      }
      case 'progress': {
        const item = str(event.item);
        if (item === null) break;
        const bytesDone = num(event.bytes_done) ?? 0;
        const bytesTotal = num(event.bytes_total);
        const reported = num(event.percent);
        const percent = reported !== null
          ? clampPercent(reported)
          : bytesTotal !== null && bytesTotal > 0
            ? clampPercent((100 * bytesDone) / bytesTotal)
            : null;
        items = { ...items, [item]: { bytesDone, bytesTotal, percent } };
        break;
      }
      case 'job_done':
        markStepDone(() => true);
        line(seq, ts, 'step', 'done');
        break;
      case 'job_failed': {
        const message = str(event.message) ?? '';
        error = { message, hint: str(event.hint) };
        line(seq, ts, 'error', message);
        break;
      }
      case 'needs_restart':
        restartCommand = str(event.command);
        break;
      default:
        // An unknown type from a newer backend is skipped, never rendered as
        // a mystery line — the log is a transcript, not a protocol dump.
        break;
    }

    onExtra?.(event, draft);
  }

  const log = lines.length > 0
    ? [...job.log, ...lines].slice(-MAX_LOG_LINES)
    : job.log;

  return {
    ...draft,
    status: page.status,
    steps,
    items,
    log,
    error,
    restartCommand,
    // Never moves backwards: an empty page returns the cursor we sent.
    cursor: Math.max(job.cursor, page.cursor),
  };
}

export interface JobFollowerOptions<J extends Job> {
  /**
   * Ask for the page after *cursor*, parking for up to *waitS* seconds.
   *
   * The signal is the follower's, and abandoning a parked request is the only
   * way to release the connection it is holding, so an implementation that
   * drops it turns `stop()` into a lie.
   */
  fetchPage: (
    jobId: string, cursor: number, signal: AbortSignal, waitS: number,
  ) => Promise<JobEventsPage>;
  /** The job the store is showing, whatever it is. May be null, or another job. */
  getOpenJob: () => J | null;
  /**
   * Record *next* — but only if *jobId* is still the open job. Following is
   * asynchronous, and a page for a job the user has moved on from must not
   * overwrite the one they are looking at.
   */
  patchJob: (jobId: string, next: J) => void;
  /**
   * The job reached an ending. *job* is the freshly folded job when it is
   * still the open one and null when it is not, which is the caller's cue
   * that there is nothing on screen to talk about.
   */
  onSettled: (jobId: string, status: JobStatus, job: J | null) => void;
  /**
   * The events endpoint failed *maxFailures* times in a row.
   *
   * Return true to say the ending has been handled — a domain that can read
   * something into the silence (a pack whose server was always going to stop
   * answering mid-restart) settles the job itself. Return false, or leave the
   * callback out, and the follower marks the job `lost`. *error* is the last
   * rejection, and the difference between a dropped connection and a server
   * that answered with a status code is usually the whole decision.
   */
  onGiveUp?: (jobId: string, error: unknown, job: J | null) => boolean;
  /** Passed straight to `reduceJobEvents` on every page. */
  onExtra?: (event: JobEvent, draft: J) => void;
  waitS?: number;
  idleMs?: number;
  retryMs?: number;
  maxFailures?: number;
}

export interface JobFollower {
  /** Follow *jobId* from *cursor*, unless it is already being followed. */
  start: (jobId: string, cursor?: number) => void;
  /** Abandon the current follower, if any. */
  stop: () => void;
  /** The job this follower is on, or null. Makes adoption idempotent. */
  followingJobId: () => string | null;
}

/**
 * A loop that tails one job at a time until it can produce no more.
 *
 * The stop condition is "terminal AND the cursor did not move", which is the
 * only one that is actually true: a finished job with a backlog still has
 * pages to hand over, and a running job that returned nothing is simply
 * between events. Phrasing it in terms of the CURSOR rather than
 * `events.length` also makes a server that answers without making progress a
 * bounded loop instead of a busy-wait.
 *
 * In-flight requests and timers are closure state, not store state: putting
 * them in a store would make every turn of the loop a re-render for
 * subscribers that only care about the data.
 */
export function createJobFollower<J extends Job>(
  options: JobFollowerOptions<J>,
): JobFollower {
  const {
    fetchPage, getOpenJob, patchJob, onSettled, onGiveUp, onExtra,
    waitS = EVENT_WAIT_S,
    idleMs = FOLLOW_IDLE_MS,
    retryMs = FOLLOW_RETRY_MS,
    maxFailures = MAX_FOLLOW_FAILURES,
  } = options;

  /** Bumped by every `stop`; a loop whose generation is stale exits. */
  let generation = 0;
  let inFlight: AbortController | null = null;
  let following: string | null = null;

  /**
   * Two mechanisms, and both are needed. `abort()` releases the parked HTTP
   * request immediately — a 25 s long poll left dangling holds a connection
   * open on a server with a small pool. The generation bump is what stops the
   * LOOP: an abort only rejects the request in flight, and without it the
   * follower would simply issue the next one.
   */
  function stop(): void {
    generation += 1;
    inFlight?.abort();
    inFlight = null;
    following = null;
  }

  /**
   * The idempotence is what lets a catalog poll adopt the server's active job
   * every time without restarting the follower — and without the
   * double-follow that would apply every event twice.
   */
  function start(jobId: string, cursor = 0): void {
    if (following === jobId) return;
    stop();
    following = jobId;
    void loop(jobId, cursor, generation);
  }

  async function loop(
    jobId: string, startCursor: number, mine: number,
  ): Promise<void> {
    let cursor = startCursor;
    let failures = 0;

    while (mine === generation) {
      const controller = new AbortController();
      inFlight = controller;
      let page: JobEventsPage;
      try {
        page = await fetchPage(jobId, cursor, controller.signal, waitS);
      } catch (error) {
        // A deliberate abort bumped the generation; anything else is the
        // network, which a long install is entitled to survive — the work
        // itself is happening server-side and does not care that we blinked.
        if (mine !== generation) return;
        failures += 1;
        if (failures >= maxFailures) {
          if (following === jobId) following = null;
          const open = getOpenJob();
          const held = open !== null && open.jobId === jobId ? open : null;
          const handled = onGiveUp?.(jobId, error, held) === true;
          if (!handled && held !== null) {
            patchJob(jobId, { ...held, status: 'lost' });
          }
          return;
        }
        await sleep(retryMs);
        continue;
      }
      if (mine !== generation) return;
      failures = 0;

      let folded: J | null = null;
      const open = getOpenJob();
      if (open !== null && open.jobId === jobId) {
        folded = reduceJobEvents(open, page, onExtra);
        patchJob(jobId, folded);
      }

      const advanced = page.cursor > cursor;
      cursor = Math.max(cursor, page.cursor);
      if (page.status !== 'running' && !advanced) {
        if (following === jobId) following = null;
        onSettled(jobId, page.status, folded);
        return;
      }
      if (!advanced) await sleep(idleMs);
    }
  }

  return { start, stop, followingJobId: () => following };
}
