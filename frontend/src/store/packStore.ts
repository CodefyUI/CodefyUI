import { create } from 'zustand';
import {
  PackApiError,
  cancelPackJob,
  fetchHealth,
  getPackJobEvents,
  installPack,
  listPacks,
  removePackItem,
  type LaunchMode,
  type PackCatalog,
  type PackGpuInfo,
  type PackInstallMode,
  type PackJobEventsPage,
  type PackJobStatus,
  type PackSummary,
} from '../api/rest';
import { confirm } from '../utils/dialog';
import { localizedPackTitle } from '../utils/packAvailability';
import { useToastStore, type ToastAction } from './toastStore';
import { useUIStore } from './uiStore';
import { useI18n, type TranslationKey } from '../i18n';

/**
 * App-level state for the Package Center (optional model / library packs).
 *
 * Deliberately NOT owned by the modal that shows it. An install is a
 * multi-gigabyte download that outlives the window it was started from: the
 * user closes the modal, switches to another tab, or reloads the page, and
 * the download has to survive all three. Everything that would die with a
 * component — the long-poll follower, the restart handshake, the job itself —
 * therefore lives here, and the panel is a pure view of it.
 *
 * ── How progress reaches the screen ────────────────────────────────────
 * One long poll, exactly like `runStore`'s run follower: `GET
 * /api/packs/jobs/{id}/events?wait=` parks server-side and answers the moment
 * an event lands, so a download bar moves at the speed of the download rather
 * than the speed of a timer, with ONE connection instead of a poll per tick.
 *
 * The follower is keyed by job id at module scope rather than per-component,
 * which is what makes three things true at once: closing the modal keeps the
 * download alive, re-opening it resumes from the cursor already applied, and
 * a job started in ANOTHER browser tab is adopted by this one the first time
 * `refresh()` reports it as active.
 *
 * ── The restart handshake ──────────────────────────────────────────────
 * The GPU PyTorch pack swaps the wheel the running interpreter is using, so
 * no process can do it to itself. The server hands back a command and ends
 * the job as `needs_restart`; under `cdui start` a supervisor performs the
 * swap and relaunches, and this store's job is to notice the NEW server
 * (`boot_id`, not mere reachability — the old process answers right up until
 * it exits) and reload the page. The pending pack id is parked in
 * `sessionStorage` so the reloaded page can report how it went.
 */

/** Long-poll parking time for the job follower, in SECONDS (server caps it). */
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
 * Consecutive failures before the job is declared `lost` — or, for a
 * restart-mode job, settled as the `needs_restart` it was heading for.
 *
 * Five retries across ten seconds outlast a restarting dev server and a
 * flaky Wi-Fi hop, which are the two things that interrupt an install on a
 * developer's machine. Beyond that the honest thing to say is that we no
 * longer know what the download is doing.
 */
export const MAX_FOLLOW_FAILURES = 5;

/** Ring-buffer bound on the rendered install log. */
export const MAX_PACK_LOG_LINES = 400;

/**
 * How much of a failed restart's `log_tail` a toast is allowed to carry.
 *
 * Only ever shown when the record carried no message at all (see
 * `checkInProgress`), and bounded because the toast has no scroll of its own:
 * an unbounded tail would grow one card until it covered the canvas.
 */
export const LOG_TAIL_TOAST_CHARS = 300;

/** How often the restart handshake asks whether the server is back. */
export const RESTART_POLL_MS = 1500;

/**
 * How long a server gets to start going down before we call it a no-show.
 *
 * A supervisor that never picked the request up leaves /api/health answering
 * from the SAME process forever; without this the overlay would sit there
 * for the full timeout saying "restarting" about a server that never tried.
 */
export const RESTART_GRACE_MS = 30_000;

/** Total budget for a restart. A wheel swap on a slow disk is minutes. */
export const RESTART_TIMEOUT_MS = 600_000;

/**
 * Where the pending pack id is parked across the reload.
 *
 * `sessionStorage`, not `localStorage`: it is scoped to this tab and dies
 * with it, which is exactly the lifetime of "the page I am about to reload".
 */
export const RESTART_PENDING_KEY = 'codefyui-pack-restart-pending';

/**
 * The server's `reason` phrases, and the toast each one deserves.
 *
 * These 409s look identical to the "this server cannot restart itself" one —
 * same status, same `command` — and only `reason` separates a permanent
 * limitation from a wait of a few seconds. Keyed on the server's exact
 * wording (a contract pinned on the backend side by
 * `test_restart_refused_while_a_graph_runs` and its neighbours); anything not
 * listed falls back to `needsCli`, whose command is always a true way through.
 *
 * A `Map` rather than an object literal, because the key comes off the wire:
 * `{}['toString']` is a function, and a plain lookup would hand `t()` one.
 */
const REFUSAL_TOASTS = new Map<string, TranslationKey>([
  ['a graph is running', 'packs.toast.restartRefusedRunning'],
  // Two spellings of one condition: the first is another restart-mode submit
  // colliding with a claim on disk, the second a LIVE install arriving while
  // this server is already on its way out.
  ['a restart-mode install is already pending', 'packs.toast.restartRefusedPending'],
  ['a restart is already pending', 'packs.toast.restartRefusedPending'],
]);

/**
 * One line of an install job's log.
 *
 * `text` is the server's own message (English, and often a pip line), kept
 * verbatim rather than translated: it is a transcript of what ran, and the
 * step LABELS the UI translates come from `PackJobStep.step` ids instead.
 */
export interface PackLogLine {
  /** Unique, ascending, and stable — the React key for the line. */
  seq: number;
  ts: string | null;
  kind: 'step' | 'log' | 'error';
  text: string;
}

/** How far one item has downloaded. */
export interface PackItemProgress {
  bytesDone: number;
  /** null when the server never learned the size (a chunked response). */
  bytesTotal: number | null;
  /** 0..100, or null when there is no total to divide by. */
  percent: number | null;
}

export interface PackJobStep {
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
export type PackJobPhase = PackJobStatus | 'lost';

export interface PackJob {
  jobId: string;
  packId: string;
  /**
   * How this job was launched.
   *
   * Recorded because `needs_restart` means two different things depending on
   * it: a RESTART-mode job ended the way it was always going to, and a LIVE
   * one hit a resolver conflict against the constraints file and is telling
   * the user to run a command. Only the first is a restart anybody asked for.
   */
  mode: PackInstallMode;
  status: PackJobPhase;
  steps: PackJobStep[];
  /** Keyed by item id, so a replayed frame overwrites instead of appending. */
  items: Record<string, PackItemProgress>;
  log: PackLogLine[];
  /** Highest event cursor applied — where the follower resumes. */
  cursor: number;
  error: { message: string; hint: string | null } | null;
  /** Set by a `needs_restart` event: what to run when we cannot restart. */
  restartCommand: string | null;
  /**
   * Set by a `needs_restart` event on a LIVE install the resolver stopped:
   * the mode that CAN finish it, when the server is able to offer one.
   *
   * Null on every restart-mode job, which needs no retry — it ended the way
   * it was always going to. Only `restart` is understood; a mode from a newer
   * backend reads as null rather than as a button that posts something this
   * build cannot follow.
   */
  retryMode: PackInstallMode | null;
  startedAt: number;
}

export type RestartPhase = 'idle' | 'waiting' | 'notStarted' | 'timeout';

export interface RestartState {
  phase: RestartPhase;
  packId: string | null;
  startedAt: number | null;
  command: string | null;
}

interface PackState {
  packs: PackSummary[];
  /** The same rows keyed by id, for the O(1) lookups the canvas does. */
  byId: Record<string, PackSummary>;
  loading: boolean;
  /** A first answer arrived — a catalog or a 404. A network error is neither. */
  loaded: boolean;
  /** The server predates the Package Center: treat everything as available. */
  unsupported: boolean;
  error: string | null;
  remoteInstallAllowed: boolean;
  launchMode: LaunchMode;
  /**
   * The server says it can install a restart-mode pack and come back.
   *
   * Narrower than `launchMode === 'start'` — it also wants the launcher still
   * on disk and the kill switch off — and it is the ONLY thing the panel and
   * the settle handler gate a restart on. False until a catalog says
   * otherwise, so nothing offers a restart to a server that never claimed one.
   */
  restartAvailable: boolean;
  gpu: PackGpuInfo | null;
  job: PackJob | null;
  /** Pack ids with an install request in flight — disables that card's button. */
  busy: Record<string, boolean>;
  cancelling: boolean;
  restart: RestartState;

  refresh: () => Promise<void>;
  install: (
    packId: string,
    opts?: { items?: string[]; mode?: PackInstallMode; variant?: string },
  ) => Promise<void>;
  cancel: () => Promise<void>;
  removeItem: (packId: string, itemId: string) => Promise<void>;
  /** Adopt *jobId* and start (or keep) following it. Idempotent per job id. */
  followJob: (jobId: string, packId: string, cursor?: number) => void;
  stopFollowing: () => void;
  /** Clear a finished job from the activity pane. Ignored while running. */
  dismissJob: () => void;
  restartFlow: (packId: string, command: string | null) => Promise<void>;
  /** Once per page load: adopt a running job and report a finished restart. */
  checkInProgress: () => Promise<void>;
}

const IDLE_RESTART: RestartState = {
  phase: 'idle', packId: null, startedAt: null, command: null,
};

/**
 * A job with nothing in it yet — what an adopted or just-started job starts as.
 *
 * `mode` defaults to `live` because that is what an ADOPTED job is until the
 * catalog says otherwise: it decides whether a `needs_restart` ending starts a
 * restart handshake, and starting one for a job that never asked for it puts a
 * blocking overlay over a server that is not going anywhere.
 */
export function emptyPackJob(
  jobId: string,
  packId: string,
  mode: PackInstallMode = 'live',
): PackJob {
  return {
    jobId,
    packId,
    mode,
    status: 'running',
    steps: [],
    items: {},
    log: [],
    cursor: 0,
    error: null,
    restartCommand: null,
    retryMode: null,
    startedAt: Date.now(),
  };
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/**
 * `action` is spread rather than always passed, so a toast without one is
 * byte-for-byte the object every existing caller was already producing.
 */
function toast(
  message: string,
  type: 'info' | 'error' | 'success' | 'warning',
  action?: ToastAction,
) {
  useToastStore.getState().addToast(message, type, action ? { action } : undefined);
}

/** The button a toast about a pack wears: it opens the panel on that pack. */
function openCenterAction(packId: string): ToastAction {
  const { t } = useI18n.getState();
  return {
    label: t('packs.toast.openCenter'),
    onClick: () => useUIStore.getState().openPackCenter(packId),
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

/** What the breadcrumb carries across the reload: which pack, which job. */
interface RestartBreadcrumb {
  packId: string;
  /** null for a payload written before this field existed. */
  jobId: string | null;
}

/**
 * `sessionStorage` can throw outright — Safari in private mode, and any
 * browser configured to block site data. Losing the restart breadcrumb costs
 * one missing toast; letting it throw would abandon the restart itself.
 */
function readPending(): RestartBreadcrumb | null {
  let raw: string | null;
  try {
    raw = sessionStorage.getItem(RESTART_PENDING_KEY);
  } catch {
    return null;
  }
  if (raw === null || raw === '') return null;
  const [packId, jobId] = raw.split(' ');
  return { packId, jobId: jobId || null };
}

function writePending(packId: string | null, jobId: string | null = null): void {
  try {
    if (packId === null) {
      sessionStorage.removeItem(RESTART_PENDING_KEY);
      return;
    }
    // The job id rides along beside the pack: `last_restart_job` carries no
    // age bound, so the page that comes back has no other way to tell this
    // install's outcome from one that finished an hour ago. A space is a safe
    // join — pack ids are slugs and job ids are hex.
    sessionStorage.setItem(
      RESTART_PENDING_KEY, jobId === null ? packId : `${packId} ${jobId}`,
    );
  } catch {
    // See readPending.
  }
}

/** Only `running` is live; everything else — ours included — is an ending. */
function isTerminalPhase(status: PackJobPhase): boolean {
  return status !== 'running';
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
 */
export function reducePackEvents(job: PackJob, page: PackJobEventsPage): PackJob {
  let steps = job.steps;
  let items = job.items;
  let error = job.error;
  let restartCommand = job.restartCommand;
  let retryMode = job.retryMode;
  const lines: PackLogLine[] = [];

  // Log keys have to be unique for React and ascending for the reader. Event
  // cursors are both, so they are used as-is; a frame that arrived without
  // one (a hand-written double, an older backend) gets the next number after
  // everything seen so far rather than a colliding zero.
  let lastSeq = job.log.length > 0 ? job.log[job.log.length - 1].seq : 0;
  if (job.cursor > lastSeq) lastSeq = job.cursor;

  const line = (
    seq: number, ts: string | null, kind: PackLogLine['kind'], text: string,
  ) => lines.push({ seq, ts, kind, text });

  const markStepDone = (predicate: (step: PackJobStep) => boolean) => {
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
        // Only the one mode this build knows how to post. The key is absent
        // unless the server can actually restart itself, so reading anything
        // else as a retry would offer a button whose request nothing here
        // knows how to make.
        retryMode = str(event.retry_mode) === 'restart' ? 'restart' : null;
        break;
      default:
        // An unknown type from a newer backend is skipped, never rendered as
        // a mystery line — the log is a transcript, not a protocol dump.
        break;
    }
  }

  const log = lines.length > 0
    ? [...job.log, ...lines].slice(-MAX_PACK_LOG_LINES)
    : job.log;

  return {
    ...job,
    status: page.status,
    steps,
    items,
    log,
    error,
    restartCommand,
    retryMode,
    // Never moves backwards: an empty page returns the cursor we sent.
    cursor: Math.max(job.cursor, page.cursor),
  };
}

// ── module-scope schedulers ──────────────────────────────────────────────
// In-flight requests and timers are process state, not store state: putting
// them in the store would make every turn of the loop a re-render for
// subscribers that only care about the data.

/** Bumped by every `stopFollowing`; a follower whose generation is stale exits. */
let followGeneration = 0;
let followAbort: AbortController | null = null;
/** The job a follower is currently on, or null. Makes adoption idempotent. */
let followingJobId: string | null = null;
/** The last job `onJobSettled` fired for — its side effects run exactly once. */
let settledJobId: string | null = null;

/** Bumped by every `restartFlow` and by the test reset; a stale loop exits. */
let restartGeneration = 0;

/** Once per page load — StrictMode mounts effects twice in development. */
let inProgressChecked = false;

/**
 * The catalog's `last_restart_job`, kept out of the store on purpose.
 *
 * It is read exactly once, by the post-reload handshake, and nothing renders
 * it; as state it would be one more field every component sees change on
 * every poll for no reason.
 */
let lastRestartRecord: Record<string, unknown> | null = null;

/**
 * Abandon the current follower, if any.
 *
 * Two mechanisms, and both are needed. `abort()` releases the parked HTTP
 * request immediately — a 25 s long poll left dangling holds a connection
 * open on a server with a small pool. The generation bump is what stops the
 * LOOP: an abort only rejects the request in flight, and without it the
 * follower would simply issue the next one.
 */
function stopFollowing(): void {
  followGeneration += 1;
  followAbort?.abort();
  followAbort = null;
  followingJobId = null;
}

/**
 * Follow *jobId* from *cursor*, unless it is already being followed.
 *
 * The idempotence is what lets `refresh()` adopt `active_job` on every poll
 * without restarting the follower — and without the double-follow that would
 * apply every event twice.
 */
function startFollowing(jobId: string, packId: string, cursor: number): void {
  if (followingJobId === jobId) return;
  stopFollowing();
  followingJobId = jobId;
  void follow(jobId, packId, cursor, followGeneration);
}

/** Record on the open job — never on a job the user has moved on from. */
function patchJob(jobId: string, patch: (job: PackJob) => PackJob): void {
  usePackStore.setState((state) => (
    state.job && state.job.jobId === jobId ? { job: patch(state.job) } : {}
  ));
}

/**
 * Tail one install job until it can produce no more.
 *
 * The stop condition is "terminal AND the cursor did not move", which is the
 * only one that is actually true: a finished job with a backlog still has
 * pages to hand over, and a running job that returned nothing is simply
 * between events. Phrasing it in terms of the CURSOR rather than
 * `events.length` also makes a server that answers without making progress a
 * bounded loop instead of a busy-wait.
 */
async function follow(
  jobId: string, packId: string, startCursor: number, generation: number,
): Promise<void> {
  let cursor = startCursor;
  let failures = 0;

  while (generation === followGeneration) {
    const controller = new AbortController();
    followAbort = controller;
    let page: PackJobEventsPage;
    try {
      page = await getPackJobEvents(jobId, {
        cursor,
        wait: EVENT_WAIT_S,
        signal: controller.signal,
      });
    } catch (error) {
      // A deliberate abort bumped the generation; anything else is the
      // network, which an install is entitled to survive — the download
      // itself is happening server-side and does not care that we blinked.
      if (generation !== followGeneration) return;
      failures += 1;
      if (failures >= MAX_FOLLOW_FAILURES) {
        if (followingJobId === jobId) followingJobId = null;
        // A RESTART-mode job is the one case where the endpoint going quiet
        // is the expected ending rather than a lost connection: the server
        // stops accepting half a second after the 202 and then exits, and a
        // LAN client or a tab the browser had throttled can miss that half
        // second entirely. `lost` would leave the user in
        // front of a "we no longer know" banner while the wheel swap runs —
        // no overlay, and worse, no breadcrumb, so the page that comes back
        // could not report how the install went. Settling it as the ending
        // it almost certainly reached runs the same handshake the last page
        // would have started; if it turns out the server never went, the
        // overlay's own thirty-second grace says so.
        //
        // Only for a connection that DROPPED, though. A `PackApiError` is
        // the server answering — a 404 for a job that aged out, a 500 — and
        // an answer is proof it is still there. Settling on those raised the
        // blocking overlay over a live server, which is the bug the note in
        // `onJobSettled` records as fixed.
        const store = usePackStore.getState();
        const open = store.job;
        if (!(error instanceof PackApiError)
            && open?.jobId === jobId && open.mode === 'restart') {
          // No `needs_restart` event ever arrived, so nothing carried the
          // command — and both give-up screens name one ("Run this command,
          // then reload:"). The catalog has it: `install_command` is what the
          // panel's own button would have run for this pack.
          const command = store.byId[packId]?.install_command ?? null;
          patchJob(jobId, (job) => ({
            ...job,
            status: 'needs_restart',
            restartCommand: job.restartCommand ?? command,
          }));
          onJobSettled(jobId, packId, 'needs_restart');
        } else {
          patchJob(jobId, (job) => ({ ...job, status: 'lost' }));
        }
        return;
      }
      await sleep(FOLLOW_RETRY_MS);
      continue;
    }
    if (generation !== followGeneration) return;
    failures = 0;

    patchJob(jobId, (job) => reducePackEvents(job, page));

    const advanced = page.cursor > cursor;
    cursor = Math.max(cursor, page.cursor);
    if (page.status !== 'running' && !advanced) {
      if (followingJobId === jobId) followingJobId = null;
      onJobSettled(jobId, packId, page.status);
      return;
    }
    if (!advanced) await sleep(FOLLOW_IDLE_MS);
  }
}

/**
 * A pack's human name for a toast, in the reader's language.
 *
 * The rule itself lives in `packAvailability` because the node side, the
 * panel and these toasts all name the same packs, and a zh-TW reader told
 * 已安裝 Word vectors (GloVe)。 about a card headed 詞向量（GloVe） is being told
 * about two different packs.
 */
function packTitle(packId: string): string {
  const { t } = useI18n.getState();
  return localizedPackTitle(t, usePackStore.getState().byId, packId);
}

/**
 * Everything that happens once, when a job reaches its end.
 *
 * Guarded by job id rather than trusted to the caller: `refresh()` can adopt
 * a job at any time, and a settled job announcing itself twice would toast
 * twice and, for `needs_restart`, start a second restart handshake.
 */
function onJobSettled(jobId: string, packId: string, status: PackJobStatus): void {
  if (settledJobId === jobId) return;
  settledJobId = jobId;

  const { t } = useI18n.getState();
  const store = usePackStore.getState();
  const title = packTitle(packId);
  // The page that ended the job was applied to `job` a moment ago — but only
  // if it is still the open one, so what is read back is checked rather than
  // assumed. A dismissed job must not lend its message to this one.
  const settled = store.job?.jobId === jobId ? store.job : null;

  switch (status) {
    case 'done':
      toast(t('packs.toast.installed', { pack: title }), 'success');
      void store.refresh();
      break;
    case 'failed':
      toast(
        t('packs.toast.installFailed', { message: settled?.error?.message ?? '' }),
        'error',
      );
      void store.refresh();
      break;
    case 'cancelled':
      toast(t('packs.toast.cancelled'), 'info');
      void store.refresh();
      break;
    case 'needs_restart': {
      const command = settled?.restartCommand ?? null;
      // The JOB's mode, not the launch mode. A restart-mode install is the
      // only one that ends by design in a wheel swap the supervisor performs;
      // a LIVE install lands here when its resolver conflicts with the
      // constraints file, and the server it was talking to is not going
      // anywhere. Keying this on `launchMode` alone put a blocking "Server
      // restarting" overlay over a running server for thirty seconds, and hid
      // the command the user actually needs behind it.
      //
      // The other half is the SERVER's own answer, not the launch mode it
      // implies: `restart_available` is what the process that has to come
      // back said about itself (its launcher still on disk, its kill switch
      // off), and a 202 for a restart-mode install is only ever issued by a
      // server that said yes.
      if (settled?.mode === 'restart' && store.restartAvailable) {
        // Parked BEFORE the reload so the page that comes back knows which
        // pack and which job to report on — neither survives the reload.
        writePending(packId, jobId);
        void store.restartFlow(packId, command);
      } else if (settled?.retryMode === 'restart' && store.restartAvailable) {
        // A LIVE install the constraints file stopped, on a server that CAN
        // restart itself and said so by offering the retry mode. `needsCli`
        // — "cannot be installed from inside the app" — is the one sentence
        // that is flatly false here: the panel's banner is rendering a
        // **Restart the server and install** button that does exactly this.
        // So the toast points at the panel, and carries the click, rather
        // than handing over a command the user does not need.
        toast(t('packs.toast.restartRetry'), 'warning', openCenterAction(packId));
      } else if (store.launchMode === 'dev') {
        // `cdui dev` reloads in place; nothing relaunches it, so no catalog
        // it serves will ever say otherwise. Its own sentence, because
        // "not from inside the app YET" would be a promise about a future
        // release when the actual answer is "not the way you started this
        // one" — and it points at the command block the banner is already
        // rendering rather than repeating the command inline.
        toast(t('packs.toast.devRestart'), 'warning');
      } else {
        // Nothing here can finish this install: no supervisor was asked for,
        // or the one that exists cannot promise to come back. The job STAYS
        // on screen — its banner is what renders the command block this
        // names.
        toast(t('packs.toast.needsCli', { command: command ?? '' }), 'warning');
      }
      break;
    }
    default:
      break;
  }
}

/**
 * Apply a whole catalog. The ONLY place `byId` is rebuilt.
 *
 * `packs` is a handful of rows, so rebuilding the index costs nothing here —
 * but doing it on every unrelated `set` would, and `byId` is read on every
 * node render through `packAvailability`.
 */
function setCatalog(catalog: PackCatalog): void {
  const byId: Record<string, PackSummary> = {};
  for (const pack of catalog.packs) byId[pack.id] = pack;
  lastRestartRecord = catalog.last_restart_job;
  usePackStore.setState({
    packs: catalog.packs,
    byId,
    remoteInstallAllowed: catalog.remote_install_allowed,
    launchMode: catalog.launch_mode,
    restartAvailable: catalog.restart_available,
    gpu: catalog.gpu,
    loading: false,
    loaded: true,
    unsupported: false,
    error: null,
  });
}

export const usePackStore = create<PackState>((set, get) => ({
  packs: [],
  byId: {},
  loading: false,
  loaded: false,
  unsupported: false,
  error: null,
  remoteInstallAllowed: true,
  launchMode: 'unknown',
  restartAvailable: false,
  gpu: null,
  job: null,
  busy: {},
  cancelling: false,
  restart: IDLE_RESTART,

  refresh: async () => {
    set({ loading: true });
    // Read BEFORE the request: a job that appears while this one is in
    // flight is not described by the answer, and must not be declared lost
    // on the strength of it.
    const jobBefore = get().job;

    let catalog: PackCatalog;
    try {
      catalog = await listPacks();
    } catch (err) {
      if (err instanceof PackApiError && err.status === 404) {
        // A server older than the Package Center. Not an error the user did
        // anything about, so it is reported by the panel, silently, and
        // every pack-gated node stays enabled.
        set({
          loading: false, loaded: true, unsupported: true,
          packs: [], byId: {}, error: null,
        });
        return;
      }
      // `loaded` stays as it was: no answer arrived, so a later mount is
      // entitled to try again. The rows already on screen are kept — a
      // dropped Wi-Fi packet is no reason to blank the catalog.
      set({ loading: false, error: errorMessage(err) });
      return;
    }

    setCatalog(catalog);

    const active = catalog.active_job;
    if (active !== null) {
      const current = get().job;
      if (!current || current.jobId !== active.job_id) {
        // `active_job` does not carry the mode it was launched with, so the
        // pack's own default is the closest thing to the truth. It is
        // right for every pack that HAS a restart mode, and a live pack has
        // no other mode to be running in.
        const mode = get().byId[active.pack_id]?.install_mode ?? 'live';
        set({ job: emptyPackJob(active.job_id, active.pack_id, mode) });
      }
      // Idempotent: this runs on every poll, and only the FIRST one starts a
      // follower. This is how a job started in another tab is adopted.
      startFollowing(active.job_id, active.pack_id, get().job?.cursor ?? 0);
      return;
    }

    const job = get().job;
    if (
      job
      && job.status === 'running'
      && jobBefore?.jobId === job.jobId
      // ...and nobody is watching it. A follower that is still parked on the
      // events endpoint has a better answer coming than this catalog read
      // does: `active_job` goes null the moment a job finishes, and the
      // follower's next page settles it as done/failed/cancelled. Marking it
      // `lost` from here is a race that turns a successful install into a
      // "lost contact with the server" banner.
      && followingJobId !== job.jobId
    ) {
      // The server has no record of a job we think is running: it restarted,
      // or the job aged out. Saying "running" would be the one answer that
      // is definitely wrong.
      set({ job: { ...job, status: 'lost' } });
    }
  },

  install: async (packId, opts = {}) => {
    const state = get();
    if (state.busy[packId]) return;
    const { t } = useI18n.getState();

    if (state.job && state.job.status === 'running') {
      toast(t('packs.toast.busy'), 'warning');
      return;
    }

    const pack = state.byId[packId];
    // The mode decides whether this needs a restart, so it is resolved the
    // same way the server would resolve it — but only what the CALLER asked
    // for is sent, because the request model refuses keys it did not declare.
    const mode: PackInstallMode = opts.mode ?? pack?.install_mode ?? 'live';
    if (mode === 'restart' && state.launchMode !== 'start') {
      toast(t('packs.toast.devRestart'), 'warning');
      return;
    }

    set((s) => ({ busy: { ...s.busy, [packId]: true } }));
    try {
      const { job_id } = await installPack(packId, {
        items: opts.items,
        mode: opts.mode,
        variant: opts.variant,
      });

      // Seed the bars the user is about to watch. Omitting `items` means
      // "the whole pack minus what is already here", which is what the
      // server does with it, so the seed mirrors that.
      const requested = opts.items
        ?? pack?.items.filter((item) => item.status !== 'present').map((item) => item.id)
        ?? [];
      // The mode resolved above, not `opts.mode`: what the job IS decides how
      // a `needs_restart` ending is handled, and a caller that let the pack's
      // default stand still launched a restart-mode install.
      const job = emptyPackJob(job_id, packId, mode);
      for (const item of requested) {
        job.items[item] = { bytesDone: 0, bytesTotal: null, percent: 0 };
      }
      set({ job });
      setPackStatus(packId, 'installing');
      startFollowing(job_id, packId, 0);
    } catch (err) {
      if (err instanceof PackApiError && typeof err.body?.command === 'string') {
        // A refusal that hands back a command is not a collision — it is the
        // server saying "not from in here" or "not right now". Both 409
        // shapes carry a `detail`, and only this one carries `command`, so
        // the command is what tells them apart. Checked FIRST: every
        // restart-mode install (the GPU pack, today) lands here, and "another
        // install is already running" would be a plain lie about a server
        // sitting idle.
        //
        // `reason` then splits the passing conditions off from the permanent
        // one. `needsCli` says the app cannot do this at all, which is true
        // of a server that cannot restart itself and FALSE of these two: the
        // button works, it just does not work this second, and what the user
        // needs to hear is what to wait for. An unknown reason from a newer
        // backend falls through to `needsCli`, which is wrong-ish but never
        // misleading — the command in it does work.
        const refusal = REFUSAL_TOASTS.get(String(err.body.reason ?? ''));
        toast(refusal
          ? t(refusal)
          : t('packs.toast.needsCli', { command: err.body.command }), 'warning');
      } else if (err instanceof PackApiError && err.status === 409) {
        // Somebody else got there first — this tab, another tab, or the CLI.
        // The refresh adopts whatever the server IS running, which is more
        // useful than the refusal.
        toast(t('packs.toast.busy'), 'warning');
        await get().refresh();
      } else if (err instanceof PackApiError && err.status === 403) {
        toast(t('packs.remoteDisabled'), 'error');
      } else if (err instanceof PackApiError && err.status === 400
                 && Array.isArray(err.body?.blocked_by)
                 && err.body.blocked_by.length > 0) {
        const first = String(err.body.blocked_by[0]);
        toast(t('packs.toast.blocked', { pack: packTitle(first) }), 'warning');
      } else {
        toast(t('packs.toast.installFailed', { message: errorMessage(err) }), 'error');
      }
    } finally {
      set((s) => {
        const busy = { ...s.busy };
        delete busy[packId];
        return { busy };
      });
    }
  },

  cancel: async () => {
    const job = get().job;
    if (!job || isTerminalPhase(job.status) || get().cancelling) return;
    set({ cancelling: true });
    try {
      // Cooperative: the flow notices between steps and inside a download, so
      // the job may still say running when this returns. The FOLLOWER is what
      // records the outcome — faking it here would show "cancelled" over a
      // download that is still writing bytes.
      await cancelPackJob(job.jobId);
    } catch (err) {
      toast(
        useI18n.getState().t('packs.toast.cancelFailed', { message: errorMessage(err) }),
        'error',
      );
    } finally {
      set({ cancelling: false });
    }
  },

  removeItem: async (packId, itemId) => {
    const { t } = useI18n.getState();
    const ok = await confirm({
      title: t('packs.item.remove'),
      message: t('packs.item.removeConfirm', { item: itemId }),
      confirmText: t('packs.item.remove'),
      variant: 'danger',
    });
    if (!ok) return;

    try {
      const outcome = await removePackItem(packId, itemId);
      if (outcome.removed === false) {
        // The sentinel is gone but the bytes are not — Windows keeps an open
        // file. Promising the space back would be a lie.
        toast(t('packs.item.removeFailed', { item: itemId }), 'warning');
      } else {
        toast(t('packs.item.removed', { item: itemId }), 'success');
      }
    } catch (err) {
      // A separate key from the `removed: false` warning above, which has no
      // message to append: putting `{message}` in that one would leave a
      // dangling separator on the commonest of the two outcomes.
      toast(
        t('packs.item.removeError', { item: itemId, message: errorMessage(err) }),
        'error',
      );
    }
    // Either way the disk is not what the catalog says it is.
    await get().refresh();
  },

  followJob: (jobId, packId, cursor = 0) => {
    const current = get().job;
    if (!current || current.jobId !== jobId) {
      // Same reasoning as the adoption in `refresh`: nothing tells us how a
      // job we did not start was launched, so the pack's own mode stands in.
      set({ job: emptyPackJob(jobId, packId, get().byId[packId]?.install_mode ?? 'live') });
    }
    startFollowing(jobId, packId, cursor);
  },

  stopFollowing: () => stopFollowing(),

  dismissJob: () => {
    const job = get().job;
    if (!job || !isTerminalPhase(job.status)) return;
    set({ job: null });
  },

  restartFlow: async (packId, command) => {
    // Nothing to follow: the job that asked for this is over, and the server
    // it was talking to is about to stop existing.
    stopFollowing();
    restartGeneration += 1;
    const generation = restartGeneration;
    const startedAt = Date.now();
    set({ restart: { phase: 'waiting', packId, startedAt, command } });

    // One read before the wait. A reachable /api/health proves only that the
    // OLD process is still answering — it does right up until it exits — so
    // the identity of the process is what a "the server came back" decision
    // is actually made of.
    let bootIdAtStart: string | undefined;
    let sawDown = false;
    try {
      bootIdAtStart = (await fetchHealth()).boot_id;
    } catch {
      // The server was already going down when we looked. An outage observed
      // is an outage: without recording it here the loop would wait for a
      // SECOND one that has no reason to happen, and call a restart that
      // worked perfectly a no-show thirty seconds later.
      sawDown = true;
    }

    while (generation === restartGeneration) {
      // Wall clock, never a tick count: a sleeping laptop fires no timers,
      // and an overlay that counts turns would still be up an hour later.
      const elapsed = Date.now() - startedAt;
      // Both give-up branches drop the breadcrumb. It exists to let the page
      // that comes back from an automatic reload report how the install went;
      // once the handshake has given up, the user reloads by hand at a time of
      // their choosing, and a breadcrumb left behind would toast an outcome
      // read off a `last_restart_job` record from some other attempt.
      if (elapsed > RESTART_TIMEOUT_MS) {
        writePending(null);
        set((state) => ({ restart: { ...state.restart, phase: 'timeout' } }));
        return;
      }
      if (!sawDown && elapsed > RESTART_GRACE_MS) {
        // Half a minute and the server never even flinched: nothing picked
        // the restart up, and the user needs the command, not a spinner.
        writePending(null);
        set((state) => ({ restart: { ...state.restart, phase: 'notStarted' } }));
        return;
      }

      try {
        const info = await fetchHealth();
        if (generation !== restartGeneration) return;
        // Either proof is enough: a gap in service, or a different process
        // than the one we started with. `boot_id` is absent on a server too
        // old to have it, and two absences are not a change.
        const rebooted = bootIdAtStart !== undefined
          && info.boot_id !== undefined
          && info.boot_id !== bootIdAtStart;
        if (sawDown || rebooted) {
          window.location.reload();
          return;
        }
      } catch {
        sawDown = true;
      }
      if (generation !== restartGeneration) return;
      await sleep(RESTART_POLL_MS);
    }
  },

  checkInProgress: async () => {
    if (inProgressChecked) return;
    inProgressChecked = true;

    await get().refresh();
    if (!get().loaded) {
      // No answer arrived — `refresh` reports a network error and leaves
      // `loaded` false precisely so a later mount can try again. The flag
      // exists to keep two same-tick mounts from both fetching, NOT to make
      // one dropped packet permanent, so it is released here.
      //
      // The restart breadcrumb is untouched for the same reason: it is read
      // against `last_restart_job`, which only a catalog that ARRIVED can
      // carry. Consuming it now would swallow the one report a user who just
      // sat through a server restart is waiting for.
      inProgressChecked = false;
      return;
    }

    if (get().unsupported) {
      // A 404 IS an answer, so the once-per-load flag stays set and nothing
      // refetches — but it is not an answer about the RESTART. The outcome is
      // read off `last_restart_job`, which rides on the catalog this server
      // does not serve, so consuming the breadcrumb here would silently drop
      // the report a user who just sat through a restart is owed. It costs
      // nothing to leave: the key is scoped to this tab and dies with it, and
      // a server that can answer gets to report it instead.
      return;
    }

    const { t } = useI18n.getState();

    // A download the user started before reloading is still going, and the
    // Package Center is closed. This toast is the only thing that says so.
    if (followingJobId !== null) {
      toast(t('packs.toast.inProgress'), 'info');
    }

    const pending = readPending();
    if (pending === null) return;
    writePending(null);

    const record = lastRestartRecord;
    // `last_restart_job` carries no age bound — the record of an install that
    // finished an hour ago is still the one this reads. Not every reload the
    // handshake performs has a restart behind it (a lost poll during a
    // network blip ends the same way), so the record has to say it is about
    // THIS job before it is reported as this job's outcome. A breadcrumb or a
    // record without an id cannot disagree, and is believed: the ids are what
    // this checks, not their presence.
    const recordJob = str(record?.job_id);
    if (pending.jobId !== null && recordJob !== null
        && recordJob !== pending.jobId) {
      return;
    }
    const status = str(record?.status);
    const title = packTitle(pending.packId);
    if (status === 'ok') {
      toast(t('packs.restart.done', { pack: title }), 'success');
    } else if (status === 'failed') {
      const message = str(record?.message) ?? '';
      toast(t('packs.restart.failed', { pack: title, message }), 'error');
      // The helper that ran the install is gone, and so is its job log: this
      // record is the only account of what went wrong that survived the
      // restart. `message` is normally the whole story, so the tail is a
      // SECOND toast only when there is no story — an installer that died
      // without a message would otherwise report a bare colon and nothing
      // else. Both are `error` toasts, which this app never auto-dismisses,
      // so the text is still there when the user comes back to the tab.
      const tail = (str(record?.log_tail) ?? '').trim();
      if (message === '' && tail !== '') {
        // A toast is not a log viewer, and the writer is under no obligation
        // to keep this short: the last stretch is the part that says how it
        // ended, and the whole record is on disk for anyone who needs more.
        const log = tail.slice(-LOG_TAIL_TOAST_CHARS);
        toast(t('packs.restart.failedLog', { log }), 'error');
      }
    }
  },
}));

/**
 * Set one pack's status without rebuilding the index.
 *
 * Used to show "Installing" the instant the POST is accepted rather than at
 * the next catalog poll. Both views are updated together: a row and its
 * `byId` entry disagreeing is what puts a stale badge on a node.
 */
function setPackStatus(packId: string, status: PackSummary['status']): void {
  usePackStore.setState((state) => {
    const current = state.byId[packId];
    if (!current || current.status === status) return {};
    const updated = { ...current, status };
    return {
      packs: state.packs.map((pack) => (pack.id === packId ? updated : pack)),
      byId: { ...state.byId, [packId]: updated },
    };
  });
}

/** Test-only: reset the module-scope schedulers and the store between cases. */
export function _resetPackStoreForTesting(): void {
  stopFollowing();
  restartGeneration += 1;
  settledJobId = null;
  inProgressChecked = false;
  lastRestartRecord = null;
  usePackStore.setState({
    packs: [],
    byId: {},
    loading: false,
    loaded: false,
    unsupported: false,
    error: null,
    remoteInstallAllowed: true,
    launchMode: 'unknown',
    restartAvailable: false,
    gpu: null,
    job: null,
    busy: {},
    cancelling: false,
    restart: IDLE_RESTART,
  });
}
