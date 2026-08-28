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
import { useToastStore } from './toastStore';
import { useI18n } from '../i18n';

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
 * Consecutive failures before the job is declared `lost`.
 *
 * Five retries across ten seconds outlast a restarting dev server and a
 * flaky Wi-Fi hop, which are the two things that interrupt an install on a
 * developer's machine. Beyond that the honest thing to say is that we no
 * longer know what the download is doing.
 */
export const MAX_FOLLOW_FAILURES = 5;

/** Ring-buffer bound on the rendered install log. */
export const MAX_PACK_LOG_LINES = 400;

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

/** A job with nothing in it yet — what an adopted or just-started job starts as. */
export function emptyPackJob(jobId: string, packId: string): PackJob {
  return {
    jobId,
    packId,
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

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function toast(message: string, type: 'info' | 'error' | 'success' | 'warning') {
  useToastStore.getState().addToast(message, type);
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
 * `sessionStorage` can throw outright — Safari in private mode, and any
 * browser configured to block site data. Losing the restart breadcrumb costs
 * one missing toast; letting it throw would abandon the restart itself.
 */
function readPending(): string | null {
  try {
    return sessionStorage.getItem(RESTART_PENDING_KEY);
  } catch {
    return null;
  }
}

function writePending(packId: string | null): void {
  try {
    if (packId === null) sessionStorage.removeItem(RESTART_PENDING_KEY);
    else sessionStorage.setItem(RESTART_PENDING_KEY, packId);
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
    } catch {
      // A deliberate abort bumped the generation; anything else is the
      // network, which an install is entitled to survive — the download
      // itself is happening server-side and does not care that we blinked.
      if (generation !== followGeneration) return;
      failures += 1;
      if (failures >= MAX_FOLLOW_FAILURES) {
        patchJob(jobId, (job) => ({ ...job, status: 'lost' }));
        if (followingJobId === jobId) followingJobId = null;
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

/** A pack's human name, falling back to its id before the catalog loads. */
function packTitle(packId: string): string {
  return usePackStore.getState().byId[packId]?.title ?? packId;
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
        `${t('packs.toast.installFailed')}: ${settled?.error?.message ?? ''}`,
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
      if (store.launchMode === 'start') {
        // Parked BEFORE the reload so the page that comes back knows which
        // pack to report on — the job itself does not survive the reload.
        writePending(packId);
        void store.restartFlow(packId, command);
      } else {
        // `cdui dev` has no supervisor to swap the wheel and relaunch. The
        // job stays on screen: it is what the command block describes.
        toast(t('packs.toast.devRestart'), 'warning');
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
        set({ job: emptyPackJob(active.job_id, active.pack_id) });
      }
      // Idempotent: this runs on every poll, and only the FIRST one starts a
      // follower. This is how a job started in another tab is adopted.
      startFollowing(active.job_id, active.pack_id, get().job?.cursor ?? 0);
      return;
    }

    const job = get().job;
    if (job && job.status === 'running' && jobBefore?.jobId === job.jobId) {
      // The server has no record of a job we think is running: it restarted,
      // or the job aged out. Saying "running" would be the one answer that
      // is definitely wrong.
      set({ job: { ...job, status: 'lost' } });
      if (followingJobId === job.jobId) stopFollowing();
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
      const job = emptyPackJob(job_id, packId);
      for (const item of requested) {
        job.items[item] = { bytesDone: 0, bytesTotal: null, percent: 0 };
      }
      set({ job });
      setPackStatus(packId, 'installing');
      startFollowing(job_id, packId, 0);
    } catch (err) {
      if (err instanceof PackApiError && err.status === 409) {
        // Somebody else got there first — this tab, another tab, or the CLI.
        // The refresh adopts whatever the server IS running, which is more
        // useful than the refusal.
        toast(t('packs.toast.busy'), 'warning');
        await get().refresh();
      } else if (err instanceof PackApiError && err.status === 403) {
        toast(t('packs.toast.remoteNotAllowed'), 'error');
      } else if (err instanceof PackApiError && err.status === 400
                 && Array.isArray(err.body?.blocked_by)
                 && err.body.blocked_by.length > 0) {
        const first = String(err.body.blocked_by[0]);
        toast(t('packs.toast.blocked', { pack: packTitle(first) }), 'warning');
      } else {
        toast(`${t('packs.toast.installFailed')}: ${errorMessage(err)}`, 'error');
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
      toast(`${useI18n.getState().t('packs.toast.cancelFailed')}: ${errorMessage(err)}`,
        'error');
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
      toast(
        `${t('packs.item.removeFailed', { item: itemId })}: ${errorMessage(err)}`,
        'error',
      );
    }
    // Either way the disk is not what the catalog says it is.
    await get().refresh();
  },

  followJob: (jobId, packId, cursor = 0) => {
    const current = get().job;
    if (!current || current.jobId !== jobId) {
      set({ job: emptyPackJob(jobId, packId) });
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
    try {
      bootIdAtStart = (await fetchHealth()).boot_id;
    } catch {
      // Already down. `sawDown` picks it up on the first turn.
    }

    let sawDown = false;
    while (generation === restartGeneration) {
      // Wall clock, never a tick count: a sleeping laptop fires no timers,
      // and an overlay that counts turns would still be up an hour later.
      const elapsed = Date.now() - startedAt;
      if (elapsed > RESTART_TIMEOUT_MS) {
        set((state) => ({ restart: { ...state.restart, phase: 'timeout' } }));
        return;
      }
      if (!sawDown && elapsed > RESTART_GRACE_MS) {
        // Half a minute and the server never even flinched: nothing picked
        // the restart up, and the user needs the command, not a spinner.
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
    const status = str(record?.status);
    const title = packTitle(pending);
    if (status === 'ok') {
      toast(t('packs.restart.done', { pack: title }), 'success');
    } else if (status === 'failed') {
      toast(
        t('packs.restart.failed', {
          pack: title,
          message: str(record?.message) ?? '',
        }),
        'error',
      );
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
    gpu: null,
    job: null,
    busy: {},
    cancelling: false,
    restart: IDLE_RESTART,
  });
}
