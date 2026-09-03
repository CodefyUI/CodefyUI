import { create } from 'zustand';
import {
  PackApiError,
  cancelPackJob,
  fetchHealth,
  getPackJobEvents,
  installPack,
  listPacks,
  removePackItem,
  type JobEvent,
  type JobEventsPage,
  type LaunchMode,
  type PackCatalog,
  type PackGpuInfo,
  type PackInstallMode,
  type PackJobStatus,
  type PackSummary,
} from '../api/rest';
import {
  EVENT_WAIT_S,
  FOLLOW_IDLE_MS,
  FOLLOW_RETRY_MS,
  MAX_FOLLOW_FAILURES,
  MAX_LOG_LINES,
  createJobFollower,
  emptyJob,
  reduceJobEvents,
  type ItemProgress,
  type Job,
  type JobPhase,
  type JobStep,
  type LogLine,
} from './jobFollower';
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

/**
 * The follower's own constants, re-exported under the names this panel and
 * its tests have always used. `jobFollower` owns the values because the
 * plugin center parks on the same endpoint shape with the same deadlines;
 * a second copy here would be two numbers to keep in step.
 */
export { EVENT_WAIT_S, FOLLOW_IDLE_MS, FOLLOW_RETRY_MS, MAX_FOLLOW_FAILURES };

/** Ring-buffer bound on the rendered install log. */
export const MAX_PACK_LOG_LINES = MAX_LOG_LINES;

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
 * The generic job model under this panel's own names.
 *
 * Aliases rather than copies: a pack job's log lines, per-item bars, steps
 * and phases are the shared ones, and a second declaration would be a second
 * thing to keep in step with the reducer that produces them. The names stay
 * because the panel, the cards and their tests all read in packs.
 */
export type PackLogLine = LogLine;
export type PackItemProgress = ItemProgress;
export type PackJobStep = JobStep;
export type PackJobPhase = JobPhase;

/** A pack install job: the generic job plus what makes it a PACK's. */
export interface PackJob extends Job {
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
  return { ...emptyJob(jobId), packId, mode, retryMode: null };
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
 * What a pack reads out of an event that the generic reducer does not.
 *
 * One key today, and it is genuinely pack-shaped: `retry_mode` is the
 * server's offer to finish a LIVE install that its resolver stopped, and
 * only this panel has a button that can take it up.
 */
function packExtras(event: JobEvent, draft: PackJob): void {
  if (event.type !== 'needs_restart') return;
  // Only the one mode this build knows how to post. The key is absent
  // unless the server can actually restart itself, so reading anything
  // else as a retry would offer a button whose request nothing here
  // knows how to make.
  draft.retryMode = str(event.retry_mode) === 'restart' ? 'restart' : null;
}

/**
 * Fold one page of events into a pack job.
 *
 * Pure and exported so the interesting part — what each event type does to
 * the steps, the per-item bars and the log — is testable without a server, a
 * timer or a React tree. The fold itself is `jobFollower`'s, shared with the
 * plugin center; only `retry_mode` is ours.
 *
 * The follower below runs THIS function rather than being handed `packExtras`
 * separately, so what the tests exercise and what a live install folds are one
 * thing. Its page is the generic `JobEventsPage`: a `PackJobEventsPage` is one
 * of those, and the extra keys are read off the index signature anyway.
 */
export function reducePackEvents(job: PackJob, page: JobEventsPage): PackJob {
  return reduceJobEvents(job, page, packExtras);
}

// ── module-scope schedulers ──────────────────────────────────────────────
// In-flight requests and timers are process state, not store state: putting
// them in the store would make every turn of the loop a re-render for
// subscribers that only care about the data. The follower's half of that now
// lives inside the closure `createJobFollower` returns; what is left here is
// the restart handshake's.

/** The last job `onJobSettled` fired for — its side effects run exactly once. */
let settledJobId: string | null = null;

/**
 * The pack the running follower is installing, and the job it was named for.
 *
 * The follower is keyed by job id and knows nothing about packs, but every
 * ending it reports is about one: which catalog row carries the fallback
 * command, which name a toast says, which pack the reloaded page reports on.
 * Set immediately before each `start`, and safe as a single slot because
 * starting a different job replaces it and abandons the old loop in the same
 * breath — a stale loop never gets to read it.
 *
 * The job id travels with the pack id so that "the ending being reported is
 * the job this slot was written for" is CHECKED rather than assumed. It costs
 * a comparison, and it is what keeps an `await` slipped between the
 * assignment and `follower.start` from naming the wrong row on the toast.
 */
let following: { jobId: string; packId: string } | null = null;

/** The pack *jobId* is installing, or '' if it is not the one being followed. */
function followedPack(jobId: string): string {
  return following !== null && following.jobId === jobId ? following.packId : '';
}

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

/** Record on the open job — never on a job the user has moved on from. */
function patchJob(jobId: string, next: PackJob): void {
  usePackStore.setState((state) => (
    state.job && state.job.jobId === jobId ? { job: next } : {}
  ));
}

/**
 * The one follower this store runs.
 *
 * Everything about HOW a job is tailed — the long poll, the generation
 * guard, the retry budget, the "terminal AND the cursor did not move" stop
 * condition — is `jobFollower`'s, and is shared with the plugin center.
 * What is left here is the three things that are about PACKS: which endpoint
 * to ask, what an ending means, and what to read into the endpoint going
 * quiet.
 */
const follower = createJobFollower<PackJob>({
  fetchPage: (jobId, cursor, signal, wait) => (
    getPackJobEvents(jobId, { cursor, wait, signal })
  ),
  getOpenJob: () => usePackStore.getState().job,
  patchJob,
  reduce: reducePackEvents,
  onSettled: (jobId, status) => {
    onJobSettled(jobId, followedPack(jobId), status);
  },
  onGiveUp: (jobId, error, open) => {
    // A RESTART-mode job is the one case where the endpoint going quiet is
    // the expected ending rather than a lost connection: the server stops
    // accepting half a second after the 202 and then exits, and a LAN client
    // or a tab the browser had throttled can miss that half second entirely.
    // `lost` would leave the user in front of a "we no longer know" banner
    // while the wheel swap runs — no overlay, and worse, no breadcrumb, so
    // the page that comes back could not report how the install went.
    // Settling it as the ending it almost certainly reached runs the same
    // handshake the last page would have started; if it turns out the server
    // never went, the overlay's own thirty-second grace says so.
    //
    // Only for a connection that DROPPED, though. A `PackApiError` is the
    // server answering — a 404 for a job that aged out, a 500 — and an
    // answer is proof it is still there. Settling on those raised the
    // blocking overlay over a live server, which is the bug the note in
    // `onJobSettled` records as fixed. Every other ending declines here and
    // takes the follower's own `lost`.
    if (error instanceof PackApiError || open === null || open.mode !== 'restart') {
      return false;
    }
    const packId = followedPack(jobId);
    // No `needs_restart` event ever arrived, so nothing carried the command
    // — and both give-up screens name one ("Run this command, then
    // reload:"). The catalog has it: `install_command` is what the panel's
    // own button would have run for this pack.
    const command = usePackStore.getState().byId[packId]?.install_command ?? null;
    patchJob(jobId, {
      ...open,
      status: 'needs_restart',
      restartCommand: open.restartCommand ?? command,
    });
    onJobSettled(jobId, packId, 'needs_restart');
    return true;
  },
});

/** Abandon the current follower, if any. */
function stopFollowing(): void {
  follower.stop();
}

/**
 * Follow *jobId* from *cursor*, unless it is already being followed.
 *
 * The idempotence is what lets `refresh()` adopt `active_job` on every poll
 * without restarting the follower — and without the double-follow that
 * would apply every event twice. Asked here as well as inside `start` so the
 * pack id is not rewritten under a loop already running on another job.
 */
function startFollowing(jobId: string, packId: string, cursor: number): void {
  if (follower.followingJobId() === jobId) return;
  following = { jobId, packId };
  follower.start(jobId, cursor);
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
      && follower.followingJobId() !== job.jobId
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
    if (follower.followingJobId() !== null) {
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
  follower.stop();
  restartGeneration += 1;
  settledJobId = null;
  following = null;
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
