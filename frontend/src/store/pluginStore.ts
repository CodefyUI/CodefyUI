import { create } from 'zustand';
import {
  ApiError,
  cancelPluginJob,
  errorDetail,
  getPluginJobEvents,
  inspectPluginSource,
  installPlugin,
  listPluginCatalog,
  setPluginEnabled,
  uninstallPlugin,
  updatePlugin,
  type JobStatus,
  type PluginCatalog,
  type PluginCatalogEntry,
  type PluginInspection,
  type PluginJobKind,
} from '../api/rest';
import { createJobFollower, emptyJob, type Job } from './jobFollower';
import { confirm } from '../utils/dialog';
import { reloadPluginFrontends } from '../plugins/PluginHost';
import { useNodeDefStore } from './nodeDefStore';
import { errorMessage, str, toast } from './storeText';
import type { ToastAction } from './toastStore';
import { useUIStore } from './uiStore';
import { useI18n, type TranslationKey } from '../i18n';

/**
 * App-level state for the Plugin Center.
 *
 * The same shape as `packStore`, and for the same reason: an install is a
 * download, an unpack and a pip resolve that outlives the modal it was
 * started from. The panel is a pure view of this store, so closing it, opening
 * a second tab or reloading the page all leave the job running and pick it
 * back up.
 *
 * ── What is different from a pack ──────────────────────────────────────
 * A plugin install changes what the SERVER can do while it is still running:
 * new node types, new frontend bundles, a changed registry generation. So
 * every settled change ends in the same three steps -- the catalog, the node
 * definitions, the plugin frontends -- rather than in a restart handshake.
 * Nothing here swaps a wheel out from under the interpreter, which is why
 * this store has no restart mode and its follower declines to read anything
 * into an endpoint going quiet.
 */

/** A plugin install or update job: the generic job plus whose it is. */
export interface PluginJob extends Job {
  pluginId: string;
  /** Decides which toast the ending gets: installed, or updated. */
  kind: PluginJobKind;
}

export type { PluginJobKind };

/**
 * What `parseGitHubSource` made of what the user typed.
 *
 * `catalog` is a bare word, which can only have been meant as one of the
 * built-in packs; the server is what decides whether that name exists.
 */
export type PluginSourceRef =
  | { kind: 'github'; owner: string; repo: string; ref: string | null }
  | { kind: 'catalog'; id: string };

/**
 * Why an inspection or an install was refused, in the three parts a review
 * card needs.
 *
 * Not a string, because the backend's 400 bodies carry no message at all --
 * `{code: reserved_id, id}`, `{code: unknown_catalog_name, known}`,
 * `{code: consent_required, missing_capabilities}`, `{code:
 * trust_author_required, allowed_modules}`. Reduced to `err.message` those
 * become the raw token `consent_required` on screen, and the keys that say
 * WHICH id is taken, which names exist, or which box is still unticked are
 * gone by the time the card is rendered.
 *
 * `message` is what to show when nothing better is known, `code` is what a
 * card switches on, and `detail` is the rest of that body.
 */
export interface InspectionFailure {
  message: string;
  code: string | null;
  detail: Record<string, unknown> | null;
}

/**
 * The install review, as a little state machine.
 *
 * An inspection is not an install: it reads the manifest at a source and
 * reports what installing it WOULD cost, so the user consents to the exact
 * version that was read. `forPluginId` is the catalog row the review was
 * started from -- null when the user typed a source by hand, which is what
 * lets the panel show the review beside that row or in the source box.
 */
export type InspectionState =
  | { phase: 'idle' }
  | { phase: 'inspecting'; source: string }
  | {
      phase: 'ready';
      data: PluginInspection;
      source: string;
      forPluginId: string | null;
      kind: PluginJobKind;
      /**
       * A refusal the review can still recover from: the server said the
       * consent was incomplete. The inspection STAYS ready, because the fix
       * is a tick box on the card in front of the user rather than a new
       * inspection.
       */
      error: InspectionFailure | null;
    }
  | { phase: 'error'; source: string; failure: InspectionFailure };

interface PluginState {
  plugins: PluginCatalogEntry[];
  /** The same rows keyed by id, for the O(1) lookups a toast or a card does. */
  byId: Record<string, PluginCatalogEntry>;
  loading: boolean;
  /** A first answer arrived — a catalog or a 404. A network error is neither. */
  loaded: boolean;
  /** The server predates the Plugin Center: the panel says so and stops. */
  unsupported: boolean;
  error: string | null;
  remoteInstallAllowed: boolean;
  /** The node registry's reload counter, as of the last catalog read. */
  generation: number;
  job: PluginJob | null;
  /** Plugin ids with a request in flight — disables that row's buttons. */
  busy: Record<string, boolean>;
  cancelling: boolean;
  inspection: InspectionState;

  refresh: () => Promise<void>;
  /** Inspect *pluginId* and install it straight away if nothing needs consent. */
  install: (pluginId: string) => Promise<void>;
  /** Inspect whatever the user typed: a catalog name, owner/repo, or a URL. */
  inspect: (source: string) => Promise<void>;
  installInspected: (opts: {
    acceptCapabilities: boolean;
    trustAuthor: boolean;
    /**
     * Reinstall over what is already there. Never needed for an update (the
     * inspection's own `mode` carries that) and never sent unless a caller
     * asks, so nothing here can overwrite an install by accident.
     */
    force?: boolean;
  }) => Promise<void>;
  clearInspection: () => void;
  update: (pluginId: string) => Promise<void>;
  uninstall: (pluginId: string) => Promise<void>;
  setEnabled: (pluginId: string, enabled: boolean) => Promise<void>;
  cancel: () => Promise<void>;
  /** Adopt *jobId* and start (or keep) following it. Idempotent per job id. */
  followJob: (
    jobId: string, pluginId: string, kind?: PluginJobKind, cursor?: number,
  ) => void;
  stopFollowing: () => void;
  /** Clear a finished job from the activity pane. Ignored while running. */
  dismissJob: () => void;
  /** Once per page load: adopt a job that outlived the last page. */
  checkInProgress: () => Promise<void>;
}

/** A job with nothing in it yet — what an adopted or just-started job starts as. */
export function emptyPluginJob(
  jobId: string,
  pluginId: string,
  kind: PluginJobKind = 'install',
): PluginJob {
  return { ...emptyJob(jobId), pluginId, kind };
}

// ── source parsing ───────────────────────────────────────────────────────

// The CLI's three forms, mirrored (`scripts/plugins.py: parse_source`): the
// point is to refuse an unparseable source without a round trip, not to
// resolve it -- the SERVER decides what a name or a repo actually is.
const GITHUB_URL =
  /^https?:\/\/(?:www\.)?github\.com\/([\w.-]+)\/([\w.-]+?)(?:\.git)?\/?(?:@(.+))?$/;
const GITHUB_SHORT = /^([\w.-]+)\/([\w.-]+?)(?:@([\w./-]+))?$/;
/**
 * One word, no slash and no scheme: it can only be a catalog name.
 *
 * The CLI's own class, character for character, because refusing here means
 * refusing WITHOUT asking the server: a narrower rule would turn `EDU` or
 * `C1` into "that is not a source" when the catalog has them. The lookup is
 * case-insensitive there (`spec.lower()`), so the id is lower-cased to match.
 */
const CATALOG_NAME = /^[A-Za-z0-9][\w.-]*$/;

/**
 * Read a plugin source the way the CLI reads one, or answer null.
 *
 * Exported because the panel wants it too: a source box that can say "that is
 * not a repo" as you type is the difference between one wrong keystroke and a
 * round trip that ends in a 400.
 */
export function parseGitHubSource(input: string): PluginSourceRef | null {
  const spec = input.trim();

  const url = GITHUB_URL.exec(spec);
  if (url !== null) {
    return { kind: 'github', owner: url[1], repo: url[2], ref: url[3] ?? null };
  }
  const short = GITHUB_SHORT.exec(spec);
  if (short !== null) {
    return { kind: 'github', owner: short[1], repo: short[2], ref: short[3] ?? null };
  }
  // Lower-cased, not echoed: the server looks a catalog name up in lower
  // case, so `EDU` and `edu` are the same pack and the panel should not show
  // them as two.
  if (CATALOG_NAME.test(spec)) return { kind: 'catalog', id: spec.toLowerCase() };
  return null;
}

// ── helpers ──────────────────────────────────────────────────────────────

/**
 * The button a toast about a plugin wears: it opens the panel on that plugin.
 *
 * An empty id opens the panel with no focus rather than scrolling to a row
 * called "": a job adopted from another tab can settle before any catalog
 * read has said whose it was.
 */
function openCenterAction(pluginId: string): ToastAction {
  const { t } = useI18n.getState();
  return {
    label: t('pluginCenter.toast.openCenter'),
    onClick: () => useUIStore.getState().openPluginCenter(pluginId || undefined),
  };
}

/**
 * The code a refusal was refused with, or null.
 *
 * `errorDetail` is the api client's, not a copy: every plugin route nests its
 * keys under `detail`, and one unwrapper shared with the panel is what keeps a
 * caller from reading `err.body.code` and finding nothing there.
 */
function refusalCode(err: unknown): string | null {
  return str(errorDetail(err)?.code);
}

/**
 * The refusals whose code IS the whole message, and what to say instead.
 *
 * Every route here answers `HTTPException(status, detail={"code": ...})` with
 * deliberately no `message`, so `readApiError` falls back to the code and
 * `err.message` is the raw token: a student is shown `inspection_expired`.
 * These three are the ones with a fix worth naming; `already_installed` is
 * not here because it is answered with a button rather than a sentence, and
 * everything else keeps the server's own message, which is at least true.
 */
const REFUSAL_KEY: Record<string, TranslationKey | undefined> = {
  unavailable: 'pluginCenter.error.unavailable',
  inspection_expired: 'pluginCenter.error.inspectionExpired',
  unknown_job: 'pluginCenter.error.unknownJob',
};

/** What a refusal should read as, once its code has had its say. */
function refusalMessage(err: unknown): string {
  const key = REFUSAL_KEY[refusalCode(err) ?? ''];
  return key === undefined ? errorMessage(err) : useI18n.getState().t(key);
}

/**
 * Everything a thrown refusal carried, in the shape the review card reads.
 *
 * One builder rather than a shape assembled at each catch, so no caller can
 * be the one that keeps only the message -- which is the state this replaces.
 */
function inspectionFailure(err: unknown): InspectionFailure {
  return {
    // 403 is the one status whose own message says nothing a user can act
    // on ("Forbidden"): it is the server refusing to install from anywhere
    // but the machine it runs on, which only this key explains.
    message: err instanceof ApiError && err.status === 403
      ? useI18n.getState().t('packs.remoteDisabled')
      : refusalMessage(err),
    code: refusalCode(err),
    detail: errorDetail(err),
  };
}

/**
 * Leave *err* on the review the user is looking at, if there still is one.
 *
 * The refusals answered ON the card -- an unticked capability, an untrusted
 * author, a plugin that turned out to be installed already -- keep the
 * inspection ready and grow an error, because the fix is a control on that
 * card rather than a new inspection. A review that is no longer ready (a
 * second tab cleared it) drops the refusal rather than resurrecting a card
 * for a decision nobody is making.
 */
function attachInspectionFailure(err: unknown): void {
  usePluginStore.setState((state) => (
    state.inspection.phase === 'ready'
      ? { inspection: { ...state.inspection, error: inspectionFailure(err) } }
      : {}
  ));
}

/**
 * Run *fn* with *pluginId* marked busy, and unmark it however *fn* ends.
 *
 * Every action that acts on one plugin disables that row for the length of
 * the request, and every one of them has to release it on a throw as well as
 * on a return. Five copies of the same `finally` are five chances to leave a
 * row disabled with nothing running behind it.
 */
async function withBusy<T>(pluginId: string, fn: () => Promise<T>): Promise<T> {
  usePluginStore.setState((s) => ({ busy: { ...s.busy, [pluginId]: true } }));
  try {
    return await fn();
  } finally {
    usePluginStore.setState((s) => {
      const busy = { ...s.busy };
      delete busy[pluginId];
      return { busy };
    });
  }
}

/** A plugin's display name for a toast, falling back to its id. */
function pluginName(pluginId: string): string {
  return usePluginStore.getState().byId[pluginId]?.name || pluginId;
}

/** Only `running` is live; everything else — `lost` included — is an ending. */
function isRunning(job: PluginJob | null): boolean {
  return job !== null && job.status === 'running';
}

/**
 * A job kind off the wire, narrowed. Anything a newer backend invents reads
 * as an install, which is the ending that says "installed" rather than one
 * this build has no toast for.
 */
function jobKind(value: unknown): PluginJobKind {
  return value === 'update' ? 'update' : 'install';
}

// ── module-scope schedulers ──────────────────────────────────────────────

/** The last job `onJobSettled` fired for — its side effects run exactly once. */
let settledJobId: string | null = null;

/**
 * Whose job the follower is on, and what it is doing.
 *
 * The follower is keyed by job id and knows nothing about plugins, but every
 * ending it reports is about one: which name a toast says, which row the
 * button on that toast opens. Set immediately before each `start`, and safe as
 * a single slot because starting a different job replaces it and abandons the
 * old loop in the same breath.
 */
let followingPluginId: string | null = null;
let followingKind: PluginJobKind = 'install';

/** Once per page load — StrictMode mounts effects twice in development. */
let inProgressChecked = false;

/** Record on the open job — never on a job the user has moved on from. */
function patchJob(jobId: string, next: PluginJob): void {
  usePluginStore.setState((state) => (
    state.job && state.job.jobId === jobId ? { job: next } : {}
  ));
}

/**
 * The one follower this store runs. Everything about HOW a job is tailed is
 * `jobFollower`'s, shared with the Package Center.
 */
const follower = createJobFollower<PluginJob>({
  fetchPage: (jobId, cursor, signal, wait) => (
    getPluginJobEvents(jobId, { cursor, wait, signal })
  ),
  getOpenJob: () => usePluginStore.getState().job,
  patchJob,
  onSettled: (jobId, status) => {
    // Nothing awaits this, so a throw between its awaits would surface as an
    // unhandled rejection in a console the user never opens. Every step of the
    // refresh is already guarded; this is the net under the rest of it.
    void onJobSettled(jobId, followingPluginId ?? '', followingKind, status)
      .catch((err: unknown) => {
        toast(
          useI18n.getState().t(
            'pluginCenter.toast.refreshFailed', { message: errorMessage(err) },
          ),
          'error',
        );
      });
  },
  // Nothing here restarts the server, so an events endpoint that stops
  // answering is exactly what it looks like: we no longer know what the job is
  // doing. Declining leaves the follower's own `lost`, which is the honest
  // answer — the pack store's restart branch has no counterpart for plugins.
  onGiveUp: () => false,
});

/** Abandon the current follower, if any. */
function stopFollowing(): void {
  follower.stop();
}

/**
 * Follow *jobId* from *cursor*, unless it is already being followed.
 *
 * The idempotence is what lets `refresh()` adopt `active_job` on every poll
 * without restarting the follower — and without the double-follow that would
 * apply every event twice.
 *
 * Asked here as well as inside `start` because the two module slots below are
 * not the follower's to guard: a second call for the job ALREADY being
 * followed keeps the first call's plugin id and kind, and does not restart
 * the loop. A call naming a different job replaces both, and the loop with
 * them.
 */
function startFollowing(
  jobId: string, pluginId: string, kind: PluginJobKind, cursor: number,
): void {
  if (follower.followingJobId() === jobId) return;
  followingPluginId = pluginId;
  followingKind = kind;
  follower.start(jobId, cursor);
}

/**
 * Everything that has to happen after a plugin's files changed on disk.
 *
 * Three steps, in this order and one at a time. The catalog first, because the
 * rest of the UI reads statuses off it. Then the node definitions -- with
 * `fetchDefinitions`, NOT `reload`: the server rediscovered its own nodes when
 * the job finished, and asking it to do it again would throw away a warm
 * registry for nothing. Then the plugin frontends, which have to come last
 * because a bundle activates against the definitions that are now loaded.
 *
 * Every step is guarded on its own. A catalog that did not come back is no
 * reason to leave a just-installed plugin's UI unloaded, and vice versa.
 */
async function refreshEverything(): Promise<void> {
  const steps: (() => Promise<unknown>)[] = [
    () => usePluginStore.getState().refresh(),
    () => useNodeDefStore.getState().fetchDefinitions(),
    () => reloadPluginFrontends(),
  ];
  for (const step of steps) {
    try {
      await step();
    } catch (err) {
      // Its own key, not `installFailed`: this runs after an uninstall and
      // after a disable too, and "install failed" beside "demo uninstalled."
      // describes something that did not happen. What DID happen is that the
      // change landed and the editor could not be brought up to date.
      toast(
        useI18n.getState().t(
          'pluginCenter.toast.refreshFailed', { message: errorMessage(err) },
        ),
        'error',
      );
    }
  }
}

/**
 * Everything that happens once, when a job reaches its end.
 *
 * Guarded by job id rather than trusted to the caller: `refresh()` can adopt a
 * job at any time, and a settled job announcing itself twice would toast twice
 * and run the three-step refresh twice.
 */
async function onJobSettled(
  jobId: string, pluginId: string, kind: PluginJobKind, status: JobStatus,
): Promise<void> {
  if (settledJobId === jobId) return;
  settledJobId = jobId;

  const { t } = useI18n.getState();
  const store = usePluginStore.getState();
  const name = pluginName(pluginId);
  // The page that ended the job was applied to `job` a moment ago — but only
  // if it is still the open one, so what is read back is checked rather than
  // assumed. A dismissed job must not lend its message to this one.
  const settled = store.job?.jobId === jobId ? store.job : null;

  switch (status) {
    case 'done':
      toast(
        t(kind === 'update'
          ? 'pluginCenter.toast.updated'
          : 'pluginCenter.toast.installed', { plugin: name }),
        'success',
      );
      await refreshEverything();
      break;
    case 'failed':
      // Only the catalog: nothing landed, so the node definitions and the
      // frontends are exactly what they were.
      toast(
        t('packs.toast.installFailed', { message: settled?.error?.message ?? '' }),
        'error',
        openCenterAction(pluginId),
      );
      await store.refresh();
      break;
    case 'cancelled':
      toast(t('packs.toast.cancelled'), 'info');
      await store.refresh();
      break;
    case 'needs_restart':
      // The files are in place and the registry could not pick them up from
      // inside the running process. The JOB stays on screen — the panel's
      // banner is what renders the command — and the catalog is refreshed
      // because the row's status changed even though the nodes did not.
      toast(
        t('pluginCenter.toast.needsRestart', { plugin: name }),
        'warning',
        openCenterAction(pluginId),
      );
      await store.refresh();
      break;
    default:
      break;
  }
}

/**
 * Apply a whole catalog. The ONLY place `byId` is rebuilt.
 */
function setCatalog(catalog: PluginCatalog): void {
  const byId: Record<string, PluginCatalogEntry> = {};
  for (const entry of catalog.entries) byId[entry.id] = entry;
  usePluginStore.setState({
    plugins: catalog.entries,
    byId,
    remoteInstallAllowed: catalog.remote_install_allowed,
    generation: catalog.generation,
    loading: false,
    loaded: true,
    unsupported: false,
    error: null,
  });
}

/**
 * Set one plugin's status without rebuilding the index.
 *
 * Used to show "Installing" the instant the 202 lands rather than at the next
 * catalog poll. Both views are updated together: a row and its `byId` entry
 * disagreeing is what puts a stale badge on a card.
 */
function setPluginStatus(pluginId: string, status: PluginCatalogEntry['status']): void {
  usePluginStore.setState((state) => {
    const current = state.byId[pluginId];
    if (!current || current.status === status) return {};
    const updated = { ...current, status };
    return {
      plugins: state.plugins.map((entry) => (entry.id === pluginId ? updated : entry)),
      byId: { ...state.byId, [pluginId]: updated },
    };
  });
}

/**
 * Inspect *source* and leave the review in whatever state it reached.
 *
 * Returns the inspection so `install` can act on it without re-reading the
 * store. *forPluginId* is the catalog row this was started from, or null for
 * a source the user typed.
 */
async function runInspect(
  source: string, forPluginId: string | null,
): Promise<PluginInspection | null> {
  const { t } = useI18n.getState();
  const spec = source.trim();

  if (parseGitHubSource(spec) === null) {
    // Refused without a round trip: the server would answer 400
    // `unparseable_source`, and this is the one error the client can be sure
    // of on its own.
    usePluginStore.setState({
      inspection: {
        phase: 'error',
        source: spec,
        // No code: nothing was refused, this build simply knows the shape is
        // not one the server could resolve.
        failure: {
          message: t('pluginCenter.source.invalid'), code: null, detail: null,
        },
      },
    });
    return null;
  }

  usePluginStore.setState({ inspection: { phase: 'inspecting', source: spec } });
  try {
    const data = await inspectPluginSource(spec);
    usePluginStore.setState({
      inspection: {
        phase: 'ready', data, source: spec, forPluginId, kind: data.mode, error: null,
      },
    });
    return data;
  } catch (err) {
    usePluginStore.setState({
      inspection: { phase: 'error', source: spec, failure: inspectionFailure(err) },
    });
    return null;
  }
}

/**
 * Post an inspected install and adopt the job it starts.
 *
 * Shared by the auto-install path (`install` on a plugin that asks for
 * nothing) and the review card's confirm (`installInspected`), so both send
 * exactly the same body and read exactly the same refusals.
 */
async function startInstall(
  data: PluginInspection,
  opts: { acceptCapabilities: boolean; trustAuthor: boolean; force?: boolean },
): Promise<void> {
  const { t } = useI18n.getState();
  const store = usePluginStore.getState();

  try {
    const { job_id } = await installPlugin({
      inspection_id: data.inspection_id,
      // The DECLARED list, never a blanket "yes". The server compares what
      // comes back with what the manifest asks for and refuses a mismatch,
      // which is the whole point: a client that echoed a boolean would be
      // consenting on the user's behalf to whatever the source grows next.
      ...(opts.acceptCapabilities ? { accept_capabilities: data.capabilities } : {}),
      ...(opts.trustAuthor ? { trust_author: true } : {}),
      ...(opts.force ? { force: true } : {}),
    });

    usePluginStore.setState({
      job: emptyPluginJob(job_id, data.plugin_id, data.mode),
      inspection: { phase: 'idle' },
    });
    setPluginStatus(data.plugin_id, 'installing');
    startFollowing(job_id, data.plugin_id, data.mode, 0);
  } catch (err) {
    const code = refusalCode(err);
    if (code === 'consent_required' || code === 'trust_author_required') {
      // Recoverable, and recoverable on the card the user is already looking
      // at: the inspection stays ready and grows a failure, so the review can
      // say which box is still unticked -- the refusal names the capabilities
      // or the modules -- instead of starting over.
      attachInspectionFailure(err);
    } else if (code === 'already_installed') {
      // Not a failure of anything: an OFFER, and the backend says so in as
      // many words. The install this refused is the one the user asked for,
      // so the review stays up carrying the code and the card grows a
      // Reinstall button -- `installInspected({force: true})`, which spends
      // this same inspection rather than reading the source again. Checked
      // before the 409 below, which would otherwise swallow it as "another
      // install is already running" and refresh the offer away.
      attachInspectionFailure(err);
    } else if (err instanceof ApiError && err.status === 409) {
      // Somebody else got there first — this tab, another tab, or the CLI.
      // The refresh adopts whatever the server IS running, which is more
      // useful than the refusal.
      toast(t('packs.toast.busy'), 'warning');
      await store.refresh();
    } else if (err instanceof ApiError && err.status === 403) {
      toast(t('packs.remoteDisabled'), 'error');
    } else {
      toast(t('packs.toast.installFailed', { message: refusalMessage(err) }), 'error');
    }
  }
}

export const usePluginStore = create<PluginState>((set, get) => ({
  plugins: [],
  byId: {},
  loading: false,
  loaded: false,
  unsupported: false,
  error: null,
  remoteInstallAllowed: true,
  generation: 0,
  job: null,
  busy: {},
  cancelling: false,
  inspection: { phase: 'idle' },

  refresh: async () => {
    set({ loading: true });
    // Read BEFORE the request: a job that appears while this one is in flight
    // is not described by the answer, and must not be declared lost on the
    // strength of it.
    const jobBefore = get().job;

    let catalog: PluginCatalog;
    try {
      catalog = await listPluginCatalog();
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        // A server older than the Plugin Center. Not an error the user did
        // anything about, so the panel reports it silently and nothing else
        // in the app changes.
        set({
          loading: false, loaded: true, unsupported: true,
          plugins: [], byId: {}, error: null,
        });
        return;
      }
      // `loaded` stays as it was: no answer arrived, so a later mount is
      // entitled to try again. The rows already on screen are kept — a
      // dropped packet is no reason to blank the catalog.
      set({ loading: false, error: errorMessage(err) });
      return;
    }

    setCatalog(catalog);

    const active = catalog.active_job;
    if (active !== null) {
      const current = get().job;
      const kind = jobKind(active.kind);
      if (!current || current.jobId !== active.job_id) {
        set({ job: emptyPluginJob(active.job_id, active.plugin_id, kind) });
      }
      // Idempotent: this runs on every poll, and only the FIRST one starts a
      // follower. This is how a job started in another tab is adopted.
      startFollowing(active.job_id, active.plugin_id, kind, get().job?.cursor ?? 0);
      return;
    }

    const job = get().job;
    if (
      job
      && job.status === 'running'
      && jobBefore?.jobId === job.jobId
      // ...and nobody is watching it. A follower still parked on the events
      // endpoint has a better answer coming than this catalog read does:
      // `active_job` goes null the moment a job finishes, and the follower's
      // next page settles it. Marking it `lost` from here is a race that turns
      // a successful install into a "we lost contact" banner.
      && follower.followingJobId() !== job.jobId
    ) {
      // The server has no record of a job we think is running: it restarted,
      // or the job aged out. Saying "running" would be the one answer that is
      // definitely wrong.
      set({ job: { ...job, status: 'lost' } });
    }
  },

  install: async (pluginId) => {
    const state = get();
    if (state.busy[pluginId]) return;
    const { t } = useI18n.getState();

    if (isRunning(state.job)) {
      // One job at a time, server-side. Said here rather than after an
      // inspection the user would then have to abandon.
      toast(t('packs.toast.busy'), 'warning');
      return;
    }

    await withBusy(pluginId, async () => {
      const data = await runInspect(pluginId, pluginId);
      if (data === null) return;
      if (data.consent_required) {
        // Capabilities to grant, or an author to trust. The review card takes
        // it from here; nothing is installed until the user says so.
        return;
      }
      // Nothing to consent to, so asking would be a dialog whose only answer
      // is yes. `trustAuthor` stays false for the same reason: an install
      // with no `allowed_modules` has no author to trust.
      await startInstall(data, { acceptCapabilities: true, trustAuthor: false });

      // The review this install passed through was never on screen. If the
      // install failed for something a review cannot fix — busy, refused,
      // offline — putting the card up now would answer a toast with a form
      // nobody asked for. A consent refusal is the exception: it left a
      // failure on the review precisely because ticking a box is the fix.
      const after = get().inspection;
      if (after.phase === 'ready' && after.error === null
          && after.forPluginId === pluginId) {
        set({ inspection: { phase: 'idle' } });
      }
    });
  },

  inspect: async (source) => {
    // No busy flag: a typed source names no row yet, so there is no button to
    // disable. `inspection.phase` is what the source box reads instead.
    await runInspect(source, null);
  },

  installInspected: async (opts) => {
    const inspection = get().inspection;
    if (inspection.phase !== 'ready') return;
    const pluginId = inspection.data.plugin_id;
    if (get().busy[pluginId]) return;

    await withBusy(pluginId, () => startInstall(inspection.data, opts));
  },

  clearInspection: () => set({ inspection: { phase: 'idle' } }),

  update: async (pluginId) => {
    const state = get();
    if (state.busy[pluginId]) return;
    const { t } = useI18n.getState();

    if (isRunning(state.job)) {
      toast(t('packs.toast.busy'), 'warning');
      return;
    }

    await withBusy(pluginId, async () => {
      try {
        const result = await updatePlugin(pluginId);
        if (result.kind === 'up_to_date') {
          toast(
            t('pluginCenter.toast.upToDate', { plugin: pluginName(pluginId) }), 'info',
          );
          return;
        }
        if (result.kind === 'needs_consent') {
          // The new version asks for something the installed one did not. Same
          // review card as a fresh install, with `capabilities_added` the only
          // part that is actually new.
          set({
            inspection: {
              phase: 'ready',
              data: result.inspection,
              source: pluginId,
              forPluginId: pluginId,
              kind: 'update',
              error: null,
            },
          });
          return;
        }
        set({ job: emptyPluginJob(result.job_id, pluginId, 'update') });
        setPluginStatus(pluginId, 'installing');
        startFollowing(result.job_id, pluginId, 'update', 0);
      } catch (err) {
        if (err instanceof ApiError && err.status === 409) {
          toast(t('packs.toast.busy'), 'warning');
          await get().refresh();
        } else if (err instanceof ApiError && err.status === 403) {
          toast(t('packs.remoteDisabled'), 'error');
        } else {
          toast(t('pluginCenter.updateFailed', { message: refusalMessage(err) }), 'error');
        }
      }
    });
  },

  uninstall: async (pluginId) => {
    const { t } = useI18n.getState();
    if (get().busy[pluginId]) return;
    // Read before the request: a github plugin's row is gone by the time the
    // toast is written, and "uninstalled foo-plugin" reads better than the id
    // the catalog no longer has a name for.
    const name = pluginName(pluginId);

    const ok = await confirm({
      title: t('pluginCenter.uninstall'),
      message: t('pluginCenter.uninstallConfirm', { plugin: name }),
      confirmText: t('pluginCenter.uninstall'),
      variant: 'danger',
    });
    if (!ok) return;

    await withBusy(pluginId, async () => {
      try {
        await uninstallPlugin(pluginId);
        // Its nodes are gone from the registry and its bundle is still
        // activated in this page: all three steps, or the canvas keeps
        // offering nodes the server will refuse to run.
        await refreshEverything();
        toast(t('pluginCenter.toast.removed', { plugin: name }), 'success');
      } catch (err) {
        const hint = str(errorDetail(err)?.hint);
        if (refusalCode(err) === 'files_locked') {
          // Windows keeps an open file: the plugin is deregistered but its
          // directory is still there. The server's hint is the only thing that
          // says what to do about it, so it is what the toast carries.
          toast(
            t('pluginCenter.toast.removeFailed', {
              plugin: name, message: hint ?? refusalMessage(err),
            }),
            'warning',
          );
          await refreshEverything();
        } else if (err instanceof ApiError && err.status === 409) {
          toast(t('packs.toast.busy'), 'warning');
          await get().refresh();
        } else if (err instanceof ApiError && err.status === 403) {
          // Same gate as an install, and the same answer: the server only
          // takes a change like this from the machine it runs on. Wrapping
          // `Forbidden` in "Could not remove Demo plugin" would tell a LAN
          // user that something went wrong rather than where to do it.
          toast(t('packs.remoteDisabled'), 'error');
        } else {
          toast(
            t('pluginCenter.toast.removeFailed', {
              plugin: name, message: refusalMessage(err),
            }),
            'error',
          );
        }
      }
    });
  },

  setEnabled: async (pluginId, enabled) => {
    const { t } = useI18n.getState();
    if (get().busy[pluginId]) return;
    const name = pluginName(pluginId);

    await withBusy(pluginId, async () => {
      try {
        await setPluginEnabled(pluginId, enabled);
        // Enabling registers nodes and activates a frontend; disabling takes
        // both away. Either way the same three things are now stale.
        await refreshEverything();
        toast(
          t(enabled ? 'pluginCenter.toast.enabled' : 'pluginCenter.toast.disabled',
            { plugin: name }),
          'success',
        );
      } catch (err) {
        toast(
          t('pluginCenter.toast.toggleFailed', {
            plugin: name, message: refusalMessage(err),
          }),
          'error',
        );
      }
    });
  },

  cancel: async () => {
    const job = get().job;
    if (job === null || !isRunning(job) || get().cancelling) return;
    set({ cancelling: true });
    try {
      // Cooperative: the flow notices between steps and inside a download, so
      // the job may still say running when this returns. The FOLLOWER is what
      // records the outcome — faking it here would show "cancelled" over an
      // unpack that is still writing files.
      await cancelPluginJob(job.jobId);
    } catch (err) {
      // A job the server has already forgotten is the common refusal here,
      // and `unknown_job` on a toast is not a sentence.
      const { t } = useI18n.getState();
      toast(t('packs.toast.cancelFailed', { message: refusalMessage(err) }), 'error');
    } finally {
      set({ cancelling: false });
    }
  },

  followJob: (jobId, pluginId, kind = 'install', cursor = 0) => {
    const current = get().job;
    if (!current || current.jobId !== jobId) {
      set({ job: emptyPluginJob(jobId, pluginId, kind) });
    }
    startFollowing(jobId, pluginId, kind, cursor);
  },

  stopFollowing: () => stopFollowing(),

  dismissJob: () => {
    const job = get().job;
    if (job === null || isRunning(job)) return;
    set({ job: null });
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
      inProgressChecked = false;
      return;
    }
    if (get().unsupported) return;

    // An install the user started before reloading is still going, and the
    // Plugin Center is closed. This toast is the only thing that says so.
    if (follower.followingJobId() !== null) {
      toast(
        useI18n.getState().t('pluginCenter.toast.inProgress'),
        'info',
        openCenterAction(get().job?.pluginId ?? ''),
      );
    }
  },
}));

/** Test-only: reset the module-scope schedulers and the store between cases. */
export function _resetPluginStoreForTesting(): void {
  follower.stop();
  settledJobId = null;
  followingPluginId = null;
  followingKind = 'install';
  inProgressChecked = false;
  usePluginStore.setState({
    plugins: [],
    byId: {},
    loading: false,
    loaded: false,
    unsupported: false,
    error: null,
    remoteInstallAllowed: true,
    generation: 0,
    job: null,
    busy: {},
    cancelling: false,
    inspection: { phase: 'idle' },
  });
}
