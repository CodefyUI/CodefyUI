import { create } from 'zustand';
import {
  ACTIVE_RUN_STATUSES,
  cancelRun,
  deleteRun,
  downloadRunMetricsCsv,
  getRun,
  getRunArtifacts,
  getRunEvents,
  getRunMetrics,
  listRuns,
  type RunArtifact,
  type RunEventsPage,
  type RunStatus,
  type RunSummary,
} from '../api/rest';
import { useToastStore } from './toastStore';
import { useI18n } from '../i18n';

/**
 * App-level state for the Runs panel (#124).
 *
 * Deliberately NOT part of `tabStore`: a run outlives the tab that started
 * it and can be inspected from a tab that never saw it, so keying any of
 * this by tab id would be a lie the first time someone reloads the page.
 * (`tabStore` also being 1400 lines is a secondary reason, not the reason.)
 *
 * ── Data flow ──────────────────────────────────────────────────────────
 * Two independent readers, because the two questions have different costs:
 *
 * 1. **The list polls.** `GET /api/runs` is one bounded query; a poll that
 *    is 2 s stale is invisible in a table of runs measured in minutes. The
 *    interval backs off to `IDLE_POLL_MS` the moment nothing is active, so
 *    an idle browser tab left open overnight costs one request per 15 s
 *    rather than one per 2 s. Polling only runs while a panel is mounted
 *    AND watching (`watch()` is reference-counted — `ResultsPanel` is
 *    instantiated once per open canvas tab, and one poller is enough).
 *
 * 2. **The open detail long-polls.** `GET /api/runs/{id}/events?wait=` parks
 *    server-side and answers the instant an event lands, so a live loss
 *    curve updates at the speed the run produces points instead of at the
 *    speed of a timer — with ONE connection, not a metric refetch per tick.
 *    The `metric` event (#122) carries whole flush batches, which is what
 *    makes charting from the event stream cheap enough to do at all.
 *
 * The socket from #121 is not used here on purpose: it is per-tab and holds
 * exactly one attachment, so borrowing it to preview a run would silently
 * detach whatever that tab was watching. Re-attach (`attach` on the tab's
 * own socket) stays an explicit, user-initiated action.
 */

/** Page size for the list. Deeper than a human scrolls, cheap to fetch. */
export const RUN_LIST_LIMIT = 50;

/** List poll while at least one run is queued or running. */
export const ACTIVE_POLL_MS = 2000;

/** List poll when everything is finished — nothing changes on its own. */
export const IDLE_POLL_MS = 15000;

/** Long-poll parking time for the open detail, in SECONDS (server caps at 60). */
export const EVENT_WAIT_S = 25;

/**
 * Floor between follower iterations that made no progress.
 *
 * The long poll is supposed to park server-side, so an unadvanced page
 * normally means its deadline passed and 500 ms of extra latency is noise.
 * It is also the thing standing between this loop and a busy-wait if a
 * server ever answers instantly without moving the cursor.
 */
export const FOLLOW_IDLE_MS = 500;

/**
 * How far back from the run's head the log tail starts.
 *
 * Cursors are gapless and 1-based, so `last_cursor - N` is exactly "the last
 * N events" without a dedicated endpoint. A finished 40-epoch run has tens
 * of thousands of events; replaying all of them to show a tail would move
 * megabytes to render a few hundred lines.
 */
export const LOG_TAIL_EVENTS = 200;

/** Ring-buffer bound on the rendered log. */
export const MAX_LOG_LINES = 500;

/** Points per series handed to the chart; more than this is downsampled. */
export const MAX_CHART_POINTS = 600;

const ACTIVE = new Set<RunStatus>(ACTIVE_RUN_STATUSES);

export function isActiveRun(status: RunStatus): boolean {
  return ACTIVE.has(status);
}

export type RunStatusFilter = 'all' | RunStatus;

/**
 * One line of the detail view's log tail.
 *
 * Structured rather than pre-rendered text so the strings stay reactive:
 * switching language must re-translate the lines already on screen, which a
 * `t()` call baked in at arrival time cannot do.
 */
export interface RunLogLine {
  cursor: number;
  ts: string;
  kind: 'started' | 'node' | 'completed' | 'failed' | 'stopped' | 'warning';
  nodeId?: string;
  status?: string;
  detail?: string;
  tone: 'info' | 'success' | 'error' | 'warning';
}

/**
 * Joins a metric name to its producing node id to form a series key.
 *
 * Two nodes in one graph may both log `loss`. Keying by name alone would
 * interleave their points into one zig-zagging line that is not any node's
 * loss, so the key carries both and `splitSeriesKey` takes them apart. The
 * panel labels a series with its plain name until there is an actual
 * collision to disambiguate.
 *
 * NUL as the separator because a metric name is an arbitrary string chosen
 * by whoever wrote the node: a space, a colon or a slash could each appear
 * in one and would split the key in the wrong place.
 */
export const SERIES_KEY_SEP = '\u0000';

export function seriesKey(name: string, nodeId: string | null): string {
  return nodeId ? `${name}${SERIES_KEY_SEP}${nodeId}` : name;
}

export function splitSeriesKey(key: string): { name: string; nodeId: string | null } {
  const at = key.indexOf(SERIES_KEY_SEP);
  return at === -1
    ? { name: key, nodeId: null }
    : { name: key.slice(0, at), nodeId: key.slice(at + 1) };
}

export interface RunDetail {
  runId: string;
  status: RunStatus;
  /**
   * The run's own row, as `GET /api/runs/{id}` returned it.
   *
   * Held here rather than looked up in `runs` because the detail must not
   * go blank when its run leaves the filtered list — which happens the
   * moment a user watching a running job switches the filter, or the job
   * finishes while `running` is selected.
   */
  row: RunSummary | null;
  /**
   * `seriesKey -> (step -> value)`.
   *
   * Keyed by step rather than an array because the seed from `/metrics` and
   * the tail from `/events` overlap: the two are read at slightly different
   * instants, so a point can legitimately arrive twice. Overwriting by step
   * makes the merge idempotent instead of drawing the same epoch twice.
   */
  series: Record<string, Record<number, number>>;
  log: RunLogLine[];
  artifacts: RunArtifact[];
  /** Highest event cursor applied — where the follower resumes. */
  cursor: number;
  loading: boolean;
  error: string | null;
}

interface RunState {
  runs: RunSummary[];
  /** Unpaged count for the current filter, so the table can say "50 of 214". */
  total: number;
  /**
   * How many runs are queued or running RIGHT NOW, app-wide.
   *
   * Kept apart from `runs` because it has to be true before the panel has
   * ever been opened (the tab badge is the only hint a detached run gives an
   * unaware user) and has to stay true while a status filter is hiding the
   * active rows. Seeded by the mount-time check, then maintained by every
   * list poll.
   */
  activeCount: number;
  loading: boolean;
  error: string | null;
  filter: RunStatusFilter;
  selectedRunId: string | null;
  detail: RunDetail | null;
  /** Run ids with a row action in flight — disables that row's buttons. */
  busy: Record<string, boolean>;

  setFilter: (filter: RunStatusFilter) => void;
  refresh: () => Promise<void>;
  select: (runId: string | null) => Promise<void>;
  cancel: (runId: string) => Promise<void>;
  remove: (runId: string) => Promise<void>;
  exportCsv: (runId: string) => Promise<void>;
  /** Start list polling; call the returned function to release this watcher. */
  watch: () => () => void;
  /** Mount-time check: how many runs are still going. */
  checkInProgress: () => Promise<number>;
}

function emptyDetail(runId: string, status: RunStatus,
                     row: RunSummary | null = null): RunDetail {
  return {
    runId,
    status,
    row,
    series: {},
    log: [],
    artifacts: [],
    cursor: 0,
    loading: true,
    error: null,
  };
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function toast(message: string, type: 'info' | 'error' | 'success' | 'warning') {
  useToastStore.getState().addToast(message, type);
}

/**
 * Fold one page of events into a detail.
 *
 * Pure and exported so the interesting part — what each event type does to
 * the chart, the log and the artifact list — is testable without a server,
 * a timer or a React tree.
 *
 * `metric` and `artifact` events never become log lines: they have their own
 * places in the UI, and a per-flush metric event would drown the tail.
 * `node_status` frames with `status: 'progress'` are dropped for the same
 * reason — a 40-epoch run emits one per epoch per node, and every one of
 * them is already represented as a point on the curve.
 */
export function reduceRunEvents(detail: RunDetail, page: RunEventsPage): RunDetail {
  let series = detail.series;
  let artifacts = detail.artifacts;
  const lines: RunLogLine[] = [];
  let touchedSeries = false;

  const line = (
    cursor: number,
    ts: string,
    kind: RunLogLine['kind'],
    tone: RunLogLine['tone'],
    extra: Partial<RunLogLine> = {},
  ) => lines.push({ cursor, ts, kind, tone, ...extra });

  for (const event of page.events) {
    // Idempotent by cursor. `/events` returns events strictly AFTER the
    // cursor we sent, so this only ever fires on a replay — but a replayed
    // log line would be a duplicate React key as well as a duplicate line,
    // and the reducer is cheaper to make safe than every caller is.
    if (event.cursor <= detail.cursor) continue;
    const payload = event.payload ?? {};
    switch (event.type) {
      case 'metric': {
        const points = Array.isArray(payload.points) ? payload.points : [];
        for (const point of points) {
          // A null value is a non-finite reading (a diverged loss). It is a
          // gap in the curve, not a zero, so it is simply not plotted.
          if (typeof point?.value !== 'number' || !Number.isFinite(point.value)) continue;
          if (typeof point.name !== 'string' || typeof point.step !== 'number') continue;
          if (!touchedSeries) {
            series = { ...series };
            touchedSeries = true;
          }
          const key = seriesKey(
            point.name, typeof point.node_id === 'string' ? point.node_id : null);
          series[key] = { ...(series[key] ?? {}), [point.step]: point.value };
        }
        break;
      }
      case 'artifact': {
        if (typeof payload.path !== 'string') break;
        // A real row id is a positive integer from sqlite. The cursor
        // fallback is PREFIXED so a synthesised key can never collide with
        // one — otherwise cursor 7 and artifact row 7 would deduplicate
        // each other and one of the two would never be listed.
        const id = typeof payload.artifact_id === 'number'
          ? payload.artifact_id
          : `cursor:${event.cursor}`;
        // The seed from /artifacts may already hold this row — the event log
        // and the table describe the same thing.
        if (artifacts.some((a) => a.id === id)) break;
        artifacts = [...artifacts, {
          id,
          kind: typeof payload.kind === 'string' ? payload.kind : 'artifact',
          path: payload.path,
          meta: (payload.meta ?? null) as Record<string, unknown> | null,
          created_at: event.ts,
        }];
        break;
      }
      case 'execution_start':
        line(event.cursor, event.ts, 'started', 'info');
        break;
      case 'node_status': {
        if (payload.status === 'progress') break;
        line(event.cursor, event.ts, 'node',
          payload.status === 'error' ? 'error'
            : payload.status === 'completed' ? 'success' : 'info',
          {
            nodeId: typeof payload.node_id === 'string' ? payload.node_id : undefined,
            status: String(payload.status ?? ''),
            detail: typeof payload.error === 'string' ? payload.error : undefined,
          });
        break;
      }
      case 'execution_complete':
        line(event.cursor, event.ts, 'completed', 'success');
        break;
      case 'execution_error':
        line(event.cursor, event.ts, 'failed', 'error', {
          detail: typeof payload.error === 'string' ? payload.error : undefined,
        });
        break;
      case 'execution_stopped':
        line(event.cursor, event.ts, 'stopped', 'warning', {
          detail: typeof payload.reason === 'string' ? payload.reason : undefined,
        });
        break;
      case 'run_warning':
        line(event.cursor, event.ts, 'warning', 'warning', {
          detail: typeof payload.detail === 'string' ? payload.detail : undefined,
        });
        break;
      default:
        // An unknown type from a newer backend is skipped, never rendered as
        // a mystery line — the log is a summary, not a protocol dump.
        break;
    }
  }

  const log = lines.length > 0
    ? [...detail.log, ...lines].slice(-MAX_LOG_LINES)
    : detail.log;

  return {
    ...detail,
    status: page.status,
    series,
    artifacts,
    log,
    // Never moves backwards: an empty page returns the cursor we sent.
    cursor: Math.max(detail.cursor, page.cursor),
  };
}

/**
 * Shape a stored `step -> value` map into chart series, downsampling when a
 * series is longer than the chart can usefully draw.
 *
 * Per-batch metrics turn a ten-minute run into six figures of points; an SVG
 * polyline with 100 000 vertices is a frozen tab, and at ~600 px wide the
 * extra vertices are not visible anyway. The LAST point is always kept so
 * the "current value" dot never lags behind the run.
 */
export function toChartSeries(
  series: Record<string, Record<number, number>>,
  keys: string[],
  maxPoints = MAX_CHART_POINTS,
): { name: string; colorKey: string; points: { x: number; y: number }[] }[] {
  // A node id is only worth showing when it is what tells two lines apart.
  const perName = new Map<string, number>();
  for (const key of keys) {
    const { name } = splitSeriesKey(key);
    perName.set(name, (perName.get(name) ?? 0) + 1);
  }
  return keys.map((key) => {
    const { name, nodeId } = splitSeriesKey(key);
    const label = (perName.get(name) ?? 0) > 1 && nodeId
      ? `${name} @${nodeId.slice(0, 8)}`
      : name;
    const steps = Object.keys(series[key] ?? {})
      .map(Number)
      .sort((a, b) => a - b);
    const stride = Math.max(1, Math.ceil(steps.length / maxPoints));
    const picked = stride === 1 ? steps : steps.filter((_, i) => i % stride === 0);
    if (stride > 1 && steps.length > 0 && picked[picked.length - 1] !== steps[steps.length - 1]) {
      picked.push(steps[steps.length - 1]);
    }
    return {
      name: label,
      // Colour follows the METRIC, not the disambiguated label, so a second
      // node joining late does not recolour the first node's line.
      colorKey: name,
      points: picked.map((x) => ({ x, y: series[key][x] })),
    };
  });
}

/** Series keys in a stable, human order: the chart legend must not shuffle. */
export function seriesNames(series: Record<string, Record<number, number>>): string[] {
  return Object.keys(series).sort();
}

// ── module-scope schedulers ──────────────────────────────────────────────
// Timers and in-flight requests are process state, not store state: putting
// them in the store would make every tick a re-render for subscribers that
// only care about the data.

let watchers = 0;
let listTimer: ReturnType<typeof setTimeout> | null = null;

/** Bumped by every `select()`; a follower whose generation is stale exits. */
let followGeneration = 0;
let followAbort: AbortController | null = null;

/** Once per page load — StrictMode mounts effects twice in development. */
let inProgressChecked = false;

function scheduleList() {
  if (listTimer !== null) clearTimeout(listTimer);
  const anyActive = useRunStore.getState().runs.some((run) => ACTIVE.has(run.status));
  listTimer = setTimeout(tickList, anyActive ? ACTIVE_POLL_MS : IDLE_POLL_MS);
}

async function tickList() {
  listTimer = null;
  if (watchers === 0) return;
  await useRunStore.getState().refresh();
  if (watchers > 0) scheduleList();
}

/**
 * Abandon the current event follower, if any.
 *
 * Two mechanisms, and both are needed. `abort()` releases the parked HTTP
 * request immediately — a 25 s long poll left dangling holds a connection
 * open on a server with a small pool. The generation bump is what stops the
 * LOOP: an abort only rejects the request in flight, and without the
 * generation check the follower would simply issue the next one.
 */
function stopFollowing() {
  followGeneration += 1;
  followAbort?.abort();
  followAbort = null;
}

/** Follow *runId* from *cursor* under a fresh generation. */
function startFollowing(runId: string, cursor: number) {
  stopFollowing();
  void follow(runId, cursor, followGeneration);
}

/**
 * Tail one run's events until it can produce no more.
 *
 * The stop condition is "terminal AND the cursor did not move", which is the
 * only one that is actually true: a terminal run with a backlog still has
 * pages to hand over, and an active run that returned nothing is simply
 * between events. Phrasing it in terms of the CURSOR rather than
 * `events.length` also makes a server that answers without making progress
 * a bounded loop instead of a busy-wait.
 */
async function follow(runId: string, startCursor: number, generation: number) {
  let cursor = startCursor;
  while (generation === followGeneration) {
    const controller = new AbortController();
    followAbort = controller;
    let page: RunEventsPage;
    try {
      page = await getRunEvents(runId, {
        cursor,
        wait: EVENT_WAIT_S,
        signal: controller.signal,
      });
    } catch {
      // Aborted by a new selection, or the server went away. Either way this
      // follower is done; the list poll is what notices a server coming back.
      return;
    }
    if (generation !== followGeneration) return;
    useRunStore.setState((state) => {
      if (!state.detail || state.detail.runId !== runId) return {};
      return { detail: reduceRunEvents(state.detail, page) };
    });
    const advanced = page.cursor > cursor;
    cursor = Math.max(cursor, page.cursor);
    if (!ACTIVE.has(page.status) && !advanced) return;
    if (!advanced) {
      await new Promise((resolve) => setTimeout(resolve, FOLLOW_IDLE_MS));
    }
  }
}

export const useRunStore = create<RunState>((set, get) => ({
  runs: [],
  total: 0,
  activeCount: 0,
  loading: false,
  error: null,
  filter: 'all',
  selectedRunId: null,
  detail: null,
  busy: {},

  setFilter: (filter) => {
    if (get().filter === filter) return;
    set({ filter, loading: true });
    void get().refresh();
  },

  refresh: async () => {
    const { filter } = get();
    try {
      const page = await listRuns({
        status: filter === 'all' ? undefined : [filter],
        limit: RUN_LIST_LIMIT,
      });
      // An unfiltered page already contains every active run — the list is
      // newest-first and an active run is by definition recent — so the
      // badge comes free. Under a filter it does not, and a stale badge on
      // the ONE affordance that reports detached runs is worse than one
      // extra bounded request while the user is deliberately filtering.
      const active = filter === 'all'
        ? page.runs.filter((run) => ACTIVE.has(run.status)).length
        : (await listRuns({ status: ACTIVE_RUN_STATUSES, limit: 1 })).total;
      // The filter can change while a request is in flight; a late page for
      // the previous filter must not repaint the table.
      if (get().filter !== filter) return;
      set({
        runs: page.runs,
        total: page.total,
        activeCount: active,
        loading: false,
        error: null,
      });
    } catch (err) {
      set({ loading: false, error: errorMessage(err) });
    }
  },

  select: async (runId) => {
    stopFollowing();
    const generation = followGeneration;
    if (runId === null) {
      set({ selectedRunId: null, detail: null });
      return;
    }
    const known = get().runs.find((run) => run.id === runId) ?? null;
    set({
      selectedRunId: runId,
      detail: emptyDetail(runId, known?.status ?? 'running', known),
    });

    let run;
    try {
      run = await getRun(runId);
    } catch (err) {
      if (generation !== followGeneration) return;
      set((state) => (state.detail?.runId === runId
        ? { detail: { ...state.detail, loading: false, error: errorMessage(err) } }
        : {}));
      return;
    }
    if (generation !== followGeneration) return;
    if (!run) {
      // Deleted or pruned between the list poll and the click.
      set((state) => (state.detail?.runId === runId
        ? {
          detail: {
            ...state.detail,
            loading: false,
            error: useI18n.getState().t('runs.error.gone'),
          },
        }
        : {}));
      void get().refresh();
      return;
    }

    // Seed the chart and the artifact list from the tables rather than from
    // the event log: both are one indexed read and both are COMPLETE, which
    // a tail of the last few hundred events is not.
    const [metrics, artifacts] = await Promise.all([
      getRunMetrics(runId).catch(() => null),
      getRunArtifacts(runId).catch(() => null),
    ]);
    if (generation !== followGeneration) return;

    const series: Record<string, Record<number, number>> = {};
    for (const point of metrics?.metrics ?? []) {
      if (point.value === null || !Number.isFinite(point.value)) continue;
      const key = seriesKey(point.name, point.node_id);
      series[key] = { ...(series[key] ?? {}), [point.step]: point.value };
    }

    const startCursor = Math.max(0, run.last_cursor - LOG_TAIL_EVENTS);
    set((state) => (state.detail?.runId === runId
      ? {
        detail: {
          ...state.detail,
          status: run.status,
          row: run,
          series,
          artifacts: artifacts?.artifacts ?? [],
          cursor: startCursor,
          loading: false,
        },
      }
      : {}));

    // Only stream while something is on screen to stream into; `watch()`
    // resumes from `detail.cursor` when a panel comes back.
    if (watchers > 0) startFollowing(runId, startCursor);
  },

  cancel: async (runId) => {
    if (get().busy[runId]) return;
    set((state) => ({ busy: { ...state.busy, [runId]: true } }));
    const { t } = useI18n.getState();
    try {
      const outcome = await cancelRun(runId);
      toast(
        outcome.cancelled ? t('runs.toast.cancelling') : t('runs.toast.alreadyDone'),
        outcome.cancelled ? 'info' : 'warning',
      );
    } catch (err) {
      toast(`${t('runs.toast.cancelFailed')}: ${errorMessage(err)}`, 'error');
    } finally {
      set((state) => {
        const busy = { ...state.busy };
        delete busy[runId];
        return { busy };
      });
    }
    await get().refresh();
  },

  remove: async (runId) => {
    if (get().busy[runId]) return;
    set((state) => ({ busy: { ...state.busy, [runId]: true } }));
    const { t } = useI18n.getState();
    try {
      await deleteRun(runId);
      if (get().selectedRunId === runId) {
        stopFollowing();
        set({ selectedRunId: null, detail: null });
      }
      toast(t('runs.toast.deleted'), 'success');
    } catch (err) {
      toast(`${t('runs.toast.deleteFailed')}: ${errorMessage(err)}`, 'error');
    } finally {
      set((state) => {
        const busy = { ...state.busy };
        delete busy[runId];
        return { busy };
      });
    }
    await get().refresh();
  },

  exportCsv: async (runId) => {
    // Guarded like the other row actions: a CSV of a long run takes long
    // enough that an impatient second click is normal, and it would start a
    // second download of the same file.
    if (get().busy[runId]) return;
    set((state) => ({ busy: { ...state.busy, [runId]: true } }));
    try {
      await downloadRunMetricsCsv(runId);
    } catch (err) {
      toast(
        `${useI18n.getState().t('runs.toast.exportFailed')}: ${errorMessage(err)}`,
        'error',
      );
    } finally {
      set((state) => {
        const busy = { ...state.busy };
        delete busy[runId];
        return { busy };
      });
    }
  },

  /**
   * Register a viewer. Both readers — the list poll and the detail's event
   * follower — live between the first `watch()` and the last release.
   *
   * The event follower is tied to the SAME refcount rather than to the
   * component that opened the detail, because `selectedRunId` is app-level:
   * `ResultsPanel` is instantiated once per canvas tab, so unmounting one
   * of them must not abandon a follower the others are still rendering.
   * When the last one goes the selection is kept but the network stops, and
   * re-watching resumes from the cursor already applied — which is also
   * what makes StrictMode's mount/unmount/mount leave a live follower
   * behind instead of a silently dead detail.
   */
  watch: () => {
    watchers += 1;
    if (watchers === 1) {
      set({ loading: true });
      void tickList();
      const { selectedRunId, detail } = get();
      if (selectedRunId !== null && detail !== null) {
        startFollowing(selectedRunId, detail.cursor);
      }
    }
    let released = false;
    return () => {
      if (released) return;
      released = true;
      watchers -= 1;
      if (watchers > 0) return;
      if (listTimer !== null) {
        clearTimeout(listTimer);
        listTimer = null;
      }
      stopFollowing();
    };
  },

  /**
   * Mount-time check: how many runs are still going.
   *
   * Also seeds `activeCount`, which is what puts a number on the Runs tab
   * before anyone has opened it. Without that seed the badge appears only
   * after the first visit — precisely backwards, since the badge exists to
   * tell a user who does not yet know that a run outlived their reload.
   *
   * The toast fires once per page load; the count is stored either way.
   */
  checkInProgress: async () => {
    if (inProgressChecked) return 0;
    inProgressChecked = true;
    try {
      const page = await listRuns({ status: ACTIVE_RUN_STATUSES, limit: 1 });
      set({ activeCount: page.total });
      if (page.total > 0) {
        toast(useI18n.getState().t('runs.toast.inProgress', { count: page.total }), 'info');
      }
      return page.total;
    } catch {
      // The server being unreachable at boot is App's problem to report, not
      // a reason to interrupt the user with a second failure toast.
      return 0;
    }
  },
}));

/** Test-only: reset the module-scope schedulers between cases. */
export function _resetRunStoreForTesting(): void {
  watchers = 0;
  if (listTimer !== null) clearTimeout(listTimer);
  listTimer = null;
  stopFollowing();
  inProgressChecked = false;
  useRunStore.setState({
    runs: [],
    total: 0,
    activeCount: 0,
    loading: false,
    error: null,
    filter: 'all',
    selectedRunId: null,
    detail: null,
    busy: {},
  });
}
