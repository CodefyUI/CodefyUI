/**
 * The plugin-facing execution event stream (`api.events.onExecution`,
 * apiVersion 3).
 *
 * ── Where the events come from ───────────────────────────────────────────
 *
 * The Run Service's live fan-out is `ExecutionWebSocket.dispatch` in
 * `api/ws.ts`, and its wildcard slot — `ws.on('*', fn)` — is the one place
 * every frame of an attached run passes through. Sockets are per canvas tab
 * (`TabState.ws`), so this module attaches to all of them and keeps up with
 * tabs opening and closing, exactly as `useGraphExecution` does.
 *
 * Not the Runs panel's data: `runStore` polls REST (`GET /api/runs`) and
 * long-polls one selected run, which is the right shape for a history table
 * and the wrong shape for a live event stream. Plugins that want history call
 * `api.runs` instead — the two halves of the contract cover the two cases.
 *
 * ── Why the wire vocabulary is not the plugin vocabulary ─────────────────
 *
 * The wire speaks `execution_start` / `execution_complete` / `execution_error`
 * / `execution_stopped`; the contract publishes `run_started` and one
 * `run_finished` carrying the run's terminal `status`. That indirection is the
 * point: `ws.ts` is an internal protocol we expect to keep changing, and a
 * published contract that forwarded it raw would freeze it. Normalising here
 * also lets the host drop the frames that would otherwise be actively
 * misleading:
 *
 *   - `execution_error` with `rejected: true` — a refused submit. Nothing ran,
 *     and its `run_id` names whichever run the socket was already attached to,
 *     so forwarding it would report an unrelated, still-running run as failed.
 *   - `execution_stopped` with `reason: "not_running"` — synthesised when a
 *     cancel had nothing to cancel.
 *   - Transport frames (`attached`, `detached`, `cache_cleared`,
 *     `reconnected`, protocol `error`) carry no cursor and are not run events.
 *
 * Everything dropped here still occupies a cursor in the durable log, which is
 * why a delivered event carries `seq` as well — see `ExecutionEvent`.
 *
 * ── Delivery ─────────────────────────────────────────────────────────────
 *
 * Events buffer and flush on the host's shared frame slot
 * (`utils/frameScheduler`) — the same one the canvas node-badge queue uses,
 * deliberately not a second timing mechanism. Each subscriber callback is
 * invoked inside its own try/catch, so one plugin throwing cannot stop
 * another plugin, or the host's own consumers, from seeing the batch.
 *
 * Before any of that, every frame passes the per-run watermark in
 * `acceptFrame` — the thing that keeps the stream a tail while the host's own
 * re-attach paths replay whole logs through the same socket, and the place
 * `seq` is handed out.
 *
 * One consequence worth knowing, because it is visible to plugin authors: the
 * watermark table belongs to this MODULE, not to a subscription. A plugin
 * that subscribes after another has already been streaming a run inherits
 * that run's watermark, so a replay it has personally never seen is swallowed
 * for it too. Documented rather than fixed: per-subscriber tables would
 * multiply the memory by the plugin count to un-swallow history the plugin
 * can fetch from `api.runs` anyway.
 */
import { useTabStore } from '../store/tabStore';
import type { ExecutionWebSocket } from '../api/ws';
import type { RunMetricPoint } from '../api/rest';
import { createFrameFlusher } from '../utils/frameScheduler';

/** Terminal state of a run, mirroring the backend's run statuses. */
export type ExecutionFinishStatus =
  | 'succeeded' | 'failed' | 'cancelled' | 'interrupted';

/**
 * A run event as normalised from the wire, before the stream stamps its `seq`.
 *
 * A `metric` entry keeps the batch it was recorded as rather than being split
 * into one event per point, for two reasons that only became visible once the
 * per-point version existed:
 *
 *  - **The log entry stays atomic.** Under overflow the buffer evicts whole
 *    events. With per-point expansion that meant dropping some points of an
 *    entry while delivering others, so a plugin holding cursor N could not
 *    conclude it had entry N. One entry, one event.
 *  - **`points` is the same type `runs.metrics()` returns.** A dashboard
 *    spans the live tail and the REST back-fill; making them one type means
 *    one fold, and means a diverged loss is `value: null` on both sides
 *    rather than a dropped point on one and an explicit gap on the other.
 */
export type ExecutionEventDraft =
  | { type: 'run_started'; run_id: string; cursor: number }
  | {
      type: 'node_status'; run_id: string; cursor: number;
      node_id: string; status: string; error?: string;
    }
  | {
      type: 'metric'; run_id: string; cursor: number;
      points: readonly RunMetricPoint[];
    }
  | {
      type: 'run_finished'; run_id: string; cursor: number;
      status: ExecutionFinishStatus; error?: string;
    };

/**
 * One delivered run event: a draft plus the stream's own sequence number.
 *
 * ── Why there are two numbers ────────────────────────────────────────────
 *
 * `cursor` is the entry's position in the run's DURABLE log, which is what
 * `runs.get(id).last_cursor` and `GET /api/runs/{id}/events` speak, so it is
 * the number that correlates an event with the REST side. What it is not is
 * dense. Plenty of durable entries never become plugin events:
 *
 *   - `artifact` and `run_warning` — outside the published union.
 *   - `execution_error` with `rejected: true`, and `execution_stopped` with
 *     `reason: "not_running"` — deliberately dropped as misleading.
 *   - a `metric` entry whose every point was malformed, or which the server
 *     collapsed because the payload was too large.
 *
 * Each of those still consumes a cursor. A run that saves a checkpoint emits
 * an `artifact` every time, so cursor jumps are not an edge case — they are
 * what an ordinary training run looks like. Telling plugins to read a jump as
 * data loss (which an earlier draft of this contract did) would have made a
 * dashboard cry wolf on every checkpoint.
 *
 * `seq` is this module's own counter, incremented once per event actually
 * handed to subscribers, per run. It is dense by construction: the only thing
 * that can put a hole in it is the buffer dropping events under
 * `MAX_BUFFERED_EXECUTION_WEIGHT`. So `seq` answers "did I miss anything?"
 * and `cursor` answers "where is this in the log?", and neither has to lie to
 * do the other's job.
 */
export type ExecutionEvent = ExecutionEventDraft & { readonly seq: number };

export type ExecutionEventHandler = (event: ExecutionEvent) => void;

/**
 * How much may wait for a frame that is not coming, measured in RETAINED
 * OBJECTS: one unit for an event, plus one for each metric point it carries.
 *
 * An occluded or backgrounded document keeps `requestAnimationFrame` but
 * never calls it back, and a dashboard watching a long training run in a
 * backgrounded tab is not a corner case — it is this API working as intended
 * — so the buffer has to be bounded by something that reflects what it
 * actually holds.
 *
 * Counting events does not. A `metric` event carries a whole recorded batch,
 * and the backend flushes metrics on a 0.5s floor, so a run writing several
 * series at a high step rate puts hundreds or low thousands of points in one
 * event. An event count of 2000 therefore bounded anywhere from 2,000 to
 * ~2,000,000 retained objects depending on the run — three orders of
 * magnitude of slack, in the one scenario the API exists for. Weighing an
 * event by `1 + points.length` bounds the thing that costs memory.
 *
 * At the cap the oldest non-lifecycle events are dropped first: a plugin that
 * misses a metric can re-read it from `api.runs.metrics()`, but one that
 * misses `run_finished` waits forever. Every drop punches a hole in `seq`,
 * which is how a subscriber finds out.
 *
 * The value is roomy on purpose: ~20k objects is a few megabytes, and the
 * backend flushes at most `METRIC_FLUSH_MAX_POINTS` (64) points per metric,
 * so this holds around 300 batches — a couple of minutes of a hidden tab —
 * before anything is sacrificed.
 */
export const MAX_BUFFERED_EXECUTION_WEIGHT = 20000;

/** What one buffered event costs against `MAX_BUFFERED_EXECUTION_WEIGHT`. */
export function executionEventWeight(event: ExecutionEvent): number {
  return event.type === 'metric' ? 1 + event.points.length : 1;
}

/**
 * How many runs keep a delivery watermark.
 *
 * Two small numbers per run, not one key per frame: see `acceptFrame`. The
 * bound exists so a session cannot grow the table without limit, and the
 * least-recently-advanced run is evicted first.
 *
 * It is set far above what a session reaches because exceeding it is not a
 * degraded mode, it is a broken promise: an evicted run has no watermark, so
 * re-attaching to it replays its log a second time and the published "a
 * cursor you have already seen will never arrive again" stops holding. 64 was
 * reachable by a user clicking through the Runs panel; a thousand distinct
 * runs attached in one page session is not. The exact condition is documented
 * for plugin authors rather than left implicit.
 */
export const EXECUTION_EVENT_WATERMARK_RUNS = 1024;

type Frame = Record<string, unknown>;

function str(value: unknown): string | undefined {
  return typeof value === 'string' && value ? value : undefined;
}

/**
 * Translate one wire frame into zero or more contract events.
 *
 * Pure and exported so the mapping — including everything it deliberately
 * drops — is testable without a socket.
 */
export function normalizeExecutionFrame(raw: unknown): ExecutionEventDraft[] {
  if (!raw || typeof raw !== 'object') return [];
  const frame = raw as Frame;
  const type = str(frame.type);
  const run_id = str(frame.run_id);
  const cursor = frame.cursor;
  // No cursor means a transport frame, not a run event.
  if (!type || !run_id || typeof cursor !== 'number') return [];

  switch (type) {
    case 'execution_start':
      return [{ type: 'run_started', run_id, cursor }];

    case 'node_status': {
      const node_id = str(frame.node_id);
      const status = str(frame.status);
      if (!node_id || !status) return [];
      const error = str(frame.error);
      return [{
        type: 'node_status', run_id, cursor, node_id, status,
        ...(error ? { error } : {}),
      }];
    }

    case 'metric': {
      const raw_points = frame.points;
      if (!Array.isArray(raw_points)) return [];
      const points: RunMetricPoint[] = [];
      for (const raw_point of raw_points) {
        if (!raw_point || typeof raw_point !== 'object') continue;
        const point = raw_point as Frame;
        const name = str(point.name);
        if (!name || typeof point.step !== 'number') continue;
        // `null` is NOT malformed. The server runs `json_safe()` over every
        // payload, which turns a non-finite float into null — so a diverged
        // loss reaches us as `value: null`, and that is exactly the moment a
        // dashboard needs to draw a break rather than see nothing at all.
        // Anything that is neither a number nor null really is malformed.
        const value = point.value;
        if (typeof value !== 'number' && value !== null) continue;
        points.push({
          node_id: str(point.node_id) ?? null,
          name, step: point.step, value,
        });
      }
      // A frame whose every point was malformed carries nothing; delivering an
      // empty batch would just be noise.
      if (points.length === 0) return [];
      return [{ type: 'metric', run_id, cursor, points }];
    }

    case 'execution_complete':
      return [{ type: 'run_finished', run_id, cursor, status: 'succeeded' }];

    case 'execution_error': {
      // A refused submit — nothing started, and `run_id` is somebody else's.
      if (frame.rejected === true) return [];
      const error = str(frame.error);
      return [{
        type: 'run_finished', run_id, cursor, status: 'failed',
        ...(error ? { error } : {}),
      }];
    }

    case 'execution_stopped': {
      const reason = str(frame.reason);
      if (reason !== 'cancelled' && reason !== 'interrupted') return [];
      return [{ type: 'run_finished', run_id, cursor, status: reason }];
    }

    default:
      // `artifact`, `run_warning` and anything a future server adds. Dropping
      // them keeps the published union closed and narrowable; they are
      // reachable through `api.runs` and the REST events endpoint.
      return [];
  }
}

// ── module state ─────────────────────────────────────────────────────────

const subscribers = new Set<ExecutionEventHandler>();
/** tabId -> the wildcard handler registered on that tab's socket. */
const attached = new Map<string, { ws: ExecutionWebSocket; handler: (d: unknown) => void }>();
let unsubscribeTabs: (() => void) | null = null;

let buffer: ExecutionEvent[] = [];
/**
 * Where the next eviction should start looking.
 *
 * It only ever moves forward through the lifecycle events it has already
 * rejected, which is what keeps a sustained overflow linear: without it,
 * every single event past the cap would rescan the whole buffer, and an
 * occluded window during a long run is exactly the case where that happens
 * millions of times.
 */
let evictFrom = 0;

/**
 * Running total of `executionEventWeight` over everything in `buffer`.
 *
 * Kept incrementally rather than recomputed: the buffer is walked on every
 * accepted frame, and summing point counts across it each time would make the
 * hidden-tab path quadratic in exactly the scenario the bound exists for.
 */
let bufferedWeight = 0;

/** What the stream remembers about one run. */
interface RunWatermark {
  /** Highest durable cursor accepted for this run. */
  cursor: number;
  /** How many events have been handed to subscribers for this run. */
  seq: number;
}

/** run_id -> watermark. Insertion order is least-recently-advanced first. */
const watermarks = new Map<string, RunWatermark>();
let droppedSinceWarning = 0;

const flusher = createFrameFlusher(flushExecutionEvents);

/**
 * The gate that makes this a tail rather than a transcript, and the source of
 * the dense `seq`.
 *
 * Two things push the same frame at us more than once:
 *
 *  - Two canvas tabs attached to the same run — every frame arrives twice.
 *  - **Re-attach.** Both of the editor's attach paths deliberately ask for
 *    `cursor: 0` (`useGraphExecution` on page load, `RunsPanel` when the user
 *    picks a run to watch) because the host's own console is empty and wants
 *    the run's whole story. The server replays the entire durable log, and
 *    every replayed frame comes through the same `'*'` slot this module taps.
 *    A 1000-event run therefore used to be delivered twice.
 *
 * Durable cursors are monotonic per run, so one number per run answers both:
 * accept a frame only if its cursor is strictly above everything already
 * accepted for that run. A replay of ground already covered is swallowed.
 *
 * `seq` is then handed out here rather than at delivery, and only to frames
 * that actually became an event — which is what makes it dense across the
 * entry types the stream drops. Assigning it here rather than at flush time
 * is also deliberate: an event evicted under buffer pressure has already
 * taken its number, so the hole it leaves is visible to subscribers.
 *
 * Returns the sequence number, or null when the frame is a replay.
 */
function acceptFrame(run_id: string, cursor: number): number | null {
  const existing = watermarks.get(run_id);
  if (existing !== undefined) {
    if (cursor <= existing.cursor) return null;
    existing.cursor = cursor;
    existing.seq += 1;
    // Re-insert so iteration order stays least-recently-advanced first.
    watermarks.delete(run_id);
    watermarks.set(run_id, existing);
    return existing.seq;
  }
  watermarks.set(run_id, { cursor, seq: 1 });
  if (watermarks.size > EXECUTION_EVENT_WATERMARK_RUNS) {
    const oldest = watermarks.keys().next();
    if (!oldest.done) watermarks.delete(oldest.value);
  }
  return 1;
}

/**
 * Drop the oldest sacrificeable event; report whether there was one.
 *
 * Returns false when nothing but lifecycle events are left, and the caller
 * must take that for an answer. The whole point of the priority is that a
 * plugin which misses a metric can re-read it from `api.runs.metrics()`,
 * while one that misses `run_finished` waits forever — so room is never
 * bought with a lifecycle event, however heavy the arriving one is.
 */
function evictOne(): boolean {
  while (
    evictFrom < buffer.length
    && buffer[evictFrom].type !== 'metric'
    && buffer[evictFrom].type !== 'node_status'
  ) {
    evictFrom += 1;
  }
  if (evictFrom >= buffer.length) return false;
  const [dropped] = buffer.splice(evictFrom, 1);
  bufferedWeight -= executionEventWeight(dropped);
  droppedSinceWarning += 1;
  return true;
}

/**
 * Freeze what every subscriber shares.
 *
 * One event object goes to all of them, so without this the first subscriber
 * could empty `points` — `readonly` is a compile-time claim a plugin bundle
 * never sees — and the second would receive an event the host never sent.
 */
function sealed(draft: ExecutionEventDraft, seq: number): ExecutionEvent {
  if (draft.type === 'metric') {
    for (const point of draft.points) Object.freeze(point);
    Object.freeze(draft.points);
  }
  return Object.freeze({ ...draft, seq }) as ExecutionEvent;
}

function onFrame(raw: unknown): void {
  if (subscribers.size === 0) return;
  const drafts = normalizeExecutionFrame(raw);
  // A frame that produced nothing takes no sequence number and does not move
  // the watermark: it never happened as far as this stream is concerned.
  if (drafts.length === 0) return;
  const [first] = drafts;
  const seq = acceptFrame(first.run_id, first.cursor);
  if (seq === null) return;
  for (const draft of drafts) {
    const event = sealed(draft, seq);
    const weight = executionEventWeight(event);
    let fits = bufferedWeight + weight <= MAX_BUFFERED_EXECUTION_WEIGHT;
    while (!fits && evictOne()) {
      fits = bufferedWeight + weight <= MAX_BUFFERED_EXECUTION_WEIGHT;
    }
    // It goes in even when it does not fit. Two ways that happens, and
    // accepting the overshoot is right for both: the buffer is empty and the
    // event alone outweighs the budget (one event is the smallest thing this
    // can hold, so the alternative is dropping it outright), or everything
    // left is lifecycle (which `evictOne` refuses to trade, by design). The
    // overshoot is bounded by one event plus the lifecycle events in flight,
    // which weigh one apiece — never a metric batch, and never unbounded.
    buffer.push(event);
    bufferedWeight += weight;
  }
  flusher.schedule();
}

/**
 * Deliver everything buffered right now.
 *
 * Exported so a test can advance the stream without a real animation frame;
 * the scheduler calls it once per frame in the browser.
 */
export function flushExecutionEvents(): void {
  flusher.cancel();
  if (buffer.length === 0) return;
  const batch = buffer;
  buffer = [];
  evictFrom = 0;
  bufferedWeight = 0;
  if (droppedSinceWarning > 0) {
    console.warn(
      `[plugins] dropped ${droppedSinceWarning} execution event(s): more than `
      + `${MAX_BUFFERED_EXECUTION_WEIGHT} retained objects (events plus metric `
      + `points) were waiting for an animation frame.`,
    );
    droppedSinceWarning = 0;
  }
  // Snapshot the subscribers: a callback may unsubscribe (or subscribe)
  // mid-batch, and iterating the live Set would then skip or double-deliver.
  //
  // The snapshot alone is not enough, though. "Unsubscribe on run_finished,
  // then tear my state down" is the obvious plugin idiom, and without the
  // membership re-check below the rest of the batch — a whole buffer's worth
  // under overflow — would still be delivered into code that has already
  // torn itself down. The host would contain the resulting throws, but the
  // plugin author would be reading warnings for an unsubscribe that did not.
  const handlers = Array.from(subscribers);
  for (const event of batch) {
    for (const handler of handlers) {
      if (!subscribers.has(handler)) continue;
      try {
        handler(event);
      } catch (err) {
        console.warn('[plugins] onExecution subscriber failed:', err);
      }
    }
  }
}

function attachTab(tabId: string, ws: ExecutionWebSocket | undefined): void {
  // A tab with no socket cannot happen through the store's own API, but this
  // runs inside a zustand subscriber: throwing here would surface as a
  // failure in whatever unrelated code wrote to the store.
  if (!ws || attached.has(tabId)) return;
  const handler = (data: unknown) => onFrame(data);
  ws.on('*', handler);
  attached.set(tabId, { ws, handler });
}

function detachTab(tabId: string): void {
  const entry = attached.get(tabId);
  if (!entry) return;
  entry.ws.off('*', entry.handler);
  attached.delete(tabId);
}

function startListening(): void {
  if (unsubscribeTabs !== null) return;
  for (const tab of useTabStore.getState().tabs) attachTab(tab.id, tab.ws);
  unsubscribeTabs = useTabStore.subscribe((state) => {
    const live = new Set(state.tabs.map((t) => t.id));
    for (const tabId of Array.from(attached.keys())) {
      if (!live.has(tabId)) detachTab(tabId);
    }
    for (const tab of state.tabs) attachTab(tab.id, tab.ws);
  });
}

function stopListening(): void {
  unsubscribeTabs?.();
  unsubscribeTabs = null;
  for (const tabId of Array.from(attached.keys())) detachTab(tabId);
  flusher.cancel();
  buffer = [];
  bufferedWeight = 0;
  evictFrom = 0;
  watermarks.clear();
  droppedSinceWarning = 0;
}

/**
 * Subscribe to the run event stream. Returns an unsubscribe function.
 *
 * The sockets are only tapped while at least one subscriber exists, so a
 * workspace with no plugins pays nothing, and unsubscribing the last listener
 * removes every handler this module installed.
 */
export function subscribeExecutionEvents(
  handler: ExecutionEventHandler,
): () => void {
  subscribers.add(handler);
  startListening();
  let done = false;
  return () => {
    if (done) return;
    done = true;
    subscribers.delete(handler);
    if (subscribers.size === 0) stopListening();
  };
}

/** Number of live subscribers. Exposed for leak tests. */
export function executionEventSubscriberCount(): number {
  return subscribers.size;
}

/** Number of sockets currently tapped. Exposed for leak tests. */
export function executionEventTapCount(): number {
  return attached.size;
}

/** Test helper — drop every subscriber and detach from every socket. */
export function _resetExecutionEvents(): void {
  subscribers.clear();
  stopListening();
}
