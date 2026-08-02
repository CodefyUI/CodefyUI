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
 * ── Delivery ─────────────────────────────────────────────────────────────
 *
 * Events buffer and flush on the host's shared frame slot
 * (`utils/frameScheduler`) — the same one the canvas node-badge queue uses,
 * deliberately not a second timing mechanism. Each subscriber callback is
 * invoked inside its own try/catch, so one plugin throwing cannot stop
 * another plugin, or the host's own consumers, from seeing the batch.
 *
 * Before any of that, every frame passes the per-run watermark in
 * `alreadyDelivered` — the thing that keeps the stream a tail while the
 * host's own re-attach paths replay whole logs through the same socket.
 */
import { useTabStore } from '../store/tabStore';
import type { ExecutionWebSocket } from '../api/ws';
import type { RunMetricPoint } from '../api/rest';
import { createFrameFlusher } from '../utils/frameScheduler';

/** Terminal state of a run, mirroring the backend's run statuses. */
export type ExecutionFinishStatus =
  | 'succeeded' | 'failed' | 'cancelled' | 'interrupted';

/**
 * One normalised run event — exactly one entry of the run's durable log.
 *
 * `cursor` is that entry's position, the same cursor
 * `GET /api/runs/{id}/events` pages by.
 *
 * A `metric` entry keeps the batch it was recorded as rather than being split
 * into one event per point, for two reasons that only became visible once the
 * per-point version existed:
 *
 *  - **`cursor` stays honest.** Under overflow the buffer evicts whole
 *    events. With per-point expansion that meant dropping some points of an
 *    entry while delivering others, so a plugin holding cursor N could not
 *    conclude it had entry N — which is precisely what this module tells it
 *    to conclude. One entry, one event, one cursor.
 *  - **`points` is the same type `runs.metrics()` returns.** A dashboard
 *    spans the live tail and the REST back-fill; making them one type means
 *    one fold, and means a diverged loss is `value: null` on both sides
 *    rather than a dropped point on one and an explicit gap on the other.
 */
export type ExecutionEvent =
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

export type ExecutionEventHandler = (event: ExecutionEvent) => void;

/**
 * How many events may wait for a frame that is not coming.
 *
 * An occluded or backgrounded document keeps `requestAnimationFrame` but
 * never calls it back, so a long training run in a hidden window would
 * otherwise grow this buffer without bound. At the cap the oldest
 * non-lifecycle events are dropped first: a plugin that misses a metric can
 * re-read it from `api.runs.metrics()`, but one that misses `run_finished`
 * waits forever.
 */
export const MAX_BUFFERED_EXECUTION_EVENTS = 2000;

/**
 * How many runs keep a delivery watermark.
 *
 * One number per run, not one key per frame: see `alreadyDelivered`. A
 * workspace watches single digits of runs at once, so this only bounds a
 * pathological session; the least-recently-advanced run is dropped first.
 */
export const EXECUTION_EVENT_WATERMARK_RUNS = 64;

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
export function normalizeExecutionFrame(raw: unknown): ExecutionEvent[] {
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
/** run_id -> the highest cursor accepted for it. Insertion order = LRU. */
const watermarks = new Map<string, number>();
let droppedSinceWarning = 0;

const flusher = createFrameFlusher(flushExecutionEvents);

/**
 * The gate that makes this a tail rather than a transcript.
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
 * Cursors are gapless and monotonic per run, so one number per run answers
 * both: accept a frame only if its cursor is strictly above everything
 * already accepted for that run. A replay of ground already covered is
 * silently swallowed, and the guarantee the contract sells — cursors that
 * strictly increase and never repeat, so a jump means real loss — becomes
 * true by construction rather than by luck.
 */
function alreadyDelivered(run_id: string, cursor: number): boolean {
  const watermark = watermarks.get(run_id);
  if (watermark !== undefined && cursor <= watermark) return true;
  // Re-insert so iteration order stays least-recently-advanced first.
  watermarks.delete(run_id);
  watermarks.set(run_id, cursor);
  if (watermarks.size > EXECUTION_EVENT_WATERMARK_RUNS) {
    const oldest = watermarks.keys().next();
    if (!oldest.done) watermarks.delete(oldest.value);
  }
  return false;
}

/** Make room at the cap, sacrificing metrics and statuses before lifecycle. */
function evictOne(): void {
  while (
    evictFrom < buffer.length
    && buffer[evictFrom].type !== 'metric'
    && buffer[evictFrom].type !== 'node_status'
  ) {
    evictFrom += 1;
  }
  // Nothing but lifecycle events left: bounded is bounded, so the oldest goes.
  if (evictFrom >= buffer.length) evictFrom = 0;
  buffer.splice(evictFrom, 1);
  droppedSinceWarning += 1;
}

function onFrame(raw: unknown): void {
  if (subscribers.size === 0) return;
  const events = normalizeExecutionFrame(raw);
  if (events.length === 0) return;
  const [first] = events;
  if (alreadyDelivered(first.run_id, first.cursor)) return;
  for (const event of events) {
    if (buffer.length >= MAX_BUFFERED_EXECUTION_EVENTS) evictOne();
    buffer.push(event);
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
  if (droppedSinceWarning > 0) {
    console.warn(
      `[plugins] dropped ${droppedSinceWarning} execution event(s): more than `
      + `${MAX_BUFFERED_EXECUTION_EVENTS} were waiting for an animation frame.`,
    );
    droppedSinceWarning = 0;
  }
  // Snapshot the subscribers: a callback may unsubscribe (or subscribe)
  // mid-batch, and iterating the live Set would then skip or double-deliver.
  //
  // The snapshot alone is not enough, though. "Unsubscribe on run_finished,
  // then tear my state down" is the obvious plugin idiom, and without the
  // membership re-check below the rest of the batch — up to the whole 2000
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
