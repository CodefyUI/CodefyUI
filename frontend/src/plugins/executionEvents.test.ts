import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { useTabStore } from '../store/tabStore';
import type { ExecutionWebSocket } from '../api/ws';
import {
  EXECUTION_EVENT_DEDUPE_WINDOW,
  MAX_BUFFERED_EXECUTION_EVENTS,
  _resetExecutionEvents,
  executionEventSubscriberCount,
  executionEventTapCount,
  flushExecutionEvents,
  normalizeExecutionFrame,
  subscribeExecutionEvents,
  type ExecutionEvent,
} from './executionEvents';

/**
 * A stand-in for `ExecutionWebSocket` that records its handler table, so a
 * test can push frames in and assert that unsubscribing really removes them.
 */
class FakeWs {
  handlers = new Map<string, Array<(d: unknown) => void>>();

  on(type: string, handler: (d: unknown) => void): void {
    const list = this.handlers.get(type) ?? [];
    list.push(handler);
    this.handlers.set(type, list);
  }

  off(type: string, handler: (d: unknown) => void): void {
    const list = this.handlers.get(type) ?? [];
    const i = list.indexOf(handler);
    if (i !== -1) list.splice(i, 1);
  }

  /** Deliver a frame the way `dispatch` would. */
  emit(frame: unknown): void {
    for (const handler of [...(this.handlers.get('*') ?? [])]) handler(frame);
  }

  get wildcardCount(): number {
    return (this.handlers.get('*') ?? []).length;
  }
}

/**
 * Replace the tab store with `n` tabs whose sockets are fakes.
 *
 * Cloned from a real tab rather than built from `{ id, ws }`: the store's own
 * autosave subscriber walks every tab's nodes, so a half-shaped tab makes an
 * unrelated part of the app throw a few frames later.
 */
function seedTabs(n: number): FakeWs[] {
  useTabStore.setState({ tabs: [], activeTabId: null } as never);
  useTabStore.getState().addTab('seed');
  const template = useTabStore.getState().tabs[0];
  const sockets: FakeWs[] = [];
  const tabs = Array.from({ length: n }, (_, i) => {
    const ws = new FakeWs();
    sockets.push(ws);
    return { ...template, id: `t${i}`, ws: ws as unknown as ExecutionWebSocket };
  });
  useTabStore.setState({ tabs: tabs as never, activeTabId: 't0' });
  return sockets;
}

/** A fake-socket tab shaped like the ones `seedTabs` makes. */
function extraTab(id: string, ws: FakeWs) {
  return { ...useTabStore.getState().tabs[0], id, ws: ws as unknown as ExecutionWebSocket };
}

function runFrame() {
  vi.advanceTimersByTime(20);
}

beforeEach(() => {
  vi.useFakeTimers();
  _resetExecutionEvents();
});

afterEach(() => {
  _resetExecutionEvents();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

// ── the wire -> contract mapping, as a table ─────────────────────────────

describe('normalizeExecutionFrame', () => {
  it('maps execution_start to run_started', () => {
    expect(normalizeExecutionFrame({
      type: 'execution_start', run_id: 'r1', cursor: 1,
    })).toEqual([{ type: 'run_started', run_id: 'r1', cursor: 1 }]);
  });

  it('maps node_status, carrying an error only when there is one', () => {
    expect(normalizeExecutionFrame({
      type: 'node_status', run_id: 'r1', cursor: 2, node_id: 'n1', status: 'running',
    })).toEqual([{
      type: 'node_status', run_id: 'r1', cursor: 2, node_id: 'n1', status: 'running',
    }]);
    expect(normalizeExecutionFrame({
      type: 'node_status', run_id: 'r1', cursor: 3,
      node_id: 'n1', status: 'error', error: 'boom',
    })).toEqual([{
      type: 'node_status', run_id: 'r1', cursor: 3,
      node_id: 'n1', status: 'error', error: 'boom',
    }]);
  });

  it('expands a batched metric frame into one event per point', () => {
    expect(normalizeExecutionFrame({
      type: 'metric', run_id: 'r1', cursor: 4,
      points: [
        { name: 'loss', value: 0.5, step: 1, node_id: 'n1' },
        { name: 'acc', value: 0.9, step: 1, node_id: null },
      ],
    })).toEqual([
      { type: 'metric', run_id: 'r1', cursor: 4, name: 'loss', value: 0.5, step: 1, node_id: 'n1' },
      { type: 'metric', run_id: 'r1', cursor: 4, name: 'acc', value: 0.9, step: 1, node_id: null },
    ]);
  });

  it('skips malformed metric points without losing the good ones', () => {
    const events = normalizeExecutionFrame({
      type: 'metric', run_id: 'r1', cursor: 5,
      points: [
        null,
        'garbage',
        { name: '', value: 1, step: 1 },
        { name: 'loss', value: null, step: 1 },
        { name: 'loss', value: 1, step: 'one' },
        { name: 'loss', value: 0.25, step: 7 },
      ],
    });
    expect(events).toEqual([{
      type: 'metric', run_id: 'r1', cursor: 5,
      name: 'loss', value: 0.25, step: 7, node_id: null,
    }]);
  });

  it('maps the three terminal frames onto one run_finished with a status', () => {
    expect(normalizeExecutionFrame({
      type: 'execution_complete', run_id: 'r1', cursor: 9,
    })).toEqual([{ type: 'run_finished', run_id: 'r1', cursor: 9, status: 'succeeded' }]);

    expect(normalizeExecutionFrame({
      type: 'execution_error', run_id: 'r1', cursor: 9, error: 'nope',
    })).toEqual([{
      type: 'run_finished', run_id: 'r1', cursor: 9, status: 'failed', error: 'nope',
    }]);

    expect(normalizeExecutionFrame({
      type: 'execution_stopped', run_id: 'r1', cursor: 9, reason: 'cancelled',
    })).toEqual([{ type: 'run_finished', run_id: 'r1', cursor: 9, status: 'cancelled' }]);

    expect(normalizeExecutionFrame({
      type: 'execution_stopped', run_id: 'r1', cursor: 9, reason: 'interrupted',
    })).toEqual([{ type: 'run_finished', run_id: 'r1', cursor: 9, status: 'interrupted' }]);
  });

  it('drops a REJECTED submit — nothing ran, and run_id names somebody else', () => {
    expect(normalizeExecutionFrame({
      type: 'execution_error', run_id: 'still-running', cursor: 3,
      error: 'refused', rejected: true,
    })).toEqual([]);
  });

  it('drops a synthesised not_running stop', () => {
    expect(normalizeExecutionFrame({
      type: 'execution_stopped', run_id: 'r1', cursor: 3, reason: 'not_running',
    })).toEqual([]);
  });

  it('drops transport frames, which carry no cursor', () => {
    for (const frame of [
      { type: 'attached', run_id: 'r1', cursor: 4, status: 'running' },
      { type: 'detached', run_id: 'r1' },
      { type: 'cache_cleared' },
      { type: 'reconnected' },
      { type: 'error', error: 'bad action' },
    ]) {
      const events = normalizeExecutionFrame(frame);
      // `attached` does carry a cursor but is not a log event, so it must not
      // become a contract event either.
      expect(events).toEqual([]);
    }
  });

  it('drops event types outside the published union', () => {
    expect(normalizeExecutionFrame({
      type: 'artifact', run_id: 'r1', cursor: 6, artifact_id: 1, kind: 'file',
    })).toEqual([]);
    expect(normalizeExecutionFrame({
      type: 'run_warning', run_id: 'r1', cursor: 7, kind: 'dropped_signals',
    })).toEqual([]);
    expect(normalizeExecutionFrame({
      type: 'something_added_in_2027', run_id: 'r1', cursor: 8,
    })).toEqual([]);
  });

  it('drops junk', () => {
    for (const junk of [null, undefined, 42, 'frame', [], {}, { type: 'metric' }]) {
      expect(normalizeExecutionFrame(junk)).toEqual([]);
    }
  });
});

// ── the stream ───────────────────────────────────────────────────────────

describe('subscribeExecutionEvents', () => {
  it('taps every tab socket on the first subscriber and releases them on the last', () => {
    const [ws0, ws1] = seedTabs(2);
    expect(ws0.wildcardCount).toBe(0);

    const offA = subscribeExecutionEvents(vi.fn());
    const offB = subscribeExecutionEvents(vi.fn());
    expect(executionEventTapCount()).toBe(2);
    // One tap per socket however many plugins subscribe.
    expect(ws0.wildcardCount).toBe(1);
    expect(ws1.wildcardCount).toBe(1);

    offA();
    expect(ws0.wildcardCount).toBe(1);
    offB();
    expect(executionEventSubscriberCount()).toBe(0);
    expect(executionEventTapCount()).toBe(0);
    expect(ws0.wildcardCount).toBe(0);
    expect(ws1.wildcardCount).toBe(0);
  });

  it('delivers nothing until the frame fires, then the whole batch in order', () => {
    const [ws] = seedTabs(1);
    const seen: ExecutionEvent[] = [];
    subscribeExecutionEvents((e) => seen.push(e));

    ws.emit({ type: 'execution_start', run_id: 'r1', cursor: 1 });
    ws.emit({
      type: 'metric', run_id: 'r1', cursor: 2,
      points: [{ name: 'loss', value: 1, step: 1, node_id: null }],
    });
    ws.emit({ type: 'execution_complete', run_id: 'r1', cursor: 3 });
    expect(seen).toHaveLength(0);

    runFrame();
    expect(seen.map((e) => e.type)).toEqual(['run_started', 'metric', 'run_finished']);
  });

  it('fans one frame out to every subscriber', () => {
    const [ws] = seedTabs(1);
    const a = vi.fn();
    const b = vi.fn();
    subscribeExecutionEvents(a);
    subscribeExecutionEvents(b);
    ws.emit({ type: 'execution_start', run_id: 'r1', cursor: 1 });
    runFrame();
    expect(a).toHaveBeenCalledTimes(1);
    expect(b).toHaveBeenCalledTimes(1);
  });

  it('isolates a throwing subscriber from the others', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const [ws] = seedTabs(1);
    const after = vi.fn();
    subscribeExecutionEvents(() => { throw new Error('plugin exploded'); });
    subscribeExecutionEvents(after);

    ws.emit({ type: 'execution_start', run_id: 'r1', cursor: 1 });
    expect(() => runFrame()).not.toThrow();
    expect(after).toHaveBeenCalledTimes(1);
    expect(warn).toHaveBeenCalled();

    // ...and the stream keeps running afterwards.
    ws.emit({ type: 'execution_complete', run_id: 'r1', cursor: 2 });
    runFrame();
    expect(after).toHaveBeenCalledTimes(2);
  });

  it('a throwing subscriber does not stop the host reading the same socket', () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const [ws] = seedTabs(1);
    const hostHandler = vi.fn();
    ws.on('*', hostHandler);
    subscribeExecutionEvents(() => { throw new Error('plugin exploded'); });

    // The plugin bridge only BUFFERS on the socket callback; it cannot throw
    // synchronously into the dispatch loop the host shares.
    expect(() => ws.emit({ type: 'execution_start', run_id: 'r1', cursor: 1 }))
      .not.toThrow();
    expect(hostHandler).toHaveBeenCalledTimes(1);
    runFrame();
    expect(hostHandler).toHaveBeenCalledTimes(1);
  });

  it('unsubscribing mid-batch stops delivery to that subscriber only', () => {
    const [ws] = seedTabs(1);
    const other = vi.fn();
    let off = () => {};
    const first = vi.fn(() => off());
    off = subscribeExecutionEvents(first);
    subscribeExecutionEvents(other);

    ws.emit({ type: 'execution_start', run_id: 'r1', cursor: 1 });
    ws.emit({ type: 'execution_complete', run_id: 'r1', cursor: 2 });
    runFrame();
    // The batch is delivered against a snapshot, so the mid-batch removal
    // takes effect from the next frame rather than corrupting this one.
    expect(other).toHaveBeenCalledTimes(2);

    ws.emit({ type: 'execution_start', run_id: 'r2', cursor: 1 });
    runFrame();
    expect(first).toHaveBeenCalledTimes(2);
    expect(other).toHaveBeenCalledTimes(3);
  });

  it('collapses the same frame arriving on two tabs attached to one run', () => {
    const [ws0, ws1] = seedTabs(2);
    const seen: ExecutionEvent[] = [];
    subscribeExecutionEvents((e) => seen.push(e));

    ws0.emit({ type: 'execution_start', run_id: 'r1', cursor: 1 });
    ws1.emit({ type: 'execution_start', run_id: 'r1', cursor: 1 });
    runFrame();
    expect(seen).toHaveLength(1);
  });

  it('does not confuse the same cursor in two different runs', () => {
    const [ws] = seedTabs(1);
    const seen: ExecutionEvent[] = [];
    subscribeExecutionEvents((e) => seen.push(e));
    ws.emit({ type: 'execution_start', run_id: 'r1', cursor: 1 });
    ws.emit({ type: 'execution_start', run_id: 'r2', cursor: 1 });
    runFrame();
    expect(seen.map((e) => e.run_id)).toEqual(['r1', 'r2']);
  });

  it('follows tabs opening and closing', () => {
    const [ws0] = seedTabs(1);
    const seen: ExecutionEvent[] = [];
    subscribeExecutionEvents((e) => seen.push(e));
    expect(executionEventTapCount()).toBe(1);

    const ws1 = new FakeWs();
    useTabStore.setState({
      tabs: [...useTabStore.getState().tabs, extraTab('t1', ws1)] as never,
    });
    expect(executionEventTapCount()).toBe(2);
    ws1.emit({ type: 'execution_start', run_id: 'r9', cursor: 1 });
    runFrame();
    expect(seen).toHaveLength(1);

    useTabStore.setState({ tabs: [] } as never);
    expect(executionEventTapCount()).toBe(0);
    expect(ws0.wildcardCount).toBe(0);
    expect(ws1.wildcardCount).toBe(0);
  });

  it('buffers nothing once the last subscriber has gone', () => {
    const [ws] = seedTabs(1);
    const seen = vi.fn();
    const off = subscribeExecutionEvents(seen);
    off();
    ws.emit({ type: 'execution_start', run_id: 'r1', cursor: 1 });
    runFrame();
    expect(seen).not.toHaveBeenCalled();
  });

  it('forgets seen cursors after a full teardown, so a re-subscribe is clean', () => {
    const [ws] = seedTabs(1);
    const first: ExecutionEvent[] = [];
    const off = subscribeExecutionEvents((e) => first.push(e));
    ws.emit({ type: 'execution_start', run_id: 'r1', cursor: 1 });
    runFrame();
    off();

    const second: ExecutionEvent[] = [];
    subscribeExecutionEvents((e) => second.push(e));
    ws.emit({ type: 'execution_start', run_id: 'r1', cursor: 1 });
    runFrame();
    expect(second).toHaveLength(1);
  });

  it('de-duplicates only within a bounded window', () => {
    const [ws] = seedTabs(1);
    const seen: ExecutionEvent[] = [];
    subscribeExecutionEvents((e) => seen.push(e));
    for (let cursor = 1; cursor <= EXECUTION_EVENT_DEDUPE_WINDOW + 1; cursor += 1) {
      ws.emit({ type: 'execution_start', run_id: 'r1', cursor });
      runFrame();
    }
    const before = seen.length;
    // The oldest key has been evicted, so replaying it is no longer detected.
    ws.emit({ type: 'execution_start', run_id: 'r1', cursor: 1 });
    runFrame();
    expect(seen.length).toBe(before + 1);
  });
});

describe('the buffer stays bounded when no frame ever comes', () => {
  it('drops metrics rather than growing without limit, and keeps lifecycle events', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const [ws] = seedTabs(1);
    const seen: ExecutionEvent[] = [];
    subscribeExecutionEvents((e) => seen.push(e));

    // An occluded document keeps requestAnimationFrame but never calls it, so
    // nothing flushes while this runs.
    ws.emit({ type: 'execution_start', run_id: 'r1', cursor: 1 });
    const overflow = MAX_BUFFERED_EXECUTION_EVENTS + 500;
    for (let i = 0; i < overflow; i += 1) {
      ws.emit({
        type: 'metric', run_id: 'r1', cursor: i + 2,
        points: [{ name: 'loss', value: i, step: i, node_id: null }],
      });
    }
    ws.emit({ type: 'execution_complete', run_id: 'r1', cursor: overflow + 2 });

    flushExecutionEvents();
    expect(seen.length).toBe(MAX_BUFFERED_EXECUTION_EVENTS);
    // Both lifecycle events survived the eviction.
    expect(seen[0].type).toBe('run_started');
    expect(seen[seen.length - 1]).toMatchObject({
      type: 'run_finished', status: 'succeeded',
    });
    // The newest metrics are the ones kept.
    const metrics = seen.filter((e) => e.type === 'metric');
    expect(metrics[metrics.length - 1]).toMatchObject({ step: overflow - 1 });
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('dropped'));
  });
});
