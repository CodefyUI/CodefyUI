import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { useTabStore } from '../store/tabStore';
import type { ExecutionWebSocket } from '../api/ws';
import {
  EXECUTION_EVENT_WATERMARK_RUNS,
  MAX_BUFFERED_EXECUTION_WEIGHT,
  executionEventWeight,
  _resetExecutionEvents,
  executionEventSubscriberCount,
  executionEventTapCount,
  flushExecutionEvents,
  normalizeExecutionFrame,
  subscribeExecutionEvents,
  type ExecutionEvent,
  type ExecutionEventDraft,
} from './executionEvents';
import type { RunMetricPoint } from '../api/rest';

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

  it('keeps a batched metric frame whole, as one event carrying its points', () => {
    expect(normalizeExecutionFrame({
      type: 'metric', run_id: 'r1', cursor: 4,
      points: [
        { name: 'loss', value: 0.5, step: 1, node_id: 'n1' },
        { name: 'acc', value: 0.9, step: 1, node_id: null },
      ],
    })).toEqual([{
      type: 'metric', run_id: 'r1', cursor: 4,
      points: [
        { node_id: 'n1', name: 'loss', step: 1, value: 0.5 },
        { node_id: null, name: 'acc', step: 1, value: 0.9 },
      ],
    }]);
  });

  // The server runs json_safe() over every payload, so a diverged loss
  // reaches the socket as `value: null`. That is a first-class value with a
  // documented meaning on the REST half of this same contract ("a gap, not a
  // zero"), and the moment a dashboard most needs to draw something. Dropping
  // it here would make the two halves of v3 disagree.
  it('DELIVERS a non-finite value as an explicit null gap', () => {
    expect(normalizeExecutionFrame({
      type: 'metric', run_id: 'r1', cursor: 4,
      points: [
        { name: 'loss', value: 0.5, step: 1, node_id: null },
        { name: 'loss', value: null, step: 2, node_id: null },
        { name: 'loss', value: 0.3, step: 3, node_id: null },
      ],
    })).toEqual([{
      type: 'metric', run_id: 'r1', cursor: 4,
      points: [
        { node_id: null, name: 'loss', step: 1, value: 0.5 },
        { node_id: null, name: 'loss', step: 2, value: null },
        { node_id: null, name: 'loss', step: 3, value: 0.3 },
      ],
    }]);
  });

  it('agrees with what runs.metrics() really returns for the same point', () => {
    // Not a shape assertion for its own sake: a dashboard folds the live tail
    // and the REST back-fill with one function, which only works if a point
    // from each side is the same thing to that function.
    //
    // The REST side here is what the SERVER actually sends
    // (`routes_runs.py` -> node_id/name/step/value/ts), not a hand-built
    // literal — `getRunMetrics` returns `res.json()` unmapped, so the extra
    // `ts` key reaches plugin code verbatim and the contract declares it.
    const restPoint: RunMetricPoint = {
      node_id: null, name: 'loss', step: 2, value: null,
      ts: '2026-08-02T09:15:00.000Z',
    };
    const [live] = normalizeExecutionFrame({
      type: 'metric', run_id: 'r1', cursor: 4,
      points: [{ name: 'loss', value: null, step: 2 }],
    });
    expect(live.type).toBe('metric');
    const livePoint = (live as Extract<ExecutionEventDraft, { type: 'metric' }>)
      .points[0];

    // Every field a fold reads is identical...
    const foldFields = (p: RunMetricPoint) =>
      ({ node_id: p.node_id, name: p.name, step: p.step, value: p.value });
    expect(foldFields(livePoint)).toEqual(foldFields(restPoint));

    // ...and `ts` is the one documented difference: the REST half records it,
    // the live half never carries it, and it is optional for exactly that
    // reason. Asserted rather than assumed, because an undeclared field
    // leaking through a typed facade is how it becomes load-bearing.
    expect(restPoint.ts).toBeDefined();
    expect(livePoint.ts).toBeUndefined();
    expect('ts' in livePoint).toBe(false);
  });

  it('skips genuinely malformed points without losing the good ones', () => {
    const events = normalizeExecutionFrame({
      type: 'metric', run_id: 'r1', cursor: 5,
      points: [
        null,
        'garbage',
        { name: '', value: 1, step: 1 },
        { name: 'loss', value: 1, step: 'one' },
        { name: 'loss', value: 'NaN', step: 4 },
        { name: 'loss', value: undefined, step: 5 },
        { name: 'loss', value: 0.25, step: 7 },
      ],
    });
    expect(events).toEqual([{
      type: 'metric', run_id: 'r1', cursor: 5,
      points: [{ node_id: null, name: 'loss', step: 7, value: 0.25 }],
    }]);
  });

  it('drops a frame whose every point was malformed', () => {
    expect(normalizeExecutionFrame({
      type: 'metric', run_id: 'r1', cursor: 5, points: ['garbage', null],
    })).toEqual([]);
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

  /**
   * The two tests above feed frames carrying a cursor, which is what makes the
   * `rejected` / `not_running` guards reachable at all. The real server does
   * not send them that way: both are answered by the WS handler itself and
   * never written to a run's durable log, so the no-cursor rule drops them
   * first and the guards are belt to its braces. Pinned because the module
   * doc now says exactly that, and a doc claim nobody checks goes stale.
   */
  it('drops the wire shapes of a refusal and a no-op cancel on the no-cursor rule', () => {
    for (const frame of [
      // ws_execution.py, handle_submit: the interactive cap / one-run-per-session
      // refusal. `run_id` names the run this socket was ALREADY following.
      {
        type: 'execution_error', error: 'too many interactive runs',
        rejected: true, run_id: 'still-running',
      },
      // ws_execution.py, handle_cancel: both "nothing to cancel" answers.
      { type: 'execution_stopped', reason: 'not_running' },
      { type: 'execution_stopped', reason: 'not_running', run_id: 'r1' },
    ]) {
      expect(normalizeExecutionFrame(frame), JSON.stringify(frame)).toEqual([]);
    }
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

  it('subscribing from inside a callback does not disturb the running batch', () => {
    const [ws] = seedTabs(1);
    const late = vi.fn();
    subscribeExecutionEvents((e) => {
      if (e.type === 'run_started') subscribeExecutionEvents(late);
    });

    ws.emit({ type: 'execution_start', run_id: 'r1', cursor: 1 });
    ws.emit({ type: 'execution_complete', run_id: 'r1', cursor: 2 });
    runFrame();
    // The batch runs against a snapshot, so a subscriber added mid-batch
    // starts at the NEXT frame rather than seeing half of this one.
    expect(late).not.toHaveBeenCalled();

    ws.emit({ type: 'execution_start', run_id: 'r2', cursor: 1 });
    runFrame();
    expect(late).toHaveBeenCalledTimes(1);
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

  /**
   * A tab keeps one socket for its whole life today, so this cannot happen
   * through the store's own API — which is exactly why it is worth pinning.
   * An id-only guard fails silently here: the tab still LOOKS covered, while
   * the tap sits on the socket nobody is using and every frame of the live one
   * goes unseen. Nothing throws and no count looks wrong.
   */
  it('re-taps a tab whose socket is replaced in place', () => {
    const [ws0] = seedTabs(1);
    const seen: ExecutionEvent[] = [];
    subscribeExecutionEvents((e) => seen.push(e));
    expect(ws0.wildcardCount).toBe(1);

    const ws1 = new FakeWs();
    useTabStore.setState({
      tabs: [{
        ...useTabStore.getState().tabs[0],
        ws: ws1 as unknown as ExecutionWebSocket,
      }] as never,
    });

    // Still one tap for the one tab — but on the socket it actually has now,
    // and the handler on the discarded one is released rather than leaked.
    expect(executionEventTapCount()).toBe(1);
    expect(ws0.wildcardCount).toBe(0);
    expect(ws1.wildcardCount).toBe(1);

    ws1.emit({ type: 'execution_start', run_id: 'r1', cursor: 1 });
    runFrame();
    expect(seen.map((e) => e.type)).toEqual(['run_started']);
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

  it('swallows a full cursor-0 re-attach replay, however long the run', () => {
    // Both of the editor's attach paths send `cursor: 0` on purpose, so the
    // server replays the entire durable log through the same '*' slot this
    // module taps. A 512-key window could not cover a real training run; a
    // per-run watermark covers any length.
    const [ws] = seedTabs(1);
    const seen: ExecutionEvent[] = [];
    subscribeExecutionEvents((e) => seen.push(e));

    const RUN_LENGTH = 1000;
    for (let cursor = 1; cursor <= RUN_LENGTH; cursor += 1) {
      ws.emit({
        type: 'metric', run_id: 'r1', cursor,
        points: [{ name: 'loss', value: 1 / cursor, step: cursor, node_id: null }],
      });
    }
    flushExecutionEvents();
    expect(seen).toHaveLength(RUN_LENGTH);

    // The user reloads or clicks re-attach: the whole log arrives again.
    for (let cursor = 1; cursor <= RUN_LENGTH; cursor += 1) {
      ws.emit({
        type: 'metric', run_id: 'r1', cursor,
        points: [{ name: 'loss', value: 1 / cursor, step: cursor, node_id: null }],
      });
    }
    flushExecutionEvents();
    expect(seen).toHaveLength(RUN_LENGTH);

    // ...and the live tail that follows the replay still gets through.
    ws.emit({ type: 'execution_complete', run_id: 'r1', cursor: RUN_LENGTH + 1 });
    flushExecutionEvents();
    expect(seen).toHaveLength(RUN_LENGTH + 1);
    expect(seen[seen.length - 1].type).toBe('run_finished');
  });

  it('delivers cursors strictly increasing per run, so a jump means real loss', () => {
    const [ws] = seedTabs(1);
    const seen: ExecutionEvent[] = [];
    subscribeExecutionEvents((e) => seen.push(e));
    for (const cursor of [1, 2, 3, 2, 1, 3, 4, 4, 5]) {
      ws.emit({ type: 'execution_start', run_id: 'r1', cursor });
    }
    flushExecutionEvents();
    expect(seen.map((e) => e.cursor)).toEqual([1, 2, 3, 4, 5]);
  });

  it('keeps a watermark per run, not one for the workspace', () => {
    const [ws] = seedTabs(1);
    const seen: ExecutionEvent[] = [];
    subscribeExecutionEvents((e) => seen.push(e));
    ws.emit({ type: 'execution_start', run_id: 'r1', cursor: 900 });
    // A different run starting at cursor 1 is not "behind" r1.
    ws.emit({ type: 'execution_start', run_id: 'r2', cursor: 1 });
    flushExecutionEvents();
    expect(seen.map((e) => [e.run_id, e.cursor])).toEqual([['r1', 900], ['r2', 1]]);
  });

  it('bounds the watermark table, forgetting the least recently advanced run', () => {
    // The documented failure mode: past the bound, an evicted run's replay is
    // delivered again. The bound is set high enough that reaching it is not a
    // session a user has, but the condition is exact and published rather
    // than left implicit.
    const [ws] = seedTabs(1);
    const seen: ExecutionEvent[] = [];
    subscribeExecutionEvents((e) => seen.push(e));
    ws.emit({ type: 'execution_start', run_id: 'old', cursor: 5 });
    for (let i = 0; i < EXECUTION_EVENT_WATERMARK_RUNS; i += 1) {
      ws.emit({ type: 'execution_start', run_id: `r${i}`, cursor: 1 });
    }
    flushExecutionEvents();
    const before = seen.length;
    ws.emit({ type: 'execution_start', run_id: 'old', cursor: 5 });
    flushExecutionEvents();
    expect(seen.length).toBe(before + 1);
  });

  it('evicts by least-recently-ADVANCED, not least-recently-first-seen', () => {
    // Without the re-insert in acceptFrame the run that must survive here is
    // the one evicted, because it was inserted first.
    const [ws] = seedTabs(1);
    const seen: ExecutionEvent[] = [];
    subscribeExecutionEvents((e) => seen.push(e));

    ws.emit({ type: 'execution_start', run_id: 'veteran', cursor: 1 });
    for (let i = 0; i < EXECUTION_EVENT_WATERMARK_RUNS - 1; i += 1) {
      ws.emit({ type: 'execution_start', run_id: `r${i}`, cursor: 1 });
    }
    // Touch the veteran again: it is now the most recently advanced run, so
    // the next insertion must evict r0, not it.
    ws.emit({
      type: 'node_status', run_id: 'veteran', cursor: 2,
      node_id: 'n', status: 'running',
    });
    ws.emit({ type: 'execution_start', run_id: 'newcomer', cursor: 1 });
    flushExecutionEvents();
    seen.length = 0;

    // The veteran kept its watermark: a replay of cursors 1-2 is swallowed.
    ws.emit({ type: 'execution_start', run_id: 'veteran', cursor: 1 });
    ws.emit({
      type: 'node_status', run_id: 'veteran', cursor: 2,
      node_id: 'n', status: 'running',
    });
    // r0 lost its watermark: its replay comes through.
    ws.emit({ type: 'execution_start', run_id: 'r0', cursor: 1 });
    flushExecutionEvents();
    expect(seen.map((e) => e.run_id)).toEqual(['r0']);
  });
});

describe('seq -- the dense counter that actually signals loss', () => {
  it('stays dense across durable entries the stream does not publish', () => {
    // The log entries here are cursors 1-5, but artifact and run_warning are
    // outside the published union. A run that saves a checkpoint emits an
    // artifact every time, so a dashboard told to read a cursor jump as data
    // loss would cry wolf on every checkpoint.
    const [ws] = seedTabs(1);
    const seen: ExecutionEvent[] = [];
    subscribeExecutionEvents((e) => seen.push(e));

    ws.emit({ type: 'execution_start', run_id: 'r1', cursor: 1 });
    ws.emit({
      type: 'artifact', run_id: 'r1', cursor: 2,
      artifact_id: 7, kind: 'checkpoint',
    });
    ws.emit({ type: 'run_warning', run_id: 'r1', cursor: 3, kind: 'dropped_signals' });
    ws.emit({
      type: 'metric', run_id: 'r1', cursor: 4,
      points: [{ name: 'loss', value: 0.5, step: 1, node_id: null }],
    });
    ws.emit({ type: 'execution_complete', run_id: 'r1', cursor: 5 });
    flushExecutionEvents();

    expect(seen.map((e) => e.cursor)).toEqual([1, 4, 5]);
    expect(seen.map((e) => e.seq)).toEqual([1, 2, 3]);
  });

  it('stays dense across an all-malformed metric entry', () => {
    const [ws] = seedTabs(1);
    const seen: ExecutionEvent[] = [];
    subscribeExecutionEvents((e) => seen.push(e));
    ws.emit({ type: 'execution_start', run_id: 'r1', cursor: 1 });
    ws.emit({ type: 'metric', run_id: 'r1', cursor: 2, points: ['garbage'] });
    ws.emit({ type: 'execution_complete', run_id: 'r1', cursor: 3 });
    flushExecutionEvents();
    expect(seen.map((e) => e.seq)).toEqual([1, 2]);
  });

  it('counts per run, not per workspace', () => {
    const [ws] = seedTabs(1);
    const seen: ExecutionEvent[] = [];
    subscribeExecutionEvents((e) => seen.push(e));
    ws.emit({ type: 'execution_start', run_id: 'a', cursor: 1 });
    ws.emit({ type: 'execution_start', run_id: 'b', cursor: 1 });
    ws.emit({ type: 'execution_complete', run_id: 'a', cursor: 2 });
    ws.emit({ type: 'execution_complete', run_id: 'b', cursor: 2 });
    flushExecutionEvents();
    expect(seen.map((e) => [e.run_id, e.seq]))
      .toEqual([['a', 1], ['b', 1], ['a', 2], ['b', 2]]);
  });

  it('does not advance for a swallowed replay', () => {
    const [ws] = seedTabs(1);
    const seen: ExecutionEvent[] = [];
    subscribeExecutionEvents((e) => seen.push(e));
    ws.emit({ type: 'execution_start', run_id: 'r1', cursor: 1 });
    ws.emit({ type: 'execution_start', run_id: 'r1', cursor: 1 });
    ws.emit({ type: 'execution_complete', run_id: 'r1', cursor: 2 });
    flushExecutionEvents();
    expect(seen.map((e) => e.seq)).toEqual([1, 2]);
  });

  it('JUMPS when the buffer drops events -- the one thing that makes a hole', () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const [ws] = seedTabs(1);
    const seen: ExecutionEvent[] = [];
    subscribeExecutionEvents((e) => seen.push(e));

    // A first frame gets through normally.
    ws.emit({ type: 'execution_start', run_id: 'r1', cursor: 1 });
    flushExecutionEvents();
    expect(seen.map((e) => e.seq)).toEqual([1]);

    // Then the window goes dark and more arrives than the buffer holds.
    const dropped = 200;
    // Every metric below carries one point, so each costs 2 against the
    // weight budget: the event itself plus its point.
    const capacity = Math.floor(MAX_BUFFERED_EXECUTION_WEIGHT / 2);
    const total = capacity + dropped;
    for (let i = 0; i < total; i += 1) {
      ws.emit({
        type: 'metric', run_id: 'r1', cursor: i + 2,
        points: [{ name: 'loss', value: i, step: i, node_id: null }],
      });
    }
    flushExecutionEvents();

    const seqs = seen.map((e) => e.seq);
    expect(seqs).toHaveLength(capacity + 1);
    // Numbers were handed out to all of them, so the survivors run to the end
    // and the ones the buffer sacrificed are simply absent.
    expect(seqs[seqs.length - 1]).toBe(total + 1);
    // The subscriber had seq 1; the next thing it sees is far past 2. That
    // jump is the signal, and it is the ONLY way to get one.
    expect(seqs[1]).toBe(dropped + 2);
    expect(seqs[1] - seqs[0]).toBeGreaterThan(1);
  });
});

describe('delivered events are shared, so they are frozen', () => {
  it('one subscriber cannot empty points for another', () => {
    const [ws] = seedTabs(1);
    subscribeExecutionEvents((e) => {
      if (e.type !== 'metric') return;
      // `readonly` is compile-time only; a plugin bundle never sees it.
      try {
        (e.points as RunMetricPoint[]).length = 0;
      } catch {
        /* frozen, so strict mode throws -- which is the point */
      }
    });
    const second: ExecutionEvent[] = [];
    subscribeExecutionEvents((e) => second.push(e));

    ws.emit({
      type: 'metric', run_id: 'r1', cursor: 1,
      points: [{ name: 'loss', value: 0.5, step: 1, node_id: null }],
    });
    flushExecutionEvents();

    const event = second[0] as Extract<ExecutionEvent, { type: 'metric' }>;
    expect(event.points).toHaveLength(1);
    expect(event.points[0].value).toBe(0.5);
  });

  it('freezes the event, its points array and each point', () => {
    const [ws] = seedTabs(1);
    const seen: ExecutionEvent[] = [];
    subscribeExecutionEvents((e) => seen.push(e));
    ws.emit({
      type: 'metric', run_id: 'r1', cursor: 1,
      points: [{ name: 'loss', value: 0.5, step: 1, node_id: null }],
    });
    flushExecutionEvents();
    const event = seen[0] as Extract<ExecutionEvent, { type: 'metric' }>;
    expect(Object.isFrozen(event)).toBe(true);
    expect(Object.isFrozen(event.points)).toBe(true);
    expect(Object.isFrozen(event.points[0])).toBe(true);
  });
});

describe('unsubscribe takes effect immediately', () => {
  it('stops delivery mid-batch for the handler that unsubscribed', () => {
    // The obvious plugin idiom is "unsubscribe on run_finished, then tear my
    // state down". Delivering the rest of the batch afterwards would call
    // into code that has already gone.
    const [ws] = seedTabs(1);
    const seen: ExecutionEvent[] = [];
    let off = () => {};
    off = subscribeExecutionEvents((e) => {
      seen.push(e);
      if (e.type === 'run_started') off();
    });

    ws.emit({ type: 'execution_start', run_id: 'r1', cursor: 1 });
    ws.emit({ type: 'execution_complete', run_id: 'r1', cursor: 2 });
    flushExecutionEvents();
    expect(seen.map((e) => e.type)).toEqual(['run_started']);
  });

  it('does not disturb the other subscribers in the same batch', () => {
    const [ws] = seedTabs(1);
    const other: ExecutionEvent[] = [];
    let off = () => {};
    off = subscribeExecutionEvents(() => off());
    subscribeExecutionEvents((e) => other.push(e));

    ws.emit({ type: 'execution_start', run_id: 'r1', cursor: 1 });
    ws.emit({ type: 'execution_complete', run_id: 'r1', cursor: 2 });
    flushExecutionEvents();
    expect(other).toHaveLength(2);
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
    const overflow = Math.floor(MAX_BUFFERED_EXECUTION_WEIGHT / 2) + 500;
    for (let i = 0; i < overflow; i += 1) {
      ws.emit({
        type: 'metric', run_id: 'r1', cursor: i + 2,
        points: [{ name: 'loss', value: i, step: i, node_id: null }],
      });
    }
    ws.emit({ type: 'execution_complete', run_id: 'r1', cursor: overflow + 2 });

    flushExecutionEvents();
    // What the cap promises is a bound on RETAINED OBJECTS, not on events, so
    // that is what gets asserted: the delivered batch never weighed more than
    // the budget.
    const delivered = seen.reduce((sum, e) => sum + executionEventWeight(e), 0);
    expect(delivered).toBeLessThanOrEqual(MAX_BUFFERED_EXECUTION_WEIGHT);
    // Two-sided on purpose. An upper bound alone is satisfied by ANY smaller
    // cap - including the event count this replaced - so without the lower
    // bound this test passes on the very defect the change exists to fix.
    expect(delivered).toBeGreaterThan(MAX_BUFFERED_EXECUTION_WEIGHT / 2);
    expect(seen.length).toBeLessThan(overflow);
    // Both lifecycle events survived the eviction.
    expect(seen[0].type).toBe('run_started');
    expect(seen[seen.length - 1]).toMatchObject({
      type: 'run_finished', status: 'succeeded',
    });
    // The newest metrics are the ones kept.
    const metrics = seen.filter((e) => e.type === 'metric');
    expect(metrics[metrics.length - 1]).toMatchObject({
      points: [{ name: 'loss', step: overflow - 1, value: overflow - 1 }],
    });
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('dropped'));
  });

  it('evicts whole entries, so a delivered cursor is always a complete one', () => {
    // The reason `metric` carries its batch instead of being expanded: with
    // one event per point, eviction removed points from the MIDDLE of an
    // entry while delivering the rest, and `cursor` then lied about what the
    // plugin held.
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const [ws] = seedTabs(1);
    const seen: ExecutionEvent[] = [];
    subscribeExecutionEvents((e) => seen.push(e));

    const POINTS_PER_FRAME = 200;
    // Comfortably more frames than the weight budget can hold, so eviction
    // definitely runs.
    const frames = Math.ceil(MAX_BUFFERED_EXECUTION_WEIGHT / POINTS_PER_FRAME) + 100;
    for (let i = 0; i < frames; i += 1) {
      ws.emit({
        type: 'metric', run_id: 'r1', cursor: i + 1,
        points: Array.from({ length: POINTS_PER_FRAME }, (_, k) => ({
          name: `series${k}`, value: k, step: i, node_id: null,
        })),
      });
    }
    flushExecutionEvents();

    expect(seen.length).toBeGreaterThan(0);
    for (const event of seen) {
      expect(event.type).toBe('metric');
      expect((event as Extract<ExecutionEvent, { type: 'metric' }>).points)
        .toHaveLength(POINTS_PER_FRAME);
    }
  });

  it('bounds retained points, so fat frames are held to the same budget as thin ones', () => {
    // The bound this pins: what a hidden tab retains is metric POINTS, not
    // events. Counting events let a run writing big batches hold three orders
    // of magnitude more than one writing single points, in the exact scenario
    // this API exists for (a dashboard watching a long run in a background
    // tab). Weight makes both runs cost the same.
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const [ws] = seedTabs(1);
    const seen: ExecutionEvent[] = [];
    subscribeExecutionEvents((e) => seen.push(e));

    const FAT = 1000;
    const frames = Math.ceil(MAX_BUFFERED_EXECUTION_WEIGHT / FAT) + 50;
    for (let i = 0; i < frames; i += 1) {
      ws.emit({
        type: 'metric', run_id: 'r1', cursor: i + 1,
        points: Array.from({ length: FAT }, (_, k) => ({
          name: 'loss', value: k, step: i * FAT + k, node_id: null,
        })),
      });
    }
    flushExecutionEvents();

    const retainedPoints = seen.reduce(
      (sum, e) => sum + (e.type === 'metric' ? e.points.length : 0), 0,
    );
    expect(retainedPoints).toBeLessThanOrEqual(MAX_BUFFERED_EXECUTION_WEIGHT);
    // Not vacuous: the budget is actually close to full, and far fewer frames
    // survived than an event-count cap of the same number would have kept.
    expect(retainedPoints).toBeGreaterThan(MAX_BUFFERED_EXECUTION_WEIGHT / 2);
    expect(seen.length).toBeLessThan(frames);

    // Eviction under weight still punches a visible hole: the survivors are
    // the newest frames, so seq starts well past 1 and runs unbroken to the end.
    const seqs = seen.map((e) => e.seq);
    expect(seqs[0]).toBeGreaterThan(1);
    expect(seqs[seqs.length - 1]).toBe(frames);
    for (let i = 1; i < seqs.length; i += 1) {
      expect(seqs[i]).toBe(seqs[i - 1] + 1);
    }
  });

  it('a teardown with a full buffer does not leave the budget spent', () => {
    // The weight total is kept incrementally, so every path that empties the
    // buffer has to zero it too. Tearing down while a hidden tab is holding
    // events is the path that is easy to miss: nothing flushes there, so a
    // leaked total never self-heals, and the next plugin to subscribe finds a
    // buffer that can hold exactly one event.
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const [ws] = seedTabs(1);
    const stop = subscribeExecutionEvents(() => {});
    for (let i = 0; i < 400; i += 1) {
      ws.emit({
        type: 'metric', run_id: 'r1', cursor: i + 1,
        points: Array.from({ length: 50 }, (_, k) => ({
          name: 'loss', value: k, step: i, node_id: null,
        })),
      });
    }
    stop(); // last subscriber leaves; the buffer is discarded unflushed

    const seen: ExecutionEvent[] = [];
    subscribeExecutionEvents((e) => seen.push(e));
    for (let i = 0; i < 100; i += 1) {
      ws.emit({
        type: 'metric', run_id: 'r2', cursor: i + 1,
        points: [{ name: 'loss', value: i, step: i, node_id: null }],
      });
    }
    flushExecutionEvents();
    // 100 events at weight 2 is 1% of the budget: nothing should be evicted.
    expect(seen).toHaveLength(100);
  });

  it('never buys room with a lifecycle event, however heavy the arrival', () => {
    // Metrics are re-readable from api.runs.metrics(); run_finished is not.
    // So an arriving event that cannot fit overshoots the budget rather than
    // cascading through the events a plugin can never recover.
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const [ws] = seedTabs(1);
    const seen: ExecutionEvent[] = [];
    subscribeExecutionEvents((e) => seen.push(e));

    ws.emit({ type: 'execution_start', run_id: 'r1', cursor: 1 });
    ws.emit({ type: 'execution_complete', run_id: 'r1', cursor: 2 });
    ws.emit({
      type: 'metric', run_id: 'r1', cursor: 3,
      points: Array.from({ length: MAX_BUFFERED_EXECUTION_WEIGHT * 3 }, (_, k) => ({
        name: 'loss', value: k, step: k, node_id: null,
      })),
    });
    flushExecutionEvents();

    expect(seen.map((e) => e.type)).toEqual(['run_started', 'run_finished', 'metric']);
  });
});
