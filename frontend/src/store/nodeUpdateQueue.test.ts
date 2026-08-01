import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import type { Node } from '@xyflow/react';

import { useTabStore } from './tabStore';
import type { NodeData } from '../types';
import {
  queueTabNodeStatus,
  queueTabNodeProgress,
  flushTabNodeUpdates,
  discardTabNodeUpdates,
  pendingTabNodeUpdateCount,
} from './nodeUpdateQueue';

function node(id: string): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position: { x: 0, y: 0 },
    data: { label: id, type: 'Test', params: {} },
  };
}

/** Install a tab store holding one tab with `count` nodes. */
function seedTab(count = 3) {
  const tab = {
    ...useTabStore.getState().tabs[0],
    id: 't1',
    nodes: Array.from({ length: count }, (_, i) => node(`n${i}`)),
    edges: [],
  };
  useTabStore.setState({ tabs: [tab as never], activeTabId: 't1' });
}

function tab() {
  return useTabStore.getState().tabs.find((t) => t.id === 't1')!;
}

/**
 * Count how many times the ACTIVE tab's nodes array is replaced. This is the
 * number the acceptance criterion is about: React Flow re-diffs every node
 * whenever that reference changes, so one rebuild per frame is the budget.
 */
function countNodeRebuilds(): { count: () => number; stop: () => void } {
  let rebuilds = 0;
  let previous = tab().nodes;
  const unsubscribe = useTabStore.subscribe((state) => {
    const nodes = state.tabs.find((t) => t.id === 't1')?.nodes;
    if (nodes && nodes !== previous) {
      previous = nodes;
      rebuilds += 1;
    }
  });
  return { count: () => rebuilds, stop: unsubscribe };
}

/** Advance past one animation frame with fake timers installed. */
function runFrame() {
  vi.advanceTimersByTime(20);
}

beforeEach(() => {
  discardTabNodeUpdates();
  seedTab();
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
  discardTabNodeUpdates();
});

describe('nodeUpdateQueue - coalescing', () => {
  it('does not touch the store until the frame fires', () => {
    queueTabNodeStatus('t1', 'n0', 'running');
    expect(tab().nodes[0].data.executionStatus).toBeUndefined();
    expect(pendingTabNodeUpdateCount()).toBe(1);

    runFrame();
    expect(tab().nodes[0].data.executionStatus).toBe('running');
    expect(pendingTabNodeUpdateCount()).toBe(0);
  });

  it('collapses a burst of events into ONE nodes-array rebuild', () => {
    const probe = countNodeRebuilds();
    for (let i = 0; i < 40; i += 1) {
      queueTabNodeStatus('t1', `n${i % 3}`, 'running');
      queueTabNodeProgress('t1', `n${i % 3}`, { event: 'batch', value: i });
    }
    expect(probe.count()).toBe(0);

    runFrame();
    expect(probe.count()).toBe(1);
    probe.stop();
  });

  it('keeps at most one rebuild per frame at 20 events/sec', () => {
    const probe = countNodeRebuilds();
    // 20 events/sec over one second, at ~60fps: 3 frames between events.
    for (let i = 0; i < 20; i += 1) {
      queueTabNodeProgress('t1', 'n0', { event: 'batch', value: i });
      vi.advanceTimersByTime(50);
    }
    flushTabNodeUpdates();
    // 20 events land in 20 distinct frames -> 20 rebuilds, never two in one
    // frame. The burst test above pins the other half of the budget.
    expect(probe.count()).toBeLessThanOrEqual(20);
    expect(probe.count()).toBeGreaterThan(0);
    expect(tab().nodes[0].data.progress).toEqual({ event: 'batch', value: 19 });
    probe.stop();
  });

  it('schedules a new frame after a flush', () => {
    queueTabNodeStatus('t1', 'n0', 'running');
    runFrame();
    queueTabNodeStatus('t1', 'n0', 'completed');
    expect(tab().nodes[0].data.executionStatus).toBe('running');
    runFrame();
    expect(tab().nodes[0].data.executionStatus).toBe('completed');
  });

  it('applies the LAST value when the same field is queued repeatedly', () => {
    queueTabNodeStatus('t1', 'n0', 'running');
    queueTabNodeStatus('t1', 'n0', 'error', 'boom');
    queueTabNodeStatus('t1', 'n0', 'completed');
    runFrame();
    expect(tab().nodes[0].data.executionStatus).toBe('completed');
    // The last status carried no error, so the previous one must not stick.
    expect(tab().nodes[0].data.error).toBeUndefined();
  });

  it('merges status and progress for the same node without losing either', () => {
    queueTabNodeStatus('t1', 'n0', 'running');
    queueTabNodeProgress('t1', 'n0', { event: 'epoch', epoch: 2 });
    runFrame();
    expect(tab().nodes[0].data.executionStatus).toBe('running');
    expect(tab().nodes[0].data.progress).toEqual({ event: 'epoch', epoch: 2 });
  });

  it('leaves untouched nodes referentially identical across a flush', () => {
    const before = tab().nodes;
    queueTabNodeStatus('t1', 'n1', 'running');
    runFrame();
    const after = tab().nodes;
    expect(after).not.toBe(before);
    expect(after[0]).toBe(before[0]);
    expect(after[2]).toBe(before[2]);
    expect(after[1]).not.toBe(before[1]);
  });
});

describe('nodeUpdateQueue - flush and discard', () => {
  it('flushTabNodeUpdates applies immediately and cancels the frame', () => {
    const probe = countNodeRebuilds();
    queueTabNodeStatus('t1', 'n0', 'running');
    flushTabNodeUpdates();
    expect(tab().nodes[0].data.executionStatus).toBe('running');
    expect(probe.count()).toBe(1);

    // The already-scheduled frame must not commit a second time.
    runFrame();
    expect(probe.count()).toBe(1);
    probe.stop();
  });

  it('flushing an empty queue does not touch the store', () => {
    const probe = countNodeRebuilds();
    flushTabNodeUpdates();
    expect(probe.count()).toBe(0);
    probe.stop();
  });

  it('discardTabNodeUpdates(tabId) drops that tab pending work only', () => {
    useTabStore.setState((s) => ({
      tabs: [...s.tabs, { ...s.tabs[0], id: 't2', nodes: [node('n0')] } as never],
    }));
    queueTabNodeStatus('t1', 'n0', 'running');
    queueTabNodeStatus('t2', 'n0', 'running');
    discardTabNodeUpdates('t1');
    runFrame();
    expect(tab().nodes[0].data.executionStatus).toBeUndefined();
    expect(
      useTabStore.getState().tabs.find((t) => t.id === 't2')!.nodes[0].data.executionStatus,
    ).toBe('running');
  });

  it('discardTabNodeUpdates() with no argument drops everything', () => {
    queueTabNodeStatus('t1', 'n0', 'running');
    discardTabNodeUpdates();
    expect(pendingTabNodeUpdateCount()).toBe(0);
    runFrame();
    expect(tab().nodes[0].data.executionStatus).toBeUndefined();
  });

  it('ignores queued updates for a tab that no longer exists', () => {
    queueTabNodeStatus('gone', 'n0', 'running');
    expect(() => runFrame()).not.toThrow();
    expect(tab().nodes[0].data.executionStatus).toBeUndefined();
  });

  it('ignores queued updates for a node that no longer exists', () => {
    const probe = countNodeRebuilds();
    queueTabNodeStatus('t1', 'ghost', 'running');
    runFrame();
    // Nothing matched, so the nodes array must be left alone entirely.
    expect(probe.count()).toBe(0);
    probe.stop();
  });
});

describe('nodeUpdateQueue - environments without requestAnimationFrame', () => {
  it('falls back to a timeout when rAF is missing', () => {
    vi.stubGlobal('requestAnimationFrame', undefined);
    vi.stubGlobal('cancelAnimationFrame', undefined);
    try {
      queueTabNodeStatus('t1', 'n0', 'running');
      expect(tab().nodes[0].data.executionStatus).toBeUndefined();
      runFrame();
      expect(tab().nodes[0].data.executionStatus).toBe('running');
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
