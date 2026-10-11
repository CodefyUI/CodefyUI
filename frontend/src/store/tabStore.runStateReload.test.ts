/**
 * #714: a reload drops node run state together with the run id.
 *
 * The Inspector shows captures only for `lastRunId`, and a finished run's id
 * does not survive a reload. Node badges must follow the same rule, or a node
 * says Completed while the Inspector beside it says nothing was captured. A
 * run still in flight keeps both, so the re-attach goes on painting it.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { Node } from '@xyflow/react';

import {
  useTabStore,
  _persistedTabsForTesting,
  _tabFromPersistedForTesting,
  type TabState,
} from './tabStore';
import type { NodeData } from '../types';
import { subgraphIdOf } from '../utils/subgraph';

vi.mock('./tabPersistence', () => ({
  readSnapshot: vi.fn(async () => null),
  writeSnapshot: vi.fn(async () => {}),
}));

const store = () => useTabStore.getState();
const tab = () => store().getActiveTab();

function node(id: string, x = 0): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position: { x, y: 0 },
    data: { label: id, type: 'X', params: {}, executionStatus: 'idle' },
  };
}

/** Autosave the active tab and read it back, as a reload does. */
function reload(): TabState {
  const record = _persistedTabsForTesting(store().tabs).find((r) => r.id === tab().id)!;
  const restored = _tabFromPersistedForTesting(record, tab());
  useTabStore.setState({ tabs: [restored], activeTabId: restored.id });
  return restored;
}

const statusOf = (id: string) => tab().nodes.find((n) => n.id === id)!.data;

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
  store().addTab('test');
  store().setNodes([node('tensor', 0), node('softmax', 100), node('print', 200)]);
});

describe('reloading a tab after its run finished', () => {
  it('shows every node idle, as the Inspector does', () => {
    store().setLastRunId(tab().id, 'run-1');
    store().setNodeExecutionStatus('tensor', 'completed');
    store().setNodeExecutionStatus('softmax', 'completed');
    store().setNodeExecutionStatus('print', 'completed');
    store().setStatus('completed');

    const restored = reload();

    expect(restored.lastRunId).toBeNull();
    for (const id of ['tensor', 'softmax', 'print']) {
      expect(statusOf(id).executionStatus).toBe('idle');
    }
  });

  it('drops a failed node error badge and a progress bar by the same rule', () => {
    store().setLastRunId(tab().id, 'run-1');
    store().setNodeExecutionStatus('tensor', 'completed');
    store().setNodeExecutionStatus('softmax', 'error', 'shape mismatch');
    store().setNodes(
      tab().nodes.map((n) =>
        n.id === 'print'
          ? { ...n, data: { ...n.data, progress: { current: 3, total: 10 } as never } }
          : n,
      ),
    );
    store().setStatus('error');

    reload();

    expect(statusOf('softmax').executionStatus).toBe('idle');
    expect(statusOf('softmax').error).toBeUndefined();
    expect('progress' in statusOf('print')).toBe(false);
    // The document survives: only run state is dropped.
    expect(tab().nodes.map((n) => [n.id, n.position.x])).toEqual([
      ['tensor', 0],
      ['softmax', 100],
      ['print', 200],
    ]);
  });

  it('shows a block and the nodes inside it idle, also when saved from inside the block', () => {
    store().setNodes(tab().nodes.map((n) => ({ ...n, selected: n.id !== 'print' })));
    expect(store().collapseSelectionToSubgraph('Block').ok).toBe(true);
    const blockId = tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
    store().setLastRunId(tab().id, 'run-1');
    store().setNodeExecutionStatus(blockId, 'completed');
    expect(store().enterSubgraph(blockId)).toBe(true);
    store().applyTabNodeUpdates(
      new Map([[
        tab().id,
        new Map([[`${blockId}/softmax`, { inner: true as const, status: { executionStatus: 'completed' as const } }]]),
      ]]),
    );
    expect(statusOf('softmax').executionStatus).toBe('completed');
    store().setStatus('completed');

    reload();

    expect(tab().subgraphStack).toEqual([]);
    expect(statusOf(blockId).executionStatus).toBe('idle');
    expect(store().enterSubgraph(blockId)).toBe(true);
    expect(tab().nodes.map((n) => n.data.executionStatus)).toEqual(['idle', 'idle']);
  });
});

describe('reloading a tab whose run is still in flight', () => {
  it('keeps the run id and the node states the re-attach goes on updating', () => {
    store().setLastRunId(tab().id, 'run-2');
    store().setStatus('running');
    store().setNodeExecutionStatus('tensor', 'completed');
    store().setNodeExecutionStatus('softmax', 'running');

    const restored = reload();

    expect(restored.lastRunId).toBe('run-2');
    expect(statusOf('tensor').executionStatus).toBe('completed');
    expect(statusOf('softmax').executionStatus).toBe('running');
    expect(statusOf('print').executionStatus).toBe('idle');
  });
});
