/**
 * #559: `applyTabNodeUpdates` keeps what a run reports for a node inside a
 * block under the id the run gives it, and paints it on every open level that
 * shows the node, leaving every other level and node as it was.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import type { Node } from '@xyflow/react';

import { useTabStore, type TabState } from './tabStore';
import { subgraphFrame } from '../test/openBlocks';
import type { NodeData } from '../types';

const store = () => useTabStore.getState();
const tab = () => store().getActiveTab();

function node(id: string): Node<NodeData> {
  return { id, type: 'baseNode', position: { x: 0, y: 0 }, data: { label: id, type: 'X', params: {} } };
}

function patchTab(patch: Partial<TabState>) {
  const { tabs, activeTabId } = useTabStore.getState();
  useTabStore.setState({ tabs: tabs.map((t) => (t.id === activeTabId ? { ...t, ...patch } : t)) });
}

function inner(runId: string, executionStatus: NodeData['executionStatus']) {
  store().applyTabNodeUpdates(
    new Map([[tab().id, new Map([[runId, { inner: true as const, status: { executionStatus } }]])]]),
  );
}

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
  store().addTab('test');
});

describe('a run state for a node inside a block', () => {
  it('leaves a stashed level that does not show the node untouched', () => {
    const level1 = { ...subgraphFrame('nest'), nodes: [node('x')] };
    patchTab({
      subgraphStack: [{ ...subgraphFrame('blk'), nodes: [node('blk')] }, level1],
      nodes: [node('m')],
    });

    inner('blk/nest/m', 'completed');

    expect(tab().nodes[0].data.executionStatus).toBe('completed');
    expect(tab().subgraphStack[1]).toBe(level1);
  });

  it('is kept at the top level, where nothing shows it, for a tab with no stack at all', () => {
    patchTab({ subgraphStack: undefined as unknown as TabState['subgraphStack'], nodes: [node('blk')] });
    const nodes = tab().nodes;

    inner('blk/m', 'running');

    expect(tab().innerRunStates).toEqual({ 'blk/m': { executionStatus: 'running', error: undefined } });
    expect(tab().nodes).toBe(nodes);
  });
});
