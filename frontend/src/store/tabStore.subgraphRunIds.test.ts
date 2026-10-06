/**
 * #621: which instance a block was entered through, and the run ids that
 * follow from it.
 *
 * The engine flattens a block before it runs, so an inner node is captured as
 * `<instance>/<inner>` -- `<outer>/<inner instance>/<node>` one level further
 * in -- while the open block shows the definition's nodes under their own ids.
 * A frame that records only the definition cannot tell the two copies of one
 * block apart; each frame records the instance it was entered through.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';

import { useTabStore } from './tabStore';
import { useNodeDefStore } from './nodeDefStore';
import type { NodeDefinition, SubgraphDefinition } from '../types';
import { buildInstanceNode } from '../utils/subgraph';
import { runNodePrefix } from '../components/InspectorPanel/portCaptures';

vi.mock('./tabPersistence', () => ({
  readSnapshot: vi.fn(async () => null),
  writeSnapshot: vi.fn(async () => {}),
}));

const store = () => useTabStore.getState();
const tab = () => useTabStore.getState().getActiveTab();
const entered = () => tab().subgraphStack.map((frame) => frame.instanceId);

function def(name: string): NodeDefinition {
  return {
    node_name: name,
    category: 'x',
    description: '',
    inputs: [{ name: 'tensor', data_type: 'TENSOR', description: '', optional: false }],
    outputs: [{ name: 'tensor', data_type: 'TENSOR', description: '', optional: false }],
    params: [],
  };
}

const inner: SubgraphDefinition = {
  id: 'inner',
  name: 'Inner',
  description: '',
  nodes: [{ id: 'mul2', type: 'Mul', position: { x: 0, y: 0 }, data: { params: {} } }],
  edges: [],
  interface: { inputs: [], outputs: [], triggerTargets: [] },
};

const outer: SubgraphDefinition = {
  id: 'outer',
  name: 'Outer',
  description: '',
  nodes: [
    { id: 'mul', type: 'Mul', position: { x: 0, y: 0 }, data: { params: {} } },
    { id: 'nest', type: 'subgraph:inner', position: { x: 200, y: 0 }, data: { params: {} } },
  ],
  edges: [],
  interface: { inputs: [], outputs: [], triggerTargets: [] },
};

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
  store().addTab('test');
  useNodeDefStore.setState({
    definitions: [def('Mul')],
    presets: [],
    categorized: {},
    loading: false,
    error: null,
  } as never);
  store().setSubgraphs([outer, inner]);
  // Two copies of one block: the same inner ids live in both.
  store().setNodes([
    buildInstanceNode(outer, { x: 0, y: 0 }, 'blk'),
    buildInstanceNode(outer, { x: 0, y: 300 }, 'blk2'),
  ]);
});

describe('runNodePrefix', () => {
  it('is empty at the top level', () => {
    expect(runNodePrefix(undefined)).toBe('');
    expect(runNodePrefix([])).toBe('');
    expect(runNodePrefix(tab().subgraphStack)).toBe('');
  });

  it('names the entered instances, outermost first, with a trailing separator', () => {
    expect(runNodePrefix([{ instanceId: 'blk' }])).toBe('blk/');
    expect(runNodePrefix([{ instanceId: 'blk' }, { instanceId: 'nest' }])).toBe('blk/nest/');
  });
});

describe('enterSubgraph records the instance it went through', () => {
  it('one level in', () => {
    expect(store().enterSubgraph('blk')).toBe(true);
    expect(tab().subgraphStack[0]).toMatchObject({ subgraphId: 'outer', instanceId: 'blk' });
    expect(runNodePrefix(tab().subgraphStack)).toBe('blk/');
  });

  it('a block inside the block', () => {
    store().enterSubgraph('blk');
    expect(store().enterSubgraph('nest')).toBe(true);
    expect(entered()).toEqual(['blk', 'nest']);
    expect(runNodePrefix(tab().subgraphStack)).toBe('blk/nest/');

    store().exitSubgraph();
    expect(entered()).toEqual(['blk']);
  });

  it('the other copy of the same block', () => {
    store().enterSubgraph('blk');
    store().exitSubgraph();
    store().enterSubgraph('blk2');
    expect(tab().subgraphStack[0]).toMatchObject({ subgraphId: 'outer', instanceId: 'blk2' });
    expect(runNodePrefix(tab().subgraphStack)).toBe('blk2/');
  });

  it('keeps it through a rename of the open block', () => {
    store().enterSubgraph('blk');
    store().enterSubgraph('nest');
    store().renameSubgraph('outer', 'Renamed');
    expect(entered()).toEqual(['blk', 'nest']);
  });
});
