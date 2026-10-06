/**
 * Stepping into and out of a block is not an edit (found during the browser
 * e2e of the fixes for #618-#625). A saved tab asked "cannot be undone" on
 * close after a mere look inside a block, or after a Ctrl+S while a block was
 * open: the step swaps the canvas, which moves `revision`, and `savedRevision`
 * stayed behind.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { Edge, Node } from '@xyflow/react';

import { tabHasUnsavedWork, useTabStore } from './tabStore';
import { useNodeDefStore } from './nodeDefStore';
import type { NodeData, NodeDefinition } from '../types';
import { subgraphIdOf } from '../utils/subgraph';

vi.mock('./tabPersistence', () => ({
  readSnapshot: vi.fn(async () => null),
  writeSnapshot: vi.fn(async () => {}),
}));

const store = () => useTabStore.getState();
const tab = () => useTabStore.getState().getActiveTab();

function def(name: string): NodeDefinition {
  return {
    node_name: name,
    category: 'x',
    description: '',
    inputs: [{ name: 'in', data_type: 'TENSOR', description: '', optional: false }],
    outputs: [{ name: 'out', data_type: 'TENSOR', description: '', optional: false }],
    params: [
      {
        name: 'scale', param_type: 'float', default: 1, description: '',
        options: [], min_value: null, max_value: null,
      },
    ],
  };
}

function node(id: string, type: string, x: number): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position: { x, y: 0 },
    data: { label: id, type, params: { scale: 1 }, definition: def(type), executionStatus: 'idle' },
  };
}

function dataEdge(id: string, source: string, target: string): Edge {
  return { id, source, target, sourceHandle: 'out', targetHandle: 'in' };
}

/** a -> [b -> c] with b and c collapsed into a block; returns the block's card id. */
function graphWithABlock(): string {
  store().setNodes([node('a', 'A', 0), node('b', 'B', 100), node('c', 'C', 200)]);
  store().setEdges([dataEdge('e1', 'a', 'b'), dataEdge('e2', 'b', 'c')]);
  store().setNodes(tab().nodes.map((n) => ({ ...n, selected: n.id !== 'a' })));
  expect(store().collapseSelectionToSubgraph('Block').ok).toBe(true);
  return tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
}

/** What a save does once the server has written the graph. */
function save() {
  const { id, revision } = tab();
  store().setTabGraphFile(id, 'f12test', 'f12test');
  store().markTabSaved(id, revision);
}

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
  store().addTab('test');
  useNodeDefStore.setState({
    definitions: [def('A'), def('B'), def('C')],
    presets: [],
    categorized: {},
    loading: false,
    error: null,
  } as never);
});

describe('stepping into and out of a block on a saved tab', () => {
  it.each([
    ['Back', () => store().exitSubgraph()],
    ['Main', () => store().exitAllSubgraphs()],
  ])('a look inside, left with %s, leaves the tab saved', (_how, leave) => {
    const card = graphWithABlock();
    save();
    expect(tabHasUnsavedWork(tab())).toBe(false);

    store().enterSubgraph(card);
    expect(tabHasUnsavedWork(tab())).toBe(false);
    leave();
    expect(tabHasUnsavedWork(tab())).toBe(false);
  });

  it('a save while the block is open, then leaving, leaves the tab saved', () => {
    const card = graphWithABlock();
    save();
    store().enterSubgraph(card);
    store().updateNodeParams('b', { scale: 5 });
    expect(tabHasUnsavedWork(tab())).toBe(true);

    save();
    store().exitSubgraph();

    expect(tabHasUnsavedWork(tab())).toBe(false);
  });

  it('an edit made inside the block still counts once the block is left', () => {
    const card = graphWithABlock();
    save();
    store().enterSubgraph(card);
    store().updateNodeParams('b', { scale: 5 });
    store().exitAllSubgraphs();

    expect(tabHasUnsavedWork(tab())).toBe(true);
  });

  it('a tab never saved still asks after a look inside a block', () => {
    const card = graphWithABlock();
    store().enterSubgraph(card);
    store().exitSubgraph();

    expect(tabHasUnsavedWork(tab())).toBe(true);
  });

  it('still moves the revision both ways, which plugins are told about', () => {
    // `onGraphChanged` and `workspace.onChanged` fire on a step into or out of
    // a block, and a revision read before it no longer matches (plugin docs).
    const card = graphWithABlock();
    save();
    const before = tab().revision;
    store().enterSubgraph(card);
    const inside = tab().revision;
    store().exitSubgraph();

    expect(inside).toBeGreaterThan(before);
    expect(tab().revision).toBeGreaterThan(inside);
  });
});
