/**
 * One deletion is one undo step.
 *
 * The Delete key deletes through React Flow's `deleteElements`, which reports
 * one deletion in two calls: the wires to `onEdgesChange`, then the nodes to
 * `onNodesChange`. Each pushed an undo snapshot, so the first Ctrl+Z put the
 * node back with none of its wires and a second one was needed for those. The
 * rule these tests pin lives in `removalUndo.ts`.
 *
 * The Delete-key tests run a real React Flow, wired to the store the way
 * FlowCanvas wires it, so they see the calls in the order React Flow makes
 * them rather than an order a test assumes.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { act, fireEvent, render } from '@testing-library/react';
import { ReactFlow, ReactFlowProvider, type Edge, type Node } from '@xyflow/react';

import { useTabStore } from './tabStore';
import { useNodeDefStore } from './nodeDefStore';
import { useDeleteKey } from '../hooks/useDeleteKey';
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
    params: [],
  };
}

function node(id: string, type: string, x = 0): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position: { x, y: 0 },
    data: { label: id, type, params: {}, definition: def(type), executionStatus: 'idle' },
  };
}

function dataEdge(id: string, source: string, target: string): Edge {
  return { id, source, target, sourceHandle: 'out', targetHandle: 'in' };
}

function triggerEdge(id: string, source: string, target: string): Edge {
  return {
    id, source, target, sourceHandle: 'trigger', targetHandle: '__trigger',
    type: 'triggerEdge', data: { type: 'trigger' },
  };
}

/** Start -> TensorCreate -> Print, the graph the bug was found on. */
function seedRepro() {
  store().setNodes([
    { ...node('start', 'Start'), type: 'start' },
    node('t', 'TensorCreate', 200),
    node('p', 'Print', 400),
  ]);
  store().setEdges([triggerEdge('go', 'start', 't'), dataEdge('wire', 't', 'p')]);
}

/** start -> a -> b -> c -> d. */
function seedChain() {
  store().setNodes([
    { ...node('start', 'Start'), type: 'start' },
    node('a', 'A', 100),
    node('b', 'B', 200),
    node('c', 'C', 300),
    node('d', 'D', 400),
  ]);
  store().setEdges([
    triggerEdge('go', 'start', 'a'),
    dataEdge('ab', 'a', 'b'),
    dataEdge('bc', 'b', 'c'),
    dataEdge('cd', 'c', 'd'),
  ]);
}

/** Collapse these nodes into one block and return its card's id. */
function collapse(...memberIds: string[]): string {
  store().setNodes(tab().nodes.map((n) => ({ ...n, selected: memberIds.includes(n.id) })));
  expect(store().collapseSelectionToSubgraph('Block').ok).toBe(true);
  return tab().nodes.find((n) => subgraphIdOf(n.data.type) !== null)!.id;
}

/**
 * The canvas as far as Delete goes: the active tab drawn by a real React Flow
 * with its own Delete binding off, its changes going into the store, and
 * `useDeleteKey` deleting through `deleteElements` (FlowCanvas).
 */
function Canvas() {
  const active = useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId)!);
  const onNodesChange = useTabStore((s) => s.onNodesChange);
  const onEdgesChange = useTabStore((s) => s.onEdgesChange);
  useDeleteKey();
  return (
    <div style={{ width: 800, height: 600 }}>
      <ReactFlow
        nodes={active.nodes}
        edges={active.edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        deleteKeyCode={null}
        // FlowCanvas's answer with no modal open.
        onBeforeDelete={async () => true}
      />
    </div>
  );
}

function mount() {
  render(
    <ReactFlowProvider>
      <Canvas />
    </ReactFlowProvider>,
  );
}

/** Select exactly these nodes and wires, as clicking them does. */
function select(...ids: string[]) {
  act(() => {
    store().setNodes(tab().nodes.map((n) => ({ ...n, selected: ids.includes(n.id) })));
    store().setEdges(tab().edges.map((e) => ({ ...e, selected: ids.includes(e.id) })));
  });
}

/** Delete pressed and let go, with the deletion run out (it awaits `onBeforeDelete`). */
async function pressDelete() {
  fireEvent.keyDown(document.body, { key: 'Delete', code: 'Delete' });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
  fireEvent.keyUp(document.body, { key: 'Delete', code: 'Delete' });
}

const undo = () => act(() => store().undo());
const redo = () => act(() => store().redo());

const ids = (items: { id: string }[]) => items.map((item) => item.id).sort();

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
  store().addTab('test');
  useNodeDefStore.setState({
    definitions: ['Start', 'TensorCreate', 'Print', 'A', 'B', 'C', 'D'].map(def),
    presets: [],
    categorized: {},
    loading: false,
    error: null,
  } as never);
});

describe('the Delete key is one undo step', () => {
  it('one undo brings a deleted node back with its wires', async () => {
    seedRepro();
    mount();
    select('t');
    await pressDelete();
    expect(ids(tab().nodes)).toEqual(['p', 'start']);
    expect(tab().edges).toEqual([]);

    undo();
    expect(ids(tab().nodes)).toEqual(['p', 'start', 't']);
    expect(ids(tab().edges)).toEqual(['go', 'wire']);
    // That was the whole deletion: nothing is left to undo.
    expect(tab().undoStack).toHaveLength(0);
  });

  it('one redo takes the node and its wires away again', async () => {
    seedRepro();
    mount();
    select('t');
    await pressDelete();
    undo();

    redo();
    expect(ids(tab().nodes)).toEqual(['p', 'start']);
    expect(tab().edges).toEqual([]);
    expect(tab().redoStack).toHaveLength(0);
    expect(tab().undoStack).toHaveLength(1);
  });

  it('several nodes and a wire deleted together come back together', async () => {
    seedChain();
    mount();
    select('a', 'd', 'bc');
    await pressDelete();
    expect(ids(tab().nodes)).toEqual(['b', 'c', 'start']);
    expect(tab().edges).toEqual([]);

    undo();
    expect(ids(tab().nodes)).toEqual(['a', 'b', 'c', 'd', 'start']);
    expect(ids(tab().edges)).toEqual(['ab', 'bc', 'cd', 'go']);
    expect(tab().undoStack).toHaveLength(0);
  });

  it('two presses of Delete are still two steps', async () => {
    seedRepro();
    mount();
    select('wire');
    await pressDelete();
    select('t');
    await pressDelete();
    expect(ids(tab().nodes)).toEqual(['p', 'start']);
    expect(tab().edges).toEqual([]);

    // The second press comes back whole, and without the first.
    undo();
    expect(ids(tab().nodes)).toEqual(['p', 'start', 't']);
    expect(ids(tab().edges)).toEqual(['go']);
    undo();
    expect(ids(tab().edges)).toEqual(['go', 'wire']);
    expect(tab().undoStack).toHaveLength(0);
  });

  it('a block card comes back with its wires and its block', async () => {
    seedChain();
    const card = collapse('b', 'c');
    const before = { nodes: ids(tab().nodes), edges: ids(tab().edges), steps: tab().undoStack.length };
    mount();
    select(card);
    await pressDelete();
    expect(ids(tab().nodes)).toEqual(['a', 'd', 'start']);
    expect(ids(tab().edges)).toEqual(['go']);

    undo();
    expect(ids(tab().nodes)).toEqual(before.nodes);
    expect(ids(tab().edges)).toEqual(before.edges);
    expect(tab().subgraphs).toHaveLength(1);
    expect(tab().undoStack).toHaveLength(before.steps);
  });

  it('inside a block, one undo brings a node back with its wires', async () => {
    seedChain();
    const card = collapse('b', 'c', 'd');
    expect(store().enterSubgraph(card)).toBe(true);
    const before = { nodes: ids(tab().nodes), edges: ids(tab().edges) };
    expect(before.nodes).toEqual(['b', 'c', 'd']);
    expect(before.edges).toHaveLength(2);
    mount();
    select('c');
    await pressDelete();
    expect(ids(tab().nodes)).toEqual(['b', 'd']);
    expect(tab().edges).toEqual([]);

    undo();
    expect(ids(tab().nodes)).toEqual(before.nodes);
    expect(ids(tab().edges)).toEqual(before.edges);
    // A block keeps a history of its own, and this was all of it.
    expect(tab().undoStack).toHaveLength(0);
  });
});

describe('the other ways a node is deleted', () => {
  it('the context menu Delete is one step', () => {
    seedRepro();
    store().deleteNode('t');
    store().undo();
    expect(ids(tab().nodes)).toEqual(['p', 'start', 't']);
    expect(ids(tab().edges)).toEqual(['go', 'wire']);
    expect(tab().undoStack).toHaveLength(0);
  });

  it('a deletion reported nodes first is one step too', () => {
    // React Flow reports the wires first today; the step does not depend on it.
    seedRepro();
    store().onNodesChange([{ id: 't', type: 'remove' }]);
    store().onEdgesChange([
      { id: 'go', type: 'remove' },
      { id: 'wire', type: 'remove' },
    ]);
    store().undo();
    expect(ids(tab().nodes)).toEqual(['p', 'start', 't']);
    expect(ids(tab().edges)).toEqual(['go', 'wire']);
    expect(tab().undoStack).toHaveLength(0);
  });
});
