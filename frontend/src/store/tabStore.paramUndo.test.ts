/**
 * A param edit is its own undo step (wave 2.8.9, T5).
 *
 * `updateNodeParams` used to push no snapshot, so Ctrl+Z after typing a value
 * restored the frame of the structural action BEFORE it: connect an edge,
 * change a param, press Ctrl+Z, and both the param AND the edge were gone.
 * The rule these tests pin lives in `paramEditUndo.ts`: a run of edits to one
 * node, with nothing else undoable in between and no pause longer than
 * `PARAM_EDIT_IDLE_MS`, is one step.
 *
 * `Date.now` is faked so the idle window is driven by the tests, not by how
 * fast the machine runs them.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import type { Edge } from '@xyflow/react';
import { useTabStore } from './tabStore';
import { PARAM_EDIT_IDLE_MS, resetParamEditMark } from './paramEditUndo';
import type { NodeDefinition } from '../types';

const store = () => useTabStore.getState();
const activeTab = () => useTabStore.getState().getActiveTab();

let clock = 0;

function makeDef(overrides: Partial<NodeDefinition> = {}): NodeDefinition {
  return {
    node_name: 'Dense',
    category: 'Layer',
    description: '',
    inputs: [{ name: 'in', data_type: 'TENSOR', description: '', optional: false }],
    outputs: [{ name: 'out', data_type: 'TENSOR', description: '', optional: false }],
    params: [
      { name: 'p1', param_type: 'int', default: 5, description: '', options: [], min_value: null, max_value: null },
      { name: 'p2', param_type: 'string', default: 'x', description: '', options: [], min_value: null, max_value: null },
    ],
    ...overrides,
  };
}

/** Patch the active tab directly, the way a loaded document would arrive. */
function patchActiveTab(patch: Record<string, unknown>) {
  useTabStore.setState((s) => ({
    tabs: s.tabs.map((t) => (t.id === s.activeTabId ? { ...t, ...patch } : t)),
  }));
}

/** Two unconnected nodes and an empty history, so every count starts at 0. */
function seedTwoNodes(): { a: string; b: string } {
  store().addNode(makeDef({ node_name: 'Source' }), { x: 0, y: 0 });
  store().addNode(makeDef({ node_name: 'Sink' }), { x: 200, y: 0 });
  const [a, b] = activeTab().nodes;
  patchActiveTab({ undoStack: [], redoStack: [], dirtyNodeIds: new Set<string>() });
  return { a: a.id, b: b.id };
}

const paramOf = (nodeId: string, name: string) =>
  activeTab().nodes.find((n) => n.id === nodeId)!.data.params[name];
const edgeIds = () => activeTab().edges.map((e) => e.id).sort();

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  store().addTab('test');
  // The mark is module state: it outlives the store reset above.
  resetParamEditMark();
  clock = 1_000_000;
  vi.spyOn(Date, 'now').mockImplementation(() => clock);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('a param edit is one undo step', () => {
  it('one undo after an edge and a param edit reverts only the param', () => {
    const { a, b } = seedTwoNodes();
    store().onConnect({ source: a, target: b, sourceHandle: 'out', targetHandle: 'in' });
    store().updateNodeParams(b, { p1: 42 });

    store().undo();
    expect(paramOf(b, 'p1')).toBe(5);
    expect(activeTab().edges).toHaveLength(1);

    store().undo();
    expect(activeTab().edges).toHaveLength(0);

    // And both come back, one redo each.
    store().redo();
    expect(activeTab().edges).toHaveLength(1);
    expect(paramOf(b, 'p1')).toBe(5);
    store().redo();
    expect(paramOf(b, 'p1')).toBe(42);
  });

  it('a run of edits to one node inside the idle window is one step', () => {
    // Typing `123` into a field: three writes, one thing to undo.
    const { a } = seedTwoNodes();
    store().updateNodeParams(a, { p1: 1 });
    clock += 300;
    store().updateNodeParams(a, { p1: 12 });
    clock += 300;
    store().updateNodeParams(a, { p2: 'y' });
    expect(activeTab().undoStack).toHaveLength(1);

    store().undo();
    expect(paramOf(a, 'p1')).toBe(5);
    expect(paramOf(a, 'p2')).toBe('x');
  });

  it('the window slides: a steady run longer than the window stays one step', () => {
    const { a } = seedTwoNodes();
    for (let value = 1; value <= 5; value += 1) {
      store().updateNodeParams(a, { p1: value });
      clock += PARAM_EDIT_IDLE_MS;
    }
    expect(activeTab().undoStack).toHaveLength(1);
  });

  it('a pause longer than the window opens a new step', () => {
    const { a } = seedTwoNodes();
    store().updateNodeParams(a, { p1: 1 });
    clock += PARAM_EDIT_IDLE_MS + 1;
    store().updateNodeParams(a, { p1: 2 });
    expect(activeTab().undoStack).toHaveLength(2);

    store().undo();
    expect(paramOf(a, 'p1')).toBe(1);
    store().undo();
    expect(paramOf(a, 'p1')).toBe(5);
  });

  it('an edit to another node opens a new step', () => {
    const { a, b } = seedTwoNodes();
    store().updateNodeParams(a, { p1: 1 });
    store().updateNodeParams(b, { p1: 2 });
    store().updateNodeParams(a, { p1: 3 });
    expect(activeTab().undoStack).toHaveLength(3);

    store().undo();
    expect(paramOf(a, 'p1')).toBe(1);
    expect(paramOf(b, 'p1')).toBe(2);
  });

  it('any other undoable action in between ends the run', () => {
    const { a, b } = seedTwoNodes();
    store().updateNodeParams(b, { p1: 1 });
    store().onConnect({ source: a, target: b, sourceHandle: 'out', targetHandle: 'in' });
    store().updateNodeParams(b, { p1: 2 });
    expect(activeTab().undoStack).toHaveLength(3);

    store().undo();
    expect(paramOf(b, 'p1')).toBe(1);
    expect(activeTab().edges).toHaveLength(1);
  });

  it('after an undo, the next edit to the same node is a new step and clears redo', () => {
    const { a } = seedTwoNodes();
    store().updateNodeParams(a, { p1: 1 });
    store().undo();
    expect(paramOf(a, 'p1')).toBe(5);
    expect(activeTab().redoStack).toHaveLength(1);

    store().updateNodeParams(a, { p1: 7 });
    expect(activeTab().undoStack).toHaveLength(1);
    expect(activeTab().redoStack).toHaveLength(0);

    store().undo();
    expect(paramOf(a, 'p1')).toBe(5);
  });

  it('an undo inside a run ends it, so a redo cannot write over the next edit', () => {
    // Edit, connect, undo: the stack's top is the run's own frame again, with
    // the connect waiting in redo. The next edit must still push, and so clear
    // redo, or a redo would put p1 = 1 back over it.
    const { a, b } = seedTwoNodes();
    store().updateNodeParams(a, { p1: 1 });
    clock += 300;
    store().onConnect({ source: a, target: b, sourceHandle: 'out', targetHandle: 'in' });
    clock += 300;
    store().undo();
    clock += 300;
    store().updateNodeParams(a, { p1: 2 });
    expect(activeTab().redoStack).toHaveLength(0);
    expect(activeTab().undoStack).toHaveLength(2);

    store().redo();
    expect(paramOf(a, 'p1')).toBe(2);
    store().undo();
    expect(paramOf(a, 'p1')).toBe(1);
  });

  it('an edit that changes nothing writes nothing and pushes nothing', () => {
    // Re-picking the option a select already shows must not leave a step
    // that undoes to the very same graph.
    const { a } = seedTwoNodes();
    const nodesBefore = activeTab().nodes;
    store().updateNodeParams(a, { p1: 5, p2: 'x' });
    expect(activeTab().undoStack).toHaveLength(0);
    expect(activeTab().nodes).toBe(nodesBefore);
    expect(activeTab().dirtyNodeIds.size).toBe(0);
  });

  it('a run stays one step across an edit that changed nothing', () => {
    const { a, b } = seedTwoNodes();
    store().updateNodeParams(a, { p1: 1 });
    store().updateNodeParams(b, { p1: 5 });
    store().updateNodeParams(a, { p1: 2 });
    expect(activeTab().undoStack).toHaveLength(1);
  });

  it('an edit that changes nothing keeps a run alive', () => {
    // Typing "-0.5": "-0" commits -0 and "-0." commits the same -0 again.
    // Only the pause before the 5 should count, not the one since "-0".
    const { a } = seedTwoNodes();
    store().updateNodeParams(a, { p1: -0 });
    clock += 700;
    store().updateNodeParams(a, { p1: -0 });
    clock += 700;
    store().updateNodeParams(a, { p1: -0.5 });
    expect(activeTab().undoStack).toHaveLength(1);

    store().undo();
    expect(paramOf(a, 'p1')).toBe(5);
  });

  it('an edit that changes nothing does not revive a run that has ended', () => {
    const { a } = seedTwoNodes();
    store().updateNodeParams(a, { p1: 1 });
    clock += PARAM_EDIT_IDLE_MS + 1;
    store().updateNodeParams(a, { p1: 1 });
    clock += 100;
    store().updateNodeParams(a, { p1: 2 });
    expect(activeTab().undoStack).toHaveLength(2);
  });

  it('a run does not continue in another tab, nor after coming back', () => {
    const { a } = seedTwoNodes();
    const first = useTabStore.getState().activeTabId;
    store().updateNodeParams(a, { p1: 1 });

    store().addTab('other');
    const { b } = seedTwoNodes();
    store().updateNodeParams(b, { p1: 9 });
    expect(activeTab().undoStack).toHaveLength(1);

    store().setActiveTab(first);
    store().updateNodeParams(a, { p1: 2 });
    expect(activeTab().undoStack).toHaveLength(2);
    store().undo();
    expect(paramOf(a, 'p1')).toBe(1);
  });

  it('does not throw with no tab open', () => {
    useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
    expect(() => store().updateNodeParams('gone', { p1: 1 })).not.toThrow();
    expect(useTabStore.getState().tabs).toEqual([]);
  });

  it('an edit to a node that is not on the canvas pushes nothing', () => {
    seedTwoNodes();
    store().updateNodeParams('gone', { p1: 1 });
    expect(activeTab().undoStack).toHaveLength(0);
  });
});

describe('a param edit that drops edges', () => {
  const splitDef = makeDef({
    node_name: 'Split',
    inputs: [],
    outputs: [
      { name: 'chunk_0', data_type: 'TENSOR', description: '', optional: false },
      { name: 'chunk_1', data_type: 'TENSOR', description: '', optional: false },
    ],
    params: [
      { name: 'chunks', param_type: 'int', default: 2, description: '', options: [], min_value: 1, max_value: 32 },
    ],
  });

  /** A Split cut into three, with one edge on the chunk that lowering it removes. */
  function seedSplit(): { split: string } {
    store().addNode(splitDef, { x: 0, y: 0 });
    store().addNode(makeDef({ node_name: 'Print' }), { x: 200, y: 0 });
    const [split, sink] = activeTab().nodes;
    const edges: Edge[] = [
      { id: 'keep', source: split.id, target: sink.id, sourceHandle: 'chunk_0', targetHandle: 'in' },
      { id: 'drop', source: split.id, target: sink.id, sourceHandle: 'chunk_2', targetHandle: 'in' },
    ];
    patchActiveTab({
      nodes: activeTab().nodes.map((n) =>
        n.id === split.id ? { ...n, data: { ...n.data, params: { chunks: 3 } } } : n,
      ),
      edges,
      undoStack: [],
      redoStack: [],
    });
    return { split: split.id };
  }

  it('pushes exactly one snapshot, and one undo restores the edges and the param', () => {
    const { split } = seedSplit();
    store().updateNodeParams(split, { chunks: 2 });
    expect(edgeIds()).toEqual(['keep']);
    expect(activeTab().undoStack).toHaveLength(1);

    store().undo();
    expect(edgeIds()).toEqual(['drop', 'keep']);
    expect(paramOf(split, 'chunks')).toBe(3);
  });

  it('opens its own step even in the middle of a run on the same node', () => {
    // The deleted edges must come back on their own, not together with
    // whatever the typing before them changed.
    const { split } = seedSplit();
    store().updateNodeParams(split, { chunks: 4 });
    clock += 100;
    store().updateNodeParams(split, { chunks: 2 });
    expect(activeTab().undoStack).toHaveLength(2);

    store().undo();
    expect(edgeIds()).toEqual(['drop', 'keep']);
    expect(paramOf(split, 'chunks')).toBe(4);
    store().undo();
    expect(paramOf(split, 'chunks')).toBe(3);
  });
});

describe('updateNodeLayers', () => {
  it('an Apply is one step', () => {
    const { a } = seedTwoNodes();
    store().updateNodeLayers(a, '[{"type":"Linear"}]');
    expect(activeTab().undoStack).toHaveLength(1);
    expect(paramOf(a, 'layers')).toBe('[{"type":"Linear"}]');

    store().undo();
    expect(paramOf(a, 'layers')).toBeUndefined();
  });

  it('an Apply that changed nothing is no step', () => {
    const { a } = seedTwoNodes();
    store().updateNodeLayers(a, '[]');
    store().updateNodeLayers(a, '[]');
    expect(activeTab().undoStack).toHaveLength(1);
  });

  it('ends a run of param edits on the same node', () => {
    const { a } = seedTwoNodes();
    store().updateNodeParams(a, { p1: 1 });
    store().updateNodeLayers(a, '[]');
    store().updateNodeParams(a, { p1: 2 });
    expect(activeTab().undoStack).toHaveLength(3);
  });
});

describe('undo and redo with no tab open', () => {
  // Ctrl+Z on the welcome screen reaches the store with `tabs: []`.
  it('do nothing instead of throwing', () => {
    useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
    expect(() => store().undo()).not.toThrow();
    expect(() => store().redo()).not.toThrow();
    expect(useTabStore.getState().tabs).toEqual([]);
  });

  it('do nothing when the active id names no tab', () => {
    const { a } = seedTwoNodes();
    store().updateNodeParams(a, { p1: 1 });
    const tabsBefore = useTabStore.getState().tabs;
    useTabStore.setState({ activeTabId: 'gone' });
    expect(() => store().undo()).not.toThrow();
    expect(() => store().redo()).not.toThrow();
    expect(useTabStore.getState().tabs).toBe(tabsBefore);
  });
});
