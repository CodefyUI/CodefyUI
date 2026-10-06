/**
 * Clear Canvas is one undo step for the whole tab (#625).
 *
 * `clear()` drops the file binding, the description, the device and the
 * read-only flag along with the graph. Its undo frame used to hold only the
 * graph, so Ctrl+Z brought the graph back unbound: Save turned into Save As,
 * and writing it over its file dropped the description and the device.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import type { Edge, Node } from '@xyflow/react';
import { flushSubgraphEditing, useTabStore, type TabState } from './tabStore';
import { useNodeDefStore } from './nodeDefStore';
import type { NodeData, NodeDefinition } from '../types';
import { subgraphIdOf } from '../utils/subgraph';

const store = () => useTabStore.getState();
const activeTab = () => store().getActiveTab();
const nodeIds = () => activeTab().nodes.map((n) => n.id);

function flowNode(id: string): Node<NodeData> {
  return { id, type: 'baseNode', position: { x: 0, y: 0 }, data: { label: id, type: 'Add', params: {} } };
}

function patchActiveTab(patch: Partial<TabState>): void {
  useTabStore.setState((s) => ({
    tabs: s.tabs.map((t) => (t.id === s.activeTabId ? { ...t, ...patch } : t)),
  }));
}

/** What a saved graph's tab holds besides the graph. */
const LAB1 = {
  currentGraphFile: 'lab1',
  currentGraphName: 'Lab 1',
  description: 'Two dense layers',
  graphDevice: 'cuda:0',
  readOnly: false,
};

const CLEARED = {
  currentGraphFile: null,
  currentGraphName: null,
  description: '',
  graphDevice: null,
  readOnly: false,
};

const documentOf = (tab: TabState) => ({
  currentGraphFile: tab.currentGraphFile,
  currentGraphName: tab.currentGraphName,
  description: tab.description,
  graphDevice: tab.graphDevice,
  readOnly: tab.readOnly,
});

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  store().addTab('lab1');
  patchActiveTab({ ...LAB1, nodes: [flowNode('n1')] });
});

describe('Clear Canvas, then undo', () => {
  it('brings back the binding, the description and the device with the graph', () => {
    store().clear();
    expect(documentOf(activeTab())).toEqual(CLEARED);
    store().undo();
    expect(nodeIds()).toEqual(['n1']);
    expect(documentOf(activeTab())).toEqual(LAB1);
  });

  it('brings a read-only graph back read-only', () => {
    patchActiveTab({ readOnly: true });
    store().clear();
    expect(activeTab().readOnly).toBe(false);
    store().undo();
    expect(activeTab().readOnly).toBe(true);
  });

  it('can be redone, which unbinds the tab again, and undone again', () => {
    store().clear();
    store().undo();
    store().redo();
    expect(nodeIds()).toEqual([]);
    expect(documentOf(activeTab())).toEqual(CLEARED);
    store().undo();
    expect(nodeIds()).toEqual(['n1']);
    expect(documentOf(activeTab())).toEqual(LAB1);
  });

  it('leaves the binding alone when the step undone is not a Clear', () => {
    // An edit, then a Save As, which is no undo step: undoing the edit keeps
    // the tab bound to the file it was saved as.
    store().pushUndoSnapshot();
    store().setNodes([flowNode('n1'), flowNode('n2')]);
    store().setCurrentGraphFile('lab2', 'Lab 2');
    store().undo();
    expect(nodeIds()).toEqual(['n1']);
    expect(activeTab().currentGraphFile).toBe('lab2');
    expect(activeTab().currentGraphName).toBe('Lab 2');
  });
});

// The binding a Clear frame holds goes stale like a tab's own when its file
// is renamed, deleted or written over, and an in-place Save never asks.
describe('a binding the history holds follows its file', () => {
  it('moves with a rename', () => {
    store().clear();
    store().rebindGraphFile('lab1', { file: 'lab1b', name: 'Lab 1b' });
    store().undo();
    expect(activeTab().currentGraphFile).toBe('lab1b');
    expect(activeTab().currentGraphName).toBe('Lab 1b');
  });

  it('is dropped with a delete, so the next Save asks for a name', () => {
    store().clear();
    store().rebindGraphFile('lab1', null);
    store().undo();
    expect(activeTab().currentGraphFile).toBeNull();
    expect(activeTab().currentGraphName).toBeNull();
    // The rest of the tab still comes back.
    expect(activeTab().description).toBe(LAB1.description);
    expect(activeTab().graphDevice).toBe(LAB1.graphDevice);
  });

  it('is dropped when another tab saves over the file', () => {
    const cleared = store().activeTabId;
    store().clear();
    store().addTab('other');
    const other = store().activeTabId;
    // What `saveActiveGraph` does after a Save As over lab1 from that tab.
    store().setCurrentGraphFile('lab1', 'Other');
    store().rebindGraphFile('lab1', null, other);
    expect(activeTab().currentGraphFile).toBe('lab1');
    store().setActiveTab(cleared);
    store().undo();
    expect(activeTab().currentGraphFile).toBeNull();
  });

  it('stays in the history of the tab that saved over the file', () => {
    store().clear();
    // Save As lab1 from this same tab: its own history is its graph's.
    store().setCurrentGraphFile('lab1', 'Lab 1');
    store().rebindGraphFile('lab1', null, store().activeTabId);
    store().undo();
    expect(activeTab().currentGraphFile).toBe('lab1');
  });

  it('moves in the history an open block keeps for the level above', () => {
    store().clear();
    const history = activeTab().undoStack;
    // Only the stacks matter here, so the frame is cast rather than built.
    const frame = {
      subgraphId: 'blk', nodes: [], edges: [], presets: [], undoStack: history, redoStack: [],
      selectedNodeId: null, subgraphs: [], segmentGroups: [], activeSegment: null,
    } as unknown as TabState['subgraphStack'][number];
    patchActiveTab({ undoStack: [], subgraphStack: [frame] });
    store().rebindGraphFile('lab1', { file: 'lab1b', name: 'Lab 1b' });
    const kept = activeTab().subgraphStack[0].undoStack;
    expect(kept[kept.length - 1]).toMatchObject({ currentGraphFile: 'lab1b', currentGraphName: 'Lab 1b' });
  });
});

// File -> Clear Canvas has no block gating, so it can be pressed inside one.
// It used to keep the block's own history as the tab's and drop the top
// level's: the second Ctrl+Z put the block's insides in place of the graph.
describe('Clear Canvas from inside an open block', () => {
  function def(name: string): NodeDefinition {
    return {
      node_name: name,
      category: 'x',
      description: '',
      inputs: [{ name: 'in', data_type: 'TENSOR', description: '', optional: false }],
      outputs: [{ name: 'out', data_type: 'TENSOR', description: '', optional: false }],
      params: [
        { name: 'scale', param_type: 'float', default: 1, description: '', options: [], min_value: null, max_value: null },
      ],
    };
  }

  function chainNode(id: string, type: string, x: number): Node<NodeData> {
    return {
      id,
      type: 'baseNode',
      position: { x, y: 0 },
      data: { label: id, type, params: { scale: 1 }, definition: def(type), executionStatus: 'idle' },
    };
  }

  const dataEdge = (id: string, source: string, target: string): Edge =>
    ({ id, source, target, sourceHandle: 'out', targetHandle: 'in' });

  /** start -> a -> [b -> c] -> sink, b and c collapsed into a block; the instance id. */
  function seedBlock(): string {
    useNodeDefStore.setState({ definitions: [def('A'), def('B'), def('C'), def('S')], presets: [] } as never);
    store().setNodes([
      { ...chainNode('start', 'Start', 0), type: 'start' },
      chainNode('a', 'A', 100),
      chainNode('b', 'B', 200),
      chainNode('c', 'C', 300),
      chainNode('sink', 'S', 400),
    ]);
    store().setEdges([
      {
        id: 't', source: 'start', target: 'a', sourceHandle: 'trigger',
        targetHandle: '__trigger', type: 'triggerEdge', data: { type: 'trigger' },
      },
      dataEdge('e1', 'a', 'b'),
      dataEdge('e2', 'b', 'c'),
      dataEdge('e3', 'c', 'sink'),
    ]);
    store().setNodes(activeTab().nodes.map((n) => ({ ...n, selected: n.id === 'b' || n.id === 'c' })));
    expect(store().collapseSelectionToSubgraph('Block').ok).toBe(true);
    return activeTab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
  }

  const graphOf = (tab: TabState) => ({ nodes: tab.nodes, edges: tab.edges, subgraphs: tab.subgraphs });

  it('keeps the top level history, so each undo puts back a whole graph, and redo returns', () => {
    const instanceId = seedBlock();
    const atEntry = graphOf(activeTab());
    store().enterSubgraph(instanceId);
    store().updateNodeParams('b', { scale: 7 });
    const beforeClear = graphOf(flushSubgraphEditing(activeTab()));

    store().clear();
    expect(activeTab().subgraphStack).toEqual([]);
    expect(activeTab().nodes).toEqual([]);

    store().undo();
    expect(graphOf(activeTab())).toEqual(beforeClear);
    // Then the visit, as leaving the block records it -- not the block's
    // inner canvas in place of the whole graph.
    store().undo();
    expect(graphOf(activeTab())).toEqual(atEntry);
    // Then what came before the block was entered: the collapse.
    store().undo();
    expect(activeTab().subgraphs).toEqual([]);
    expect(activeTab().nodes.map((n) => n.id).sort()).toEqual(['a', 'b', 'c', 'sink', 'start']);
    expect(activeTab().subgraphStack).toEqual([]);

    store().redo();
    expect(graphOf(activeTab())).toEqual(atEntry);
    store().redo();
    expect(graphOf(activeTab())).toEqual(beforeClear);
    store().redo();
    expect(activeTab().nodes).toEqual([]);
    expect(activeTab().subgraphs).toEqual([]);
  });

  it('adds only the Clear step when nothing in the block changed', () => {
    const instanceId = seedBlock();
    const depth = activeTab().undoStack.length;
    store().enterSubgraph(instanceId);
    store().clear();
    expect(activeTab().undoStack).toHaveLength(depth + 1);
    store().undo();
    store().undo();
    expect(activeTab().subgraphs).toEqual([]);
  });
});
