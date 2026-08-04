/**
 * core#137 store behaviour: collapse/expand as ONE undo step, entering and
 * leaving a sub-canvas, and shared-definition propagation across save/load.
 *
 * The propagation test deliberately goes through `getSerializedGraph` and
 * back through `setSubgraphs`/`resolveSerializedNodes`: two instances built
 * from the same object in memory would "share" a definition by aliasing,
 * which proves nothing about a graph reopened from disk.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { Edge, Node } from '@xyflow/react';

import { useTabStore } from './tabStore';
import { useNodeDefStore } from './nodeDefStore';
import type { NodeData, NodeDefinition } from '../types';
import { resolveSerializedEdges, resolveSerializedNodes } from '../utils';
import { buildInstanceNode, subgraphIdOf } from '../utils/subgraph';

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
    inputs: [
      { name: 'in', data_type: 'TENSOR', description: '', optional: false },
    ],
    outputs: [
      { name: 'out', data_type: 'TENSOR', description: '', optional: false },
    ],
    params: [
      {
        name: 'scale', param_type: 'float', default: 1, description: '',
        options: [], min_value: null, max_value: null,
      },
    ],
  };
}

function node(id: string, type: string, x = 0, y = 0): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position: { x, y },
    data: {
      label: id,
      type,
      params: { scale: 1 },
      definition: def(type),
      executionStatus: 'idle',
    },
  };
}

function dataEdge(id: string, source: string, target: string): Edge {
  return { id, source, target, sourceHandle: 'out', targetHandle: 'in' };
}

/** start -> a -> b -> c -> sink (a is the triggered root). */
function seedChain() {
  const nodes: Node<NodeData>[] = [
    { ...node('start', 'Start', 0, 0), type: 'start' },
    node('a', 'A', 100, 0),
    node('b', 'B', 200, 40),
    node('c', 'C', 300, 20),
    node('sink', 'S', 400, 0),
  ];
  const edges: Edge[] = [
    {
      id: 't', source: 'start', target: 'a', sourceHandle: 'trigger',
      targetHandle: '__trigger', type: 'triggerEdge', data: { type: 'trigger' },
    },
    dataEdge('e1', 'a', 'b'),
    dataEdge('e2', 'b', 'c'),
    dataEdge('e3', 'c', 'sink'),
  ];
  store().setNodes(nodes);
  store().setEdges(edges);
}

function select(...ids: string[]) {
  store().setNodes(
    tab().nodes.map((n) => ({ ...n, selected: ids.includes(n.id) })),
  );
}

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
  store().addTab('test');
  useNodeDefStore.setState({
    definitions: [def('A'), def('B'), def('C'), def('S')],
    presets: [],
    categorized: {},
    presetCategorized: {},
    loading: false,
    error: null,
  } as never);
});

// ── Collapse / expand ───────────────────────────────────────────────────

describe('collapseSelectionToSubgraph', () => {
  it('replaces the selection with an instance and records the definition', () => {
    seedChain();
    select('b', 'c');
    const result = store().collapseSelectionToSubgraph('Block');
    expect(result.ok).toBe(true);
    const instances = tab().nodes.filter((n) => subgraphIdOf(n.data.type));
    expect(instances).toHaveLength(1);
    expect(
      tab().nodes
        .filter((n) => !subgraphIdOf(n.data.type))
        .map((n) => n.id)
        .sort(),
    ).toEqual(['a', 'sink', 'start']);
    expect(tab().subgraphs).toHaveLength(1);
    expect(tab().subgraphs[0].name).toBe('Block');
    expect(subgraphIdOf(instances[0].data.type)).toBe(tab().subgraphs[0].id);
    // The entry point is untouched -- `a` stays outside the block and keeps
    // its trigger -- and the block is wired in where `a` used to feed `b`.
    expect(
      tab().edges.some((e) => e.source === 'start' && e.target === 'a'),
    ).toBe(true);
    expect(
      tab().edges.some((e) => e.source === 'a' && e.target === instances[0].id),
    ).toBe(true);
    expect(
      tab().edges.some(
        (e) => e.source === instances[0].id && e.target === 'sink',
      ),
    ).toBe(true);
  });

  it('refuses without touching the graph, and pushes NO undo entry', () => {
    seedChain();
    const before = tab().nodes;
    const undoDepth = tab().undoStack.length;
    select('a', 'c'); // b sits between them
    const result = store().collapseSelectionToSubgraph();
    expect(result).toMatchObject({ ok: false, reason: 'not-convex' });
    // `select` replaced the array, so compare ids rather than identity.
    expect(tab().nodes.map((n) => n.id)).toEqual(before.map((n) => n.id));
    expect(tab().subgraphs).toEqual([]);
    expect(tab().undoStack.length).toBe(undoDepth);
  });
});

describe('undo across collapse and expand', () => {
  it('undoes a collapse in exactly ONE step', () => {
    seedChain();
    const beforeNodes = tab().nodes.map((n) => n.id).sort();
    const beforeEdges = tab().edges.map((e) => e.id).sort();
    const depth = tab().undoStack.length;

    select('b', 'c');
    expect(store().collapseSelectionToSubgraph('Block').ok).toBe(true);
    // Exactly one entry, not one per part of the change.
    expect(tab().undoStack.length).toBe(depth + 1);

    store().undo();

    expect(tab().nodes.map((n) => n.id).sort()).toEqual(beforeNodes);
    expect(tab().edges.map((e) => e.id).sort()).toEqual(beforeEdges);
    // The definition list is part of the snapshot: without it, undo would
    // leave the graph holding a block nobody defines.
    expect(tab().subgraphs).toEqual([]);
    expect(tab().undoStack.length).toBe(depth);
  });

  it('undoes an expand in exactly ONE step, definition included', () => {
    seedChain();
    select('b', 'c');
    store().collapseSelectionToSubgraph('Block');
    const instanceId = tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
    const collapsedNodes = tab().nodes.map((n) => n.id).sort();
    const depth = tab().undoStack.length;

    expect(store().expandSubgraphInstance(instanceId)).toBe(true);
    expect(tab().nodes.map((n) => n.id).sort()).toEqual(
      ['a', 'b', 'c', 'sink', 'start'],
    );
    expect(tab().subgraphs).toEqual([]);
    expect(tab().undoStack.length).toBe(depth + 1);

    store().undo();

    expect(tab().nodes.map((n) => n.id).sort()).toEqual(collapsedNodes);
    expect(tab().subgraphs).toHaveLength(1);
    expect(tab().subgraphs[0].name).toBe('Block');
  });

  it('redo puts the collapse back, definition and all', () => {
    seedChain();
    select('b', 'c');
    store().collapseSelectionToSubgraph('Block');
    store().undo();
    expect(tab().subgraphs).toEqual([]);
    store().redo();
    expect(tab().subgraphs).toHaveLength(1);
    expect(tab().nodes.some((n) => subgraphIdOf(n.data.type))).toBe(true);
  });
});

// ── Entering and leaving ────────────────────────────────────────────────

describe('enterSubgraph / exitSubgraph', () => {
  function collapseBC(): string {
    seedChain();
    select('b', 'c');
    store().collapseSelectionToSubgraph('Block');
    return tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
  }

  it('swaps the definition onto the canvas and back', () => {
    const instanceId = collapseBC();
    const outerIds = tab().nodes.map((n) => n.id).sort();

    expect(store().enterSubgraph(instanceId)).toBe(true);
    expect(tab().subgraphStack).toHaveLength(1);
    expect(tab().nodes.map((n) => n.id).sort()).toEqual(['b', 'c']);
    expect(tab().edges.map((e) => e.id)).toEqual(['e2']);

    store().exitSubgraph();
    expect(tab().subgraphStack).toEqual([]);
    expect(tab().nodes.map((n) => n.id).sort()).toEqual(outerIds);
  });

  it('keeps the outer undo history out of reach while inside', () => {
    const instanceId = collapseBC();
    const outerDepth = tab().undoStack.length;
    expect(outerDepth).toBeGreaterThan(0);

    store().enterSubgraph(instanceId);
    expect(tab().undoStack).toEqual([]);

    store().exitSubgraph();
    expect(tab().undoStack.length).toBe(outerDepth);
  });

  it('writes an edit inside the block back into the definition', () => {
    const instanceId = collapseBC();
    store().enterSubgraph(instanceId);
    store().updateNodeParams('b', { scale: 7 });
    store().exitSubgraph();

    const inner = tab().subgraphs[0].nodes.find((n) => n.id === 'b');
    expect(inner.data.params.scale).toBe(7);
  });

  it('drops a boundary port whose inner node was deleted inside', () => {
    const instanceId = collapseBC();
    store().enterSubgraph(instanceId);
    store().deleteNode('c'); // 'c' produced the block's only output
    store().exitSubgraph();

    expect(tab().subgraphs[0].interface.outputs).toEqual([]);
    // ... and the outer edge that named it is gone rather than dangling.
    expect(tab().edges.some((e) => e.source === instanceId)).toBe(false);
  });

  it('serializes the ROOT graph while a sub-canvas is open', () => {
    const instanceId = collapseBC();
    store().enterSubgraph(instanceId);
    const graph = store().getSerializedGraph();
    expect(graph.nodes.map((n: any) => n.id).sort()).toEqual(
      ['a', instanceId, 'sink', 'start'].sort(),
    );
    expect(graph.subgraphs).toHaveLength(1);
    // Still inside: serializing must not close the editor behind the user.
    expect(tab().subgraphStack).toHaveLength(1);
  });
});

// ── Shared definition, proven across a save/load round trip ─────────────

describe('shared definitions', () => {
  it('two instances follow one definition after a save/load round trip', () => {
    seedChain();
    select('b', 'c');
    store().collapseSelectionToSubgraph('Block');
    const firstId = tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;

    // A second, independent use of the same block.
    const second = {
      ...tab().nodes.find((n) => n.id === firstId)!,
      id: 'inst2',
      position: { x: 600, y: 300 },
      selected: false,
    };
    store().setNodes([...tab().nodes, second]);

    // ── Round trip: serialize, then rebuild from the serialized form only.
    const saved = JSON.parse(JSON.stringify(store().getSerializedGraph()));
    expect(saved.subgraphs).toHaveLength(1);
    expect(
      saved.nodes.filter((n: any) => subgraphIdOf(n.type)),
    ).toHaveLength(2);

    store().clear();
    const defs = useNodeDefStore.getState().definitions;
    const reloadedNodes = resolveSerializedNodes(
      saved.nodes, defs, [], saved.subgraphs,
    );
    store().setNodes(reloadedNodes);
    store().setEdges(resolveSerializedEdges(saved.edges, reloadedNodes));
    store().setSubgraphs(saved.subgraphs);

    const instanceIds = tab().nodes
      .filter((n) => subgraphIdOf(n.data.type))
      .map((n) => n.id);
    expect(instanceIds).toHaveLength(2);

    // ── Both instances must render the SHARED interface straight off the
    //    reload, before anything is edited. Without this the test would pass
    //    even if loading gave every instance an empty definition, because the
    //    edit below rebuilds them anyway.
    for (const id of instanceIds) {
      const instance = tab().nodes.find((n) => n.id === id)!;
      expect(instance.data.definition!.inputs.map((p) => p.name)).toEqual(['in']);
      expect(instance.data.definition!.outputs.map((p) => p.name)).toEqual(['out']);
      expect(instance.data.label).toBe('Block');
    }

    // ── Edit the definition through ONE instance.
    store().enterSubgraph(instanceIds[0]);
    store().updateNodeParams('b', { scale: 42 });
    store().deleteNode('c');
    store().exitSubgraph();

    // ── Both instances show the new boundary, and there is still ONE
    //    definition carrying the edit.
    expect(tab().subgraphs).toHaveLength(1);
    const definition = tab().subgraphs[0];
    expect(definition.nodes.map((n: any) => n.id)).toEqual(['b']);
    expect(definition.nodes[0].data.params.scale).toBe(42);

    for (const id of instanceIds) {
      const instance = tab().nodes.find((n) => n.id === id)!;
      expect(instance.data.definition!.outputs).toEqual([]);
      expect(instance.data.definition!.inputs.map((p) => p.name)).toEqual(['in']);
    }

    // And it survives another save.
    const resaved = store().getSerializedGraph();
    expect(resaved.subgraphs).toHaveLength(1);
    expect(resaved.subgraphs![0].nodes[0].data.params.scale).toBe(42);
  });

  it('a graph reloaded without its definitions still renders the instance', () => {
    seedChain();
    select('b', 'c');
    store().collapseSelectionToSubgraph('Block');
    const saved = JSON.parse(JSON.stringify(store().getSerializedGraph()));
    const reloaded = resolveSerializedNodes(saved.nodes, [], [], []);
    const instance = reloaded.find((n) => subgraphIdOf(n.data.type))!;
    expect(instance.type).toBe('subgraphNode');
    expect(instance.data.definition!.inputs).toEqual([]);
  });
});

// ── Clipboard ───────────────────────────────────────────────────────────

describe('copy/paste of an instance', () => {
  it('pasting in the same tab makes a second instance of ONE definition', () => {
    seedChain();
    select('b', 'c');
    store().collapseSelectionToSubgraph('Block');
    const instanceId = tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;

    select(instanceId);
    store().copySelectedNodes();
    store().pasteNodes();

    const instances = tab().nodes.filter((n) => subgraphIdOf(n.data.type));
    expect(instances).toHaveLength(2);
    expect(new Set(instances.map((n) => subgraphIdOf(n.data.type))).size).toBe(1);
    expect(tab().subgraphs).toHaveLength(1);
  });

  it('carries the definition into a tab that does not have it', () => {
    seedChain();
    select('b', 'c');
    store().collapseSelectionToSubgraph('Block');
    const instanceId = tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
    select(instanceId);
    store().copySelectedNodes();

    store().addTab('other');
    expect(tab().subgraphs).toEqual([]);
    store().pasteNodes();

    // Without the clipboard carrying it, this tab would hold an instance the
    // server would refuse with "Unknown subgraph".
    expect(tab().subgraphs).toHaveLength(1);
    expect(tab().subgraphs[0].name).toBe('Block');
    const pasted = tab().nodes.find((n) => subgraphIdOf(n.data.type))!;
    expect(subgraphIdOf(pasted.data.type)).toBe(tab().subgraphs[0].id);
  });
});

// ── Review MAJOR 3: leaving a block is its OWN undo step ─────────────────
//
// Before the fix `exitSubgraph` restored the outer undo stack verbatim while
// committing the edited definition, so the definition edit landed in the
// graph with NO undo entry behind it. The next Ctrl+Z therefore reached past
// the block edit and undid whatever the user had done OUTSIDE it -- while
// silently taking the block edit with it, because `undo` restores
// `subgraphs` from the snapshot too.

describe('leaving a sub-canvas is one undoable step', () => {
  function collapseBC(): string {
    seedChain();
    select('b', 'c');
    store().collapseSelectionToSubgraph('Block');
    return tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
  }

  const blockScale = () =>
    tab().subgraphs[0].nodes.find((n: any) => n.id === 'b').data.params.scale;
  const outerLabel = () => tab().nodes.find((n) => n.id === 'a')!.data.label;

  it('ONE undo reverts the block edit and LEAVES an unrelated outer edit alone', () => {
    const instanceId = collapseBC();
    store().renameNode('a', 'OUTER-EDIT'); // an unrelated action at the top level
    store().enterSubgraph(instanceId);
    store().updateNodeParams('b', { scale: 77 });
    store().exitSubgraph();

    expect(blockScale()).toBe(77);
    expect(outerLabel()).toBe('OUTER-EDIT');

    store().undo();

    expect(blockScale()).toBe(1);
    expect(outerLabel()).toBe('OUTER-EDIT'); // the rename survives

    // A SECOND undo is what reaches the rename -- one step per action.
    store().undo();
    expect(outerLabel()).toBe('a');

    store().redo();
    store().redo();
    expect(blockScale()).toBe(77);
    expect(outerLabel()).toBe('OUTER-EDIT');
  });

  it('pushes exactly one entry, however many things changed inside', () => {
    const instanceId = collapseBC();
    const depth = tab().undoStack.length;
    store().enterSubgraph(instanceId);
    store().updateNodeParams('b', { scale: 3 });
    store().updateNodeParams('c', { scale: 4 });
    store().deleteNode('c');
    store().exitSubgraph();
    expect(tab().undoStack.length).toBe(depth + 1);
  });

  it('clears redo, so a stale redo cannot resurrect a graph that never was', () => {
    const instanceId = collapseBC();
    store().renameNode('a', 'x');
    store().undo(); // fills the redo stack
    expect(tab().redoStack.length).toBe(1);
    store().enterSubgraph(instanceId);
    store().updateNodeParams('b', { scale: 5 });
    store().exitSubgraph();
    expect(tab().redoStack).toEqual([]);
  });

  // The false positive the fix has to avoid: `definitionFromCanvas` rebuilds
  // positions and edges from scratch, so a naive "did the object change?"
  // compare would report a change for merely looking inside a block.
  it('pushes NOTHING when the user enters and leaves without editing', () => {
    const instanceId = collapseBC();
    const depth = tab().undoStack.length;
    const before = JSON.stringify(tab().subgraphs);

    store().enterSubgraph(instanceId);
    store().exitSubgraph();

    expect(tab().undoStack.length).toBe(depth);
    expect(JSON.stringify(tab().subgraphs)).toBe(before);
  });

  it('exitAllSubgraphs is one step too', () => {
    const instanceId = collapseBC();
    store().renameNode('a', 'OUTER-EDIT');
    store().enterSubgraph(instanceId);
    store().updateNodeParams('b', { scale: 55 });
    store().exitAllSubgraphs();

    expect(tab().subgraphStack).toEqual([]);
    expect(blockScale()).toBe(55);

    store().undo();
    expect(blockScale()).toBe(1);
    expect(outerLabel()).toBe('OUTER-EDIT');
  });

  it('exitAllSubgraphs pushes nothing when nothing was edited', () => {
    const instanceId = collapseBC();
    const depth = tab().undoStack.length;
    store().enterSubgraph(instanceId);
    store().exitAllSubgraphs();
    expect(tab().undoStack.length).toBe(depth);
  });
});

// ── Review MAJOR 6 / MINOR 9: nested definitions ────────────────────────

/**
 * Collapse b+c into "Inner", then collapse `a` together with that instance
 * into "Outer" -- so Outer's DEFINITION holds an instance of Inner and the
 * canvas holds no direct reference to Inner at all.
 */
function collapseNested(): { outerInstanceId: string } {
  seedChain();
  select('b', 'c');
  store().collapseSelectionToSubgraph('Inner');
  const innerInstance = tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
  select('a', innerInstance);
  store().collapseSelectionToSubgraph('Outer');
  const outerInstanceId = tab().nodes.find(
    (n) => subgraphIdOf(n.data.type) && n.data.label === 'Outer',
  )!.id;
  return { outerInstanceId };
}

describe('nested definitions', () => {
  it('the fixture really does nest one definition inside another', () => {
    collapseNested();
    const outer = tab().subgraphs.find((d) => d.name === 'Outer')!;
    const inner = tab().subgraphs.find((d) => d.name === 'Inner')!;
    expect(outer.nodes.some((n: any) => n.type === `subgraph:${inner.id}`)).toBe(true);
    // Nothing on the canvas points at Inner directly.
    expect(
      tab().nodes.some((n) => subgraphIdOf(n.data.type) === inner.id),
    ).toBe(false);
  });

  it('copy/paste into ANOTHER tab carries the nested definition too', () => {
    const { outerInstanceId } = collapseNested();
    select(outerInstanceId);
    store().copySelectedNodes();

    store().addTab('other');
    expect(tab().subgraphs).toEqual([]);
    store().pasteNodes();

    // Without the nested one, the pasted tab holds an Outer definition that
    // references `subgraph:<inner>` -- a node the canvas draws and the server
    // refuses to run.
    expect(tab().subgraphs.map((d) => d.name).sort()).toEqual(['Inner', 'Outer']);
  });

  it('serializing keeps a definition reachable only from INSIDE another', () => {
    collapseNested();
    const graph = store().getSerializedGraph();
    expect(graph.subgraphs!.map((d: any) => d.name).sort()).toEqual([
      'Inner', 'Outer',
    ]);
  });
});

describe('orphaned definitions are never serialized', () => {
  function collapseBC(): string {
    seedChain();
    select('b', 'c');
    store().collapseSelectionToSubgraph('Block');
    return tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
  }

  it('deleting the only instance with deleteNode leaves no orphan in the save', () => {
    const instanceId = collapseBC();
    store().deleteNode(instanceId);
    expect(store().getSerializedGraph().subgraphs).toEqual([]);
  });

  it('deleting the only instance with the Delete key leaves no orphan either', () => {
    // The Delete key never reaches `deleteNode`: React Flow routes it through
    // onNodesChange, which is the path most users actually take.
    const instanceId = collapseBC();
    store().onNodesChange([{ type: 'remove', id: instanceId } as never]);
    expect(tab().nodes.some((n) => n.id === instanceId)).toBe(false);
    expect(store().getSerializedGraph().subgraphs).toEqual([]);
  });

  it('keeps the definition in memory, so undoing the delete brings it back', () => {
    const instanceId = collapseBC();
    store().deleteNode(instanceId);
    expect(tab().subgraphs).toHaveLength(1); // still there, just not saved
    store().undo();
    expect(store().getSerializedGraph().subgraphs).toHaveLength(1);
  });
});

// ── Review MINOR 14: collapse cleans up what deleteNode cleans up ────────

describe('collapse drops references to the nodes it swallowed', () => {
  it('drops segments, clears activeSegment and unbinds notes', () => {
    seedChain();
    const boundNote: Node<NodeData> = {
      id: 'note1',
      type: 'noteNode',
      position: { x: 210, y: 120 },
      data: {
        label: 'Note', type: 'note', params: {},
        noteKind: 'text', noteContent: 'about b', noteColor: '#3d3d1a',
        boundToNodeId: 'b', boundOffset: { x: 10, y: 60 },
      },
    };
    const otherNote: Node<NodeData> = {
      ...boundNote,
      id: 'note2',
      data: { ...boundNote.data, boundToNodeId: 'a', boundOffset: { x: 1, y: 2 } },
    };
    store().setNodes([...tab().nodes, boundNote, otherNote]);
    store().setSegmentGroups([
      { id: 's-inside', headNodeId: 'b', tailNodeId: 'c' },
      { id: 's-outside', headNodeId: 'a', tailNodeId: 'sink' },
    ]);
    store().setActiveSegment({ id: 's-inside', headNodeId: 'b', tailNodeId: 'c' });

    select('b', 'c');
    expect(store().collapseSelectionToSubgraph('Block').ok).toBe(true);

    // A segment can never resolve a path once an endpoint moved inside a
    // block -- exactly the reason deleteNode drops one.
    expect(tab().segmentGroups.map((s) => s.id)).toEqual(['s-outside']);
    expect(tab().activeSegment).toBeNull();
    // The note bound to `b` is unbound; the one bound to a surviving node
    // keeps its binding.
    expect(tab().nodes.find((n) => n.id === 'note1')!.data.boundToNodeId).toBeNull();
    expect(tab().nodes.find((n) => n.id === 'note1')!.data.boundOffset).toBeNull();
    expect(tab().nodes.find((n) => n.id === 'note2')!.data.boundToNodeId).toBe('a');
  });
});

// ── Review MINOR 17: readOnly covers mutations, never navigation ─────────

function makeReadOnly() {
  useTabStore.setState({
    tabs: useTabStore.getState().tabs.map((t) => ({ ...t, readOnly: true })),
  });
}

describe('readOnly on the subgraph actions', () => {
  function collapseBC(): string {
    seedChain();
    select('b', 'c');
    store().collapseSelectionToSubgraph('Block');
    return tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
  }

  it('refuses renameSubgraph, the third user-initiated mutation', () => {
    collapseBC();
    const sgId = tab().subgraphs[0].id;
    const depth = tab().undoStack.length;
    makeReadOnly();
    store().renameSubgraph(sgId, 'Renamed');
    expect(tab().subgraphs[0].name).toBe('Block');
    expect(tab().undoStack.length).toBe(depth);
  });

  it('still lets the user walk into and out of a block they cannot edit', () => {
    const instanceId = collapseBC();
    makeReadOnly();
    expect(store().enterSubgraph(instanceId)).toBe(true);
    expect(tab().nodes.map((n) => n.id).sort()).toEqual(['b', 'c']);
    store().exitSubgraph();
    expect(tab().subgraphStack).toEqual([]);
  });

  it('still lets a read-only graph LOAD its definitions', () => {
    seedChain();
    makeReadOnly();
    store().setSubgraphs([
      {
        id: 'loaded', name: 'Loaded', description: '', nodes: [], edges: [],
        interface: { inputs: [], outputs: [], triggerTargets: [] },
      },
    ]);
    expect(tab().subgraphs.map((d) => d.id)).toEqual(['loaded']);
  });
});

// ── Review MINOR 13: a layout-less definition is laid out on first entry ─

describe('enterSubgraph lays out a definition that has no positions', () => {
  const flatDefinition = () => ({
    id: 'flat', name: 'Flat', description: '',
    nodes: [
      { id: 'x', type: 'A', data: { params: { scale: 1 } } },
      { id: 'y', type: 'B', data: { params: { scale: 1 } } },
      { id: 'z', type: 'C', data: { params: { scale: 1 } } },
    ],
    edges: [
      { id: 'i1', source: 'x', target: 'y', sourceHandle: 'out', targetHandle: 'in' },
      { id: 'i2', source: 'y', target: 'z', sourceHandle: 'out', targetHandle: 'in' },
    ],
    interface: { inputs: [], outputs: [], triggerTargets: [] },
  });

  it('does not pile every node at the origin', () => {
    const definition = flatDefinition();
    store().setSubgraphs([definition as never]);
    store().setNodes([
      buildInstanceNode(definition as never, { x: 0, y: 0 }, 'inst'),
    ]);

    expect(store().enterSubgraph('inst')).toBe(true);
    const positions = tab().nodes.map((n) => `${n.position.x},${n.position.y}`);
    expect(tab().nodes).toHaveLength(3);
    expect(new Set(positions).size).toBe(3);
  });

  it('leaves a definition that already has positions exactly where it was', () => {
    seedChain();
    select('b', 'c');
    store().collapseSelectionToSubgraph('Block');
    const instanceId = tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
    const stored = tab().subgraphs[0].nodes.map((n: any) => ({
      id: n.id, x: n.position.x, y: n.position.y,
    }));

    store().enterSubgraph(instanceId);

    for (const { id, x, y } of stored) {
      const live = tab().nodes.find((n) => n.id === id)!;
      expect(live.position).toEqual({ x, y });
    }
  });
});
