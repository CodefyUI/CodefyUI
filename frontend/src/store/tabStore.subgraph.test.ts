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

import {
  useTabStore,
  _buildPersistedTabForTesting,
  _tabFromPersistedForTesting,
} from './tabStore';
import { useNodeDefStore } from './nodeDefStore';
import type { NodeData, NodeDefinition, PresetDefinition } from '../types';
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

// ── #200 item 2: the overlays a subgraph action dropped come back ────────

describe('undo restores the segments a subgraph action dropped', () => {
  it('undo of a collapse puts the swallowed segment back, focused', () => {
    seedChain();
    const inside = { id: 's-inside', headNodeId: 'b', tailNodeId: 'c' };
    store().setSegmentGroups([inside]);
    store().setActiveSegment(inside);

    select('b', 'c');
    expect(store().collapseSelectionToSubgraph('Block').ok).toBe(true);
    expect(tab().segmentGroups).toEqual([]);

    // The collapse is ONE undo step and the overlay travels in its frame, so
    // the segment is back with the graph rather than gone for good.
    store().undo();
    expect(tab().segmentGroups.map((s) => s.id)).toEqual(['s-inside']);
    expect(tab().activeSegment?.id).toBe('s-inside');
    // ...and redo drops it again, in step with the collapse it belongs to.
    store().redo();
    expect(tab().segmentGroups).toEqual([]);
    expect(tab().activeSegment).toBeNull();
  });

  it('undo of an expand puts the segment through the instance back', () => {
    seedChain();
    select('b', 'c');
    store().collapseSelectionToSubgraph('Block');
    const instanceId = tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
    const through = { id: 's-through', headNodeId: 'a', tailNodeId: instanceId };
    store().setSegmentGroups([through]);
    store().setActiveSegment(through);

    expect(store().expandSubgraphInstance(instanceId)).toBe(true);
    expect(tab().segmentGroups).toEqual([]);

    store().undo();
    expect(tab().segmentGroups.map((s) => s.id)).toEqual(['s-through']);
    expect(tab().activeSegment?.id).toBe('s-through');
  });

  // ── an overlay-only visit is still a visit (item 2 review) ────────────
  //
  // The exit used to ask "did anything change in there?" of the definitions
  // alone. Overlays are mutable from inside a block -- Compare Segment works on
  // two inner nodes, Clear active clears a top-level segment -- and an
  // overlay-only visit leaves every definition byte-identical. So the exit
  // pushed nothing, the inner entry went out with the inner stack, and the
  // restored outer stack's top frame still held the PRE-entry overlays: the
  // next unrelated Ctrl+Z wiped the overlay as a side effect.

  it('keeps a segment created inside a block when a later undo is about something else', () => {
    seedChain();
    select('b', 'c');
    store().collapseSelectionToSubgraph('Block');
    const instanceId = tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
    store().renameNode('a', 'OUTER-EDIT'); // the unrelated step, one frame

    store().enterSubgraph(instanceId);
    store().addSegmentGroup({ id: 's-inner', headNodeId: 'b', tailNodeId: 'c' });
    store().exitSubgraph();
    expect(tab().segmentGroups.map((s) => s.id)).toEqual(['s-inner']);

    // The visit is its own step, so the FIRST undo is the visit -- and the
    // rename it did not touch survives it.
    store().undo();
    expect(tab().segmentGroups).toEqual([]);
    expect(tab().nodes.find((n) => n.id === 'a')!.data.label).toBe('OUTER-EDIT');

    // ...and the rename's own undo is not what took the overlay away.
    store().undo();
    expect(tab().nodes.find((n) => n.id === 'a')!.data.label).toBe('a');
  });

  it('gives a segment cleared inside a block its own undo entry', () => {
    seedChain();
    select('b', 'c');
    store().collapseSelectionToSubgraph('Block');
    const instanceId = tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
    const top = { id: 's-top', headNodeId: 'a', tailNodeId: 'sink' };
    store().setSegmentGroups([top]);
    store().setActiveSegment(top);

    store().enterSubgraph(instanceId);
    // Clear active stays enabled inside a block, and it clears the TOP-LEVEL
    // segment -- the mirror of the case above.
    store().removeSegmentGroup('s-top');
    store().exitSubgraph();
    expect(tab().segmentGroups).toEqual([]);

    store().undo();
    expect(tab().segmentGroups.map((s) => s.id)).toEqual(['s-top']);
    expect(tab().activeSegment?.id).toBe('s-top');
  });

  it('costs nothing for a visit that leaves the overlays as it found them', () => {
    seedChain();
    select('b', 'c');
    store().collapseSelectionToSubgraph('Block');
    const instanceId = tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
    const top = { id: 's-top', headNodeId: 'a', tailNodeId: 'sink' };
    store().setSegmentGroups([top]);
    const depth = tab().undoStack.length;

    store().enterSubgraph(instanceId);
    store().exitSubgraph();

    expect(tab().undoStack.length).toBe(depth);
  });

  it('costs nothing when an overlay is created and undone inside the block', () => {
    // The phantom-step trap: the inner undo promotes the frame's COPY of the
    // list, so the array is equal to the pre-entry one and a different object.
    // An identity compare would charge a step for a visit that changed nothing.
    seedChain();
    select('b', 'c');
    store().collapseSelectionToSubgraph('Block');
    const instanceId = tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
    const top = { id: 's-top', headNodeId: 'a', tailNodeId: 'sink' };
    store().setSegmentGroups([top]);
    const onEntry = tab().segmentGroups;
    const depth = tab().undoStack.length;

    store().enterSubgraph(instanceId);
    store().addSegmentGroup({ id: 's-inner', headNodeId: 'b', tailNodeId: 'c' });
    store().undo(); // inside the block: back to just s-top, in a fresh array
    expect(tab().segmentGroups.map((s) => s.id)).toEqual(['s-top']);
    expect(tab().segmentGroups).not.toBe(onEntry); // same value, different array
    store().exitSubgraph();

    expect(tab().undoStack.length).toBe(depth);
  });

  it('undo of a block VISIT restores a segment deleted inside the block', () => {
    // The one path with no snapshot of its own: an undo taken inside a block
    // stays inside it, and the inner stack is thrown away on the way out. The
    // single frame the exit pushes describes the state on ENTRY -- so the
    // overlays it carries have to be the pre-entry ones, exactly as its
    // `subgraphs` already were.
    seedChain();
    select('b', 'c');
    store().collapseSelectionToSubgraph('Block');
    const instanceId = tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
    const inner = { id: 's-inner', headNodeId: 'b', tailNodeId: 'c' };
    store().setSegmentGroups([inner]);
    store().setActiveSegment(inner);

    store().enterSubgraph(instanceId);
    store().deleteNode('b'); // prunes the segment naming it
    expect(tab().segmentGroups).toEqual([]);
    store().exitSubgraph();

    store().undo();
    expect(tab().segmentGroups.map((s) => s.id)).toEqual(['s-inner']);
    expect(tab().activeSegment?.id).toBe('s-inner');
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

// ── Round-2 review findings ─────────────────────────────────────────────

function presetDefinition(name = 'KeyedChat') {
  return {
    preset_name: name,
    category: 'c',
    description: '',
    tags: [],
    nodes: [{ id: 'inner', type: 'LLMChat', params: {} }],
    edges: [],
    exposed_inputs: [],
    exposed_outputs: [],
    exposed_params: [],
  };
}

function presetNode(id: string, x = 0, y = 0, name = 'KeyedChat'): Node<NodeData> {
  return {
    id,
    type: 'presetNode',
    position: { x, y },
    data: {
      label: name,
      type: `preset:${name}`,
      params: {},
      definition: def(name),
      isPreset: true,
      presetDefinition: presetDefinition(name) as never,
      internalParams: { inner: { model: 'gpt-5.2' } },
      executionStatus: 'idle',
    },
  };
}

describe('getSerializedGraph carries what a definition depends on', () => {
  it('keeps the portable preset definition of a preset node collapsed into a block', () => {
    // A collapsed preset serializes as `preset:<name>` + internalParams
    // inside the definition, but `presets[]` was collected from the CANVAS
    // only -- so the file kept the block and dropped the thing that makes
    // the preset inside it resolvable. Export 400s; save writes an
    // unrunnable file.
    //
    // The preset is in the node-def registry, which is where every path that
    // puts a preset node on a canvas leaves it: installed presets come from
    // `/api/presets`, and a graph/example/import that ships one merges it in
    // before resolving its nodes.
    useNodeDefStore.setState({ presets: [presetDefinition()] } as never);
    store().setNodes([
      { ...node('start', 'Start', 0, 0), type: 'start' },
      node('a', 'A', 100, 0),
      presetNode('p', 200, 0),
    ]);
    store().setEdges([dataEdge('e1', 'a', 'p')]);
    expect(store().getSerializedGraph().presets.map((p) => p.preset_name))
      .toEqual(['KeyedChat']);

    select('a', 'p');
    const result = store().collapseSelectionToSubgraph('Block');
    expect(result.ok).toBe(true);

    const serialized = store().getSerializedGraph();
    // The definition really did swallow the preset node.
    expect(
      serialized.subgraphs[0].nodes.map((n: any) => n.type),
    ).toContain('preset:KeyedChat');
    expect(serialized.presets.map((p) => p.preset_name)).toEqual(['KeyedChat']);
  });

  it('blanks SECRET params inside a definition', () => {
    const secretDef: NodeDefinition = {
      ...def('LLMChat'),
      params: [
        {
          name: 'api_key', param_type: 'secret', default: '', description: '',
          options: [], min_value: null, max_value: null,
        },
      ],
    };
    useNodeDefStore.setState({ definitions: [def('A'), secretDef] } as never);
    const keyed = node('k', 'LLMChat', 100, 0);
    store().setNodes([
      node('a', 'A', 0, 0),
      { ...keyed, data: { ...keyed.data, definition: secretDef, params: { api_key: 'sk-LEAK' } } },
    ]);
    store().setEdges([dataEdge('e1', 'a', 'k')]);
    select('a', 'k');
    expect(store().collapseSelectionToSubgraph('Block').ok).toBe(true);

    const serialized = store().getSerializedGraph();
    expect(JSON.stringify(serialized)).not.toContain('sk-LEAK');
    const inner = serialized.subgraphs[0].nodes.find((n: any) => n.id === 'k');
    expect(inner.data.params.api_key).toBe('');
  });
});

// A node inside a block keeps no definition of its own, so the strip learns
// its SECRET params from the node list, and a preset's SECRET slots from the
// preset list. Every fetch replaces both lists whole: a custom node disabled
// in the Custom Nodes manager, or a preset that came in with an opened file,
// is missing from the next one, and a key typed into that node used to be
// saved, exported and autosaved as typed (#537 review).
describe('a key typed into a block stays blank once the lists stop naming it', () => {
  /** `def(name)` plus a SECRET `api_key`, the way an LLM node declares its key. */
  function keyedDef(name: string): NodeDefinition {
    return {
      ...def(name),
      params: [
        ...def(name).params,
        {
          name: 'api_key', param_type: 'secret', default: '', description: '',
          options: [], min_value: null, max_value: null,
        },
      ],
    };
  }

  // MyChat is a custom node, listed while the keys are typed. Chat stays
  // listed throughout.
  beforeEach(() => {
    useNodeDefStore.setState({
      definitions: [def('A'), keyedDef('Chat'), keyedDef('MyChat')],
    } as never);
  });

  /** An old preset that still exposes its inner node's key as a field. */
  function keyedPreset(name: string): PresetDefinition {
    return {
      ...presetDefinition(name),
      exposed_params: [{
        internal_node: 'inner', param_name: 'api_key', display_name: 'Chat - api_key',
        group: 'Chat', param_def: keyedDef('Chat').params[1],
      }],
    };
  }

  /** A canvas node of `type` whose own definition is `keyedDef(type)`. */
  function keyed(id: string, type: string, y: number): Node<NodeData> {
    const n = node(id, type, 100, y);
    return {
      ...n,
      data: { ...n.data, definition: keyedDef(type), params: { scale: 1, api_key: '' } },
    };
  }

  /** The Custom Nodes manager disables MyChat and fetches the list again. */
  const dropMyChat = () =>
    useNodeDefStore.setState({ definitions: [def('A'), keyedDef('Chat')] } as never);

  /**
   * a feeds k (MyChat) and j (Chat); the three are collapsed into a block and
   * a key is typed into k and j from inside it, where both show the masked
   * field. Leaves the canvas inside the block.
   */
  function typeKeysInsideBlock() {
    store().setNodes([node('a', 'A', 0, 0), keyed('k', 'MyChat', 0), keyed('j', 'Chat', 80)]);
    store().setEdges([dataEdge('e1', 'a', 'k'), dataEdge('e2', 'a', 'j')]);
    select('a', 'k', 'j');
    expect(store().collapseSelectionToSubgraph('Block').ok).toBe(true);
    const instanceId = tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
    expect(store().enterSubgraph(instanceId)).toBe(true);
    store().updateNodeParams('k', { api_key: 'sk-MYCHAT' });
    store().updateNodeParams('j', { api_key: 'sk-CHAT' });
  }

  /** What the graph's one block holds for node `id`. */
  const innerParams = (graph: { subgraphs: { nodes: any[] }[] }, id: string) =>
    graph.subgraphs[0].nodes.find((n) => n.id === id).data.params;

  it('on a node whose type the node list no longer has', () => {
    typeKeysInsideBlock();
    store().exitSubgraph();
    dropMyChat();
    // A later fetch without MyChat (after Export as preset, say) does not
    // make its key stop being a key.
    useNodeDefStore.setState({ definitions: [def('A'), keyedDef('Chat'), def('B')] } as never);
    // The premise: the block really holds the key.
    expect(JSON.stringify(tab().subgraphs)).toContain('sk-MYCHAT');

    const saved = store().getSerializedGraph();
    expect(JSON.stringify(saved)).not.toContain('sk-MYCHAT');
    expect(JSON.stringify(saved)).not.toContain('sk-CHAT');
    // Blanked in place; the node's other params are kept.
    expect(innerParams(saved, 'k')).toEqual({ scale: 1, api_key: '' });
    expect(JSON.stringify(_buildPersistedTabForTesting(tab()))).not.toContain('sk-MYCHAT');
  });

  it('while the canvas is still inside the block', () => {
    typeKeysInsideBlock();
    dropMyChat();
    // The node on screen still shows the masked field: its own definition,
    // taken when the block was opened, still calls the key SECRET.
    const k = tab().nodes.find((n) => n.id === 'k')!;
    expect(k.data.definition!.params.some((p) => p.param_type === 'secret')).toBe(true);

    const saved = store().getSerializedGraph();
    expect(innerParams(saved, 'k')).toEqual({ scale: 1, api_key: '' });
    expect(innerParams(saved, 'j')).toEqual({ scale: 1, api_key: '' });
    expect(JSON.stringify(_buildPersistedTabForTesting(tab()))).not.toContain('sk-MYCHAT');
    // The Run message's serializer, called directly: Run pressed in here
    // stops at "No entry points", because it reads them off the canvas on
    // screen. It keeps the key of the type the server still has, and blanks
    // the one the server could not keep out of the run it stores.
    const run = store().getSerializedGraph({ keepSecrets: true });
    expect(innerParams(run, 'k').api_key).toBe('');
    expect(innerParams(run, 'j').api_key).toBe('sk-CHAT');
    // Serializing did not close the block behind the user.
    expect(tab().subgraphStack).toHaveLength(1);
  });

  it('in the slot of a preset the preset list no longer has', () => {
    // Merged into the list by the reader of the file that carried it.
    useNodeDefStore.setState({ presets: [keyedPreset('FileChat')] } as never);
    store().setNodes([{
      id: 'inst', type: 'subgraphNode', position: { x: 0, y: 0 },
      data: { label: 'Block', type: 'subgraph:blk', params: {} },
    }]);
    store().setSubgraphs([{
      id: 'blk', name: 'Block', description: '',
      nodes: [{
        id: 'p', type: 'preset:FileChat', position: { x: 0, y: 0 },
        data: { params: {}, internalParams: { inner: { api_key: 'sk-IN-PRESET', model: 'gpt-5.2' } } },
      }],
      edges: [],
      interface: { inputs: [], outputs: [], triggerTargets: [] },
    }]);
    // A fetch replaces the list with the server's, which never had FileChat.
    useNodeDefStore.setState({ presets: [] } as never);

    const saved = store().getSerializedGraph();
    expect(JSON.stringify(saved)).not.toContain('sk-IN-PRESET');
    expect(saved.subgraphs[0].nodes[0].data.internalParams.inner)
      .toEqual({ api_key: '', model: 'gpt-5.2' });
    expect(JSON.stringify(_buildPersistedTabForTesting(tab()))).not.toContain('sk-IN-PRESET');
  });

  it('leaves a block with nothing to blank as the same list, for the autosave cache', () => {
    // The lists have named MyChat's key; nothing in this block is one.
    dropMyChat();
    seedChain();
    select('b', 'c');
    expect(store().collapseSelectionToSubgraph('Block').ok).toBe(true);
    expect(_buildPersistedTabForTesting(tab()).subgraphs).toBe(tab().subgraphs);
  });

  // A node autosave brings back after a reload keeps the definition it was
  // saved with, so it still shows its masked field whether or not a list
  // this session names its type: a custom node disabled before the reload, a
  // preset that came in with a file the previous session opened.
  //
  // The node-def store's record lives as long as this file's module does, so
  // each case below names its own type: one another case had already folded
  // would be remembered from that case.

  /**
   * A page reload for the active tab: its nodes go through the autosave
   * record and come back with the definitions they were saved with. No list
   * is replayed, so no list this session names their types.
   */
  function reload() {
    const record = _buildPersistedTabForTesting(tab());
    const restored = _tabFromPersistedForTesting(record, tab());
    useTabStore.setState({ tabs: [restored], activeTabId: restored.id });
  }

  it('on a node restored after a reload, of a type no list has named since', () => {
    store().setNodes([node('a', 'A', 0, 0), keyed('k', 'RestoredChat', 0)]);
    store().setEdges([dataEdge('e1', 'a', 'k')]);
    reload();
    const k = tab().nodes.find((n) => n.id === 'k')!;
    expect(k.data.definition!.params.some((p) => p.param_type === 'secret')).toBe(true);
    store().updateNodeParams('k', { api_key: 'sk-RESTORED-NODE' });
    select('a', 'k');
    expect(store().collapseSelectionToSubgraph('Block').ok).toBe(true);
    // The premise: the block holds the key.
    expect(JSON.stringify(tab().subgraphs)).toContain('sk-RESTORED-NODE');

    const saved = store().getSerializedGraph();
    expect(JSON.stringify(saved)).not.toContain('sk-RESTORED-NODE');
    expect(innerParams(saved, 'k')).toEqual({ scale: 1, api_key: '' });
    expect(JSON.stringify(_buildPersistedTabForTesting(tab()))).not.toContain('sk-RESTORED-NODE');
  });

  it('on a preset node restored after a reload, of a preset no list has named since', () => {
    const p = presetNode('p', 100, 0, 'RestoredPreset');
    store().setNodes([
      node('a', 'A', 0, 0),
      { ...p, data: { ...p.data, presetDefinition: keyedPreset('RestoredPreset') } },
    ]);
    store().setEdges([dataEdge('e1', 'a', 'p')]);
    reload();
    store().updatePresetInternalParam('p', 'inner', 'api_key', 'sk-RESTORED-PRESET');
    select('a', 'p');
    expect(store().collapseSelectionToSubgraph('Block').ok).toBe(true);
    expect(JSON.stringify(tab().subgraphs)).toContain('sk-RESTORED-PRESET');

    const saved = store().getSerializedGraph();
    expect(JSON.stringify(saved)).not.toContain('sk-RESTORED-PRESET');
    expect(saved.subgraphs[0].nodes.find((n: any) => n.id === 'p').data.internalParams.inner)
      .toEqual({ model: 'gpt-5.2', api_key: '' });
    expect(JSON.stringify(_buildPersistedTabForTesting(tab()))).not.toContain('sk-RESTORED-PRESET');
  });

  /**
   * A restored node of `type`, copied and pasted into an open block, where
   * `key` is typed into the copy. Leaves the canvas inside the block.
   */
  function pasteRestoredNodeIntoBlock(type: string, key: string) {
    store().setNodes([keyed('r', type, 0), {
      id: 'inst', type: 'subgraphNode', position: { x: 300, y: 0 },
      data: { label: 'Block', type: 'subgraph:blk', params: {} },
    }]);
    store().setSubgraphs([{
      id: 'blk', name: 'Block', description: '',
      nodes: [{ id: 'x', type: 'A', position: { x: 0, y: 0 }, data: { params: { scale: 1 } } }],
      edges: [],
      interface: { inputs: [], outputs: [], triggerTargets: [] },
    }]);
    reload();
    select('r');
    store().copySelectedNodes();
    expect(store().enterSubgraph('inst')).toBe(true);
    store().pasteNodes();
    const pasted = tab().nodes.find((n) => n.data.type === type)!;
    store().updateNodeParams(pasted.id, { api_key: key });
  }

  it('on a restored node pasted into an open block, saved from inside it', () => {
    pasteRestoredNodeIntoBlock('PastedChat', 'sk-SAVED-INSIDE');

    // Every save folds the open block first; this one never left it.
    expect(JSON.stringify(store().getSerializedGraph())).not.toContain('sk-SAVED-INSIDE');
    expect(JSON.stringify(_buildPersistedTabForTesting(tab()))).not.toContain('sk-SAVED-INSIDE');
    expect(tab().subgraphStack).toHaveLength(1);
  });

  it('on a restored node pasted into an open block, saved after stepping out', () => {
    pasteRestoredNodeIntoBlock('SteppedOutChat', 'sk-SAVED-OUTSIDE');
    store().exitSubgraph();
    expect(JSON.stringify(tab().subgraphs)).toContain('sk-SAVED-OUTSIDE');

    expect(JSON.stringify(store().getSerializedGraph())).not.toContain('sk-SAVED-OUTSIDE');
    expect(JSON.stringify(_buildPersistedTabForTesting(tab()))).not.toContain('sk-SAVED-OUTSIDE');
  });
});

describe('a definition entry a graph file carries but nobody validated', () => {
  it('is coerced at the door instead of wedging save and autosave', () => {
    // `File > Import` on `{"subgraphs":[{"id":"x"}]}` produces exactly this
    // call: the readers check `Array.isArray` on the LIST and nothing at all
    // on its ENTRIES, and `resolveSerializedNodes` reads only `sg.id`, so the
    // entry arrives intact. The server accepts it too -- `id` is the only
    // required field on `SubgraphDefinition` -- so the editor must not
    // delete a block the server would happily run.
    seedChain();
    store().setNodes([
      ...tab().nodes,
      {
        id: 'inst',
        type: 'subgraphNode',
        position: { x: 500, y: 0 },
        data: { label: 'x', type: 'subgraph:x', params: {} },
      } as never,
    ]);

    store().setSubgraphs([{ id: 'x' }] as never);

    const [installed] = tab().subgraphs;
    expect(installed.nodes).toEqual([]);
    expect(installed.edges).toEqual([]);
    expect(installed.interface).toEqual({
      inputs: [],
      outputs: [],
      triggerTargets: [],
    });

    // Save and Run walk `definition.nodes` (`reachableSubgraphIds`), and so
    // does the secret strip on every autosave. Both threw on this input, and
    // the autosave kept throwing for the rest of the session -- silently, so
    // the user goes on working in a tab nothing is writing to disk any more.
    expect(() => store().getSerializedGraph()).not.toThrow();
  });

  it('drops only an entry with no usable id, since every lookup is by id', () => {
    seedChain();
    store().setSubgraphs([
      null,
      'nope',
      { name: 'no id at all' },
      { id: '' },
      { id: 'keep' },
    ] as never);
    expect(tab().subgraphs.map((d) => d.id)).toEqual(['keep']);
  });
});

describe('clear() while inside a sub-canvas', () => {
  it('snapshots the WHOLE graph, so undo restores the outer canvas', () => {
    seedChain();
    select('b', 'c');
    store().collapseSelectionToSubgraph('Block');
    const outerIds = tab().nodes.map((n) => n.id).sort();
    const instanceId = tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
    store().enterSubgraph(instanceId);
    expect(tab().nodes.map((n) => n.id).sort()).toEqual(['b', 'c']);

    store().clear();
    store().undo();

    // Not ['b','c'] -- the block's insides are not the user's graph.
    expect(tab().nodes.map((n) => n.id).sort()).toEqual(outerIds);
    expect(tab().subgraphs).toHaveLength(1);
    expect(tab().subgraphStack).toHaveLength(0);
  });
});

describe('expandSubgraphInstance cleans up after the instance it removes', () => {
  it('drops segments, note bindings and an open detail modal that named it', () => {
    seedChain();
    select('b', 'c');
    store().collapseSelectionToSubgraph('Block');
    const instanceId = tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
    // A segment through the instance, a note bound to it, and its detail modal.
    store().setSegmentGroups([
      { id: 's1', headNodeId: 'a', tailNodeId: instanceId } as never,
    ]);
    store().setActiveSegment({ headNodeId: 'a', tailNodeId: instanceId } as never);
    store().setNodes([
      ...tab().nodes,
      {
        id: 'note', type: 'noteNode', position: { x: 0, y: 0 },
        data: { label: 'Note', type: 'note', params: {}, boundToNodeId: instanceId, boundOffset: { x: 1, y: 1 } },
      } as never,
    ]);
    store().openNodeDetail(instanceId);

    expect(store().expandSubgraphInstance(instanceId)).toBe(true);

    expect(tab().nodes.some((n) => n.id === instanceId)).toBe(false);
    expect(tab().segmentGroups).toEqual([]);
    expect(tab().activeSegment).toBeNull();
    expect(tab().nodes.find((n) => n.id === 'note')!.data.boundToNodeId).toBeNull();
    expect(tab().nodeDetailNodeId).toBeNull();
  });

  it('closes a viz viewer that named the instance, and keeps one that named another node (core#324)', () => {
    seedChain();
    select('b', 'c');
    store().collapseSelectionToSubgraph('Block');
    const instanceId = tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;

    store().openVizModal('a');
    expect(store().expandSubgraphInstance(instanceId)).toBe(true);
    expect(tab().vizModalNodeId).toBe('a');

    store().undo();
    const again = tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
    store().openVizModal(again);
    expect(store().expandSubgraphInstance(again)).toBe(true);
    expect(tab().vizModalNodeId).toBeNull();
  });
});

describe('collapseSelectionToSubgraph closes a detail modal it swallowed', () => {
  it('clears nodeDetailNodeId when the node moved into the block', () => {
    seedChain();
    store().openNodeDetail('b');
    select('b', 'c');
    expect(store().collapseSelectionToSubgraph('Block').ok).toBe(true);
    expect(tab().nodes.some((n) => n.id === 'b')).toBe(false);
    expect(tab().nodeDetailNodeId).toBeNull();
  });

  it('clears vizModalNodeId when the node moved into the block, and keeps one that stayed outside (core#324)', () => {
    seedChain();
    store().openVizModal('a');
    select('b', 'c');
    expect(store().collapseSelectionToSubgraph('Block').ok).toBe(true);
    expect(tab().vizModalNodeId).toBe('a');

    store().undo();
    store().openVizModal('b');
    select('b', 'c');
    expect(store().collapseSelectionToSubgraph('Block').ok).toBe(true);
    expect(tab().vizModalNodeId).toBeNull();
  });
});

describe('pasteNodes on a definition-id collision', () => {
  it('re-renders the pasted instance from the LOCAL definition', () => {
    seedChain();
    select('b', 'c');
    store().collapseSelectionToSubgraph('Block');
    const instanceId = tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
    const definitionId = subgraphIdOf(
      tab().nodes.find((n) => n.id === instanceId)!.data.type,
    )!;
    select(instanceId);
    store().copySelectedNodes();

    // The local definition now exposes a DIFFERENT port from the clipboard's.
    store().setSubgraphs(
      tab().subgraphs.map((d) =>
        d.id === definitionId
          ? {
              ...d,
              interface: {
                ...d.interface,
                inputs: [{ port: 'local_in', innerNode: 'b', innerPort: 'in', data_type: 'TENSOR' }],
              },
            }
          : d,
      ),
    );
    store().pasteNodes();

    // Paste selects exactly what it added, so `selected` identifies the new
    // instance. The pre-existing one is deliberately NOT re-rendered here:
    // refreshing live instances is `refreshInstances`' job on sub-canvas
    // exit, and this test is about the node paste itself just created.
    const all = tab().nodes.filter(
      (n) => subgraphIdOf(n.data.type) === definitionId,
    );
    expect(all).toHaveLength(2);
    const pasted = all.filter((n) => n.selected);
    expect(pasted).toHaveLength(1);
    expect(pasted[0].data.definition!.inputs.map((p) => p.name)).toEqual([
      'local_in',
    ]);
  });
});
