/**
 * `TabState.revision` — the compare-and-swap token plugin API v5 hands out
 * (#341).
 *
 * The rule under test is a construction rule, not a per-action one: the
 * store's `set` is wrapped once, so a tab whose DOCUMENT changed in a
 * transition comes out of it with `revision + 1`. These tests drive that
 * through the PUBLIC actions, because "a new action forgot to bump" is
 * exactly the failure the wrapper exists to make impossible -- and the only
 * honest way to check it is to mutate the way the editor does.
 *
 * The other half of the contract, and the harder half, is what must NOT bump.
 * React Flow rewrites `nodes` for reasons that have nothing to do with the
 * document: selecting a card, and reporting a card's measured size when it
 * remounts after being culled by `onlyRenderVisibleElements`. A counter that
 * compared array references would climb while the user panned the canvas, and
 * every plugin's compare-and-swap would fail against a graph nobody had
 * touched. So the comparison is by CONTENT, and these tests are where that is
 * pinned.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { Edge, Node } from '@xyflow/react';
import { useTabStore, _documentChangedForTesting } from './tabStore';
import { useNodeDefStore } from './nodeDefStore';
import type { NodeData, NodeDefinition } from '../types';

vi.mock('./tabPersistence', () => ({
  readSnapshot: vi.fn(async () => null),
  writeSnapshot: vi.fn(async () => {}),
}));

const store = () => useTabStore.getState();
const activeTab = () => useTabStore.getState().getActiveTab();
const revision = () => activeTab().revision;

function def(name: string): NodeDefinition {
  return {
    node_name: name, category: 'Layer', description: '',
    inputs: [{ name: 'in', data_type: 'TENSOR', description: '', optional: false }],
    outputs: [{ name: 'out', data_type: 'TENSOR', description: '', optional: false }],
    params: [
      { name: 'size', param_type: 'int', default: 8, description: '', options: [], min_value: 1, max_value: 64 },
    ],
  };
}

function node(id: string, x = 0): Node<NodeData> {
  return {
    id, type: 'baseNode', position: { x, y: 0 },
    data: { label: id, type: 'A', params: {}, definition: def('A'), executionStatus: 'idle' },
  };
}

function dataEdge(id: string, source: string, target: string): Edge {
  return { id, source, target, sourceHandle: 'out', targetHandle: 'in' };
}

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  store().addTab('test');
  useNodeDefStore.setState({ definitions: [def('A')] } as never);
});

describe('TabState.revision', () => {
  it('starts at 1 on a fresh tab', () => {
    expect(revision()).toBe(1);
  });

  it('bumps exactly once per document-changing action', () => {
    // One `set` per action, one bump per `set`: the table is the contract.
    const steps: Array<[string, () => void]> = [
      ['setNodes', () => store().setNodes([node('a'), node('b', 200)])],
      ['setEdges', () => store().setEdges([dataEdge('e1', 'a', 'b')])],
      ['addNote', () => store().addNote('text', { x: 10, y: 10 })],
      ['deleteNode', () => store().deleteNode('b')],
      ['renameNode', () => store().renameNode('a', 'Renamed')],
      ['addSegmentGroup', () => store().addSegmentGroup({ id: 's1', headNodeId: 'a', tailNodeId: 'a' })],
      ['removeSegmentGroup', () => store().removeSegmentGroup('s1')],
      ['setSegmentGroups', () => store().setSegmentGroups([{ id: 's2', headNodeId: 'a', tailNodeId: 'a' }])],
      // A real definition, not `[]`: `documentChanged` compares definitions
      // with `sameSubgraphs`, and an empty list replaced by another empty
      // list is honestly not a change.
      ['setSubgraphs', () => store().setSubgraphs([{
        id: 'blk', name: 'Encoder', description: '',
        nodes: [], edges: [], interface: { inputs: [], outputs: [], triggerTargets: [] },
      }])],
      ['renameSubgraph', () => store().renameSubgraph('blk', 'Decoder')],
      ['updateNodeParams', () => store().updateNodeParams('a', { size: 16 })],
      ['clear', () => store().clear()],
    ];
    for (const [name, run] of steps) {
      const before = revision();
      run();
      expect(revision(), `${name} must bump revision by exactly 1`).toBe(before + 1);
    }
  });

  it('bumps on undo and on redo -- the document changed either way', () => {
    store().setNodes([node('a')]);
    store().pushUndoSnapshot();
    store().setNodes([node('a'), node('b', 200)]);
    const afterEdit = revision();

    store().undo();
    expect(revision()).toBe(afterEdit + 1);
    expect(activeTab().nodes).toHaveLength(1);

    store().redo();
    expect(revision()).toBe(afterEdit + 2);
    expect(activeTab().nodes).toHaveLength(2);
  });

  it('bumps on loadGraphDocument and never goes backwards', () => {
    store().setNodes([node('a')]);
    const before = revision();
    store().loadGraphDocument({ nodes: [], edges: [], boundFile: null });
    expect(revision()).toBe(before + 1);
    expect(revision()).toBeGreaterThan(1);
  });

  it('does NOT bump for rename, activate, a modal id or run status', () => {
    const id = activeTab().id;
    const before = revision();
    store().renameTab(id, 'Renamed tab');
    store().setActiveTab(id);
    store().setSelectedNodeId(null);
    store().openPresetModal('a');
    store().closePresetModal();
    store().closeNodeDetail();
    store().setTabStatus(id, 'running');
    expect(useTabStore.getState().getTab(id)!.revision).toBe(before);
  });
});

describe('revision ignores what is not the document', () => {
  beforeEach(() => {
    store().setNodes([node('a'), node('b', 200)]);
    store().setEdges([dataEdge('e1', 'a', 'b')]);
  });

  it('a plain click through setSelectedNodeId does not bump', () => {
    const before = revision();
    store().setSelectedNodeId('a');
    expect(revision()).toBe(before);
  });

  it('selectNodeExclusively does not bump', () => {
    // It writes `node.selected` onto every node, so the array reference moves
    // and the objects are fresh. Selection is not the document.
    const before = revision();
    store().selectNodeExclusively('a');
    expect(activeTab().nodes.find((n) => n.id === 'a')!.selected).toBe(true);
    expect(activeTab().selectedNodeId).toBe('a');
    expect(revision()).toBe(before);
  });

  it('openNodeDetail does not bump', () => {
    // Same write as above plus four scalar fields and the request nonce.
    const before = revision();
    store().openNodeDetail('a');
    expect(activeTab().nodeDetailNodeId).toBe('a');
    expect(revision()).toBe(before);
  });

  it('a select-only onNodesChange batch does not bump', () => {
    const before = revision();
    store().onNodesChange([
      { id: 'a', type: 'select', selected: true },
      { id: 'b', type: 'select', selected: false },
    ]);
    expect(activeTab().nodes.find((n) => n.id === 'a')!.selected).toBe(true);
    expect(revision()).toBe(before);
  });

  it('a select-only onEdgesChange batch does not bump', () => {
    const before = revision();
    store().onEdgesChange([{ id: 'e1', type: 'select', selected: true }]);
    expect(activeTab().edges[0].selected).toBe(true);
    expect(revision()).toBe(before);
  });

  it('a dimensions batch does not bump -- panning must not break a plugin CAS', () => {
    // This is the shape React Flow sends when a card that was culled by
    // `onlyRenderVisibleElements` remounts and re-measures itself: it sets
    // `measured`, and with `setAttributes` also `width` and `height`. Under a
    // reference-comparing counter, scrolling around a large graph would
    // invalidate every revision a plugin was holding.
    const before = revision();
    store().onNodesChange([
      { id: 'a', type: 'dimensions', dimensions: { width: 214, height: 88 }, setAttributes: true },
      { id: 'b', type: 'dimensions', dimensions: { width: 214, height: 88 } },
    ]);
    expect(activeTab().nodes[0].measured).toEqual({ width: 214, height: 88 });
    expect(revision()).toBe(before);
  });

  it('a resize flag on its own does not bump', () => {
    const before = revision();
    store().onNodesChange([{ id: 'a', type: 'dimensions', resizing: true }]);
    expect(revision()).toBe(before);
  });

  it('a drag START, which carries no coordinates, does not bump', () => {
    // `NodePositionChange.position` is optional; the first change of a drag
    // sets only `dragging`. Nothing about the document has changed yet.
    const before = revision();
    store().onNodesChange([{ id: 'a', type: 'position', dragging: true }]);
    expect(revision()).toBe(before);
  });

  it('a drag that actually moves a node bumps', () => {
    const before = revision();
    store().onNodesChange([
      { id: 'a', type: 'position', position: { x: 40, y: 12 }, dragging: true },
    ]);
    expect(activeTab().nodes[0].position).toEqual({ x: 40, y: 12 });
    expect(revision()).toBe(before + 1);
  });

  it('a position change to the SAME coordinates does not bump', () => {
    // React Flow hands over a fresh position object either way, so this is
    // only true because the comparison reads x and y rather than the
    // reference.
    const before = revision();
    store().onNodesChange([
      { id: 'a', type: 'position', position: { x: 0, y: 0 }, dragging: false },
    ]);
    expect(revision()).toBe(before);
  });

  it('a params edit bumps, because it replaces data', () => {
    const before = revision();
    store().updateNodeParams('a', { size: 16 });
    expect(revision()).toBe(before + 1);
  });

  it('documentChanged reads position by value and data by reference', () => {
    // The helper directly, since it is the whole rule.
    const base = activeTab();
    const sameShape = {
      ...base,
      nodes: base.nodes.map((n) => ({ ...n, position: { ...n.position }, selected: !n.selected })),
    };
    expect(_documentChangedForTesting(base, sameShape)).toBe(false);

    const moved = {
      ...base,
      nodes: base.nodes.map((n, i) => (i === 0 ? { ...n, position: { x: 1, y: 0 } } : n)),
    };
    expect(_documentChangedForTesting(base, moved)).toBe(true);

    const relabelled = {
      ...base,
      nodes: base.nodes.map((n, i) => (i === 0 ? { ...n, data: { ...n.data } } : n)),
    };
    expect(_documentChangedForTesting(base, relabelled)).toBe(true);

    const shorter = { ...base, nodes: base.nodes.slice(0, 1) };
    expect(_documentChangedForTesting(base, shorter)).toBe(true);
  });
});

describe('TabState.revision, continued', () => {

  it('does not bump a tab that did not change when a sibling tab does', () => {
    const first = activeTab().id;
    store().addTab('second');
    const second = activeTab().id;
    const firstRevision = useTabStore.getState().getTab(first)!.revision;
    store().setNodes([node('a')]);
    expect(useTabStore.getState().getTab(second)!.revision).toBeGreaterThan(1);
    expect(useTabStore.getState().getTab(first)!.revision).toBe(firstRevision);
  });

  it('pushUndoSnapshot alone does not bump -- it changes no document field', () => {
    store().setNodes([node('a')]);
    const before = revision();
    store().pushUndoSnapshot();
    expect(revision()).toBe(before);
  });

  it('a whole drag session bumps, and only for the batches that moved it', () => {
    // The real sequence: grab (no coordinates), two pointer moves, release
    // (same coordinates as the last move). Three of the four batches are not
    // document changes.
    store().setNodes([node('a')]);
    const before = revision();
    store().onNodesChange([{ id: 'a', type: 'position', dragging: true }]);
    store().onNodesChange([{ id: 'a', type: 'position', position: { x: 20, y: 5 }, dragging: true }]);
    store().onNodesChange([{ id: 'a', type: 'position', position: { x: 40, y: 9 }, dragging: true }]);
    store().onNodesChange([{ id: 'a', type: 'position', position: { x: 40, y: 9 }, dragging: false }]);
    expect(activeTab().nodes[0].position).toEqual({ x: 40, y: 9 });
    expect(revision()).toBe(before + 2);
  });
});

describe('revision persistence', () => {
  it('round-trips through buildPersistedTab / tabFromPersisted', async () => {
    const { _buildPersistedTabForTesting, _tabFromPersistedForTesting } = await import('./tabStore');
    store().setNodes([node('a')]);
    store().setEdges([]);
    const record = _buildPersistedTabForTesting(activeTab());
    expect(record.revision).toBe(activeTab().revision);

    const restored = _tabFromPersistedForTesting(record, activeTab());
    expect(restored.revision).toBe(record.revision);
  });

  it('a record written before this feature restores as revision 1', async () => {
    const { _buildPersistedTabForTesting, _tabFromPersistedForTesting } = await import('./tabStore');
    const record = _buildPersistedTabForTesting(activeTab());
    delete (record as { revision?: number }).revision;
    const restored = _tabFromPersistedForTesting(record, activeTab());
    expect(restored.revision).toBe(1);
  });
});
