/**
 * The tab-addressed half of the store, added for plugin API v5 (#341).
 *
 * Everything the editor does happens to the ACTIVE tab; everything a plugin
 * does happens to a tab it names. These are the actions that make the second
 * possible without duplicating the first -- each active-tab action now
 * delegates to its tab-addressed twin, so the two cannot drift.
 *
 * Also here: the Delete-key segment pruning fix (#341 section 5.5), because it
 * is the same "a tab's document must stay self-consistent" invariant one
 * gesture over, and an agent writing segments at volume is what surfaces it.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { Edge, Node } from '@xyflow/react';
import {
  useTabStore,
  _buildPersistedTabForTesting,
  _persistedTabsForTesting,
} from './tabStore';
import { useNodeDefStore } from './nodeDefStore';
import type { NodeData, NodeDefinition } from '../types';

vi.mock('./tabPersistence', () => ({
  readSnapshot: vi.fn(async () => null),
  writeSnapshot: vi.fn(async () => {}),
}));

const store = () => useTabStore.getState();
const activeTab = () => useTabStore.getState().getActiveTab();

function def(name: string): NodeDefinition {
  return {
    node_name: name, category: 'Layer', description: '',
    inputs: [{ name: 'in', data_type: 'TENSOR', description: '', optional: false }],
    outputs: [{ name: 'out', data_type: 'TENSOR', description: '', optional: false }],
    params: [],
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
  useNodeDefStore.setState({ definitions: [def('A')], presets: [] } as never);
});

describe('createTab', () => {
  it('returns the new id and activates by default', () => {
    const before = activeTab().id;
    const id = store().createTab({ title: 'Candidate' });
    expect(id).not.toBe(before);
    expect(store().activeTabId).toBe(id);
    expect(store().getTab(id)!.name).toBe('Candidate');
    expect(store().getTab(id)!.revision).toBe(1);
  });

  it('can create a tab WITHOUT stealing the user focus', () => {
    const before = store().activeTabId;
    const id = store().createTab({ title: 'Background', activate: false });
    expect(store().activeTabId).toBe(before);
    expect(store().getTab(id)).toBeDefined();
  });

  it('addTab still activates and still numbers unnamed tabs', () => {
    store().addTab();
    expect(activeTab().name).toBe('Tab 2');
  });
});

describe('loadGraphDocumentInto', () => {
  it('installs a document into a NAMED tab, leaving the active one alone', () => {
    store().setNodes([node('live')]);
    const target = store().createTab({ title: 'Target', activate: false });
    const activeBefore = store().activeTabId;

    const readOnly = store().loadGraphDocumentInto(target, {
      nodes: [node('a'), node('b', 200)],
      edges: [dataEdge('e1', 'a', 'b')],
      boundFile: null,
      segmentGroups: [{ id: 's1', headNodeId: 'a', tailNodeId: 'b' }],
      description: 'from a plugin',
    });

    expect(readOnly).toBe(false);
    expect(store().activeTabId).toBe(activeBefore);
    expect(activeTab().nodes.map((n) => n.id)).toEqual(['live']);
    const loaded = store().getTab(target)!;
    expect(loaded.nodes.map((n) => n.id)).toEqual(['a', 'b']);
    expect(loaded.segmentGroups).toHaveLength(1);
    expect(loaded.description).toBe('from a plugin');
    expect(loaded.revision).toBe(2);
  });

  it('returns the read-only verdict for a newer format, on the named tab', () => {
    const target = store().createTab({ title: 'Newer', activate: false });
    const readOnly = store().loadGraphDocumentInto(target, {
      nodes: [], edges: [], boundFile: null, formatVersion: 9_999,
    });
    expect(readOnly).toBe(true);
    expect(store().getTab(target)!.readOnly).toBe(true);
  });

  it('loadGraphDocument still targets the active tab', () => {
    store().createTab({ title: 'Other', activate: false });
    store().loadGraphDocument({ nodes: [node('x')], edges: [], boundFile: null });
    expect(activeTab().nodes.map((n) => n.id)).toEqual(['x']);
  });
});

describe('getSerializedGraphOf', () => {
  it('serializes a BACKGROUND tab while a block is open in the active one', () => {
    // The active tab is inside a block; the background tab must still answer
    // with its own whole graph rather than with what is on screen.
    const target = store().createTab({ title: 'Background', activate: false });
    store().loadGraphDocumentInto(target, {
      nodes: [node('bg')], edges: [], boundFile: null,
    });
    store().setNodes([node('a'), node('b', 200)]);
    // Both, not one: `checkCollapse` refuses a selection of fewer than two.
    store().setNodes(activeTab().nodes.map((n) => ({ ...n, selected: true })));
    expect(store().collapseSelectionToSubgraph('Blk').ok).toBe(true);
    const instance = activeTab().nodes.find((n) => n.data.type.startsWith('subgraph:'))!;
    expect(store().enterSubgraph(instance.id)).toBe(true);

    const graph = store().getSerializedGraphOf(store().getTab(target)!);
    expect(graph.nodes.map((n) => n.id)).toEqual(['bg']);
    // Reading a background tab must not close the block the user is in.
    expect(activeTab().subgraphStack).toHaveLength(1);
  });

  it('getSerializedGraph() still answers for the active tab', () => {
    store().setNodes([node('a')]);
    expect(store().getSerializedGraph().nodes.map((n) => n.id)).toEqual(['a']);
  });
});

describe('commitDocument + pushUndoSnapshotFor', () => {
  it('writes nodes, edges and segments in ONE transition and bumps once', () => {
    const target = store().createTab({ title: 'Target', activate: false });
    store().loadGraphDocumentInto(target, {
      nodes: [node('a'), node('b', 200)], edges: [], boundFile: null,
    });
    const before = store().getTab(target)!.revision;

    store().pushUndoSnapshotFor(target);
    store().commitDocument(target, {
      nodes: [node('a'), node('b', 200), node('c', 400)],
      edges: [dataEdge('e1', 'a', 'b')],
      segmentGroups: [{ id: 's1', headNodeId: 'a', tailNodeId: 'b' }],
      dirtyIds: ['c'],
    });

    const tab = store().getTab(target)!;
    expect(tab.revision).toBe(before + 1);
    expect(tab.nodes).toHaveLength(3);
    expect(tab.edges).toHaveLength(1);
    expect(tab.segmentGroups).toHaveLength(1);
    expect([...tab.dirtyNodeIds]).toContain('c');
    expect(tab.undoStack).toHaveLength(1);
  });

  it('clears activeSegment when the commit drops the segment it named', () => {
    const target = store().createTab({ title: 'Target', activate: false });
    store().loadGraphDocumentInto(target, {
      nodes: [node('a')], edges: [], boundFile: null,
      segmentGroups: [{ id: 's1', headNodeId: 'a', tailNodeId: 'a' }],
    });
    // `setActiveSegment` is active-tab only, and this tab is deliberately in
    // the background, so the highlight is installed directly.
    useTabStore.setState({
      tabs: store().tabs.map((t) =>
        t.id === target ? { ...t, activeSegment: { id: 's1', headNodeId: 'a', tailNodeId: 'a' } } : t,
      ),
    });

    store().commitDocument(target, {
      nodes: [node('a')], edges: [], segmentGroups: [], dirtyIds: [],
    });
    expect(store().getTab(target)!.activeSegment).toBeNull();
  });

  it('closes a detail modal the commit removed the node for', () => {
    // The same cleanup `onNodesChange`'s remove branch does (#167). A plugin's
    // `remove_node` never reaches that reducer, so without this the modal
    // points at a node that is gone -- and pops back open on its own the
    // moment an undo restores it (#341 section 4.4 step 7).
    const live = store().activeTabId;
    store().loadGraphDocumentInto(live, {
      nodes: [node('a'), node('b', 200)], edges: [], boundFile: null,
    });
    store().openNodeDetail('b');
    expect(activeTab().nodeDetailNodeId).toBe('b');
    expect(activeTab().selectedNodeId).toBe('b');

    store().commitDocument(live, {
      nodes: [node('a')], edges: [], segmentGroups: [], dirtyIds: [],
    });
    expect(activeTab().nodeDetailNodeId).toBeNull();
    expect(activeTab().selectedNodeId).toBeNull();
  });

  it('closes a viz viewer the commit removed the node for, and keeps one it kept (core#324)', () => {
    const live = store().activeTabId;
    store().loadGraphDocumentInto(live, {
      nodes: [node('a'), node('b', 200)], edges: [], boundFile: null,
    });
    store().openVizModal('b');

    store().commitDocument(live, {
      nodes: [node('a'), node('b', 200)], edges: [], segmentGroups: [], dirtyIds: [],
    });
    expect(activeTab().vizModalNodeId).toBe('b');

    store().commitDocument(live, {
      nodes: [node('a')], edges: [], segmentGroups: [], dirtyIds: [],
    });
    expect(activeTab().vizModalNodeId).toBeNull();
  });

  it('leaves both alone when the committed document still has the node', () => {
    const live = store().activeTabId;
    store().loadGraphDocumentInto(live, {
      nodes: [node('a'), node('b', 200)], edges: [], boundFile: null,
    });
    store().openNodeDetail('b');

    store().commitDocument(live, {
      nodes: [node('a'), node('b', 200), node('c', 400)],
      edges: [], segmentGroups: [], dirtyIds: [],
    });
    expect(activeTab().nodeDetailNodeId).toBe('b');
    expect(activeTab().selectedNodeId).toBe('b');
  });

  it('pushUndoSnapshotFor on a background tab does not touch the active one', () => {
    const target = store().createTab({ title: 'Target', activate: false });
    store().pushUndoSnapshotFor(target);
    expect(store().getTab(target)!.undoStack).toHaveLength(1);
    expect(activeTab().undoStack).toHaveLength(0);
  });
});

describe('setTabMeta and transient persistence', () => {
  it('sets only the keys it was given', () => {
    const id = store().createTab({ title: 'Meta', activate: false });
    store().setTabMeta(id, { readOnly: true, source: { kind: 'agent-variant', pluginId: 'p' } });
    expect(store().getTab(id)!.readOnly).toBe(true);
    expect(store().getTab(id)!.transient).toBe(false);

    store().setTabMeta(id, { transient: true });
    expect(store().getTab(id)!.readOnly).toBe(true);
    expect(store().getTab(id)!.source!.pluginId).toBe('p');
    expect(store().getTab(id)!.transient).toBe(true);
  });

  it('a transient tab is never written to storage', () => {
    const keep = store().createTab({ title: 'Keep', activate: false });
    const drop = store().createTab({ title: 'Drop', activate: false });
    store().setTabMeta(drop, { transient: true });

    const records = _persistedTabsForTesting(store().tabs);
    expect(records.map((r) => r.id)).toContain(keep);
    expect(records.map((r) => r.id)).not.toContain(drop);
  });

  it('source persists with a non-transient tab; transient itself never does', () => {
    const id = store().createTab({ title: 'Kept', activate: false });
    store().setTabMeta(id, { source: { kind: 'agent-variant', pluginId: 'p', jobId: 'j' } });
    const record = _buildPersistedTabForTesting(store().getTab(id)!);
    expect(record.source).toEqual({ kind: 'agent-variant', pluginId: 'p', jobId: 'j' });
    expect('transient' in record).toBe(false);
  });

  it('changing only the source still produces a fresh record', () => {
    // The record cache compares node/edge references plus a scalar signature;
    // setTabMeta moves neither array, so `source` has to be in the signature
    // or the tab persists without it.
    const id = store().createTab({ title: 'Cached', activate: false });
    const first = _persistedTabsForTesting(store().tabs).find((r) => r.id === id)!;
    store().setTabMeta(id, { source: { kind: 'agent-variant', pluginId: 'p' } });
    const second = _persistedTabsForTesting(store().tabs).find((r) => r.id === id)!;
    expect(second).not.toBe(first);
    expect(second.source).toEqual({ kind: 'agent-variant', pluginId: 'p' });
  });
});

describe('Delete key prunes segments (#341 section 5.5)', () => {
  it('removing a segment head through onNodesChange drops the segment', () => {
    store().setNodes([node('a'), node('b', 200)]);
    store().setEdges([dataEdge('e1', 'a', 'b')]);
    store().addSegmentGroup({ id: 's1', headNodeId: 'a', tailNodeId: 'b' });
    store().setActiveSegment({ id: 's1', headNodeId: 'a', tailNodeId: 'b' });

    store().onNodesChange([{ id: 'a', type: 'remove' }]);

    expect(activeTab().segmentGroups).toHaveLength(0);
    expect(activeTab().activeSegment).toBeNull();
    // ...and it is gone from what the next save would write.
    expect(store().getSerializedGraph().segmentGroups).toHaveLength(0);
  });

  it('leaves an unrelated segment, and its array identity, alone', () => {
    store().setNodes([node('a'), node('b', 200), node('c', 400)]);
    store().setEdges([dataEdge('e1', 'a', 'b')]);
    store().addSegmentGroup({ id: 's1', headNodeId: 'a', tailNodeId: 'b' });
    const groupsBefore = activeTab().segmentGroups;

    store().onNodesChange([{ id: 'c', type: 'remove' }]);

    expect(activeTab().segmentGroups).toBe(groupsBefore);
  });

  it('undo brings the segment back with the node', () => {
    store().setNodes([node('a'), node('b', 200)]);
    store().setEdges([dataEdge('e1', 'a', 'b')]);
    store().addSegmentGroup({ id: 's1', headNodeId: 'a', tailNodeId: 'b' });
    store().onNodesChange([{ id: 'a', type: 'remove' }]);
    store().undo();
    expect(activeTab().segmentGroups.map((s) => s.id)).toEqual(['s1']);
  });
});
