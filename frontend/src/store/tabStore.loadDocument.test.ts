/**
 * `loadGraphDocument` -- the one door a whole graph document comes through
 * (#200 items 4 and 8).
 *
 * Before it there were three readers (Toolbar's load, Toolbar's import,
 * `openExample`) each hand-sequencing setNodes/setEdges/setSubgraphs/
 * setSegmentGroups/setDescription/setTabReadOnly. Two of the three got the
 * whole sequence right; the third dropped `description` and the
 * format-version gate. These tests pin the properties that made that class
 * of bug possible: ONE update, and every field of the document written
 * whether the file carries it or not.
 *
 * #200 item 9 added the save binding (`boundFile` -> `currentGraphFile`) to
 * that set for the same reason, one field later: it was the last piece of
 * "which graph is this" still being assigned OUTSIDE the install, and the
 * reader that never assigned it opened examples onto another file's binding.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import type { Edge, Node } from '@xyflow/react';

import { useTabStore } from './tabStore';
import type { NodeData, SegmentGroup, SubgraphDefinition } from '../types';

const store = () => useTabStore.getState();
const tab = () => useTabStore.getState().getActiveTab();

function node(id: string): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position: { x: 0, y: 0 },
    data: { label: id, type: 'Add', params: {} },
  } as Node<NodeData>;
}

function edge(id: string, source: string, target: string): Edge {
  return { id, source, target };
}

function definition(id: string): SubgraphDefinition {
  return {
    id,
    name: id,
    description: '',
    nodes: [],
    edges: [],
    interface: { inputs: [], outputs: [], triggerTargets: [] },
  } as unknown as SubgraphDefinition;
}

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('Tab 1');
  localStorage.clear();
});

describe('loadGraphDocument', () => {
  it('installs the whole document in ONE store emission', () => {
    let emissions = 0;
    const unsubscribe = useTabStore.subscribe(() => {
      emissions += 1;
    });

    store().loadGraphDocument({
      nodes: [node('a')],
      edges: [edge('e1', 'a', 'a')],
      boundFile: 'doc-file',
      subgraphs: [definition('blk')],
      segmentGroups: [{ id: 's1' } as unknown as SegmentGroup],
      name: 'Doc',
      description: 'a description',
      formatVersion: 1,
    });
    unsubscribe();

    // The point of the action: no window in which the tab holds the new
    // nodes but the old definitions (or any other half-installed mix).
    expect(emissions).toBe(1);
    const t = tab();
    expect(t.nodes.map((n) => n.id)).toEqual(['a']);
    expect(t.edges.map((e) => e.id)).toEqual(['e1']);
    expect(t.subgraphs.map((d) => d.id)).toEqual(['blk']);
    expect(t.segmentGroups.map((s) => s.id)).toEqual(['s1']);
    expect(t.name).toBe('Doc');
    expect(t.description).toBe('a description');
    expect(t.readOnly).toBe(false);
    // The save target lands in the SAME emission as the graph it belongs to
    // (#200 item 9) -- there is no window in which the tab holds this
    // document and the previous graph's file.
    expect(t.currentGraphFile).toBe('doc-file');
  });

  it('writes every field even for a document that omits them, so nothing survives from the previous graph', () => {
    store().loadGraphDocument({
      nodes: [node('old')],
      edges: [],
      boundFile: 'the-previous-file',
      subgraphs: [definition('old-blk')],
      segmentGroups: [{ id: 'stale' } as unknown as SegmentGroup],
      description: 'the previous graph',
    });
    store().setActiveSegment({ id: 'stale' } as unknown as SegmentGroup);
    // Really bound, so the clear below is not passing vacuously.
    expect(tab().currentGraphFile).toBe('the-previous-file');

    // A bare document -- exactly what a hand-written example file looks like.
    store().loadGraphDocument({ nodes: [node('new')], edges: [], boundFile: null });

    const t = tab();
    expect(t.nodes.map((n) => n.id)).toEqual(['new']);
    expect(t.subgraphs).toEqual([]);
    // `description` and `segmentGroups` are both persisted through save: a
    // leftover would be written to disk as if it belonged to the new graph.
    expect(t.description).toBe('');
    expect(t.segmentGroups).toEqual([]);
    // The overlay the Teaching Inspector is pointing at names head/tail ids
    // the new graph does not have.
    expect(t.activeSegment).toBeNull();
    // The save target is the sharpest case of "written either way": the
    // previous file's binding under the new graph is what made an example
    // overwrite it on the next Save (#200 item 9).
    expect(t.currentGraphFile).toBeNull();
  });

  // -- #200 item 9: the save binding is part of installing a document --

  it('binds the tab to the file the document names, and unbinds when it names none', () => {
    store().loadGraphDocument({ nodes: [], edges: [], boundFile: 'classifier' });
    expect(tab().currentGraphFile).toBe('classifier');

    // A template/import: bound to nothing, so the next Save has to ask.
    store().loadGraphDocument({ nodes: [], edges: [], boundFile: null });
    expect(tab().currentGraphFile).toBeNull();

    // ...and back again, so this is a real write and not a one-way clear.
    store().loadGraphDocument({ nodes: [], edges: [], boundFile: 'other' });
    expect(tab().currentGraphFile).toBe('other');
  });

  it('never lets a document inherit the previous graph binding', () => {
    store().setCurrentGraphFile('foo');

    store().loadGraphDocument({ nodes: [node('example')], edges: [], boundFile: null });

    // The whole of item 9: the graph on screen is the example, so the file
    // Save would overwrite must not still be foo.
    expect(tab().currentGraphFile).toBeNull();
  });

  it('keeps the tab name when the document ships none, and adopts a trimmed one when it does', () => {
    store().loadGraphDocument({ nodes: [], edges: [], boundFile: null });
    expect(tab().name).toBe('Tab 1');

    store().loadGraphDocument({ nodes: [], edges: [], boundFile: null, name: '  My Model  ' });
    expect(tab().name).toBe('My Model');

    // A blank name is not a name: it would leave the tab labelled "".
    store().loadGraphDocument({ nodes: [], edges: [], boundFile: null, name: '   ' });
    expect(tab().name).toBe('My Model');
  });

  it('fails closed on a newer format_version: read-only, and the verdict is returned', () => {
    const tooNew = store().loadGraphDocument({ nodes: [], edges: [], boundFile: null, formatVersion: 999 });

    expect(tooNew).toBe(true);
    expect(tab().readOnly).toBe(true);
  });

  it('clears a stale read-only flag when a current-format document is loaded', () => {
    store().loadGraphDocument({ nodes: [], edges: [], boundFile: null, formatVersion: 999 });
    expect(tab().readOnly).toBe(true);

    const tooNew = store().loadGraphDocument({ nodes: [], edges: [], boundFile: null, formatVersion: 1 });

    expect(tooNew).toBe(false);
    expect(tab().readOnly).toBe(false);
  });

  it('treats a missing format_version as current, not as too new', () => {
    expect(store().loadGraphDocument({ nodes: [], edges: [], boundFile: null })).toBe(false);
    expect(tab().readOnly).toBe(false);
  });

  it('normalizes definitions read off a file', () => {
    // Same door `setSubgraphs` normalized through: an entry missing its
    // interface must not reach the four walkers that later read one.
    store().loadGraphDocument({
      nodes: [],
      edges: [],
      boundFile: null,
      subgraphs: [{ id: 'x' } as unknown as SubgraphDefinition],
    });

    const [installed] = tab().subgraphs;
    expect(installed.id).toBe('x');
    expect(installed.interface).toBeDefined();
    expect(installed.nodes).toEqual([]);
  });

  it('drops an open sub-canvas and installs the document as the top level', () => {
    // The ordering contract `setSubgraphs` documents, now enforced by
    // construction: the incoming nodes ARE the top level, so an editing
    // stack left standing would leave a dead block's insides on screen as
    // if they were the whole graph -- and the next save would write them.
    useTabStore.setState({
      tabs: useTabStore.getState().tabs.map((t) => ({
        ...t,
        nodes: [node('inside-a-block')],
        subgraphStack: [
          {
            subgraphId: 'blk',
            nodes: [node('outer')],
            edges: [],
            undoStack: [],
            redoStack: [],
            selectedNodeId: null,
            subgraphs: [definition('blk')],
          },
        ],
      })) as never,
    });

    store().loadGraphDocument({
      nodes: [node('top')],
      edges: [],
      boundFile: null,
      subgraphs: [definition('blk')],
    });

    const t = tab();
    expect(t.subgraphStack).toEqual([]);
    expect(t.nodes.map((n) => n.id)).toEqual(['top']);
    // What a save/run sees is the document, not the block that was open.
    expect(store().getSerializedGraph().nodes.map((n: { id: string }) => n.id)).toEqual(['top']);
  });
});

/**
 * The per-document UI residue (core#337).
 *
 * `clear()` nulls the selection and the three modal ids; the atomic install
 * did not, so every one of them survived an open — and a new graph that
 * happens to reuse a node id showed the previous graph's selection, popped
 * the previous graph's modal, or drew the previous graph's output summaries
 * on a node that has never run.
 *
 * Each of these seeds ONE field and opens a bare document, so a field that
 * stops being cleared fails on its own line.
 */
describe('loadGraphDocument clears the previous document UI residue', () => {
  /** Open a document that reuses the seeded node id, the way #337 describes. */
  const openAnother = () =>
    store().loadGraphDocument({ nodes: [node('same-id')], edges: [], boundFile: null });

  it('clears the selection', () => {
    store().setSelectedNodeId('same-id');
    openAnother();
    expect(tab().selectedNodeId).toBeNull();
  });

  it('closes the preset modal', () => {
    store().openPresetModal('same-id');
    openAnother();
    expect(tab().presetModalNodeId).toBeNull();
  });

  it('closes the layers editor', () => {
    store().openLayersModal('same-id');
    openAnother();
    expect(tab().layersModalNodeId).toBeNull();
  });

  it('closes the detail modal, deep link and all', () => {
    store().openNodeDetail('same-id', { tab: 'code', port: 'same-id::out' });
    openAnother();
    const t = tab();
    expect(t.nodeDetailNodeId).toBeNull();
    // The tab and port travel with the id; leaving them would aim the next
    // open of the modal at a port off the graph that just closed.
    expect(t.nodeDetailTab).toBeNull();
    expect(t.nodeDetailPort).toBeNull();
  });

  it('keeps the detail-modal request counter, which is not per-document', () => {
    // `nodeDetailRequest` is a monotonic tick the modal watches to notice a
    // SECOND deep link into the node it is already showing (#129). It names
    // no node and belongs to no document; resetting it to 0 would make the
    // next open look to that effect like nothing had happened.
    store().openNodeDetail('same-id');
    const before = tab().nodeDetailRequest;
    openAnother();
    expect(tab().nodeDetailRequest).toBe(before);
  });

  it('clears the partial-re-execution hint', () => {
    store().markDirty('same-id');
    openAnother();
    expect(tab().dirtyNodeIds.size).toBe(0);
  });

  it('clears the output summaries', () => {
    // The sharpest of the set: these are captured VALUES, and drawn on the
    // card of whatever node wears the id. Left behind, the new graph's node
    // shows the old graph's tensor.
    store().setTabOutputSummary(tab().id, 'same-id', {
      out: { type: 'tensor', shape: [2, 2] },
    });
    openAnother();
    expect(tab().outputSummaries).toEqual({});
  });
});
