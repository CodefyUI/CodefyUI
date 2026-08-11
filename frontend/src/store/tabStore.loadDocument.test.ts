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
  });

  it('writes every field even for a document that omits them, so nothing survives from the previous graph', () => {
    store().loadGraphDocument({
      nodes: [node('old')],
      edges: [],
      subgraphs: [definition('old-blk')],
      segmentGroups: [{ id: 'stale' } as unknown as SegmentGroup],
      description: 'the previous graph',
    });
    store().setActiveSegment({ id: 'stale' } as unknown as SegmentGroup);

    // A bare document -- exactly what a hand-written example file looks like.
    store().loadGraphDocument({ nodes: [node('new')], edges: [] });

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
  });

  it('keeps the tab name when the document ships none, and adopts a trimmed one when it does', () => {
    store().loadGraphDocument({ nodes: [], edges: [] });
    expect(tab().name).toBe('Tab 1');

    store().loadGraphDocument({ nodes: [], edges: [], name: '  My Model  ' });
    expect(tab().name).toBe('My Model');

    // A blank name is not a name: it would leave the tab labelled "".
    store().loadGraphDocument({ nodes: [], edges: [], name: '   ' });
    expect(tab().name).toBe('My Model');
  });

  it('fails closed on a newer format_version: read-only, and the verdict is returned', () => {
    const tooNew = store().loadGraphDocument({ nodes: [], edges: [], formatVersion: 999 });

    expect(tooNew).toBe(true);
    expect(tab().readOnly).toBe(true);
  });

  it('clears a stale read-only flag when a current-format document is loaded', () => {
    store().loadGraphDocument({ nodes: [], edges: [], formatVersion: 999 });
    expect(tab().readOnly).toBe(true);

    const tooNew = store().loadGraphDocument({ nodes: [], edges: [], formatVersion: 1 });

    expect(tooNew).toBe(false);
    expect(tab().readOnly).toBe(false);
  });

  it('treats a missing format_version as current, not as too new', () => {
    expect(store().loadGraphDocument({ nodes: [], edges: [] })).toBe(false);
    expect(tab().readOnly).toBe(false);
  });

  it('normalizes definitions read off a file', () => {
    // Same door `setSubgraphs` normalized through: an entry missing its
    // interface must not reach the four walkers that later read one.
    store().loadGraphDocument({
      nodes: [],
      edges: [],
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

    store().loadGraphDocument({ nodes: [node('top')], edges: [], subgraphs: [definition('blk')] });

    const t = tab();
    expect(t.subgraphStack).toEqual([]);
    expect(t.nodes.map((n) => n.id)).toEqual(['top']);
    // What a save/run sees is the document, not the block that was open.
    expect(store().getSerializedGraph().nodes.map((n: { id: string }) => n.id)).toEqual(['top']);
  });
});
