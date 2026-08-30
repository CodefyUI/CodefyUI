import { describe, it, expect } from 'vitest';
import type { Node, Edge } from '@xyflow/react';
import type { NodeData, NodeDefinition, SegmentGroup } from '../types';
import { applyGraphOps, type GraphOp } from './ops';
import { buildFlowNode } from '../utils';

function def(name: string, overrides: Partial<NodeDefinition> = {}): NodeDefinition {
  return {
    node_name: name,
    category: 'Layer',
    description: '',
    inputs: [],
    outputs: [],
    params: [],
    ...overrides,
  };
}

const DEFS: NodeDefinition[] = [
  def('Source', {
    outputs: [{ name: 'out', data_type: 'TENSOR', description: '', optional: false }],
    params: [
      { name: 'size', param_type: 'int', default: 8, description: '', options: [], min_value: 1, max_value: 64 },
      { name: 'mode', param_type: 'select', default: 'a', description: '', options: ['a', 'b'], min_value: null, max_value: null },
    ],
  }),
  def('Sink', {
    inputs: [{ name: 'x', data_type: 'TENSOR', description: '', optional: false }],
  }),
  def('ModelSink', {
    inputs: [{ name: 'm', data_type: 'MODEL', description: '', optional: false }],
  }),
];

function run(
  ops: GraphOp[],
  nodes: Node<NodeData>[] = [],
  edges: Edge[] = [],
  segmentGroups: SegmentGroup[] = [],
) {
  return applyGraphOps({ nodes, edges, segmentGroups }, DEFS, ops);
}

describe('applyGraphOps — add_node', () => {
  it('adds a node with defaults and returns its id', () => {
    const r = run([{ op: 'add_node', node_type: 'Source', ref: 's' }]);
    expect(r.results[0]).toMatchObject({ ok: true });
    expect(r.nodes).toHaveLength(1);
    expect(r.refs.s).toBe(r.nodes[0].id);
    expect(r.nodes[0].data.params).toEqual({ size: 8, mode: 'a' });
    expect(r.mutated).toBe(true);
  });

  it('applies provided params and position', () => {
    const r = run([{ op: 'add_node', node_type: 'Source', params: { size: 16 }, position: { x: 5, y: 6 } }]);
    expect(r.nodes[0].data.params.size).toBe(16);
    expect(r.nodes[0].position).toEqual({ x: 5, y: 6 });
  });

  it('fails on unknown node type without adding', () => {
    const r = run([{ op: 'add_node', node_type: 'Nope' }]);
    expect(r.results[0].ok).toBe(false);
    expect(r.results[0].error).toContain('Unknown node type');
    expect(r.nodes).toHaveLength(0);
    expect(r.mutated).toBe(false);
  });

  it('fails on bad params (unknown name, range, options, type)', () => {
    const cases: Array<Record<string, unknown>> = [
      { ghost: 1 },
      { size: 0 },
      { size: 999 },
      { size: 'big' },
      { mode: 'z' },
    ];
    for (const params of cases) {
      const r = run([{ op: 'add_node', node_type: 'Source', params }]);
      expect(r.results[0].ok).toBe(false);
      expect(r.nodes).toHaveLength(0);
    }
  });
});

describe('applyGraphOps — connect', () => {
  it('connects two refs created in the same batch', () => {
    const r = run([
      { op: 'add_node', node_type: 'Source', ref: 'a' },
      { op: 'add_node', node_type: 'Sink', ref: 'b' },
      { op: 'connect', source: 'a', source_handle: 'out', target: 'b', target_handle: 'x' },
    ]);
    expect(r.results.map((x) => x.ok)).toEqual([true, true, true]);
    expect(r.edges).toHaveLength(1);
    expect(r.edges[0]).toMatchObject({ source: r.refs.a, target: r.refs.b, sourceHandle: 'out', targetHandle: 'x' });
  });

  it('rejects type-incompatible connections', () => {
    const r = run([
      { op: 'add_node', node_type: 'Source', ref: 'a' },
      { op: 'add_node', node_type: 'ModelSink', ref: 'b' },
      { op: 'connect', source: 'a', source_handle: 'out', target: 'b', target_handle: 'm' },
    ]);
    expect(r.results[2].ok).toBe(false);
    expect(r.results[2].error).toMatch(/TENSOR.*MODEL|incompatible/i);
    expect(r.edges).toHaveLength(0);
  });

  it('rejects unknown nodes, unknown ports, and duplicates', () => {
    const base: GraphOp[] = [
      { op: 'add_node', node_type: 'Source', ref: 'a' },
      { op: 'add_node', node_type: 'Sink', ref: 'b' },
    ];
    const dup = run([
      ...base,
      { op: 'connect', source: 'a', source_handle: 'out', target: 'b', target_handle: 'x' },
      { op: 'connect', source: 'a', source_handle: 'out', target: 'b', target_handle: 'x' },
    ]);
    expect(dup.results[3].ok).toBe(false);
    expect(dup.edges).toHaveLength(1);

    const ghost = run([{ op: 'connect', source: 'nope', source_handle: 'out', target: 'nope2', target_handle: 'x' }]);
    expect(ghost.results[0].ok).toBe(false);

    const badPort = run([
      ...base,
      { op: 'connect', source: 'a', source_handle: 'ghost', target: 'b', target_handle: 'x' },
    ]);
    expect(badPort.results[2].ok).toBe(false);
    expect(badPort.results[2].error).toContain('ghost');
  });
});

describe('applyGraphOps — set_params / remove_node / remove_edge / clear / layout', () => {
  function seeded() {
    const a = buildFlowNode(DEFS[0], { x: 0, y: 0 });
    const b = buildFlowNode(DEFS[1], { x: 100, y: 0 });
    const e: Edge = { id: 'e1', source: a.id, target: b.id, sourceHandle: 'out', targetHandle: 'x' };
    return { nodes: [a, b], edges: [e], a, b };
  }

  it('set_params merges valid values and reports invalid ones', () => {
    const { nodes, edges, a } = seeded();
    const ok = run([{ op: 'set_params', node_id: a.id, params: { size: 32 } }], nodes, edges);
    expect(ok.results[0].ok).toBe(true);
    expect(ok.nodes.find((n) => n.id === a.id)!.data.params.size).toBe(32);

    const bad = run([{ op: 'set_params', node_id: a.id, params: { size: -1 } }], nodes, edges);
    expect(bad.results[0].ok).toBe(false);
  });

  it('remove_node drops the node and its edges', () => {
    const { nodes, edges, a } = seeded();
    const r = run([{ op: 'remove_node', node_id: a.id }], nodes, edges);
    expect(r.results[0].ok).toBe(true);
    expect(r.nodes).toHaveLength(1);
    expect(r.edges).toHaveLength(0);
  });

  it('remove_edge matches by endpoints (handles optional)', () => {
    const { nodes, edges, a, b } = seeded();
    const r = run([{ op: 'remove_edge', source: a.id, target: b.id }], nodes, edges);
    expect(r.results[0].ok).toBe(true);
    expect(r.edges).toHaveLength(0);

    const miss = run([{ op: 'remove_edge', source: b.id, target: a.id }], nodes, edges);
    expect(miss.results[0].ok).toBe(false);
  });

  it('clear_graph empties everything; auto_layout repositions', () => {
    const { nodes, edges } = seeded();
    const cleared = run([{ op: 'clear_graph' }], nodes, edges);
    expect(cleared.nodes).toHaveLength(0);
    expect(cleared.edges).toHaveLength(0);

    const laid = run([{ op: 'auto_layout' }], nodes, edges);
    expect(laid.results[0].ok).toBe(true);
    expect(laid.nodes).toHaveLength(2);
  });
});

describe('applyGraphOps — move_node', () => {
  function seeded() {
    const a = buildFlowNode(DEFS[0], { x: 0, y: 0 });
    const note: Node<NodeData> = {
      id: 'note1', type: 'noteNode', position: { x: 30, y: 40 },
      data: {
        label: 'Note', type: 'note', params: {},
        noteKind: 'text', noteContent: 'hi', noteColor: '#3d3d1a',
        boundToNodeId: a.id, boundOffset: { x: 30, y: 40 },
        noteWidth: 200,
      },
    };
    const loose: Node<NodeData> = { ...note, id: 'note2', data: { ...note.data, boundToNodeId: null, boundOffset: null } };
    return { nodes: [a, note, loose], a };
  }

  it('moves a node to an exact position', () => {
    const { nodes, a } = seeded();
    const r = run([{ op: 'move_node', node_id: a.id, position: { x: 500, y: 250 } }], nodes);
    expect(r.results[0]).toMatchObject({ ok: true, node_id: a.id });
    expect(r.nodes.find((n) => n.id === a.id)!.position).toEqual({ x: 500, y: 250 });
    expect(r.mutated).toBe(true);
  });

  it('carries a bound note along by the same delta, and leaves a loose one', () => {
    const { nodes, a } = seeded();
    const r = run([{ op: 'move_node', node_id: a.id, position: { x: 100, y: 60 } }], nodes);
    expect(r.nodes.find((n) => n.id === 'note1')!.position).toEqual({ x: 130, y: 100 });
    expect(r.nodes.find((n) => n.id === 'note2')!.position).toEqual({ x: 30, y: 40 });
  });

  it('rejects an unknown node and a non-finite position', () => {
    const { nodes, a } = seeded();
    const ghost = run([{ op: 'move_node', node_id: 'nope', position: { x: 0, y: 0 } }], nodes);
    expect(ghost.results[0].ok).toBe(false);
    expect(ghost.mutated).toBe(false);

    const nan = run([{ op: 'move_node', node_id: a.id, position: { x: Number.NaN, y: 0 } }], nodes);
    expect(nan.results[0].ok).toBe(false);
    expect(nan.results[0].error).toContain('finite');
  });

  it('accepts a same-batch ref', () => {
    const r = run([
      { op: 'add_node', node_type: 'Source', ref: 's' },
      { op: 'move_node', node_id: 's', position: { x: 12, y: 34 } },
    ]);
    expect(r.results[1].ok).toBe(true);
    expect(r.nodes[0].position).toEqual({ x: 12, y: 34 });
  });

  it('re-derives the offset when the moved node IS the bound note', () => {
    const { nodes, a } = seeded();
    const r = run([{ op: 'move_node', node_id: 'note1', position: { x: 200, y: 100 } }], nodes);
    const note = r.nodes.find((n) => n.id === 'note1')!;
    expect(note.position).toEqual({ x: 200, y: 100 });
    // `onNodesChange`'s FIRST pass: the offset follows the note that was moved.
    expect(note.data.boundOffset).toEqual({ x: 200, y: 100 });

    // The SECOND pass, replayed by hand -- what the store does on the next
    // drag of the parent. A stale offset would snap the note back to where it
    // sat before the plugin moved it, reading as if the edit had been undone.
    const parent = r.nodes.find((n) => n.id === a.id)!;
    const dragged = { x: parent.position.x + 40, y: parent.position.y + 15 };
    const rederived = {
      x: dragged.x + note.data.boundOffset!.x,
      y: dragged.y + note.data.boundOffset!.y,
    };
    expect(rederived).toEqual({ x: 240, y: 115 });
  });

  it('leaves a plain node\'s data object untouched', () => {
    const { nodes, a } = seeded();
    const before = nodes.find((n) => n.id === a.id)!.data;
    const r = run([{ op: 'move_node', node_id: a.id, position: { x: 7, y: 8 } }], nodes);
    expect(r.nodes.find((n) => n.id === a.id)!.data).toBe(before);
  });
});

describe('applyGraphOps — set_segment / remove_segment', () => {
  function chain() {
    const a = buildFlowNode(DEFS[0], { x: 0, y: 0 });
    const b = buildFlowNode(DEFS[1], { x: 100, y: 0 });
    const e: Edge = { id: 'e1', source: a.id, target: b.id, sourceHandle: 'out', targetHandle: 'x' };
    return { nodes: [a, b], edges: [e], a, b };
  }

  it('appends a segment and reports the id it generated', () => {
    const { nodes, edges, a, b } = chain();
    const r = run([{ op: 'set_segment', head_node_id: a.id, tail_node_id: b.id }], nodes, edges);
    expect(r.results[0].ok).toBe(true);
    expect(typeof r.results[0].segment_id).toBe('string');
    expect(r.segmentGroups).toEqual([
      { id: r.results[0].segment_id, headNodeId: a.id, tailNodeId: b.id },
    ]);
  });

  it('replaces by id rather than duplicating', () => {
    const { nodes, edges, a, b } = chain();
    const r = run(
      [{ op: 'set_segment', segment_id: 's1', head_node_id: b.id, tail_node_id: b.id }],
      nodes, edges,
      [{ id: 's1', headNodeId: a.id, tailNodeId: b.id }],
    );
    expect(r.segmentGroups).toHaveLength(1);
    expect(r.segmentGroups[0]).toEqual({ id: 's1', headNodeId: b.id, tailNodeId: b.id });
  });

  it('refuses head and tail with no data-edge path between them', () => {
    const { nodes, edges, a, b } = chain();
    const backwards = run(
      [{ op: 'set_segment', head_node_id: b.id, tail_node_id: a.id }],
      nodes, edges,
    );
    expect(backwards.results[0].ok).toBe(false);
    expect(backwards.results[0].error).toContain('no data-edge path');
    expect(backwards.segmentGroups).toHaveLength(0);
    expect(backwards.mutated).toBe(false);
  });

  it('refuses a note as an endpoint and an unknown node', () => {
    const { nodes, edges, a } = chain();
    const note: Node<NodeData> = {
      id: 'note1', type: 'noteNode', position: { x: 0, y: 0 },
      data: { label: 'Note', type: 'note', params: {}, noteKind: 'text', noteContent: 'x', noteColor: '#3d3d1a', boundToNodeId: null, boundOffset: null, noteWidth: 200 },
    };
    const withNote = run(
      [{ op: 'set_segment', head_node_id: a.id, tail_node_id: 'note1' }],
      [...nodes, note], edges,
    );
    expect(withNote.results[0].ok).toBe(false);
    expect(withNote.results[0].error).toContain('note');

    const ghost = run([{ op: 'set_segment', head_node_id: a.id, tail_node_id: 'nope' }], nodes, edges);
    expect(ghost.results[0].ok).toBe(false);
  });

  it('remove_segment drops a known id and reports an unknown one', () => {
    const { nodes, edges, a, b } = chain();
    const groups: SegmentGroup[] = [{ id: 's1', headNodeId: a.id, tailNodeId: b.id }];
    const ok = run([{ op: 'remove_segment', segment_id: 's1' }], nodes, edges, groups);
    expect(ok.results[0].ok).toBe(true);
    expect(ok.segmentGroups).toHaveLength(0);

    const miss = run([{ op: 'remove_segment', segment_id: 'nope' }], nodes, edges, groups);
    expect(miss.results[0].ok).toBe(false);
    expect(miss.mutated).toBe(false);
  });

  it('hands back the SAME segmentGroups array when no op touched it', () => {
    // The persistence record cache compares this array by reference, so a
    // batch of add_node that handed back a fresh (identical) segment list
    // would rewrite the tab's IndexedDB record for nothing.
    const { nodes, edges, a, b } = chain();
    const groups: SegmentGroup[] = [{ id: 's1', headNodeId: a.id, tailNodeId: b.id }];
    const r = run([{ op: 'add_node', node_type: 'Source' }], nodes, edges, groups);
    expect(r.segmentGroups).toBe(groups);
  });

  it('remove_node drops the segments that named it, and only those', () => {
    const { nodes, edges, a, b } = chain();
    const groups: SegmentGroup[] = [
      { id: 's1', headNodeId: a.id, tailNodeId: b.id },
      { id: 's2', headNodeId: b.id, tailNodeId: b.id },
    ];
    const head = run([{ op: 'remove_node', node_id: a.id }], nodes, edges, groups);
    expect(head.results[0].ok).toBe(true);
    expect(head.segmentGroups).toEqual([{ id: 's2', headNodeId: b.id, tailNodeId: b.id }]);

    // A delete that names no endpoint leaves the list alone -- by reference,
    // so the autosave record is not rewritten for a list that did not move.
    const note: Node<NodeData> = {
      id: 'note1', type: 'noteNode', position: { x: 0, y: 0 },
      data: { label: 'Note', type: 'note', params: {}, noteKind: 'text', noteContent: 'x', noteColor: '#3d3d1a', boundToNodeId: null, boundOffset: null, noteWidth: 200 },
    };
    const other = run([{ op: 'remove_node', node_id: 'note1' }], [...nodes, note], edges, groups);
    expect(other.segmentGroups).toBe(groups);
  });

  it('clear_graph empties the segment list too', () => {
    const { nodes, edges, a, b } = chain();
    const groups: SegmentGroup[] = [{ id: 's1', headNodeId: a.id, tailNodeId: b.id }];
    const r = run([{ op: 'clear_graph' }], nodes, edges, groups);
    expect(r.results[0].ok).toBe(true);
    expect(r.segmentGroups).toEqual([]);

    const empty: SegmentGroup[] = [];
    expect(run([{ op: 'clear_graph' }], nodes, edges, empty).segmentGroups).toBe(empty);
  });
});
