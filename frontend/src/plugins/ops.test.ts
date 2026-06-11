import { describe, it, expect } from 'vitest';
import type { Node, Edge } from '@xyflow/react';
import type { NodeData, NodeDefinition } from '../types';
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

function run(ops: GraphOp[], nodes: Node<NodeData>[] = [], edges: Edge[] = []) {
  return applyGraphOps({ nodes, edges }, DEFS, ops);
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
