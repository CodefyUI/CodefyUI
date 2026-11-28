import { describe, it, expect } from 'vitest';
import type { Edge, Node } from '@xyflow/react';
import type { NodeData, NodeDefinition, PortDefinition } from '../types';
import { applyGraphOps, type GraphOp } from './ops';
import { resolveDynamicInputs, resolveDynamicOutputs } from '../utils';
import { definitionsWithDynamicPorts, dynamicPortsOf, livePorts } from '../utils/livePorts';

// Round 6 H1: the plugin `connect` checks a node instance's LIVE ports --
// the ones its params give it, the ones the canvas draws -- not the static
// definition. Measured with a real model: `add_node PythonScript
// {output_ports: 2}` then `connect source_handle: "out2"` was refused with
// "'PythonScript' has no output 'out2'; outputs: out1".

const port = (name: string, data_type = 'ANY'): PortDefinition =>
  ({ name, data_type, description: '', optional: false });

const intParam = (name: string, value: number, max: number) => ({
  name, param_type: 'int' as const, default: value, description: '', options: [], min_value: 1, max_value: max,
});
const strParam = (name: string) => ({
  name, param_type: 'string' as const, default: '', description: '', options: [], min_value: null, max_value: null,
});

const DEFS: NodeDefinition[] = [
  {
    node_name: 'PythonScript', category: 'Utility', description: '',
    inputs: [port('in1', 'TENSOR')], outputs: [port('out1')],
    params: [intParam('input_ports', 1, 8), intParam('output_ports', 1, 8),
      strParam('input_types'), strParam('output_types'), strParam('code')],
  },
  {
    node_name: 'Split', category: 'Tensor', description: '',
    inputs: [port('tensor', 'TENSOR')], outputs: [port('chunk_0', 'TENSOR'), port('chunk_1', 'TENSOR')],
    params: [intParam('chunks', 2, 32)],
  },
  {
    node_name: 'Switch', category: 'Control', description: '',
    inputs: [port('input_0'), port('input_1'), port('input_2'), port('input_3'), port('selector', 'SCALAR')],
    outputs: [port('output')],
    params: [intParam('inputs', 4, 16)],
  },
  {
    node_name: 'Sink', category: 'Utility', description: '',
    inputs: [port('x', 'TENSOR')], outputs: [], params: [],
  },
];

function run(ops: GraphOp[], nodes: Node<NodeData>[] = [], edges: Edge[] = []) {
  return applyGraphOps({ nodes, edges, segmentGroups: [] }, DEFS, ops);
}

describe('connect checks the ports the node has now', () => {
  it('wires out2 of a PythonScript added with two outputs in the same batch', () => {
    const r = run([
      { op: 'add_node', node_type: 'PythonScript', ref: 'py', params: { output_ports: 2 } },
      { op: 'add_node', node_type: 'PythonScript', ref: 'sink', params: { input_ports: 2 } },
      { op: 'connect', source: 'py', source_handle: 'out2', target: 'sink', target_handle: 'in2' },
    ]);
    expect(r.results.map((x) => x.ok)).toEqual([true, true, true]);
    expect(r.edges[0]).toMatchObject({ sourceHandle: 'out2', targetHandle: 'in2' });
  });

  it('wires a port a set_params earlier in the batch created', () => {
    const r = run([
      { op: 'add_node', node_type: 'PythonScript', ref: 'py' },
      { op: 'set_params', node_id: 'py', params: { output_ports: 3 } },
      { op: 'add_node', node_type: 'Sink', ref: 's' },
      { op: 'connect', source: 'py', source_handle: 'out3', target: 's', target_handle: 'x' },
    ]);
    expect(r.results.every((x) => x.ok)).toBe(true);
  });

  it('still refuses a port the node does not have, listing the live ones', () => {
    const r = run([
      { op: 'add_node', node_type: 'PythonScript', ref: 'py', params: { output_ports: 2 } },
      { op: 'add_node', node_type: 'Sink', ref: 's' },
      { op: 'connect', source: 'py', source_handle: 'out3', target: 's', target_handle: 'x' },
    ]);
    expect(r.results[2].ok).toBe(false);
    expect(r.results[2].error).toBe("connect: 'PythonScript' has no output 'out3'; outputs: out1, out2");
  });

  it('type-checks against the live port type', () => {
    const r = run([
      { op: 'add_node', node_type: 'PythonScript', ref: 'py', params: { output_ports: 2, output_types: 'ANY, MODEL' } },
      { op: 'add_node', node_type: 'Sink', ref: 's' },
      { op: 'connect', source: 'py', source_handle: 'out2', target: 's', target_handle: 'x' },
    ]);
    expect(r.results[2].ok).toBe(false);
    expect(r.results[2].error).toBe('connect: incompatible types MODEL -> TENSOR');
  });

  it('wires a Split chunk past the default two', () => {
    const r = run([
      { op: 'add_node', node_type: 'Split', ref: 'sp', params: { chunks: 4 } },
      { op: 'add_node', node_type: 'Sink', ref: 's' },
      { op: 'connect', source: 'sp', source_handle: 'chunk_3', target: 's', target_handle: 'x' },
    ]);
    expect(r.results[2].ok).toBe(true);
  });

  it("grows a Switch when its last input is wired, as the canvas does", () => {
    const r = run([
      { op: 'add_node', node_type: 'PythonScript', ref: 'py' },
      { op: 'add_node', node_type: 'Switch', ref: 'sw', params: { inputs: 2 } },
      { op: 'connect', source: 'py', source_handle: 'out1', target: 'sw', target_handle: 'input_1' },
      { op: 'connect', source: 'py', source_handle: 'out1', target: 'sw', target_handle: 'input_2' },
    ]);
    expect(r.results.map((x) => x.ok)).toEqual([true, true, true, true]);
    const sw = r.nodes.find((n) => n.id === r.refs.sw)!;
    expect(sw.data.params.inputs).toBe(4);
  });
});

describe('set_params that removes ports drops their wires, as the canvas does', () => {
  it('lowering output_ports removes the wire on out2 and names it', () => {
    const built = run([
      { op: 'add_node', node_type: 'PythonScript', ref: 'py', params: { output_ports: 2 } },
      { op: 'add_node', node_type: 'Sink', ref: 'a' },
      { op: 'add_node', node_type: 'Sink', ref: 'b' },
      { op: 'connect', source: 'py', source_handle: 'out1', target: 'a', target_handle: 'x' },
      { op: 'connect', source: 'py', source_handle: 'out2', target: 'b', target_handle: 'x' },
    ]);
    const out2 = built.edges.find((e) => e.sourceHandle === 'out2')!;
    const r = run([{ op: 'set_params', node_id: built.refs.py, params: { output_ports: 1 } }],
      built.nodes, built.edges);
    expect(r.results[0]).toMatchObject({ ok: true, removed_edge_ids: [out2.id] });
    expect(r.edges.map((e) => e.sourceHandle)).toEqual(['out1']);
    // The consumer that lost its input is dirty too.
    expect(r.dirtyIds).toEqual(expect.arrayContaining([built.refs.py, built.refs.b]));
  });

  it('a param change that keeps every port removes nothing', () => {
    const built = run([
      { op: 'add_node', node_type: 'PythonScript', ref: 'py', params: { output_ports: 2 } },
      { op: 'add_node', node_type: 'Sink', ref: 'a' },
      { op: 'connect', source: 'py', source_handle: 'out1', target: 'a', target_handle: 'x' },
    ]);
    const r = run([{ op: 'set_params', node_id: built.refs.py, params: { code: 'x = 1' } }],
      built.nodes, built.edges);
    expect(r.results[0].removed_edge_ids).toBeUndefined();
    expect(r.edges).toHaveLength(1);
  });
});

describe('getNodeDefinitions says which ports are dynamic', () => {
  it('adds dynamic_ports to those definitions only, leaving the rest as they are', () => {
    const out = definitionsWithDynamicPorts(DEFS);
    expect(out.find((d) => d.node_name === 'Sink')).toBe(DEFS[3]);
    expect(out.find((d) => d.node_name === 'PythonScript')!.dynamic_ports!.outputs)
      .toMatchObject({ count_param: 'output_ports', prefix: 'out', first_index: 1 });
    expect(definitionsWithDynamicPorts(DEFS)).toBe(out);
  });

  it('a plugin-qualified name is described by its bare name', () => {
    expect(dynamicPortsOf({ node_name: 'pkg:Split' })?.outputs?.count_param).toBe('chunks');
  });

  // Every description against the resolvers, at every count: the catalogue
  // an agent reads cannot drift from the handles the canvas draws.
  for (const definition of DEFS) {
    const info = dynamicPortsOf(definition);
    if (!info) continue;
    for (const side of ['inputs', 'outputs'] as const) {
      const d = info[side];
      if (!d) continue;
      it(`${definition.node_name}.${side} matches the resolver for ${d.min}..${d.max}`, () => {
        for (let n = d.min; n <= d.max; n += 1) {
          const params = { [d.count_param]: n };
          const ports = side === 'inputs'
            ? resolveDynamicInputs(definition, params)
            : resolveDynamicOutputs(definition, params);
          const expected = [
            ...Array.from({ length: n }, (_, i) => `${d.prefix}${i + d.first_index}`),
            ...(d.fixed ?? []),
          ];
          expect(ports.map((p) => p.name)).toEqual(expected);
          expect(livePorts({ data: { definition, params } as unknown as NodeData })[side]
            .map((p) => p.name)).toEqual(expected);
        }
      });
    }
  }
});
