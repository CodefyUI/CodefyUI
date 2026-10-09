import { describe, it, expect } from 'vitest';
import type { Edge, Node } from '@xyflow/react';
import type { NodeData, NodeDefinition } from '../types';
import { fanInInputs, withSwitchFor } from './fanIn';

const SWITCH: NodeDefinition = {
  node_name: 'Switch', category: 'Data Flow', description: '',
  inputs: [], outputs: [{ name: 'output', data_type: 'ANY', description: '', optional: false }],
  params: [
    { name: 'selector', param_type: 'int', default: 0, description: '', options: [], min_value: 0, max_value: 15 },
    { name: 'inputs', param_type: 'int', default: 4, description: '', options: [], min_value: 2, max_value: 16 },
  ],
};

function card(id: string, x = 0, y = 0): Node<NodeData> {
  return { id, type: 'baseNode', position: { x, y }, data: { label: id, type: 'X', params: {} } };
}
const wire = (id: string, source: string, target: string, targetHandle = 'value'): Edge => ({
  id, source, sourceHandle: 'out', target, targetHandle,
});

describe('fanInInputs', () => {
  it('lists each input with more than one wire, its wires in file order', () => {
    const edges = [wire('e1', 'a', 'sink'), wire('e2', 'b', 'other'), wire('e3', 'c', 'sink')];
    expect(fanInInputs(edges)).toEqual([
      { target: 'sink', handle: 'value', edges: [edges[0], edges[2]] },
    ]);
  });

  it('ignores triggers, a wire with no handle, and different inputs of one node', () => {
    const edges: Edge[] = [
      { id: 't1', source: 's1', target: 'sink', targetHandle: '__trigger', type: 'triggerEdge' },
      { id: 't2', source: 's2', target: 'sink', targetHandle: '__trigger' },
      { id: 't3', source: 's3', sourceHandle: 'trigger', target: 'sink', targetHandle: 'value' },
      { id: 't4', source: 's4', target: 'sink', targetHandle: 'value', data: { type: 'trigger' } },
      { id: 'n1', source: 'a', target: 'sink' },
      { id: 'n2', source: 'b', target: 'sink' },
      wire('d1', 'a', 'sink', 'left'),
      wire('d2', 'b', 'sink', 'right'),
    ];
    expect(fanInInputs(edges)).toEqual([]);
  });
});

describe('withSwitchFor', () => {
  it('routes the wires through a Switch that selects the last of them', () => {
    const nodes = [card('a'), card('b'), card('sink', 500, 80)];
    const edges = [wire('e1', 'a', 'sink'), wire('e2', 'b', 'sink'), wire('keep', 'a', 'elsewhere')];
    const [fanIn] = fanInInputs(edges);
    const next = withSwitchFor(nodes, edges, fanIn, SWITCH)!;

    const sw = next.nodes.find((n) => n.id === next.switchId)!;
    expect(sw.data.params).toEqual({ selector: 1, inputs: 4 });
    expect(sw.position).toEqual({ x: 260, y: 80 });
    const into = next.edges.filter((e) => e.target === next.switchId);
    expect(into.map((e) => [e.source, e.targetHandle])).toEqual([['a', 'input_0'], ['b', 'input_1']]);
    expect(next.edges.filter((e) => e.target === 'sink')).toEqual([
      expect.objectContaining({ source: next.switchId, sourceHandle: 'output', targetHandle: 'value' }),
    ]);
    expect(next.edges.some((e) => e.id === 'keep')).toBe(true);
    expect(next.edges.some((e) => e.id === 'e1' || e.id === 'e2')).toBe(false);
  });

  it('leaves an empty input after the wires, and places the card at the origin when the input is gone', () => {
    const edges = Array.from({ length: 5 }, (_, i) => wire(`e${i}`, `s${i}`, 'ghost'));
    const next = withSwitchFor([], edges, fanInInputs(edges)[0], SWITCH)!;
    const sw = next.nodes[0];
    expect(sw.data.params).toEqual({ selector: 4, inputs: 6 });
    expect(sw.position).toEqual({ x: 0, y: 0 });
  });

  it('declines more wires than a Switch has inputs', () => {
    const edges = Array.from({ length: 17 }, (_, i) => wire(`e${i}`, `s${i}`, 'sink'));
    expect(withSwitchFor([card('sink')], edges, fanInInputs(edges)[0], SWITCH)).toBeNull();
  });
});
