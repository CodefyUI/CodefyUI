import { describe, it, expect } from 'vitest';
import type { Edge, Node } from '@xyflow/react';
import type { NodeData, NodeDefinition } from '../../types';
import { portDataType, resolveInputSources, resolveSingleNodePorts } from './portCaptures';

function def(outputs: { name: string; data_type: string }[]): NodeDefinition {
  return {
    node_name: 'N',
    category: 'c',
    description: '',
    inputs: [],
    outputs: outputs.map((o) => ({ ...o, description: '', optional: false })),
    params: [],
  };
}

function node(
  id: string,
  over: { label?: string; definition?: NodeDefinition } = {},
): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position: { x: 0, y: 0 },
    data: {
      label: over.label ?? id,
      type: 'N',
      params: {},
      ...(over.definition !== undefined ? { definition: over.definition } : {}),
    },
  };
}

function edge(
  id: string,
  source: string,
  target: string,
  over: Partial<Edge> = {},
): Edge {
  return { id, source, target, sourceHandle: 'out', ...over } as Edge;
}

describe('portDataType', () => {
  it('reads the declared type off the source node definition', () => {
    const nodes = [node('a', { definition: def([{ name: 'out', data_type: 'TENSOR' }]) })];
    expect(portDataType(nodes, 'a', 'out')).toBe('TENSOR');
  });

  it('returns undefined for an unknown node, port, or a node with no definition', () => {
    const nodes = [
      node('a', { definition: def([{ name: 'out', data_type: 'TENSOR' }]) }),
      node('b'),
    ];
    expect(portDataType(nodes, 'ghost', 'out')).toBeUndefined();
    expect(portDataType(nodes, 'a', 'nope')).toBeUndefined();
    expect(portDataType(nodes, 'b', 'out')).toBeUndefined();
  });
});

describe('resolveInputSources', () => {
  it('collects the connected upstream (node, port) pairs', () => {
    const edges = [edge('e1', 's1', 'target', { sourceHandle: 'x' }), edge('e2', 's2', 'other')];
    expect(resolveInputSources('target', edges)).toEqual([{ nodeId: 's1', port: 'x' }]);
  });

  it('skips trigger edges in both shapes, and edges naming no source port', () => {
    const edges = [
      edge('e1', 's1', 'target', { type: 'triggerEdge' }),
      edge('e2', 's2', 'target', { data: { type: 'trigger' } }),
      edge('e3', 's3', 'target', { sourceHandle: null }),
    ];
    expect(resolveInputSources('target', edges)).toEqual([]);
  });
});

describe('resolveSingleNodePorts', () => {
  it('labels inputs with their provenance and lists the node’s own outputs', () => {
    const nodes = [
      node('n1', { definition: def([{ name: 'logits', data_type: 'TENSOR' }]) }),
      node('src', { label: 'Source', definition: def([{ name: 'y', data_type: 'SCALAR' }]) }),
    ];
    const edges = [edge('e1', 'src', 'n1', { sourceHandle: 'y' })];
    expect(resolveSingleNodePorts('n1', nodes, edges)).toEqual({
      inputs: [{ nodeId: 'src', port: 'y', displayName: 'Source.y', dataType: 'SCALAR' }],
      outputs: [{ nodeId: 'n1', port: 'logits', dataType: 'TENSOR' }],
    });
  });

  it('returns nothing for a node that is not on the canvas', () => {
    expect(resolveSingleNodePorts('ghost', [], [])).toEqual({ inputs: [], outputs: [] });
  });

  it('falls back to a truncated id when the upstream node has no usable label', () => {
    // Source is absent from `nodes` entirely, and its label would be blank
    // anyway — either way the row still names where the value came from.
    const nodes = [node('n1', { definition: def([]) })];
    const edges = [edge('e1', 'sourcenode123', 'n1', { sourceHandle: 'y' })];
    expect(resolveSingleNodePorts('n1', nodes, edges).inputs).toEqual([
      { nodeId: 'sourcenode123', port: 'y', displayName: 'source.y', dataType: undefined },
    ]);
  });

  it('treats a node with no definition as having no outputs', () => {
    const nodes = [node('n1')];
    expect(resolveSingleNodePorts('n1', nodes, [])).toEqual({ inputs: [], outputs: [] });
  });
});
