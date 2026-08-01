import type { Edge, Node } from '@xyflow/react';
import type { NodeData, NodeDefinition } from '../types';

/**
 * A deterministic synthetic graph, big enough to make the canvas hurt (#125).
 *
 * Shared by the perf harness and anything else that needs a realistic large
 * graph. "Realistic" matters: a node whose `data` is `{}` costs nothing to
 * copy or serialize, so a graph of those would flatter every measurement.
 * Each node here carries a definition with ports, a handful of params and a
 * label — roughly what a Linear/Conv layer looks like on a real canvas.
 */

const CATEGORIES = ['Layers', 'Data', 'Training', 'Utility', 'Metrics'];

/** Nodes are laid out in a grid so positions are spread, not stacked. */
const COLUMNS = 20;
const COLUMN_WIDTH = 260;
const ROW_HEIGHT = 140;

function makeDefinition(index: number): NodeDefinition {
  return {
    node_name: `Synth${index % 12}`,
    category: CATEGORIES[index % CATEGORIES.length],
    description: 'Synthetic node used by the canvas perf harness.',
    inputs: [
      { name: 'input', data_type: 'TENSOR', description: 'in', optional: false },
      { name: 'aux', data_type: 'NUMBER', description: 'aux', optional: true },
    ],
    outputs: [
      { name: 'output', data_type: 'TENSOR', description: 'out', optional: false },
    ],
    params: [
      {
        name: 'units',
        param_type: 'int',
        default: 64,
        description: 'width',
        options: [],
        min_value: 1,
        max_value: 4096,
      },
      {
        name: 'activation',
        param_type: 'select',
        default: 'relu',
        description: 'activation',
        options: ['relu', 'gelu', 'tanh'],
        min_value: null,
        max_value: null,
      },
    ],
  };
}

/** `count` nodes on a grid, each with a full definition and params. */
export function syntheticNodes(count: number): Node<NodeData>[] {
  const nodes: Node<NodeData>[] = [];
  for (let i = 0; i < count; i += 1) {
    const definition = makeDefinition(i);
    nodes.push({
      id: `perf-n${i}`,
      type: 'baseNode',
      position: {
        x: (i % COLUMNS) * COLUMN_WIDTH,
        y: Math.floor(i / COLUMNS) * ROW_HEIGHT,
      },
      data: {
        label: `${definition.node_name} ${i}`,
        type: definition.node_name,
        params: { units: 32 + (i % 8) * 32, activation: 'relu' },
        definition,
        executionStatus: 'idle',
      },
    });
  }
  return nodes;
}

/**
 * `count` edges over `nodes`, deterministic and acyclic.
 *
 * Each node feeds the next, plus a skip connection every third node, which is
 * what pushes the count past one-per-node without inventing random topology.
 */
export function syntheticEdges(nodes: Node<NodeData>[], count: number): Edge[] {
  const edges: Edge[] = [];
  for (let i = 0; edges.length < count && i < nodes.length - 1; i += 1) {
    edges.push({
      id: `perf-e${edges.length}`,
      source: nodes[i].id,
      target: nodes[i + 1].id,
      sourceHandle: 'output',
      targetHandle: 'input',
      style: { stroke: '#555', strokeWidth: 2 },
    });
    if (i % 3 === 0 && i + 4 < nodes.length && edges.length < count) {
      edges.push({
        id: `perf-e${edges.length}`,
        source: nodes[i].id,
        target: nodes[i + 4].id,
        sourceHandle: 'output',
        targetHandle: 'aux',
        style: { stroke: '#555', strokeWidth: 2 },
      });
    }
  }
  return edges;
}

/** The harness default: 300 nodes, 400 edges. */
export function syntheticGraph(nodeCount = 300, edgeCount = 400) {
  const nodes = syntheticNodes(nodeCount);
  return { nodes, edges: syntheticEdges(nodes, edgeCount) };
}
