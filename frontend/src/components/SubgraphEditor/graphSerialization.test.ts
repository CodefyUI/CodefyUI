import { describe, it, expect } from 'vitest';
import type { Node, Edge } from '@xyflow/react';
import {
  isMergeType,
  detectImportFormat,
  convertWorkflowToGraphSpec,
  flowToGraphJson,
  graphToFlow,
  emptyGraph,
  autoLayoutSubgraph,
  validateGraph,
  type GraphSpec,
  type LayerNodeData,
  type WorkflowData,
} from './graphSerialization';

// ── Helpers ──────────────────────────────────────────────────────────────────

function flowNode(
  id: string,
  data: Partial<LayerNodeData>,
  extra: Partial<Node<LayerNodeData>> = {},
): Node<LayerNodeData> {
  return {
    id,
    type: 'layerNode',
    position: { x: 0, y: 0 },
    data: {
      layerType: 'Conv2d',
      params: {},
      color: '#000',
      ...data,
    },
    ...extra,
  };
}

function edge(id: string, source: string, target: string, extra: Partial<Edge> = {}): Edge {
  return { id, source, target, ...extra };
}

/** Build an Input node for validation tests. */
function inputNode(id: string, ports: { id: string; name: string }[]): Node<LayerNodeData> {
  return flowNode(id, { layerType: 'Input', ports, isBoundary: true });
}
function outputNode(id: string, ports: { id: string; name: string }[]): Node<LayerNodeData> {
  return flowNode(id, { layerType: 'Output', ports, isBoundary: true });
}

// ── isMergeType ────────────────────────────────────────────────────────────

describe('isMergeType', () => {
  it('returns true for merge layer types', () => {
    expect(isMergeType('Add')).toBe(true);
    expect(isMergeType('Concat')).toBe(true);
    expect(isMergeType('Stack')).toBe(true);
  });
  it('returns false for non-merge types', () => {
    expect(isMergeType('Conv2d')).toBe(false);
    expect(isMergeType('Input')).toBe(false);
  });
});

// ── detectImportFormat ─────────────────────────────────────────────────────

describe('detectImportFormat', () => {
  it('returns unknown for invalid JSON', () => {
    expect(detectImportFormat('{not json')).toEqual({ kind: 'unknown' });
  });

  it('detects a GraphSpec v2', () => {
    const json = JSON.stringify({ version: 2, nodes: [], edges: [] });
    expect(detectImportFormat(json)).toEqual({ kind: 'graphspec', json });
  });

  it('detects workflow-layers when a known layer node is present', () => {
    const json = JSON.stringify({
      nodes: [{ id: 'a', type: 'Conv2d' }],
      edges: [],
    });
    const res = detectImportFormat(json);
    expect(res.kind).toBe('workflow-layers');
    if (res.kind === 'workflow-layers') {
      expect(res.workflowData.nodes).toHaveLength(1);
    }
  });

  it('detects workflow-layers when an Activation node maps to a known activation', () => {
    const json = JSON.stringify({
      nodes: [{ id: 'a', type: 'Activation', data: { params: { function: 'relu' } } }],
      edges: [],
    });
    expect(detectImportFormat(json).kind).toBe('workflow-layers');
  });

  it('treats an Activation with no/unknown function as not a layer node', () => {
    // function defaults to '' which is not in ACTIVATION_MAP -> not workflow-layers.
    const json = JSON.stringify({
      nodes: [{ id: 'a', type: 'Activation', data: { params: {} } }],
      edges: [],
    });
    expect(detectImportFormat(json).kind).toBe('unknown');
  });

  it('treats a node with no type as not a layer node', () => {
    // Exercises `n.type ?? ''` falling back to ''.
    const json = JSON.stringify({ nodes: [{ id: 'a' }], edges: [] });
    expect(detectImportFormat(json).kind).toBe('unknown');
  });

  it('detects workflow-sequential when SequentialModel nodes carry a layers string', () => {
    const json = JSON.stringify({
      nodes: [
        { id: 's1', type: 'SequentialModel', data: { label: 'M1', params: { layers: '[]' } } },
        { id: 's2', type: 'SequentialModel', data: { params: { layers: '[]' } } },
      ],
      edges: [],
    });
    const res = detectImportFormat(json);
    expect(res.kind).toBe('workflow-sequential');
    if (res.kind === 'workflow-sequential') {
      expect(res.models).toHaveLength(2);
      expect(res.models[0].label).toBe('M1');
      // Falls back to nodeId when label missing.
      expect(res.models[1].label).toBe('s2');
    }
  });

  it('returns unknown for a workflow with arrays but no layer/sequential nodes', () => {
    const json = JSON.stringify({
      nodes: [{ id: 'x', type: 'SomethingElse' }],
      edges: [],
    });
    expect(detectImportFormat(json).kind).toBe('unknown');
  });

  it('returns unknown when nodes/edges are not arrays', () => {
    expect(detectImportFormat(JSON.stringify({ nodes: 1, edges: 2 })).kind).toBe('unknown');
  });

  it('ignores SequentialModel nodes whose layers param is not a string', () => {
    const json = JSON.stringify({
      nodes: [{ id: 's1', type: 'SequentialModel', data: { params: { layers: 123 } } }],
      edges: [],
    });
    expect(detectImportFormat(json).kind).toBe('unknown');
  });
});

// ── convertWorkflowToGraphSpec ─────────────────────────────────────────────

describe('convertWorkflowToGraphSpec', () => {
  it('converts a simple linear workflow with Input/Output wrappers', () => {
    const workflow: WorkflowData = {
      nodes: [
        { id: 'a', type: 'Conv2d', position: { x: 0, y: 0 }, data: { params: { in_channels: 3 } } },
        { id: 'b', type: 'ReLU' },
      ],
      edges: [{ id: 'e1', source: 'a', target: 'b' }],
    };
    const spec = convertWorkflowToGraphSpec(workflow);
    expect(spec.version).toBe(2);
    // Input + 2 layers + Output.
    const types = spec.nodes.map((n) => n.type);
    expect(types[0]).toBe('Input');
    expect(types).toContain('Conv2d');
    expect(types).toContain('ReLU');
    expect(types[types.length - 1]).toBe('Output');

    // Conv2d carries its params; ReLU has none -> params undefined.
    const conv = spec.nodes.find((n) => n.type === 'Conv2d')!;
    expect(conv.params).toEqual({ in_channels: 3 });
    const relu = spec.nodes.find((n) => n.type === 'ReLU')!;
    expect(relu.params).toBeUndefined();

    // The Conv2d preserves its position; nodes without one are undefined.
    expect(conv.position).toEqual({ x: 0, y: 0 });
  });

  it('maps Activation nodes via the activation map and strips the function param', () => {
    const workflow: WorkflowData = {
      nodes: [{ id: 'a', type: 'Activation', data: { params: { function: 'gelu', extra: 1 } } }],
      edges: [],
    };
    const spec = convertWorkflowToGraphSpec(workflow);
    const mapped = spec.nodes.find((n) => n.type === 'GELU')!;
    expect(mapped).toBeDefined();
    expect(mapped.params).toEqual({ extra: 1 });
    expect(mapped.params?.function).toBeUndefined();
  });

  it('defaults Activation function to relu when omitted', () => {
    const workflow: WorkflowData = {
      nodes: [{ id: 'a', type: 'Activation', data: { params: {} } }],
      edges: [],
    };
    const spec = convertWorkflowToGraphSpec(workflow);
    expect(spec.nodes.some((n) => n.type === 'ReLU')).toBe(true);
  });

  it('skips Activation nodes with an unmapped function', () => {
    const workflow: WorkflowData = {
      nodes: [
        { id: 'a', type: 'Activation', data: { params: { function: 'mystery' } } },
        { id: 'b', type: 'Conv2d' },
      ],
      edges: [],
    };
    const spec = convertWorkflowToGraphSpec(workflow);
    // 'a' skipped; only Conv2d among layer nodes.
    expect(spec.nodes.some((n) => n.id === 'a')).toBe(false);
    expect(spec.nodes.some((n) => n.type === 'Conv2d')).toBe(true);
  });

  it('skips nodes whose type is unknown / missing', () => {
    const workflow: WorkflowData = {
      nodes: [
        { id: 'a', type: 'NotALayer' },
        { id: 'b' as unknown as string, type: undefined as unknown as string },
        { id: 'c', type: 'Linear' },
      ],
      edges: [{ id: 'e', source: 'a', target: 'c' }],
    };
    const spec = convertWorkflowToGraphSpec(workflow);
    expect(spec.nodes.some((n) => n.id === 'a')).toBe(false);
    expect(spec.nodes.some((n) => n.id === 'b')).toBe(false);
    expect(spec.nodes.some((n) => n.type === 'Linear')).toBe(true);
    // The edge referencing the dropped node 'a' is filtered out.
    expect(spec.edges.some((e) => e.source === 'a')).toBe(false);
  });

  it('handles a workflow with no edges (single isolated layer is both root and leaf)', () => {
    const workflow: WorkflowData = {
      nodes: [{ id: 'a', type: 'Conv2d' }],
      edges: [],
    };
    const spec = convertWorkflowToGraphSpec(workflow);
    // Input -> a, a -> Output, with the single node as the chosen last leaf.
    const inputN = spec.nodes.find((n) => n.type === 'Input')!;
    const outputN = spec.nodes.find((n) => n.type === 'Output')!;
    expect(spec.edges.some((e) => e.source === inputN.id && e.target === 'a')).toBe(true);
    expect(spec.edges.some((e) => e.source === 'a' && e.target === outputN.id)).toBe(true);
  });

  it('creates extra named input ports (x, x2, ...) for multiple roots', () => {
    const workflow: WorkflowData = {
      nodes: [
        { id: 'a', type: 'Conv2d' },
        { id: 'b', type: 'Conv2d' },
        { id: 'c', type: 'Add' },
      ],
      edges: [
        { id: 'e1', source: 'a', target: 'c' },
        { id: 'e2', source: 'b', target: 'c' },
      ],
    };
    const spec = convertWorkflowToGraphSpec(workflow);
    const inputN = spec.nodes.find((n) => n.type === 'Input')!;
    expect(inputN.ports).toHaveLength(2);
    expect(inputN.ports!.map((p) => p.name)).toEqual(['x', 'x2']);
  });

  it('handles a cyclic workflow (no roots/leaves) without throwing', () => {
    // a <-> b form a cycle: neither is a root or a leaf. The internal topological
    // sort cannot drain the cycle, so the trailing "append unreached ids" path runs.
    const workflow: WorkflowData = {
      nodes: [
        { id: 'a', type: 'Conv2d' },
        { id: 'b', type: 'Conv2d' },
      ],
      edges: [
        { id: 'e1', source: 'a', target: 'b' },
        { id: 'e2', source: 'b', target: 'a' },
      ],
    };
    const spec = convertWorkflowToGraphSpec(workflow);
    expect(spec.version).toBe(2);
    // Input has no ports (no roots) and both layer nodes are still present.
    const inputN = spec.nodes.find((n) => n.type === 'Input')!;
    expect(inputN.ports).toEqual([]);
    expect(spec.nodes.some((n) => n.id === 'a')).toBe(true);
    expect(spec.nodes.some((n) => n.id === 'b')).toBe(true);
  });

  it('keeps only edges connecting two converted layer nodes (strips handles)', () => {
    const workflow: WorkflowData = {
      nodes: [
        { id: 'a', type: 'Conv2d' },
        { id: 'b', type: 'ReLU' },
      ],
      edges: [
        { id: 'e1', source: 'a', target: 'b', sourceHandle: 'h1', targetHandle: 'h2' },
        { id: 'e2', source: 'a', target: 'ghost' },
      ],
    };
    const spec = convertWorkflowToGraphSpec(workflow);
    const ab = spec.edges.find((e) => e.id === 'e1')!;
    expect(ab.sourceHandle).toBeNull();
    expect(ab.targetHandle).toBeNull();
    expect(spec.edges.some((e) => e.id === 'e2')).toBe(false);
  });
});

// ── flowToGraphJson ────────────────────────────────────────────────────────

describe('flowToGraphJson', () => {
  it('serializes nodes and edges with defaults filled in', () => {
    const nodes = [
      flowNode('n1', { layerType: 'Linear', params: { out: 10 }, ports: [{ id: 'p', name: 'x' }] }),
      // params omitted -> {} default via `?? {}`.
      flowNode('n2', { layerType: 'ReLU', params: undefined as unknown as Record<string, unknown> }),
    ];
    const edges = [
      edge('e1', 'n1', 'n2', { sourceHandle: 'sh', targetHandle: 'th' }),
      // handles omitted -> null defaults.
      edge('e2', 'n2', 'n1'),
    ];
    const spec: GraphSpec = JSON.parse(flowToGraphJson(nodes, edges));
    expect(spec.version).toBe(2);
    expect(spec.nodes[0]).toMatchObject({ id: 'n1', type: 'Linear', params: { out: 10 } });
    expect(spec.nodes[0].ports).toEqual([{ id: 'p', name: 'x' }]);
    expect(spec.nodes[1].params).toEqual({});
    expect(spec.edges[0]).toMatchObject({ sourceHandle: 'sh', targetHandle: 'th' });
    expect(spec.edges[1].sourceHandle).toBeNull();
    expect(spec.edges[1].targetHandle).toBeNull();
  });

  it('round-trips through graphToFlow (serialize -> deserialize)', () => {
    const original = graphToFlow(
      JSON.stringify({
        version: 2,
        nodes: [
          { id: 'i', type: 'Input', ports: [{ id: 'ip', name: 'x' }], position: { x: 1, y: 2 } },
          { id: 'c', type: 'Conv2d', params: { k: 3 }, position: { x: 3, y: 4 } },
          { id: 'o', type: 'Output', ports: [{ id: 'op', name: 'y' }], position: { x: 5, y: 6 } },
        ],
        edges: [
          { id: 'e1', source: 'i', sourceHandle: 'ip', target: 'c', targetHandle: null },
          { id: 'e2', source: 'c', sourceHandle: null, target: 'o', targetHandle: 'op' },
        ],
      }),
    );
    const json = flowToGraphJson(original.nodes, original.edges);
    const round = graphToFlow(json);
    expect(round.nodes.map((n) => n.id)).toEqual(['i', 'c', 'o']);
    expect(round.nodes.map((n) => n.data.layerType)).toEqual(['Input', 'Conv2d', 'Output']);
    expect(round.edges.map((e) => e.id)).toEqual(['e1', 'e2']);
  });
});

// ── graphToFlow ────────────────────────────────────────────────────────────

describe('graphToFlow', () => {
  it('returns an empty graph for invalid JSON', () => {
    const res = graphToFlow('}}}not json');
    // emptyGraph yields exactly one Input + one Output node and no edges.
    expect(res.nodes).toHaveLength(2);
    expect(res.edges).toHaveLength(0);
  });

  it('returns an empty graph when version is not 2', () => {
    const res = graphToFlow(JSON.stringify({ version: 1, nodes: [], edges: [] }));
    expect(res.nodes).toHaveLength(2);
  });

  it('returns an empty graph when nodes is not an array', () => {
    const res = graphToFlow(JSON.stringify({ version: 2, nodes: null, edges: [] }));
    expect(res.nodes).toHaveLength(2);
  });

  it('returns an empty graph when edges is not an array', () => {
    const res = graphToFlow(JSON.stringify({ version: 2, nodes: [], edges: null }));
    expect(res.nodes).toHaveLength(2);
  });

  it('maps node types: Input -> inputNode, Output -> outputNode, merge & plain -> layerNode', () => {
    const res = graphToFlow(
      JSON.stringify({
        version: 2,
        nodes: [
          { id: 'i', type: 'Input', ports: [{ id: 'p', name: 'x' }], position: { x: 1, y: 1 } },
          { id: 'm', type: 'Add', position: { x: 2, y: 2 } },
          { id: 'c', type: 'Conv2d', position: { x: 3, y: 3 } },
          { id: 'o', type: 'Output', ports: [{ id: 'q', name: 'y' }], position: { x: 4, y: 4 } },
        ],
        edges: [],
      }),
    );
    const byId = Object.fromEntries(res.nodes.map((n) => [n.id, n]));
    expect(byId.i.type).toBe('inputNode');
    expect(byId.o.type).toBe('outputNode');
    expect(byId.m.type).toBe('layerNode');
    expect(byId.c.type).toBe('layerNode');
    // Merge node flagged.
    expect(byId.m.data.isMerge).toBe(true);
    expect(byId.c.data.isMerge).toBe(false);
    // Boundary nodes keep ports; non-boundary get undefined ports.
    expect(byId.i.data.ports).toEqual([{ id: 'p', name: 'x' }]);
    expect(byId.c.data.ports).toBeUndefined();
    expect(byId.i.data.isBoundary).toBe(true);
    expect(byId.c.data.isBoundary).toBe(false);
  });

  it('defaults params to {} and position to origin when missing', () => {
    const res = graphToFlow(
      JSON.stringify({
        version: 2,
        nodes: [{ id: 'c', type: 'Conv2d' }],
        edges: [],
      }),
    );
    expect(res.nodes[0].data.params).toEqual({});
    expect(res.nodes[0].position).toEqual({ x: 0, y: 0 });
  });

  it('preserves edge handles when present and converts null handles to undefined', () => {
    const res = graphToFlow(
      JSON.stringify({
        version: 2,
        nodes: [
          { id: 'a', type: 'Conv2d', position: { x: 1, y: 1 } },
          { id: 'b', type: 'Conv2d', position: { x: 2, y: 2 } },
        ],
        edges: [
          { id: 'e1', source: 'a', sourceHandle: 'sh', target: 'b', targetHandle: 'th' },
          { id: 'e2', source: 'b', sourceHandle: null, target: 'a', targetHandle: null },
        ],
      }),
    );
    expect(res.edges[0].sourceHandle).toBe('sh');
    expect(res.edges[0].targetHandle).toBe('th');
    expect(res.edges[1].sourceHandle).toBeUndefined();
    expect(res.edges[1].targetHandle).toBeUndefined();
    expect(res.edges[0].style).toMatchObject({ stroke: '#555', strokeWidth: 2 });
  });

  it('auto-lays out nodes when all positions are at origin/missing (needsLayout=true)', () => {
    // 3 nodes all at origin -> assignPositionsFromTopology runs and spreads them.
    const res = graphToFlow(
      JSON.stringify({
        version: 2,
        nodes: [
          { id: 'a', type: 'Conv2d' },
          { id: 'b', type: 'Conv2d' },
          { id: 'c', type: 'Conv2d' },
        ],
        edges: [
          { id: 'e1', source: 'a', target: 'b' },
          { id: 'e2', source: 'b', target: 'c' },
        ],
      }),
    );
    const positions = res.nodes.map((n) => `${n.position.x},${n.position.y}`);
    // After layout, not all nodes remain stacked at the same coordinate.
    expect(new Set(positions).size).toBeGreaterThan(1);
  });

  it('does NOT auto-layout when a single node is present (needsLayout=false via length>1)', () => {
    const res = graphToFlow(
      JSON.stringify({ version: 2, nodes: [{ id: 'a', type: 'Conv2d' }], edges: [] }),
    );
    expect(res.nodes[0].position).toEqual({ x: 0, y: 0 });
  });

  it('does NOT auto-layout when nodes already have non-origin positions', () => {
    const res = graphToFlow(
      JSON.stringify({
        version: 2,
        nodes: [
          { id: 'a', type: 'Conv2d', position: { x: 10, y: 20 } },
          { id: 'b', type: 'Conv2d', position: { x: 30, y: 40 } },
        ],
        edges: [{ id: 'e', source: 'a', target: 'b' }],
      }),
    );
    expect(res.nodes[0].position).toEqual({ x: 10, y: 20 });
    expect(res.nodes[1].position).toEqual({ x: 30, y: 40 });
  });

  it('treats a node at x=0 but y!=0 as already positioned (needsLayout=false)', () => {
    // Exercises the `n.position.y === 0` half of the origin check short-circuiting.
    const res = graphToFlow(
      JSON.stringify({
        version: 2,
        nodes: [
          { id: 'a', type: 'Conv2d', position: { x: 0, y: 5 } },
          { id: 'b', type: 'Conv2d', position: { x: 0, y: 0 } },
        ],
        edges: [{ id: 'e', source: 'a', target: 'b' }],
      }),
    );
    // a has y!=0 so the `every(... x===0 && y===0)` is false -> no relayout.
    expect(res.nodes[0].position).toEqual({ x: 0, y: 5 });
    expect(res.nodes[1].position).toEqual({ x: 0, y: 0 });
  });

  it('auto-layout keeps a short chain in a single column', () => {
    // 5-node chain with no skips: the valley pass is a no-op, so dagre's
    // single-column TB layout comes through untouched.
    const nodes = Array.from({ length: 5 }, (_, i) => ({ id: `s${i}`, type: 'Conv2d' }));
    const edges = Array.from({ length: 4 }, (_, i) => ({
      id: `se${i}`,
      source: `s${i}`,
      target: `s${i + 1}`,
    }));
    const res = graphToFlow(JSON.stringify({ version: 2, nodes, edges }));
    // All nodes land in a single column (one distinct x).
    const xs = new Set(res.nodes.map((n) => n.position.x));
    expect(xs.size).toBe(1);
    // But they are spread vertically by dagre.
    const ys = new Set(res.nodes.map((n) => n.position.y));
    expect(ys.size).toBeGreaterThan(1);
  });

  it('colors each known layer category and merges, with a fallback for unknowns', () => {
    const res = graphToFlow(
      JSON.stringify({
        version: 2,
        nodes: [
          { id: 'i', type: 'Input', ports: [{ id: 'p1', name: 'x' }], position: { x: 1, y: 1 } },
          { id: 'o', type: 'Output', ports: [{ id: 'p2', name: 'y' }], position: { x: 2, y: 2 } },
          { id: 'add', type: 'Add', position: { x: 3, y: 3 } },
          { id: 'conv', type: 'Conv2d', position: { x: 4, y: 4 } },
          { id: 'bn', type: 'BatchNorm2d', position: { x: 5, y: 5 } },
          { id: 'pool', type: 'MaxPool2d', position: { x: 6, y: 6 } },
          { id: 'drop', type: 'Dropout', position: { x: 7, y: 7 } },
          { id: 'lin', type: 'Linear', position: { x: 8, y: 8 } },
          { id: 'flat', type: 'Flatten', position: { x: 9, y: 9 } },
          { id: 'wat', type: 'TotallyUnknown', position: { x: 10, y: 10 } },
        ],
        edges: [],
      }),
    );
    const color = (id: string) => res.nodes.find((n) => n.id === id)!.data.color;
    expect(color('i')).toBe('#4CAF50');
    expect(color('o')).toBe('#F44336');
    expect(color('add')).toBe('#FF9800');
    expect(color('conv')).toBe('#4CAF50');
    expect(color('bn')).toBe('#9C27B0');
    expect(color('pool')).toBe('#2196F3');
    expect(color('drop')).toBe('#FF9800');
    expect(color('lin')).toBe('#00BCD4');
    expect(color('flat')).toBe('#607D8B');
    // Unknown type falls back to red.
    expect(color('wat')).toBe('#F44336');
  });

  it('ignores edges referencing unknown node ids during layout', () => {
    // Forces the `nodeIds.has(...)` false branch in assignPositionsFromTopology.
    const res = graphToFlow(
      JSON.stringify({
        version: 2,
        nodes: [
          { id: 'a', type: 'Conv2d' },
          { id: 'b', type: 'Conv2d' },
        ],
        edges: [
          { id: 'e1', source: 'a', target: 'b' },
          { id: 'ghost', source: 'a', target: 'missing' },
        ],
      }),
    );
    expect(res.nodes).toHaveLength(2);
  });

  it('a long chain never wraps into columns — single column preserved', () => {
    // 40-node vertical chain: column wrapping is gone, so however tall the
    // layout gets it stays one column (pan/zoom handles the length).
    const n = 40;
    const nodes = Array.from({ length: n }, (_, i) => ({ id: `n${i}`, type: 'Conv2d' }));
    const edges = Array.from({ length: n - 1 }, (_, i) => ({
      id: `e${i}`,
      source: `n${i}`,
      target: `n${i + 1}`,
    }));
    const res = graphToFlow(JSON.stringify({ version: 2, nodes, edges }));
    const xs = new Set(res.nodes.map((node) => node.position.x));
    expect(xs.size).toBe(1);
    // y strictly increases down the chain (no carriage-return columns).
    const ys = res.nodes.map((node) => node.position.y);
    for (let i = 1; i < ys.length; i++) expect(ys[i]).toBeGreaterThan(ys[i - 1]);
  });

  it('a skip connection sinks its covered ranks rightward (transposed valley)', () => {
    // 12-node TB chain with a skip n1 -> n7: ranks 2..6 are covered and shift
    // right by a full valley step; the skip endpoints stay in the base column
    // (within dagre's dummy-lane jitter, which is smaller than one step).
    const n = 12;
    const nodes: GraphSpec['nodes'] = Array.from({ length: n }, (_, i) => ({
      id: `n${i}`,
      type: 'Conv2d',
    }));
    const edges: GraphSpec['edges'] = Array.from({ length: n - 1 }, (_, i) => ({
      id: `e${i}`,
      source: `n${i}`,
      target: `n${i + 1}`,
    }));
    edges.push({ id: 'skip', source: 'n1', target: 'n7' });
    const res = graphToFlow(JSON.stringify({ version: 2, nodes, edges }));
    const xOf = (id: string) => res.nodes.find((node) => node.id === id)!.position.x;
    // Covered trunk (n4) sits clearly right of both skip endpoints.
    expect(xOf('n4')).toBeGreaterThan(Math.max(xOf('n1'), xOf('n7')) + 60);
    // Outside the skip, head and tail share the base column band.
    expect(Math.abs(xOf('n0') - xOf('n11'))).toBeLessThan(1);
  });
});

// ── emptyGraph ─────────────────────────────────────────────────────────────

describe('emptyGraph', () => {
  it('produces an Input and an Output node with one port each and no edges', () => {
    const g = emptyGraph();
    expect(g.edges).toHaveLength(0);
    expect(g.nodes).toHaveLength(2);
    const [input, output] = g.nodes;
    expect(input.type).toBe('inputNode');
    expect(input.data.layerType).toBe('Input');
    expect(input.data.ports).toHaveLength(1);
    expect(input.data.ports![0].name).toBe('x');
    expect(output.type).toBe('outputNode');
    expect(output.data.layerType).toBe('Output');
    expect(output.data.ports![0].name).toBe('out');
    // Distinct generated ids.
    expect(input.id).not.toBe(output.id);
  });
});

// ── autoLayoutSubgraph ─────────────────────────────────────────────────────

describe('autoLayoutSubgraph', () => {
  it('returns the same array reference for an empty node list', () => {
    const nodes: Node<LayerNodeData>[] = [];
    expect(autoLayoutSubgraph(nodes, [])).toBe(nodes);
  });

  it('positions nodes using measured/explicit/default dimensions', () => {
    const nodes = [
      // measured dims
      flowNode('a', {}, { measured: { width: 200, height: 60 } }),
      // explicit width/height (no measured)
      flowNode('b', {}, { width: 120, height: 30 }),
      // neither -> defaults
      flowNode('c', {}),
    ];
    const edges = [edge('e1', 'a', 'b'), edge('e2', 'b', 'c')];
    const laid = autoLayoutSubgraph(nodes, edges);
    expect(laid).toHaveLength(3);
    // Layout assigns finite, non-identical positions down the chain.
    const ys = laid.map((n) => n.position.y);
    expect(new Set(ys).size).toBeGreaterThan(1);
    ys.forEach((y) => expect(Number.isFinite(y)).toBe(true));
  });

  it('ignores edges that reference ids not in the node set', () => {
    const nodes = [flowNode('a', {}), flowNode('b', {})];
    const edges = [edge('e1', 'a', 'b'), edge('ghost', 'a', 'zzz')];
    const laid = autoLayoutSubgraph(nodes, edges);
    expect(laid).toHaveLength(2);
  });

  it('very tall nodes still lay out in a single column (no wrapping by pixel height)', () => {
    // Column wrapping used to fire on pixel height; with the valley pass a
    // skip-free chain is returned untouched regardless of how tall it renders.
    const tall = (id: string): Node<LayerNodeData> =>
      flowNode(id, {}, { width: 160, height: 2000 });
    const nodes = [tall('t0'), tall('t1'), tall('t2'), tall('t3')];
    const edges = [edge('e0', 't0', 't1'), edge('e1', 't1', 't2'), edge('e2', 't2', 't3')];
    const laid = autoLayoutSubgraph(nodes, edges);
    expect(laid).toHaveLength(4);
    // No horizontal wrapping: single column.
    const xs = new Set(laid.map((n) => n.position.x));
    expect(xs.size).toBe(1);
  });

  it('chooses dagre layout config buckets by node count (>25 and >50 thresholds)', () => {
    // Exercise the >25 branch.
    const mk = (count: number) => {
      const nodes = Array.from({ length: count }, (_, i) => flowNode(`n${i}`, {}));
      const edges = Array.from({ length: count - 1 }, (_, i) =>
        edge(`e${i}`, `n${i}`, `n${i + 1}`),
      );
      return autoLayoutSubgraph(nodes, edges);
    };
    expect(mk(30)).toHaveLength(30); // >25 bucket
    expect(mk(60)).toHaveLength(60); // >50 bucket
  });
});

// ── validateGraph ──────────────────────────────────────────────────────────

describe('validateGraph', () => {
  it('passes a valid linear graph', () => {
    const i = inputNode('i', [{ id: 'ip', name: 'x' }]);
    const c = flowNode('c', { layerType: 'Conv2d' });
    const o = outputNode('o', [{ id: 'op', name: 'y' }]);
    const edges = [
      edge('e1', 'i', 'c', { sourceHandle: 'ip' }),
      edge('e2', 'c', 'o', { targetHandle: 'op' }),
    ];
    expect(validateGraph([i, c, o], edges)).toBeNull();
  });

  it('requires exactly one Input node', () => {
    const o = outputNode('o', [{ id: 'op', name: 'y' }]);
    expect(validateGraph([o], [])).toEqual({
      message: 'Graph must have exactly one Input node',
    });
  });

  it('requires exactly one Output node', () => {
    const i = inputNode('i', [{ id: 'ip', name: 'x' }]);
    expect(validateGraph([i], [])).toEqual({
      message: 'Graph must have exactly one Output node',
    });
  });

  it('requires the Input node to have at least one port', () => {
    const i = inputNode('i', []);
    const o = outputNode('o', [{ id: 'op', name: 'y' }]);
    expect(validateGraph([i, o], [])).toEqual({
      message: 'Input node must have at least one port',
    });
  });

  it('requires the Output node to have at least one port', () => {
    const i = inputNode('i', [{ id: 'ip', name: 'x' }]);
    const o = outputNode('o', []);
    expect(validateGraph([i, o], [])).toEqual({
      message: 'Output node must have at least one port',
    });
  });

  it('uses [] when an Input node has undefined ports', () => {
    // Covers the `input.data.ports ?? []` fallback for the input side.
    const i = flowNode('i', { layerType: 'Input', isBoundary: true }); // no ports field
    const o = outputNode('o', [{ id: 'op', name: 'y' }]);
    expect(validateGraph([i, o], [])).toEqual({
      message: 'Input node must have at least one port',
    });
  });

  it('uses [] when an Output node has undefined ports', () => {
    const i = inputNode('i', [{ id: 'ip', name: 'x' }]);
    const o = flowNode('o', { layerType: 'Output', isBoundary: true }); // no ports field
    expect(validateGraph([i, o], [])).toEqual({
      message: 'Output node must have at least one port',
    });
  });

  it('rejects duplicate input port names', () => {
    const i = inputNode('i', [
      { id: 'ip1', name: 'x' },
      { id: 'ip2', name: 'x' },
    ]);
    const o = outputNode('o', [{ id: 'op', name: 'y' }]);
    expect(validateGraph([i, o], [])).toEqual({
      message: 'Input port names must be unique',
    });
  });

  it('rejects duplicate output port names', () => {
    const i = inputNode('i', [{ id: 'ip', name: 'x' }]);
    const o = outputNode('o', [
      { id: 'op1', name: 'y' },
      { id: 'op2', name: 'y' },
    ]);
    expect(validateGraph([i, o], [])).toEqual({
      message: 'Output port names must be unique',
    });
  });

  it('requires each output port to have exactly one incoming edge (too many)', () => {
    const i = inputNode('i', [{ id: 'ip', name: 'x' }]);
    const c = flowNode('c', { layerType: 'Conv2d' });
    const o = outputNode('o', [{ id: 'op', name: 'y' }]);
    const edges = [
      edge('e1', 'i', 'c', { sourceHandle: 'ip' }),
      edge('e2', 'c', 'o', { targetHandle: 'op' }),
      edge('e3', 'i', 'o', { sourceHandle: 'ip', targetHandle: 'op' }),
    ];
    expect(validateGraph([i, c, o], edges)).toEqual({
      message: "Output port 'y' must have exactly 1 incoming edge (got 2)",
    });
  });

  it('requires each output port to have at least one incoming edge (zero)', () => {
    const i = inputNode('i', [{ id: 'ip', name: 'x' }]);
    const o = outputNode('o', [{ id: 'op', name: 'y' }]);
    // input port used so it does not trip the unused-input check first.
    const edges = [edge('e1', 'i', 'o', { sourceHandle: 'ip', targetHandle: 'WRONG' })];
    expect(validateGraph([i, o], edges)).toEqual({
      message: "Output port 'y' must have exactly 1 incoming edge (got 0)",
    });
  });

  it('rejects an unused input port', () => {
    const i = inputNode('i', [{ id: 'ip', name: 'x' }]);
    const o = outputNode('o', [{ id: 'op', name: 'y' }]);
    // Output port satisfied, but input port has no outgoing edge.
    const edges = [edge('e1', 'i', 'o', { sourceHandle: 'OTHER', targetHandle: 'op' })];
    expect(validateGraph([i, o], edges)).toEqual({
      message: "Input port 'x' is unused",
    });
  });

  it('requires plain layers to have exactly one incoming edge (zero)', () => {
    const i = inputNode('i', [{ id: 'ip', name: 'x' }]);
    const c = flowNode('c', { layerType: 'Conv2d' });
    const o = outputNode('o', [{ id: 'op', name: 'y' }]);
    // c has no incoming edge.
    const edges = [
      edge('e1', 'i', 'o', { sourceHandle: 'ip', targetHandle: 'op' }),
      edge('e2', 'c', 'o', { targetHandle: 'op' }),
    ];
    // Output already has 2 incoming -> but the layer check is AFTER the output-port check,
    // so craft a clean case: give c zero incoming but a valid single-incoming output.
    const i2 = inputNode('i2', [{ id: 'ip2', name: 'x' }]);
    const c2 = flowNode('c2', { layerType: 'Linear' });
    const o2 = outputNode('o2', [{ id: 'op2', name: 'y' }]);
    const edges2 = [
      edge('a', 'i2', 'o2', { sourceHandle: 'ip2', targetHandle: 'op2' }),
    ];
    // c2 is isolated -> 0 incoming.
    expect(validateGraph([i2, c2, o2], edges2)).toEqual({
      message: "Layer 'Linear' must have exactly 1 incoming edge (got 0)",
    });
    void edges;
    void i;
    void c;
    void o;
  });

  it('skips the incoming-edge rule for merge layers (they may take many)', () => {
    const i = inputNode('i', [{ id: 'ip', name: 'x' }]);
    const c1 = flowNode('c1', { layerType: 'Conv2d' });
    const c2 = flowNode('c2', { layerType: 'Conv2d' });
    const add = flowNode('add', { layerType: 'Add', isMerge: true });
    const o = outputNode('o', [{ id: 'op', name: 'y' }]);
    const edges = [
      edge('e1', 'i', 'c1', { sourceHandle: 'ip' }),
      edge('e2', 'i', 'c2', { sourceHandle: 'ip' }),
      edge('e3', 'c1', 'add'),
      edge('e4', 'c2', 'add'),
      edge('e5', 'add', 'o', { targetHandle: 'op' }),
    ];
    expect(validateGraph([i, c1, c2, add, o], edges)).toBeNull();
  });

  it('detects a cycle', () => {
    const i = inputNode('i', [{ id: 'ip', name: 'x' }]);
    // Merge nodes bypass the plain-layer "exactly 1 incoming" rule so we reach
    // the cycle check with a back-edge present.
    const a = flowNode('a', { layerType: 'Add', isMerge: true });
    const b = flowNode('b', { layerType: 'Add', isMerge: true });
    const o = outputNode('o', [{ id: 'op', name: 'y' }]);
    const edges = [
      edge('e1', 'i', 'a', { sourceHandle: 'ip' }),
      edge('e2', 'a', 'b'),
      edge('e3', 'b', 'a'), // back-edge -> cycle
      edge('e4', 'b', 'o', { targetHandle: 'op' }),
    ];
    expect(validateGraph([i, a, b, o], edges)).toEqual({ message: 'Graph contains a cycle' });
  });

  it('reports a node unreachable from Input', () => {
    const i = inputNode('i', [{ id: 'ip', name: 'x' }]);
    // 'a' is a merge node fed by both the input and 'island'.
    const a = flowNode('a', { layerType: 'Add', isMerge: true });
    // 'island' is a merge node with no incoming edge: it can reach the output (via a)
    // but is not reachable from the input. Merge bypasses the plain-layer rule, and
    // 0 incoming keeps the cycle check happy (it becomes a Kahn root).
    const island = flowNode('island', { layerType: 'Concat', isMerge: true });
    const o = outputNode('o', [{ id: 'op', name: 'y' }]);
    const edges = [
      edge('e1', 'i', 'a', { sourceHandle: 'ip' }),
      edge('e2', 'a', 'o', { targetHandle: 'op' }),
      edge('e3', 'island', 'a'),
    ];
    expect(validateGraph([i, a, island, o], edges)).toEqual({
      message: "Node 'Concat' is not reachable from Input",
    });
  });

  it('passes a graph that pushes a node onto each DFS stack twice (re-visit `continue` guards)', () => {
    // Topology: i -> split -> {x, b}, b -> x, x -> o. With stack-based DFS, the
    // shared successor `x` is pushed before `b` is processed, then re-pushed by
    // `b -> x` while still unvisited; popping the duplicate hits the
    // `if (reachableFromInput.has(id)) continue;` line. The mirror happens in the
    // backward pass for `split`. All nodes are merge/boundary so the per-layer
    // single-incoming rule is bypassed and we actually reach the reachability code.
    const i = inputNode('i', [{ id: 'ip', name: 'x' }]);
    const split = flowNode('split', { layerType: 'Add', isMerge: true });
    const b = flowNode('b', { layerType: 'Concat', isMerge: true });
    const x = flowNode('x', { layerType: 'Multiply', isMerge: true });
    const o = outputNode('o', [{ id: 'op', name: 'y' }]);
    const edges = [
      edge('e1', 'i', 'split', { sourceHandle: 'ip' }),
      edge('e2', 'split', 'x'),
      edge('e3', 'split', 'b'),
      edge('e4', 'b', 'x'),
      edge('e5', 'x', 'o', { targetHandle: 'op' }),
    ];
    expect(validateGraph([i, split, b, x, o], edges)).toBeNull();
  });

  it('reports a node that cannot reach Output', () => {
    const i = inputNode('i', [{ id: 'ip', name: 'x' }]);
    const a = flowNode('a', { layerType: 'Conv2d' });
    const dead = flowNode('dead', { layerType: 'Dropout' });
    const o = outputNode('o', [{ id: 'op', name: 'y' }]);
    const edges = [
      edge('e1', 'i', 'a', { sourceHandle: 'ip' }),
      edge('e2', 'a', 'o', { targetHandle: 'op' }),
      // dead is fed from input but never reaches output.
      edge('e3', 'i', 'dead', { sourceHandle: 'ip' }),
    ];
    // dead has 1 incoming (ok for plain layer), reachable from input, but cannot reach output.
    expect(validateGraph([i, a, dead, o], edges)).toEqual({
      message: "Node 'Dropout' cannot reach Output",
    });
  });
});
