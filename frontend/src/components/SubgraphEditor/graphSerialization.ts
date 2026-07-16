// frontend/src/components/SubgraphEditor/graphSerialization.ts
import dagre from '@dagrejs/dagre';
import type { Node, Edge } from '@xyflow/react';
import { generateId } from '../../utils';
import { applyValleyPass, isTriggerEdge } from '../../utils/autoLayout';

export interface PortDef {
  id: string;
  name: string;
}

export interface LayerNodeData {
  layerType: string;
  params: Record<string, any>;
  color: string;
  ports?: PortDef[]; // only Input / Output
  isMerge?: boolean;
  isBoundary?: boolean;
  [key: string]: unknown;
}

export interface GraphSpec {
  version: 2;
  nodes: Array<{
    id: string;
    type: string;
    params?: Record<string, any>;
    ports?: PortDef[];
    position?: { x: number; y: number };
  }>;
  edges: Array<{
    id: string;
    source: string;
    sourceHandle?: string | null;
    target: string;
    targetHandle?: string | null;
  }>;
}

const MERGE_TYPES = new Set(['Add', 'Concat', 'Multiply', 'Subtract', 'Mean', 'Stack']);

/** All layer types that the SubgraphEditor can represent. */
const KNOWN_LAYER_TYPES = new Set([
  'Conv2d', 'Conv1d', 'ConvTranspose2d',
  'BatchNorm2d', 'BatchNorm1d', 'LayerNorm', 'GroupNorm', 'InstanceNorm2d',
  'MaxPool2d', 'AvgPool2d', 'AdaptiveAvgPool2d',
  'Dropout',
  'Linear', 'Embedding',
  'Flatten',
  'ReLU', 'LeakyReLU', 'GELU', 'SiLU', 'Mish', 'ELU', 'SELU', 'PReLU',
  'Sigmoid', 'Tanh', 'Hardswish', 'Softmax',
  'Add', 'Concat', 'Multiply', 'Subtract', 'Mean', 'Stack',
]);

/** Map main-editor Activation node's `function` param → SubgraphEditor layer type. */
const ACTIVATION_MAP: Record<string, string> = {
  relu: 'ReLU', leaky_relu: 'LeakyReLU', gelu: 'GELU', silu: 'SiLU',
  mish: 'Mish', elu: 'ELU', selu: 'SELU', prelu: 'PReLU',
  sigmoid: 'Sigmoid', tanh: 'Tanh', hardswish: 'Hardswish', softmax: 'Softmax',
};

export function isMergeType(t: string): boolean {
  return MERGE_TYPES.has(t);
}

// ── Main-editor workflow import types & functions ──

export interface WorkflowData {
  nodes: Array<{
    id: string;
    type: string;
    position?: { x: number; y: number };
    data?: { params?: Record<string, any> };
  }>;
  edges: Array<{
    id: string;
    source: string;
    target: string;
    sourceHandle?: string;
    targetHandle?: string;
  }>;
}

export interface SequentialModelEntry {
  nodeId: string;
  label: string;
  layersJson: string;
}

export type ImportDetection =
  | { kind: 'graphspec'; json: string }
  | { kind: 'workflow-layers'; workflowData: WorkflowData }
  | { kind: 'workflow-sequential'; models: SequentialModelEntry[] }
  | { kind: 'unknown' };

export function detectImportFormat(text: string): ImportDetection {
  let data: any;
  try {
    data = JSON.parse(text);
  } catch {
    return { kind: 'unknown' };
  }

  // GraphSpec v2
  if (data.version === 2 && Array.isArray(data.nodes) && Array.isArray(data.edges)) {
    return { kind: 'graphspec', json: text };
  }

  // Main-editor workflow
  if (Array.isArray(data.nodes) && Array.isArray(data.edges)) {
    const hasLayerNodes = data.nodes.some((n: any) => {
      const t = n.type ?? '';
      if (KNOWN_LAYER_TYPES.has(t)) return true;
      if (t === 'Activation') return (n.data?.params?.function ?? '') in ACTIVATION_MAP;
      return false;
    });

    if (hasLayerNodes) {
      return { kind: 'workflow-layers', workflowData: { nodes: data.nodes, edges: data.edges } };
    }

    const models: SequentialModelEntry[] = data.nodes
      .filter((n: any) => n.type === 'SequentialModel' && typeof n.data?.params?.layers === 'string')
      .map((n: any) => ({
        nodeId: n.id,
        label: n.data?.label ?? n.id,
        layersJson: n.data.params.layers,
      }));

    if (models.length > 0) {
      return { kind: 'workflow-sequential', models };
    }
  }

  return { kind: 'unknown' };
}

export function convertWorkflowToGraphSpec(workflow: WorkflowData): GraphSpec {
  const layerNodeIds = new Set<string>();
  const convertedNodes: GraphSpec['nodes'] = [];

  for (const n of workflow.nodes) {
    const rawType = n.type ?? '';
    let layerType = rawType;
    const params = { ...(n.data?.params ?? {}) };

    if (rawType === 'Activation') {
      const fn = params.function ?? 'relu';
      const mapped = ACTIVATION_MAP[fn];
      if (!mapped) continue;
      layerType = mapped;
      delete params.function;
    } else if (!KNOWN_LAYER_TYPES.has(rawType)) {
      continue;
    }

    layerNodeIds.add(n.id);
    convertedNodes.push({
      id: n.id,
      type: layerType,
      params: Object.keys(params).length > 0 ? params : undefined,
      position: n.position,
    });
  }

  // Keep only edges between converted layer nodes, strip handles
  const convertedEdges: GraphSpec['edges'] = workflow.edges
    .filter((e) => layerNodeIds.has(e.source) && layerNodeIds.has(e.target))
    .map((e) => ({
      id: e.id,
      source: e.source,
      sourceHandle: null,
      target: e.target,
      targetHandle: null,
    }));

  // Compute in-degree / out-degree
  const inDeg = new Map<string, number>();
  const outDeg = new Map<string, number>();
  for (const id of layerNodeIds) { inDeg.set(id, 0); outDeg.set(id, 0); }
  for (const e of convertedEdges) {
    // inDeg/outDeg are pre-seeded for every layer id above and convertedEdges only
    // reference converted layer nodes, so the `?? 0` fallbacks are dead.
    /* v8 ignore start */
    inDeg.set(e.target, (inDeg.get(e.target) ?? 0) + 1);
    outDeg.set(e.source, (outDeg.get(e.source) ?? 0) + 1);
    /* v8 ignore stop */
  }

  // inDeg/outDeg are pre-seeded for every layer id above, so the `?? 0` fallbacks are dead.
  /* v8 ignore start */
  const roots = [...layerNodeIds].filter((id) => (inDeg.get(id) ?? 0) === 0);
  const leaves = [...layerNodeIds].filter((id) => (outDeg.get(id) ?? 0) === 0);
  /* v8 ignore stop */

  // Create Input node with one port per root
  const inputNodeId = generateId();
  const inputPorts: PortDef[] = roots.map((_, i) => ({
    id: generateId(),
    name: i === 0 ? 'x' : `x${i + 1}`,
  }));
  convertedNodes.unshift({ id: inputNodeId, type: 'Input', ports: inputPorts });
  for (let i = 0; i < roots.length; i++) {
    convertedEdges.push({
      id: generateId(),
      source: inputNodeId,
      sourceHandle: inputPorts[i].id,
      target: roots[i],
      targetHandle: null,
    });
  }

  // Create Output node connected to the topologically-last leaf
  const outputNodeId = generateId();
  const outputPorts: PortDef[] = [{ id: generateId(), name: 'y' }];
  convertedNodes.push({ id: outputNodeId, type: 'Output', ports: outputPorts });

  const sorted = topoSort(
    convertedNodes.map((n) => n.id),
    convertedEdges,
  );
  let lastLeaf = leaves[0];
  for (let i = sorted.length - 1; i >= 0; i--) {
    if (leaves.includes(sorted[i])) { lastLeaf = sorted[i]; break; }
  }
  convertedEdges.push({
    id: generateId(),
    source: lastLeaf,
    sourceHandle: null,
    target: outputNodeId,
    targetHandle: outputPorts[0].id,
  });

  return { version: 2, nodes: convertedNodes, edges: convertedEdges };
}

export function flowToGraphJson(nodes: Node<LayerNodeData>[], edges: Edge[]): string {
  const spec: GraphSpec = {
    version: 2,
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.data.layerType,
      params: n.data.params ?? {},
      ports: n.data.ports,
      position: n.position,
    })),
    edges: edges.map((e) => ({
      id: e.id,
      source: e.source,
      sourceHandle: e.sourceHandle ?? null,
      target: e.target,
      targetHandle: e.targetHandle ?? null,
    })),
  };
  return JSON.stringify(spec);
}

const SUBGRAPH_NODE_W = 160;
const SUBGRAPH_NODE_H = 40;
const SUBGRAPH_NODESEP = 40;
const SUBGRAPH_RANKSEP = 60;
function getSubgraphLayoutConfig(
  nodeCount: number,
): { nodesep: number; ranksep: number } {
  if (nodeCount > 50) return { nodesep: 28, ranksep: 42 };
  if (nodeCount > 25) return { nodesep: 32, ranksep: 50 };
  return { nodesep: SUBGRAPH_NODESEP, ranksep: SUBGRAPH_RANKSEP };
}

type SubgraphPos = { x: number; y: number; width: number; height: number };

// Column wrapping was removed in favor of the shared skip-aware valley pass
// (applyValleyPass with axis 'x'): a TB subgraph never splits into columns;
// ranks covered by a skip connection shift rightward instead.

/**
 * Topological-sort ids and return the order.
 */
function topoSort(
  nodeIds: string[],
  edges: Array<{ source: string; target: string }>,
): string[] {
  const inDegree: Record<string, number> = {};
  const adj: Record<string, string[]> = {};
  for (const id of nodeIds) {
    inDegree[id] = 0;
    adj[id] = [];
  }
  const idSet = new Set(nodeIds);
  for (const e of edges) {
    // The sole caller passes pre-filtered edges whose endpoints are all in nodeIds,
    // and inDegree/adj are pre-seeded for every id, so the guard/`??` fallbacks are dead.
    /* v8 ignore start */
    if (!idSet.has(e.source) || !idSet.has(e.target)) continue;
    inDegree[e.target] = (inDegree[e.target] ?? 0) + 1;
    adj[e.source] = [...(adj[e.source] ?? []), e.target];
    /* v8 ignore stop */
  }

  const queue: string[] = Object.keys(inDegree).filter((k) => inDegree[k] === 0);
  const sorted: string[] = [];
  while (queue.length) {
    const id = queue.shift()!;
    sorted.push(id);
    for (const t of adj[id]) {
      inDegree[t]--;
      if (inDegree[t] === 0) queue.push(t);
    }
  }
  // Append any nodes not reached (e.g. cycles / isolated)
  for (const id of nodeIds) {
    if (!sorted.includes(id)) sorted.push(id);
  }
  return sorted;
}

/**
 * Use dagre to assign positions when loading a graph without saved positions.
 * Uses default dimensions since nodes haven't been rendered yet.
 */
function assignPositionsFromTopology(spec: GraphSpec): void {
  // Only ever called when needsLayout is true, which requires spec.nodes.length > 1.
  /* v8 ignore start */
  if (spec.nodes.length === 0) return;
  /* v8 ignore stop */

  const g = new dagre.graphlib.Graph();
  const cfg = getSubgraphLayoutConfig(spec.nodes.length);
  g.setGraph({
    rankdir: 'TB',
    nodesep: cfg.nodesep,
    ranksep: cfg.ranksep,
  });
  g.setDefaultEdgeLabel(() => ({}));

  for (const n of spec.nodes) {
    g.setNode(n.id, { width: SUBGRAPH_NODE_W, height: SUBGRAPH_NODE_H });
  }
  const nodeIds = new Set(spec.nodes.map((n) => n.id));
  for (const e of spec.edges) {
    if (nodeIds.has(e.source) && nodeIds.has(e.target)) {
      g.setEdge(e.source, e.target);
    }
  }

  dagre.layout(g);

  const raw = new Map<string, SubgraphPos>();
  for (const n of spec.nodes) {
    const pos = g.node(n.id);
    // Every node was registered via g.setNode above, so dagre always returns a pos.
    /* v8 ignore start */
    if (!pos) continue;
    /* v8 ignore stop */
    raw.set(n.id, {
      x: pos.x - SUBGRAPH_NODE_W / 2,
      y: pos.y - SUBGRAPH_NODE_H / 2,
      width: SUBGRAPH_NODE_W,
      height: SUBGRAPH_NODE_H,
    });
  }
  const wrapped = applyValleyPass(raw, spec.edges, {
    axis: 'x',
    skipPredicate: (e) => !isTriggerEdge(e),
  });
  for (const n of spec.nodes) {
    const p = wrapped.get(n.id);
    // applyValleyPass returns an entry for every input id, so `p` is always
    // defined (the falsy branch is unreachable).
    /* v8 ignore start */
    if (p) {
      /* v8 ignore stop */
      n.position = { x: p.x, y: p.y };
    }
  }
}

/**
 * Auto-layout React Flow nodes top-to-bottom using dagre for proper
 * branching support and measured-dimension-aware spacing.
 */
export function autoLayoutSubgraph(
  nodes: Node<LayerNodeData>[],
  edges: Edge[],
): Node<LayerNodeData>[] {
  if (nodes.length === 0) return nodes;

  const g = new dagre.graphlib.Graph();
  const cfg = getSubgraphLayoutConfig(nodes.length);
  g.setGraph({
    rankdir: 'TB',
    nodesep: cfg.nodesep,
    ranksep: cfg.ranksep,
  });
  g.setDefaultEdgeLabel(() => ({}));

  for (const node of nodes) {
    const w = node.measured?.width ?? node.width ?? SUBGRAPH_NODE_W;
    const h = node.measured?.height ?? node.height ?? SUBGRAPH_NODE_H;
    g.setNode(node.id, { width: w, height: h });
  }

  const nodeIds = new Set(nodes.map((n) => n.id));
  for (const edge of edges) {
    if (nodeIds.has(edge.source) && nodeIds.has(edge.target)) {
      g.setEdge(edge.source, edge.target);
    }
  }

  dagre.layout(g);

  const raw = new Map<string, SubgraphPos>();
  for (const n of nodes) {
    const pos = g.node(n.id);
    // Every node was registered via g.setNode above, so dagre always returns a pos.
    /* v8 ignore start */
    if (!pos) continue;
    /* v8 ignore stop */
    const w = n.measured?.width ?? n.width ?? SUBGRAPH_NODE_W;
    const h = n.measured?.height ?? n.height ?? SUBGRAPH_NODE_H;
    raw.set(n.id, {
      x: pos.x - w / 2,
      y: pos.y - h / 2,
      width: w,
      height: h,
    });
  }
  const wrapped = applyValleyPass(raw, edges, {
    axis: 'x',
    skipPredicate: (e) => !isTriggerEdge(e),
  });

  return nodes.map((n) => {
    const p = wrapped.get(n.id);
    // applyValleyPass returns an entry for every input id, so `p` is always defined.
    /* v8 ignore start */
    if (!p) return n;
    /* v8 ignore stop */
    return {
      ...n,
      position: { x: p.x, y: p.y },
    };
  });
}

export function graphToFlow(json: string): { nodes: Node<LayerNodeData>[]; edges: Edge[] } {
  let spec: GraphSpec;
  try {
    spec = JSON.parse(json);
  } catch {
    return emptyGraph();
  }
  if (spec.version !== 2 || !Array.isArray(spec.nodes) || !Array.isArray(spec.edges)) {
    return emptyGraph();
  }

  // Auto-assign positions when nodes lack them (all at origin or missing)
  const needsLayout = spec.nodes.length > 1 &&
    spec.nodes.every((n) => !n.position || (n.position.x === 0 && n.position.y === 0));
  if (needsLayout) {
    assignPositionsFromTopology(spec);
  }

  const nodes: Node<LayerNodeData>[] = spec.nodes.map((n) => {
    const isInput = n.type === 'Input';
    const isOutput = n.type === 'Output';
    const isBoundary = isInput || isOutput;
    const isMerge = isMergeType(n.type);

    return {
      id: n.id,
      type: isInput ? 'inputNode' : isOutput ? 'outputNode' : 'layerNode',
      position: n.position ?? { x: 0, y: 0 },
      data: {
        layerType: n.type,
        params: n.params ?? {},
        color: colorForType(n.type),
        ports: isBoundary ? n.ports : undefined,
        isMerge,
        isBoundary,
      },
    };
  });

  const edges: Edge[] = spec.edges.map((e) => ({
    id: e.id,
    source: e.source,
    sourceHandle: e.sourceHandle ?? undefined,
    target: e.target,
    targetHandle: e.targetHandle ?? undefined,
    style: { stroke: '#555', strokeWidth: 2 },
  }));

  return { nodes, edges };
}

export function emptyGraph(): { nodes: Node<LayerNodeData>[]; edges: Edge[] } {
  const inId = generateId();
  const outId = generateId();
  const inPortId = generateId();
  const outPortId = generateId();
  return {
    nodes: [
      {
        id: inId,
        type: 'inputNode',
        position: { x: 50, y: 50 },
        data: {
          layerType: 'Input',
          params: {},
          color: '#4CAF50',
          ports: [{ id: inPortId, name: 'x' }],
          isBoundary: true,
        },
      },
      {
        id: outId,
        type: 'outputNode',
        position: { x: 50, y: 400 },
        data: {
          layerType: 'Output',
          params: {},
          color: '#F44336',
          ports: [{ id: outPortId, name: 'out' }],
          isBoundary: true,
        },
      },
    ],
    edges: [],
  };
}

function colorForType(type: string): string {
  if (type === 'Input') return '#4CAF50';
  if (type === 'Output') return '#F44336';
  if (MERGE_TYPES.has(type)) return '#FF9800';
  // fallback colors by category — duplicated from SubgraphEditorModal LAYER_DEFS
  const colors: Record<string, string> = {
    Conv2d: '#4CAF50', Conv1d: '#4CAF50', ConvTranspose2d: '#4CAF50',
    BatchNorm2d: '#9C27B0', BatchNorm1d: '#9C27B0', LayerNorm: '#9C27B0',
    GroupNorm: '#9C27B0', InstanceNorm2d: '#9C27B0',
    MaxPool2d: '#2196F3', AvgPool2d: '#2196F3', AdaptiveAvgPool2d: '#2196F3',
    Dropout: '#FF9800',
    Linear: '#00BCD4', Embedding: '#00BCD4',
    Flatten: '#607D8B',
  };
  return colors[type] ?? '#F44336';
}

export interface ValidationError {
  message: string;
}

export function validateGraph(nodes: Node<LayerNodeData>[], edges: Edge[]): ValidationError | null {
  const inputs = nodes.filter((n) => n.data.layerType === 'Input');
  const outputs = nodes.filter((n) => n.data.layerType === 'Output');
  if (inputs.length !== 1) return { message: 'Graph must have exactly one Input node' };
  if (outputs.length !== 1) return { message: 'Graph must have exactly one Output node' };

  const input = inputs[0];
  const output = outputs[0];
  const inPorts = input.data.ports ?? [];
  const outPorts = output.data.ports ?? [];
  if (inPorts.length === 0) return { message: 'Input node must have at least one port' };
  if (outPorts.length === 0) return { message: 'Output node must have at least one port' };

  const inNames = inPorts.map((p) => p.name);
  if (new Set(inNames).size !== inNames.length) return { message: 'Input port names must be unique' };
  const outNames = outPorts.map((p) => p.name);
  if (new Set(outNames).size !== outNames.length) return { message: 'Output port names must be unique' };

  // Output port: exactly 1 incoming
  for (const p of outPorts) {
    const count = edges.filter((e) => e.target === output.id && e.targetHandle === p.id).length;
    if (count !== 1) {
      return { message: `Output port '${p.name}' must have exactly 1 incoming edge (got ${count})` };
    }
  }

  // Input port: at least 1 outgoing
  for (const p of inPorts) {
    const count = edges.filter((e) => e.source === input.id && e.sourceHandle === p.id).length;
    if (count < 1) {
      return { message: `Input port '${p.name}' is unused` };
    }
  }

  // Plain layers (non-merge, non-boundary): exactly 1 incoming
  for (const n of nodes) {
    if (n.data.isBoundary || n.data.isMerge) continue;
    const incoming = edges.filter((e) => e.target === n.id);
    if (incoming.length !== 1) {
      return { message: `Layer '${n.data.layerType}' must have exactly 1 incoming edge (got ${incoming.length})` };
    }
  }

  // Cycle check via Kahn's
  const inDegree: Record<string, number> = {};
  const outAdj: Record<string, string[]> = {};
  for (const n of nodes) {
    inDegree[n.id] = 0;
    outAdj[n.id] = [];
  }
  for (const e of edges) {
    // inDegree/outAdj are pre-seeded for every node id above, so the `?? 0`/`?? []`
    // fallbacks never fire (edges only reference existing nodes).
    /* v8 ignore start */
    inDegree[e.target] = (inDegree[e.target] ?? 0) + 1;
    outAdj[e.source] = [...(outAdj[e.source] ?? []), e.target];
    /* v8 ignore stop */
  }
  const queue: string[] = Object.keys(inDegree).filter((k) => inDegree[k] === 0);
  let visited = 0;
  while (queue.length) {
    const id = queue.shift()!;
    visited++;
    for (const t of outAdj[id]) {
      inDegree[t]--;
      if (inDegree[t] === 0) queue.push(t);
    }
  }
  if (visited !== nodes.length) return { message: 'Graph contains a cycle' };

  // Forward reachability: every node must be reachable from the Input node
  const reachableFromInput = new Set<string>();
  const stack = [input.id];
  while (stack.length) {
    const id = stack.pop()!;
    if (reachableFromInput.has(id)) continue;
    reachableFromInput.add(id);
    for (const e of edges) {
      if (e.source === id && !reachableFromInput.has(e.target)) {
        stack.push(e.target);
      }
    }
  }
  for (const n of nodes) {
    if (!reachableFromInput.has(n.id)) {
      return { message: `Node '${n.data.layerType}' is not reachable from Input` };
    }
  }

  // Backward reachability: every node must be able to reach the Output node
  const canReachOutput = new Set<string>();
  const backStack = [output.id];
  while (backStack.length) {
    const id = backStack.pop()!;
    if (canReachOutput.has(id)) continue;
    canReachOutput.add(id);
    for (const e of edges) {
      if (e.target === id && !canReachOutput.has(e.source)) {
        backStack.push(e.source);
      }
    }
  }
  for (const n of nodes) {
    if (!canReachOutput.has(n.id)) {
      return { message: `Node '${n.data.layerType}' cannot reach Output` };
    }
  }

  return null;
}
