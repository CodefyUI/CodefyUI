// frontend/src/components/LayersEditor/graphSerialization.ts
import dagre from '@dagrejs/dagre';
import type { Node, Edge } from '@xyflow/react';
import { generateId } from '../../utils';
import { applyValleyPass, isTriggerEdge } from '../../utils/autoLayout';
import { LAYER_TYPE_COLORS } from '../../styles/theme';
import type { TranslationKey } from '../../i18n';

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
    /**
     * `Readonly` on purpose. `workflowToGraphSpec` aliases the React Flow
     * node's own position object in here rather than copying it, so mutating
     * `position.x` through a spec node would move the node on the live canvas
     * — action at a distance, from a function whose job is serialization.
     * Replacing the whole object (`n.position = { x, y }`, as `layoutSpec`
     * does) is the safe pattern and the only one the type permits.
     */
    position?: Readonly<{ x: number; y: number }>;
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

/**
 * Merges whose result depends on operand order. Each operand gets its own
 * named target handle on the card ("a"/"b" for Subtract, "in0", "in1", ...
 * for Concat and Stack), and the backend takes the operands in handle order.
 * Add, Multiply and Mean are order-free and keep a single handle.
 */
const ORDERED_MERGE_TYPES = new Set(['Subtract', 'Concat', 'Stack']);

/**
 * Canvas layer types the "Import from workflow" path converts.
 *
 * `Mean` is absent on purpose: on the canvas it reduces ONE tensor over a
 * dim, while the layer editor's `Mean` averages two or more branches. The
 * names collide; the operations do not.
 */
const KNOWN_LAYER_TYPES = new Set([
  'Conv2d', 'Conv1d', 'ConvTranspose2d',
  'BatchNorm2d', 'BatchNorm1d', 'LayerNorm', 'GroupNorm', 'InstanceNorm2d',
  'MaxPool2d', 'AvgPool2d', 'AdaptiveAvgPool2d',
  'Dropout', 'Dropout2d',
  'Linear', 'Embedding',
  'LSTM', 'GRU', 'RNN',
  'MultiHeadAttention', 'TransformerEncoder', 'TransformerDecoder',
  'Flatten', 'Reshape', 'SelectIndex',
  'ReLU', 'LeakyReLU', 'GELU', 'SiLU', 'Mish', 'ELU', 'SELU', 'PReLU',
  'Sigmoid', 'Tanh', 'Hardswish', 'Softmax',
  'Add', 'Concat', 'Multiply', 'Subtract', 'Stack',
]);

/** Map main-editor Activation node's `function` param → layers-editor layer type. */
const ACTIVATION_MAP: Record<string, string> = {
  relu: 'ReLU', leaky_relu: 'LeakyReLU', gelu: 'GELU', silu: 'SiLU',
  mish: 'Mish', elu: 'ELU', selu: 'SELU', prelu: 'PReLU',
  sigmoid: 'Sigmoid', tanh: 'Tanh', hardswish: 'Hardswish', softmax: 'Softmax',
};

export function isMergeType(t: string): boolean {
  return MERGE_TYPES.has(t);
}

export function isOrderedMergeType(t: string): boolean {
  return ORDERED_MERGE_TYPES.has(t);
}

/** Target handle id of operand `index` (0-based) on an order-sensitive merge. */
export function mergeOperandHandle(type: string, index: number): string {
  if (type === 'Subtract') return index === 0 ? 'a' : 'b';
  return `in${index}`;
}

/** Operand position a target handle stands for, or null when it names none. */
export function mergeOperandIndex(type: string, handle: string | null | undefined): number | null {
  if (typeof handle !== 'string') return null;
  if (type === 'Subtract') return handle === 'a' ? 0 : handle === 'b' ? 1 : null;
  const match = /^in(\d+)$/.exec(handle);
  return match ? Number(match[1]) : null;
}

/**
 * The target handles an order-sensitive merge card shows. Subtract always
 * has `a` and `b`. Concat and Stack show one handle per wired operand plus
 * one free handle to connect the next branch to, and never fewer than two.
 * Any other type returns [] (it keeps the single unnamed handle).
 */
export function mergeInputHandles(
  type: string,
  usedHandles: ReadonlyArray<string | null | undefined>,
): string[] {
  if (!ORDERED_MERGE_TYPES.has(type)) return [];
  if (type === 'Subtract') return ['a', 'b'];
  let highest = -1;
  for (const h of usedHandles) {
    const index = mergeOperandIndex(type, h);
    if (index !== null) highest = Math.max(highest, index);
  }
  const count = Math.max(2, highest + 2);
  return Array.from({ length: count }, (_, i) => `in${i}`);
}

/**
 * Give the incoming edges of every order-sensitive merge a named handle,
 * in edge-list order, when none of them has one. A spec saved before the
 * handles existed is built by the backend in edge-list order, so this
 * assignment shows on the card the order the model already computes in.
 * A merge whose edges already name handles is left alone.
 */
function assignMergeOperandHandles(nodes: GraphSpec['nodes'], edges: Edge[]): void {
  for (const n of nodes) {
    if (!ORDERED_MERGE_TYPES.has(n.type)) continue;
    const incoming = edges.filter((e) => e.target === n.id);
    if (incoming.some((e) => mergeOperandIndex(n.type, e.targetHandle) !== null)) continue;
    const limit = n.type === 'Subtract' ? 2 : incoming.length;
    incoming.slice(0, limit).forEach((e, i) => {
      e.targetHandle = mergeOperandHandle(n.type, i);
    });
  }
}

/**
 * Which slot of the layer palette a layer type belongs to (core#228).
 *
 * The one place a layer type is classified. Both the colour below and the
 * palette groups in `LayersEditorModal` read it, so the two cannot disagree
 * about what a `BatchNorm2d` is — which they previously could, because each
 * carried its own hex table and the only thing holding them together was a
 * comment saying "duplicated from".
 */
export type LayerCategory = keyof typeof LAYER_TYPE_COLORS;

const LAYER_CATEGORIES: Record<string, LayerCategory> = {
  Conv2d: 'Convolution', Conv1d: 'Convolution', ConvTranspose2d: 'Convolution',
  BatchNorm2d: 'Normalization', BatchNorm1d: 'Normalization',
  LayerNorm: 'Normalization', GroupNorm: 'Normalization',
  InstanceNorm2d: 'Normalization',
  MaxPool2d: 'Pooling', AvgPool2d: 'Pooling', AdaptiveAvgPool2d: 'Pooling',
  Dropout: 'Regularization', Dropout2d: 'Regularization',
  Linear: 'Linear', Embedding: 'Linear',
  LSTM: 'Recurrent', GRU: 'Recurrent', RNN: 'Recurrent',
  MultiHeadAttention: 'Attention',
  TransformerEncoder: 'Attention', TransformerDecoder: 'Attention',
  PositionalEncoding: 'Attention',
  Flatten: 'Utility', Reshape: 'Utility', SelectIndex: 'Utility',
  ReLU: 'Activation', LeakyReLU: 'Activation', GELU: 'Activation',
  SiLU: 'Activation', Mish: 'Activation', ELU: 'Activation',
  SELU: 'Activation', PReLU: 'Activation', Sigmoid: 'Activation',
  Tanh: 'Activation', Hardswish: 'Activation', Softmax: 'Activation',
  Input: 'Input',
  Output: 'Output',
};

export function layerCategoryOf(type: string): LayerCategory {
  if (MERGE_TYPES.has(type)) return 'Merge';
  // A type the palette does not list gets its own name rather than falling
  // into whichever category happens to share its colour.
  return LAYER_CATEGORIES[type] ?? 'Unknown';
}

/**
 * Colour for a layer type, straight off the token layer.
 *
 * Used to be a hex table here and a second, hand-synced one in
 * `LAYER_DEFS` — both on the pre-lift Material tones, so the layers editor
 * painted a different purple, blue, red and blue-grey than the rest of the
 * app. Now there is one table and it lives in `tokens.css`, where the
 * contrast gate can see it.
 */
export function colorForType(type: string): string {
  return LAYER_TYPE_COLORS[layerCategoryOf(type)];
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

/**
 * Something the import had to say about one canvas node. `error` stops the
 * import (the layer editor cannot express what the canvas node computes);
 * `warning` imports it and tells the user what changed. Carried as a locale
 * key plus values, like `ValidationError`, so the toast is in the user's
 * language.
 */
export interface ImportIssue {
  level: 'error' | 'warning';
  key: TranslationKey;
  params: Record<string, string | number>;
}

export interface WorkflowImport {
  spec: GraphSpec;
  issues: ImportIssue[];
}

/** Canvas port names of a two-input tensor op, in operand order. */
const CANVAS_OPERAND_PORTS: Record<string, number> = { tensor_a: 0, tensor_b: 1 };

function parseIntList(text: unknown): number[] | null {
  if (typeof text === 'number' && Number.isInteger(text)) return [text];
  if (typeof text !== 'string') return null;
  const parts = text.split(',').map((p) => p.trim()).filter((p) => p !== '');
  if (parts.length === 0 || parts.some((p) => !/^-?\d+$/.test(p))) return null;
  return parts.map(Number);
}

/**
 * Rewrite one canvas node's params into what the layer builder means by
 * the same layer, or report why it cannot be done. The canvas and the layer
 * editor share type names, and several params do not mean the same thing
 * on both sides; copying them verbatim built a different model.
 * Returns null when the node must not be imported.
 */
function translateCanvasParams(
  type: string,
  nodeId: string,
  params: Record<string, any>,
  issues: ImportIssue[],
): Record<string, any> | null {
  const out = { ...params };
  const refuse = (key: TranslationKey, extra: Record<string, string | number> = {}) => {
    issues.push({ level: 'error', key, params: { node: nodeId, ...extra } });
    return null;
  };
  const warn = (key: TranslationKey, extra: Record<string, string | number> = {}) => {
    issues.push({ level: 'warning', key, params: { node: nodeId, ...extra } });
  };

  switch (type) {
    case 'Embedding': {
      // The canvas spells "no padding" as -1. nn.Embedding reads a negative
      // padding_idx as counted from the end, so -1 zeroed the last row.
      const idx = Number(out.padding_idx);
      if (out.padding_idx === undefined || out.padding_idx === null || !(idx >= 0)) {
        delete out.padding_idx;
      }
      return out;
    }
    case 'Reshape': {
      // The canvas shape includes the batch dim; the layer's Reshape keeps
      // the incoming batch size and takes the remaining dims only.
      const shape = String(out.shape ?? '');
      const dims = parseIntList(shape);
      if (!dims || dims.length < 2) return refuse('layersEditor.import.issue.reshapeNoBatch', { shape });
      if (dims[0] !== -1) warn('layersEditor.import.issue.reshapeBatchDropped', { shape, batch: dims[0] });
      out.shape = dims.slice(1).join(',');
      return out;
    }
    case 'Add': {
      // The canvas Add computes a + alpha * b; the merge Add has no alpha.
      const alpha = out.alpha === undefined ? 1 : Number(out.alpha);
      if (alpha !== 1) return refuse('layersEditor.import.issue.addAlpha', { alpha: String(out.alpha) });
      delete out.alpha;
      return out;
    }
    case 'LayerNorm': {
      // The canvas stores "512" or "4,8"; nn.LayerNorm wants an int or a list.
      const dims = parseIntList(out.normalized_shape);
      if (!dims) {
        return refuse('layersEditor.import.issue.layerNormShape', { shape: String(out.normalized_shape) });
      }
      out.normalized_shape = dims.length === 1 ? dims[0] : dims;
      return out;
    }
    case 'AdaptiveAvgPool2d': {
      // The canvas has output_height / output_width; the builder takes output_size.
      const h = Number(out.output_height ?? 1);
      const w = Number(out.output_width ?? 1);
      delete out.output_height;
      delete out.output_width;
      out.output_size = h === w ? h : [h, w];
      return out;
    }
    case 'MultiHeadAttention': {
      // The layer block is always batch-first. A canvas node with
      // batch_first off read (seq, batch, embed), so say so.
      if (!out.batch_first) warn('layersEditor.import.issue.attentionBatchFirst');
      delete out.batch_first;
      return out;
    }
    default:
      return out;
  }
}

/**
 * Convert a canvas workflow's layer nodes into a layer-editor graph,
 * translating params the two sides spell differently and listing what
 * could not be carried over. The caller refuses the import when any issue
 * has level `error`.
 */
export function convertWorkflowForImport(workflow: WorkflowData): WorkflowImport {
  const issues: ImportIssue[] = [];
  const layerNodeIds = new Set<string>();
  const convertedNodes: GraphSpec['nodes'] = [];
  const typeById = new Map<string, string>();

  for (const n of workflow.nodes) {
    const rawType = n.type ?? '';
    let layerType = rawType;
    let params: Record<string, any> | null = { ...(n.data?.params ?? {}) };

    if (rawType === 'Activation') {
      const fn = params.function ?? 'relu';
      const mapped = ACTIVATION_MAP[fn];
      if (!mapped) continue;
      layerType = mapped;
      delete params.function;
    } else if (rawType === 'Mean') {
      issues.push({ level: 'warning', key: 'layersEditor.import.issue.meanSkipped', params: { node: n.id } });
      continue;
    } else if (!KNOWN_LAYER_TYPES.has(rawType)) {
      continue;
    } else {
      params = translateCanvasParams(rawType, n.id, params, issues);
      if (params === null) continue;
    }

    layerNodeIds.add(n.id);
    typeById.set(n.id, layerType);
    convertedNodes.push({
      id: n.id,
      type: layerType,
      params: Object.keys(params).length > 0 ? params : undefined,
      position: n.position,
    });
  }

  // Keep only edges between converted layer nodes. Canvas handles mean
  // nothing here, except the operand port of an order-sensitive merge
  // (tensor_a / tensor_b), which becomes that merge's operand handle.
  const convertedEdges: GraphSpec['edges'] = workflow.edges
    .filter((e) => layerNodeIds.has(e.source) && layerNodeIds.has(e.target))
    .map((e) => {
      // The filter above keeps only edges between converted nodes, all in typeById.
      const targetType = typeById.get(e.target)!;
      const operand = e.targetHandle !== undefined ? CANVAS_OPERAND_PORTS[e.targetHandle] : undefined;
      const targetHandle =
        ORDERED_MERGE_TYPES.has(targetType) && operand !== undefined
          ? mergeOperandHandle(targetType, operand)
          : null;
      return {
        id: e.id,
        source: e.source,
        sourceHandle: null,
        target: e.target,
        targetHandle,
      };
    });

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

  return { spec: { version: 2, nodes: convertedNodes, edges: convertedEdges }, issues };
}

/** `convertWorkflowForImport` without the issue list. */
export function convertWorkflowToGraphSpec(workflow: WorkflowData): GraphSpec {
  return convertWorkflowForImport(workflow).spec;
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

const LAYER_NODE_W = 160;
const LAYER_NODE_H = 40;
const LAYER_NODESEP = 40;
const LAYER_RANKSEP = 60;
function getLayerLayoutConfig(
  nodeCount: number,
): { nodesep: number; ranksep: number } {
  if (nodeCount > 50) return { nodesep: 28, ranksep: 42 };
  if (nodeCount > 25) return { nodesep: 32, ranksep: 50 };
  return { nodesep: LAYER_NODESEP, ranksep: LAYER_RANKSEP };
}

type LayerPos = { x: number; y: number; width: number; height: number };

// Column wrapping was removed in favor of the shared skip-aware valley pass
// (applyValleyPass with axis 'x'): a TB layer graph never splits into columns;
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
 * True when a node has no real position of its own: either the field is
 * absent, or it sits at the exact literal origin (the placeholder every node
 * falls back to when a spec omits `position` -- see `graphToFlow`'s mapping
 * below). Shared by the whole-spec gate and `assignPositionsFromTopology` so
 * the two always agree on which nodes need laying out.
 */
function needsPosition(n: GraphSpec['nodes'][number]): boolean {
  return !n.position || (n.position.x === 0 && n.position.y === 0);
}

/**
 * Use dagre to assign positions to the nodes of `spec` that lack one, leaving
 * any already-positioned node's coordinates untouched. Uses default
 * dimensions for every node (including already-positioned ones, which still
 * take part in the topology) since nodes haven't been rendered yet.
 */
function assignPositionsFromTopology(spec: GraphSpec): void {
  // Only ever called when needsLayout is true, which requires spec.nodes.length > 1.
  /* v8 ignore start */
  if (spec.nodes.length === 0) return;
  /* v8 ignore stop */

  const g = new dagre.graphlib.Graph();
  const cfg = getLayerLayoutConfig(spec.nodes.length);
  g.setGraph({
    rankdir: 'TB',
    nodesep: cfg.nodesep,
    ranksep: cfg.ranksep,
  });
  g.setDefaultEdgeLabel(() => ({}));

  for (const n of spec.nodes) {
    g.setNode(n.id, { width: LAYER_NODE_W, height: LAYER_NODE_H });
  }
  const nodeIds = new Set(spec.nodes.map((n) => n.id));
  for (const e of spec.edges) {
    if (nodeIds.has(e.source) && nodeIds.has(e.target)) {
      g.setEdge(e.source, e.target);
    }
  }

  dagre.layout(g);

  const raw = new Map<string, LayerPos>();
  for (const n of spec.nodes) {
    const pos = g.node(n.id);
    // Every node was registered via g.setNode above, so dagre always returns a pos.
    /* v8 ignore start */
    if (!pos) continue;
    /* v8 ignore stop */
    raw.set(n.id, {
      x: pos.x - LAYER_NODE_W / 2,
      y: pos.y - LAYER_NODE_H / 2,
      width: LAYER_NODE_W,
      height: LAYER_NODE_H,
    });
  }
  const wrapped = applyValleyPass(raw, spec.edges, {
    axis: 'x',
    skipPredicate: (e) => !isTriggerEdge(e),
  });
  for (const n of spec.nodes) {
    // Preserve nodes that already have a real position -- only the ones that
    // triggered needsLayout in the first place get the computed coordinates.
    if (!needsPosition(n)) continue;
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
export function autoLayoutLayers(
  nodes: Node<LayerNodeData>[],
  edges: Edge[],
): Node<LayerNodeData>[] {
  if (nodes.length === 0) return nodes;

  const g = new dagre.graphlib.Graph();
  const cfg = getLayerLayoutConfig(nodes.length);
  g.setGraph({
    rankdir: 'TB',
    nodesep: cfg.nodesep,
    ranksep: cfg.ranksep,
  });
  g.setDefaultEdgeLabel(() => ({}));

  for (const node of nodes) {
    const w = node.measured?.width ?? node.width ?? LAYER_NODE_W;
    const h = node.measured?.height ?? node.height ?? LAYER_NODE_H;
    g.setNode(node.id, { width: w, height: h });
  }

  const nodeIds = new Set(nodes.map((n) => n.id));
  for (const edge of edges) {
    if (nodeIds.has(edge.source) && nodeIds.has(edge.target)) {
      g.setEdge(edge.source, edge.target);
    }
  }

  dagre.layout(g);

  const raw = new Map<string, LayerPos>();
  for (const n of nodes) {
    const pos = g.node(n.id);
    // Every node was registered via g.setNode above, so dagre always returns a pos.
    /* v8 ignore start */
    if (!pos) continue;
    /* v8 ignore stop */
    const w = n.measured?.width ?? n.width ?? LAYER_NODE_W;
    const h = n.measured?.height ?? n.height ?? LAYER_NODE_H;
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

  // Auto-assign positions when ANY node lacks one (at origin or missing).
  // Per-node, not whole-spec: one already-positioned node must not block
  // layout for the rest, or they all fall through to the shared `{x:0,y:0}`
  // default below and land stacked on top of each other.
  const needsLayout = spec.nodes.length > 1 && spec.nodes.some(needsPosition);
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
  assignMergeOperandHandles(spec.nodes, edges);

  return { nodes, edges };
}

/**
 * The number on the SequentialModel card: how many nodes the Layers editor
 * shows for this `layers` value (#594). Same acceptance rule as
 * `graphToFlow`, so a version-2 spec counts every node, Input and Output
 * included. Anything else (not JSON, another version, the legacy layer
 * array, no value) counts 0: the backend cannot build it, and the editor
 * opens it on a blank starter graph instead of showing its contents.
 */
export function layerNodeCount(json: string | undefined): number {
  if (typeof json !== 'string') return 0;
  let spec: unknown;
  try {
    spec = JSON.parse(json);
  } catch {
    return 0;
  }
  if (typeof spec !== 'object' || spec === null) return 0;
  const { version, nodes, edges } = spec as Record<string, unknown>;
  return version === 2 && Array.isArray(nodes) && Array.isArray(edges) ? nodes.length : 0;
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
          color: colorForType('Input'),
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
          color: colorForType('Output'),
          ports: [{ id: outPortId, name: 'out' }],
          isBoundary: true,
        },
      },
    ],
    edges: [],
  };
}

/**
 * A failed rule, carried as its locale key plus the values that fill it.
 * These reach the student as a toast, which has to be in the language they
 * picked, so the sentence is built at render time: `t(error.key, error.params)`.
 */
export interface ValidationError {
  key: TranslationKey;
  params?: Record<string, string | number>;
}

export function validateGraph(nodes: Node<LayerNodeData>[], edges: Edge[]): ValidationError | null {
  const inputs = nodes.filter((n) => n.data.layerType === 'Input');
  const outputs = nodes.filter((n) => n.data.layerType === 'Output');
  if (inputs.length !== 1) return { key: 'layersEditor.validation.noInput' };
  if (outputs.length !== 1) return { key: 'layersEditor.validation.noOutput' };

  const input = inputs[0];
  const output = outputs[0];
  const inPorts = input.data.ports ?? [];
  const outPorts = output.data.ports ?? [];
  if (inPorts.length === 0) return { key: 'layersEditor.validation.inputNeedsPort' };
  if (outPorts.length === 0) return { key: 'layersEditor.validation.outputNeedsPort' };

  const inNames = inPorts.map((p) => p.name);
  if (new Set(inNames).size !== inNames.length) {
    return { key: 'layersEditor.validation.dupInputPorts' };
  }
  const outNames = outPorts.map((p) => p.name);
  if (new Set(outNames).size !== outNames.length) {
    return { key: 'layersEditor.validation.dupOutputPorts' };
  }

  // Output port: exactly 1 incoming
  for (const p of outPorts) {
    const count = edges.filter((e) => e.target === output.id && e.targetHandle === p.id).length;
    if (count !== 1) {
      return {
        key: 'layersEditor.validation.outputEdgeCount',
        params: { port: p.name, count },
      };
    }
  }

  // Input port: at least 1 outgoing
  for (const p of inPorts) {
    const count = edges.filter((e) => e.source === input.id && e.sourceHandle === p.id).length;
    if (count < 1) {
      return { key: 'layersEditor.validation.inputPortUnused', params: { port: p.name } };
    }
  }

  // Plain layers (non-merge, non-boundary): exactly 1 incoming
  for (const n of nodes) {
    if (n.data.isBoundary || n.data.isMerge) continue;
    const incoming = edges.filter((e) => e.target === n.id);
    if (incoming.length !== 1) {
      return {
        key: 'layersEditor.validation.layerEdgeCount',
        params: { layer: n.data.layerType, count: incoming.length },
      };
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
  if (visited !== nodes.length) return { key: 'layersEditor.validation.cycle' };

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
      return {
        key: 'layersEditor.validation.unreachable',
        params: { layer: n.data.layerType },
      };
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
      return {
        key: 'layersEditor.validation.cannotReachOutput',
        params: { layer: n.data.layerType },
      };
    }
  }

  return null;
}
