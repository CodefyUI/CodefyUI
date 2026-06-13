/**
 * Pure graph-operation reducer behind CodefyUIPluginAPI.graph.applyOperations.
 *
 * Operates on copies of the active tab's nodes/edges and never touches the
 * store — the commit wrapper in ./api.ts handles the undo snapshot and the
 * store write. Failing ops are skipped (and reported) rather than aborting
 * the batch, so an agent driving this API can self-correct from per-op
 * errors.
 */
import type { Edge, Node } from '@xyflow/react';
import type { NodeData, NodeDefinition, ParamDefinition } from '../types';
import { buildFlowNode, generateId, isValidConnection } from '../utils';
import { autoLayout } from '../utils/autoLayout';

export type GraphOp =
  | { op: 'add_node'; node_type: string; ref?: string;
      params?: Record<string, unknown>; position?: { x: number; y: number } }
  | { op: 'connect'; source: string; source_handle: string;
      target: string; target_handle: string }
  | { op: 'set_params'; node_id: string; params: Record<string, unknown> }
  | { op: 'remove_node'; node_id: string }
  | { op: 'remove_edge'; source: string; target: string;
      source_handle?: string; target_handle?: string }
  | { op: 'clear_graph' }
  | { op: 'auto_layout' };

export interface OpResult {
  index: number;
  ok: boolean;
  error?: string;
  node_id?: string;
}

export interface ApplyOutcome {
  nodes: Node<NodeData>[];
  edges: Edge[];
  results: OpResult[];
  refs: Record<string, string>;
  dirtyIds: string[];
  mutated: boolean;
}

function validateParamValue(p: ParamDefinition, value: unknown): string | null {
  switch (p.param_type) {
    case 'int':
      if (typeof value !== 'number' || !Number.isInteger(value)) {
        return `param '${p.name}' expects an integer`;
      }
      break;
    case 'float':
      if (typeof value !== 'number' || Number.isNaN(value)) {
        return `param '${p.name}' expects a number`;
      }
      break;
    case 'bool':
      if (typeof value !== 'boolean') return `param '${p.name}' expects a boolean`;
      break;
    case 'select':
      if (typeof value !== 'string' || !p.options.includes(value)) {
        return `param '${p.name}' must be one of: ${p.options.join(', ')}`;
      }
      break;
    case 'string':
      if (typeof value !== 'string') return `param '${p.name}' expects a string`;
      break;
    default:
      // model_file / image_file / tensor_grid carry editor-managed payloads;
      // accept whatever the caller sends.
      return null;
  }
  if (typeof value === 'number') {
    if (p.min_value !== null && value < p.min_value) {
      return `param '${p.name}' must be >= ${p.min_value}`;
    }
    if (p.max_value !== null && value > p.max_value) {
      return `param '${p.name}' must be <= ${p.max_value}`;
    }
  }
  return null;
}

function validateParams(
  def: NodeDefinition,
  params: Record<string, unknown>,
): string | null {
  for (const [name, value] of Object.entries(params)) {
    const pd = def.params.find((p) => p.name === name);
    if (!pd) {
      const known = def.params.map((p) => p.name).join(', ') || '(none)';
      return `unknown param '${name}' for ${def.node_name}; known params: ${known}`;
    }
    const err = validateParamValue(pd, value);
    if (err) return err;
  }
  return null;
}

export function applyGraphOps(
  current: { nodes: Node<NodeData>[]; edges: Edge[] },
  definitions: NodeDefinition[],
  ops: GraphOp[],
): ApplyOutcome {
  let nodes = [...current.nodes];
  let edges = [...current.edges];
  const results: OpResult[] = [];
  const refs: Record<string, string> = {};
  const dirty = new Set<string>();
  let mutated = false;
  let staggered = 0;

  const defByName = new Map(definitions.map((d) => [d.node_name, d]));
  const resolveId = (idOrRef: string): string | null => {
    const viaRef = refs[idOrRef];
    if (viaRef && nodes.some((n) => n.id === viaRef)) return viaRef;
    return nodes.some((n) => n.id === idOrRef) ? idOrRef : null;
  };

  ops.forEach((op, index) => {
    const fail = (error: string) => results.push({ index, ok: false, error });

    switch (op.op) {
      case 'add_node': {
        const def = defByName.get(op.node_type);
        if (!def) {
          fail(`Unknown node type '${op.node_type}' — use exact names from the node catalog`);
          return;
        }
        if (op.params) {
          const err = validateParams(def, op.params);
          if (err) {
            fail(err);
            return;
          }
        }
        const position = op.position ?? { x: 160 + (staggered % 4) * 90, y: 120 + staggered * 70 };
        staggered += 1;
        const node = buildFlowNode(def, position);
        if (op.params) {
          node.data.params = { ...node.data.params, ...op.params };
        }
        nodes = [...nodes, node];
        if (op.ref) refs[op.ref] = node.id;
        dirty.add(node.id);
        mutated = true;
        results.push({ index, ok: true, node_id: node.id });
        return;
      }

      case 'connect': {
        const sourceId = resolveId(op.source);
        const targetId = resolveId(op.target);
        if (!sourceId) return fail(`connect: unknown source node '${op.source}'`);
        if (!targetId) return fail(`connect: unknown target node '${op.target}'`);
        const sourceNode = nodes.find((n) => n.id === sourceId)!;
        const targetNode = nodes.find((n) => n.id === targetId)!;
        if (sourceNode.type === 'noteNode' || targetNode.type === 'noteNode') {
          return fail('connect: note nodes cannot be connected');
        }

        const isTrigger = op.source_handle === 'trigger';
        const targetHandle = isTrigger ? '__trigger' : op.target_handle;

        if (!isTrigger) {
          const sDef = sourceNode.data.definition;
          const tDef = targetNode.data.definition;
          if (sDef) {
            const out = sDef.outputs.find((o) => o.name === op.source_handle);
            if (!out) {
              const names = sDef.outputs.map((o) => o.name).join(', ') || '(none)';
              return fail(`connect: '${sDef.node_name}' has no output '${op.source_handle}'; outputs: ${names}`);
            }
            if (tDef) {
              const inp = tDef.inputs.find((i) => i.name === op.target_handle);
              if (!inp) {
                const names = tDef.inputs.map((i) => i.name).join(', ') || '(none)';
                return fail(`connect: '${tDef.node_name}' has no input '${op.target_handle}'; inputs: ${names}`);
              }
              if (!isValidConnection(out.data_type, inp.data_type)) {
                return fail(`connect: incompatible types ${out.data_type} -> ${inp.data_type}`);
              }
            }
          }
        }

        const duplicate = edges.some(
          (e) => e.source === sourceId && e.target === targetId
            && (e.sourceHandle ?? '') === op.source_handle
            && (e.targetHandle ?? '') === targetHandle,
        );
        if (duplicate) return fail('connect: edge already exists');

        const edge: Edge = isTrigger
          ? { id: generateId(), source: sourceId, target: targetId,
              sourceHandle: 'trigger', targetHandle: '__trigger',
              animated: false, type: 'triggerEdge', data: { type: 'trigger' } }
          : { id: generateId(), source: sourceId, target: targetId,
              sourceHandle: op.source_handle, targetHandle,
              animated: false, style: { stroke: '#555', strokeWidth: 2 } };
        edges = [...edges, edge];
        dirty.add(targetId);
        mutated = true;
        results.push({ index, ok: true });
        return;
      }

      case 'set_params': {
        const id = resolveId(op.node_id);
        if (!id) return fail(`set_params: unknown node '${op.node_id}'`);
        const node = nodes.find((n) => n.id === id)!;
        const def = node.data.definition;
        if (def) {
          const err = validateParams(def, op.params);
          if (err) return fail(err);
        }
        nodes = nodes.map((n) =>
          n.id === id
            ? { ...n, data: { ...n.data, params: { ...n.data.params, ...op.params } } }
            : n,
        );
        dirty.add(id);
        mutated = true;
        results.push({ index, ok: true, node_id: id });
        return;
      }

      case 'remove_node': {
        const id = resolveId(op.node_id);
        if (!id) return fail(`remove_node: unknown node '${op.node_id}'`);
        nodes = nodes
          .filter((n) => n.id !== id)
          .map((n) =>
            n.type === 'noteNode' && n.data.boundToNodeId === id
              ? { ...n, data: { ...n.data, boundToNodeId: null, boundOffset: null } }
              : n,
          );
        edges = edges.filter((e) => e.source !== id && e.target !== id);
        mutated = true;
        results.push({ index, ok: true });
        return;
      }

      case 'remove_edge': {
        const sourceId = resolveId(op.source);
        const targetId = resolveId(op.target);
        if (!sourceId || !targetId) {
          return fail('remove_edge: unknown source or target node');
        }
        const matches = edges.filter(
          (e) => e.source === sourceId && e.target === targetId
            && (op.source_handle === undefined || (e.sourceHandle ?? '') === op.source_handle)
            && (op.target_handle === undefined || (e.targetHandle ?? '') === op.target_handle),
        );
        if (matches.length === 0) return fail('remove_edge: no matching edge');
        const drop = new Set(matches.map((e) => e.id));
        edges = edges.filter((e) => !drop.has(e.id));
        mutated = true;
        results.push({ index, ok: true });
        return;
      }

      case 'clear_graph': {
        nodes = [];
        edges = [];
        for (const k of Object.keys(refs)) delete refs[k];
        mutated = true;
        results.push({ index, ok: true });
        return;
      }

      case 'auto_layout': {
        nodes = autoLayout(nodes, edges, 'all') as Node<NodeData>[];
        mutated = true;
        results.push({ index, ok: true });
        return;
      }

      default:
        fail(`Unknown op '${(op as { op?: string }).op}'`);
    }
  });

  return {
    nodes,
    edges,
    results,
    refs,
    dirtyIds: [...dirty].filter((id) => nodes.some((n) => n.id === id)),
    mutated,
  };
}
