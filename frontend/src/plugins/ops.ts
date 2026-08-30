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
import type { NodeData, NodeDefinition, ParamDefinition, SegmentGroup } from '../types';
import {
  buildFlowNode, buildNoteNode, DEFAULT_NOTE_COLOR, generateId, isValidConnection,
} from '../utils';
import { autoLayout } from '../utils/autoLayout';
import { computeSegmentNodes } from '../utils/segmentPath';

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
  | { op: 'auto_layout' }
  /**
   * Put one node at an exact position (#342). Notes bound to it ride along by
   * the same delta, the way they do when the user drags it -- an agent that
   * tidies a corner and strands three sticky notes has not tidied anything.
   */
  | { op: 'move_node'; node_id: string; position: { x: number; y: number } }
  /**
   * Create or replace a Teaching Inspector segment: the orange bubble the
   * canvas draws around every node on a data path from head to tail (#342).
   * Membership is derived, not stored, so head and tail have to be joined by
   * data edges or the overlay would render nothing.
   */
  | { op: 'set_segment'; segment_id?: string; head_node_id: string; tail_node_id: string }
  | { op: 'remove_segment'; segment_id: string }
  /**
   * Add a text note -- the sticky the canvas already renders, and which the
   * validator, the exporters and every fingerprint already skip (#342).
   */
  | { op: 'add_note'; ref?: string; text: string;
      position?: { x: number; y: number }; color?: string; bind_to?: string }
  | { op: 'update_note'; node_id: string; text?: string; color?: string }
  /** Name one node. The label is metadata beside `params`, never inside it. */
  | { op: 'set_node_meta'; node_id: string; label: string };

export interface OpResult {
  index: number;
  ok: boolean;
  error?: string;
  node_id?: string;
  /** Set by `set_segment`, which may have generated the id itself. */
  segment_id?: string;
}

export interface ApplyOutcome {
  nodes: Node<NodeData>[];
  edges: Edge[];
  segmentGroups: SegmentGroup[];
  results: OpResult[];
  refs: Record<string, string>;
  dirtyIds: string[];
  mutated: boolean;
}

/** Longest note an op may write. An agent inlining a document is not a note. */
const NOTE_TEXT_MAX = 4000;
/** Longest node label. Long enough for a sentence fragment, short enough for a card. */
const NODE_LABEL_MAX = 120;
/**
 * The six colours the canvas's note menu offers are all `#rrggbb`
 * (`NodeContextMenu.tsx:33-40`), so one pattern covers "a palette colour" and
 * "a hex colour" without importing a component's private constant.
 */
const NOTE_COLOR_RE = /^#[0-9a-fA-F]{6}$/;
/** Where an unpositioned bound note lands, relative to the node it explains. */
const NOTE_BIND_OFFSET = { x: 240, y: -20 };

/**
 * A bound note's offset from the node it explains, derived from where the
 * note actually sits. One function so `add_note`'s initial binding and
 * `move_node`'s re-derivation cannot drift: they are the same rule, and a
 * disagreement between them shows up only as a note that jumps on the next
 * drag of its parent.
 */
function boundOffsetFrom(
  notePosition: { x: number; y: number },
  parentPosition: { x: number; y: number },
): { x: number; y: number } {
  return { x: notePosition.x - parentPosition.x, y: notePosition.y - parentPosition.y };
}

/**
 * True when `text` holds a C0 control character or DEL.
 *
 * Written as a codepoint scan rather than a regex character class on
 * purpose: a class of `\u00xx` escapes is the kind of literal that gets
 * mangled by one careless copy-paste and then silently accepts nothing, or
 * everything. Tab, newline and carriage return are legitimate in a note and
 * are the only three let through.
 */
function hasControlChars(text: string): boolean {
  for (const ch of text) {
    const code = ch.codePointAt(0)!;
    if (code === 9 || code === 10 || code === 13) continue;
    if (code < 0x20 || code === 0x7f) return true;
  }
  return false;
}

function validateNoteText(text: unknown): string | null {
  if (typeof text !== 'string') return 'text must be a string';
  if (text.length < 1 || text.length > NOTE_TEXT_MAX) {
    return `text must be 1..${NOTE_TEXT_MAX} characters`;
  }
  if (hasControlChars(text)) {
    return 'text may not contain control characters (newline and tab are fine)';
  }
  return null;
}

function validateNoteColor(color: unknown): string | null {
  if (typeof color !== 'string' || !NOTE_COLOR_RE.test(color)) {
    return "color must be a '#rrggbb' hex string";
  }
  return null;
}

function validateNodeLabel(label: unknown): string | null {
  if (typeof label !== 'string') return 'label must be a string';
  if (/[\r\n]/.test(label)) return 'label must be a single line';
  const trimmed = label.trim();
  if (trimmed.length < 1 || trimmed.length > NODE_LABEL_MAX) {
    return `label must be 1..${NODE_LABEL_MAX} characters`;
  }
  return null;
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
      // model_file / image_file / data_file / tensor_grid carry
      // editor-managed payloads; accept whatever the caller sends.
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
  current: { nodes: Node<NodeData>[]; edges: Edge[]; segmentGroups: SegmentGroup[] },
  definitions: NodeDefinition[],
  ops: GraphOp[],
): ApplyOutcome {
  let nodes = [...current.nodes];
  let edges = [...current.edges];
  // NOT copied, unlike the two above. The commit path writes whatever comes
  // back, and the autosave record cache compares this array by reference, so
  // a fresh (identical) list on every batch of add_node would rewrite the
  // tab's stored record for nothing.
  let segmentGroups = current.segmentGroups;
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
        // The rule the Delete key follows (tabStore.ts:2206-2237): a group
        // whose head or tail has just been deleted renders nothing and would
        // otherwise be written to the saved file for good.
        const keptSegments = segmentGroups.filter(
          (s) => s.headNodeId !== id && s.tailNodeId !== id,
        );
        if (keptSegments.length !== segmentGroups.length) segmentGroups = keptSegments;
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
        // Every group named nodes that are gone. Guarded so an already-empty
        // list comes back by reference.
        if (segmentGroups.length > 0) segmentGroups = [];
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

      case 'move_node': {
        const id = resolveId(op.node_id);
        if (!id) return fail(`move_node: unknown node '${op.node_id}'`);
        const target = op.position;
        if (
          !target
          || !Number.isFinite(target.x)
          || !Number.isFinite(target.y)
        ) {
          return fail('move_node: position must be two finite numbers');
        }
        const moved = nodes.find((n) => n.id === id)!;
        const dx = target.x - moved.position.x;
        const dy = target.y - moved.position.y;
        nodes = nodes.map((n) => {
          if (n.id === id) {
            const next = { ...n, position: { x: target.x, y: target.y } };
            // Mirrors `onNodesChange`'s FIRST pass (tabStore.ts:2133-2150): a
            // bound note that is itself moved re-derives its offset, or the
            // next drag of its parent would snap it back to where it sat
            // before -- reading as if the plugin's edit had been undone.
            if (n.type === 'noteNode' && n.data.boundToNodeId && n.data.boundOffset) {
              const parent = nodes.find((p) => p.id === n.data.boundToNodeId);
              if (parent) {
                next.data = {
                  ...n.data,
                  boundOffset: boundOffsetFrom(target, parent.position),
                };
              }
            }
            return next;
          }
          // Mirrors `onNodesChange`'s second pass (tabStore.ts:2152-2178): a
          // bound note follows its parent. By DELTA rather than by
          // re-deriving from `boundOffset`, so a note the user has nudged
          // keeps the offset they nudged it to.
          if (n.type === 'noteNode' && n.data.boundToNodeId === id && n.data.boundOffset) {
            return { ...n, position: { x: n.position.x + dx, y: n.position.y + dy } };
          }
          return n;
        });
        mutated = true;
        results.push({ index, ok: true, node_id: id });
        return;
      }

      case 'set_segment': {
        const headId = resolveId(op.head_node_id);
        const tailId = resolveId(op.tail_node_id);
        if (!headId) return fail(`set_segment: unknown head node '${op.head_node_id}'`);
        if (!tailId) return fail(`set_segment: unknown tail node '${op.tail_node_id}'`);
        const head = nodes.find((n) => n.id === headId)!;
        const tail = nodes.find((n) => n.id === tailId)!;
        if (head.type === 'noteNode' || tail.type === 'noteNode') {
          return fail('set_segment: a note node cannot be a segment endpoint');
        }
        // The overlay derives its members by BFS over data edges and draws
        // nothing when the tail is unreachable, so an unreachable pair is a
        // segment the user would never see. Refused with the reason rather
        // than stored invisibly.
        if (computeSegmentNodes(headId, tailId, nodes, edges).size === 0) {
          return fail(
            `set_segment: no data-edge path from '${op.head_node_id}' to '${op.tail_node_id}'`,
          );
        }
        const id = op.segment_id ?? generateId();
        segmentGroups = [
          ...segmentGroups.filter((s) => s.id !== id),
          { id, headNodeId: headId, tailNodeId: tailId },
        ];
        mutated = true;
        results.push({ index, ok: true, segment_id: id });
        return;
      }

      case 'remove_segment': {
        if (!segmentGroups.some((s) => s.id === op.segment_id)) {
          return fail(`remove_segment: unknown segment '${op.segment_id}'`);
        }
        segmentGroups = segmentGroups.filter((s) => s.id !== op.segment_id);
        mutated = true;
        results.push({ index, ok: true });
        return;
      }

      case 'add_note': {
        const textError = validateNoteText(op.text);
        if (textError) return fail(`add_note: ${textError}`);
        if (op.color !== undefined) {
          const colorError = validateNoteColor(op.color);
          if (colorError) return fail(`add_note: ${colorError}`);
        }
        let bound: Node<NodeData> | undefined;
        if (op.bind_to !== undefined) {
          const boundId = resolveId(op.bind_to);
          if (!boundId) return fail(`add_note: unknown node '${op.bind_to}'`);
          bound = nodes.find((n) => n.id === boundId)!;
          if (bound.type === 'noteNode') {
            return fail('add_note: a note cannot be bound to another note');
          }
        }
        const position = op.position
          ?? (bound
            ? {
                x: bound.position.x + NOTE_BIND_OFFSET.x,
                y: bound.position.y + NOTE_BIND_OFFSET.y,
              }
            : { x: 160 + (staggered % 4) * 90, y: 120 + staggered * 70 });
        staggered += 1;
        const note = buildNoteNode({
          text: op.text,
          position,
          color: op.color ?? DEFAULT_NOTE_COLOR,
          boundToNodeId: bound ? bound.id : null,
          // Derived from where the note actually ends up, so an explicit
          // `position` and the default one both leave a consistent binding.
          boundOffset: bound ? boundOffsetFrom(position, bound.position) : null,
        });
        nodes = [...nodes, note];
        if (op.ref) refs[op.ref] = note.id;
        mutated = true;
        results.push({ index, ok: true, node_id: note.id });
        return;
      }

      case 'update_note': {
        const id = resolveId(op.node_id);
        if (!id) return fail(`update_note: unknown node '${op.node_id}'`);
        const note = nodes.find((n) => n.id === id)!;
        if (note.type !== 'noteNode') {
          return fail(`update_note: node '${op.node_id}' is not a note`);
        }
        if (op.text === undefined && op.color === undefined) {
          return fail('update_note: nothing to change — pass text, color, or both');
        }
        if (op.text !== undefined) {
          const textError = validateNoteText(op.text);
          if (textError) return fail(`update_note: ${textError}`);
          // An image note's `noteContent` is its data URL. Writing prose over
          // it would blank the picture, and the 4000-char cap is not a
          // meaningful guard for a field that is meant to hold base64.
          if (note.data.noteKind === 'image') {
            return fail('update_note: cannot set text on an image note');
          }
        }
        if (op.color !== undefined) {
          const colorError = validateNoteColor(op.color);
          if (colorError) return fail(`update_note: ${colorError}`);
        }
        nodes = nodes.map((n) =>
          n.id === id
            ? {
                ...n,
                data: {
                  ...n.data,
                  ...(op.text !== undefined ? { noteContent: op.text } : {}),
                  ...(op.color !== undefined ? { noteColor: op.color } : {}),
                },
              }
            : n,
        );
        mutated = true;
        results.push({ index, ok: true, node_id: id });
        return;
      }

      case 'set_node_meta': {
        const id = resolveId(op.node_id);
        if (!id) return fail(`set_node_meta: unknown node '${op.node_id}'`);
        const target = nodes.find((n) => n.id === id)!;
        if (target.type === 'noteNode') {
          return fail('set_node_meta: a note has no label — use update_note');
        }
        const labelError = validateNodeLabel(op.label);
        if (labelError) return fail(`set_node_meta: ${labelError}`);
        const label = op.label.trim();
        nodes = nodes.map((n) =>
          n.id === id ? { ...n, data: { ...n.data, label } } : n,
        );
        mutated = true;
        results.push({ index, ok: true, node_id: id });
        return;
      }

      default:
        fail(`Unknown op '${(op as { op?: string }).op}'`);
    }
  });

  return {
    nodes,
    edges,
    segmentGroups,
    results,
    refs,
    dirtyIds: [...dirty].filter((id) => nodes.some((n) => n.id === id)),
    mutated,
  };
}
