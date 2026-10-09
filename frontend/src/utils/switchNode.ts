import type { Edge, Node } from '@xyflow/react';
import type { NodeData, PortDefinition } from '../types';

/**
 * The Switch node's ports and the graph rules its ports cannot express
 * (#655). Mirrors `backend/app/nodes/dataflow/switch_node.py`.
 *
 * - `inputs` sets how many `input_N` ports a Switch has. A saved graph
 *   without it gets the default, 4: the ports Switch had before the param.
 * - Wiring the last empty input adds another (`grownSwitchParams`).
 * - The output carries the one type its wired inputs carry
 *   (`switchOutputType`); the backend's `validate_graph` refuses the same
 *   graphs this module refuses on the canvas.
 */

export const SWITCH_TYPE = 'Switch';
export const SWITCH_MIN_INPUTS = 2;
export const SWITCH_MAX_INPUTS = 16;
export const SWITCH_DEFAULT_INPUTS = 4;
export const SWITCH_SELECTOR = 'selector';
export const SWITCH_OUTPUT = 'output';

const INPUT_PREFIX = 'input_';

function bareName(qualifiedName: string | undefined): string {
  const name = qualifiedName ?? '';
  const idx = name.lastIndexOf(':');
  return idx >= 0 ? name.slice(idx + 1) : name;
}

/** Whether `node` is a Switch, by its definition's name or its type. */
export function isSwitchNode(node: Pick<Node<NodeData>, 'data'> | undefined): boolean {
  if (!node?.data) return false;
  return bareName(node.data.definition?.node_name ?? node.data.type) === SWITCH_TYPE;
}

/** The input count `params` name, clamped as `resolve_count_param` clamps it. */
export function switchInputCount(params: Record<string, unknown> | undefined): number {
  const raw = params?.inputs ?? SWITCH_DEFAULT_INPUTS;
  const parsed =
    typeof raw === 'boolean'
      ? NaN
      : typeof raw === 'number'
        ? Math.floor(raw)
        : parseInt(String(raw), 10);
  const count = Number.isFinite(parsed) ? parsed : SWITCH_DEFAULT_INPUTS;
  return Math.max(SWITCH_MIN_INPUTS, Math.min(SWITCH_MAX_INPUTS, count));
}

/** `input_3` -> 3; anything else -> null. */
export function switchInputIndex(handle: string | null | undefined): number | null {
  if (!handle?.startsWith(INPUT_PREFIX)) return null;
  const rest = handle.slice(INPUT_PREFIX.length);
  return /^\d+$/.test(rest) ? Number(rest) : null;
}

/** The live input ports: the options first, the selector last (see the backend). */
export function switchInputs(params: Record<string, unknown> | undefined): PortDefinition[] {
  return [
    ...Array.from({ length: switchInputCount(params) }, (_, i) => ({
      name: `${INPUT_PREFIX}${i}`,
      data_type: 'ANY',
      description: `Option ${i}`,
      optional: true,
    })),
    {
      name: SWITCH_SELECTOR,
      data_type: 'SCALAR',
      description: 'Index of the input to forward; overrides the selector param when wired',
      optional: true,
    },
  ];
}

/**
 * The params a Switch takes once a wire lands on `targetHandle`: one more
 * input when that was its last, else null. Never past the maximum.
 */
export function grownSwitchParams(
  node: Pick<Node<NodeData>, 'data'> | undefined,
  targetHandle: string | null | undefined,
): Record<string, unknown> | null {
  if (!node || !isSwitchNode(node)) return null;
  const params = node.data.params ?? {};
  const count = switchInputCount(params);
  if (switchInputIndex(targetHandle) !== count - 1 || count >= SWITCH_MAX_INPUTS) return null;
  return { ...params, inputs: count + 1 };
}

/** What a node's output port is typed, a Switch's by what feeds it. */
function outputTypeOf(
  node: Node<NodeData> | undefined,
  port: string | null | undefined,
  byId: Map<string, Node<NodeData>>,
  edges: readonly Edge[],
  outputsOf: (node: Node<NodeData>) => PortDefinition[],
  visiting: Set<string>,
): string {
  if (!node) return 'ANY';
  if (isSwitchNode(node) && (port ?? '') === SWITCH_OUTPUT) {
    return switchTypeOf(node.id, byId, edges, outputsOf, visiting, null);
  }
  return outputsOf(node).find((o) => o.name === port)?.data_type ?? 'ANY';
}

function switchTypeOf(
  switchId: string,
  byId: Map<string, Node<NodeData>>,
  edges: readonly Edge[],
  outputsOf: (node: Node<NodeData>) => PortDefinition[],
  visiting: Set<string>,
  ignoreHandle: string | null,
): string {
  if (visiting.has(switchId)) return 'ANY';
  const next = new Set(visiting).add(switchId);
  const types = new Set<string>();
  for (const edge of edges) {
    if (edge.target !== switchId || switchInputIndex(edge.targetHandle) === null) continue;
    if (ignoreHandle !== null && edge.targetHandle === ignoreHandle) continue;
    const type = outputTypeOf(byId.get(edge.source), edge.sourceHandle, byId, edges, outputsOf, next);
    if (type !== 'ANY') types.add(type);
  }
  return types.size === 1 ? [...types][0] : 'ANY';
}

/**
 * The type a Switch's output carries: the one concrete type its wired inputs
 * carry, else `ANY`. `ignoreHandle` leaves out the wires into one input, for
 * asking what a new wire into it would have to match.
 */
export function switchOutputType(
  switchId: string,
  nodes: readonly Node<NodeData>[],
  edges: readonly Edge[],
  outputsOf: (node: Node<NodeData>) => PortDefinition[],
  ignoreHandle: string | null = null,
): string {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  return switchTypeOf(switchId, byId, edges, outputsOf, new Set(), ignoreHandle);
}

/** The type `node.port` carries in this graph: its declared one, or a Switch's inferred one. */
export function liveOutputType(
  node: Node<NodeData> | undefined,
  port: string | null | undefined,
  nodes: readonly Node<NodeData>[],
  edges: readonly Edge[],
  outputsOf: (node: Node<NodeData>) => PortDefinition[],
): string {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  return outputTypeOf(node, port, byId, edges, outputsOf, new Set());
}

/**
 * Whether a wire carrying `sourceType` may land on a Switch's option input
 * `targetHandle`: it must match the type the Switch's other inputs carry,
 * and every port the Switch already feeds must accept it.
 */
export function switchAcceptsType(
  switchNode: Node<NodeData>,
  targetHandle: string | null | undefined,
  sourceType: string,
  nodes: readonly Node<NodeData>[],
  edges: readonly Edge[],
  outputsOf: (node: Node<NodeData>) => PortDefinition[],
  inputsOf: (node: Node<NodeData>) => PortDefinition[],
  isCompatible: (source: string, target: string) => boolean,
  visiting: ReadonlySet<string> = new Set(),
): boolean {
  if (!targetHandle || switchInputIndex(targetHandle) === null || sourceType === 'ANY') return true;
  if (visiting.has(switchNode.id)) return true;
  const next = new Set(visiting).add(switchNode.id);
  const others = switchOutputType(switchNode.id, nodes, edges, outputsOf, targetHandle);
  if (others !== 'ANY' && others !== sourceType) return false;
  const byId = new Map(nodes.map((n) => [n.id, n]));
  for (const edge of edges) {
    if (edge.source !== switchNode.id || (edge.sourceHandle ?? '') !== SWITCH_OUTPUT) continue;
    const target = byId.get(edge.target);
    if (!target) continue;
    const port = inputsOf(target).find((i) => i.name === edge.targetHandle);
    if (!port) continue;
    if (isSwitchNode(target)) {
      if (
        !switchAcceptsType(
          target, edge.targetHandle, sourceType, nodes, edges, outputsOf, inputsOf, isCompatible, next,
        )
      ) {
        return false;
      }
      continue;
    }
    if (!isCompatible(sourceType, port.data_type)) return false;
  }
  return true;
}
