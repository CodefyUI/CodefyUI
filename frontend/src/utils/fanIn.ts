import type { Edge, Node } from '@xyflow/react';
import type { NodeData, NodeDefinition } from '../types';
import { buildFlowNode, generateId } from '.';
import { SWITCH_DEFAULT_INPUTS, SWITCH_MAX_INPUTS, SWITCH_OUTPUT } from './switchNode';

/**
 * Inputs fed by more than one wire, and the Switch that replaces them
 * (#562, #658).
 *
 * A data input takes one source. A graph saved before that rule can still
 * have several, which the run used to merge by edge order: the LAST wire in
 * the file won. Run and Export now refuse such a graph, and the fix here
 * keeps what it computed: the wires go into a new Switch in file order, and
 * its selector names the last one.
 */

/** One input with more than one wire into it, its wires in file order. */
export interface FanIn {
  target: string;
  handle: string;
  edges: Edge[];
}

function isTriggerWire(edge: Edge): boolean {
  return (
    edge.type === 'triggerEdge' ||
    (edge.data as { type?: string } | undefined)?.type === 'trigger' ||
    edge.sourceHandle === 'trigger' ||
    edge.targetHandle === '__trigger'
  );
}

/** Every data input with more than one wire, in the order they first appear. */
export function fanInInputs(edges: readonly Edge[]): FanIn[] {
  const byInput = new Map<string, FanIn>();
  for (const edge of edges) {
    if (isTriggerWire(edge) || !edge.targetHandle) continue;
    const key = `${edge.target}\n${edge.targetHandle}`;
    const entry = byInput.get(key) ?? { target: edge.target, handle: edge.targetHandle, edges: [] };
    entry.edges.push(edge);
    byInput.set(key, entry);
  }
  return [...byInput.values()].filter((entry) => entry.edges.length > 1);
}

/** Room for the card to the left of the input it feeds. */
const SWITCH_OFFSET_X = 240;

const wireStyle = { stroke: '#555', strokeWidth: 2 };

/**
 * `nodes` and `edges` with the wires into `fanIn` routed through a new
 * Switch, or null when `fanIn` has more wires than a Switch takes inputs.
 */
export function withSwitchFor(
  nodes: readonly Node<NodeData>[],
  edges: readonly Edge[],
  fanIn: FanIn,
  switchDefinition: NodeDefinition,
): { nodes: Node<NodeData>[]; edges: Edge[]; switchId: string } | null {
  const count = fanIn.edges.length;
  if (count > SWITCH_MAX_INPUTS) return null;
  const target = nodes.find((n) => n.id === fanIn.target);
  const position = target
    ? { x: target.position.x - SWITCH_OFFSET_X, y: target.position.y }
    : { x: 0, y: 0 };
  const built = buildFlowNode(switchDefinition, position);
  const switchNode: Node<NodeData> = {
    ...built,
    data: {
      ...built.data,
      params: {
        ...built.data.params,
        // The last wire in the file is the one the old rule read.
        selector: count - 1,
        // Room for every wire, and one empty input as wiring would leave.
        inputs: Math.min(SWITCH_MAX_INPUTS, Math.max(SWITCH_DEFAULT_INPUTS, count + 1)),
      },
    },
  };
  const replaced = new Set(fanIn.edges.map((e) => e.id));
  const intoSwitch: Edge[] = fanIn.edges.map((edge, index) => ({
    ...edge,
    id: generateId(),
    target: switchNode.id,
    targetHandle: `input_${index}`,
  }));
  const fromSwitch: Edge = {
    id: generateId(),
    source: switchNode.id,
    sourceHandle: SWITCH_OUTPUT,
    target: fanIn.target,
    targetHandle: fanIn.handle,
    animated: false,
    style: wireStyle,
  };
  return {
    nodes: [...nodes, switchNode],
    edges: [...edges.filter((e) => !replaced.has(e.id)), ...intoSwitch, fromSwitch],
    switchId: switchNode.id,
  };
}
