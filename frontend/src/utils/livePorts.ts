import type { Edge, Node } from '@xyflow/react';
import type { DynamicPortsInfo, NodeData, NodeDefinition, PortDefinition } from '../types';
import {
  COMPOSE_MAX_STEPS, COMPOSE_MIN_STEPS, SCRIPT_MAX_PORTS, SPLIT_MAX_CHUNKS,
  resolveDynamicInputs, resolveDynamicOutputs,
} from './index';
import { SWITCH_MAX_INPUTS, SWITCH_MIN_INPUTS, SWITCH_SELECTOR, SWITCH_TYPE } from './switchNode';

/**
 * The ports one node instance has right now: its definition expanded by its
 * own params (PythonScript's `input_ports`/`output_ports`, Split's `chunks`,
 * ComposeTransform's `steps`, Switch's cases). The canvas draws these
 * handles; the plugin ops check a `connect` against the same answer (round 6
 * H1), so a port the canvas shows is a port an agent can wire.
 */
export function livePorts(
  node: Pick<Node<NodeData>, 'data'>,
  params: Record<string, unknown> | undefined = node.data.params,
): { inputs: PortDefinition[]; outputs: PortDefinition[] } {
  const def = node.data.definition;
  return {
    inputs: resolveDynamicInputs(def, params),
    outputs: resolveDynamicOutputs(def, params),
  };
}

export const NO_STALE_EDGES: ReadonlySet<string> = new Set<string>();

/**
 * Ids of edges a param edit would leave hanging off a handle that no longer
 * exists (core#131).
 *
 * Only ports the resolver itself produced are candidates, so trigger handles
 * and preset ports — which this knows nothing about — are never touched. For
 * every node whose ports do NOT depend on params the two resolvers hand back
 * the definition's own arrays, so the identity check below settles it in two
 * comparisons; the set work only runs for the handful of nodes the two
 * resolvers actually expand (Split, PythonScript, ComposeTransform today).
 */
export function staleEdges(
  node: Node<NodeData> | undefined,
  nextParams: Record<string, unknown>,
  edges: Edge[],
): ReadonlySet<string> {
  const def = node?.data.definition;
  if (!node || !def) return NO_STALE_EDGES;

  const inputsBefore = resolveDynamicInputs(def, node.data.params);
  const inputsAfter = resolveDynamicInputs(def, nextParams);
  const outputsBefore = resolveDynamicOutputs(def, node.data.params);
  const outputsAfter = resolveDynamicOutputs(def, nextParams);
  if (inputsBefore === inputsAfter && outputsBefore === outputsAfter) {
    return NO_STALE_EDGES;
  }

  const names = (ports: { name: string }[]) => new Set(ports.map((p) => p.name));
  const hadInput = names(inputsBefore);
  const hasInput = names(inputsAfter);
  const hadOutput = names(outputsBefore);
  const hasOutput = names(outputsAfter);

  const stale = new Set<string>();
  for (const edge of edges) {
    const from = edge.sourceHandle;
    const to = edge.targetHandle;
    if (edge.source === node.id && from && hadOutput.has(from) && !hasOutput.has(from)) {
      stale.add(edge.id);
    }
    if (edge.target === node.id && to && hadInput.has(to) && !hasInput.has(to)) {
      stale.add(edge.id);
    }
  }
  return stale;
}


/**
 * How a node's ports depend on its params, for `getNodeDefinitions()` (round
 * 6 H1): an agent reads the catalogue, not this file, and has to know that
 * `PythonScript {output_ports: 2}` has `out1` and `out2`. `livePorts.test.ts`
 * checks every entry against the resolvers, so the two cannot drift.
 */
const DYNAMIC_PORTS: Record<string, DynamicPortsInfo> = {
  Split: {
    outputs: { count_param: 'chunks', prefix: 'chunk_', first_index: 0, min: 1, max: SPLIT_MAX_CHUNKS },
  },
  PythonScript: {
    inputs: { count_param: 'input_ports', prefix: 'in', first_index: 1, min: 1,
      max: SCRIPT_MAX_PORTS, types_param: 'input_types' },
    outputs: { count_param: 'output_ports', prefix: 'out', first_index: 1, min: 1,
      max: SCRIPT_MAX_PORTS, types_param: 'output_types' },
  },
  ComposeTransform: {
    inputs: { count_param: 'steps', prefix: 'step_', first_index: 1,
      min: COMPOSE_MIN_STEPS, max: COMPOSE_MAX_STEPS },
  },
  [SWITCH_TYPE]: {
    inputs: { count_param: 'inputs', prefix: 'input_', first_index: 0,
      min: SWITCH_MIN_INPUTS, max: SWITCH_MAX_INPUTS, fixed: [SWITCH_SELECTOR],
      grows_on_connect: true },
  },
};

/** The dynamic-port description of `definition`, or undefined for a node
 *  whose ports are its definition's. */
export function dynamicPortsOf(definition: Pick<NodeDefinition, 'node_name'>): DynamicPortsInfo | undefined {
  const name = definition.node_name;
  const bare = name.slice(name.lastIndexOf(':') + 1);
  return DYNAMIC_PORTS[bare];
}

const withDynamic = new WeakMap<NodeDefinition[], NodeDefinition[]>();

/**
 * The catalogue as plugins get it: each definition whose ports follow its
 * params carries `dynamic_ports`. Every other field is the definition's own.
 * Cached per catalogue array, so repeated calls hand back the same list.
 */
export function definitionsWithDynamicPorts(definitions: NodeDefinition[]): NodeDefinition[] {
  const cached = withDynamic.get(definitions);
  if (cached) return cached;
  const out = definitions.map((d) => {
    const info = dynamicPortsOf(d);
    return info ? { ...d, dynamic_ports: info } : d;
  });
  withDynamic.set(definitions, out);
  return out;
}
