/**
 * Collapse a selection into a subgraph, and expand it back (core#137).
 *
 * Everything here is PURE: it takes canvas nodes/edges plus the definition
 * list and returns the next ones. The store does nothing but push one undo
 * snapshot and commit the result, which is what makes collapse and expand
 * one undo step each -- and what lets the interesting rules (boundary port
 * derivation, convexity, trigger fan-out) be tested without a canvas.
 */
import type { Edge, Node } from '@xyflow/react';

import type {
  NodeData,
  NodeDefinition,
  SubgraphDefinition,
  SubgraphInterface,
  SubgraphPort,
} from '../types';
import { generateId } from './ids';

export const SUBGRAPH_TYPE_PREFIX = 'subgraph:';
export const TRIGGER_TARGET_HANDLE = '__trigger';

/** `subgraph:blk` -> `blk`; anything else -> `null`. */
export function subgraphIdOf(nodeType: unknown): string | null {
  const text = typeof nodeType === 'string' ? nodeType : '';
  return text.startsWith(SUBGRAPH_TYPE_PREFIX)
    ? text.slice(SUBGRAPH_TYPE_PREFIX.length)
    : null;
}

export function isSubgraphInstance(node: Node<NodeData> | undefined): boolean {
  return !!node && subgraphIdOf(node.data?.type) !== null;
}

function isTriggerEdge(edge: Edge): boolean {
  return (
    edge.type === 'triggerEdge' ||
    (edge.data as { type?: string } | undefined)?.type === 'trigger'
  );
}

/**
 * The NodeDefinition an instance node shows on the canvas.
 *
 * Synthesized from the interface, exactly as `addPresetNode` synthesizes one
 * from a preset's exposed ports -- so BaseNode renders an instance with no
 * knowledge that subgraphs exist.
 */
export function instanceDefinition(
  definition: SubgraphDefinition,
): NodeDefinition {
  return {
    node_name: definition.name || definition.id,
    category: 'Subgraph',
    description: definition.description || '',
    inputs: definition.interface.inputs.map((p) => ({
      name: p.port,
      data_type: p.data_type || 'ANY',
      description: `${p.innerNode}.${p.innerPort}`,
      optional: false,
    })),
    outputs: definition.interface.outputs.map((p) => ({
      name: p.port,
      data_type: p.data_type || 'ANY',
      description: `${p.innerNode}.${p.innerPort}`,
      optional: false,
    })),
    params: [],
  };
}

/** A React Flow node for one instance of *definition*. */
export function buildInstanceNode(
  definition: SubgraphDefinition,
  position: { x: number; y: number },
  id: string = generateId(),
): Node<NodeData> {
  return {
    id,
    type: 'subgraphNode',
    position,
    data: {
      label: definition.name || definition.id,
      type: `${SUBGRAPH_TYPE_PREFIX}${definition.id}`,
      params: {},
      definition: instanceDefinition(definition),
      subgraphId: definition.id,
      executionStatus: 'idle',
    },
  };
}

/**
 * Nodes OUTSIDE the selection that sit on a path between two selected nodes.
 *
 * Collapsing around such a node would make the canvas show a loop
 * (`instance -> blocker -> instance`) for a graph that is perfectly acyclic
 * once flattened. Rather than draw a lie, collapse refuses and names the
 * node so the user can add it to the selection. It also keeps the exported
 * script honest: a subgraph's members can then always be scheduled together.
 */
export function findConvexityBlockers(
  selected: Set<string>,
  nodes: Node<NodeData>[],
  edges: Edge[],
): string[] {
  const dataEdges = edges.filter((e) => !isTriggerEdge(e));
  const forward = new Map<string, string[]>();
  const backward = new Map<string, string[]>();
  for (const edge of dataEdges) {
    if (!forward.has(edge.source)) forward.set(edge.source, []);
    forward.get(edge.source)!.push(edge.target);
    if (!backward.has(edge.target)) backward.set(edge.target, []);
    backward.get(edge.target)!.push(edge.source);
  }

  // Reachable FROM the selection, walking outside nodes only.
  const walk = (
    seeds: string[],
    adjacency: Map<string, string[]>,
  ): Set<string> => {
    const seen = new Set<string>();
    const stack = [...seeds];
    while (stack.length) {
      const current = stack.pop()!;
      for (const next of adjacency.get(current) ?? []) {
        if (selected.has(next) || seen.has(next)) continue;
        seen.add(next);
        stack.push(next);
      }
    }
    return seen;
  };

  const downstream = walk([...selected], forward);
  const upstream = walk([...selected], backward);
  const knownIds = new Set(nodes.map((n) => n.id));
  return [...downstream]
    .filter((id) => upstream.has(id) && knownIds.has(id))
    .sort();
}

function uniquePortName(base: string, taken: Set<string>): string {
  const cleaned = base || 'port';
  if (!taken.has(cleaned)) {
    taken.add(cleaned);
    return cleaned;
  }
  let counter = 2;
  while (taken.has(`${cleaned}_${counter}`)) counter += 1;
  const name = `${cleaned}_${counter}`;
  taken.add(name);
  return name;
}

function portTypeOf(
  node: Node<NodeData> | undefined,
  portName: string,
  side: 'inputs' | 'outputs',
): string {
  const ports = node?.data?.definition?.[side] ?? [];
  return ports.find((p) => p.name === portName)?.data_type || 'ANY';
}

export interface CollapseFailure {
  ok: false;
  reason:
    | 'too-few'
    | 'contains-start'
    | 'contains-note'
    | 'not-convex'
    | 'read-only';
  blockers: string[];
}

export interface CollapseSuccess {
  ok: true;
  nodes: Node<NodeData>[];
  edges: Edge[];
  subgraphs: SubgraphDefinition[];
  definition: SubgraphDefinition;
  instanceId: string;
}

export type CollapseResult = CollapseFailure | CollapseSuccess;

/**
 * Replace *selectedIds* with one instance node.
 *
 * The instance lands on the selection's top-left corner and inner positions
 * are stored relative to it, so expanding puts every node back exactly where
 * it was.
 */
export function collapseSelection(
  nodes: Node<NodeData>[],
  edges: Edge[],
  subgraphs: SubgraphDefinition[],
  selectedIds: string[],
  options: { name?: string; id?: string; instanceId?: string } = {},
): CollapseResult {
  const selected = new Set(selectedIds);
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const members = selectedIds
    .map((id) => byId.get(id))
    .filter((n): n is Node<NodeData> => n !== undefined);

  if (members.length < 2) {
    return { ok: false, reason: 'too-few', blockers: [] };
  }
  const notes = members.filter((n) => n.type === 'noteNode');
  if (notes.length) {
    return { ok: false, reason: 'contains-note', blockers: notes.map((n) => n.id) };
  }
  // Start is a GRAPH-level entry marker. Inside a reusable block it would
  // mean "every instance adds its own entry point", which is not a thing v1
  // has an answer for, so it is refused rather than half-supported.
  const starts = members.filter((n) => n.data?.type === 'Start');
  if (starts.length) {
    return {
      ok: false,
      reason: 'contains-start',
      blockers: starts.map((n) => n.id),
    };
  }
  const blockers = findConvexityBlockers(selected, nodes, edges);
  if (blockers.length) {
    return { ok: false, reason: 'not-convex', blockers };
  }

  const definitionId = options.id ?? generateId();
  const instanceId = options.instanceId ?? generateId();

  const minX = Math.min(...members.map((n) => n.position.x));
  const minY = Math.min(...members.map((n) => n.position.y));

  // ── Boundary ports ───────────────────────────────────────────────────
  //
  // Keyed by (innerNode, innerPort): several outside edges feeding one inner
  // port share ONE boundary port, so the fan-in the engine resolves
  // last-edge-wins survives the round trip.
  // Separate pools: an input handle and an output handle live in different
  // namespaces (React Flow keys them by type, and expansion looks each side
  // up in its own map), so a block with one in and one out should read
  // "tensor" on both rather than "tensor" and "tensor_3".
  const takenInputNames = new Set<string>();
  const takenOutputNames = new Set<string>();
  const inputs: SubgraphPort[] = [];
  const outputs: SubgraphPort[] = [];
  const inputByInner = new Map<string, SubgraphPort>();
  const outputByInner = new Map<string, SubgraphPort>();
  const triggerTargets: string[] = [];

  const innerEdges: Edge[] = [];
  const outerEdges: Edge[] = [];
  const triggerSources = new Map<string, Edge>();

  for (const edge of edges) {
    const sourceIn = selected.has(edge.source);
    const targetIn = selected.has(edge.target);
    if (sourceIn && targetIn) {
      innerEdges.push(edge);
      continue;
    }
    if (!sourceIn && !targetIn) {
      outerEdges.push(edge);
      continue;
    }
    if (isTriggerEdge(edge)) {
      if (targetIn) {
        if (!triggerTargets.includes(edge.target)) triggerTargets.push(edge.target);
        if (!triggerSources.has(edge.source)) triggerSources.set(edge.source, edge);
      }
      // A trigger leaving the selection cannot happen: only Start emits one,
      // and a Start inside the selection was refused above.
      continue;
    }
    if (targetIn) {
      const handle = edge.targetHandle ?? '';
      const key = `${edge.target} ${handle}`;
      let port = inputByInner.get(key);
      if (!port) {
        port = {
          port: uniquePortName(handle || 'in', takenInputNames),
          innerNode: edge.target,
          innerPort: handle,
          data_type: portTypeOf(byId.get(edge.target), handle, 'inputs'),
        };
        inputByInner.set(key, port);
        inputs.push(port);
      }
      outerEdges.push({
        ...edge,
        target: instanceId,
        targetHandle: port.port,
      });
    } else {
      const handle = edge.sourceHandle ?? '';
      const key = `${edge.source} ${handle}`;
      let port = outputByInner.get(key);
      if (!port) {
        port = {
          port: uniquePortName(handle || 'out', takenOutputNames),
          innerNode: edge.source,
          innerPort: handle,
          data_type: portTypeOf(byId.get(edge.source), handle, 'outputs'),
        };
        outputByInner.set(key, port);
        outputs.push(port);
      }
      outerEdges.push({
        ...edge,
        source: instanceId,
        sourceHandle: port.port,
      });
    }
  }

  const interfaceSpec: SubgraphInterface = { inputs, outputs, triggerTargets };
  const definition: SubgraphDefinition = {
    id: definitionId,
    name: options.name ?? 'Subgraph',
    description: '',
    nodes: members.map((n) => serializeInnerNode(n, minX, minY)),
    edges: innerEdges.map(serializeInnerEdge),
    interface: interfaceSpec,
  };

  const instance = buildInstanceNode(
    definition,
    { x: Math.round(minX), y: Math.round(minY) },
    instanceId,
  );

  const nextEdges = [...outerEdges];
  for (const [source, edge] of triggerSources) {
    nextEdges.push({
      ...edge,
      id: `${edge.id}-sg`,
      source,
      target: instanceId,
      targetHandle: TRIGGER_TARGET_HANDLE,
    });
  }

  return {
    ok: true,
    nodes: [...nodes.filter((n) => !selected.has(n.id)), instance],
    edges: nextEdges,
    subgraphs: [...subgraphs, definition],
    definition,
    instanceId,
  };
}

function serializeInnerNode(
  node: Node<NodeData>,
  offsetX: number,
  offsetY: number,
): any {
  return {
    id: node.id,
    type: node.data.type,
    position: {
      x: Math.round(node.position.x - offsetX),
      y: Math.round(node.position.y - offsetY),
    },
    data: {
      params: node.data.params ?? {},
      ...(node.data.isPreset
        ? { internalParams: node.data.internalParams ?? {} }
        : {}),
      ...(node.data.bypassed ? { bypassed: true } : {}),
    },
  };
}

function serializeInnerEdge(edge: Edge): any {
  return {
    id: edge.id,
    source: edge.source,
    target: edge.target,
    sourceHandle: edge.sourceHandle ?? '',
    targetHandle: edge.targetHandle ?? '',
    ...(isTriggerEdge(edge) ? { type: 'trigger' } : {}),
  };
}

export interface ExpandResult {
  ok: boolean;
  nodes: Node<NodeData>[];
  edges: Edge[];
  subgraphs: SubgraphDefinition[];
  restoredIds: string[];
}

/**
 * Put an instance's definition back on the canvas, in place.
 *
 * Inner ids are restored verbatim when free (so collapse then expand gives
 * back the graph that was collapsed, ids and all) and prefixed with the
 * instance id when they are not -- which is what happens when the graph
 * still holds another instance of the same definition.
 */
export function expandInstance(
  nodes: Node<NodeData>[],
  edges: Edge[],
  subgraphs: SubgraphDefinition[],
  instanceId: string,
  resolve: (raw: any[]) => Node<NodeData>[],
): ExpandResult {
  const instance = nodes.find((n) => n.id === instanceId);
  const definitionId = subgraphIdOf(instance?.data?.type);
  const definition = subgraphs.find((s) => s.id === definitionId);
  if (!instance || !definition) {
    return { ok: false, nodes, edges, subgraphs, restoredIds: [] };
  }

  const used = new Set(nodes.map((n) => n.id));
  used.delete(instanceId);
  const idMap = new Map<string, string>();
  for (const raw of definition.nodes) {
    let candidate = String(raw.id);
    if (used.has(candidate)) {
      candidate = `${instanceId}-${raw.id}`;
      let counter = 2;
      while (used.has(candidate)) {
        candidate = `${instanceId}-${raw.id}-${counter}`;
        counter += 1;
      }
    }
    used.add(candidate);
    idMap.set(String(raw.id), candidate);
  }

  const restored = resolve(
    definition.nodes.map((raw) => ({
      ...raw,
      id: idMap.get(String(raw.id)),
      position: {
        x: instance.position.x + (raw.position?.x ?? 0),
        y: instance.position.y + (raw.position?.y ?? 0),
      },
    })),
  );

  const innerEdges: Edge[] = definition.edges.map((raw, index) => ({
    id: `${instanceId}-e${index}-${raw.id ?? index}`,
    source: idMap.get(String(raw.source)) ?? String(raw.source),
    target: idMap.get(String(raw.target)) ?? String(raw.target),
    sourceHandle: raw.sourceHandle || undefined,
    targetHandle: raw.targetHandle || undefined,
    ...(raw.type === 'trigger'
      ? { type: 'triggerEdge', data: { type: 'trigger' } }
      : {}),
  }));

  const inputByPort = new Map(
    definition.interface.inputs.map((p) => [p.port, p]),
  );
  const outputByPort = new Map(
    definition.interface.outputs.map((p) => [p.port, p]),
  );

  const nextEdges: Edge[] = [];
  for (const edge of edges) {
    const touchesTarget = edge.target === instanceId;
    const touchesSource = edge.source === instanceId;
    if (!touchesTarget && !touchesSource) {
      nextEdges.push(edge);
      continue;
    }
    if (isTriggerEdge(edge) && touchesTarget) {
      definition.interface.triggerTargets.forEach((innerId, position) => {
        const mapped = idMap.get(String(innerId));
        if (!mapped) return;
        nextEdges.push({
          ...edge,
          id: `${edge.id}-x${position}`,
          target: mapped,
          targetHandle: TRIGGER_TARGET_HANDLE,
        });
      });
      continue;
    }
    let next = { ...edge };
    if (touchesTarget) {
      const port = inputByPort.get(edge.targetHandle ?? '');
      if (!port) continue; // a port the definition no longer exposes
      next = {
        ...next,
        target: idMap.get(port.innerNode) ?? port.innerNode,
        targetHandle: port.innerPort || undefined,
      };
    }
    if (touchesSource) {
      const port = outputByPort.get(edge.sourceHandle ?? '');
      if (!port) continue;
      next = {
        ...next,
        source: idMap.get(port.innerNode) ?? port.innerNode,
        sourceHandle: port.innerPort || undefined,
      };
    }
    nextEdges.push(next);
  }

  const nextNodes = [
    ...nodes.filter((n) => n.id !== instanceId),
    ...restored,
  ];
  // The definition stays as long as ANOTHER instance still uses it; expanding
  // the last one takes it with it, so a graph never carries a block nobody
  // can reach from the canvas.
  const stillUsed = nextNodes.some(
    (n) => subgraphIdOf(n.data?.type) === definition.id,
  );

  return {
    ok: true,
    nodes: nextNodes,
    edges: [...nextEdges, ...innerEdges],
    subgraphs: stillUsed
      ? subgraphs
      : subgraphs.filter((s) => s.id !== definition.id),
    restoredIds: [...idMap.values()],
  };
}

/**
 * Refresh every instance node of *definition* from the definition.
 *
 * Called after a sub-canvas edit: the instances' rendered ports come from
 * the interface, so this is what makes an edit to one definition show up on
 * every instance of it.
 */
export function refreshInstances(
  nodes: Node<NodeData>[],
  definition: SubgraphDefinition,
): Node<NodeData>[] {
  const rendered = instanceDefinition(definition);
  let changed = false;
  const next = nodes.map((node) => {
    if (subgraphIdOf(node.data?.type) !== definition.id) return node;
    changed = true;
    return {
      ...node,
      data: {
        ...node.data,
        label: definition.name || definition.id,
        definition: rendered,
      },
    };
  });
  return changed ? next : nodes;
}

/**
 * Edges whose endpoint on an instance no longer exists in the interface.
 *
 * A sub-canvas edit can delete the node a boundary port stood for. Dropping
 * those edges keeps the canvas consistent with what expansion would accept
 * (which refuses an undeclared port outright).
 */
export function pruneStaleBoundaryEdges(
  nodes: Node<NodeData>[],
  edges: Edge[],
  subgraphs: SubgraphDefinition[],
): Edge[] {
  const byDefinition = new Map(subgraphs.map((s) => [s.id, s]));
  const instances = new Map<string, SubgraphDefinition>();
  for (const node of nodes) {
    const sid = subgraphIdOf(node.data?.type);
    const definition = sid ? byDefinition.get(sid) : undefined;
    if (definition) instances.set(node.id, definition);
  }
  if (!instances.size) return edges;
  return edges.filter((edge) => {
    if (isTriggerEdge(edge)) return true;
    const asTarget = instances.get(edge.target);
    if (
      asTarget &&
      !asTarget.interface.inputs.some((p) => p.port === (edge.targetHandle ?? ''))
    ) {
      return false;
    }
    const asSource = instances.get(edge.source);
    if (
      asSource &&
      !asSource.interface.outputs.some(
        (p) => p.port === (edge.sourceHandle ?? ''),
      )
    ) {
      return false;
    }
    return true;
  });
}

/**
 * Rebuild a definition's `nodes`/`edges` from a sub-canvas the user edited.
 *
 * The interface is carried across by inner node id, and any port whose inner
 * node was deleted is dropped -- the block's boundary can only name nodes it
 * still contains.
 */
export function definitionFromCanvas(
  definition: SubgraphDefinition,
  nodes: Node<NodeData>[],
  edges: Edge[],
): SubgraphDefinition {
  const real = nodes.filter((n) => n.type !== 'noteNode');
  const alive = new Set(real.map((n) => n.id));
  const minX = real.length ? Math.min(...real.map((n) => n.position.x)) : 0;
  const minY = real.length ? Math.min(...real.map((n) => n.position.y)) : 0;
  const keepPort = (p: SubgraphPort) => alive.has(p.innerNode);
  return {
    ...definition,
    nodes: real.map((n) => serializeInnerNode(n, minX, minY)),
    edges: edges
      .filter((e) => alive.has(e.source) && alive.has(e.target))
      .map(serializeInnerEdge),
    interface: {
      inputs: definition.interface.inputs.filter(keepPort),
      outputs: definition.interface.outputs.filter(keepPort),
      triggerTargets: definition.interface.triggerTargets.filter((id) =>
        alive.has(id),
      ),
    },
  };
}

