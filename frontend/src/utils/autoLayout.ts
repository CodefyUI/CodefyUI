import dagre from '@dagrejs/dagre';
import type { Node, Edge } from '@xyflow/react';

export type LayoutMode = 'experiments' | 'all' | 'selected';

const NODE_W = 200;
const NODE_H = 80;
const NODESEP = 40;
const RANKSEP = 80;
const LANE_GAP = 60;
const TARGET_ROW_WIDTH = 2400;
const ROW_GAP = 120;
const MIN_NODES_FOR_WRAP = 4;

function getLayoutConfig(nodeCount: number): { nodesep: number; ranksep: number } {
  if (nodeCount > 50) return { nodesep: 28, ranksep: 56 };
  if (nodeCount > 25) return { nodesep: 32, ranksep: 64 };
  return { nodesep: NODESEP, ranksep: RANKSEP };
}

function isEntryPointOrStart(node: Node): boolean {
  return node.type === 'start' || (node.data as any)?.type === 'Start';
}

function findConnectedComponents(targetIds: Set<string>, edges: Edge[]): string[][] {
  // Union-find on targetIds, treating ALL edges (data + trigger) as connecting.
  const parent = new Map<string, string>();
  for (const id of targetIds) parent.set(id, id);
  const find = (x: string): string => {
    let root = x;
    while (parent.get(root) !== root) root = parent.get(root)!;
    let cur = x;
    while (cur !== root) {
      const next = parent.get(cur)!;
      parent.set(cur, root);
      cur = next;
    }
    return root;
  };
  const union = (a: string, b: string) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent.set(ra, rb);
  };
  for (const e of edges) {
    if (targetIds.has(e.source) && targetIds.has(e.target)) {
      union(e.source, e.target);
    }
  }
  const groups = new Map<string, string[]>();
  for (const id of targetIds) {
    const root = find(id);
    if (!groups.has(root)) groups.set(root, []);
    groups.get(root)!.push(id);
  }
  return Array.from(groups.values());
}

function layoutComponentWithDagre(
  componentNodeIds: string[],
  allNodes: Node[],
  allEdges: Edge[],
): Map<string, { x: number; y: number; width: number; height: number }> {
  const g = new dagre.graphlib.Graph();
  const cfg = getLayoutConfig(componentNodeIds.length);
  g.setGraph({ rankdir: 'LR', nodesep: cfg.nodesep, ranksep: cfg.ranksep, ranker: 'network-simplex' });
  g.setDefaultEdgeLabel(() => ({}));

  const idSet = new Set(componentNodeIds);
  for (const id of componentNodeIds) {
    const node = allNodes.find((n) => n.id === id)!;
    const w = node.measured?.width ?? node.width ?? NODE_W;
    const h = node.measured?.height ?? node.height ?? NODE_H;
    g.setNode(id, { width: w, height: h });
  }
  for (const e of allEdges) {
    if (idSet.has(e.source) && idSet.has(e.target)) {
      g.setEdge(e.source, e.target);
    }
  }
  dagre.layout(g);

  const result = new Map<string, { x: number; y: number; width: number; height: number }>();
  for (const id of componentNodeIds) {
    const dn = g.node(id);
    // Dagre returns center coordinates; convert to top-left for React Flow.
    result.set(id, {
      x: dn.x - dn.width / 2,
      y: dn.y - dn.height / 2,
      width: dn.width,
      height: dn.height,
    });
  }
  return result;
}

interface LaidOutComponent {
  ids: string[];
  positions: Map<string, { x: number; y: number; width: number; height: number }>;
  hasEntryPoint: boolean;
  bounds: { minY: number; maxY: number };
}

/**
 * If a component's trunk exceeds TARGET_ROW_WIDTH, split its ranks into multiple
 * rows and shift each row downward. Keeps intra-rank lane layout intact (nodes
 * sharing a rank stay vertically aligned).
 */
function wrapIntoGrid(
  positions: Map<string, { x: number; y: number; width: number; height: number }>,
): Map<string, { x: number; y: number; width: number; height: number }> {
  if (positions.size < MIN_NODES_FOR_WRAP) return positions;

  const values = Array.from(positions.values());
  let minX = Infinity;
  let maxX = -Infinity;
  for (const p of values) {
    if (p.x < minX) minX = p.x;
    if (p.x + p.width > maxX) maxX = p.x + p.width;
  }
  const width = maxX - minX;
  if (width <= TARGET_ROW_WIDTH) return positions;

  const rankUnit = NODE_W + RANKSEP;
  const rankSet = new Set<number>();
  const rankOf = new Map<string, number>();
  for (const [id, p] of positions) {
    const rank = Math.round((p.x - minX) / rankUnit);
    rankOf.set(id, rank);
    rankSet.add(rank);
  }
  // width > TARGET_ROW_WIDTH implies nodes exist, so rankSet.size is always ≥ 1
  /* v8 ignore start */
  const totalRanks = rankSet.size || 1;
  /* v8 ignore stop */
  const maxRanksPerRow = Math.max(1, Math.floor(TARGET_ROW_WIDTH / rankUnit));
  if (totalRanks <= maxRanksPerRow) return positions;

  const rowCount = Math.ceil(totalRanks / maxRanksPerRow);
  const ranksPerRow = Math.ceil(totalRanks / rowCount);
  const rowWidth = ranksPerRow * rankUnit;

  const rowOf = new Map<string, number>();
  for (const [id, rank] of rankOf) {
    const row = Math.min(rowCount - 1, Math.floor(rank / ranksPerRow));
    rowOf.set(id, row);
  }

  const rowYOffset: number[] = new Array(rowCount).fill(0);
  let cumY = 0;
  for (let r = 0; r < rowCount; r++) {
    let rMinY = Infinity;
    let rMaxY = -Infinity;
    for (const [id, p] of positions) {
      if (rowOf.get(id) !== r) continue;
      if (p.y < rMinY) rMinY = p.y;
      if (p.y + p.height > rMaxY) rMaxY = p.y + p.height;
    }
    // Every intermediate grid row gets at least one rank, so a row is never empty
    /* v8 ignore start */
    if (rMinY === Infinity) {
      rMinY = 0;
      rMaxY = 0;
    }
    /* v8 ignore stop */
    rowYOffset[r] = cumY - rMinY;
    cumY += rMaxY - rMinY + ROW_GAP;
  }

  const result = new Map<string, { x: number; y: number; width: number; height: number }>();
  for (const [id, p] of positions) {
    const row = rowOf.get(id)!;
    result.set(id, {
      x: p.x - row * rowWidth,
      y: p.y + rowYOffset[row],
      width: p.width,
      height: p.height,
    });
  }
  return result;
}

function packIntoSwimLanes(
  components: LaidOutComponent[],
): Map<string, { x: number; y: number }> {
  // Sort: entry-pointed first, then drafts; within each group, larger first
  components.sort((a, b) => {
    if (a.hasEntryPoint !== b.hasEntryPoint) return a.hasEntryPoint ? -1 : 1;
    return b.ids.length - a.ids.length;
  });

  const finalPositions = new Map<string, { x: number; y: number }>();
  let currentY = 0;
  for (const comp of components) {
    const yOffset = currentY - comp.bounds.minY;
    let laneMaxY = -Infinity;
    for (const [id, pos] of comp.positions) {
      finalPositions.set(id, { x: pos.x, y: pos.y + yOffset });
      const bottom = pos.y + yOffset + pos.height;
      if (bottom > laneMaxY) laneMaxY = bottom;
    }
    currentY = laneMaxY + LANE_GAP;
  }
  return finalPositions;
}

function isNoteNode(node: Node): boolean {
  return node.type === 'noteNode';
}

function pickTargetIds(
  nodes: Node[],
  edges: Edge[],
  mode: LayoutMode,
  selectedIds?: Set<string>,
): Set<string> {
  // Exclude note nodes from layout computation
  const computationalNodes = nodes.filter((n) => !isNoteNode(n));

  if (mode === 'all') {
    return new Set(computationalNodes.map((n) => n.id));
  }
  if (mode === 'selected') {
    // For selected mode, also exclude notes
    const filtered = new Set(selectedIds ?? []);
    for (const id of filtered) {
      const node = nodes.find((n) => n.id === id);
      if (node && isNoteNode(node)) filtered.delete(id);
    }
    return filtered;
  }
  // mode === 'experiments': only nodes in connected components that contain
  // at least one entry point
  const allComponents = findConnectedComponents(
    new Set(computationalNodes.map((n) => n.id)),
    edges,
  );
  const targets = new Set<string>();
  for (const comp of allComponents) {
    const compNodes = comp.map((id) => nodes.find((n) => n.id === id)!);
    if (compNodes.some(isEntryPointOrStart)) {
      for (const id of comp) targets.add(id);
    }
  }
  return targets;
}

export function autoLayout(
  nodes: Node[],
  edges: Edge[],
  mode: LayoutMode,
  selectedIds?: Set<string>,
): Node[] {
  const targetIds = pickTargetIds(nodes, edges, mode, selectedIds);
  if (targetIds.size === 0) return nodes;

  const componentIds = findConnectedComponents(targetIds, edges);

  // For 'selected' mode, record original centroid
  let originalCentroid: { x: number; y: number } | null = null;
  if (mode === 'selected' && selectedIds) {
    const sel = nodes.filter((n) => selectedIds.has(n.id));
    originalCentroid = {
      x: sel.reduce((s, n) => s + n.position.x + (n.measured?.width ?? n.width ?? NODE_W) / 2, 0) / sel.length,
      y: sel.reduce((s, n) => s + n.position.y + (n.measured?.height ?? n.height ?? NODE_H) / 2, 0) / sel.length,
    };
  }

  // Lay out each component independently
  const laidOut: LaidOutComponent[] = componentIds.map((ids) => {
    const rawPositions = layoutComponentWithDagre(ids, nodes, edges);
    const positions = wrapIntoGrid(rawPositions);
    const ys = Array.from(positions.values()).map((p) => p.y);
    const heights = Array.from(positions.values()).map((p) => p.height);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys.map((y, i) => y + heights[i]));
    const compNodes = ids.map((id) => nodes.find((n) => n.id === id)!);
    return {
      ids,
      positions,
      hasEntryPoint: compNodes.some(isEntryPointOrStart),
      bounds: { minY, maxY },
    };
  });

  let finalPositions = packIntoSwimLanes(laidOut);

  // Selected mode: shift result so the centroid matches the original
  if (mode === 'selected' && originalCentroid) {
    const sel = Array.from(finalPositions.entries());
    const nodeMap = new Map(nodes.map((n) => [n.id, n]));
    const newCentroid = {
      x: sel.reduce((s, [id, p]) => s + p.x + (nodeMap.get(id)?.measured?.width ?? nodeMap.get(id)?.width ?? NODE_W) / 2, 0) / sel.length,
      y: sel.reduce((s, [id, p]) => s + p.y + (nodeMap.get(id)?.measured?.height ?? nodeMap.get(id)?.height ?? NODE_H) / 2, 0) / sel.length,
    };
    const dx = originalCentroid.x - newCentroid.x;
    const dy = originalCentroid.y - newCentroid.y;
    finalPositions = new Map(
      Array.from(finalPositions.entries()).map(([id, p]) => [id, { x: p.x + dx, y: p.y + dy }]),
    );
  }

  // Build result: only target nodes get new positions; others unchanged
  const result = nodes.map((n) => {
    const newPos = finalPositions.get(n.id);
    if (!newPos) return n;
    return { ...n, position: newPos };
  });

  // Reposition bound notes relative to their parent's new position
  return result.map((n) => {
    if (!isNoteNode(n)) return n;
    const data = n.data as any;
    if (!data.boundToNodeId || !data.boundOffset) return n;
    const parent = result.find((p) => p.id === data.boundToNodeId);
    if (!parent) return n;
    return {
      ...n,
      position: {
        x: parent.position.x + data.boundOffset.x,
        y: parent.position.y + data.boundOffset.y,
      },
    };
  });
}

/**
 * Deterministically place UNBOUND note nodes in an offset column. dagre skips
 * notes (they have no edges), so a `layout_missing` load would otherwise leave
 * them stacked at the origin. Bound notes are already repositioned relative to
 * their parent by autoLayout. Ordering is by node id for stability (spec 6.3).
 */
export function stackUnboundNotes(nodes: Node[]): Node[] {
  const NOTE_X = -320;
  const NOTE_GAP = 140;
  const unbound = nodes
    .filter((n) => n.type === 'noteNode' && !(n.data as { boundToNodeId?: string }).boundToNodeId)
    .map((n) => n.id)
    .sort();
  const order = new Map(unbound.map((id, i) => [id, i]));
  return nodes.map((n) => {
    if (n.type !== 'noteNode') return n;
    const idx = order.get(n.id);
    if (idx === undefined) return n; // bound note -> untouched
    return { ...n, position: { x: NOTE_X, y: idx * NOTE_GAP } };
  });
}
