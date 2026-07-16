import dagre from '@dagrejs/dagre';
import type { Node, Edge } from '@xyflow/react';

export type LayoutMode = 'experiments' | 'all' | 'selected';

const NODE_W = 200;
const NODE_H = 80;
const NODESEP = 40;
const RANKSEP = 80;
const LANE_GAP = 60;
const VALLEY_MIN_SPAN = 3;
const VALLEY_GAP = 60;

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

export type ValleyEdge = {
  source: string;
  target: string;
  type?: string;
  data?: unknown;
  sourceHandle?: string | null;
};

export function isTriggerEdge(e: ValleyEdge): boolean {
  return (
    e.type === 'triggerEdge' ||
    (e.data as { type?: string } | undefined)?.type === 'trigger' ||
    e.sourceHandle === 'trigger'
  );
}

type ValleyPos = { x: number; y: number; width: number; height: number };

/**
 * Skip-aware "valley" pass (replaces the old width wrapping). Ranks spanned by
 * a skip connection are shifted along the cross axis, so the trunk dips under
 * its skips: a UNet reads as a U, residual blocks as small dips, and graphs
 * without skips are returned untouched. A skip is a non-trigger edge whose
 * endpoints both lie on the spine (longest forward path) and whose rank span
 * is at least `minSpan`. Cycle-safe: only forward edges (rank increasing) are
 * walked, so back edges are simply ignored.
 */
export function applyValleyPass(
  positions: Map<string, ValleyPos>,
  edges: ValleyEdge[],
  opts: {
    axis: 'y' | 'x';
    skipPredicate?: (e: ValleyEdge) => boolean;
    minSpan?: number;
    gap?: number;
  },
): Map<string, ValleyPos> {
  const { axis, skipPredicate = () => true, minSpan = VALLEY_MIN_SPAN, gap = VALLEY_GAP } = opts;
  if (positions.size === 0) return positions;

  // 1. Geometric ranks: cluster distinct centers along the flow axis. Dagre
  //    gives every node of a rank the same center coordinate, so this recovers
  //    true ranks regardless of node sizes or spacing tier.
  const centerOf = (p: ValleyPos) => (axis === 'y' ? p.x + p.width / 2 : p.y + p.height / 2);
  const centers = Array.from(positions.values(), centerOf).sort((a, b) => a - b);
  const rankCenters: number[] = [];
  for (const c of centers) {
    if (rankCenters.length === 0 || c - rankCenters[rankCenters.length - 1] > 1) {
      rankCenters.push(c);
    }
  }
  const rankOf = new Map<string, number>();
  for (const [id, p] of positions) {
    const c = centerOf(p);
    let lo = 0;
    let hi = rankCenters.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (rankCenters[mid] < c - 1) lo = mid + 1;
      else hi = mid;
    }
    rankOf.set(id, lo);
  }
  const maxRank = rankCenters.length - 1;
  if (maxRank < minSpan) return positions;

  // 2. Spine: longest path over FORWARD edges (rank strictly increasing),
  //    deterministic tie-break by id.
  const ids = Array.from(positions.keys()).sort();
  const byRank = ids
    .slice()
    .sort((a, b) => rankOf.get(a)! - rankOf.get(b)! || a.localeCompare(b));
  const forwardIn = new Map<string, string[]>(ids.map((id) => [id, []]));
  for (const e of edges) {
    if (!positions.has(e.source) || !positions.has(e.target)) continue;
    if (rankOf.get(e.target)! > rankOf.get(e.source)!) {
      forwardIn.get(e.target)!.push(e.source);
    }
  }
  const pathLen = new Map<string, number>();
  const pred = new Map<string, string | null>();
  for (const id of byRank) {
    let best = 0;
    let bestPred: string | null = null;
    for (const p of forwardIn.get(id)!.sort()) {
      const cand = pathLen.get(p)! + 1;
      if (cand > best) {
        best = cand;
        bestPred = p;
      }
    }
    pathLen.set(id, best);
    pred.set(id, bestPred);
  }
  let tail = ids[0];
  for (const id of ids) {
    if (
      pathLen.get(id)! > pathLen.get(tail)! ||
      (pathLen.get(id) === pathLen.get(tail) && id < tail)
    ) {
      tail = id;
    }
  }
  const spine = new Set<string>();
  for (let cur: string | null = tail; cur !== null; cur = pred.get(cur) ?? null) {
    spine.add(cur);
  }

  // 3. Skips: eligible edges with both endpoints on the spine spanning >= minSpan.
  const coverage = new Array(maxRank + 1).fill(0);
  let hasSkip = false;
  for (const e of edges) {
    if (!skipPredicate(e)) continue;
    if (!spine.has(e.source) || !spine.has(e.target)) continue;
    // Off-map endpoints are filtered by the spine check (spine ⊆ positions keys)
    const rs = rankOf.get(e.source)!;
    const rt = rankOf.get(e.target)!;
    if (rt - rs < minSpan) continue;
    hasSkip = true;
    for (let r = rs + 1; r < rt; r++) coverage[r]++;
  }
  if (!hasSkip) return positions;

  // 4. Shift whole ranks along the cross axis by coverage * step.
  let maxCross = 0;
  for (const p of positions.values()) {
    maxCross = Math.max(maxCross, axis === 'y' ? p.height : p.width);
  }
  const step = maxCross + gap;
  const result = new Map<string, ValleyPos>();
  for (const [id, p] of positions) {
    const shift = coverage[rankOf.get(id)!] * step;
    result.set(id, axis === 'y' ? { ...p, y: p.y + shift } : { ...p, x: p.x + shift });
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
    const positions = applyValleyPass(rawPositions, edges, {
      axis: 'y',
      skipPredicate: (e) => !isTriggerEdge(e),
    });
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
