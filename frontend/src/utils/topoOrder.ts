import type { Edge } from '@xyflow/react';

/**
 * Trigger edges are execution markers, not data dependencies — the backend's
 * `topological_sort` skips them and so must we, or a Start node would appear
 * to gate everything it points at.
 *
 * Two shapes carry the marker in saved / live graphs: the xyflow edge `type`
 * (`'triggerEdge'`) and the payload (`data.type === 'trigger'`). Both are
 * checked everywhere else in the codebase (InspectorPanel, BaseNode), so both
 * are checked here.
 */
function isTriggerEdge(e: Pick<Edge, 'type' | 'data'>): boolean {
  return (
    e.type === 'triggerEdge' ||
    (e.data as { type?: string } | undefined)?.type === 'trigger'
  );
}

/**
 * Order node ids the way the engine would execute them (Kahn's algorithm),
 * mirroring `backend/app/core/graph_engine.py::topological_sort`.
 *
 * Two deliberate differences, because this drives *navigation* rather than
 * execution and must never leave a node unreachable:
 *
 * - Edges with an endpoint outside `nodes` are ignored. The backend would let
 *   such an edge pin its target's in-degree above zero forever and then raise;
 *   here that would silently drop a perfectly visitable node from the
 *   prev/next walk.
 * - A cycle does not throw. Whatever Kahn's could not drain is appended in
 *   the caller's node order, so every node is still visited exactly once.
 *
 * Ties are broken by the caller's node order (roots are seeded in that order
 * and adjacency lists are built in edge order), which keeps the walk stable
 * across renders and matches what the engine does with an equally ambiguous
 * graph.
 */
export function topologicalOrder(
  nodes: readonly { id: string }[],
  edges: readonly Pick<Edge, 'source' | 'target' | 'type' | 'data'>[],
): string[] {
  const inDegree = new Map<string, number>();
  for (const n of nodes) inDegree.set(n.id, 0);

  const adjacency = new Map<string, string[]>();
  for (const e of edges) {
    if (isTriggerEdge(e)) continue;
    if (!inDegree.has(e.source) || !inDegree.has(e.target)) continue;
    const list = adjacency.get(e.source);
    if (list) list.push(e.target);
    else adjacency.set(e.source, [e.target]);
    // Non-null: the `has(e.target)` guard above is what makes this safe.
    inDegree.set(e.target, inDegree.get(e.target)! + 1);
  }

  const queue: string[] = nodes.filter((n) => inDegree.get(n.id) === 0).map((n) => n.id);
  const order: string[] = [];
  const emitted = new Set<string>();

  // Index-based cursor rather than shift(): the queue is append-only here, so
  // walking it costs O(n) instead of O(n²) re-indexing on large graphs.
  for (let cursor = 0; cursor < queue.length; cursor++) {
    const id = queue[cursor];
    order.push(id);
    emitted.add(id);
    for (const next of adjacency.get(id) ?? []) {
      // Non-null: only edges whose endpoints are both known nodes made it
      // into `adjacency`, so every neighbour has an in-degree entry.
      const remaining = inDegree.get(next)! - 1;
      inDegree.set(next, remaining);
      if (remaining === 0) queue.push(next);
    }
  }

  if (order.length !== nodes.length) {
    for (const n of nodes) {
      if (!emitted.has(n.id)) order.push(n.id);
    }
  }

  return order;
}
