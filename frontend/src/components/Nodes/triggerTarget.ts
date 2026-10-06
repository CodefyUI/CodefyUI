import { useStore, type Edge } from '@xyflow/react';

/** The nodes a trigger wire goes into, built once for each edges array. */
const targetsByEdges = new WeakMap<readonly Edge[], ReadonlySet<string>>();

function triggerTargets(edges: readonly Edge[]): ReadonlySet<string> {
  let targets = targetsByEdges.get(edges);
  if (!targets) {
    const found = new Set<string>();
    for (const e of edges) {
      if ((e.data as { type?: string } | undefined)?.type === 'trigger') found.add(e.target);
    }
    targets = found;
    targetsByEdges.set(edges, targets);
  }
  return targets;
}

/**
 * Whether a Start node's trigger wire goes into node `id`, for the entry-point
 * marker on its card.
 *
 * Subscribed to React Flow's edges, so the marker follows a trigger wire that
 * is moved or deleted. Read once at render with `getEdges()`, it stayed on the
 * old card and never reached the new one until something else re-rendered
 * them (#619). Every card's selector runs on every React Flow store update (a
 * pan or drag frame), so the answer is a lookup in a set built once per edges
 * array, and a boolean: a card re-renders only when its own answer changes.
 */
export function useIsTriggerTarget(id: string): boolean {
  return useStore((s) => triggerTargets(s.edges).has(id));
}
