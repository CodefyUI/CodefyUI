import type { Edge } from '@xyflow/react';

/** The hidden handle every card carries for a Start node's trigger wire. */
const TRIGGER_HANDLE = '__trigger';

/** The two ends of a wire about to be added or moved. */
export interface WireEnds {
  source: string | null;
  target: string | null;
  sourceHandle?: string | null;
  targetHandle?: string | null;
}

/**
 * The data wires already feeding the input `wire` lands on (#562).
 *
 * A data input takes one source. A new wire onto an input that already has
 * one replaces it, so every gesture and op that adds a wire removes these in
 * the same undo step. `ignoreId` is a wire being moved, which does not
 * occupy the input it is leaving.
 *
 * A trigger wire is control flow: the `__trigger` handle takes a trigger from
 * each Start node that runs the card, so nothing occupies it. A wire whose
 * target names no handle is left alone too, since there is no input to
 * compare.
 */
export function occupantsOf(
  edges: readonly Edge[],
  wire: WireEnds,
  ignoreId: string | null = null,
): Edge[] {
  const handle = wire.targetHandle ?? '';
  if (!wire.target || !handle || handle === TRIGGER_HANDLE) return [];
  if (wire.sourceHandle === 'trigger') return [];
  return edges.filter(
    (e) =>
      e.id !== ignoreId &&
      e.target === wire.target &&
      (e.targetHandle ?? '') === handle &&
      e.type !== 'triggerEdge',
  );
}

/** `edges` without the wires {@link occupantsOf} names. */
export function withoutOccupants(
  edges: readonly Edge[],
  wire: WireEnds,
  ignoreId: string | null = null,
): Edge[] {
  const occupants = new Set(occupantsOf(edges, wire, ignoreId).map((e) => e.id));
  return occupants.size === 0 ? [...edges] : edges.filter((e) => !occupants.has(e.id));
}
