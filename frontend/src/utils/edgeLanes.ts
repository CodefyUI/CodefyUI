/**
 * Deterministic, sibling-aware lane assignment for canvas edges.
 *
 * Why this exists
 * ---------------
 * Two connection lines must never render as one line. The previous scheme hashed
 * the edge id into four buckets and applied the result to detour paths only, so
 * two edges that shared a route collided one time in four, and the ordinary
 * (non-detour) paths were never offset at all. A hash can never promise
 * separation: it cannot see the other edges on the canvas.
 *
 * This module looks at the whole edge list and hands every edge a *slot* that is
 * unique among the edges it can actually collide with. `SmartDataEdge` turns that
 * slot into a routing coordinate, so two edges in the same group are guaranteed to
 * bend at different places. Crossings are untouched by design - the owner asked
 * for no superposition, not for fewer crossings.
 *
 * Grouping
 * --------
 * - out group: every edge leaving the same source NODE, not merely the same
 *   handle. Every output handle sits on the same vertical edge of the card, so two
 *   edges leaving *different* handles still bend at the same x unless they are
 *   told apart here.
 * - in group: every edge arriving at the same target NODE, for the mirror reason.
 *
 * Ordering inside a group is by (handle id, edge id). Both are stable across
 * reloads and independent of the order the edges happen to sit in the array, so
 * the same graph always yields the same lanes - no randomness and no dependence on
 * insertion order.
 *
 * Cost is O(E log E) over the edge list and depends on topology only, never on
 * node positions, so dragging a node never recomputes it and the lane map keeps a
 * stable identity across drags, selection changes and style changes.
 *
 * What this guarantees
 * --------------------
 * Two edges that share a handle coincide ONLY inside the port stub of that
 * handle - never anywhere else, and never for longer than
 * `laneDistance(min slot, count)`. That stub is not a tolerance chosen for
 * convenience, it is a floor: orthogonal routes leaving a common point in a common
 * direction each occupy some prefix [0, L] of the departure line, so two of them
 * always share min(L_i, L_j), and the L values must differ by at least a lane
 * pitch or the turns land on top of each other instead. Sorted, L_(k) >=
 * LANE_BASE + k * step, so the worst pair in a fan of n shares exactly
 * LANE_BASE + (n - 2) * step and no routing can do better. That is 12px for a
 * pair, 20px for three, 28px for four - measured on the real ResNet-18 example in
 * `resnetOverlap.test.ts`, where those same pairs used to share 30, 44 and 62px.
 *
 * What it does not
 * ----------------
 * 1. Edges between two pairs of nodes with no handle in common, whose routes
 *    happen to land on the same line. The concrete mechanism is worth naming
 *    because it is systematic rather than freak: slots are numbered per source
 *    node and every group starts from the same base, so two nodes sitting in the
 *    SAME COLUMN hand their k-th edge the same turn distance - and the same turn
 *    distance from the same x is the same line. Three pairs of the ResNet-18
 *    example are exactly this, the worst sharing 372px, and they are named in
 *    `resnetOverlap.test.ts` so a new one fails the suite. Separating them needs a
 *    router that reads where the nodes actually are and re-runs whenever one
 *    moves; the common instance - a wire skipping over an intervening node in the
 *    same row - is already routed around by the detour in `SmartDataEdge`.
 * 2. Two edges whose *whole route* is one axis line: both endpoints identical, or
 *    both targets at exactly the source handle's height. No choice of turning
 *    distance separates those - only lifting one off the row would, and that would
 *    put a gratuitous hop on every aligned wire whose sibling merely happens to be
 *    nearby. It needs the two target cards to touch or overlap: give the far
 *    target any clearance and it crosses the skip threshold and detours.
 *
 * Both remain possible on a real graph, and both are visible rather than hidden:
 * `SmartDataEdge.overlap.test.tsx` asserts that each still happens, and
 * `resnetOverlap.test.ts` names every instance on the flagship example. A reader
 * cannot mistake the guarantee for a universal one.
 *
 * Every kind of line on the canvas is covered, trigger edges included: they are
 * ordinary members of the edge array, their diamond is a Right handle and
 * `__trigger` is a Left one, so `TriggerEdge` shares `resolveEdgePath` and this
 * map and keeps only its own stroke.
 */

/** The only edge fields lane assignment needs. */
export interface LaneEdgeInput {
  id: string;
  source: string;
  target: string;
  sourceHandle?: string | null;
  targetHandle?: string | null;
}

/** Slot of an edge inside its two conflict groups, plus each group's size. */
export interface EdgeLane {
  /** 0-based slot among all edges leaving the same source node. */
  outSlot: number;
  /** Number of edges leaving that source node (1 when this edge is alone). */
  outCount: number;
  /** 0-based slot among all edges arriving at the same target node. */
  inSlot: number;
  /** Number of edges arriving at that target node (1 when this edge is alone). */
  inCount: number;
}

/** Lane of an edge with no siblings: every offset resolves to "route as before". */
export const SOLO_LANE: EdgeLane = Object.freeze({
  outSlot: 0,
  outCount: 1,
  inSlot: 0,
  inCount: 1,
});

export type EdgeLaneMap = ReadonlyMap<string, EdgeLane>;

/** Shared empty map so consumers without a provider allocate nothing. */
export const EMPTY_LANE_MAP: EdgeLaneMap = new Map<string, EdgeLane>();

/**
 * Distance from the anchor handle to the *first* lane.
 *
 * This is the port stub - the one stretch two wires leaving a shared handle are
 * allowed to have in common - so it is deliberately as short as it can be while
 * still clearing the handle graphic (about 12px across) and leaving room for a
 * readable corner. `SmartDataEdge` builds laned routes itself rather than through
 * `getSmoothStepPath` precisely so this is not forced up to 28 by xyflow's fixed
 * 20px handle gap.
 */
export const LANE_BASE = 12;
/**
 * Spacing between adjacent lanes for small groups.
 *
 * Stroke width is 2 *flow* units, so it scales with zoom exactly as this does:
 * the gap-to-stroke ratio is zoom-invariant, and 8 units is four stroke widths -
 * unambiguously two traces at any zoom the canvas is usable at.
 */
export const LANE_STEP_MAX = 8;
/** Floor on lane spacing: three stroke widths, still clearly two traces. */
export const LANE_STEP_MIN = 6;
/** Ladder width budget; keeps a 20-edge bus from spreading across the canvas. */
export const LANE_SPREAD_MAX = 120;

/** Extra offset between adjacent detour corridors for skip routes. */
export const DETOUR_STEP_MAX = 26;
export const DETOUR_STEP_MIN = 10;
export const DETOUR_SPREAD_MAX = 160;

/**
 * A forward edge (target far enough along the handle axis) is routed by xyflow
 * through a perpendicular split, so its free coordinate is the cross axis. Below
 * this separation xyflow flips to the other split. Mirrors `getDirection` in
 * `@xyflow/system`, which compares the 20px-gapped endpoints.
 */
export const LANE_FORWARD_MIN = 40;

function clampStep(count: number, spread: number, min: number, max: number): number {
  if (count <= 2) return max;
  return Math.max(min, Math.min(max, spread / (count - 1)));
}

/**
 * Spacing between adjacent lanes of a group of `count` edges. Depends on `count`
 * alone, which every member of a group shares - that is what makes
 * {@link laneDistance} strictly increasing in `slot` for all members, and hence
 * what makes their bends provably distinct.
 */
export function laneStep(count: number): number {
  return clampStep(count, LANE_SPREAD_MAX, LANE_STEP_MIN, LANE_STEP_MAX);
}

/** Spacing between adjacent skip-detour corridors. */
export function detourStep(count: number): number {
  return clampStep(count, DETOUR_SPREAD_MAX, DETOUR_STEP_MIN, DETOUR_STEP_MAX);
}

/**
 * Distance from the anchor handle to this edge's lane. Strictly increasing in
 * `slot`, so two edges of the same group never share a lane.
 */
export function laneDistance(slot: number, count: number): number {
  return LANE_BASE + slot * laneStep(count);
}

/** Extra corridor offset for a skip detour. Zero for slot 0, so lone skips are unchanged. */
export function detourOffset(slot: number, count: number): number {
  return slot * detourStep(count);
}

/** Which end of the edge the lane offset is measured from. */
export type LaneAnchor = 'none' | 'source' | 'target';

export interface LaneChoice {
  anchor: LaneAnchor;
  slot: number;
  count: number;
}

/**
 * Pick the end to anchor this edge's lane to.
 *
 * The source side wins when it has siblings, because a fan-out shares a *point*:
 * the wires can only be told apart by bending at different distances from it, and
 * bending late would leave them superimposed for most of their length. When the
 * source is alone the target side gets the free coordinate instead, which
 * separates a fan-in. When neither end has siblings the edge routes exactly as it
 * did before this module existed.
 */
export function pickLaneAnchor(lane: EdgeLane): LaneChoice {
  if (lane.outCount > 1) return { anchor: 'source', slot: lane.outSlot, count: lane.outCount };
  if (lane.inCount > 1) return { anchor: 'target', slot: lane.inSlot, count: lane.inCount };
  return { anchor: 'none', slot: 0, count: 1 };
}

function handleKey(handle: string | null | undefined): string {
  return handle ?? '';
}

/** Total order used inside a group: stable across reloads, blind to array order. */
function bySortKey(a: { key: string }, b: { key: string }): number {
  return a.key < b.key ? -1 : a.key > b.key ? 1 : 0;
}

interface Member {
  id: string;
  key: string;
}

function assign(
  groups: Map<string, Member[]>,
  write: (id: string, slot: number, count: number) => void,
): void {
  for (const members of groups.values()) {
    if (members.length > 1) members.sort(bySortKey);
    for (let i = 0; i < members.length; i++) write(members[i].id, i, members.length);
  }
}

/**
 * Assign every edge a lane in its source-node group and its target-node group.
 *
 * Pure and deterministic: the result depends only on edge ids, endpoints and
 * handle ids, never on array order, node positions or wall-clock state.
 */
export function computeEdgeLanes(edges: readonly LaneEdgeInput[]): EdgeLaneMap {
  const lanes = new Map<string, EdgeLane>();
  if (edges.length === 0) return lanes;

  const outGroups = new Map<string, Member[]>();
  const inGroups = new Map<string, Member[]>();

  for (const edge of edges) {
    // A duplicated id would make one of the two entries unreachable; keep the
    // first so the map stays a function of the edge list.
    if (lanes.has(edge.id)) continue;
    lanes.set(edge.id, { outSlot: 0, outCount: 1, inSlot: 0, inCount: 1 });

    const outMembers = outGroups.get(edge.source);
    const outMember = { id: edge.id, key: `${handleKey(edge.sourceHandle)}\u0000${edge.id}` };
    if (outMembers) outMembers.push(outMember);
    else outGroups.set(edge.source, [outMember]);

    const inMembers = inGroups.get(edge.target);
    const inMember = { id: edge.id, key: `${handleKey(edge.targetHandle)}\u0000${edge.id}` };
    if (inMembers) inMembers.push(inMember);
    else inGroups.set(edge.target, [inMember]);
  }

  assign(outGroups, (id, slot, count) => {
    const lane = lanes.get(id)!;
    lane.outSlot = slot;
    lane.outCount = count;
  });
  assign(inGroups, (id, slot, count) => {
    const lane = lanes.get(id)!;
    lane.inSlot = slot;
    lane.inCount = count;
  });

  return lanes;
}

/**
 * Cheap key covering everything {@link computeEdgeLanes} reads.
 *
 * Memoising on this instead of the edges array keeps the lane map identity stable
 * when an edge is merely selected, recoloured or re-created by `applyEdgeChanges`,
 * so those very common updates re-render no edges at all.
 */
export function edgeLaneSignature(edges: readonly LaneEdgeInput[]): string {
  // Field and record separators are control characters, so they cannot occur in a
  // uuid or a port name: distinct edge lists cannot collide on one signature.
  let sig = '';
  for (const e of edges) {
    sig += `${e.id}\u0001${e.source}\u0001${handleKey(e.sourceHandle)}\u0001${e.target}\u0001${handleKey(e.targetHandle)}\u0002`;
  }
  return sig;
}
