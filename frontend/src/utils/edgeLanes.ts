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
 * Two edges that share an endpoint node get different routing coordinates, so the
 * only stretch they can still have in common is the stub between the handle and
 * the first of them to turn. That stub is unavoidable: orthogonal traces leaving
 * one point in one direction must run together until one of them turns, and
 * turning at the same distance would put them on the same corridor instead. The
 * stub is bounded by `laneDistance(count - 2, count)` and measured in
 * `SmartDataEdge.overlap.test.tsx` - 44px for a three-way fan-out that used to
 * share 224px, 170px for a twenty-wire bus that used to share 986px.
 *
 * What it does not
 * ----------------
 * 1. Edges between two entirely unrelated pairs of nodes. Nothing here can see
 *    that their routes happen to land on the same line, because that depends on
 *    where the nodes sit and this pass deliberately never looks. Catching it needs
 *    a global orthogonal router that re-runs whenever a node moves; the common
 *    instance - a wire that skips over an intervening node in the same row - is
 *    already routed around by the detour in `SmartDataEdge`.
 * 2. Two edges whose *whole route* is one axis line: both endpoints identical, or
 *    both targets at exactly the source handle's height. No choice of turning
 *    distance separates those - only lifting one off the row would, and that would
 *    put a gratuitous hop on every aligned wire whose sibling merely happens to be
 *    nearby. It needs the two target cards to touch or overlap: give the far
 *    target any clearance and it crosses the skip threshold and detours.
 * 3. Trigger edges (`TriggerEdge`), which are drawn as plain cubics in their own
 *    colour and dash. Two cubics to distinct targets share only their endpoint.
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
 * Distance from the anchor handle to the *first* lane. Must exceed xyflow's
 * smoothstep gap (20px) so a laned bend never lands inside the handle stub.
 */
export const LANE_BASE = 28;
/** Spacing between adjacent lanes for small groups. */
export const LANE_STEP_MAX = 18;
/** Floor on lane spacing: below this two traces stop reading as two at low zoom. */
export const LANE_STEP_MIN = 8;
/** Ladder width budget; keeps a 20-edge bus from spreading across the canvas. */
export const LANE_SPREAD_MAX = 140;

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
