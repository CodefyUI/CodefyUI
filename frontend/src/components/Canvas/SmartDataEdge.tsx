import {
  BaseEdge,
  getBezierPath,
  getSmoothStepPath,
  Position,
  type EdgeProps,
} from '@xyflow/react';
import { useUIStore } from '../../store/uiStore';
import { useEdgeLane } from './EdgeLaneContext';
import {
  detourOffset,
  laneDistance,
  LANE_FORWARD_MIN,
  pickLaneAnchor,
  SOLO_LANE,
  type EdgeLane,
} from '../../utils/edgeLanes';

const HORIZONTAL_SKIP_THRESHOLD = 380;
const VERTICAL_SKIP_THRESHOLD = 150;
const MINOR_TOLERANCE = 80;
const ROW_TRANSITION_DY_THRESHOLD = 200;
const COL_TRANSITION_DX_THRESHOLD = 200;
const ARC_OFFSET_BASE = 100;
const ARC_OFFSET_MAX_EXTRA = 120;
const ARC_OFFSET_SCALE = 0.15;
const PULL_OUT = 50;
/** Floor on the curved arc's pull-out so a high lane slot cannot invert it. */
const ARC_PULL_OUT_MIN = 12;
const MINOR_FLAT_EPSILON = 20;
const SMOOTH_STEP_BORDER_RADIUS = 20;
/** Corner radius for circuit-board style routing (small = crisp PCB look). */
export const CIRCUIT_BORDER_RADIUS = 8;
/** Stroke width bump for the selected edge; the inline per-data-type style
 * overrides the `.selected` CSS rule, so the bump is applied here instead. */
const SELECTED_STROKE_WIDTH = 3;

function isHorizontalPosition(p: Position): boolean {
  return p === Position.Left || p === Position.Right;
}

/** Sign in which the source handle points along its own axis: Right/Bottom = +1. */
function majorDirection(p: Position): number {
  return p === Position.Left || p === Position.Top ? -1 : 1;
}

function computeArcOffset(major: number): number {
  return ARC_OFFSET_BASE + Math.min(Math.abs(major) * ARC_OFFSET_SCALE, ARC_OFFSET_MAX_EXTRA);
}

function computeArcDirection(minor: number): number {
  if (Math.abs(minor) < MINOR_FLAT_EPSILON) return -1;
  return Math.sign(minor);
}

function buildSkipPath(
  sourceX: number,
  sourceY: number,
  targetX: number,
  targetY: number,
  horizontal: boolean,
  arcDir: number,
  arcOffset: number,
  pullOut: number = PULL_OUT,
): string {
  if (horizontal) {
    const c1x = sourceX + pullOut;
    const c1y = sourceY + arcDir * arcOffset;
    const c2x = targetX - pullOut;
    const c2y = targetY + arcDir * arcOffset;
    return `M ${sourceX},${sourceY} C ${c1x},${c1y} ${c2x},${c2y} ${targetX},${targetY}`;
  }
  const c1x = sourceX + arcDir * arcOffset;
  const c1y = sourceY + pullOut;
  const c2x = targetX + arcDir * arcOffset;
  const c2y = targetY - pullOut;
  return `M ${sourceX},${sourceY} C ${c1x},${c1y} ${c2x},${c2y} ${targetX},${targetY}`;
}

interface Point {
  x: number;
  y: number;
}

/**
 * Turn a sequence of axis-aligned waypoints into an SVG path whose corners
 * are rounded with quadratic arcs (same corner treatment as xyflow's
 * smoothstep `getBend`). Consecutive duplicate points are dropped and the
 * bend radius is clamped to half of each adjacent segment so short legs
 * never overshoot. Points MUST be axis-aligned pair-to-pair.
 */
export function buildRoundedOrthogonalPath(points: Point[], radius: number): string {
  const pts: Point[] = [];
  for (const p of points) {
    const prev = pts[pts.length - 1];
    if (!prev || prev.x !== p.x || prev.y !== p.y) pts.push(p);
  }
  if (pts.length === 0) return '';
  let d = `M ${pts[0].x},${pts[0].y}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const a = pts[i - 1];
    const b = pts[i];
    const c = pts[i + 1];
    // Colinear middle point: no corner, just pass through.
    if ((a.x === b.x && b.x === c.x) || (a.y === b.y && b.y === c.y)) {
      d += ` L ${b.x},${b.y}`;
      continue;
    }
    // Segments are axis-aligned, so Manhattan length == segment length.
    const abLen = Math.abs(b.x - a.x) + Math.abs(b.y - a.y);
    const bcLen = Math.abs(c.x - b.x) + Math.abs(c.y - b.y);
    const r = Math.min(radius, abLen / 2, bcLen / 2);
    const inX = Math.sign(b.x - a.x);
    const inY = Math.sign(b.y - a.y);
    const outX = Math.sign(c.x - b.x);
    const outY = Math.sign(c.y - b.y);
    const entryX = b.x - inX * r;
    const entryY = b.y - inY * r;
    const exitX = b.x + outX * r;
    const exitY = b.y + outY * r;
    d += ` L ${entryX},${entryY} Q ${b.x},${b.y} ${exitX},${exitY}`;
  }
  if (pts.length > 1) {
    const last = pts[pts.length - 1];
    d += ` L ${last.x},${last.y}`;
  }
  return d;
}

/**
 * Circuit-mode counterpart of {@link buildSkipPath}: instead of arcing over
 * intermediate nodes with a cubic, route orthogonally through a detour lane
 * offset from the node row (horizontal flows) or column (vertical flows).
 * `arcDir` picks the side exactly like the curved arc does, `laneOffset` (base
 * offset plus the edge's lane slot) puts parallel skips on separate corridors, and
 * `pullOut` moves the climb onto the corridor to its own column - without that
 * second lane two skips leaving one node would still climb along the same line
 * even though their corridors differ.
 */
export function buildCircuitSkipPath(
  sourceX: number,
  sourceY: number,
  targetX: number,
  targetY: number,
  horizontal: boolean,
  arcDir: number,
  laneOffset: number,
  pullOut: number = PULL_OUT,
): string {
  if (horizontal) {
    const laneY = (sourceY + targetY) / 2 + arcDir * laneOffset;
    return buildRoundedOrthogonalPath(
      [
        { x: sourceX, y: sourceY },
        { x: sourceX + pullOut, y: sourceY },
        { x: sourceX + pullOut, y: laneY },
        { x: targetX - pullOut, y: laneY },
        { x: targetX - pullOut, y: targetY },
        { x: targetX, y: targetY },
      ],
      CIRCUIT_BORDER_RADIUS,
    );
  }
  const laneX = (sourceX + targetX) / 2 + arcDir * laneOffset;
  return buildRoundedOrthogonalPath(
    [
      { x: sourceX, y: sourceY },
      { x: sourceX, y: sourceY + pullOut },
      { x: laneX, y: sourceY + pullOut },
      { x: laneX, y: targetY - pullOut },
      { x: targetX, y: targetY - pullOut },
      { x: targetX, y: targetY },
    ],
    CIRCUIT_BORDER_RADIUS,
  );
}

/** Everything the routing decision depends on: geometry, style, and the lane. */
export interface EdgeRoute {
  sourceX: number;
  sourceY: number;
  sourcePosition: Position;
  targetX: number;
  targetY: number;
  targetPosition: Position;
  /** True for the orthogonal circuit-board style, false for the curve style. */
  circuit: boolean;
  lane?: EdgeLane;
}

/** Free coordinate handed to `getSmoothStepPath`; empty means "xyflow's default". */
interface LaneCenters {
  centerX?: number;
  centerY?: number;
}

/**
 * Turn a lane slot into the one coordinate a smoothstep route leaves free.
 *
 * A forward route (`getDirection` in `@xyflow/system` compares the 20px-gapped
 * endpoints) splits perpendicular to the handle axis, so its free coordinate runs
 * *along* that axis: it is the distance at which the trace turns. Those are the
 * routes that superimpose on a fan-out, and the cure is an absolute ladder
 * measured from the shared end. Every sibling shares that end's coordinate and
 * `laneDistance` is strictly increasing in the slot, so their turns are provably
 * distinct and the only stretch they still share is the short stub before the
 * first rung.
 *
 * A backward route splits the other way, so its free coordinate is the cross-axis
 * corridor. The ladder is absolute there too, and for the same reason: two
 * backward edges leaving one handle first drop to their corridors along the *same*
 * gap line, so a merely relative offset would still leave them superimposed for
 * the whole of the shorter drop. Anchoring the corridors near the handle keeps
 * those drops short and provably distinct.
 */
export function resolveLaneCenters(route: EdgeRoute): LaneCenters {
  const lane = route.lane ?? SOLO_LANE;
  const { anchor, slot, count } = pickLaneAnchor(lane);
  if (anchor === 'none') return {};

  const horizontal = isHorizontalPosition(route.sourcePosition);
  const dir = majorDirection(route.sourcePosition);
  const dx = route.targetX - route.sourceX;
  const dy = route.targetY - route.sourceY;
  const major = horizontal ? dx : dy;
  const minor = horizontal ? dy : dx;

  if (major * dir > LANE_FORWARD_MIN) {
    const distance = laneDistance(slot, count);
    const rung =
      anchor === 'source'
        ? (horizontal ? route.sourceX : route.sourceY) + dir * distance
        : (horizontal ? route.targetX : route.targetY) - dir * distance;
    return horizontal ? { centerX: rung } : { centerY: rung };
  }

  const crossDir = Math.sign(minor) || -1;
  const distance = laneDistance(slot, count);
  const corridor =
    anchor === 'source'
      ? (horizontal ? route.sourceY : route.sourceX) + crossDir * distance
      : (horizontal ? route.targetY : route.targetX) - crossDir * distance;
  return horizontal ? { centerY: corridor } : { centerX: corridor };
}

/**
 * The whole routing decision as a pure function, so the no-superposition
 * guarantee can be asserted against real path strings without rendering anything.
 */
export function resolveEdgePath(route: EdgeRoute): string {
  const { sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition, circuit } = route;
  const lane = route.lane ?? SOLO_LANE;

  const horizontal = isHorizontalPosition(sourcePosition);
  const dx = targetX - sourceX;
  const dy = targetY - sourceY;
  const major = horizontal ? dx : dy;
  const minor = horizontal ? dy : dx;

  const isRowTransition = horizontal
    ? Math.abs(dy) > ROW_TRANSITION_DY_THRESHOLD
    : Math.abs(dx) > COL_TRANSITION_DX_THRESHOLD;

  const majorThreshold = horizontal ? HORIZONTAL_SKIP_THRESHOLD : VERTICAL_SKIP_THRESHOLD;
  const isSkip =
    !isRowTransition && Math.abs(major) > majorThreshold && Math.abs(minor) < MINOR_TOLERANCE;

  if (isSkip) {
    const arcDir = computeArcDirection(minor);
    const { slot, count } = pickLaneAnchor(lane);
    // Slot 0 rides the base corridor and every sibling gets one of its own. The
    // old scheme instead shifted *every* skip by a hash of its id into one of four
    // buckets, so a lone skip sat at a corridor chosen at random and any two skips
    // collided outright one time in four. A lone skip now sits closer to its row
    // than it used to, which is the one deliberate visual change here.
    const spread = detourOffset(slot, count);
    const laneOffset = computeArcOffset(major) + spread;
    if (circuit) {
      // Widening the pull-out as well moves each climb onto its own column;
      // without it two skips leaving one node climb along the same line even
      // though their corridors differ.
      return buildCircuitSkipPath(
        sourceX,
        sourceY,
        targetX,
        targetY,
        horizontal,
        arcDir,
        laneOffset,
        PULL_OUT + spread,
      );
    }
    // The curved arc has no columns to separate: sibling arcs leave the shared
    // handle on the same tangent and hug until it turns. Narrowing the pull-out
    // while the corridor widens rotates that tangent instead of merely
    // lengthening it, which is what actually pulls them apart early.
    const arcPullOut = Math.max(PULL_OUT - spread, ARC_PULL_OUT_MIN);
    return buildSkipPath(
      sourceX,
      sourceY,
      targetX,
      targetY,
      horizontal,
      arcDir,
      laneOffset,
      arcPullOut,
    );
  }

  if (isRowTransition || circuit) {
    const [smoothStepPath] = getSmoothStepPath({
      sourceX,
      sourceY,
      sourcePosition,
      targetX,
      targetY,
      targetPosition,
      borderRadius: circuit ? CIRCUIT_BORDER_RADIUS : SMOOTH_STEP_BORDER_RADIUS,
      ...resolveLaneCenters(route),
    });
    return smoothStepPath;
  }

  // Curve style, short hop. Two cubics leaving one handle for different targets
  // already pull apart inside the same stub the orthogonal ladder needs, so a lane
  // would buy nothing here and the familiar bezier is left exactly as it was.
  const [bezier] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });
  return bezier;
}

export function SmartDataEdge(props: EdgeProps) {
  const {
    id,
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    style,
    markerEnd,
    interactionWidth,
    selected,
  } = props;

  const circuit = useUIStore((s) => s.edgeStyle) === 'circuit';
  const lane = useEdgeLane(id);

  const path = resolveEdgePath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    circuit,
    lane,
  });

  // Inline per-data-type styles override App.css's `.selected` width bump,
  // so widen the selected edge here; the stroke color stays untouched.
  const appliedStyle = selected ? { ...style, strokeWidth: SELECTED_STROKE_WIDTH } : style;

  return (
    <BaseEdge
      id={id}
      path={path}
      style={appliedStyle}
      markerEnd={markerEnd}
      interactionWidth={interactionWidth}
    />
  );
}
