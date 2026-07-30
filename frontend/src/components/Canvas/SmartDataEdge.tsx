import {
  BaseEdge,
  getBezierPath,
  getSmoothStepPath,
  Position,
  type EdgeProps,
} from '@xyflow/react';
import { useUIStore } from '../../store/uiStore';

const HORIZONTAL_SKIP_THRESHOLD = 380;
const VERTICAL_SKIP_THRESHOLD = 150;
const MINOR_TOLERANCE = 80;
const ROW_TRANSITION_DY_THRESHOLD = 200;
const COL_TRANSITION_DX_THRESHOLD = 200;
const ARC_OFFSET_BASE = 100;
const ARC_OFFSET_MAX_EXTRA = 120;
const ARC_OFFSET_SCALE = 0.15;
const PULL_OUT = 50;
const MINOR_FLAT_EPSILON = 20;
const ARC_JITTER_BUCKETS = 4;
const ARC_JITTER_STEP = 28;
const SMOOTH_STEP_BORDER_RADIUS = 20;
/** Corner radius for circuit-board style routing (small = crisp PCB look). */
export const CIRCUIT_BORDER_RADIUS = 8;
/** Lane spacing between parallel circuit skip detours (per-edge-id jitter). */
const CIRCUIT_JITTER_STEP = 24;
/** Stroke width bump for the selected edge; the inline per-data-type style
 * overrides the `.selected` CSS rule, so the bump is applied here instead. */
const SELECTED_STROKE_WIDTH = 3;

function isHorizontalPosition(p: Position): boolean {
  return p === Position.Left || p === Position.Right;
}

function computeArcOffset(major: number): number {
  return ARC_OFFSET_BASE + Math.min(Math.abs(major) * ARC_OFFSET_SCALE, ARC_OFFSET_MAX_EXTRA);
}

function computeArcDirection(minor: number): number {
  if (Math.abs(minor) < MINOR_FLAT_EPSILON) return -1;
  return Math.sign(minor);
}

function hashString(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

function buildSkipPath(
  sourceX: number,
  sourceY: number,
  targetX: number,
  targetY: number,
  horizontal: boolean,
  arcDir: number,
  arcOffset: number,
): string {
  if (horizontal) {
    const c1x = sourceX + PULL_OUT;
    const c1y = sourceY + arcDir * arcOffset;
    const c2x = targetX - PULL_OUT;
    const c2y = targetY + arcDir * arcOffset;
    return `M ${sourceX},${sourceY} C ${c1x},${c1y} ${c2x},${c2y} ${targetX},${targetY}`;
  }
  const c1x = sourceX + arcDir * arcOffset;
  const c1y = sourceY + PULL_OUT;
  const c2x = targetX + arcDir * arcOffset;
  const c2y = targetY - PULL_OUT;
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
 * `arcDir` picks the side exactly like the curved arc does and `laneOffset`
 * (base + per-edge jitter) keeps parallel skip edges in separate lanes.
 */
export function buildCircuitSkipPath(
  sourceX: number,
  sourceY: number,
  targetX: number,
  targetY: number,
  horizontal: boolean,
  arcDir: number,
  laneOffset: number,
): string {
  if (horizontal) {
    const laneY = (sourceY + targetY) / 2 + arcDir * laneOffset;
    return buildRoundedOrthogonalPath(
      [
        { x: sourceX, y: sourceY },
        { x: sourceX + PULL_OUT, y: sourceY },
        { x: sourceX + PULL_OUT, y: laneY },
        { x: targetX - PULL_OUT, y: laneY },
        { x: targetX - PULL_OUT, y: targetY },
        { x: targetX, y: targetY },
      ],
      CIRCUIT_BORDER_RADIUS,
    );
  }
  const laneX = (sourceX + targetX) / 2 + arcDir * laneOffset;
  return buildRoundedOrthogonalPath(
    [
      { x: sourceX, y: sourceY },
      { x: sourceX, y: sourceY + PULL_OUT },
      { x: laneX, y: sourceY + PULL_OUT },
      { x: laneX, y: targetY - PULL_OUT },
      { x: targetX, y: targetY - PULL_OUT },
      { x: targetX, y: targetY },
    ],
    CIRCUIT_BORDER_RADIUS,
  );
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
    !isRowTransition &&
    Math.abs(major) > majorThreshold &&
    Math.abs(minor) < MINOR_TOLERANCE;

  let path: string;
  if (isRowTransition) {
    const [smoothStepPath] = getSmoothStepPath({
      sourceX,
      sourceY,
      sourcePosition,
      targetX,
      targetY,
      targetPosition,
      borderRadius: circuit ? CIRCUIT_BORDER_RADIUS : SMOOTH_STEP_BORDER_RADIUS,
    });
    path = smoothStepPath;
  } else if (isSkip) {
    const arcDir = computeArcDirection(minor);
    if (circuit) {
      const jitter = (hashString(id) % ARC_JITTER_BUCKETS) * CIRCUIT_JITTER_STEP;
      const laneOffset = computeArcOffset(major) + jitter;
      path = buildCircuitSkipPath(sourceX, sourceY, targetX, targetY, horizontal, arcDir, laneOffset);
    } else {
      const jitter = (hashString(id) % ARC_JITTER_BUCKETS) * ARC_JITTER_STEP;
      const arcOffset = computeArcOffset(major) + jitter;
      path = buildSkipPath(sourceX, sourceY, targetX, targetY, horizontal, arcDir, arcOffset);
    }
  } else if (circuit) {
    const [smoothStepPath] = getSmoothStepPath({
      sourceX,
      sourceY,
      sourcePosition,
      targetX,
      targetY,
      targetPosition,
      borderRadius: CIRCUIT_BORDER_RADIUS,
    });
    path = smoothStepPath;
  } else {
    const [bezier] = getBezierPath({
      sourceX,
      sourceY,
      sourcePosition,
      targetX,
      targetY,
      targetPosition,
    });
    path = bezier;
  }

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
