/**
 * Measure how much of one SVG path is drawn on top of another.
 *
 * This exists so the "two connection lines must never render as one" rule can be
 * asserted on the *output* - real path strings - instead of on the routing code
 * that produced them. It knows nothing about lanes, slots or edge styles, so a
 * test written against it keeps its meaning if the routing is ever rewritten.
 *
 * The measure is the longest contiguous stretch of path A that stays within
 * `tolerance` of path B, in flow-coordinate pixels. Two traces read as one wire
 * exactly when that number is large. A crossing scores near zero (the paths are
 * only close where they touch), which is what the requirement asks for: crossings
 * are allowed, superposition is not.
 */

export interface Point {
  x: number;
  y: number;
}

const COMMAND_RE = /([MLQCZmlqcz])([^MLQCZmlqcz]*)/g;
const NUMBER_RE = /-?\d*\.?\d+(?:e[-+]?\d+)?/gi;

/** Points consumed by each absolute path command this parser supports. */
const ARITY: Record<string, number> = { M: 2, L: 2, Q: 4, C: 6, Z: 0 };

function quadraticAt(p0: Point, c: Point, p1: Point, t: number): Point {
  const u = 1 - t;
  return {
    x: u * u * p0.x + 2 * u * t * c.x + t * t * p1.x,
    y: u * u * p0.y + 2 * u * t * c.y + t * t * p1.y,
  };
}

function cubicAt(p0: Point, c1: Point, c2: Point, p1: Point, t: number): Point {
  const u = 1 - t;
  return {
    x: u * u * u * p0.x + 3 * u * u * t * c1.x + 3 * u * t * t * c2.x + t * t * t * p1.x,
    y: u * u * u * p0.y + 3 * u * u * t * c1.y + 3 * u * t * t * c2.y + t * t * t * p1.y,
  };
}

/**
 * Flatten a path into a polyline. Curves are sampled, so the result approximates
 * bezier and quadratic-cornered routes closely enough to compare them by distance.
 */
export function flattenPath(d: string, curveSamples = 32): Point[] {
  const points: Point[] = [];
  let cursor: Point = { x: 0, y: 0 };
  let start: Point = { x: 0, y: 0 };

  COMMAND_RE.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = COMMAND_RE.exec(d)) !== null) {
    const command = match[1].toUpperCase();
    if (match[1] !== command) throw new Error(`relative path command not supported: ${match[1]}`);
    const arity = ARITY[command];
    if (arity === undefined) throw new Error(`unsupported path command: ${command}`);
    const nums = (match[2].match(NUMBER_RE) ?? []).map(Number);
    if (arity === 0) {
      if (points.length) points.push({ ...start });
      cursor = { ...start };
      continue;
    }
    if (nums.length === 0 || nums.length % arity !== 0) {
      throw new Error(`command ${command} got ${nums.length} numbers, expected a multiple of ${arity}`);
    }
    // A command letter may be followed by several coordinate sets.
    for (let base = 0; base < nums.length; base += arity) {
      if (command === 'M') {
        cursor = { x: nums[base], y: nums[base + 1] };
        start = { ...cursor };
        points.push({ ...cursor });
      } else if (command === 'L') {
        cursor = { x: nums[base], y: nums[base + 1] };
        points.push({ ...cursor });
      } else if (command === 'Q') {
        const c = { x: nums[base], y: nums[base + 1] };
        const end = { x: nums[base + 2], y: nums[base + 3] };
        for (let i = 1; i <= curveSamples; i++) {
          points.push(quadraticAt(cursor, c, end, i / curveSamples));
        }
        cursor = end;
      } else {
        const c1 = { x: nums[base], y: nums[base + 1] };
        const c2 = { x: nums[base + 2], y: nums[base + 3] };
        const end = { x: nums[base + 4], y: nums[base + 5] };
        for (let i = 1; i <= curveSamples; i++) {
          points.push(cubicAt(cursor, c1, c2, end, i / curveSamples));
        }
        cursor = end;
      }
    }
  }
  return points;
}

function pointToSegment(p: Point, a: Point, b: Point): number {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const lenSq = dx * dx + dy * dy;
  if (lenSq === 0) return Math.hypot(p.x - a.x, p.y - a.y);
  let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / lenSq;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}

/** Shortest distance from a point to a polyline. */
export function distanceToPolyline(p: Point, poly: readonly Point[]): number {
  if (poly.length === 0) return Infinity;
  if (poly.length === 1) return Math.hypot(p.x - poly[0].x, p.y - poly[0].y);
  let best = Infinity;
  for (let i = 1; i < poly.length; i++) {
    const d = pointToSegment(p, poly[i - 1], poly[i]);
    if (d < best) best = d;
  }
  return best;
}

/** Walk a polyline emitting a point every `step` pixels of arc length. */
export function resample(poly: readonly Point[], step: number): Point[] {
  if (poly.length === 0) return [];
  const out: Point[] = [{ ...poly[0] }];
  let carry = 0;
  for (let i = 1; i < poly.length; i++) {
    const a = poly[i - 1];
    const b = poly[i];
    const len = Math.hypot(b.x - a.x, b.y - a.y);
    if (len === 0) continue;
    let travelled = step - carry;
    while (travelled <= len) {
      const t = travelled / len;
      out.push({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });
      travelled += step;
    }
    carry = (carry + len) % step;
  }
  return out;
}

export interface SharedRunOptions {
  /** Centre-to-centre distance below which two 2px traces read as one. */
  tolerance?: number;
  /** Sampling pitch along the measured path, in flow pixels. */
  step?: number;
}

export interface SharedRun {
  /** Length of the shared stretch in flow pixels; 0 when the paths never coincide. */
  length: number;
  /** Every sampled point of the shared stretch, so a test can say where it is. */
  points: Point[];
}

/**
 * Longest contiguous stretch of `a` that runs within `tolerance` of `b`.
 *
 * Direction-agnostic on purpose: a wire drawn right-to-left on top of one drawn
 * left-to-right still looks like a single wire. The points are returned as well
 * as the length, because "how long do they coincide" and "where do they coincide"
 * are different questions and the second is what separates an unavoidable stub at
 * a shared port from a shared lane out in open canvas.
 */
export function longestSharedRun(
  a: string,
  b: string,
  options: SharedRunOptions = {},
): SharedRun {
  const tolerance = options.tolerance ?? 1.5;
  const step = options.step ?? 2;
  const samples = resample(flattenPath(a), step);
  const other = flattenPath(b);
  let best: Point[] = [];
  let run: Point[] = [];
  for (const p of samples) {
    if (distanceToPolyline(p, other) <= tolerance) {
      run.push(p);
      if (run.length > best.length) best = run.slice();
    } else {
      run = [];
    }
  }
  return { length: best.length * step, points: best };
}

/**
 * Longest contiguous stretch of `a` that runs within `tolerance` of `b`, in
 * pixels.
 */
export function sharedRunLength(a: string, b: string, options: SharedRunOptions = {}): number {
  return longestSharedRun(a, b, options).length;
}

/**
 * Worst superposition across a whole set of paths: the largest
 * {@link sharedRunLength} over every unordered pair, measured both ways round
 * because the two paths can have very different lengths.
 */
export function worstSharedRun(
  paths: readonly string[],
  options: SharedRunOptions = {},
): { length: number; pair: [number, number] } {
  let worst = { length: 0, pair: [-1, -1] as [number, number] };
  for (let i = 0; i < paths.length; i++) {
    for (let j = i + 1; j < paths.length; j++) {
      const forward = sharedRunLength(paths[i], paths[j], options);
      const backward = sharedRunLength(paths[j], paths[i], options);
      const length = Math.max(forward, backward);
      if (length > worst.length) worst = { length, pair: [i, j] };
    }
  }
  return worst;
}
