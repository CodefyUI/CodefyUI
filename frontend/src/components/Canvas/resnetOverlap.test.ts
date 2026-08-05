/**
 * The no-superposition invariant, on the real ResNet-18 / CIFAR-10 example.
 *
 * Not a synthetic fixture: the wiring is `examples/Usage_Example/
 * ResNet18-CIFAR10-Baseline/graph.json` as merged, and the handle coordinates in
 * `src/test/fixtures/resnet18EdgeGeometry.json` were read off that graph rendered
 * by the real app in Chrome. Routing does not move handles, so those coordinates
 * are ground truth for any routing we care to test.
 *
 * The assertion is an invariant, not a threshold. For every pair of edges:
 *
 *   - if they share no handle, they must not coincide anywhere, at all; and
 *   - if they do share a handle, every point they have in common must lie within
 *     the port stub of that shared handle.
 *
 * The port stub is the one concession, and it is forced by geometry rather than
 * chosen for convenience. Orthogonal routes leaving a common point in a common
 * direction each occupy some prefix [0, L] of the departure line, so two of them
 * always share min(L_i, L_j); and the L values have to differ by at least a lane
 * pitch or the wires' turns land on top of each other instead. Sorting them,
 * L_(k) >= LANE_BASE + k * step, so the worst pair in a fan of n shares exactly
 * LANE_BASE + (n - 2) * step and no routing can do better. That is 12px for a
 * pair, 20px for three, 28px for four.
 */
import { readFileSync } from 'node:fs';
import { describe, it, expect } from 'vitest';
import { Position } from '@xyflow/react';
import { longestSharedRun, type Point } from '../../test/pathOverlap';
import geometry from '../../test/fixtures/resnet18EdgeGeometry.json';
import {
  computeEdgeLanes,
  laneDistance,
  pickLaneAnchor,
  type EdgeLane,
  type LaneEdgeInput,
} from '../../utils/edgeLanes';
import { resolveEdgePath } from './SmartDataEdge';

const GRAPH_PATH = '../examples/Usage_Example/ResNet18-CIFAR10-Baseline/graph.json';

/**
 * Slack on the stub bound: the 2px sampling pitch plus the 1.5px coincidence
 * tolerance, which can carry a run a little past the point the wires part.
 */
const MEASUREMENT_SLACK = 8;

/**
 * Below this, a coincidence is a crossing rather than a superposition, and
 * crossings are explicitly allowed. Two traces meeting at an angle stay inside the
 * 1.5px coincidence tolerance for roughly 2 * 1.5 / sin(angle) - about 6px at 30
 * degrees, more as the angle shallows - so a floor is needed to tell "they cross"
 * from "they are drawn as one". 12px is the floor the owner's own Chrome
 * measurement used, kept so the two measurements are comparable.
 */
const CROSSING_FLOOR = 12;

interface RawEdge extends LaneEdgeInput {
  type?: string;
}

interface Wire extends RawEdge {
  sourceX: number;
  sourceY: number;
  targetX: number;
  targetY: number;
}

function loadWires(): Wire[] {
  let raw: string;
  try {
    raw = readFileSync(GRAPH_PATH, 'utf8');
  } catch {
    throw new Error(
      `${GRAPH_PATH} is missing. If the example moved, repoint this test rather than deleting it - it is the only overlap measurement taken from a graph users actually open.`,
    );
  }
  const graph = JSON.parse(raw) as { edges: RawEdge[] };
  const at = new Map(
    (geometry.edges as Array<[string, number, number, number, number]>).map((row) => [
      row[0],
      { sourceX: row[1], sourceY: row[2], targetX: row[3], targetY: row[4] },
    ]),
  );
  return graph.edges.map((e) => {
    const g = at.get(e.id);
    if (!g) {
      throw new Error(
        `no captured geometry for edge ${e.id}. The example gained an edge; re-capture resnet18EdgeGeometry.json (the recipe is in the file).`,
      );
    }
    return { ...e, ...g };
  });
}

const WIRES = loadWires();
const LANES = computeEdgeLanes(WIRES);

function pathsFor(circuit: boolean): string[] {
  return WIRES.map((w) =>
    resolveEdgePath({
      sourceX: w.sourceX,
      sourceY: w.sourceY,
      targetX: w.targetX,
      targetY: w.targetY,
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
      circuit,
      lane: LANES.get(w.id),
    }),
  );
}

const sourceKey = (w: Wire) => `${w.source}\u0001${w.sourceHandle ?? ''}`;
const targetKey = (w: Wire) => `${w.target}\u0001${w.targetHandle ?? ''}`;

/** Where two edges are pinned together, and how far from it they may coincide. */
interface SharedEnd {
  at: Point;
  stub: number;
}

function sharedEnds(a: Wire, b: Wire, laneA: EdgeLane, laneB: EdgeLane): SharedEnd[] {
  const ends: SharedEnd[] = [];
  if (sourceKey(a) === sourceKey(b)) {
    const slots = [pickLaneAnchor(laneA), pickLaneAnchor(laneB)];
    const count = Math.max(laneA.outCount, laneB.outCount);
    const first = Math.min(...slots.map((s) => s.slot));
    ends.push({ at: { x: a.sourceX, y: a.sourceY }, stub: laneDistance(first, count) });
  }
  if (targetKey(a) === targetKey(b)) {
    const count = Math.max(laneA.inCount, laneB.inCount);
    const first = Math.min(laneA.inSlot, laneB.inSlot);
    ends.push({ at: { x: a.targetX, y: a.targetY }, stub: laneDistance(first, count) });
  }
  return ends;
}

const dist = (p: Point, q: Point) => Math.hypot(p.x - q.x, p.y - q.y);

interface Violation {
  pair: string;
  length: number;
  reason: string;
}

function findViolations(circuit: boolean): { violations: Violation[]; shared: string[] } {
  const paths = pathsFor(circuit);
  const violations: Violation[] = [];
  const shared: string[] = [];
  for (let i = 0; i < WIRES.length; i++) {
    for (let j = i + 1; j < WIRES.length; j++) {
      const forward = longestSharedRun(paths[i], paths[j]);
      const backward = longestSharedRun(paths[j], paths[i]);
      const run = forward.length >= backward.length ? forward : backward;
      if (run.length < CROSSING_FLOOR) continue;
      const a = WIRES[i];
      const b = WIRES[j];
      const pair = `${a.id} / ${b.id}`;
      const ends = sharedEnds(a, b, LANES.get(a.id)!, LANES.get(b.id)!);
      if (ends.length === 0) {
        violations.push({ pair, length: run.length, reason: 'coincide without sharing a handle' });
        continue;
      }
      const explained = ends.some((end) =>
        run.points.every((p) => dist(p, end.at) <= end.stub + MEASUREMENT_SLACK),
      );
      if (!explained) {
        const nearest = Math.min(...ends.map((e) => e.stub));
        violations.push({
          pair,
          length: run.length,
          reason: `run reaches beyond the ${nearest}px port stub of the shared handle`,
        });
        continue;
      }
      shared.push(`${pair} ${Math.round(run.length)}px`);
    }
  }
  return { violations, shared };
}

/**
 * The known exceptions, named rather than hidden behind a threshold.
 *
 * All three are the documented residual: two edges with no handle in common whose
 * routes happen to land on the same line. The mechanism is worth stating exactly,
 * because it is not the port stub and no choice of stub size affects it. Lane
 * slots are numbered per source node, each group starting from the same base, so
 * two nodes that sit in the SAME COLUMN hand their k-th edge the same turn
 * distance - and the same turn distance from the same x is the same line.
 * `opt` and `ds-test` are both at x=1264.5 here; `e-d3` and `e-m1` are the mirror
 * case on the arrival side, both ending at x=1715.5.
 *
 * Separating these needs a router that looks at where the nodes actually are and
 * re-runs whenever one moves - deliberately out of scope, see the residuals in
 * the edgeLanes module docs. They are listed here so they stay visible, so that
 * any NEW pair fails this test loudly, and so nobody reads the suite as promising
 * more than it delivers.
 */
const KNOWN_CROSS_GROUP_COLLISIONS = ['e-d2 / e-c2', 'e-d3 / e-m1', 'e-o1 / e-e2'];

describe.each([
  ['circuit', true],
  ['curve', false],
])('ResNet-18 example, %s style', (_style, circuit) => {
  it('never draws two edges sharing a handle outside that handle stub', () => {
    // The hard invariant. Anything that shares a handle - every pair in the
    // owner's Chrome table - must coincide only inside the port stub.
    const { violations } = findViolations(circuit as boolean);
    const stubBreaks = violations.filter((v) => !v.reason.startsWith('coincide without'));
    expect(stubBreaks.map((v) => `${v.pair}: ${Math.round(v.length)}px, ${v.reason}`)).toEqual([]);
  });

  it('grows no new collision between edges that share no handle', () => {
    const { violations } = findViolations(circuit as boolean);
    const crossGroup = violations
      .filter((v) => v.reason.startsWith('coincide without'))
      .map((v) => v.pair)
      .sort();
    expect(crossGroup).toEqual(
      KNOWN_CROSS_GROUP_COLLISIONS.filter((p) => crossGroup.includes(p)),
    );
    expect(crossGroup.every((p) => KNOWN_CROSS_GROUP_COLLISIONS.includes(p))).toBe(true);
  });

  it('keeps every shared-handle stretch inside the geometric floor for its fan', () => {
    // The largest fan-out here is four wires, so nothing that shares a handle may
    // share more than LANE_BASE + 2 * step of path. Stated as a number as well as
    // an invariant, so loosening the constants shows up here.
    const worstFan = 4;
    const ceiling = laneDistance(worstFan - 2, worstFan) + MEASUREMENT_SLACK;
    const { shared } = findViolations(circuit as boolean);
    const lengths = shared.map((s) => Number(s.match(/(\d+)px$/)![1]));
    expect(lengths.length).toBeGreaterThan(0);
    for (const len of lengths) expect(len).toBeLessThanOrEqual(ceiling);
    expect(ceiling).toBeLessThanOrEqual(36);
  });
});

describe('ResNet-18 example, the shape of the graph', () => {
  it('is the graph these tests claim it is', () => {
    expect(WIRES).toHaveLength(27);
    const triggers = WIRES.filter((w) => w.type === 'trigger');
    expect(triggers).toHaveLength(4);
    expect(new Set(triggers.map((t) => sourceKey(t))).size).toBe(1);
    // The fan-out the owner sees first, and the one the whole fix is for.
    expect(LANES.get('e-trigger-1')!.outCount).toBe(4);
  });

  it('has no edge whose captured geometry is stale', () => {
    for (const w of WIRES) {
      expect(Number.isFinite(w.sourceX)).toBe(true);
      expect(Number.isFinite(w.targetY)).toBe(true);
    }
  });
});
