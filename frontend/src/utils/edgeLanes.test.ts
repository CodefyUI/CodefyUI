import { describe, it, expect } from 'vitest';
import demo from '../test/fixtures/edgeOverlapDemo.json';
import {
  computeEdgeLanes,
  detourOffset,
  detourStep,
  edgeLaneSignature,
  laneDistance,
  laneStep,
  pickLaneAnchor,
  DETOUR_STEP_MAX,
  DETOUR_STEP_MIN,
  LANE_BASE,
  LANE_SPREAD_MAX,
  LANE_STEP_MAX,
  LANE_STEP_MIN,
  SOLO_LANE,
  type LaneEdgeInput,
} from './edgeLanes';

function wire(
  id: string,
  source: string,
  target: string,
  sourceHandle?: string,
  targetHandle?: string,
): LaneEdgeInput {
  return { id, source, target, sourceHandle, targetHandle };
}

describe('computeEdgeLanes', () => {
  it('leaves an edge with no siblings on the solo lane', () => {
    const lanes = computeEdgeLanes([wire('e', 'A', 'B')]);
    expect(lanes.get('e')).toEqual(SOLO_LANE);
  });

  it('returns an empty map for an empty edge list', () => {
    expect(computeEdgeLanes([]).size).toBe(0);
  });

  it('gives every edge leaving one node a distinct out slot', () => {
    const edges = [
      wire('e1', 'A', 'B', 'out'),
      wire('e2', 'A', 'C', 'out'),
      wire('e3', 'A', 'D', 'out'),
    ];
    const lanes = computeEdgeLanes(edges);
    const slots = edges.map((e) => lanes.get(e.id)!.outSlot);
    expect(new Set(slots).size).toBe(3);
    expect(slots.every((s) => s >= 0 && s < 3)).toBe(true);
    expect(edges.every((e) => lanes.get(e.id)!.outCount === 3)).toBe(true);
  });

  it('groups by source NODE, not by source handle', () => {
    // Every output handle sits on the same vertical edge of the card, so two
    // edges leaving different handles still bend at the same x by default.
    // Grouping per handle would leave that pair colliding.
    const edges = [wire('e1', 'A', 'B', 'left'), wire('e2', 'A', 'C', 'right')];
    const lanes = computeEdgeLanes(edges);
    expect(lanes.get('e1')!.outCount).toBe(2);
    expect(lanes.get('e2')!.outCount).toBe(2);
    expect(lanes.get('e1')!.outSlot).not.toBe(lanes.get('e2')!.outSlot);
  });

  it('gives every edge arriving at one node a distinct in slot', () => {
    const edges = [
      wire('e1', 'A', 'D', 'o', 'x'),
      wire('e2', 'B', 'D', 'o', 'y'),
      wire('e3', 'C', 'D', 'o', 'z'),
    ];
    const lanes = computeEdgeLanes(edges);
    const slots = edges.map((e) => lanes.get(e.id)!.inSlot);
    expect(new Set(slots).size).toBe(3);
    expect(edges.every((e) => lanes.get(e.id)!.inCount === 3)).toBe(true);
    // None of them shares a source, so the out groups stay solo.
    expect(edges.every((e) => lanes.get(e.id)!.outCount === 1)).toBe(true);
  });

  it('separates parallel edges between the same node pair', () => {
    const edges = [wire('e1', 'A', 'B', 'p', 'x'), wire('e2', 'A', 'B', 'q', 'y')];
    const lanes = computeEdgeLanes(edges);
    expect(lanes.get('e1')!.outSlot).not.toBe(lanes.get('e2')!.outSlot);
    expect(lanes.get('e1')!.inSlot).not.toBe(lanes.get('e2')!.inSlot);
  });

  it('separates an exact duplicate connection', () => {
    // `onConnect` appends without de-duplicating, so two edges can carry the
    // same endpoints and handles. Identical geometry means identical paths
    // unless the lanes tell them apart.
    const edges = [wire('e1', 'A', 'B', 'o', 'i'), wire('e2', 'A', 'B', 'o', 'i')];
    const lanes = computeEdgeLanes(edges);
    expect(lanes.get('e1')!.outSlot).not.toBe(lanes.get('e2')!.outSlot);
  });

  it('is independent of the order the edges sit in the array', () => {
    const edges = [
      wire('c', 'A', 'B', 'out'),
      wire('a', 'A', 'C', 'out'),
      wire('b', 'A', 'D', 'out'),
    ];
    const forward = computeEdgeLanes(edges);
    const reversed = computeEdgeLanes([...edges].reverse());
    for (const e of edges) {
      expect(reversed.get(e.id)).toEqual(forward.get(e.id));
    }
  });

  it('orders a group by handle then id, so a reload reproduces it exactly', () => {
    const lanes = computeEdgeLanes([
      wire('zzz', 'A', 'B', 'aaa'),
      wire('aaa', 'A', 'C', 'bbb'),
    ]);
    // 'aaa' sorts before 'bbb' as a handle, so the edge on handle 'aaa' takes
    // slot 0 even though its id sorts last.
    expect(lanes.get('zzz')!.outSlot).toBe(0);
    expect(lanes.get('aaa')!.outSlot).toBe(1);
  });

  it('treats a missing handle as its own stable key rather than throwing', () => {
    // The layers editor connects handles that carry no id at all.
    const lanes = computeEdgeLanes([
      { id: 'e1', source: 'A', target: 'B' },
      { id: 'e2', source: 'A', target: 'C', sourceHandle: null },
    ]);
    expect(lanes.get('e1')!.outCount).toBe(2);
    expect(lanes.get('e1')!.outSlot).not.toBe(lanes.get('e2')!.outSlot);
  });

  it('keeps only the first of two edges sharing an id', () => {
    const lanes = computeEdgeLanes([wire('dup', 'A', 'B'), wire('dup', 'C', 'D')]);
    expect(lanes.size).toBe(1);
    expect(lanes.get('dup')).toEqual(SOLO_LANE);
  });
});

describe('lane geometry', () => {
  it('spaces small groups at the full step', () => {
    expect(laneStep(1)).toBe(LANE_STEP_MAX);
    expect(laneStep(2)).toBe(LANE_STEP_MAX);
  });

  it('tightens the step so a wide bus stays inside the spread budget', () => {
    // The budget holds until the floor takes over; past that the ladder grows
    // rather than letting two traces close to within an invisible gap.
    for (const count of [4, 9, 20, 40]) {
      const spread = (count - 1) * laneStep(count);
      const floor = (count - 1) * LANE_STEP_MIN;
      expect(spread).toBeLessThanOrEqual(Math.max(LANE_SPREAD_MAX, floor));
      expect(laneStep(count)).toBeGreaterThanOrEqual(LANE_STEP_MIN);
      expect(laneStep(count)).toBeLessThanOrEqual(LANE_STEP_MAX);
    }
    expect(laneStep(20)).toBeLessThan(LANE_STEP_MAX);
  });

  it('never lets the step fall below the floor, however wide the group', () => {
    expect(laneStep(500)).toBe(LANE_STEP_MIN);
    expect(detourStep(500)).toBe(DETOUR_STEP_MIN);
    expect(detourStep(2)).toBe(DETOUR_STEP_MAX);
  });

  it('is strictly increasing in the slot, which is what makes bends distinct', () => {
    for (const count of [2, 3, 8, 20, 64]) {
      for (let slot = 1; slot < count; slot++) {
        expect(laneDistance(slot, count)).toBeGreaterThan(laneDistance(slot - 1, count));
        expect(detourOffset(slot, count)).toBeGreaterThan(detourOffset(slot - 1, count));
      }
    }
  });

  it('keeps the port stub short enough to be a stub', () => {
    // LANE_BASE is the whole allowance two wires leaving one handle get, so it is
    // the number that decides whether they read as one wire. It only has to clear
    // the handle graphic (about 12px across) and leave room for a corner; it used
    // to be forced up to 28 by xyflow's fixed 20px gap, which is why laned routes
    // are drawn directly now. If this ever climbs back above ~16 the fan-out is
    // sharing a lane again, not a stub.
    expect(LANE_BASE).toBeLessThanOrEqual(16);
    expect(LANE_BASE).toBeGreaterThanOrEqual(10);
    expect(laneDistance(0, 5)).toBe(LANE_BASE);
    expect(detourOffset(0, 5)).toBe(0);
  });

  it('keeps the worst pair in a four-way fan inside half of what it was', () => {
    // The owner measured 59px on the real ResNet graph with the old constants.
    // laneDistance(n - 2, n) is the worst any pair of a fan of n can share.
    expect(laneDistance(2, 4)).toBeLessThanOrEqual(30);
    expect(laneDistance(0, 2)).toBeLessThanOrEqual(16);
  });
});

describe('pickLaneAnchor', () => {
  it('routes by geometry alone when neither end has siblings', () => {
    expect(pickLaneAnchor(SOLO_LANE)).toEqual({ anchor: 'none', slot: 0, count: 1 });
  });

  it('prefers the source end, because a fan-out shares a point', () => {
    const choice = pickLaneAnchor({ outSlot: 2, outCount: 4, inSlot: 1, inCount: 3 });
    expect(choice).toEqual({ anchor: 'source', slot: 2, count: 4 });
  });

  it('falls back to the target end for a fan-in', () => {
    const choice = pickLaneAnchor({ outSlot: 0, outCount: 1, inSlot: 2, inCount: 3 });
    expect(choice).toEqual({ anchor: 'target', slot: 2, count: 3 });
  });
});

describe('the edge-overlap demo fixture', () => {
  // src/test/fixtures/edgeOverlapDemo.json is what a human imports to look at
  // this fix. If someone edits it into a graph that no longer contains the
  // shapes it claims to, this notices.
  it('still contains every wiring shape it advertises', () => {
    const lanes = computeEdgeLanes(demo.edges as LaneEdgeInput[]);
    const lane = (id: string) => {
      const found = lanes.get(id);
      if (!found) throw new Error(`fixture lost edge ${id}`);
      return found;
    };
    // Band 1: four wires leaving one output of one node.
    expect(lane('b1-fan-a').outCount).toBe(4);
    expect(new Set(['b1-fan-a', 'b1-fan-b', 'b1-fan-c', 'b1-fan-d'].map((id) => lane(id).outSlot)).size).toBe(4);
    // Band 2: two outputs of one node, targets stacked at one x.
    expect(lane('b2-tensor').outCount).toBe(2);
    expect(lane('b2-tensor').outSlot).not.toBe(lane('b2-labels').outSlot);
    // Band 3: two wires between one pair of nodes.
    expect(lane('b3-tensor').outCount).toBe(2);
    expect(lane('b3-tensor').inCount).toBe(2);
    // Band 4: three sources in a column arriving at one node.
    expect(lane('b4-a').inCount).toBe(3);
    expect(new Set(['b4-a', 'b4-b', 'b4-c'].map((id) => lane(id).inSlot)).size).toBe(3);
    // Band 5: a fan-out whose far targets are long enough to take the detour.
    expect(lane('b5-far-a').outCount).toBe(3);
    expect(lane('b5-far-a').outSlot).not.toBe(lane('b5-far-b').outSlot);
  });
});

describe('edgeLaneSignature', () => {
  it('ignores fields lanes do not read, so selection never rebuilds the map', () => {
    const a = [wire('e1', 'A', 'B', 'o', 'i')];
    const b = [wire('e1', 'A', 'B', 'o', 'i')];
    expect(edgeLaneSignature(a)).toBe(edgeLaneSignature(b));
  });

  it('changes when an edge is rewired', () => {
    expect(edgeLaneSignature([wire('e1', 'A', 'B', 'o', 'i')])).not.toBe(
      edgeLaneSignature([wire('e1', 'A', 'C', 'o', 'i')]),
    );
  });

  it('cannot be spoofed by shifting characters between adjacent fields', () => {
    // Field separators are control characters, so 'ab'+'c' and 'a'+'bc' differ.
    expect(edgeLaneSignature([wire('ab', 'c', 'X')])).not.toBe(
      edgeLaneSignature([wire('a', 'bc', 'X')]),
    );
  });
});
