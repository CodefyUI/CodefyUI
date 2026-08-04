/**
 * The no-superposition guarantee, asserted on rendered path strings.
 *
 * These tests deliberately know nothing about lanes or slots: they take the paths
 * the router produced and measure the longest stretch on which two of them are
 * drawn on top of each other (see `src/test/pathOverlap.ts`). A crossing scores
 * near zero, which is what the requirement asks for - crossings are allowed,
 * superposition is not.
 *
 * Each limit below is derived in its own comment. Where a limit is not zero it is
 * the shared stub at a shared handle: two orthogonal traces leaving one point in
 * one direction must run together until the first of them turns, and no routing
 * can avoid that. What the lanes buy is that the stub is a small constant instead
 * of half the length of the wire.
 */
import { describe, it, expect } from 'vitest';
import { Position, type EdgeProps } from '@xyflow/react';
import { renderWithFlow } from '../../test/utils';
import { useUIStore } from '../../store/uiStore';
import { worstSharedRun } from '../../test/pathOverlap';
import { computeEdgeLanes, laneDistance, type LaneEdgeInput } from '../../utils/edgeLanes';

/** xyflow's smoothstep handle gap; every route leaves straight for this far. */
const SMOOTHSTEP_GAP = 20;
import { resolveEdgePath, SmartDataEdge } from './SmartDataEdge';
import { EdgeLaneProvider } from './EdgeLaneContext';

interface Wire extends LaneEdgeInput {
  sourceX: number;
  sourceY: number;
  targetX: number;
  targetY: number;
}

interface Options {
  circuit?: boolean;
  /** Top/Bottom handles, as used by the subgraph layer editor. */
  vertical?: boolean;
  /** Drop the lane map, reproducing how these edges routed before the fix. */
  unlaned?: boolean;
}

function pathsFor(wires: Wire[], options: Options = {}): string[] {
  const lanes = computeEdgeLanes(wires);
  return wires.map((w) =>
    resolveEdgePath({
      sourceX: w.sourceX,
      sourceY: w.sourceY,
      targetX: w.targetX,
      targetY: w.targetY,
      sourcePosition: options.vertical ? Position.Bottom : Position.Right,
      targetPosition: options.vertical ? Position.Top : Position.Left,
      circuit: options.circuit ?? true,
      lane: options.unlaned ? undefined : lanes.get(w.id),
    }),
  );
}

function worst(wires: Wire[], options: Options = {}): number {
  return worstSharedRun(pathsFor(wires, options)).length;
}

/** Both edge styles, since the routing branches on it. */
const STYLES: Array<[string, boolean]> = [
  ['circuit', true],
  ['curve', false],
];

function fanOut(n: number, spreadY: number): Wire[] {
  return Array.from({ length: n }, (_, i) => ({
    id: `e${i}`,
    source: 'A',
    target: `T${i}`,
    sourceHandle: 'out',
    targetHandle: 'in',
    sourceX: 0,
    sourceY: 0,
    targetX: 420 + (i % 3) * 30,
    targetY: (i + 1) * spreadY,
  }));
}

/**
 * Upper bound on the stub a fan-out of `count` wires can share: the last wire to
 * turn runs alongside the second-to-last until that one turns, plus a little
 * slack for the rounded corner and the measurement tolerance.
 */
function fanOutStubLimit(count: number): number {
  return laneDistance(Math.max(count - 2, 0), count) + 12;
}

describe('fan-out: several wires leaving one output', () => {
  for (const [style, circuit] of STYLES) {
    it(`${style}: three wires share only the stub, where they used to share half the run`, () => {
      const wires = fanOut(3, 160);
      const before = worst(wires, { circuit, unlaned: true });
      const after = worst(wires, { circuit });
      expect(before).toBeGreaterThan(200);
      expect(after).toBeLessThanOrEqual(fanOutStubLimit(3));
      expect(after).toBeLessThan(before / 4);
    });

    it(`${style}: eight wires stay inside the stub bound`, () => {
      const wires = fanOut(8, 90);
      expect(worst(wires, { circuit, unlaned: true })).toBeGreaterThan(600);
      expect(worst(wires, { circuit })).toBeLessThanOrEqual(fanOutStubLimit(8));
    });

    it(`${style}: a twenty-wire bus does not explode`, () => {
      // The ladder tightens as the group grows, so the worst pair still shares
      // far less than the wires used to, and the fan stays a fan.
      const wires = fanOut(20, 45);
      expect(worst(wires, { circuit, unlaned: true })).toBeGreaterThan(900);
      expect(worst(wires, { circuit })).toBeLessThanOrEqual(fanOutStubLimit(20));
    });
  }
});

describe('two outputs of one node', () => {
  // Both handles sit on the same edge of the card, so without node-level
  // grouping these two bend at the same x and their risers superimpose.
  const wires: Wire[] = [
    { id: 'p', source: 'A', target: 'B', sourceHandle: 'a', targetHandle: 'in', sourceX: 0, sourceY: 0, targetX: 500, targetY: 90 },
    { id: 'q', source: 'A', target: 'C', sourceHandle: 'b', targetHandle: 'in', sourceX: 0, sourceY: 24, targetX: 500, targetY: 150 },
  ];

  it('circuit: risers no longer share a column', () => {
    expect(worst(wires, { circuit: true, unlaned: true })).toBeGreaterThan(50);
    expect(worst(wires, { circuit: true })).toBeLessThanOrEqual(8);
  });
});

describe('parallel wires between one pair of nodes', () => {
  const wires: Wire[] = [
    { id: 'p', source: 'A', target: 'B', sourceHandle: 'a', targetHandle: 'x', sourceX: 0, sourceY: 0, targetX: 400, targetY: 100 },
    { id: 'q', source: 'A', target: 'B', sourceHandle: 'b', targetHandle: 'y', sourceX: 0, sourceY: 24, targetX: 400, targetY: 124 },
  ];

  it('circuit: the two wires are separated end to end', () => {
    expect(worst(wires, { circuit: true, unlaned: true })).toBeGreaterThan(50);
    expect(worst(wires, { circuit: true })).toBeLessThanOrEqual(8);
  });

});

describe('fan-in: several wires arriving at one node', () => {
  // Three sources in a column, so every wire would otherwise turn at the same x
  // and the three risers would run down the same line.
  const wires: Wire[] = [
    { id: 'p', source: 'A', target: 'D', sourceHandle: 'o', targetHandle: 'x', sourceX: 0, sourceY: 0, targetX: 480, targetY: 300 },
    { id: 'q', source: 'B', target: 'D', sourceHandle: 'o', targetHandle: 'y', sourceX: 0, sourceY: 60, targetX: 480, targetY: 324 },
    { id: 'r', source: 'C', target: 'D', sourceHandle: 'o', targetHandle: 'z', sourceX: 0, sourceY: 120, targetX: 480, targetY: 348 },
  ];

  for (const [style, circuit] of STYLES) {
    it(`${style}: the approach columns are separated`, () => {
      expect(worst(wires, { circuit, unlaned: true })).toBeGreaterThan(200);
      expect(worst(wires, { circuit })).toBeLessThanOrEqual(8);
    });
  }
});

describe('skip detours over a row', () => {
  const wires: Wire[] = [
    { id: 'p', source: 'A', target: 'X', sourceHandle: 'out', targetHandle: 'in', sourceX: 0, sourceY: 0, targetX: 900, targetY: 0 },
    { id: 'q', source: 'A', target: 'Y', sourceHandle: 'out', targetHandle: 'in', sourceX: 0, sourceY: 0, targetX: 940, targetY: 10 },
  ];

  it('circuit: two skips climb in different columns and ride different corridors', () => {
    // They used to share the climb at sourceX + 50 for the whole height of the
    // corridor; now only the pull-out stub is common.
    expect(worst(wires, { circuit: true, unlaned: true })).toBeGreaterThan(200);
    expect(worst(wires, { circuit: true })).toBeLessThanOrEqual(60);
  });

  it('curve: two arcs leave the handle on different tangents', () => {
    expect(worst(wires, { circuit: false, unlaned: true })).toBeGreaterThan(80);
    expect(worst(wires, { circuit: false })).toBeLessThanOrEqual(30);
  });
});

describe('backward wires', () => {
  const wires: Wire[] = [
    { id: 'p', source: 'A', target: 'B', sourceHandle: 'out', targetHandle: 'in', sourceX: 600, sourceY: 0, targetX: 300, targetY: 120 },
    { id: 'q', source: 'A', target: 'C', sourceHandle: 'out', targetHandle: 'in', sourceX: 600, sourceY: 0, targetX: 320, targetY: 150 },
  ];

  it('circuit: the drops onto the corridors do not share a column-length', () => {
    // A backward route leaves rightwards through xyflow's 20px gap before it can
    // drop, so the stub here is that gap plus the distance to the nearer
    // corridor, rather than the corridor distance alone.
    const limit = SMOOTHSTEP_GAP + laneDistance(0, 2) + 12;
    expect(worst(wires, { circuit: true, unlaned: true })).toBeGreaterThan(70);
    expect(worst(wires, { circuit: true })).toBeLessThanOrEqual(limit);
  });
});

describe('vertical sub-canvas (Top/Bottom handles)', () => {
  // The subgraph layer editor flows downwards and its handles carry no id.
  const wires: Wire[] = [
    { id: 'p', source: 'A', target: 'B', sourceX: 0, sourceY: 0, targetX: 160, targetY: 300 },
    { id: 'q', source: 'A', target: 'C', sourceX: 0, sourceY: 0, targetX: 200, targetY: 420 },
    { id: 'r', source: 'A', target: 'D', sourceX: 0, sourceY: 0, targetX: 240, targetY: 540 },
  ];

  it('circuit: the same guarantee holds on the other axis', () => {
    expect(worst(wires, { circuit: true, vertical: true, unlaned: true })).toBeGreaterThan(200);
    expect(worst(wires, { circuit: true, vertical: true })).toBeLessThanOrEqual(fanOutStubLimit(3));
  });
});

describe('known gap: two wires on the same axis line', () => {
  it('cannot separate two edges that share both endpoints', () => {
    // `onConnect` does not de-duplicate, so the very same connection can be drawn
    // twice. Both routes are pinned at both ends and their approach runs along
    // the target handle's own row, so nothing short of bowing one of them off
    // that row would separate them - and the two describe the same connection
    // anyway, so one wire is arguably the honest picture.
    const duplicated: Wire[] = [
      { id: 'p', source: 'A', target: 'B', sourceHandle: 'o', targetHandle: 'i', sourceX: 0, sourceY: 0, targetX: 400, targetY: 100 },
      { id: 'q', source: 'A', target: 'B', sourceHandle: 'o', targetHandle: 'i', sourceX: 0, sourceY: 0, targetX: 400, targetY: 100 },
    ];
    expect(worst(duplicated, { circuit: true })).toBeGreaterThan(150);
  });

  it('still superimposes when both targets sit at the source handle height', () => {
    // A route from (x, y) to (X, y) IS the axis line, so no choice of turning
    // distance can pull two of them apart - only lifting one off the row could,
    // and that would put a gratuitous hop on every aligned wire whose sibling is
    // merely nearby. It needs the two target cards to touch or overlap: give the
    // far one any clearance and it crosses the 380px skip threshold and detours.
    //
    // If this test ever starts failing the gap has been closed. Delete it and
    // update the residuals in the edgeLanes module docs.
    const wires: Wire[] = [
      { id: 'p', source: 'A', target: 'B', sourceHandle: 'out', targetHandle: 'in', sourceX: 0, sourceY: 0, targetX: 200, targetY: 0 },
      { id: 'q', source: 'A', target: 'C', sourceHandle: 'out', targetHandle: 'in', sourceX: 0, sourceY: 0, targetX: 340, targetY: 0 },
    ];
    expect(worst(wires, { circuit: true })).toBeGreaterThan(150);
  });

  it('is separated as soon as the far target clears the skip threshold', () => {
    const wires: Wire[] = [
      { id: 'p', source: 'A', target: 'B', sourceHandle: 'out', targetHandle: 'in', sourceX: 0, sourceY: 0, targetX: 220, targetY: 0 },
      { id: 'q', source: 'A', target: 'C', sourceHandle: 'out', targetHandle: 'in', sourceX: 0, sourceY: 0, targetX: 640, targetY: 0 },
    ];
    expect(worst(wires, { circuit: true })).toBeLessThanOrEqual(80);
  });
});

describe('determinism', () => {
  it('produces identical paths however the edges are ordered in the array', () => {
    const wires = fanOut(6, 120);
    const forward = pathsFor(wires);
    const shuffled = [...wires].reverse();
    const back = pathsFor(shuffled);
    for (let i = 0; i < wires.length; i++) {
      expect(back[shuffled.length - 1 - i]).toBe(forward[i]);
    }
  });

  it('produces identical paths on a second run', () => {
    const wires = fanOut(5, 120);
    expect(pathsFor(wires)).toEqual(pathsFor(wires));
  });

  it('leaves a lone edge routed exactly as it was before lanes existed', () => {
    const lone: Wire[] = [
      { id: 'e', source: 'A', target: 'B', sourceHandle: 'o', targetHandle: 'i', sourceX: 0, sourceY: 0, targetX: 300, targetY: 90 },
    ];
    for (const [, circuit] of STYLES) {
      expect(pathsFor(lone, { circuit })).toEqual(pathsFor(lone, { circuit, unlaned: true }));
    }
  });
});

describe('the provider reaches the rendered edges', () => {
  function renderFan(withProvider: boolean): string[] {
    useUIStore.setState({ edgeStyle: 'circuit' });
    const wires = fanOut(3, 160);
    const edges = wires.map((w) => (
      <SmartDataEdge
        key={w.id}
        {...({
          id: w.id,
          source: w.source,
          target: w.target,
          sourceX: w.sourceX,
          sourceY: w.sourceY,
          targetX: w.targetX,
          targetY: w.targetY,
          sourcePosition: Position.Right,
          targetPosition: Position.Left,
        } as EdgeProps)}
      />
    ));
    const tree = withProvider ? <EdgeLaneProvider edges={wires}>{edges}</EdgeLaneProvider> : edges;
    const { container } = renderWithFlow(<svg>{tree}</svg>);
    return Array.from(container.querySelectorAll('path.react-flow__edge-path')).map(
      (p) => p.getAttribute('d') ?? '',
    );
  }

  it('separates a rendered fan-out, and does not without the provider', () => {
    const laned = renderFan(true);
    expect(laned).toHaveLength(3);
    expect(worstSharedRun(laned).length).toBeLessThanOrEqual(fanOutStubLimit(3));

    // Same component, same geometry, no lane map: this is what the canvas drew
    // before the provider was wired in.
    const bare = renderFan(false);
    expect(worstSharedRun(bare).length).toBeGreaterThan(200);
  });
});
