import { describe, it, expect, beforeEach } from 'vitest';
import { Position, type EdgeProps } from '@xyflow/react';
import { renderWithFlow } from '../../test/utils';
import { useUIStore } from '../../store/uiStore';
import {
  SmartDataEdge,
  buildRoundedOrthogonalPath,
  buildCircuitSkipPath,
} from './SmartDataEdge';

function makeProps(over: Partial<EdgeProps> = {}): EdgeProps {
  return {
    id: 'e1',
    source: 'a',
    target: 'b',
    sourceX: 0,
    sourceY: 0,
    targetX: 100,
    targetY: 0,
    sourcePosition: Position.Right,
    targetPosition: Position.Left,
    style: { stroke: '#abc' },
    markerEnd: 'url(#m)',
    interactionWidth: 20,
    ...over,
  } as EdgeProps;
}

function renderEdge(over: Partial<EdgeProps> = {}) {
  const { container } = renderWithFlow(
    <svg>
      <SmartDataEdge {...makeProps(over)} />
    </svg>,
  );
  const path = container.querySelector('path.react-flow__edge-path');
  return { path, d: path?.getAttribute('d') ?? '' };
}

describe('SmartDataEdge (curve mode)', () => {
  beforeEach(() => {
    useUIStore.setState({ edgeStyle: 'curve' });
  });

  it('passes through style and markerEnd to BaseEdge (default bezier branch)', () => {
    // Short horizontal hop: not a row transition, not a skip -> bezier.
    const { path, d } = renderEdge({ targetX: 60, targetY: 0 });
    expect(path).toBeTruthy();
    expect(d.startsWith('M')).toBe(true);
    expect(path?.getAttribute('marker-end')).toBe('url(#m)');
    expect(path?.getAttribute('style') ?? '').toContain('stroke: #abc');
  });

  it('uses smoothStep path for a horizontal row transition (|dy| > threshold)', () => {
    // horizontal source, large vertical delta triggers isRowTransition.
    const { d } = renderEdge({ targetX: 100, targetY: 300 });
    // smoothStep path uses L commands (line segments).
    expect(d).toContain('L');
  });

  it('uses smoothStep path for a vertical column transition (|dx| > threshold)', () => {
    // vertical source positions: major axis is dy. Large dx -> isRowTransition.
    const { d } = renderEdge({
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
      targetX: 300,
      targetY: 100,
    });
    expect(d).toContain('L');
  });

  it('builds a horizontal skip arc when major is large and minor is tiny (arcDir = -1 path)', () => {
    // horizontal, |dx|>380, |dy|<80, |dy|<20 -> computeArcDirection returns -1.
    const { d } = renderEdge({ id: 'skip-h', sourceX: 0, sourceY: 0, targetX: 500, targetY: 0 });
    // Custom skip path is a single cubic bezier: "M x,y C ..." with no L.
    expect(d).toMatch(/^M /);
    expect(d).toContain('C');
    expect(d).not.toContain('L');
    // Source pull-out: first control x is sourceX + PULL_OUT (50).
    expect(d).toContain('M 0,0 C 50,');
  });

  it('builds a horizontal skip arc with arcDir from sign(minor) when minor is moderate', () => {
    // |dy| between MINOR_FLAT_EPSILON(20) and MINOR_TOLERANCE(80): arcDir = sign(dy).
    const { d } = renderEdge({ id: 'skip-h2', sourceX: 0, sourceY: 0, targetX: 500, targetY: 40 });
    expect(d).toMatch(/^M /);
    expect(d).toContain('C');
    expect(d).not.toContain('L');
  });

  it('builds a vertical skip arc (horizontal=false branch of buildSkipPath)', () => {
    // vertical positions, |dy|>150, |dx|<80 -> vertical skip.
    const { d } = renderEdge({
      id: 'skip-v',
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
      sourceX: 0,
      sourceY: 0,
      targetX: 0,
      targetY: 300,
    });
    expect(d).toMatch(/^M /);
    expect(d).toContain('C');
    expect(d).not.toContain('L');
    // Vertical pull-out: first control point y is sourceY + PULL_OUT (50),
    // and its x is sourceX + arcDir*arcOffset (negative for arcDir = -1).
    expect(d).toMatch(/^M 0,0 C -?\d+(?:\.\d+)?,50 /);
  });

  it('gives two sibling-less skips the same arc regardless of their ids', () => {
    // The arc used to be nudged by a hash of the id, so identical wiring drew
    // differently. It is geometry-only now; separation comes from lanes instead.
    const a = renderEdge({ id: 'arc-a', sourceX: 0, sourceY: 0, targetX: 500, targetY: 0 }).d;
    const b = renderEdge({ id: 'completely-different-id-xyz', sourceX: 0, sourceY: 0, targetX: 500, targetY: 0 }).d;
    expect(a).toContain('C');
    expect(b).toBe(a);
  });

  it('renders an edge with an empty id', () => {
    // An id that is not in the lane map falls back to SOLO_LANE rather than throwing.
    const { d } = renderEdge({ id: '', sourceX: 0, sourceY: 0, targetX: 500, targetY: 0 });
    expect(d).toContain('C');
  });

  it('bumps stroke width for a selected edge without touching the stroke color', () => {
    const { path } = renderEdge({ targetX: 60, targetY: 0, selected: true });
    const style = path?.getAttribute('style') ?? '';
    expect(style).toContain('stroke: #abc');
    expect(style).toContain('stroke-width: 3');
  });
});

describe('SmartDataEdge (circuit mode)', () => {
  beforeEach(() => {
    useUIStore.setState({ edgeStyle: 'circuit' });
  });

  it('renders the default branch as orthogonal segments (no cubic commands)', () => {
    const { d } = renderEdge({ targetX: 120, targetY: 60 });
    expect(d.startsWith('M')).toBe(true);
    expect(d).not.toContain('C');
    expect(d).toContain('L');
  });

  it('renders a horizontal row transition orthogonally', () => {
    const { d } = renderEdge({ targetX: 100, targetY: 300 });
    expect(d).not.toContain('C');
    expect(d).toContain('L');
  });

  it('renders a vertical column transition orthogonally', () => {
    const { d } = renderEdge({
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
      targetX: 300,
      targetY: 100,
    });
    expect(d).not.toContain('C');
    expect(d).toContain('L');
  });

  it('routes a horizontal skip through an orthogonal detour lane above the row', () => {
    // No siblings, so slot 0: laneOffset = 100 + min(500*0.15, 120) = 175 and
    // arcDir -1 (flat minor) puts the corridor at laneY = (0+0)/2 - 175 = -175.
    const { d } = renderEdge({ id: 'skip-h', sourceX: 0, sourceY: 0, targetX: 500, targetY: 0 });
    expect(d.startsWith('M 0,0')).toBe(true);
    expect(d).not.toContain('C');
    expect(d).toContain('Q');
    expect(d).toContain('-175');
  });

  it('routes a vertical skip through an orthogonal detour lane beside the column', () => {
    // major = 300 -> laneOffset = 100 + min(300*0.15, 120) + 0 (bucket 0) = 145;
    // arcDir -1 -> laneX = -145.
    const { d } = renderEdge({
      id: 'skip-v',
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
      sourceX: 0,
      sourceY: 0,
      targetX: 0,
      targetY: 300,
    });
    expect(d).not.toContain('C');
    expect(d).toContain('Q');
    expect(d).toContain('-145');
  });

  it('offsets the detour to the minor side when minor is moderate', () => {
    // dy = 40 -> arcDir = +1: the lane sits below the source row (positive y).
    const { d } = renderEdge({ id: 'skip-h2', sourceX: 0, sourceY: 0, targetX: 500, targetY: 40 });
    // bucket 0 -> laneOffset = 175; laneY = 20 + 175 = 195.
    expect(d).toContain('195');
    expect(d).not.toContain('C');
  });

  it('routes purely from geometry, not from the edge id', () => {
    // The id used to feed a hash that nudged the detour, which meant the same
    // wiring drawn twice could land on different corridors and two different
    // wires could land on the same one. Lanes replaced it: an edge with no
    // siblings routes by geometry alone, and the id no longer moves anything.
    // Sibling separation is asserted in SmartDataEdge.overlap.test.ts.
    const geometry = { sourceX: 0, sourceY: 0, targetX: 500, targetY: 0 };
    const first = renderEdge({ id: 'lane-0', ...geometry }).d;
    const again = renderEdge({ id: 'lane-0', ...geometry }).d;
    const other = renderEdge({ id: 'lane-1', ...geometry }).d;
    expect(first).toBe(again);
    expect(other).toBe(first);
  });

  it('keeps a plain straight trace below the skip threshold', () => {
    // |dx| = 300 < 380: default branch; horizontally aligned ports need no bend.
    const { d } = renderEdge({ targetX: 300, targetY: 0 });
    expect(d).not.toContain('C');
    expect(d).not.toContain('Q');
  });

  it('bumps stroke width for a selected edge in circuit mode too', () => {
    const { path } = renderEdge({ targetX: 60, targetY: 0, selected: true });
    expect(path?.getAttribute('style') ?? '').toContain('stroke-width: 3');
  });

  it('leaves the incoming style untouched when not selected', () => {
    const { path } = renderEdge({ targetX: 60, targetY: 0 });
    const style = path?.getAttribute('style') ?? '';
    expect(style).toContain('stroke: #abc');
    expect(style).not.toContain('stroke-width');
  });
});

describe('buildRoundedOrthogonalPath', () => {
  it('returns an empty string for no points', () => {
    expect(buildRoundedOrthogonalPath([], 8)).toBe('');
  });

  it('emits only a move command for a single point', () => {
    expect(buildRoundedOrthogonalPath([{ x: 5, y: 7 }], 8)).toBe('M 5,7');
  });

  it('drops consecutive duplicate points', () => {
    const d = buildRoundedOrthogonalPath(
      [
        { x: 0, y: 0 },
        { x: 0, y: 0 },
        { x: 10, y: 0 },
      ],
      8,
    );
    expect(d).toBe('M 0,0 L 10,0');
  });

  it('passes through colinear points with plain line segments', () => {
    const d = buildRoundedOrthogonalPath(
      [
        { x: 0, y: 0 },
        { x: 5, y: 0 },
        { x: 10, y: 0 },
      ],
      8,
    );
    expect(d).toBe('M 0,0 L 5,0 L 10,0');
  });

  it('rounds a corner with a line + quadratic pair', () => {
    const d = buildRoundedOrthogonalPath(
      [
        { x: 0, y: 0 },
        { x: 10, y: 0 },
        { x: 10, y: 10 },
      ],
      4,
    );
    expect(d).toBe('M 0,0 L 6,0 Q 10,0 10,4 L 10,10');
  });

  it('clamps the corner radius to half the shorter adjacent segment', () => {
    const d = buildRoundedOrthogonalPath(
      [
        { x: 0, y: 0 },
        { x: 4, y: 0 },
        { x: 4, y: 40 },
      ],
      8,
    );
    expect(d).toBe('M 0,0 L 2,0 Q 4,0 4,2 L 4,40');
  });
});

describe('buildCircuitSkipPath', () => {
  it('builds the exact horizontal detour with rounded corners', () => {
    const d = buildCircuitSkipPath(0, 0, 500, 0, true, -1, 100);
    expect(d).toBe(
      'M 0,0 L 42,0 Q 50,0 50,-8 L 50,-92 Q 50,-100 58,-100' +
        ' L 442,-100 Q 450,-100 450,-92 L 450,-8 Q 450,0 458,0 L 500,0',
    );
  });

  it('builds a vertical detour offset to the arcDir side', () => {
    const d = buildCircuitSkipPath(0, 0, 0, 300, false, -1, 100);
    expect(d.startsWith('M 0,0')).toBe(true);
    expect(d).not.toContain('C');
    expect(d).toContain('Q');
    // Lane column at x = (0+0)/2 - 100.
    expect(d).toContain('-100');
    expect(d.endsWith('L 0,300')).toBe(true);
  });
});
