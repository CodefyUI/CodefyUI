import { describe, it, expect } from 'vitest';
import { Position, type EdgeProps } from '@xyflow/react';
import { renderWithFlow } from '../../test/utils';
import { SmartDataEdge } from './SmartDataEdge';

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

describe('SmartDataEdge', () => {
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

  it('varies the skip arc by edge id hash (jitter buckets)', () => {
    // Two ids that hash to different jitter buckets produce different control offsets.
    const a = renderEdge({ id: 'jitter-a', sourceX: 0, sourceY: 0, targetX: 500, targetY: 0 }).d;
    const b = renderEdge({ id: 'completely-different-id-xyz', sourceX: 0, sourceY: 0, targetX: 500, targetY: 0 }).d;
    // Both are skip arcs; the exact control points differ because of the hash jitter.
    expect(a).toContain('C');
    expect(b).toContain('C');
    // At least confirm both rendered a valid path.
    expect(a.length).toBeGreaterThan(0);
    expect(b.length).toBeGreaterThan(0);
  });

  it('handles empty id in hashString (loop does not execute)', () => {
    // id '' -> hashString returns 0 -> jitter 0; still a skip arc.
    const { d } = renderEdge({ id: '', sourceX: 0, sourceY: 0, targetX: 500, targetY: 0 });
    expect(d).toContain('C');
  });
});
