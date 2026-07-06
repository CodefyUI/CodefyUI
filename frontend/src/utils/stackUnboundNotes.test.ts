import { describe, it, expect } from 'vitest';
import type { Node } from '@xyflow/react';
import { stackUnboundNotes } from './autoLayout';

function note(id: string, boundTo: string | null = null): Node {
  return { id, type: 'noteNode', position: { x: 0, y: 0 },
    data: { boundToNodeId: boundTo } } as unknown as Node;
}
function comp(id: string): Node {
  return { id, type: 'baseNode', position: { x: 50, y: 60 }, data: {} } as unknown as Node;
}

describe('stackUnboundNotes', () => {
  it('places unbound notes in a deterministic offset column', () => {
    const out = stackUnboundNotes([note('n2'), note('n1'), comp('c')]);
    const n1 = out.find((n) => n.id === 'n1')!;
    const n2 = out.find((n) => n.id === 'n2')!;
    // Sorted by id -> n1 first, n2 below it; same x; distinct y.
    expect(n1.position.x).toBe(n2.position.x);
    expect(n2.position.y).toBeGreaterThan(n1.position.y);
    expect(n1.position.x).toBeLessThan(0); // offset column, off to the side
  });

  it('leaves bound notes and computational nodes untouched', () => {
    const bound = note('b', 'c');
    const out = stackUnboundNotes([bound, comp('c')]);
    expect(out.find((n) => n.id === 'b')!.position).toEqual({ x: 0, y: 0 });
    expect(out.find((n) => n.id === 'c')!.position).toEqual({ x: 50, y: 60 });
  });

  it('is deterministic across calls', () => {
    const a = stackUnboundNotes([note('z'), note('a')]);
    const b = stackUnboundNotes([note('a'), note('z')]);
    expect(a.find((n) => n.id === 'a')!.position)
      .toEqual(b.find((n) => n.id === 'a')!.position);
  });
});
