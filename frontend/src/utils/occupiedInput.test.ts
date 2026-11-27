import { describe, expect, it } from 'vitest';
import type { Edge } from '@xyflow/react';
import { occupantsOf, reconnectedEdges, withoutOccupants } from './occupiedInput';

const wire = (id: string, over: Partial<Edge> = {}): Edge => ({
  id,
  source: 'a',
  target: 'b',
  sourceHandle: 'out',
  targetHandle: 'x',
  ...over,
});

describe('occupantsOf (#562)', () => {
  it('names the wire already feeding the input', () => {
    const edges = [wire('e1'), wire('e2', { targetHandle: 'y' }), wire('e3', { target: 'c' })];
    expect(occupantsOf(edges, { source: 'z', target: 'b', sourceHandle: 'out', targetHandle: 'x' }))
      .toEqual([edges[0]]);
  });

  it('names every wire a graph saved before the rule left on one input', () => {
    const edges = [wire('e1'), wire('e2', { source: 'q' })];
    expect(occupantsOf(edges, { source: 'z', target: 'b', targetHandle: 'x' }).map((e) => e.id))
      .toEqual(['e1', 'e2']);
  });

  it('leaves out the wire being moved', () => {
    const edges = [wire('e1')];
    expect(occupantsOf(edges, { source: 'z', target: 'b', targetHandle: 'x' }, 'e1')).toEqual([]);
  });

  it('treats a saved wire with no target handle as on no input', () => {
    const edges = [wire('e1', { targetHandle: undefined })];
    expect(occupantsOf(edges, { source: 'z', target: 'b', targetHandle: 'x' })).toEqual([]);
  });

  it.each([
    ['no target', { source: 'z', target: null, targetHandle: 'x' }],
    ['no target handle', { source: 'z', target: 'b', targetHandle: null }],
    ['an empty target handle', { source: 'z', target: 'b' }],
    ['the trigger handle', { source: 's', target: 'b', targetHandle: '__trigger' }],
    ['a trigger source', { source: 's', target: 'b', sourceHandle: 'trigger', targetHandle: 'x' }],
  ])('finds nothing for %s', (_, ends) => {
    const edges = [
      wire('e1'),
      wire('t1', { source: 's', sourceHandle: 'trigger', targetHandle: '__trigger', type: 'triggerEdge' }),
    ];
    expect(occupantsOf(edges, ends)).toEqual([]);
  });

  it('never names a trigger wire, whatever handle it was saved on', () => {
    const edges = [wire('t1', { type: 'triggerEdge' })];
    expect(occupantsOf(edges, { source: 'z', target: 'b', targetHandle: 'x' })).toEqual([]);
  });
});

describe('withoutOccupants', () => {
  it('drops the occupants and keeps the rest in order', () => {
    const edges = [wire('e1'), wire('e2', { target: 'c' }), wire('e3', { source: 'q' })];
    expect(withoutOccupants(edges, { source: 'z', target: 'b', targetHandle: 'x' }).map((e) => e.id))
      .toEqual(['e2']);
  });

  it('returns a copy when nothing is occupied', () => {
    const edges = [wire('e1')];
    const out = withoutOccupants(edges, { source: 'z', target: 'c', targetHandle: 'x' });
    expect(out).toEqual(edges);
    expect(out).not.toBe(edges);
  });
});

describe('reconnectedEdges', () => {
  const wire = (id: string, source: string, target: string, targetHandle = 'x'): Edge =>
    ({ id, source, target, sourceHandle: 'out', targetHandle, type: 'dataEdge' }) as Edge;

  it('keeps the moved wire where it was in the list', () => {
    const edges = [wire('a', 'n1', 'n2'), wire('b', 'n1', 'n3'), wire('c', 'n2', 'n3', 'y')];
    const out = reconnectedEdges(edges, edges[0], {
      source: 'n1', target: 'n4', sourceHandle: 'out', targetHandle: 'x',
    });
    expect(out.map((e) => e.id)).toEqual(['a', 'b', 'c']);
    expect(out[0].target).toBe('n4');
    expect(out[1]).toBe(edges[1]);
  });

  it('drops the wire already feeding the input it lands on, in the same step', () => {
    const edges = [wire('a', 'n1', 'n2'), wire('b', 'n5', 'n3'), wire('c', 'n2', 'n4')];
    const out = reconnectedEdges(edges, edges[0], {
      source: 'n1', target: 'n3', sourceHandle: 'out', targetHandle: 'x',
    });
    expect(out.map((e) => e.id)).toEqual(['a', 'c']);
    expect(out[0].target).toBe('n3');
  });
});
