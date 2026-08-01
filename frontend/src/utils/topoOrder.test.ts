import { describe, it, expect } from 'vitest';
import type { Edge } from '@xyflow/react';
import { topologicalOrder } from './topoOrder';

function edge(
  source: string,
  target: string,
  extra: Partial<Pick<Edge, 'type' | 'data'>> = {},
): Pick<Edge, 'source' | 'target' | 'type' | 'data'> {
  return { source, target, ...extra };
}

describe('topologicalOrder', () => {
  it('orders a linear chain from source to sink', () => {
    const nodes = [{ id: 'c' }, { id: 'a' }, { id: 'b' }];
    const edges = [edge('a', 'b'), edge('b', 'c')];
    expect(topologicalOrder(nodes, edges)).toEqual(['a', 'b', 'c']);
  });

  it('returns the caller order when there are no edges', () => {
    const nodes = [{ id: 'x' }, { id: 'y' }, { id: 'z' }];
    expect(topologicalOrder(nodes, [])).toEqual(['x', 'y', 'z']);
  });

  it('places a node after every one of its dependencies in a diamond', () => {
    const nodes = [{ id: 'a' }, { id: 'b' }, { id: 'c' }, { id: 'd' }];
    const edges = [edge('a', 'b'), edge('a', 'c'), edge('b', 'd'), edge('c', 'd')];
    const order = topologicalOrder(nodes, edges);
    expect(order[0]).toBe('a');
    expect(order[3]).toBe('d');
    expect(new Set(order)).toEqual(new Set(['a', 'b', 'c', 'd']));
  });

  it('breaks ties by the caller node order', () => {
    const nodes = [{ id: 'r2' }, { id: 'r1' }];
    expect(topologicalOrder(nodes, [])).toEqual(['r2', 'r1']);
  });

  it('ignores trigger edges typed as triggerEdge', () => {
    // Without the skip, `b` would be forced after `a`; a trigger is a marker.
    const nodes = [{ id: 'b' }, { id: 'a' }];
    const edges = [edge('a', 'b', { type: 'triggerEdge' })];
    expect(topologicalOrder(nodes, edges)).toEqual(['b', 'a']);
  });

  it('ignores trigger edges marked via data.type', () => {
    const nodes = [{ id: 'b' }, { id: 'a' }];
    const edges = [edge('a', 'b', { data: { type: 'trigger' } })];
    expect(topologicalOrder(nodes, edges)).toEqual(['b', 'a']);
  });

  it('ignores edges whose source is not a known node', () => {
    const nodes = [{ id: 'a' }, { id: 'b' }];
    const edges = [edge('ghost', 'a'), edge('a', 'b')];
    expect(topologicalOrder(nodes, edges)).toEqual(['a', 'b']);
  });

  it('ignores edges whose target is not a known node', () => {
    const nodes = [{ id: 'a' }];
    const edges = [edge('a', 'ghost')];
    expect(topologicalOrder(nodes, edges)).toEqual(['a']);
  });

  it('still visits every node when the graph contains a cycle', () => {
    const nodes = [{ id: 'a' }, { id: 'b' }, { id: 'c' }];
    // b <-> c is a cycle; a is a clean root.
    const edges = [edge('a', 'b'), edge('b', 'c'), edge('c', 'b')];
    const order = topologicalOrder(nodes, edges);
    expect(order).toHaveLength(3);
    expect(new Set(order)).toEqual(new Set(['a', 'b', 'c']));
    expect(order[0]).toBe('a');
  });

  it('handles an entirely cyclic graph by falling back to caller order', () => {
    const nodes = [{ id: 'a' }, { id: 'b' }];
    const edges = [edge('a', 'b'), edge('b', 'a')];
    expect(topologicalOrder(nodes, edges)).toEqual(['a', 'b']);
  });

  it('returns an empty list for an empty graph', () => {
    expect(topologicalOrder([], [])).toEqual([]);
  });

  it('counts duplicate edges consistently so the target still drains', () => {
    const nodes = [{ id: 'a' }, { id: 'b' }];
    const edges = [edge('a', 'b'), edge('a', 'b')];
    expect(topologicalOrder(nodes, edges)).toEqual(['a', 'b']);
  });
});
