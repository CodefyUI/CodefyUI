import { describe, it, expect } from 'vitest';
import { autoLayout, applyValleyPass, isTriggerEdge, type ValleyEdge } from './autoLayout';
import type { Node, Edge } from '@xyflow/react';

function makeStartNode(id: string, x = 0, y = 0): Node {
  return {
    id,
    position: { x, y },
    data: { id, type: 'Start' },
    type: 'start',
    width: 80,
    height: 40,
  };
}

function makeNode(id: string, x = 0, y = 0): Node {
  return {
    id,
    position: { x, y },
    data: { id, type: 'Dataset' },
    type: 'baseNode',
    width: 200,
    height: 80,
  };
}

function makeEdge(id: string, source: string, target: string, type: 'data' | 'trigger' = 'data'): Edge {
  return { id, source, target, data: { type } };
}

function makeNoteNode(
  id: string,
  x = 0,
  y = 0,
  extra: Record<string, unknown> = {},
): Node {
  return {
    id,
    position: { x, y },
    data: { id, type: 'note', ...extra },
    type: 'noteNode',
    width: 200,
    height: 100,
  };
}

// Node carrying `measured` dimensions (the preferred size source over width/height).
function makeMeasuredNode(id: string, x = 0, y = 0): Node {
  return {
    id,
    position: { x, y },
    data: { id, type: 'Dataset' },
    type: 'baseNode',
    measured: { width: 220, height: 90 },
  };
}

// Node with no width/height/measured at all → layout falls back to NODE_W/NODE_H.
function makeBareNode(id: string, x = 0, y = 0): Node {
  return {
    id,
    position: { x, y },
    data: { id, type: 'Dataset' },
    type: 'baseNode',
  };
}

describe('autoLayout', () => {
  it('linear chain: A→B→C→D produces strictly increasing X at same Y', () => {
    const nodes = [
      makeStartNode('s'),
      makeNode('A'),
      makeNode('B'),
      makeNode('C'),
      makeNode('D'),
    ];
    const edges = [
      makeEdge('et', 's', 'A', 'trigger'),
      makeEdge('e1', 'A', 'B'),
      makeEdge('e2', 'B', 'C'),
      makeEdge('e3', 'C', 'D'),
    ];
    const result = autoLayout(nodes, edges, 'all');
    const sorted = ['A', 'B', 'C', 'D'].map((id) => result.find((n) => n.id === id)!);
    // Strictly increasing X
    expect(sorted[0].position.x).toBeLessThan(sorted[1].position.x);
    expect(sorted[1].position.x).toBeLessThan(sorted[2].position.x);
    expect(sorted[2].position.x).toBeLessThan(sorted[3].position.x);
    // Same Y (within rounding)
    const ys = sorted.map((n) => n.position.y);
    expect(Math.max(...ys) - Math.min(...ys)).toBeLessThan(5);
  });

  it('diamond: A→B,A→C,B→D,C→D — B and C stack vertically at same X', () => {
    const nodes = ['A', 'B', 'C', 'D'].map((id) => makeNode(id));
    const edges = [
      makeEdge('e1', 'A', 'B'),
      makeEdge('e2', 'A', 'C'),
      makeEdge('e3', 'B', 'D'),
      makeEdge('e4', 'C', 'D'),
    ];
    const result = autoLayout(nodes, edges, 'all');
    const B = result.find((n) => n.id === 'B')!;
    const C = result.find((n) => n.id === 'C')!;
    // B and C at the same X
    expect(Math.abs(B.position.x - C.position.x)).toBeLessThan(5);
    // Different Y
    expect(B.position.y).not.toBe(C.position.y);
  });

  it('two disconnected components → distinct Y bands', () => {
    const nodes = [
      makeStartNode('s1'),
      makeNode('A1'),
      makeNode('A2'),
      makeStartNode('s2'),
      makeNode('B1'),
      makeNode('B2'),
    ];
    const edges = [
      makeEdge('et1', 's1', 'A1', 'trigger'),
      makeEdge('e1', 'A1', 'A2'),
      makeEdge('et2', 's2', 'B1', 'trigger'),
      makeEdge('e2', 'B1', 'B2'),
    ];
    const result = autoLayout(nodes, edges, 'all');
    const A1y = result.find((n) => n.id === 'A1')!.position.y;
    const B1y = result.find((n) => n.id === 'B1')!.position.y;
    expect(A1y).not.toBe(B1y);
  });

  it('cycle A→B→C→A does not crash and produces valid coordinates', () => {
    const nodes = ['A', 'B', 'C'].map((id) => makeNode(id));
    const edges = [
      makeEdge('e1', 'A', 'B'),
      makeEdge('e2', 'B', 'C'),
      makeEdge('e3', 'C', 'A'),
    ];
    const result = autoLayout(nodes, edges, 'all');
    for (const n of result) {
      expect(Number.isFinite(n.position.x)).toBe(true);
      expect(Number.isFinite(n.position.y)).toBe(true);
    }
  });

  it('mode=experiments leaves draft components untouched', () => {
    const nodes = [
      makeStartNode('s'),
      makeNode('live1', 100, 100),
      makeNode('live2', 200, 100),
      makeNode('draft1', 500, 500),
      makeNode('draft2', 700, 500),
    ];
    const edges = [
      makeEdge('et', 's', 'live1', 'trigger'),
      makeEdge('e1', 'live1', 'live2'),
      makeEdge('e2', 'draft1', 'draft2'),
    ];
    const result = autoLayout(nodes, edges, 'experiments');
    const draft1 = result.find((n) => n.id === 'draft1')!;
    const draft2 = result.find((n) => n.id === 'draft2')!;
    expect(draft1.position).toEqual({ x: 500, y: 500 });
    expect(draft2.position).toEqual({ x: 700, y: 500 });
  });

  it('mode=selected only moves selected nodes and preserves centroid', () => {
    const nodes = [
      makeNode('A', 100, 100),
      makeNode('B', 200, 100),
      makeNode('C', 300, 100),
      makeNode('untouched', 999, 999),
    ];
    const edges = [
      makeEdge('e1', 'A', 'B'),
      makeEdge('e2', 'B', 'C'),
    ];
    const selected = new Set(['A', 'B', 'C']);
    const result = autoLayout(nodes, edges, 'selected', selected);
    const untouched = result.find((n) => n.id === 'untouched')!;
    expect(untouched.position).toEqual({ x: 999, y: 999 });
    // Centroid of selection should be roughly preserved
    const beforeCentroid = { x: 200, y: 100 }; // (100+200+300)/3, 100
    const movedNodes = result.filter((n) => selected.has(n.id));
    const afterCentroid = {
      x: movedNodes.reduce((s, n) => s + n.position.x, 0) / movedNodes.length,
      y: movedNodes.reduce((s, n) => s + n.position.y, 0) / movedNodes.length,
    };
    expect(Math.abs(afterCentroid.x - beforeCentroid.x)).toBeLessThan(50);
    expect(Math.abs(afterCentroid.y - beforeCentroid.y)).toBeLessThan(50);
  });

  it('returns nodes unchanged when no targets are selected (empty target set)', () => {
    const nodes = [makeNode('A', 10, 20), makeNode('B', 30, 40)];
    const edges = [makeEdge('e1', 'A', 'B')];
    // 'selected' mode with an empty selection → targetIds.size === 0 → early return.
    const result = autoLayout(nodes, edges, 'selected', new Set());
    expect(result).toBe(nodes);
  });

  it('mode=experiments with no entry points anywhere returns nodes unchanged', () => {
    const nodes = [makeNode('A', 1, 1), makeNode('B', 2, 2)];
    const edges = [makeEdge('e1', 'A', 'B')];
    const result = autoLayout(nodes, edges, 'experiments');
    // No Start/entry node → no component qualifies → empty targets → early return.
    expect(result).toBe(nodes);
    expect(result[0].position).toEqual({ x: 1, y: 1 });
  });

  it('excludes note nodes from layout in mode=all and leaves unbound notes put', () => {
    const nodes = [
      makeStartNode('s'),
      makeNode('A'),
      makeNode('B'),
      makeNoteNode('freeNote', 700, 700),
    ];
    const edges = [
      makeEdge('et', 's', 'A', 'trigger'),
      makeEdge('e1', 'A', 'B'),
    ];
    const result = autoLayout(nodes, edges, 'all');
    const note = result.find((n) => n.id === 'freeNote')!;
    // Note is not a layout target and has no binding → unchanged.
    expect(note.position).toEqual({ x: 700, y: 700 });
  });

  it('repositions a bound note to follow its parent using boundOffset', () => {
    const nodes = [
      makeStartNode('s'),
      makeNode('A'),
      makeNode('B'),
      makeNoteNode('boundNote', 0, 0, {
        boundToNodeId: 'A',
        boundOffset: { x: 15, y: -25 },
      }),
    ];
    const edges = [
      makeEdge('et', 's', 'A', 'trigger'),
      makeEdge('e1', 'A', 'B'),
    ];
    const result = autoLayout(nodes, edges, 'all');
    const parentA = result.find((n) => n.id === 'A')!;
    const note = result.find((n) => n.id === 'boundNote')!;
    expect(note.position.x).toBeCloseTo(parentA.position.x + 15);
    expect(note.position.y).toBeCloseTo(parentA.position.y - 25);
  });

  it('leaves a bound note untouched when its parent id is not found', () => {
    const nodes = [
      makeStartNode('s'),
      makeNode('A'),
      makeNoteNode('orphanNote', 50, 60, {
        boundToNodeId: 'DOES_NOT_EXIST',
        boundOffset: { x: 10, y: 10 },
      }),
    ];
    const edges = [makeEdge('et', 's', 'A', 'trigger')];
    const result = autoLayout(nodes, edges, 'all');
    const note = result.find((n) => n.id === 'orphanNote')!;
    // boundToNodeId set but parent missing → note stays at its original position.
    expect(note.position).toEqual({ x: 50, y: 60 });
  });

  it('excludes selected note nodes from mode=selected layout', () => {
    const nodes = [
      makeNode('A', 100, 100),
      makeNode('B', 200, 100),
      makeNoteNode('selNote', 300, 300),
    ];
    const edges = [makeEdge('e1', 'A', 'B')];
    // Note id is included in the selection but must be filtered out.
    const selected = new Set(['A', 'B', 'selNote']);
    const result = autoLayout(nodes, edges, 'selected', selected);
    const note = result.find((n) => n.id === 'selNote')!;
    expect(note.position).toEqual({ x: 300, y: 300 });
  });

  it('mode=selected with undefined selectedIds set treats selection as empty', () => {
    const nodes = [makeNode('A', 5, 5), makeNode('B', 6, 6)];
    const edges = [makeEdge('e1', 'A', 'B')];
    // selectedIds omitted → `selectedIds ?? []` path → empty → unchanged.
    const result = autoLayout(nodes, edges, 'selected');
    expect(result).toBe(nodes);
  });

  it('uses measured dimensions when present', () => {
    const nodes = [makeMeasuredNode('A'), makeMeasuredNode('B')];
    const edges = [makeEdge('e1', 'A', 'B')];
    const result = autoLayout(nodes, edges, 'all');
    const A = result.find((n) => n.id === 'A')!;
    const B = result.find((n) => n.id === 'B')!;
    expect(A.position.x).toBeLessThan(B.position.x);
    expect(Number.isFinite(A.position.x)).toBe(true);
  });

  it('falls back to default node dimensions when none are provided', () => {
    const nodes = [makeBareNode('A'), makeBareNode('B')];
    const edges = [makeEdge('e1', 'A', 'B')];
    const result = autoLayout(nodes, edges, 'all');
    const A = result.find((n) => n.id === 'A')!;
    const B = result.find((n) => n.id === 'B')!;
    expect(A.position.x).toBeLessThan(B.position.x);
  });

  it('uses larger layout spacing tiers for medium and large graphs', () => {
    // >25 and >50 node components select tighter nodesep/ranksep tiers
    // (getLayoutConfig branches). A 60-node chain exercises the >50 tier.
    const N = 60;
    const ids = Array.from({ length: N }, (_, i) => `g${i}`);
    const nodes = ids.map((id) => makeNode(id));
    const edges = ids.slice(1).map((id, i) => makeEdge(`e${i}`, ids[i], id));
    const result = autoLayout(nodes, edges, 'all');
    expect(result).toHaveLength(N);
    for (const n of result) {
      expect(Number.isFinite(n.position.x)).toBe(true);
      expect(Number.isFinite(n.position.y)).toBe(true);
    }
  });

  it('handles a medium graph (>25 nodes) in the mid spacing tier', () => {
    const N = 30;
    const ids = Array.from({ length: N }, (_, i) => `m${i}`);
    const nodes = ids.map((id) => makeNode(id));
    const edges = ids.slice(1).map((id, i) => makeEdge(`e${i}`, ids[i], id));
    const result = autoLayout(nodes, edges, 'all');
    expect(result).toHaveLength(N);
  });

  it('selected mode falls back to default dimensions for bare nodes when centering', () => {
    // Bare nodes (no measured/width/height) force the NODE_W / NODE_H fallbacks
    // in both the original-centroid and new-centroid calculations.
    const nodes = [
      makeBareNode('A', 100, 100),
      makeBareNode('B', 200, 100),
      makeBareNode('untouched', 900, 900),
    ];
    const edges = [makeEdge('e1', 'A', 'B')];
    const result = autoLayout(nodes, edges, 'selected', new Set(['A', 'B']));
    expect(result.find((n) => n.id === 'untouched')!.position).toEqual({ x: 900, y: 900 });
    const A = result.find((n) => n.id === 'A')!;
    const B = result.find((n) => n.id === 'B')!;
    expect(A.position.x).toBeLessThan(B.position.x);
    expect(Number.isFinite(A.position.y)).toBe(true);
  });

  it('orders entry-pointed components before draft components in swim lanes', () => {
    // A draft component (no Start) and a live one (with Start). The live lane
    // must be placed above (smaller Y) the draft lane after sorting.
    const nodes = [
      makeStartNode('s'),
      makeNode('live'),
      makeNode('draftA'),
      makeNode('draftB'),
    ];
    const edges = [
      makeEdge('et', 's', 'live', 'trigger'),
      makeEdge('e1', 'draftA', 'draftB'),
    ];
    const result = autoLayout(nodes, edges, 'all');
    const liveY = result.find((n) => n.id === 'live')!.position.y;
    const draftAY = result.find((n) => n.id === 'draftA')!.position.y;
    expect(liveY).toBeLessThan(draftAY);
  });

  it('still ranks the entry-pointed lane first when the draft appears first in input order', () => {
    // The draft component precedes the live one in the node array, so the swim-lane
    // comparator is invoked with the draft as its left argument — exercising the
    // `a.hasEntryPoint ? -1 : 1` false arm. The live lane must still sort above.
    const nodes = [
      makeNode('draftA'),
      makeNode('draftB'),
      makeStartNode('s'),
      makeNode('live'),
    ];
    const edges = [
      makeEdge('e1', 'draftA', 'draftB'),
      makeEdge('et', 's', 'live', 'trigger'),
    ];
    const result = autoLayout(nodes, edges, 'all');
    const liveY = result.find((n) => n.id === 'live')!.position.y;
    const draftAY = result.find((n) => n.id === 'draftA')!.position.y;
    expect(liveY).toBeLessThan(draftAY);
  });
});

describe('applyValleyPass (unit)', () => {
  const W = 200;
  const H = 80;
  const UNIT = 280;
  const STEP = H + 60; // max node height in set + default gap

  type Pos = { x: number; y: number; width: number; height: number };
  // Hand-built layered positions: n0..n{k-1} left-to-right, all at y=0.
  function chainPositions(k: number): Map<string, Pos> {
    const m = new Map<string, Pos>();
    for (let i = 0; i < k; i++) m.set(`n${i}`, { x: i * UNIT, y: 0, width: W, height: H });
    return m;
  }
  function chainEdges(k: number, skips: Array<[number, number]> = []): ValleyEdge[] {
    const edges: ValleyEdge[] = [];
    for (let i = 0; i < k - 1; i++) edges.push({ source: `n${i}`, target: `n${i + 1}` });
    for (const [a, b] of skips) edges.push({ source: `n${a}`, target: `n${b}` });
    return edges;
  }
  const yOf = (m: Map<string, Pos>, id: string) => m.get(id)!.y;

  it('no skips -> returns the input unchanged (identity)', () => {
    const positions = chainPositions(10);
    const result = applyValleyPass(positions, chainEdges(10), { axis: 'y' });
    expect(result).toBe(positions);
  });

  it('UNet-style nested skips: exact staircase, level skip endpoints', () => {
    // outer n1->n13 covers ranks 2..12; inner n4->n10 covers ranks 5..9.
    const result = applyValleyPass(chainPositions(18), chainEdges(18, [[1, 13], [4, 10]]), {
      axis: 'y',
    });
    expect(yOf(result, 'n0')).toBe(0);
    expect(yOf(result, 'n1')).toBe(0);
    expect(yOf(result, 'n13')).toBe(0); // outer endpoints level
    expect(yOf(result, 'n2')).toBe(STEP); // under outer only
    expect(yOf(result, 'n4')).toBe(STEP); // inner endpoints level...
    expect(yOf(result, 'n10')).toBe(STEP);
    expect(yOf(result, 'n7')).toBe(2 * STEP); // under both
    expect(yOf(result, 'n15')).toBe(0); // tail back to top
  });

  it('sequential residuals (ResNet-style) dip independently to equal depth', () => {
    const result = applyValleyPass(chainPositions(14), chainEdges(14, [[1, 5], [6, 10]]), {
      axis: 'y',
    });
    expect(yOf(result, 'n3')).toBe(STEP); // inside block 1
    expect(yOf(result, 'n8')).toBe(STEP); // inside block 2
    expect(yOf(result, 'n5')).toBe(0); // between blocks
    expect(yOf(result, 'n0')).toBe(0);
    expect(yOf(result, 'n13')).toBe(0);
  });

  it('short span (< 3) does not dip', () => {
    const positions = chainPositions(8);
    const result = applyValleyPass(positions, chainEdges(8, [[2, 4]]), { axis: 'y' });
    expect(result).toBe(positions);
  });

  it('skips whose endpoint is off the spine are ignored', () => {
    // t sits beside rank 2 and feeds n6/n10: long spans, but t is off-spine.
    const positions = chainPositions(12);
    positions.set('t', { x: 2 * UNIT, y: 300, width: W, height: H });
    const edges = [
      ...chainEdges(12),
      { source: 'n1', target: 't' },
      { source: 't', target: 'n6' },
      { source: 't', target: 'n10' },
    ];
    const result = applyValleyPass(positions, edges, { axis: 'y' });
    expect(result).toBe(positions);
  });

  it('skipPredicate excludes trigger edges', () => {
    const positions = chainPositions(10);
    const edges: ValleyEdge[] = [...chainEdges(10), { source: 'n1', target: 'n8', type: 'triggerEdge' }];
    const result = applyValleyPass(positions, edges, {
      axis: 'y',
      skipPredicate: (e) => !isTriggerEdge(e),
    });
    expect(result).toBe(positions);
  });

  it('axis "x" transposes the shift for TB layouts', () => {
    const m = new Map<string, Pos>();
    for (let i = 0; i < 10; i++) m.set(`n${i}`, { x: 0, y: i * 140, width: 260, height: H });
    const result = applyValleyPass(m, chainEdges(10, [[1, 7]]), { axis: 'x' });
    const xStep = 260 + 60; // max width + gap
    expect(result.get('n0')!.x).toBe(0);
    expect(result.get('n1')!.x).toBe(0);
    expect(result.get('n7')!.x).toBe(0);
    expect(result.get('n4')!.x).toBe(xStep);
    expect(result.get('n4')!.y).toBe(4 * 140); // flow axis untouched
  });

  it('cycle edges (backward in rank) are ignored safely', () => {
    const positions = chainPositions(8);
    const edges = [...chainEdges(8), { source: 'n6', target: 'n1' }]; // back edge
    const result = applyValleyPass(positions, edges, { axis: 'y' });
    expect(result).toBe(positions);
  });
});

describe('isTriggerEdge', () => {
  it('recognizes all three trigger edge shapes', () => {
    expect(isTriggerEdge({ source: 'a', target: 'b', type: 'triggerEdge' })).toBe(true);
    expect(isTriggerEdge({ source: 'a', target: 'b', data: { type: 'trigger' } })).toBe(true);
    expect(isTriggerEdge({ source: 'a', target: 'b', sourceHandle: 'trigger' })).toBe(true);
    expect(isTriggerEdge({ source: 'a', target: 'b' })).toBe(false);
  });
});

describe('valley layout through autoLayout (integration)', () => {
  // Chain helper: start -> n0 -> n1 -> ... -> n{k-1}, plus extra data skips.
  function chainWithSkips(k: number, skips: Array<[number, number]>) {
    const nodes = [makeStartNode('s'), ...Array.from({ length: k }, (_, i) => makeNode(`n${i}`))];
    const edges = [
      makeEdge('et', 's', 'n0', 'trigger'),
      ...Array.from({ length: k - 1 }, (_, i) => makeEdge(`e${i}`, `n${i}`, `n${i + 1}`)),
      ...skips.map(([a, b], i) => makeEdge(`sk${i}`, `n${a}`, `n${b}`)),
    ];
    return { nodes, edges };
  }
  const posOf = (result: Node[], id: string) => result.find((n) => n.id === id)!.position;

  it('long plain chain stays on ONE row — wrapping is gone', () => {
    const { nodes, edges } = chainWithSkips(14, []);
    const result = autoLayout(nodes, edges, 'all');
    const ys = new Set(
      Array.from({ length: 14 }, (_, i) => Math.round(posOf(result, `n${i}`).y / 10)),
    );
    expect(ys.size).toBe(1);
    const xs = Array.from({ length: 14 }, (_, i) => posOf(result, `n${i}`).x);
    for (let i = 1; i < xs.length; i++) expect(xs[i]).toBeGreaterThan(xs[i - 1]);
  });

  it('nested skips produce a valley: covered trunk sinks by full steps, x stays monotone', () => {
    const { nodes, edges } = chainWithSkips(18, [[1, 13], [4, 10]]);
    const result = autoLayout(nodes, edges, 'all');
    // STEP = max node height (200x80 base nodes) + 60 = 140; dagre lane jitter
    // from skip-edge dummy nodes is < 130, so a full step is distinguishable.
    expect(posOf(result, 'n7').y - posOf(result, 'n1').y).toBeGreaterThan(140);
    expect(posOf(result, 'n7').y - posOf(result, 'n2').y).toBeGreaterThan(70);
    // tail comes back up to the outer level
    expect(Math.abs(posOf(result, 'n16').y - posOf(result, 'n0').y)).toBeLessThan(70);
    // no wrapping: trunk x strictly increases (no carriage-return rows)
    const xs = Array.from({ length: 18 }, (_, i) => posOf(result, `n${i}`).x);
    for (let i = 1; i < xs.length; i++) expect(xs[i]).toBeGreaterThan(xs[i - 1]);
  });

  it('valley output is deterministic across runs', () => {
    const { nodes, edges } = chainWithSkips(18, [[1, 13], [4, 10]]);
    const a = autoLayout(nodes, edges, 'all').map((n) => ({ id: n.id, ...n.position }));
    const b = autoLayout(nodes, edges, 'all').map((n) => ({ id: n.id, ...n.position }));
    expect(a).toEqual(b);
  });
});
