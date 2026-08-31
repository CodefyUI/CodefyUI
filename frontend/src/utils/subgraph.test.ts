import { describe, it, expect } from 'vitest';
import type { Edge, Node } from '@xyflow/react';

import type { NodeData, SubgraphDefinition } from '../types';
import {
  collapseSelection,
  definitionFromCanvas,
  expandInstance,
  findConvexityBlockers,
  instanceDefinition,
  normalizeSubgraphs,
  pruneStaleBoundaryEdges,
  refreshInstances,
  checkCollapse,
  reachableSubgraphIds,
  subgraphIdOf,
} from './subgraph';

// ── Fixtures ────────────────────────────────────────────────────────────

function node(
  id: string,
  type: string,
  position = { x: 0, y: 0 },
  extra: Partial<NodeData> = {},
): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position,
    data: {
      label: id,
      type,
      params: {},
      definition: {
        node_name: type,
        category: 'x',
        description: '',
        inputs: [
          { name: 'in', data_type: 'TENSOR', description: '', optional: false },
          { name: 'in2', data_type: 'TENSOR', description: '', optional: true },
        ],
        outputs: [
          { name: 'out', data_type: 'TENSOR', description: '', optional: false },
        ],
        params: [],
      },
      ...extra,
    },
  };
}

function edge(id: string, source: string, target: string, opts: Partial<Edge> = {}): Edge {
  return {
    id,
    source,
    target,
    sourceHandle: 'out',
    targetHandle: 'in',
    ...opts,
  };
}

function triggerEdge(id: string, source: string, target: string): Edge {
  return {
    id,
    source,
    target,
    sourceHandle: 'trigger',
    targetHandle: '__trigger',
    type: 'triggerEdge',
    data: { type: 'trigger' },
  };
}

/** start -> a -> b -> c -> sink, with a triggering `a`. */
function chain() {
  const nodes = [
    node('start', 'Start', { x: 0, y: 0 }),
    node('a', 'A', { x: 100, y: 40 }),
    node('b', 'B', { x: 200, y: 60 }),
    node('c', 'C', { x: 300, y: 20 }),
    node('sink', 'S', { x: 400, y: 0 }),
  ];
  const edges = [
    triggerEdge('t', 'start', 'a'),
    edge('e1', 'a', 'b'),
    edge('e2', 'b', 'c'),
    edge('e3', 'c', 'sink'),
  ];
  return { nodes, edges };
}

const resolveNoop = (raw: any[]): Node<NodeData>[] =>
  raw.map((r) => node(r.id, r.type, r.position));

// ── Boundary derivation ─────────────────────────────────────────────────

describe('collapseSelection', () => {
  it('replaces the selection with one instance node', () => {
    const { nodes, edges } = chain();
    const result = collapseSelection(nodes, edges, [], ['b', 'c'], {
      id: 'sg1',
      instanceId: 'inst',
      name: 'Block',
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.nodes.map((n) => n.id).sort()).toEqual(
      ['a', 'inst', 'sink', 'start'],
    );
    expect(subgraphIdOf(result.nodes.find((n) => n.id === 'inst')!.data.type))
      .toBe('sg1');
    expect(result.definition.nodes.map((n) => n.id)).toEqual(['b', 'c']);
    expect(result.definition.edges.map((e) => e.id)).toEqual(['e2']);
  });

  it('derives one boundary port per crossed inner port, with its type', () => {
    const { nodes, edges } = chain();
    const result = collapseSelection(nodes, edges, [], ['b', 'c'], {
      id: 'sg1', instanceId: 'inst',
    });
    if (!result.ok) throw new Error('expected collapse to succeed');
    expect(result.definition.interface.inputs).toEqual([
      { port: 'in', innerNode: 'b', innerPort: 'in', data_type: 'TENSOR' },
    ]);
    expect(result.definition.interface.outputs).toEqual([
      { port: 'out', innerNode: 'c', innerPort: 'out', data_type: 'TENSOR' },
    ]);
    const rewired = result.edges.filter(
      (e) => e.source === 'inst' || e.target === 'inst',
    );
    expect(rewired.map((e) => [e.source, e.sourceHandle, e.target, e.targetHandle]))
      .toEqual([
        ['a', 'out', 'inst', 'in'],
        ['inst', 'out', 'sink', 'in'],
      ]);
  });

  it('gives two edges into the same inner port ONE boundary port', () => {
    // Fan-in is resolved last-edge-wins by the engine; splitting it into two
    // boundary ports would change which edge wins.
    const nodes = [
      node('p', 'P', { x: 0, y: 0 }),
      node('q', 'Q', { x: 0, y: 80 }),
      node('m', 'M', { x: 100, y: 0 }),
      node('n', 'N', { x: 200, y: 0 }),
    ];
    const edges = [
      edge('e1', 'p', 'm'),
      edge('e2', 'q', 'm'),
      edge('e3', 'm', 'n'),
    ];
    const result = collapseSelection(nodes, edges, [], ['m', 'n'], {
      id: 'sg', instanceId: 'inst',
    });
    if (!result.ok) throw new Error('expected collapse to succeed');
    expect(result.definition.interface.inputs).toHaveLength(1);
    const intoInstance = result.edges.filter((e) => e.target === 'inst');
    expect(intoInstance).toHaveLength(2);
    expect(new Set(intoInstance.map((e) => e.targetHandle))).toEqual(
      new Set(['in']),
    );
  });

  it('deduplicates boundary port names when two inner ports share one', () => {
    const nodes = [
      node('p', 'P', { x: 0, y: 0 }),
      node('q', 'Q', { x: 0, y: 80 }),
      node('m', 'M', { x: 100, y: 0 }),
      node('n', 'N', { x: 100, y: 80 }),
    ];
    const edges = [edge('e1', 'p', 'm'), edge('e2', 'q', 'n')];
    const result = collapseSelection(nodes, edges, [], ['m', 'n'], {
      id: 'sg', instanceId: 'inst',
    });
    if (!result.ok) throw new Error('expected collapse to succeed');
    const names = result.definition.interface.inputs.map((p) => p.port);
    expect(names).toEqual(['in', 'in_2']);
    expect(new Set(names).size).toBe(names.length);
  });

  it('records the triggered inner nodes and rewires Start to the instance', () => {
    const nodes = [
      node('start', 'Start'),
      node('a', 'A', { x: 100, y: 0 }),
      node('b', 'B', { x: 100, y: 80 }),
    ];
    const edges = [triggerEdge('t1', 'start', 'a'), triggerEdge('t2', 'start', 'b')];
    const result = collapseSelection(nodes, edges, [], ['a', 'b'], {
      id: 'sg', instanceId: 'inst',
    });
    if (!result.ok) throw new Error('expected collapse to succeed');
    expect(result.definition.interface.triggerTargets).toEqual(['a', 'b']);
    // Two Start edges into the block become ONE edge into the instance; the
    // engine fans it back out to both on expansion.
    const triggers = result.edges.filter((e) => e.target === 'inst');
    expect(triggers).toHaveLength(1);
    expect(triggers[0].targetHandle).toBe('__trigger');
  });

  it('stores inner positions relative to the instance corner', () => {
    const { nodes, edges } = chain();
    const result = collapseSelection(nodes, edges, [], ['b', 'c'], {
      id: 'sg', instanceId: 'inst',
    });
    if (!result.ok) throw new Error('expected collapse to succeed');
    const instance = result.nodes.find((n) => n.id === 'inst')!;
    expect(instance.position).toEqual({ x: 200, y: 20 });
    expect(result.definition.nodes.map((n) => n.position)).toEqual([
      { x: 0, y: 40 },
      { x: 100, y: 0 },
    ]);
  });

  it('refuses a selection of fewer than two nodes', () => {
    const { nodes, edges } = chain();
    const result = collapseSelection(nodes, edges, [], ['b']);
    expect(result).toMatchObject({ ok: false, reason: 'too-few' });
  });

  it('refuses a selection containing Start', () => {
    const { nodes, edges } = chain();
    const result = collapseSelection(nodes, edges, [], ['start', 'a']);
    expect(result).toMatchObject({ ok: false, reason: 'contains-start' });
  });

  it('refuses a selection with a node in the middle left out', () => {
    // a -> b -> c with only {a, c} selected: the instance would feed b and be
    // fed by it, drawing a loop the flattened graph does not have.
    const { nodes, edges } = chain();
    const result = collapseSelection(nodes, edges, [], ['a', 'c']);
    expect(result).toMatchObject({ ok: false, reason: 'not-convex' });
    if (result.ok) return;
    expect(result.blockers).toEqual(['b']);
  });

  it('allows a selection whose in-between node is also selected', () => {
    const { nodes, edges } = chain();
    const result = collapseSelection(nodes, edges, [], ['a', 'b', 'c'], {
      id: 'sg', instanceId: 'inst',
    });
    expect(result.ok).toBe(true);
  });
});

describe('findConvexityBlockers', () => {
  it('ignores a node that is only downstream', () => {
    const { nodes, edges } = chain();
    expect(findConvexityBlockers(new Set(['a', 'b']), nodes, edges)).toEqual([]);
  });

  it('names every node that sits between two selected ones', () => {
    const nodes = ['a', 'm1', 'm2', 'z'].map((id, i) =>
      node(id, 'X', { x: i * 10, y: 0 }),
    );
    const edges = [
      edge('e1', 'a', 'm1'), edge('e2', 'm1', 'm2'), edge('e3', 'm2', 'z'),
    ];
    expect(findConvexityBlockers(new Set(['a', 'z']), nodes, edges)).toEqual(
      ['m1', 'm2'],
    );
  });
});

// ── Expansion ───────────────────────────────────────────────────────────

describe('expandInstance', () => {
  it('restores the original ids, positions and wiring', () => {
    const { nodes, edges } = chain();
    const collapsed = collapseSelection(nodes, edges, [], ['b', 'c'], {
      id: 'sg', instanceId: 'inst',
    });
    if (!collapsed.ok) throw new Error('expected collapse to succeed');

    const expanded = expandInstance(
      collapsed.nodes, collapsed.edges, collapsed.subgraphs, 'inst', resolveNoop,
    );
    expect(expanded.ok).toBe(true);
    expect(expanded.nodes.map((n) => n.id).sort()).toEqual(
      ['a', 'b', 'c', 'sink', 'start'],
    );
    const byId = new Map(expanded.nodes.map((n) => [n.id, n]));
    expect(byId.get('b')!.position).toEqual({ x: 200, y: 60 });
    expect(byId.get('c')!.position).toEqual({ x: 300, y: 20 });

    const wiring = expanded.edges
      .map((e) => `${e.source}.${e.sourceHandle}->${e.target}.${e.targetHandle}`)
      .sort();
    expect(wiring).toEqual([
      'a.out->b.in',
      'b.out->c.in',
      'c.out->sink.in',
      'start.trigger->a.__trigger',
    ]);
    // The definition nobody instantiates any more goes with it.
    expect(expanded.subgraphs).toEqual([]);
  });

  it('fans a trigger edge back out to every recorded target', () => {
    const nodes = [
      node('start', 'Start'),
      node('a', 'A', { x: 100, y: 0 }),
      node('b', 'B', { x: 100, y: 80 }),
    ];
    const edges = [triggerEdge('t1', 'start', 'a'), triggerEdge('t2', 'start', 'b')];
    const collapsed = collapseSelection(nodes, edges, [], ['a', 'b'], {
      id: 'sg', instanceId: 'inst',
    });
    if (!collapsed.ok) throw new Error('expected collapse to succeed');
    const expanded = expandInstance(
      collapsed.nodes, collapsed.edges, collapsed.subgraphs, 'inst', resolveNoop,
    );
    const triggered = expanded.edges
      .filter((e) => e.source === 'start')
      .map((e) => e.target)
      .sort();
    expect(triggered).toEqual(['a', 'b']);
    expect(new Set(expanded.edges.map((e) => e.id)).size)
      .toBe(expanded.edges.length);
  });

  it('prefixes inner ids that a sibling instance already holds', () => {
    const { nodes, edges } = chain();
    const collapsed = collapseSelection(nodes, edges, [], ['b', 'c'], {
      id: 'sg', instanceId: 'inst',
    });
    if (!collapsed.ok) throw new Error('expected collapse to succeed');
    // A second instance of the same definition, expanded first, occupies the
    // definition's own ids.
    const withSecond = {
      nodes: [
        ...collapsed.nodes,
        { ...collapsed.nodes.find((n) => n.id === 'inst')!, id: 'inst2' },
      ],
      edges: collapsed.edges,
    };
    const first = expandInstance(
      withSecond.nodes, withSecond.edges, collapsed.subgraphs, 'inst', resolveNoop,
    );
    const second = expandInstance(
      first.nodes, first.edges, first.subgraphs, 'inst2', resolveNoop,
    );
    expect(second.restoredIds).toEqual(['inst2-b', 'inst2-c']);
    expect(new Set(second.nodes.map((n) => n.id)).size).toBe(second.nodes.length);
  });

  it('leaves the definition alone while another instance still uses it', () => {
    const { nodes, edges } = chain();
    const collapsed = collapseSelection(nodes, edges, [], ['b', 'c'], {
      id: 'sg', instanceId: 'inst',
    });
    if (!collapsed.ok) throw new Error('expected collapse to succeed');
    const twoInstances = [
      ...collapsed.nodes,
      { ...collapsed.nodes.find((n) => n.id === 'inst')!, id: 'inst2' },
    ];
    const expanded = expandInstance(
      twoInstances, collapsed.edges, collapsed.subgraphs, 'inst', resolveNoop,
    );
    expect(expanded.subgraphs.map((s) => s.id)).toEqual(['sg']);
  });
});

// ── Definition <-> canvas ───────────────────────────────────────────────

describe('definitionFromCanvas', () => {
  const base: SubgraphDefinition = {
    id: 'sg',
    name: 'Block',
    description: '',
    nodes: [{ id: 'x', type: 'X', position: { x: 0, y: 0 }, data: { params: {} } }],
    edges: [],
    interface: {
      inputs: [{ port: 'in', innerNode: 'x', innerPort: 'in', data_type: 'TENSOR' }],
      outputs: [{ port: 'out', innerNode: 'x', innerPort: 'out', data_type: 'TENSOR' }],
      triggerTargets: ['x'],
    },
  };

  it('drops boundary ports whose inner node was deleted', () => {
    const next = definitionFromCanvas(base, [node('y', 'Y')], []);
    expect(next.interface.inputs).toEqual([]);
    expect(next.interface.outputs).toEqual([]);
    expect(next.interface.triggerTargets).toEqual([]);
    expect(next.nodes.map((n) => n.id)).toEqual(['y']);
  });

  it('keeps ports whose inner node survived, and re-normalises positions', () => {
    const next = definitionFromCanvas(base, [node('x', 'X', { x: 40, y: 90 })], []);
    expect(next.interface.inputs).toEqual(base.interface.inputs);
    expect(next.nodes[0].position).toEqual({ x: 0, y: 0 });
  });

  it('leaves note nodes out of the definition', () => {
    const note = { ...node('n1', 'note'), type: 'noteNode' };
    const next = definitionFromCanvas(base, [node('x', 'X'), note], []);
    expect(next.nodes.map((n) => n.id)).toEqual(['x']);
  });
});

// ── Inner-node labels (#400) ────────────────────────────────────────────

/**
 * A renamed node keeps its name when it is collapsed into a block.
 *
 * #395 made `data.label` survive a save at the TOP level: the serializer
 * writes it when it differs from the node's type, and the reader restores it
 * (`label: raw.data?.label ?? nodeType`). `serializeInnerNode` never learned
 * the rule, so collapsing a renamed node threw its name away for good -- the
 * original node is gone, and the definition is the only record left of it.
 *
 * The rule is copied rather than loosened. Emitting a label that EQUALS the
 * type would make every existing definition grow a key the reader would have
 * defaulted to anyway; emitting one for a preset or a nested block would
 * write a name the reader ignores, since those two read their label off their
 * definition on every load.
 */
describe('inner nodes keep the name the user gave them', () => {
  const renamed = (id: string, type: string, label: string, x = 0) =>
    node(id, type, { x, y: 0 }, { label });

  const block: SubgraphDefinition = {
    id: 'sg',
    name: 'Block',
    description: '',
    nodes: [{ id: 'x', type: 'X', position: { x: 0, y: 0 }, data: { params: {} } }],
    edges: [],
    interface: { inputs: [], outputs: [], triggerTargets: [] },
  };

  it('collapse writes a renamed member into the definition', () => {
    const nodes = [renamed('b', 'B', 'Encoder'), renamed('c', 'C', 'C', 100)];
    const result = collapseSelection(nodes, [edge('e', 'b', 'c')], [], ['b', 'c'], {
      id: 'sg1', instanceId: 'inst',
    });
    if (!result.ok) throw new Error('expected collapse to succeed');
    const inner = result.definition.nodes as any[];
    expect(inner.find((n) => n.id === 'b').data.label).toBe('Encoder');
    // ...and leaves the member nobody renamed byte-identical to before.
    expect(inner.find((n) => n.id === 'c').data).not.toHaveProperty('label');
  });

  it('carries a rename made INSIDE the block back into the definition', () => {
    const next = definitionFromCanvas(block, [renamed('x', 'X', 'Latent')], []);
    expect((next.nodes[0] as any).data.label).toBe('Latent');
  });

  it('writes no label for a preset or a nested block', () => {
    // Both read their name off their own definition every time they are
    // resolved, so `data.type` never equals the label and an unguarded rule
    // would emit for every one of them -- a key the reader then ignores.
    const preset = renamed('p', 'preset:KeyedChat', 'KeyedChat');
    const nodes = [
      { ...preset, type: 'presetNode', data: { ...preset.data, isPreset: true } },
      renamed('g', 'subgraph:inner', 'Inner Block', 100),
    ];
    const result = collapseSelection(nodes, [edge('e', 'p', 'g')], [], ['p', 'g'], {
      id: 'sg1', instanceId: 'inst',
    });
    if (!result.ok) throw new Error('expected collapse to succeed');
    for (const inner of result.definition.nodes as any[]) {
      expect(inner.data).not.toHaveProperty('label');
    }
  });

  it('writes no label for an inner node that has no type to compare against', () => {
    // Serialization is on the path of every Run and every save, so a node
    // that reached the canvas without a `data.type` must not throw here --
    // and it has no fallback for the reader to restore from either.
    const typeless = node('t', 'T', { x: 0, y: 0 }, {
      label: 'Named', type: undefined as unknown as string,
    });
    const next = definitionFromCanvas(block, [typeless], []);
    expect((next.nodes[0] as any).data).not.toHaveProperty('label');
  });
});

describe('refreshInstances', () => {
  it('rewrites every instance of the definition, and nothing else', () => {
    const definition: SubgraphDefinition = {
      id: 'sg', name: 'Renamed', description: '', nodes: [], edges: [],
      interface: {
        inputs: [{ port: 'z', innerNode: 'x', innerPort: 'in', data_type: 'MODEL' }],
        outputs: [], triggerTargets: [],
      },
    };
    const instance = (id: string) => ({
      ...node(id, 'subgraph:sg'),
      type: 'subgraphNode',
    });
    const other = node('plain', 'Plain');
    const next = refreshInstances([instance('i1'), instance('i2'), other], definition);
    expect(next[0].data.definition!.inputs.map((p) => p.name)).toEqual(['z']);
    expect(next[1].data.definition!.inputs.map((p) => p.name)).toEqual(['z']);
    expect(next[0].data.label).toBe('Renamed');
    expect(next[2]).toBe(other);
  });
});

describe('pruneStaleBoundaryEdges', () => {
  it('drops an edge naming a port the interface no longer exposes', () => {
    const definition: SubgraphDefinition = {
      id: 'sg', name: '', description: '', nodes: [], edges: [],
      interface: { inputs: [], outputs: [], triggerTargets: [] },
    };
    const instance = { ...node('i1', 'subgraph:sg'), type: 'subgraphNode' };
    const edges = [edge('e', 'other', 'i1'), edge('keep', 'other', 'plain')];
    const kept = pruneStaleBoundaryEdges(
      [instance, node('plain', 'P'), node('other', 'O')], edges, [definition],
    );
    expect(kept.map((e) => e.id)).toEqual(['keep']);
  });

  it('never drops a trigger edge', () => {
    const definition: SubgraphDefinition = {
      id: 'sg', name: '', description: '', nodes: [], edges: [],
      interface: { inputs: [], outputs: [], triggerTargets: [] },
    };
    const instance = { ...node('i1', 'subgraph:sg'), type: 'subgraphNode' };
    const edges = [triggerEdge('t', 'start', 'i1')];
    expect(pruneStaleBoundaryEdges([instance], edges, [definition])).toEqual(edges);
  });
});

describe('instanceDefinition', () => {
  it('turns the interface into renderable ports', () => {
    const rendered = instanceDefinition({
      id: 'sg', name: 'Block', description: 'd', nodes: [], edges: [],
      interface: {
        inputs: [{ port: 'a', innerNode: 'x', innerPort: 'in', data_type: 'TENSOR' }],
        outputs: [{ port: 'b', innerNode: 'y', innerPort: 'out', data_type: '' }],
        triggerTargets: [],
      },
    });
    expect(rendered.node_name).toBe('Block');
    expect(rendered.inputs).toEqual([
      { name: 'a', data_type: 'TENSOR', description: 'x.in', optional: false },
    ]);
    // An untyped boundary falls back to ANY so the handle still renders.
    expect(rendered.outputs[0].data_type).toBe('ANY');
  });
});

// ── reachableSubgraphIds (review MINOR 9 / MAJOR 6) ─────────────────────

describe('reachableSubgraphIds', () => {
  function definition(
    id: string,
    innerTypes: string[] = [],
  ): SubgraphDefinition {
    return {
      id,
      name: id,
      description: '',
      nodes: innerTypes.map((type, index) => ({
        id: `${id}-n${index}`,
        type,
        position: { x: 0, y: 0 },
        data: { params: {} },
      })),
      edges: [],
      interface: { inputs: [], outputs: [], triggerTargets: [] },
    };
  }

  it('follows a nested reference the canvas cannot see directly', () => {
    // The canvas holds only an instance of `outer`; `inner` is referenced
    // exclusively from INSIDE outer's definition.
    const subgraphs = [definition('outer', ['subgraph:inner']), definition('inner')];
    const canvas = [node('i1', 'subgraph:outer')];
    expect([...reachableSubgraphIds(canvas, subgraphs)].sort()).toEqual([
      'inner',
      'outer',
    ]);
  });

  it('reports nothing for a definition no canvas node can reach', () => {
    const subgraphs = [definition('used'), definition('orphan')];
    expect([...reachableSubgraphIds([node('i1', 'subgraph:used')], subgraphs)]).toEqual([
      'used',
    ]);
  });

  it('terminates on a definition cycle rather than hanging the editor', () => {
    // Collapse can never build this, but a hand-edited or malicious file can,
    // and the walk runs on every serialize.
    const subgraphs = [
      definition('a', ['subgraph:b']),
      definition('b', ['subgraph:a']),
    ];
    expect([...reachableSubgraphIds([node('i', 'subgraph:a')], subgraphs)].sort()).toEqual(
      ['a', 'b'],
    );
  });

  it('ignores a reference to a definition the graph does not carry', () => {
    const subgraphs = [definition('a', ['subgraph:ghost'])];
    expect([...reachableSubgraphIds([node('i', 'subgraph:a')], subgraphs)]).toEqual(['a']);
  });
});

// ── checkCollapse (review NIT 21) ───────────────────────────────────────

describe('checkCollapse', () => {
  it('is the SAME verdict collapseSelection reaches, without doing the work', () => {
    const { nodes, edges } = chain();
    // a and c with b in the way: the convexity refusal.
    expect(checkCollapse(nodes, edges, ['a', 'c'])).toEqual({
      ok: false,
      reason: 'not-convex',
      blockers: ['b'],
    });
    expect(collapseSelection(nodes, edges, [], ['a', 'c'])).toEqual({
      ok: false,
      reason: 'not-convex',
      blockers: ['b'],
    });
  });

  it('accepts a selection collapse accepts, and refuses each guard', () => {
    const { nodes, edges } = chain();
    expect(checkCollapse(nodes, edges, ['b', 'c'])).toEqual({ ok: true });
    expect(checkCollapse(nodes, edges, ['b'])).toMatchObject({ reason: 'too-few' });
    expect(checkCollapse(nodes, edges, ['start', 'a'])).toMatchObject({
      reason: 'contains-start',
    });
    const withNote: Node<NodeData>[] = [
      ...nodes,
      { ...node('note', 'note'), type: 'noteNode' },
    ];
    expect(checkCollapse(withNote, edges, ['b', 'note'])).toMatchObject({
      reason: 'contains-note',
    });
  });
});

describe('normalizeSubgraphs', () => {
  it('fills a hand-written entry out to the shape every consumer walks', () => {
    // The server's own contract: `id` required, everything else defaulted
    // (`schemas/models.py SubgraphDefinition`). A file it accepts must not
    // be a file the editor throws on.
    expect(normalizeSubgraphs([{ id: 'x' }])).toEqual([
      {
        id: 'x',
        name: '',
        description: '',
        nodes: [],
        edges: [],
        interface: { inputs: [], outputs: [], triggerTargets: [] },
      },
    ]);
  });

  it('replaces only the fields that are the wrong type', () => {
    const [fixed] = normalizeSubgraphs([
      {
        id: 'x',
        name: 'Block',
        description: null,
        nodes: 'not a list',
        edges: [{ id: 'e' }],
        interface: { inputs: [{ port: 'in' }] },
      },
    ]);
    expect(fixed.name).toBe('Block');
    expect(fixed.description).toBe('');
    expect(fixed.nodes).toEqual([]);
    expect(fixed.edges).toEqual([{ id: 'e' }]);
    expect(fixed.interface.inputs).toEqual([{ port: 'in' }]);
    expect(fixed.interface.outputs).toEqual([]);
    expect(fixed.interface.triggerTargets).toEqual([]);
  });

  it('keeps a field the format has not grown yet', () => {
    const [fixed] = normalizeSubgraphs([{ id: 'x', someFutureField: 7 }]);
    expect((fixed as unknown as { someFutureField: number }).someFutureField)
      .toBe(7);
  });

  it('drops what has no usable id, and a list that is not one', () => {
    expect(
      normalizeSubgraphs([null, undefined, 'x', 3, {}, { id: '' }, { id: 'k' }])
        .map((d) => d.id),
    ).toEqual(['k']);
    expect(normalizeSubgraphs(undefined)).toEqual([]);
    expect(normalizeSubgraphs({ id: 'x' })).toEqual([]);
  });

  it('returns the SAME array and objects when nothing needed changing', () => {
    // Not cosmetic: the persistence record cache compares `subgraphs` by
    // identity, and a fresh array here would rebuild every tab's record on
    // every keystroke.
    const definition: SubgraphDefinition = {
      id: 'x',
      name: 'Block',
      description: '',
      nodes: [],
      edges: [],
      interface: { inputs: [], outputs: [], triggerTargets: [] },
    };
    const list = [definition];
    const out = normalizeSubgraphs(list);
    expect(out).toBe(list);
    expect(out[0]).toBe(definition);
  });
});
