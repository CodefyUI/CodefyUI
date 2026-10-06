/**
 * Collapse and expand inside an open block keep the block's boundary.
 *
 * An open block is wired to the level above through its definition's
 * interface, which names inner nodes by id; no edge on the block's own canvas
 * draws that boundary. Collapsing the node an interface entry names into a
 * smaller block used to leave the new block without that port, and leaving the
 * outer block then dropped the outer port and the wire on it at the level
 * above -- Run reported the downstream input as unconnected.
 *
 * `runWiring` is the check that matters: what the engine wires together once
 * every block is inlined (`expand_subgraphs`). The same wiring before and after
 * an edit means the same run. It is the stricter of the two: the engine then
 * also runs every node of a block once one of them runs.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { Edge, Node } from '@xyflow/react';

import { useTabStore } from './tabStore';
import { useNodeDefStore } from './nodeDefStore';
import type { NodeData, NodeDefinition, SubgraphPort } from '../types';
import { subgraphIdOf, triggerStarts, type CollapseSuccess } from '../utils/subgraph';

vi.mock('./tabPersistence', () => ({
  readSnapshot: vi.fn(async () => null),
  writeSnapshot: vi.fn(async () => {}),
}));

const store = () => useTabStore.getState();
const tab = () => useTabStore.getState().getActiveTab();

function def(name: string, inputs: string[] = ['in']): NodeDefinition {
  return {
    node_name: name,
    category: 'x',
    description: '',
    inputs: inputs.map((input) => ({
      name: input, data_type: 'TENSOR', description: '', optional: false,
    })),
    outputs: [
      { name: 'out', data_type: 'TENSOR', description: '', optional: false },
    ],
    params: [],
  };
}

/** `J` joins two inputs; every other type has one input and one output. */
const DEFS: Record<string, NodeDefinition> = {
  A: def('A'),
  M: def('M'),
  J: def('J', ['a', 'b']),
  S: def('S'),
};

function node(id: string, type: string, x = 0): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position: { x, y: 0 },
    data: { label: id, type, params: {}, definition: DEFS[type], executionStatus: 'idle' },
  };
}

const start: Node<NodeData> = { ...node('start', 'Start'), type: 'start' };

function wire(id: string, source: string, target: string, targetHandle = 'in'): Edge {
  return { id, source, target, sourceHandle: 'out', targetHandle };
}

function trigger(id: string, source: string, target: string): Edge {
  return {
    id, source, target, sourceHandle: 'trigger', targetHandle: '__trigger',
    type: 'triggerEdge', data: { type: 'trigger' },
  };
}

/** Puts a graph on the canvas and answers how it runs. */
function seed(nodes: Node<NodeData>[], edges: Edge[]): string[] {
  store().setNodes(nodes);
  store().setEdges(edges);
  return runWiring();
}

/** Collapses the given nodes of the canvas on screen; answers the new card's id. */
function collapse(name: string, ...ids: string[]): string {
  store().setNodes(tab().nodes.map((n) => ({ ...n, selected: ids.includes(n.id) })));
  const result = store().collapseSelectionToSubgraph(name);
  expect(result.ok).toBe(true);
  return (result as CollapseSuccess).instanceId;
}

function definitionNamed(name: string) {
  const found = tab().subgraphs.find((d) => d.name === name);
  if (!found) throw new Error(`no block named ${name}`);
  return found;
}

/** `[port, innerNode, innerPort]` of each port, for compact assertions. */
function ports(list: SubgraphPort[]): string[][] {
  return list.map((p) => [p.port, p.innerNode, p.innerPort]);
}

/**
 * What a run wires together: the saved graph with every block inlined by the
 * rules the engine expands one with (`expand_subgraphs`,
 * backend/app/core/graph_engine.py), each edge written between the nodes that
 * actually run, named by their own ids. A port the interface lost shows up as
 * a missing edge; an edge naming a port the block does not have throws, as the
 * engine refuses it.
 */
function runWiring(): string[] {
  const graph = store().getSerializedGraph();
  const byId = new Map(graph.subgraphs.map((d) => [d.id, d]));
  let nodes: { id: string; type: string }[] = graph.nodes.map((n) => ({
    id: String(n.id), type: String(n.type),
  }));
  let edges: any[] = graph.edges.map((e) => ({ ...e }));
  for (;;) {
    const card = nodes.find((n) => subgraphIdOf(n.type) !== null);
    if (!card) break;
    const definition = byId.get(subgraphIdOf(card.type)!)!;
    const at = (id: string) => `${card.id}/${id}`;
    const innerIds = definition.nodes.map((n) => String(n.id));
    const fed = new Set(definition.edges.map((e) => String(e.target)));
    const declared = definition.interface.triggerTargets;
    for (const id of declared) {
      if (!innerIds.includes(id)) throw new Error(`${definition.name}: trigger target ${id} is not inside`);
    }
    const started = declared.length ? declared : innerIds.filter((id) => !fed.has(id));
    const port = (list: SubgraphPort[], name: string) => {
      const found = list.find((p) => p.port === name);
      if (!found) throw new Error(`${definition.name} has no port '${name}'`);
      return found;
    };
    edges = edges.flatMap((e) => {
      if (e.type === 'trigger' && e.target === card.id) {
        return started.map((id) => ({ ...e, target: at(id) }));
      }
      let next = { ...e };
      if (e.target === card.id) {
        const p = port(definition.interface.inputs, e.targetHandle);
        next = { ...next, target: at(p.innerNode), targetHandle: p.innerPort };
      }
      if (e.source === card.id) {
        const p = port(definition.interface.outputs, e.sourceHandle);
        next = { ...next, source: at(p.innerNode), sourceHandle: p.innerPort };
      }
      return [next];
    });
    edges.push(...definition.edges.map((e) => ({
      ...e, source: at(String(e.source)), target: at(String(e.target)),
    })));
    nodes = [
      ...nodes.filter((n) => n !== card),
      ...definition.nodes.map((n) => ({ id: at(String(n.id)), type: String(n.type) })),
    ];
  }
  const own = (id: string) => id.split('/').pop();
  return edges
    .map((e) => `${own(e.source)}.${e.sourceHandle} -> ${own(e.target)}.${e.targetHandle}`
      + (e.type === 'trigger' ? ' (trigger)' : ''))
    .sort();
}

/**
 * start -> src -> m1 -> m2 -> m3 -> sink, with m1..m3 in the block "Outer":
 * Outer's input feeds m1, its output is m3's.
 */
function seedOuter() {
  const wiring = seed(
    [start, node('src', 'A'), node('m1', 'M'), node('m2', 'M'), node('m3', 'M'), node('sink', 'S')],
    [
      trigger('t', 'start', 'src'),
      wire('e0', 'src', 'm1'),
      wire('e1', 'm1', 'm2'),
      wire('e2', 'm2', 'm3'),
      wire('e3', 'm3', 'sink'),
    ],
  );
  const outer = collapse('Outer', 'm1', 'm2', 'm3');
  expect(runWiring()).toEqual(wiring);
  return { outer, wiring };
}

/**
 * start -> src -> m1 -> [Deep: m2 -> m3 -> m4] -> sink, with m1 and the Deep
 * card in the block "Outer": Outer's output is the Deep card's.
 */
function seedNested() {
  const wiring = seed(
    [
      start, node('src', 'A'), node('m1', 'M'), node('m2', 'M'), node('m3', 'M'),
      node('m4', 'M'), node('sink', 'S'),
    ],
    [
      trigger('t', 'start', 'src'),
      wire('e0', 'src', 'm1'),
      wire('e1', 'm1', 'm2'),
      wire('e2', 'm2', 'm3'),
      wire('e3', 'm3', 'm4'),
      wire('e4', 'm4', 'sink'),
    ],
  );
  const deep = collapse('Deep', 'm2', 'm3', 'm4');
  const outer = collapse('Outer', 'm1', deep);
  expect(ports(definitionNamed('Outer').interface.outputs)).toEqual([['out', deep, 'out']]);
  expect(runWiring()).toEqual(wiring);
  return { outer, deep, wiring };
}

/**
 * start triggers m1 directly; m1 -> m2 -> j.a and src -> r -> j.b; j -> sink.
 * m1, m2, r and j are the block "Outer", whose trigger starts m1 only -- r is a
 * root of Outer that the trigger does not reach.
 */
function seedTriggered() {
  const wiring = seed(
    [
      start, node('src', 'A'), node('m1', 'M'), node('m2', 'M'), node('r', 'M'),
      node('j', 'J'), node('sink', 'S'),
    ],
    [
      trigger('t', 'start', 'm1'),
      trigger('t2', 'start', 'src'),
      wire('e1', 'm1', 'm2'),
      wire('e2', 'm2', 'j', 'a'),
      wire('e3', 'src', 'r'),
      wire('e4', 'r', 'j', 'b'),
      wire('e5', 'j', 'sink'),
    ],
  );
  const outer = collapse('Outer', 'm1', 'm2', 'r', 'j');
  expect(definitionNamed('Outer').interface.triggerTargets).toEqual(['m1']);
  expect(runWiring()).toEqual(wiring);
  return { outer, wiring };
}

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
  store().addTab('test');
  useNodeDefStore.setState({
    definitions: Object.values(DEFS),
    presets: [],
    categorized: {},
    loading: false,
    error: null,
  } as never);
});

describe('collapsing inside an open block keeps the block boundary', () => {
  it('keeps the output the block hands to the level above', () => {
    const { outer, wiring } = seedOuter();
    store().enterSubgraph(outer);

    const inner = collapse('Inner', 'm2', 'm3');

    // The new block passes m3's result out ...
    expect(ports(definitionNamed('Inner').interface.outputs)).toEqual([['out', 'm3', 'out']]);
    const card = tab().nodes.find((n) => n.id === inner)!;
    expect(card.data.definition?.outputs.map((p) => p.name)).toEqual(['out']);
    // ... and the open block's output now comes out of the new block.
    expect(ports(definitionNamed('Outer').interface.outputs)).toEqual([['out', inner, 'out']]);
    // Saved or run from in here, the graph still runs as it did.
    expect(runWiring()).toEqual(wiring);

    store().exitSubgraph();
    expect(definitionNamed('Outer').interface.outputs).toHaveLength(1);
    expect(tab().edges.some((e) => e.source === outer && e.target === 'sink')).toBe(true);
    expect(runWiring()).toEqual(wiring);
  });

  it('keeps the input the level above feeds into the block', () => {
    const { outer, wiring } = seedOuter();
    store().enterSubgraph(outer);

    const inner = collapse('Inner', 'm1', 'm2');

    expect(ports(definitionNamed('Inner').interface.inputs)).toEqual([['in', 'm1', 'in']]);
    expect(ports(definitionNamed('Outer').interface.inputs)).toEqual([['in', inner, 'in']]);

    store().exitSubgraph();
    expect(tab().edges.some((e) => e.source === 'src' && e.target === outer)).toBe(true);
    expect(runWiring()).toEqual(wiring);
  });

  it('keeps both when the selection holds the input and the output', () => {
    const { outer, wiring } = seedOuter();
    store().enterSubgraph(outer);

    const inner = collapse('Inner', 'm1', 'm2', 'm3');

    const inside = definitionNamed('Inner').interface;
    expect(ports(inside.inputs)).toEqual([['in', 'm1', 'in']]);
    expect(ports(inside.outputs)).toEqual([['out', 'm3', 'out']]);
    expect(inside.inputs[0].data_type).toBe('TENSOR');
    expect(inside.outputs[0].data_type).toBe('TENSOR');
    const around = definitionNamed('Outer').interface;
    expect(ports(around.inputs)).toEqual([['in', inner, 'in']]);
    expect(ports(around.outputs)).toEqual([['out', inner, 'out']]);

    store().exitSubgraph();
    expect(runWiring()).toEqual(wiring);
  });

  it('shares the port collapse made for an inner port that also leaves the selection', () => {
    const wiring = seed(
      [
        start, node('src', 'A'), node('m1', 'M'), node('m2', 'M'), node('m3', 'M'),
        node('m4', 'M'), node('sink', 'S'), node('sink2', 'S'),
      ],
      [
        trigger('t', 'start', 'src'),
        wire('e0', 'src', 'm1'),
        wire('e1', 'm1', 'm2'),
        wire('e2', 'm2', 'm3'),
        wire('e3', 'm3', 'sink'),
        wire('e4', 'm3', 'm4'),
        wire('e5', 'm4', 'sink2'),
      ],
    );
    const outer = collapse('Outer', 'm1', 'm2', 'm3', 'm4');
    expect(ports(definitionNamed('Outer').interface.outputs)).toEqual([
      ['out', 'm3', 'out'],
      ['out_2', 'm4', 'out'],
    ]);
    store().enterSubgraph(outer);

    // m3 feeds m4, outside the selection, AND the open block's output.
    const inner = collapse('Inner', 'm2', 'm3');

    expect(ports(definitionNamed('Inner').interface.outputs)).toEqual([['out', 'm3', 'out']]);
    expect(ports(definitionNamed('Outer').interface.outputs)).toEqual([
      ['out', inner, 'out'],
      ['out_2', 'm4', 'out'],
    ]);

    store().exitSubgraph();
    expect(runWiring()).toEqual(wiring);
  });

  it('moves a trigger target into the new block, and its card takes the place', () => {
    const { outer, wiring } = seedTriggered();
    store().enterSubgraph(outer);

    const inner = collapse('Inner', 'm1', 'm2');

    expect(definitionNamed('Inner').interface.triggerTargets).toEqual(['m1']);
    expect(definitionNamed('Outer').interface.triggerTargets).toEqual([inner]);

    store().exitSubgraph();
    // Without the move, Outer would fall back to starting every root -- r too.
    expect(runWiring()).toEqual(wiring);
  });

  it('works two blocks deep', () => {
    const { outer, deep, wiring } = seedNested();
    store().enterSubgraph(outer);
    store().enterSubgraph(deep);

    const inner = collapse('Inner', 'm3', 'm4');

    expect(ports(definitionNamed('Inner').interface.outputs)).toEqual([['out', 'm4', 'out']]);
    expect(ports(definitionNamed('Deep').interface.outputs)).toEqual([['out', inner, 'out']]);
    expect(runWiring()).toEqual(wiring);

    store().exitSubgraph();
    // The level in between still hands Deep's output up.
    expect(ports(definitionNamed('Deep').interface.outputs)).toEqual([['out', inner, 'out']]);
    expect(ports(definitionNamed('Outer').interface.outputs)).toEqual([['out', deep, 'out']]);
    expect(tab().nodes.find((n) => n.id === deep)?.data.definition?.outputs).toHaveLength(1);

    store().exitSubgraph();
    expect(tab().edges.some((e) => e.source === outer && e.target === 'sink')).toBe(true);
    expect(runWiring()).toEqual(wiring);
  });

  it('keeps a port that names a block card inside the block', () => {
    const { outer, deep, wiring } = seedNested();
    store().enterSubgraph(outer);

    const wrap = collapse('Wrap', 'm1', deep);

    // Outer's output was the Deep card's; Wrap now passes it through.
    expect(ports(definitionNamed('Wrap').interface.outputs)).toEqual([['out', deep, 'out']]);
    expect(ports(definitionNamed('Outer').interface.outputs)).toEqual([['out', wrap, 'out']]);
    expect(ports(definitionNamed('Outer').interface.inputs)).toEqual([['in', wrap, 'in']]);

    store().exitSubgraph();
    expect(runWiring()).toEqual(wiring);
  });
});

describe('undo and redo of a collapse inside an open block', () => {
  it('undo puts the open block\'s interface back with its nodes, redo re-points it', () => {
    const { outer, wiring } = seedOuter();
    store().enterSubgraph(outer);
    const inner = collapse('Inner', 'm2', 'm3');

    store().undo();
    expect(tab().nodes.map((n) => n.id).sort()).toEqual(['m1', 'm2', 'm3']);
    expect(ports(definitionNamed('Outer').interface.outputs)).toEqual([['out', 'm3', 'out']]);
    expect(tab().subgraphs.some((d) => d.name === 'Inner')).toBe(false);
    expect(runWiring()).toEqual(wiring);

    store().redo();
    expect(ports(definitionNamed('Outer').interface.outputs)).toEqual([['out', inner, 'out']]);
    expect(runWiring()).toEqual(wiring);

    store().exitSubgraph();
    expect(runWiring()).toEqual(wiring);
  });

  it('one undo at the top level takes back the visit, and redo brings it back whole', () => {
    const { outer, wiring } = seedOuter();
    store().enterSubgraph(outer);
    const inner = collapse('Inner', 'm2', 'm3');
    store().exitSubgraph();

    store().undo();
    expect(tab().subgraphs.map((d) => d.name)).toEqual(['Outer']);
    expect(ports(definitionNamed('Outer').interface.outputs)).toEqual([['out', 'm3', 'out']]);
    expect(runWiring()).toEqual(wiring);

    store().redo();
    expect(ports(definitionNamed('Outer').interface.outputs)).toEqual([['out', inner, 'out']]);
    expect(tab().edges.some((e) => e.source === outer && e.target === 'sink')).toBe(true);
    expect(runWiring()).toEqual(wiring);
  });
});

describe('expanding inside an open block keeps the block boundary', () => {
  it('re-points a port that named the expanded card at the node behind it', () => {
    const { outer, deep, wiring } = seedNested();
    store().enterSubgraph(outer);

    expect(store().expandSubgraphInstance(deep)).toBe(true);

    expect(ports(definitionNamed('Outer').interface.outputs)).toEqual([['out', 'm4', 'out']]);
    expect(runWiring()).toEqual(wiring);

    store().exitSubgraph();
    expect(tab().edges.some((e) => e.source === outer && e.target === 'sink')).toBe(true);
    expect(runWiring()).toEqual(wiring);
  });

  it('gives back the ports and the trigger target a collapse in there moved', () => {
    const { outer, wiring } = seedTriggered();
    store().enterSubgraph(outer);
    const inner = collapse('Inner', 'm1', 'm2');

    expect(store().expandSubgraphInstance(inner)).toBe(true);

    expect(definitionNamed('Outer').interface.triggerTargets).toEqual(['m1']);
    expect(ports(definitionNamed('Outer').interface.inputs)).toEqual([['in', 'r', 'in']]);
    expect(ports(definitionNamed('Outer').interface.outputs)).toEqual([['out', 'j', 'out']]);

    store().exitSubgraph();
    expect(runWiring()).toEqual(wiring);
  });

  it('round-trips a collapse of the block\'s output node', () => {
    const { outer, wiring } = seedOuter();
    store().enterSubgraph(outer);
    const inner = collapse('Inner', 'm2', 'm3');

    expect(store().expandSubgraphInstance(inner)).toBe(true);

    expect(ports(definitionNamed('Outer').interface.outputs)).toEqual([['out', 'm3', 'out']]);
    store().exitSubgraph();
    expect(runWiring()).toEqual(wiring);
  });

  it('hands a trigger on to the roots of a card whose block names no trigger target', () => {
    seed(
      [
        start, node('src', 'A'), node('m1', 'M'), node('m2', 'M'), node('r', 'M'),
        node('j', 'J'), node('sink', 'S'),
      ],
      [
        trigger('t2', 'start', 'src'),
        wire('e1', 'm1', 'm2'),
        wire('e2', 'm2', 'j', 'a'),
        wire('e3', 'src', 'r'),
        wire('e4', 'r', 'j', 'b'),
        wire('e5', 'j', 'sink'),
      ],
    );
    const deep = collapse('Deep', 'm1', 'm2');
    expect(definitionNamed('Deep').interface.triggerTargets).toEqual([]);
    // Start wired to the card afterwards: the engine starts Deep's roots, m1.
    store().setEdges([...tab().edges, trigger('t', 'start', deep)]);
    const wiring = runWiring();
    const outer = collapse('Outer', deep, 'r', 'j');
    expect(definitionNamed('Outer').interface.triggerTargets).toEqual([deep]);
    store().enterSubgraph(outer);

    expect(store().expandSubgraphInstance(deep)).toBe(true);

    // m1 only: r is a root of Outer too, and was never started.
    expect(definitionNamed('Outer').interface.triggerTargets).toEqual(['m1']);
    store().exitSubgraph();
    expect(runWiring()).toEqual(wiring);
  });

  it('follows a node expansion could not give its old id back', () => {
    const { outer, deep } = seedNested();
    store().enterSubgraph(outer);
    store().duplicateNode(deep);
    const copy = tab().nodes.find((n) => n.id !== deep && n.data.type === `subgraph:${definitionNamed('Deep').id}`)!;
    // The copy takes m2..m4 back as they were, so the wired card's nodes are renamed.
    expect(store().expandSubgraphInstance(copy.id)).toBe(true);
    expect(store().expandSubgraphInstance(deep)).toBe(true);

    expect(ports(definitionNamed('Outer').interface.outputs)).toEqual([['out', `${deep}-m4`, 'out']]);
    expect(tab().nodes.some((n) => n.id === `${deep}-m4`)).toBe(true);

    store().exitSubgraph();
    expect(tab().edges.some((e) => e.source === outer && e.target === 'sink')).toBe(true);
    expect(runWiring()).toContain(`${deep}-m4.out -> sink.in`);
  });

  it('keeps the definition of a block whose other copy is outside the open block', () => {
    seed(
      [start, node('src', 'A'), node('m1', 'M'), node('m2', 'M'), node('m3', 'M'), node('sink', 'S')],
      [
        trigger('t', 'start', 'src'),
        wire('e0', 'src', 'm1'),
        wire('e1', 'm1', 'm2'),
        wire('e2', 'm2', 'm3'),
        wire('e3', 'm3', 'sink'),
      ],
    );
    const deep = collapse('Deep', 'm2', 'm3');
    store().duplicateNode(deep);
    const wiring = runWiring();
    const outer = collapse('Outer', 'm1', deep);
    store().enterSubgraph(outer);

    expect(store().expandSubgraphInstance(deep)).toBe(true);
    store().exitSubgraph();

    // The copy at the top level still has its block, in the tab and in the save.
    expect(tab().subgraphs.some((d) => d.name === 'Deep')).toBe(true);
    expect(store().getSerializedGraph().subgraphs.map((d) => d.name).sort()).toEqual(['Deep', 'Outer']);
    expect(runWiring()).toEqual(wiring);
  });
});

describe('expanding a card that Start triggers', () => {
  it('starts the roots of a block that names no trigger target', () => {
    seed(
      [start, node('src', 'A'), node('m1', 'M'), node('sink', 'S')],
      [wire('e1', 'src', 'm1'), wire('e2', 'm1', 'sink')],
    );
    const card = collapse('Block', 'src', 'm1');
    expect(definitionNamed('Block').interface.triggerTargets).toEqual([]);
    // Start wired to the card after the collapse: the engine starts the
    // block's roots, src.
    store().setEdges([...tab().edges, trigger('t', 'start', card)]);
    const wiring = runWiring();
    expect(wiring).toContain('start.trigger -> src.__trigger (trigger)');

    expect(store().expandSubgraphInstance(card)).toBe(true);

    expect(tab().edges.some((e) =>
      e.source === 'start' && e.target === 'src' && e.type === 'triggerEdge')).toBe(true);
    expect(runWiring()).toEqual(wiring);
  });

  it('starts only the named targets of a block that names some', () => {
    const { outer, wiring } = seedTriggered();

    expect(store().expandSubgraphInstance(outer)).toBe(true);

    // m1 only: r is a root too, and was never started.
    expect(runWiring()).toEqual(wiring);
  });
});

describe('triggerStarts', () => {
  const block = (nodes: unknown[], edges: unknown[], triggerTargets: string[]) => ({
    id: 'b', name: 'B', description: '', nodes, edges,
    interface: { inputs: [], outputs: [], triggerTargets },
  });

  it('answers the targets a block names, and its roots when it names none', () => {
    const nodes = [{ id: 'a', type: 'M' }, { id: 'b', type: 'M' }, { id: 'c', type: 'M' }];
    const edges = [{ source: 'a', target: 'b' }];
    expect(triggerStarts(block(nodes, edges, ['b']))).toEqual(['b']);
    expect(triggerStarts(block(nodes, edges, []))).toEqual(['a', 'c']);
  });

  it('leaves notes out first, as the engine does', () => {
    const nodes = [{ id: 'n', type: 'note' }, { id: 'a', type: 'M' }, { id: 'b', type: 'M' }];
    // Without the note, b is fed by nothing, and the only named target is gone.
    const edges = [{ source: 'n', target: 'b' }, { source: 'a', target: 'n' }];
    expect(triggerStarts(block(nodes, edges, ['n']))).toEqual(['a', 'b']);
  });
});
