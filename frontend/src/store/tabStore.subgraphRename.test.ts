/**
 * #620: undo and redo of a block rename keep every card of the block and the
 * definition on one name.
 *
 * A card carries a copy of its block's name (`data.label`, and the
 * `node_name` of the ports it renders), and the canvas, the validation toasts
 * and the export messages all read that copy. Leaving a block pushes one undo
 * step built from what the frame stashed on entry: the canvas above and the
 * definitions as they were. The rename used to write its new name into that
 * stashed canvas, so undoing the visit put the old name back on the definition
 * and left the new one on the cards.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { Edge, Node } from '@xyflow/react';

import { flushSubgraphEditing, useTabStore } from './tabStore';
import { useNodeDefStore } from './nodeDefStore';
import { useToastStore } from './toastStore';
import { useI18n } from '../i18n';
import type { NodeData, NodeDefinition, SubgraphDefinition } from '../types';
import { buildInstanceNode, subgraphIdOf } from '../utils/subgraph';
import { dismissValidationToasts, showValidationIssues } from '../utils/validationToasts';

vi.mock('./tabPersistence', () => ({
  readSnapshot: vi.fn(async () => null),
  writeSnapshot: vi.fn(async () => {}),
}));

const store = () => useTabStore.getState();
const tab = () => useTabStore.getState().getActiveTab();

function def(name: string): NodeDefinition {
  return {
    node_name: name,
    category: 'x',
    description: '',
    inputs: [{ name: 'in', data_type: 'TENSOR', description: '', optional: false }],
    outputs: [{ name: 'out', data_type: 'TENSOR', description: '', optional: false }],
    params: [
      {
        name: 'scale', param_type: 'float', default: 1, description: '',
        options: [], min_value: null, max_value: null,
      },
    ],
  };
}

function node(id: string, type: string, x: number): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position: { x, y: 0 },
    data: { label: id, type, params: { scale: 1 }, definition: def(type), executionStatus: 'idle' },
  };
}

function dataEdge(id: string, source: string, target: string): Edge {
  return { id, source, target, sourceHandle: 'out', targetHandle: 'in' };
}

/** The name a card shows, both ways it is read: its label and its ports' `node_name`. */
function cardName(nodeId: string): [string | undefined, string | undefined] {
  const data = tab().nodes.find((n) => n.id === nodeId)!.data;
  return [data.label, data.definition?.node_name];
}

const blockName = (subgraphId: string) => tab().subgraphs.find((d) => d.id === subgraphId)!.name;

/** a -> b -> c with b and c collapsed into "Scale", and a second card of that block. */
function twoCardsOfScale(): { cards: string[]; subgraphId: string } {
  store().setNodes([node('a', 'A', 0), node('b', 'B', 100), node('c', 'C', 200)]);
  store().setEdges([dataEdge('e1', 'a', 'b'), dataEdge('e2', 'b', 'c')]);
  store().setNodes(tab().nodes.map((n) => ({ ...n, selected: n.id !== 'a' })));
  store().collapseSelectionToSubgraph('Scale');
  const first = tab().nodes.find((n) => subgraphIdOf(n.data.type))!;
  const definition = tab().subgraphs[0];
  store().setNodes([...tab().nodes, buildInstanceNode(definition, { x: 0, y: 300 }, 'copy')]);
  return { cards: [first.id, 'copy'], subgraphId: definition.id };
}

function expectEveryCard(cards: string[], name: string) {
  for (const id of cards) expect(cardName(id)).toEqual([name, name]);
}

/** A block holding a card of another block: Outer > Inner. */
const inner: SubgraphDefinition = {
  id: 'inner',
  name: 'Inner',
  description: '',
  nodes: [{ id: 'mul2', type: 'Mul', position: { x: 0, y: 0 }, data: { params: {} } }],
  edges: [],
  interface: { inputs: [], outputs: [], triggerTargets: [] },
};
const outer: SubgraphDefinition = {
  id: 'outer',
  name: 'Outer',
  description: '',
  nodes: [
    { id: 'mul', type: 'Mul', position: { x: 0, y: 0 }, data: { params: {} } },
    { id: 'nest', type: 'subgraph:inner', position: { x: 200, y: 0 }, data: { params: {} } },
  ],
  edges: [],
  interface: { inputs: [], outputs: [], triggerTargets: [] },
};

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
  store().addTab('test');
  useNodeDefStore.setState({
    definitions: [def('A'), def('B'), def('C'), def('Mul')],
    presets: [],
    categorized: {},
    loading: false,
    error: null,
  } as never);
});

describe('undo and redo of a block rename (#620)', () => {
  it.each([
    ['Back, one level', () => store().exitSubgraph()],
    ['Main, every level', () => store().exitAllSubgraphs()],
  ])('renamed inside, left with %s: undo and redo move the cards with the definition', (_how, leave) => {
    const { cards, subgraphId } = twoCardsOfScale();
    store().enterSubgraph(cards[0]);
    store().renameSubgraph(subgraphId, 'R1');
    leave();
    expect(blockName(subgraphId)).toBe('R1');
    expectEveryCard(cards, 'R1');

    store().undo();
    expect(blockName(subgraphId)).toBe('Scale');
    expectEveryCard(cards, 'Scale');

    store().redo();
    expect(blockName(subgraphId)).toBe('R1');
    expectEveryCard(cards, 'R1');
  });

  it('a rename undone inside the block does not come back when the visit is undone', () => {
    // The visit also changes a parameter, so leaving pushes an undo step.
    const { cards, subgraphId } = twoCardsOfScale();
    store().enterSubgraph(cards[0]);
    store().updateNodeParams('b', { scale: 5 });
    store().renameSubgraph(subgraphId, 'R1');
    store().undo();
    expect(blockName(subgraphId)).toBe('Scale');
    store().exitSubgraph();
    expectEveryCard(cards, 'Scale');

    store().undo();
    expect(blockName(subgraphId)).toBe('Scale');
    expectEveryCard(cards, 'Scale');

    store().redo();
    expect(blockName(subgraphId)).toBe('Scale');
    expectEveryCard(cards, 'Scale');
  });

  it('a rename undone and redone inside the block is undone with the visit', () => {
    const { cards, subgraphId } = twoCardsOfScale();
    store().enterSubgraph(cards[0]);
    store().renameSubgraph(subgraphId, 'R1');
    store().undo();
    store().redo();
    expect(blockName(subgraphId)).toBe('R1');
    store().exitSubgraph();
    expectEveryCard(cards, 'R1');

    store().undo();
    expect(blockName(subgraphId)).toBe('Scale');
    expectEveryCard(cards, 'Scale');
  });

  it('a block inside a block: undo one level up moves its card with the definition', () => {
    store().setSubgraphs([outer, inner]);
    store().setNodes([buildInstanceNode(outer, { x: 0, y: 0 }, 'blk')]);
    store().enterSubgraph('blk');
    expect(cardName('nest')).toEqual(['Inner', 'Inner']);

    store().enterSubgraph('nest');
    store().renameSubgraph('inner', 'R1');
    store().exitSubgraph();
    expect(cardName('nest')).toEqual(['R1', 'R1']);

    store().undo();
    expect(blockName('inner')).toBe('Inner');
    expect(cardName('nest')).toEqual(['Inner', 'Inner']);

    store().redo();
    expect(blockName('inner')).toBe('R1');
    expect(cardName('nest')).toEqual(['R1', 'R1']);
  });
});

describe('a copy of a block elsewhere in the graph (#620)', () => {
  /** Outer on the top canvas with a second copy of Inner beside it; rename Inner from inside Outer > Inner. */
  function renameInnerTwoLevelsDeep() {
    store().setSubgraphs([outer, inner]);
    store().setNodes([
      buildInstanceNode(outer, { x: 0, y: 0 }, 'blk'),
      buildInstanceNode(inner, { x: 0, y: 300 }, 'top-inner'),
    ]);
    store().enterSubgraph('blk');
    store().enterSubgraph('nest');
    store().renameSubgraph('inner', 'R1');
  }

  it.each([
    ['Main', () => store().exitAllSubgraphs()],
    ['Back twice', () => {
      store().exitSubgraph();
      store().exitSubgraph();
    }],
  ])('left to the top with %s: the copy on the top canvas takes the new name', (_how, leave) => {
    renameInnerTwoLevelsDeep();
    leave();
    expect(tab().subgraphStack).toEqual([]);
    expect(blockName('inner')).toBe('R1');
    expect(cardName('top-inner')).toEqual(['R1', 'R1']);

    store().undo();
    expect(blockName('inner')).toBe('Inner');
    expect(cardName('top-inner')).toEqual(['Inner', 'Inner']);

    store().redo();
    expect(blockName('inner')).toBe('R1');
    expect(cardName('top-inner')).toEqual(['R1', 'R1']);
  });

  it('a save or run from two levels deep writes the copy with the new name', () => {
    renameInnerTwoLevelsDeep();
    const copy = flushSubgraphEditing(tab()).nodes.find((n) => n.id === 'top-inner')!;
    expect([copy.data.label, copy.data.definition?.node_name]).toEqual(['R1', 'R1']);
  });
});

describe('a validation message about a block renamed inside it (#620)', () => {
  it('names the card by the new name while the block is still open', () => {
    // What a Run from inside the block, or an Export the server refuses,
    // shows: the server's id for an inner node is `<card>/<inner>`.
    useI18n.setState({ locale: 'en' });
    useToastStore.setState({ toasts: [] });
    dismissValidationToasts();
    const { cards, subgraphId } = twoCardsOfScale();
    store().enterSubgraph(cards[0]);
    store().renameSubgraph(subgraphId, 'R1');

    showValidationIssues(tab().id, [
      {
        message: `Node ${cards[0]}/b is missing required input 'in'`,
        code: 'missing_input',
        node_id: `${cards[0]}/b`,
        params: { port: 'in', type: 'B' },
      },
    ]);

    expect(useToastStore.getState().toasts.map((toast) => toast.message)).toEqual([
      'R1: input "in" is not connected',
    ]);
  });
});
