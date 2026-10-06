import { describe, it, expect, beforeEach } from 'vitest';
import { screen } from '@testing-library/react';
import { nodeProps, renderWithFlow } from '../../test/utils';
import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';
import type { AppNode, SubgraphDefinition } from '../../types';
import { buildInstanceNode } from '../../utils/subgraph';
import SubgraphInstanceNode from './SubgraphInstanceNode';

/** A note the way a graph file stores one: the canvas never puts one in a
 *  block, but Import JSON or a hand-edited file can (#624). */
const NOTE = { id: 'n1', type: 'note', position: { x: 0, y: 0 }, data: { text: 'hi' } };

function inner(id: string, type: string) {
  return { id, type, position: { x: 0, y: 0 }, data: { params: {} } };
}

function block(nodes: SubgraphDefinition['nodes']): SubgraphDefinition {
  return {
    id: 'e',
    name: 'NoteOnly',
    description: '',
    nodes,
    edges: [],
    interface: { inputs: [], outputs: [], triggerTargets: [] },
  };
}

/** The block's card on the active tab, as the canvas renders it. */
function renderCard(definition: SubgraphDefinition) {
  const id = 'tab-block';
  useTabStore.setState((s) => ({
    activeTabId: id,
    tabs: [{ ...s.tabs[0], id, name: 'Tab', nodes: [], edges: [], subgraphs: [definition] }],
  }));
  const node = buildInstanceNode(definition, { x: 0, y: 0 }, 'e');
  return renderWithFlow(
    <SubgraphInstanceNode {...nodeProps<AppNode>({ id: node.id, type: node.type, data: node.data })} />,
  );
}

/** The count in the card's footer, after the block badge. */
function count(): string {
  return screen.getByTestId('subgraph-footer').lastElementChild?.textContent ?? '';
}

beforeEach(() => {
  useI18n.setState({ locale: 'zh-TW' });
});

describe('SubgraphInstanceNode', () => {
  it('counts no note, so a block holding only a note shows 0 nodes', () => {
    // The run refuses a trigger into this block because it has no node; the
    // card said 1.
    renderCard(block([NOTE]));

    expect(count()).toBe('0 個節點');
  });

  it('counts the nodes beside a note', () => {
    renderCard(block([NOTE, inner('x', 'TextInput'), inner('p', 'Print')]));

    expect(count()).toBe('2 個節點');
  });

  it('says "1 node" in English, not "1 nodes"', () => {
    useI18n.setState({ locale: 'en' });
    renderCard(block([NOTE, inner('x', 'TextInput')]));

    expect(count()).toBe('1 node');
  });

  it('says "2 nodes" in English for two', () => {
    useI18n.setState({ locale: 'en' });
    renderCard(block([inner('x', 'TextInput'), inner('p', 'Print')]));

    expect(count()).toBe('2 nodes');
  });

  it('keeps the Chinese count for one node', () => {
    renderCard(block([inner('x', 'TextInput')]));

    expect(count()).toBe('1 個節點');
  });
});
