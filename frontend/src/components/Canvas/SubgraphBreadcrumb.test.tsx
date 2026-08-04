/**
 * The sub-canvas breadcrumb (core#137).
 *
 * The bar is the ONLY thing telling the user which level they are editing,
 * so what it offers has to match what the store will actually accept: it
 * used to hand a read-only graph a click-to-rename affordance that
 * `renameSubgraph` then silently discarded (core#137 review MINOR 17).
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import type { Edge, Node } from '@xyflow/react';

import { SubgraphBreadcrumb } from './SubgraphBreadcrumb';
import { useTabStore } from '../../store/tabStore';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useI18n } from '../../i18n';
import type { NodeData, NodeDefinition } from '../../types';
import { subgraphIdOf } from '../../utils/subgraph';

vi.mock('../../store/tabPersistence', () => ({
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
    params: [],
  };
}

function node(id: string, x: number): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position: { x, y: 0 },
    data: { label: id, type: id.toUpperCase(), params: {}, definition: def(id.toUpperCase()) },
  };
}

/** Collapse b+c into "Block" and step inside it. */
function enterABlock() {
  const nodes = [node('a', 0), node('b', 100), node('c', 200)];
  const edges: Edge[] = [
    { id: 'e1', source: 'a', target: 'b', sourceHandle: 'out', targetHandle: 'in' },
    { id: 'e2', source: 'b', target: 'c', sourceHandle: 'out', targetHandle: 'in' },
  ];
  store().setNodes(nodes);
  store().setEdges(edges);
  store().setNodes(tab().nodes.map((n) => ({ ...n, selected: n.id !== 'a' })));
  store().collapseSelectionToSubgraph('Block');
  const instanceId = tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
  store().enterSubgraph(instanceId);
}

function makeReadOnly() {
  useTabStore.setState({
    tabs: useTabStore.getState().tabs.map((t) => ({ ...t, readOnly: true })),
  });
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  store().addTab('test');
  useNodeDefStore.setState({
    definitions: [def('A'), def('B'), def('C')],
    presets: [], categorized: {}, presetCategorized: {}, loading: false, error: null,
  } as never);
});

describe('SubgraphBreadcrumb', () => {
  it('renders nothing at the top level', () => {
    const { container } = render(<SubgraphBreadcrumb />);
    expect(container.firstChild).toBeNull();
  });

  it('shows the trail and renames through the current crumb', () => {
    enterABlock();
    render(<SubgraphBreadcrumb />);

    fireEvent.click(screen.getByTitle('Click to rename this subgraph'));
    const input = screen.getByLabelText('Subgraph name');
    fireEvent.change(input, { target: { value: 'Encoder' } });
    fireEvent.keyDown(input, { key: 'Enter' });

    expect(tab().subgraphs[0].name).toBe('Encoder');
  });

  it('offers NO rename affordance on a read-only graph', () => {
    enterABlock();
    makeReadOnly();
    render(<SubgraphBreadcrumb />);

    // The name still has to be visible -- the user is reading the block.
    expect(screen.getByText('Block')).toBeTruthy();
    // ... but nothing invites them to edit a name the store will discard.
    expect(screen.queryByTitle('Click to rename this subgraph')).toBeNull();
    fireEvent.click(screen.getByText('Block'));
    expect(screen.queryByLabelText('Subgraph name')).toBeNull();
    expect(tab().subgraphs[0].name).toBe('Block');
  });

  it('still lets a read-only user navigate back out', () => {
    enterABlock();
    makeReadOnly();
    render(<SubgraphBreadcrumb />);

    fireEvent.click(screen.getByTestId('subgraph-exit'));
    expect(tab().subgraphStack).toEqual([]);
  });

  it('the Main crumb leaves every level at once', () => {
    enterABlock();
    render(<SubgraphBreadcrumb />);
    fireEvent.click(screen.getByText('Main'));
    expect(tab().subgraphStack).toEqual([]);
  });
});
