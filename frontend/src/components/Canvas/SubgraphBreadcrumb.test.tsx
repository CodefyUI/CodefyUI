/**
 * The sub-canvas breadcrumb (core#137).
 *
 * The bar is the ONLY thing telling the user which level they are editing,
 * so what it offers has to match what the store will actually accept: it
 * used to hand a read-only graph a click-to-rename affordance that
 * `renameSubgraph` then silently discarded (core#137 review MINOR 17).
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { act, render, renderHook, screen, fireEvent } from '@testing-library/react';
import type { Edge, Node } from '@xyflow/react';

import { SubgraphBreadcrumb } from './SubgraphBreadcrumb';
import { useTabStore } from '../../store/tabStore';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useI18n } from '../../i18n';
import { useKeyboardShortcuts } from '../../hooks/useKeyboardShortcuts';
import { saveActiveGraph } from '../../utils/saveActiveGraph';
import type { NodeData, NodeDefinition } from '../../types';
import { subgraphIdOf } from '../../utils/subgraph';

vi.mock('../../store/tabPersistence', () => ({
  readSnapshot: vi.fn(async () => null),
  writeSnapshot: vi.fn(async () => {}),
}));

vi.mock('../../utils/saveActiveGraph', () => ({ saveActiveGraph: vi.fn() }));

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
    presets: [], categorized: {}, loading: false, error: null,
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

  it('Escape leaves the name as it was', () => {
    enterABlock();
    render(<SubgraphBreadcrumb />);
    const undoDepth = tab().undoStack.length;

    fireEvent.click(screen.getByTitle('Click to rename this subgraph'));
    const input = screen.getByLabelText('Subgraph name');
    fireEvent.change(input, { target: { value: 'Discarded' } });
    fireEvent.keyDown(input, { key: 'Escape' });

    expect(screen.queryByLabelText('Subgraph name')).toBeNull();
    expect(tab().subgraphs[0].name).toBe('Block');
    expect(tab().undoStack).toHaveLength(undoDepth);
  });

  it('a name left as it was adds no undo step and keeps Redo', () => {
    // `renameSubgraph` pushes an undo frame for any name, the block's own
    // included, and a pushed frame empties Redo: clicking the name to read it
    // and clicking away cost an empty Ctrl+Z and the Ctrl+Y the user held.
    enterABlock();
    render(<SubgraphBreadcrumb />);
    fireEvent.click(screen.getByTitle('Click to rename this subgraph'));
    fireEvent.change(screen.getByLabelText('Subgraph name'), { target: { value: 'Encoder' } });
    fireEvent.keyDown(screen.getByLabelText('Subgraph name'), { key: 'Enter' });
    act(() => store().undo());
    expect(tab().subgraphs[0].name).toBe('Block');
    const undoDepth = tab().undoStack.length;
    expect(tab().redoStack).toHaveLength(1);

    fireEvent.click(screen.getByTitle('Click to rename this subgraph'));
    fireEvent.blur(screen.getByLabelText('Subgraph name'));
    // Spaces around the name do not make it a new one: the store trims them.
    fireEvent.click(screen.getByTitle('Click to rename this subgraph'));
    fireEvent.change(screen.getByLabelText('Subgraph name'), { target: { value: '  Block  ' } });
    fireEvent.keyDown(screen.getByLabelText('Subgraph name'), { key: 'Enter' });

    expect(screen.queryByLabelText('Subgraph name')).toBeNull();
    expect(tab().undoStack).toHaveLength(undoDepth);
    expect(tab().redoStack).toHaveLength(1);
  });

  // Ctrl+S saves from a field (#607), and this field holds the typed name
  // until Enter or blur. On a tab bound to a file the save writes at once,
  // with no name dialog to blur the field first (#620).
  describe('Ctrl+S while the name is being edited (#620)', () => {
    beforeEach(() => {
      vi.mocked(saveActiveGraph).mockReset();
      renderHook(() => useKeyboardShortcuts());
    });

    it('keeps the typed name and saves it', () => {
      let savedName: string | undefined;
      vi.mocked(saveActiveGraph).mockImplementation(async () => {
        savedName = tab().subgraphs[0].name;
      });
      enterABlock();
      render(<SubgraphBreadcrumb />);
      const undoDepth = tab().undoStack.length;

      fireEvent.click(screen.getByTitle('Click to rename this subgraph'));
      const input = screen.getByLabelText('Subgraph name');
      fireEvent.change(input, { target: { value: 'Renamed' } });
      // `false`: the browser's own "Save page as" was refused.
      expect(fireEvent.keyDown(input, { key: 's', ctrlKey: true })).toBe(false);

      expect(saveActiveGraph).toHaveBeenCalledTimes(1);
      expect(savedName).toBe('Renamed');
      expect(screen.queryByLabelText('Subgraph name')).toBeNull();
      expect(screen.getByTitle('Click to rename this subgraph').textContent).toBe('Renamed');
      // One rename, one undo step.
      expect(tab().undoStack).toHaveLength(undoDepth + 1);
    });

    it('Ctrl+Shift+S is not Save: the field stays open and nothing is saved', () => {
      enterABlock();
      render(<SubgraphBreadcrumb />);

      fireEvent.click(screen.getByTitle('Click to rename this subgraph'));
      const input = screen.getByLabelText('Subgraph name');
      fireEvent.change(input, { target: { value: 'Half typed' } });
      fireEvent.keyDown(input, { key: 's', ctrlKey: true, shiftKey: true });

      expect(screen.getByLabelText('Subgraph name')).toBe(input);
      expect(tab().subgraphs[0].name).toBe('Block');
      expect(saveActiveGraph).not.toHaveBeenCalled();
    });
  });
});
