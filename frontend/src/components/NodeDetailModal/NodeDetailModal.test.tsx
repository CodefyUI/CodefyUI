import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, act, within } from '@testing-library/react';
import { ReactFlowProvider } from '@xyflow/react';
import type { Edge, Node } from '@xyflow/react';
import type { NodeData, NodeDefinition, OutputData, TensorOutput } from '../../types';

// ── Module mocks ─────────────────────────────────────────────────────────────
// Captures come from the same client the InspectorPanel uses; mocking it here
// is what lets the parity test drive BOTH surfaces from one fixture.
vi.mock('../../api/executionOutputs', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('../../api/executionOutputs')>();
  return {
    ...actual,
    fetchOutput: vi.fn(),
    fetchStepIndex: vi.fn(),
    fetchGradIndex: vi.fn(),
  };
});

// Only `fetchNodeDefinition` is stubbed — ParamField's file backends stay real
// and are simply never mounted (no model_file / image_file params in fixtures).
vi.mock('../../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/rest')>();
  return { ...actual, fetchNodeDefinition: vi.fn() };
});

// EmptyCanvasOverlay fires a REST call on mount and is irrelevant here.
vi.mock('../Canvas/EmptyCanvasOverlay', () => ({
  EmptyCanvasOverlay: () => <div data-testid="empty-overlay" />,
}));

import {
  fetchOutput,
  fetchStepIndex,
  fetchGradIndex,
} from '../../api/executionOutputs';
import { fetchNodeDefinition } from '../../api/rest';
import { NodeDetailModal } from './NodeDetailModal';
import {
  BUILTIN_NODE_DETAIL_TABS,
  getNodeDetailTabs,
  registerNodeDetailTab,
  unregisterNodeDetailTab,
  type NodeDetailTabContext,
} from './tabs';
import { InspectorPanel } from '../InspectorPanel/InspectorPanel';
import { NodeConfigPanel } from '../ConfigPanel/NodeConfigPanel';
import { FlowCanvas } from '../Canvas/FlowCanvas';
import { useKeyboardShortcuts } from '../../hooks/useKeyboardShortcuts';
import { useTabStore, type TabState } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import { useDialogStore } from '../../store/dialogStore';
import { useI18n } from '../../i18n';

const mockOutput = vi.mocked(fetchOutput);
const mockStepIndex = vi.mocked(fetchStepIndex);
const mockGradIndex = vi.mocked(fetchGradIndex);
const mockNodeDef = vi.mocked(fetchNodeDefinition);

// ── Fixtures ─────────────────────────────────────────────────────────────────

function tensor(values: unknown, extra: Partial<TensorOutput> = {}): TensorOutput {
  return {
    type: 'tensor',
    run_id: 'r',
    node_id: 'n',
    port: 'p',
    full_shape: [2, 2],
    dtype: 'float32',
    slice: ':',
    sliced_shape: [2, 2],
    values,
    truncated: false,
    ...extra,
  };
}

function def(over: Partial<NodeDefinition> = {}): NodeDefinition {
  return {
    node_name: 'Linear',
    category: 'CNN',
    description: 'A dense layer.',
    inputs: [],
    outputs: [],
    params: [],
    ...over,
  };
}

function outputsDef(names: string[], over: Partial<NodeDefinition> = {}): NodeDefinition {
  return def({
    outputs: names.map((name) => ({
      name,
      data_type: 'TENSOR',
      description: `the ${name} port`,
      optional: false,
    })),
    ...over,
  });
}

function node(
  id: string,
  over: { label?: string; type?: string; definition?: NodeDefinition; data?: Partial<NodeData>; nodeType?: string } = {},
): Node<NodeData> {
  return {
    id,
    type: over.nodeType ?? 'baseNode',
    position: { x: 0, y: 0 },
    data: {
      label: over.label ?? id,
      type: over.type ?? 'Linear',
      params: {},
      definition: over.definition ?? def(),
      executionStatus: 'idle',
      ...over.data,
    },
  };
}

function edge(id: string, source: string, target: string, sourceHandle = 'out'): Edge {
  return { id, source, target, sourceHandle, targetHandle: 'in' } as Edge;
}

/** Replace the single active tab with the given partial state. */
function seedTab(partial: Partial<TabState>) {
  const active = useTabStore.getState().tabs[0];
  const next: TabState = {
    ...active,
    nodes: [],
    edges: [],
    selectedNodeId: null,
    nodeDetailNodeId: null,
    presetModalNodeId: null,
    subgraphModalNodeId: null,
    activeSegment: null,
    lastRunId: null,
    recordOutputs: true,
    outputSummaries: {},
    dirtyNodeIds: new Set(),
    undoStack: [],
    redoStack: [],
    ...partial,
  };
  useTabStore.setState({ tabs: [next], activeTabId: next.id });
}

function activeTab() {
  return useTabStore.getState().getActiveTab();
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useUIStore.setState({ isCanvasPanning: false, shortcutsModalOpen: false });
  useDialogStore.setState({ active: null });
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('t');
  mockOutput.mockReset();
  mockStepIndex.mockReset();
  mockGradIndex.mockReset();
  mockNodeDef.mockReset();
  mockStepIndex.mockResolvedValue([]);
  mockGradIndex.mockResolvedValue([]);
  mockOutput.mockResolvedValue(tensor([[1, 2], [3, 4]], { min: 1, max: 4 }));
  mockNodeDef.mockRejectedValue(new Error('offline'));
});

afterEach(() => {
  vi.useRealTimers();
});

// ── Open paths ───────────────────────────────────────────────────────────────

function ShortcutHarness() {
  useKeyboardShortcuts();
  return null;
}

/** Render a real canvas (real React Flow) plus the modal, timers flushed. */
function renderCanvasWithModal() {
  const result = render(
    <ReactFlowProvider>
      <FlowCanvas />
      <NodeDetailModal />
    </ReactFlowProvider>,
  );
  act(() => {
    vi.advanceTimersByTime(200);
  });
  return result;
}

describe('NodeDetailModal — open paths', () => {
  it('opens on a node double-click', () => {
    vi.useFakeTimers();
    seedTab({ nodes: [node('n1', { label: 'Dense A' })] });
    const { container } = renderCanvasWithModal();

    const card = container.querySelector('.react-flow__node')!.firstElementChild!;
    act(() => {
      fireEvent.dblClick(card);
    });

    expect(activeTab().nodeDetailNodeId).toBe('n1');
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Dense A' })).toBeInTheDocument();
  });

  it('opens from the node context menu "Open details" entry', () => {
    vi.useFakeTimers();
    seedTab({ nodes: [node('n1', { label: 'Dense A' })] });
    const { container } = renderCanvasWithModal();

    act(() => {
      fireEvent.contextMenu(container.querySelector('.react-flow__node')!);
    });
    act(() => {
      fireEvent.click(screen.getByText('Open details'));
    });

    expect(activeTab().nodeDetailNodeId).toBe('n1');
    expect(screen.getByRole('dialog')).toBeInTheDocument();
  });

  it('opens on Enter when a node is selected', () => {
    seedTab({ nodes: [node('n1')], selectedNodeId: 'n1' });
    render(
      <>
        <ShortcutHarness />
        <NodeDetailModal />
      </>,
    );
    fireEvent.keyDown(document, { key: 'Enter' });
    expect(activeTab().nodeDetailNodeId).toBe('n1');
    expect(screen.getByRole('dialog')).toBeInTheDocument();
  });

  it('selecting the node is part of opening, so the side panels follow', () => {
    seedTab({ nodes: [node('n1')], selectedNodeId: null });
    useTabStore.getState().openNodeDetail('n1');
    expect(activeTab().selectedNodeId).toBe('n1');
  });

  it('ignores Enter with no selection, behind a dialog, or with a modal already open', () => {
    render(<ShortcutHarness />);

    seedTab({ nodes: [node('n1')], selectedNodeId: null });
    fireEvent.keyDown(document, { key: 'Enter' });
    expect(activeTab().nodeDetailNodeId).toBeNull();

    seedTab({ nodes: [node('n1')], selectedNodeId: 'n1' });
    useDialogStore.setState({
      active: { kind: 'confirm', title: 'x', resolve: () => {} } as never,
    });
    fireEvent.keyDown(document, { key: 'Enter' });
    expect(activeTab().nodeDetailNodeId).toBeNull();
    useDialogStore.setState({ active: null });

    seedTab({ nodes: [node('n1')], selectedNodeId: 'n1', presetModalNodeId: 'n1' });
    fireEvent.keyDown(document, { key: 'Enter' });
    expect(activeTab().nodeDetailNodeId).toBeNull();

    seedTab({ nodes: [node('n1')], selectedNodeId: 'n1', subgraphModalNodeId: 'n1' });
    fireEvent.keyDown(document, { key: 'Enter' });
    expect(activeTab().nodeDetailNodeId).toBeNull();

    seedTab({ nodes: [node('n1')], selectedNodeId: 'n1', nodeDetailNodeId: 'n1' });
    const before = activeTab().nodes;
    fireEvent.keyDown(document, { key: 'Enter' });
    expect(activeTab().nodes).toBe(before); // untouched — the guard returned early
  });

  it('ignores Enter while the shortcuts overlay is up, on a note node, or on a stale selection', () => {
    render(<ShortcutHarness />);

    seedTab({ nodes: [node('n1')], selectedNodeId: 'n1' });
    useUIStore.setState({ shortcutsModalOpen: true });
    fireEvent.keyDown(document, { key: 'Enter' });
    expect(activeTab().nodeDetailNodeId).toBeNull();
    useUIStore.setState({ shortcutsModalOpen: false });

    seedTab({
      nodes: [node('note1', { nodeType: 'noteNode', type: 'note' })],
      selectedNodeId: 'note1',
    });
    fireEvent.keyDown(document, { key: 'Enter' });
    expect(activeTab().nodeDetailNodeId).toBeNull();

    seedTab({ nodes: [], selectedNodeId: 'ghost' });
    fireEvent.keyDown(document, { key: 'Enter' });
    expect(activeTab().nodeDetailNodeId).toBeNull();
  });

  it('ignores Enter with no active tab, and opens for a node with no xyflow type', () => {
    render(<ShortcutHarness />);

    seedTab({ nodes: [node('n1')], selectedNodeId: 'n1' });
    const realTabs = useTabStore.getState().tabs;
    useTabStore.setState({ activeTabId: 'no-such-tab' });
    fireEvent.keyDown(document, { key: 'Enter' });
    expect(realTabs[0].nodeDetailNodeId).toBeNull();
    useTabStore.setState({ activeTabId: realTabs[0].id });

    const untyped = node('n1');
    delete (untyped as { type?: string }).type;
    seedTab({ nodes: [untyped], selectedNodeId: 'n1' });
    fireEvent.keyDown(document, { key: 'Enter' });
    expect(activeTab().nodeDetailNodeId).toBe('n1');
  });

  it('leaves Enter alone for controls that answer it themselves', () => {
    seedTab({ nodes: [node('n1')], selectedNodeId: 'n1' });
    render(
      <>
        <ShortcutHarness />
        <button type="button">a button</button>
      </>,
    );
    fireEvent.keyDown(screen.getByText('a button'), { key: 'Enter' });
    expect(activeTab().nodeDetailNodeId).toBeNull();
  });

  it('renders nothing when no node detail is open', () => {
    seedTab({ nodes: [node('n1')] });
    const { container } = render(<NodeDetailModal />);
    expect(container.firstChild).toBeNull();
    expect(screen.queryByRole('dialog')).toBeNull();
  });
});

// ── The double-click split (regression) ──────────────────────────────────────

describe('NodeDetailModal — double-click split regression', () => {
  it('node double-click opens the modal and NOT QuickNodeSearch', () => {
    vi.useFakeTimers();
    seedTab({ nodes: [node('n1')] });
    const { container } = renderCanvasWithModal();

    act(() => {
      fireEvent.dblClick(container.querySelector('.react-flow__node')!.firstElementChild!);
    });

    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.queryByPlaceholderText('Search nodes...')).toBeNull();
  });

  it('empty-canvas double-click still opens QuickNodeSearch and NOT the modal', () => {
    vi.useFakeTimers();
    seedTab({ nodes: [node('n1')] });
    const { container } = renderCanvasWithModal();

    act(() => {
      fireEvent.dblClick(container.querySelector('.react-flow__pane')!, {
        clientX: 40,
        clientY: 40,
      });
    });

    expect(screen.getByPlaceholderText('Search nodes...')).toBeInTheDocument();
    expect(activeTab().nodeDetailNodeId).toBeNull();
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('a SequentialModel double-click keeps opening the subgraph editor, not the modal', () => {
    vi.useFakeTimers();
    seedTab({
      nodes: [
        node('seq', {
          type: 'SequentialModel',
          definition: def({ node_name: 'SequentialModel' }),
          data: { params: { layers: '[]' } },
        }),
      ],
    });
    const { container } = renderCanvasWithModal();

    act(() => {
      fireEvent.click(container.querySelector('.react-flow__node')!.firstElementChild!, {
        detail: 2,
      });
      fireEvent.dblClick(container.querySelector('.react-flow__node')!.firstElementChild!);
    });

    expect(activeTab().subgraphModalNodeId).toBe('seq');
    expect(activeTab().nodeDetailNodeId).toBeNull();
  });
});

// ── Header ───────────────────────────────────────────────────────────────────

describe('NodeDetailModal — header', () => {
  it('shows the icon, name, type, category and run status', () => {
    seedTab({
      nodes: [
        node('n1', {
          label: 'Encoder',
          type: 'Linear',
          data: { executionStatus: 'completed' },
        }),
      ],
      nodeDetailNodeId: 'n1',
    });
    render(<NodeDetailModal />);
    expect(screen.getByRole('button', { name: 'Encoder' })).toBeInTheDocument();
    expect(screen.getByText('Linear')).toBeInTheDocument();
    expect(screen.getByText('CNN')).toBeInTheDocument();
    expect(screen.getByText('Completed')).toBeInTheDocument();
    // Icon is the first letter of the node type, never an emoji.
    expect(screen.getByText('L')).toBeInTheDocument();
  });

  it('falls back to Utility and Idle for a node with no definition', () => {
    seedTab({
      nodes: [node('n1', { type: 'Mystery', data: { definition: undefined, executionStatus: undefined } })],
      nodeDetailNodeId: 'n1',
    });
    render(<NodeDetailModal />);
    expect(screen.getByText('Utility')).toBeInTheDocument();
    expect(screen.getByText('Idle')).toBeInTheDocument();
    expect(screen.getByText('Mystery')).toBeInTheDocument();
  });

  it('renders a plugin category with no palette colour, and a nameless type', () => {
    seedTab({
      nodes: [
        node('n1', {
          type: '',
          definition: def({ node_name: '', category: 'SomePluginCategory' }),
        }),
      ],
      nodeDetailNodeId: 'n1',
    });
    render(<NodeDetailModal />);
    expect(screen.getByText('SomePluginCategory')).toBeInTheDocument();
    // Icon falls back to a placeholder rather than rendering blank.
    expect(screen.getByText('?')).toBeInTheDocument();
  });

  it('still opens for a node outside the walk order, showing 0 of n', () => {
    // Notes are excluded from the walk, but `openNodeDetail` is a public
    // action — the header must not render "NaN / n" if something calls it.
    seedTab({
      nodes: [node('note1', { nodeType: 'noteNode', type: 'note' }), node('n1')],
      nodeDetailNodeId: 'note1',
    });
    render(<NodeDetailModal />);
    expect(screen.getByText('0 / 1')).toBeInTheDocument();
    expect(screen.getByLabelText('Previous node')).toBeDisabled();
    expect(screen.getByLabelText('Next node')).toBeDisabled();
  });

  it('closes via the close button', () => {
    seedTab({ nodes: [node('n1')], nodeDetailNodeId: 'n1' });
    render(<NodeDetailModal />);
    fireEvent.click(screen.getByLabelText('Close node details'));
    expect(activeTab().nodeDetailNodeId).toBeNull();
  });

  it('renames inline: Enter commits through renameNode as one undo step', () => {
    seedTab({ nodes: [node('n1', { label: 'Old' })], nodeDetailNodeId: 'n1' });
    render(<NodeDetailModal />);

    fireEvent.click(screen.getByRole('button', { name: 'Old' }));
    const input = screen.getByLabelText('Node name') as HTMLInputElement;
    fireEvent.change(input, { target: { value: '  New name  ' } });
    fireEvent.keyDown(input, { key: 'Enter' });

    expect(activeTab().nodes[0].data.label).toBe('New name');
    expect(activeTab().undoStack).toHaveLength(1);
    expect(screen.getByRole('button', { name: 'New name' })).toBeInTheDocument();
  });

  it('uses the preset accent for a preset node', () => {
    seedTab({
      nodes: [
        node('p1', {
          label: 'My Preset',
          definition: def({ node_name: 'My Preset', category: 'CNN' }),
          data: { isPreset: true },
        }),
      ],
      nodeDetailNodeId: 'p1',
    });
    render(<NodeDetailModal />);
    expect(screen.getByText('M')).toHaveStyle({ background: '#D4A017' });
    expect(screen.getByText('PRESET')).toBeInTheDocument();
  });

  it('flags a bypassed node, which the canvas card underneath cannot (core#128)', () => {
    seedTab({
      nodes: [node('n1', { data: { bypassed: true } })],
      nodeDetailNodeId: 'n1',
    });
    render(<NodeDetailModal />);
    expect(screen.getByText('BYPASS')).toBeInTheDocument();
  });

  it('carries no bypass chip for an ordinary node', () => {
    seedTab({ nodes: [node('n1')], nodeDetailNodeId: 'n1' });
    render(<NodeDetailModal />);
    expect(screen.queryByText('BYPASS')).toBeNull();
  });

  it('surfaces the commit/cancel keys only while the name editor is open', () => {
    seedTab({ nodes: [node('n1', { label: 'Old' })], nodeDetailNodeId: 'n1' });
    render(<NodeDetailModal />);
    expect(screen.queryByText('Enter to apply, Esc to cancel')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Old' }));
    expect(screen.getByText('Enter to apply, Esc to cancel')).toBeInTheDocument();
  });

  it('keeps the rename editor open for keys other than Enter', () => {
    seedTab({ nodes: [node('n1', { label: 'Old' })], nodeDetailNodeId: 'n1' });
    render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('button', { name: 'Old' }));
    fireEvent.keyDown(screen.getByLabelText('Node name'), { key: 'a' });
    expect(screen.getByLabelText('Node name')).toBeInTheDocument();
    expect(activeTab().nodes[0].data.label).toBe('Old');
  });

  it('renames on blur, and skips the commit when the name is unchanged or blank', () => {
    seedTab({ nodes: [node('n1', { label: 'Old' })], nodeDetailNodeId: 'n1' });
    render(<NodeDetailModal />);

    // Blank -> no rename, editor closes.
    fireEvent.click(screen.getByRole('button', { name: 'Old' }));
    fireEvent.change(screen.getByLabelText('Node name'), { target: { value: '   ' } });
    fireEvent.blur(screen.getByLabelText('Node name'));
    expect(activeTab().nodes[0].data.label).toBe('Old');
    expect(activeTab().undoStack).toHaveLength(0);

    // Same value -> no rename either.
    fireEvent.click(screen.getByRole('button', { name: 'Old' }));
    fireEvent.blur(screen.getByLabelText('Node name'));
    expect(activeTab().undoStack).toHaveLength(0);

    // Real change on blur -> renamed.
    fireEvent.click(screen.getByRole('button', { name: 'Old' }));
    fireEvent.change(screen.getByLabelText('Node name'), { target: { value: 'Blurred' } });
    fireEvent.blur(screen.getByLabelText('Node name'));
    expect(activeTab().nodes[0].data.label).toBe('Blurred');
  });

  it('Esc cancels an in-progress rename without closing the modal', () => {
    seedTab({ nodes: [node('n1', { label: 'Old' })], nodeDetailNodeId: 'n1' });
    render(<NodeDetailModal />);

    fireEvent.click(screen.getByRole('button', { name: 'Old' }));
    fireEvent.change(screen.getByLabelText('Node name'), { target: { value: 'Discarded' } });
    fireEvent.keyDown(window, { key: 'Escape' });

    expect(activeTab().nodeDetailNodeId).toBe('n1');
    expect(activeTab().nodes[0].data.label).toBe('Old');
    expect(screen.getByRole('button', { name: 'Old' })).toBeInTheDocument();

    // A second Esc, with no draft open, closes.
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(activeTab().nodeDetailNodeId).toBeNull();
  });
});

// ── Close paths ──────────────────────────────────────────────────────────────

describe('NodeDetailModal — close paths', () => {
  it('Esc closes the modal', () => {
    seedTab({ nodes: [node('n1')], nodeDetailNodeId: 'n1' });
    render(<NodeDetailModal />);
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(activeTab().nodeDetailNodeId).toBeNull();
  });

  it('a backdrop press closes it; a press inside the surface does not', () => {
    seedTab({ nodes: [node('n1')], nodeDetailNodeId: 'n1' });
    render(<NodeDetailModal />);
    const dialog = screen.getByRole('dialog');

    fireEvent.mouseDown(dialog);
    expect(activeTab().nodeDetailNodeId).toBe('n1');

    fireEvent.mouseDown(dialog.parentElement!);
    expect(activeTab().nodeDetailNodeId).toBeNull();
  });

  it('closes itself when the open node is deleted', () => {
    seedTab({ nodes: [node('n1')], nodeDetailNodeId: 'n1' });
    render(<NodeDetailModal />);
    act(() => {
      useTabStore.getState().deleteNode('n1');
    });
    expect(activeTab().nodeDetailNodeId).toBeNull();
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('closes itself when the Delete key removes the open node', () => {
    // React Flow's own Delete key never calls `deleteNode` — it emits a
    // `remove` change straight into `onNodesChange`. A stale id here would
    // let an undo that restores the node pop the modal open unannounced.
    seedTab({ nodes: [node('n1'), node('n2')], nodeDetailNodeId: 'n1' });
    render(<NodeDetailModal />);
    act(() => {
      useTabStore.getState().onNodesChange([{ type: 'remove', id: 'n1' }]);
    });
    expect(activeTab().nodeDetailNodeId).toBeNull();
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('leaves the modal alone when the Delete key removes a different node', () => {
    seedTab({ nodes: [node('n1'), node('n2')], nodeDetailNodeId: 'n1' });
    render(<NodeDetailModal />);
    act(() => {
      useTabStore.getState().onNodesChange([{ type: 'remove', id: 'n2' }]);
    });
    expect(activeTab().nodeDetailNodeId).toBe('n1');
    expect(screen.getByRole('dialog')).toBeInTheDocument();
  });

  it('restores focus to whatever had it when the modal opened', () => {
    seedTab({ nodes: [node('n1')], nodeDetailNodeId: 'n1' });
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();
    expect(document.activeElement).toBe(opener);

    const { unmount } = render(<NodeDetailModal />);
    unmount();
    expect(document.activeElement).toBe(opener);
    opener.remove();
  });

  it('does not chase focus to an element that left the page meanwhile', () => {
    seedTab({ nodes: [node('n1')], nodeDetailNodeId: 'n1' });
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();

    const { unmount } = render(<NodeDetailModal />);
    // The node card the user double-clicked can be re-rendered away while the
    // modal is open; focusing a detached element must not throw or matter.
    opener.remove();
    expect(() => unmount()).not.toThrow();
  });

  it('renders nothing when the open id no longer names a node', () => {
    seedTab({ nodes: [], nodeDetailNodeId: 'ghost' });
    const { container } = render(<NodeDetailModal />);
    expect(container.firstChild).toBeNull();
  });
});

// ── Prev / next navigation ───────────────────────────────────────────────────

/** c <- a -> b in canvas order, but a -> b -> c topologically. */
function chainTab(openId: string) {
  seedTab({
    nodes: [node('c'), node('a'), node('b')],
    edges: [edge('e1', 'a', 'b'), edge('e2', 'b', 'c')],
    nodeDetailNodeId: openId,
  });
}

describe('NodeDetailModal — prev/next navigation', () => {
  it('walks nodes in topological order, not canvas order', () => {
    chainTab('a');
    render(<NodeDetailModal />);
    expect(screen.getByText('1 / 3')).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText('Next node'));
    expect(activeTab().nodeDetailNodeId).toBe('b');
    expect(screen.getByText('2 / 3')).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText('Next node'));
    expect(activeTab().nodeDetailNodeId).toBe('c');

    fireEvent.click(screen.getByLabelText('Previous node'));
    expect(activeTab().nodeDetailNodeId).toBe('b');
  });

  it('disables the arrows at each end of the walk', () => {
    chainTab('a');
    const { unmount } = render(<NodeDetailModal />);
    expect(screen.getByLabelText('Previous node')).toBeDisabled();
    expect(screen.getByLabelText('Next node')).toBeEnabled();
    unmount();

    chainTab('c');
    render(<NodeDetailModal />);
    expect(screen.getByLabelText('Previous node')).toBeEnabled();
    expect(screen.getByLabelText('Next node')).toBeDisabled();
  });

  it('navigates with the arrow keys', () => {
    chainTab('a');
    render(<NodeDetailModal />);
    fireEvent.keyDown(window, { key: 'ArrowRight' });
    expect(activeTab().nodeDetailNodeId).toBe('b');
    fireEvent.keyDown(window, { key: 'ArrowLeft' });
    expect(activeTab().nodeDetailNodeId).toBe('a');
    // Already at the start: ArrowLeft is a no-op rather than a wrap.
    fireEvent.keyDown(window, { key: 'ArrowLeft' });
    expect(activeTab().nodeDetailNodeId).toBe('a');
  });

  it('leaves the arrow keys to whichever control owns them', () => {
    seedTab({
      nodes: [
        node('a', { definition: def({ params: [numberParam('units')] }), data: { params: { units: 4 } } }),
        node('b'),
      ],
      edges: [edge('e1', 'a', 'b')],
      nodeDetailNodeId: 'a',
    });
    render(<NodeDetailModal />);

    const input = document.body.querySelector('input[type="number"]') as HTMLInputElement;
    fireEvent.keyDown(input, { key: 'ArrowRight' });
    expect(activeTab().nodeDetailNodeId).toBe('a');

    // The other three kinds the guard names, driven directly.
    for (const tag of ['textarea', 'select'] as const) {
      const el = document.createElement(tag);
      document.body.appendChild(el);
      fireEvent.keyDown(el, { key: 'ArrowRight' });
      expect(activeTab().nodeDetailNodeId).toBe('a');
      el.remove();
    }
    const editable = document.createElement('div');
    editable.contentEditable = 'true';
    Object.defineProperty(editable, 'isContentEditable', { value: true });
    document.body.appendChild(editable);
    fireEvent.keyDown(editable, { key: 'ArrowRight' });
    expect(activeTab().nodeDetailNodeId).toBe('a');
    editable.remove();

    // …but a plain element does not swallow them.
    fireEvent.keyDown(window, { key: 'ArrowRight' });
    expect(activeTab().nodeDetailNodeId).toBe('b');
  });

  it('skips note nodes, and tolerates a node with no xyflow type', () => {
    const untyped = node('u');
    delete (untyped as { type?: string }).type;
    seedTab({
      nodes: [node('a'), node('note1', { nodeType: 'noteNode', type: 'note' }), untyped],
      edges: [edge('e1', 'a', 'u')],
      nodeDetailNodeId: 'a',
    });
    render(<NodeDetailModal />);
    expect(screen.getByText('1 / 2')).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText('Next node'));
    expect(activeTab().nodeDetailNodeId).toBe('u');
  });
});

// ── Tabs ─────────────────────────────────────────────────────────────────────

function numberParam(name: string, over: Record<string, unknown> = {}) {
  return {
    name,
    param_type: 'int' as const,
    default: 1,
    description: '',
    options: [],
    min_value: null,
    max_value: null,
    ...over,
  };
}

describe('NodeDetailModal — tabs', () => {
  it('shows Steps and Backward only once a run exists', () => {
    seedTab({ nodes: [node('n1')], nodeDetailNodeId: 'n1', lastRunId: null });
    const { unmount } = render(<NodeDetailModal />);
    expect(screen.getByRole('tab', { name: 'Inputs' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Outputs' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Stats' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Docs' })).toBeInTheDocument();
    expect(screen.queryByRole('tab', { name: 'Steps' })).toBeNull();
    expect(screen.queryByRole('tab', { name: 'Backward' })).toBeNull();
    unmount();

    seedTab({ nodes: [node('n1')], nodeDetailNodeId: 'n1', lastRunId: 'run1' });
    render(<NodeDetailModal />);
    expect(screen.getByRole('tab', { name: 'Steps' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Backward' })).toBeInTheDocument();
  });

  it('renders captured outputs on the Outputs tab', async () => {
    seedTab({
      nodes: [node('n1', { definition: outputsDef(['logits']) })],
      nodeDetailNodeId: 'n1',
      lastRunId: 'run1',
    });
    render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('tab', { name: 'Outputs' }));

    expect(screen.getByText('Outputs (1)')).toBeInTheDocument();
    expect(screen.getByText('logits')).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('shape [2, 2]')).toBeInTheDocument());
    expect(mockOutput).toHaveBeenCalledWith('run1', 'n1', 'logits');
  });

  it('renders captured inputs with their upstream provenance', async () => {
    seedTab({
      nodes: [node('n1'), node('src', { label: 'Source', definition: outputsDef(['y']) })],
      edges: [edge('e1', 'src', 'n1', 'y')],
      nodeDetailNodeId: 'n1',
      lastRunId: 'run1',
    });
    render(<NodeDetailModal />);

    expect(screen.getByText('Inputs (1)')).toBeInTheDocument();
    expect(screen.getByText('Source.y')).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('shape [2, 2]')).toBeInTheDocument());
  });

  it('warns when Record outputs is off but still shows whatever was captured', async () => {
    seedTab({
      nodes: [node('n1', { definition: outputsDef(['logits']) })],
      nodeDetailNodeId: 'n1',
      lastRunId: 'run1',
      recordOutputs: false,
    });
    render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('tab', { name: 'Outputs' }));

    expect(
      screen.getByText('Record outputs is off — re-run with Rec on to capture values'),
    ).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('shape [2, 2]')).toBeInTheDocument());
  });

  it('falls back to the last known shapes with a hint before anything has run', () => {
    seedTab({
      nodes: [node('n1', { definition: outputsDef(['logits']) })],
      nodeDetailNodeId: 'n1',
      lastRunId: null,
      outputSummaries: { n1: { logits: { type: 'tensor', shape: [4, 8], dtype: 'float32' } } },
    });
    render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('tab', { name: 'Outputs' }));

    expect(screen.getByText('No captured data yet')).toBeInTheDocument();
    expect(screen.getByText('Shapes from the last run')).toBeInTheDocument();
    expect(screen.getByText('tensor · [4, 8] · float32')).toBeInTheDocument();
    expect(mockOutput).not.toHaveBeenCalled();
  });

  it('says so when a port has no summary, and shows a list length when there is no shape', () => {
    seedTab({
      nodes: [node('n1', { definition: outputsDef(['a', 'b']) })],
      nodeDetailNodeId: 'n1',
      lastRunId: null,
      outputSummaries: { n1: { b: { type: 'list', length: 12 } } },
    });
    render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('tab', { name: 'Outputs' }));

    expect(screen.getByText('Nothing recorded for this node yet')).toBeInTheDocument();
    expect(screen.getByText('list · length 12')).toBeInTheDocument();
  });

  it('shows the side-specific empty text when a node has no ports at all', () => {
    seedTab({ nodes: [node('n1')], nodeDetailNodeId: 'n1', lastRunId: null });
    render(<NodeDetailModal />);
    expect(screen.getByText('No inputs connected')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('tab', { name: 'Outputs' }));
    expect(screen.getByText('Run the graph to see outputs')).toBeInTheDocument();
  });

  it('renders StepTraceView on the Steps tab', async () => {
    seedTab({ nodes: [node('n1')], nodeDetailNodeId: 'n1', lastRunId: 'run1' });
    render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('tab', { name: 'Steps' }));
    await waitFor(() =>
      expect(screen.getByText('This node does not record steps')).toBeInTheDocument(),
    );
    expect(mockStepIndex).toHaveBeenCalledWith('run1', 'n1');
  });

  it('renders BackwardView on the Backward tab', async () => {
    seedTab({ nodes: [node('n1')], nodeDetailNodeId: 'n1', lastRunId: 'run1' });
    render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('tab', { name: 'Backward' }));
    await waitFor(() =>
      expect(screen.getByText('No gradients captured')).toBeInTheDocument(),
    );
    expect(mockGradIndex).toHaveBeenCalledWith('run1', 'n1');
  });

  it('shows the Stats tab pre-run empty state before anything has run', () => {
    seedTab({ nodes: [node('n1')], nodeDetailNodeId: 'n1', lastRunId: null });
    render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('tab', { name: 'Stats' }));
    expect(screen.getByText('Node statistics')).toBeInTheDocument();
    expect(
      screen.getByText('Run the graph with Rec on to capture this node’s values'),
    ).toBeInTheDocument();
  });

  it('resets to the Inputs tab when navigating to another node', () => {
    chainTab('a');
    render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('tab', { name: 'Docs' }));
    expect(screen.getByRole('tab', { name: 'Docs' })).toHaveAttribute('aria-selected', 'true');

    fireEvent.click(screen.getByLabelText('Next node'));
    expect(screen.getByRole('tab', { name: 'Inputs' })).toHaveAttribute('aria-selected', 'true');
  });

  // ── deep link from the edge tooltip (#129) ─────────────────────────────────

  it('opens on the tab the caller asked for', () => {
    seedTab({ nodes: [node('n1')], nodeDetailNodeId: 'n1', nodeDetailTab: 'stats' });
    render(<NodeDetailModal />);
    expect(screen.getByRole('tab', { name: 'Stats' })).toHaveAttribute(
      'aria-selected',
      'true',
    );
  });

  it('lets the user leave the deep-linked tab without being dragged back', () => {
    seedTab({ nodes: [node('n1')], nodeDetailNodeId: 'n1', nodeDetailTab: 'stats' });
    const { rerender } = render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('tab', { name: 'Docs' }));
    rerender(<NodeDetailModal />);
    expect(screen.getByRole('tab', { name: 'Docs' })).toHaveAttribute(
      'aria-selected',
      'true',
    );
  });

  it('honours a SECOND deep link into the node already on screen', () => {
    // Following "View stats" from another edge into the same consumer changes
    // neither the node id nor the requested tab — only the request nonce.
    seedTab({ nodes: [node('n1'), node('n2')] });
    act(() => {
      useTabStore.getState().openNodeDetail('n1', { tab: 'stats', port: 'a::x' });
    });
    render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('tab', { name: 'Docs' }));
    expect(screen.getByRole('tab', { name: 'Docs' })).toHaveAttribute(
      'aria-selected',
      'true',
    );

    act(() => {
      useTabStore.getState().openNodeDetail('n1', { tab: 'stats', port: 'b::y' });
    });
    expect(screen.getByRole('tab', { name: 'Stats' })).toHaveAttribute(
      'aria-selected',
      'true',
    );
    expect(activeTab().nodeDetailPort).toBe('b::y');
  });

  it('mounts straight onto the deep-linked tab, never Inputs first', () => {
    // A one-commit-late correction still MOUNTS the Inputs tab, and the Inputs
    // tab fetches every connected port before being thrown away. The node
    // therefore needs a real wired input — with no edges `usePortFetches`
    // bails on an empty port set and the assertion below proves nothing.
    seedTab({
      nodes: [node('src'), node('n1')],
      edges: [edge('e1', 'src', 'n1')],
      nodeDetailNodeId: 'n1',
      nodeDetailTab: 'docs',
      lastRunId: 'run1',
    });
    render(<NodeDetailModal />);
    expect(screen.getByRole('tab', { name: 'Docs' })).toHaveAttribute(
      'aria-selected',
      'true',
    );
    expect(mockOutput).not.toHaveBeenCalled();
  });

  it('the previous test is not vacuous: Inputs first DOES fetch', () => {
    // Guards the guard. Same fixture, no deep link, so the modal opens on
    // Inputs and the wired port is fetched — which is exactly what mounting
    // Inputs before correcting to Docs would have done.
    seedTab({
      nodes: [node('src'), node('n1')],
      edges: [edge('e1', 'src', 'n1')],
      nodeDetailNodeId: 'n1',
      lastRunId: 'run1',
    });
    render(<NodeDetailModal />);
    expect(mockOutput).toHaveBeenCalledWith('run1', 'src', 'out');
  });

  it('hands the requested port to every tab as focusPort', () => {
    const seen: (string | null)[] = [];
    registerNodeDetailTab({
      id: 'probe',
      labelKey: 'nodeDetail.tabs.docs',
      order: 1,
      render: (c) => {
        seen.push(c.focusPort);
        return <div data-testid="probe" />;
      },
    });
    seedTab({
      nodes: [node('n1')],
      nodeDetailNodeId: 'n1',
      nodeDetailTab: 'probe',
      nodeDetailPort: 'src::out',
    });
    render(<NodeDetailModal />);
    expect(screen.getByTestId('probe')).toBeInTheDocument();
    expect(seen).toContain('src::out');
    unregisterNodeDetailTab('probe');
  });

  it('falls back to the first tab when the selected one disappears', () => {
    seedTab({ nodes: [node('n1')], nodeDetailNodeId: 'n1', lastRunId: 'run1' });
    const { rerender } = render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('tab', { name: 'Steps' }));
    expect(screen.getByRole('tab', { name: 'Steps' })).toHaveAttribute('aria-selected', 'true');

    act(() => {
      useTabStore.setState((s) => ({
        tabs: s.tabs.map((t) => (t.id === s.activeTabId ? { ...t, lastRunId: null } : t)),
      }));
    });
    rerender(<NodeDetailModal />);
    expect(screen.queryByRole('tab', { name: 'Steps' })).toBeNull();
    expect(screen.getByRole('tab', { name: 'Inputs' })).toHaveAttribute('aria-selected', 'true');
  });
});

// ── Docs tab ─────────────────────────────────────────────────────────────────

describe('NodeDetailModal — Docs tab', () => {
  it('documents the description, params and ports from the local definition', async () => {
    seedTab({
      nodes: [
        node('n1', {
          definition: def({
            description: 'A dense layer.',
            inputs: [{ name: 'x', data_type: 'TENSOR', description: 'the input', optional: true }],
            outputs: [{ name: 'y', data_type: 'TENSOR', description: 'the output', optional: false }],
            params: [
              numberParam('units', { description: 'width', min_value: 1, max_value: 512 }),
              { ...numberParam('mode'), param_type: 'select' as const, options: ['a', 'b'], default: 'a' },
            ],
          }),
        }),
      ],
      nodeDetailNodeId: 'n1',
    });
    render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('tab', { name: 'Docs' }));
    const docs = within(screen.getByRole('tabpanel'));

    expect(docs.getByText('A dense layer.')).toBeInTheDocument();
    expect(docs.getByText('units')).toBeInTheDocument();
    expect(docs.getByText('width')).toBeInTheDocument();
    expect(docs.getByText(/default: 1/)).toBeInTheDocument();
    expect(docs.getByText(/range: 1 — 512/)).toBeInTheDocument();
    expect(docs.getByText(/options: a, b/)).toBeInTheDocument();
    expect(docs.getByText('x')).toBeInTheDocument();
    expect(docs.getByText('the input')).toBeInTheDocument();
    expect(docs.getByText('optional')).toBeInTheDocument();
    expect(docs.getByText('the output')).toBeInTheDocument();
    // A failing refresh is silent — the local definition already documents it.
    await waitFor(() => expect(mockNodeDef).toHaveBeenCalledWith('Linear'));
    expect(docs.getByText('A dense layer.')).toBeInTheDocument();
  });

  it('prefers the server definition once GET /api/nodes/{name} answers', async () => {
    mockNodeDef.mockResolvedValue(
      def({ description: 'Server-side truth.', params: [numberParam('units')] }),
    );
    seedTab({
      nodes: [node('n1', { definition: def({ description: 'Stale local copy.' }) })],
      nodeDetailNodeId: 'n1',
    });
    render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('tab', { name: 'Docs' }));
    const docs = within(screen.getByRole('tabpanel'));

    await waitFor(() => expect(docs.getByText('Server-side truth.')).toBeInTheDocument());
    expect(docs.queryByText('Stale local copy.')).toBeNull();
  });

  it('shows the empty states for a node with no description, params or ports', async () => {
    seedTab({
      nodes: [node('n1', { definition: def({ description: '' }) })],
      nodeDetailNodeId: 'n1',
    });
    render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('tab', { name: 'Docs' }));
    const docs = within(screen.getByRole('tabpanel'));

    expect(docs.getByText('This node ships no description.')).toBeInTheDocument();
    expect(docs.getByText('This node has no parameters.')).toBeInTheDocument();
    expect(docs.getAllByText('This node has no ports.')).toHaveLength(2);
    await waitFor(() => expect(mockNodeDef).toHaveBeenCalled());
  });

  it('asks the server for the node type when the node carries no definition', async () => {
    seedTab({
      nodes: [node('n1', { type: 'Mystery', data: { definition: undefined } })],
      nodeDetailNodeId: 'n1',
    });
    render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('tab', { name: 'Docs' }));
    await waitFor(() => expect(mockNodeDef).toHaveBeenCalledWith('Mystery'));
  });

  it('renders open-ended ranges and a port with no description', () => {
    seedTab({
      nodes: [
        node('n1', {
          definition: def({
            inputs: [{ name: 'x', data_type: 'TENSOR', description: '', optional: false }],
            params: [
              numberParam('lo', { min_value: 2 }),
              numberParam('hi', { max_value: 9 }),
            ],
          }),
        }),
      ],
      nodeDetailNodeId: 'n1',
    });
    render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('tab', { name: 'Docs' }));
    const docs = within(screen.getByRole('tabpanel'));
    expect(docs.getByText(/range: 2 — \+∞/)).toBeInTheDocument();
    expect(docs.getByText(/range: -∞ — 9/)).toBeInTheDocument();
    expect(docs.getByText('x')).toBeInTheDocument();
    expect(docs.queryByText('optional')).toBeNull();
  });

  it('drops a definition that arrives after the tab is gone', async () => {
    let resolveDef!: (d: NodeDefinition) => void;
    mockNodeDef.mockReturnValue(
      new Promise<NodeDefinition>((r) => {
        resolveDef = r;
      }),
    );
    seedTab({ nodes: [node('n1')], nodeDetailNodeId: 'n1' });
    const { unmount } = render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('tab', { name: 'Docs' }));
    await waitFor(() => expect(mockNodeDef).toHaveBeenCalled());
    unmount();
    resolveDef(def({ description: 'too late' }));
    await Promise.resolve();
    expect(screen.queryByText('too late')).toBeNull();
  });
});

// ── Parameter editing: parity with ConfigPanel ───────────────────────────────

function paramNode() {
  return node('n1', {
    definition: def({
      params: [
        numberParam('units', { description: 'width', min_value: 1, max_value: 512 }),
        numberParam('hidden', { visible_when: { units: 999 } } as never),
      ],
    }),
    data: { params: { units: 4, hidden: 1 } },
  });
}

/** Type into the first number field of `ui` and report what the store did. */
function editUnitsThrough(ui: React.ReactElement) {
  seedTab({ nodes: [paramNode()], selectedNodeId: 'n1', nodeDetailNodeId: 'n1' });
  const { unmount } = render(ui);
  const input = document.body.querySelector('input[type="number"]') as HTMLInputElement;
  fireEvent.change(input, { target: { value: '7' } });
  const tab = activeTab();
  const observed = {
    params: { ...tab.nodes[0].data.params },
    dirty: [...tab.dirtyNodeIds].sort(),
    undoDepth: tab.undoStack.length,
    redoDepth: tab.redoStack.length,
  };
  unmount();
  return observed;
}

describe('NodeDetailModal — parameter editing', () => {
  it('has exactly the same store effect as editing in ConfigPanel', () => {
    const viaPanel = editUnitsThrough(<NodeConfigPanel />);
    const viaModal = editUnitsThrough(<NodeDetailModal />);
    expect(viaModal).toEqual(viaPanel);
    // Pin the shared semantics rather than only their equality: the node goes
    // dirty (so partial re-execution reruns it) and typing pushes no undo
    // snapshot per keystroke.
    expect(viaModal.params.units).toBe(7);
    expect(viaModal.dirty).toEqual(['n1']);
    expect(viaModal.undoDepth).toBe(0);
  });

  it('honours visible_when exactly like ConfigPanel does', () => {
    seedTab({ nodes: [paramNode()], selectedNodeId: 'n1', nodeDetailNodeId: 'n1' });
    render(<NodeDetailModal />);
    expect(document.body.querySelectorAll('input[type="number"]')).toHaveLength(1);
    expect(screen.getByText('width')).toBeInTheDocument();
    expect(screen.getByText('Range: 1 — 512')).toBeInTheDocument();
  });

  it('shows the no-params message for a node with none', () => {
    seedTab({ nodes: [node('n1')], nodeDetailNodeId: 'n1' });
    render(<NodeDetailModal />);
    expect(
      screen.getByText('This node has no configurable parameters'),
    ).toBeInTheDocument();
  });

  it('offers the preset editor instead of claiming a preset has no parameters', () => {
    // `addPresetNode` synthesizes `definition.params: []`, so the plain
    // no-params message would be a lie — a preset's params live on the nodes
    // inside it. Same branch, and same destination, as the config panel.
    seedTab({
      nodes: [
        node('p1', {
          label: 'My Preset',
          definition: def({ node_name: 'My Preset', params: [] }),
          data: {
            isPreset: true,
            presetDefinition: { nodes: [{}, {}, {}] } as never,
          },
        }),
      ],
      nodeDetailNodeId: 'p1',
    });
    render(<NodeDetailModal />);

    expect(screen.queryByText('This node has no configurable parameters')).toBeNull();
    expect(screen.getByText('3 nodes inside')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Configure Preset'));
    // Hands over rather than stacking: the preset editor sits at a lower
    // z-index and has no Escape handler of its own.
    expect(activeTab().presetModalNodeId).toBe('p1');
    expect(activeTab().nodeDetailNodeId).toBeNull();
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('reports zero inner nodes when a preset definition is missing', () => {
    seedTab({
      nodes: [
        node('p1', {
          definition: def({ params: [] }),
          data: { isPreset: true, presetDefinition: undefined },
        }),
      ],
      nodeDetailNodeId: 'p1',
    });
    render(<NodeDetailModal />);
    expect(screen.getByText('0 nodes inside')).toBeInTheDocument();
  });
});

// ── Capture parity with the InspectorPanel ───────────────────────────────────

describe('NodeDetailModal — capture parity with InspectorPanel', () => {
  /** Port names and tensor shapes a surface put on screen. */
  function readPorts(root: HTMLElement) {
    const names = [...root.querySelectorAll('[class*="portName"]')].map((el) =>
      (el.textContent ?? '').trim(),
    );
    const shapes = [...root.querySelectorAll('[class*="tensorMeta"]')].map((el) =>
      (el.textContent ?? '').trim(),
    );
    return { names: names.sort(), shapes: shapes.sort() };
  }

  it('shows the same tensors the InspectorPanel shows, from one fixture', async () => {
    const fixture = {
      nodes: [
        node('n1', { definition: outputsDef(['logits']) }),
        node('src', { label: 'Source', definition: outputsDef(['y']) }),
      ],
      edges: [edge('e1', 'src', 'n1', 'y')],
      selectedNodeId: 'n1',
      nodeDetailNodeId: 'n1',
      lastRunId: 'run1',
    };
    mockOutput.mockImplementation(async (_run, nodeId) =>
      nodeId === 'src'
        ? (tensor([[1, 2], [3, 4]], { min: 1, max: 4 }) as OutputData)
        : (tensor([[5, 6, 7]], {
            full_shape: [1, 3],
            sliced_shape: [1, 3],
            min: 5,
            max: 7,
          }) as OutputData),
    );

    seedTab(fixture);
    const inspector = render(<InspectorPanel />);
    await waitFor(() =>
      expect(inspector.container.querySelectorAll('[class*="tensorMeta"]').length).toBe(2),
    );
    const fromInspector = readPorts(inspector.container);
    inspector.unmount();

    seedTab(fixture);
    render(<NodeDetailModal />);
    // Inputs first, then Outputs — the modal splits into two tabs, so gather
    // both sides before comparing with the panel's single Forward view.
    await waitFor(() =>
      expect(document.body.querySelectorAll('[class*="tensorMeta"]').length).toBe(1),
    );
    const modalInputs = readPorts(document.body);
    fireEvent.click(screen.getByRole('tab', { name: 'Outputs' }));
    await waitFor(() =>
      expect(document.body.querySelectorAll('[class*="tensorMeta"]').length).toBe(1),
    );
    const modalOutputs = readPorts(document.body);

    const fromModal = {
      names: [...modalInputs.names, ...modalOutputs.names].sort(),
      shapes: [...modalInputs.shapes, ...modalOutputs.shapes].sort(),
    };
    expect(fromModal).toEqual(fromInspector);
    expect(fromModal.names).toEqual(['Source.y', 'logits']);
    expect(fromModal.shapes).toEqual([
      'shape [1, 3]float32min 5 · max 7',
      'shape [2, 2]float32min 1 · max 4',
    ]);
  });

  it('surfaces a failed capture fetch as a per-port error, like the panel does', async () => {
    seedTab({
      nodes: [node('n1', { definition: outputsDef(['logits']) })],
      nodeDetailNodeId: 'n1',
      lastRunId: 'run1',
    });
    mockOutput.mockRejectedValue(new Error('port failed'));
    render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('tab', { name: 'Outputs' }));
    await waitFor(() => expect(screen.getByText('port failed')).toBeInTheDocument());
  });
});

// ── Tab registry (container contract for #129 / #131) ────────────────────────

// ── #130: charts a node emitted, on its Outputs tab ──────────────────────────
// A chart rides the node_status stream rather than the captures API, so the
// tab reads it from the tab's log. That is also why it shows with Record
// outputs off, when the port fetches below it have nothing.

describe('NodeDetailModal — chart outputs (#130)', () => {
  function chartLog(nodeId: string, chart: Record<string, unknown>) {
    return {
      timestamp: 1_700_000_000_000,
      nodeId,
      message: '',
      type: 'info' as const,
      kind: 'chart' as const,
      chart: chart as any,
    };
  }

  const CONFUSION = {
    kind: 'heatmap',
    title: 'Confusion matrix',
    matrix: [[13, 0], [1, 12]],
    row_labels: ['setosa', 'versicolor'],
    col_labels: ['setosa', 'versicolor'],
    vmin: 0,
    vmax: 13,
  };

  it('renders a confusion matrix as a heatmap on the Outputs tab', async () => {
    seedTab({
      nodes: [node('n1', { definition: outputsDef(['matrix']) })],
      nodeDetailNodeId: 'n1',
      lastRunId: 'run1',
      logs: [chartLog('n1', CONFUSION)],
    });
    render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('tab', { name: 'Outputs' }));

    expect(screen.getByText('Charts (1)')).toBeInTheDocument();
    // The modal renders through a portal, so query the document, not the root.
    expect(document.querySelector('[data-chart-kind="heatmap"]')).toBeInTheDocument();
    expect(document.querySelectorAll('rect[data-i]')).toHaveLength(4);
    expect(screen.getByText('Confusion matrix')).toBeInTheDocument();
    // The captured ports still render underneath.
    await waitFor(() => expect(screen.getByText('shape [2, 2]')).toBeInTheDocument());
  });

  it('shows only the charts belonging to the node being inspected', () => {
    seedTab({
      nodes: [node('n1'), node('n2')],
      nodeDetailNodeId: 'n1',
      lastRunId: 'run1',
      logs: [
        chartLog('n1', { kind: 'bar', title: 'Mine', bars: [] }),
        chartLog('n2', { kind: 'bar', title: 'Someone else', bars: [] }),
      ],
    });
    render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('tab', { name: 'Outputs' }));

    expect(screen.getByText('Mine')).toBeInTheDocument();
    expect(screen.queryByText('Someone else')).toBeNull();
  });

  it('keeps charts off the Inputs tab', () => {
    seedTab({
      nodes: [node('n1')],
      nodeDetailNodeId: 'n1',
      lastRunId: 'run1',
      logs: [chartLog('n1', CONFUSION)],
    });
    render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('tab', { name: 'Inputs' }));
    expect(screen.queryByText('Charts (1)')).toBeNull();
  });

  it('shows the chart before a run has captured anything', () => {
    seedTab({
      nodes: [node('n1')],
      nodeDetailNodeId: 'n1',
      lastRunId: null,
      logs: [chartLog('n1', CONFUSION)],
    });
    render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('tab', { name: 'Outputs' }));
    expect(document.querySelector('[data-chart-kind="heatmap"]')).toBeInTheDocument();
  });

  it('renders no chart block when the node emitted none', () => {
    seedTab({ nodes: [node('n1')], nodeDetailNodeId: 'n1', lastRunId: 'run1', logs: [] });
    render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('tab', { name: 'Outputs' }));
    expect(screen.queryByText(/^Charts \(/)).toBeNull();
  });
});

describe('NodeDetailModal — tab registry', () => {
  function ctx(over: Partial<NodeDetailTabContext> = {}): NodeDetailTabContext {
    const n = node('n1');
    return {
      nodeId: 'n1',
      node: n,
      runId: null,
      nodes: [n],
      edges: [],
      recordOutputs: true,
      outputSummaries: {},
      focusPort: null,
      ...over,
    };
  }

  afterEach(() => {
    unregisterNodeDetailTab('extra');
    for (const spec of BUILTIN_NODE_DETAIL_TABS) unregisterNodeDetailTab(spec.id);
  });

  it('returns the built-ins sorted by order, filtered by isEnabled', () => {
    expect(getNodeDetailTabs(ctx()).map((t) => t.id)).toEqual([
      'inputs',
      'outputs',
      'stats',
      'docs',
    ]);
    expect(getNodeDetailTabs(ctx({ runId: 'run1' })).map((t) => t.id)).toEqual([
      'inputs',
      'outputs',
      'steps',
      'backward',
      'stats',
      'docs',
    ]);
    // Every built-in declares a distinct order so the list is deterministic.
    const orders = BUILTIN_NODE_DETAIL_TABS.map((t) => t.order);
    expect(new Set(orders).size).toBe(orders.length);
  });

  it('inserts a registered tab at its declared order and renders it', () => {
    registerNodeDetailTab({
      id: 'extra',
      labelKey: 'nodeDetail.tabs.stats',
      order: 45,
      render: (c) => <div data-testid="extra-tab">extra:{c.nodeId}</div>,
    });
    expect(getNodeDetailTabs(ctx()).map((t) => t.id)).toEqual([
      'inputs',
      'outputs',
      'extra',
      'stats',
      'docs',
    ]);

    seedTab({ nodes: [node('n1')], nodeDetailNodeId: 'n1' });
    render(<NodeDetailModal />);
    // Two tabs carry the Stats label here; pick the registered one by order.
    fireEvent.click(screen.getAllByRole('tab', { name: 'Stats' })[0]);
    expect(screen.getByTestId('extra-tab')).toHaveTextContent('extra:n1');
  });

  it('replaces a built-in in place when re-registered under its id', () => {
    registerNodeDetailTab({
      id: 'stats',
      labelKey: 'nodeDetail.tabs.stats',
      order: 50,
      render: () => <div data-testid="real-stats">real stats</div>,
    });
    seedTab({ nodes: [node('n1')], nodeDetailNodeId: 'n1' });
    render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('tab', { name: 'Stats' }));
    expect(screen.getByTestId('real-stats')).toBeInTheDocument();
    expect(screen.queryByText('Node statistics')).toBeNull();
    // The slot is kept, not appended.
    expect(getNodeDetailTabs(ctx()).map((t) => t.id)).toEqual([
      'inputs',
      'outputs',
      'stats',
      'docs',
    ]);
  });

  it('reverts to the built-in after unregistering', () => {
    registerNodeDetailTab({
      id: 'stats',
      labelKey: 'nodeDetail.tabs.stats',
      order: 50,
      render: () => <div data-testid="real-stats">real stats</div>,
    });
    unregisterNodeDetailTab('stats');
    seedTab({ nodes: [node('n1')], nodeDetailNodeId: 'n1' });
    render(<NodeDetailModal />);
    fireEvent.click(screen.getByRole('tab', { name: 'Stats' }));
    expect(screen.getByText('Node statistics')).toBeInTheDocument();
  });

  it('contains a throwing tab inside the panel instead of unmounting the app', () => {
    // Registered tabs are third-party render code; a throw must cost the
    // panel, not the editor and the user's unsaved graph with it.
    const boom = vi.spyOn(console, 'error').mockImplementation(() => {});
    registerNodeDetailTab({
      id: 'extra',
      labelKey: 'nodeDetail.tabs.stats',
      order: 15,
      render: () => {
        throw new Error('tab blew up');
      },
    });
    seedTab({ nodes: [node('n1', { label: 'Still here' })], nodeDetailNodeId: 'n1' });
    render(<NodeDetailModal />);
    fireEvent.click(screen.getAllByRole('tab', { name: 'Stats' })[0]);

    expect(screen.getByText('This tab failed to render')).toBeInTheDocument();
    expect(screen.getByText('tab blew up')).toBeInTheDocument();
    // Header, param column and tab strip all survive, so the user can leave.
    expect(screen.getByRole('button', { name: 'Still here' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('tab', { name: 'Docs' }));
    expect(screen.queryByText('This tab failed to render')).toBeNull();
    boom.mockRestore();
  });

  it('survives a registry that leaves no tab enabled', () => {
    // Nothing in the product does this, but the registry is open to third
    // parties — the modal must still show its header and parameter form
    // rather than crash on an absent active tab.
    for (const spec of BUILTIN_NODE_DETAIL_TABS) {
      registerNodeDetailTab({ ...spec, isEnabled: () => false });
    }
    seedTab({ nodes: [node('n1', { label: 'Alone' })], nodeDetailNodeId: 'n1' });
    render(<NodeDetailModal />);

    expect(screen.queryAllByRole('tab')).toHaveLength(0);
    expect(screen.getByRole('button', { name: 'Alone' })).toBeInTheDocument();
    expect(screen.getByRole('tabpanel')).not.toHaveAttribute('aria-labelledby');
  });

  it('wires each tab button to the panel it controls', () => {
    seedTab({ nodes: [node('n1')], nodeDetailNodeId: 'n1' });
    render(<NodeDetailModal />);
    const inputsTab = screen.getByRole('tab', { name: 'Inputs' });
    const panel = screen.getByRole('tabpanel');
    expect(inputsTab).toHaveAttribute('id', 'node-detail-tab-inputs');
    expect(inputsTab).toHaveAttribute('aria-controls', panel.id);
    expect(panel).toHaveAttribute('aria-labelledby', 'node-detail-tab-inputs');

    fireEvent.click(screen.getByRole('tab', { name: 'Docs' }));
    expect(screen.getByRole('tabpanel')).toHaveAttribute(
      'aria-labelledby',
      'node-detail-tab-docs',
    );
  });
});
