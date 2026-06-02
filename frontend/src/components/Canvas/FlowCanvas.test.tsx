import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { screen, fireEvent, act } from '@testing-library/react';
import type { Node, Edge } from '@xyflow/react';
import type { NodeData, NodeDefinition } from '../../types';

// ── Capture everything <ReactFlow> receives ─────────────────────────────────
// FlowCanvas wires ~20 handlers onto <ReactFlow>. jsdom can't drive real
// drag/connect/pane gestures, so we replace ReactFlow with a stub that records
// the props and renders a `.react-flow__pane` (the dblclick effect attaches to
// it) plus the children. Every other xyflow export stays real.
type RFProps = Record<string, any>;
const captured: { rf: RFProps; minimap: RFProps } = { rf: {}, minimap: {} };
// When false, the stubbed ReactFlow renders WITHOUT a `.react-flow__pane`,
// so the dblclick effect's `if (pane)` guards take their false branch.
const renderPane = { value: true };

vi.mock('@xyflow/react', async (importActual) => {
  const actual = await importActual<typeof import('@xyflow/react')>();
  return {
    ...actual,
    ReactFlow: (props: RFProps) => {
      captured.rf = props;
      return (
        <div data-testid="reactflow">
          {renderPane.value && <div className="react-flow__pane" data-testid="pane" />}
          {props.children}
        </div>
      );
    },
    MiniMap: (props: RFProps) => {
      captured.minimap = props;
      return <div data-testid="minimap" />;
    },
    Background: () => <div data-testid="background" />,
    Controls: () => <div data-testid="controls" />,
  };
});

// EmptyCanvasOverlay fires a REST call (listExamples) on mount; stub it out.
vi.mock('./EmptyCanvasOverlay', () => ({
  EmptyCanvasOverlay: () => <div data-testid="empty-overlay" />,
}));

// QuickNodeSearch / PaneContextMenu are exercised in their own suites; keep
// FlowCanvas focused on its own handlers by stubbing them to readable markers.
vi.mock('./QuickNodeSearch', () => ({
  QuickNodeSearch: (props: any) => (
    <div data-testid="quick-search" onClick={props.onClose}>
      quick:{props.flowPos.x},{props.flowPos.y}
    </div>
  ),
}));
vi.mock('./PaneContextMenu', () => ({
  PaneContextMenu: (props: any) => (
    <div data-testid="pane-menu" onClick={props.onClose}>
      pane-menu:{props.flow.x},{props.flow.y}
    </div>
  ),
}));

import { FlowCanvas } from './FlowCanvas';
import { renderWithFlow } from '../../test/utils';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useDialogStore } from '../../store/dialogStore';
import { useI18n } from '../../i18n';

// ── Store helpers ───────────────────────────────────────────────────────────

const ORIGINAL_TABS = useTabStore.getState().tabs;
const ORIGINAL_ACTIVE = useTabStore.getState().activeTabId;
const TAB_ID = 'tab-canvas-test';

function makeDef(over: Partial<NodeDefinition> = {}): NodeDefinition {
  return {
    node_name: 'Linear',
    category: 'Utility',
    description: 'd',
    inputs: [{ name: 'in', data_type: 'TENSOR', description: '', optional: false }],
    outputs: [{ name: 'out', data_type: 'TENSOR', description: '', optional: false }],
    params: [],
    ...over,
  };
}

function node(id: string, over: Partial<Node<NodeData>> = {}): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position: { x: 0, y: 0 },
    data: { label: id, type: 'Linear', params: {}, definition: makeDef() },
    ...over,
  } as Node<NodeData>;
}

function setTab(partial: Partial<{ nodes: Node<NodeData>[]; edges: Edge[]; outputSummaries: any }>) {
  useTabStore.setState((s) => ({
    tabs: s.tabs.map((t) =>
      t.id === TAB_ID ? { ...t, ...partial } : t,
    ),
  }));
}

function activeTab() {
  return useTabStore.getState().tabs.find((t) => t.id === TAB_ID)!;
}

function renderCanvas() {
  // useReactFlow / useDragAndDrop need the zustand provider context, even
  // though we stub the <ReactFlow> component itself.
  return renderWithFlow(<FlowCanvas />);
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  // Fresh single tab with a stable id.
  const base = ORIGINAL_TABS[0];
  useTabStore.setState({
    tabs: [{ ...base, id: TAB_ID, name: 'test', nodes: [], edges: [], outputSummaries: {} }],
    activeTabId: TAB_ID,
    clipboard: null,
  });
  useUIStore.setState({ gridSnapEnabled: false, draggingSourceType: null, isCanvasPanning: false });
  useNodeDefStore.setState({ definitions: [makeDef()], presets: [] });
  useDialogStore.setState({ active: null, resolve: null });
  captured.rf = {};
  captured.minimap = {};
  renderPane.value = true;
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
  useTabStore.setState({ tabs: ORIGINAL_TABS, activeTabId: ORIGINAL_ACTIVE, clipboard: null });
});

// ── Rendering ───────────────────────────────────────────────────────────────

describe('FlowCanvas rendering', () => {
  it('renders the empty overlay when the active tab has no nodes', () => {
    renderCanvas();
    expect(screen.getByTestId('empty-overlay')).toBeTruthy();
    expect(screen.getByTestId('reactflow')).toBeTruthy();
  });

  it('does not render the empty overlay when nodes exist', () => {
    setTab({ nodes: [node('a')] });
    renderCanvas();
    expect(screen.queryByTestId('empty-overlay')).toBeNull();
  });

  it('passes nodes/edges and snapToGrid through to ReactFlow', () => {
    useUIStore.setState({ gridSnapEnabled: true });
    setTab({ nodes: [node('a')] });
    renderCanvas();
    expect(captured.rf.nodes).toHaveLength(1);
    expect(captured.rf.snapToGrid).toBe(true);
    expect(captured.rf.deleteKeyCode).toBe('Delete');
  });
});

// ── minimapNodeColor ────────────────────────────────────────────────────────

describe('minimapNodeColor', () => {
  it('colors note / preset / known-category / unknown-category / missing-definition nodes', () => {
    renderCanvas();
    const color = captured.minimap.nodeColor as (n: any) => string;
    expect(color({ type: 'noteNode', data: {} })).toBe('#FFD700');
    expect(color({ type: 'baseNode', data: { isPreset: true } })).toBe('#D4A017');
    expect(color({ type: 'baseNode', data: { definition: { category: 'Training' } } })).toBe('#F44336');
    // Unknown category falls back to the gray default.
    expect(color({ type: 'baseNode', data: { definition: { category: 'Nonexistent' } } })).toBe('#607D8B');
    // No definition -> category defaults to 'Utility'.
    expect(color({ type: 'baseNode', data: {} })).toBe('#607D8B');
  });
});

// ── handleConnect ───────────────────────────────────────────────────────────

describe('handleConnect', () => {
  it('converts a trigger connection into a triggerEdge and leaves other edges untouched', () => {
    // A pre-existing unrelated edge exercises the `: e` (no-match) branch.
    setTab({
      nodes: [node('start', { type: 'start' }), node('b'), node('c'), node('d')],
      edges: [{ id: 'other', source: 'c', target: 'd' }],
    });
    renderCanvas();
    act(() => {
      captured.rf.onConnect({ source: 'start', target: 'b', sourceHandle: 'trigger', targetHandle: '__trigger' });
    });
    const edges = activeTab().edges;
    const triggerEdge = edges.find((e) => e.source === 'start');
    expect(triggerEdge?.type).toBe('triggerEdge');
    expect(triggerEdge?.targetHandle).toBe('__trigger');
    expect((triggerEdge!.data as any).type).toBe('trigger');
    // The unrelated edge is preserved as-is.
    const other = edges.find((e) => e.id === 'other');
    expect(other?.type).toBeUndefined();
  });

  it('colors a data edge by the source port data type', () => {
    // The def lookup keys on `node.type` (== definition.node_name), so the
    // source node's xyflow type must equal the registered definition name for
    // the color branch to fire.
    useNodeDefStore.setState({ definitions: [makeDef({ node_name: 'Linear' })], presets: [] });
    setTab({
      nodes: [
        node('a', { type: 'Linear', data: { label: 'a', type: 'Linear', params: {}, definition: makeDef({ node_name: 'Linear' }) } }),
        node('b'),
        node('c'),
        node('d'),
      ],
      // A pre-existing unrelated edge exercises the `: e` (no-match) branch in
      // the recolor map.
      edges: [{ id: 'other', source: 'c', target: 'd', style: { stroke: '#123' } }],
    });
    renderCanvas();
    act(() => {
      captured.rf.onConnect({ source: 'a', target: 'b', sourceHandle: 'out', targetHandle: 'in' });
    });
    const edge = activeTab().edges.find((e) => e.source === 'a');
    expect(edge).toBeTruthy();
    // TENSOR -> green (#4CAF50).
    expect((edge!.style as any).stroke).toBe('#4CAF50');
    expect((edge!.style as any).strokeWidth).toBe(2);
    // The unrelated edge keeps its original stroke.
    expect((activeTab().edges.find((e) => e.id === 'other')!.style as any).stroke).toBe('#123');
  });

  it('skips coloring when the definition has no output of that name', () => {
    // Definition is found (node.type === node_name) but has no output "missing".
    useNodeDefStore.setState({ definitions: [makeDef({ node_name: 'Linear' })], presets: [] });
    setTab({
      nodes: [
        node('a', { type: 'Linear', data: { label: 'a', type: 'Linear', params: {}, definition: makeDef({ node_name: 'Linear' }) } }),
        node('b'),
      ],
    });
    renderCanvas();
    act(() => {
      captured.rf.onConnect({ source: 'a', target: 'b', sourceHandle: 'missing', targetHandle: 'in' });
    });
    const edge = activeTab().edges.find((e) => e.source === 'a');
    // Edge added by store with the default gray stroke; not recolored.
    expect((edge!.style as any).stroke).toBe('#555');
  });

  it('skips coloring when the source node id is unknown', () => {
    setTab({ nodes: [node('a'), node('b')] });
    renderCanvas();
    act(() => {
      captured.rf.onConnect({ source: 'ghost', target: 'b', sourceHandle: 'out', targetHandle: 'in' });
    });
    const edge = activeTab().edges.find((e) => e.source === 'ghost');
    expect((edge!.style as any).stroke).toBe('#555');
  });

  it('does nothing extra when sourceHandle is missing (no color branch)', () => {
    setTab({ nodes: [node('a'), node('b')] });
    renderCanvas();
    act(() => {
      captured.rf.onConnect({ source: 'a', target: 'b', sourceHandle: null, targetHandle: 'in' });
    });
    expect(activeTab().edges).toHaveLength(1);
  });
});

// ── handleIsValidConnection ─────────────────────────────────────────────────

describe('handleIsValidConnection', () => {
  function valid(conn: any) {
    return captured.rf.isValidConnection(conn) as boolean;
  }

  beforeEach(() => {
    setTab({
      nodes: [
        node('src', { data: { label: 'src', type: 'Linear', params: {}, definition: makeDef() } }),
        node('dst', { data: { label: 'dst', type: 'Linear', params: {}, definition: makeDef({ inputs: [{ name: 'in', data_type: 'TENSOR', description: '', optional: false }] }) } }),
        node('note', { type: 'noteNode', data: { label: 'n', type: 'note', params: {} } }),
        node('nodef', { data: { label: 'x', type: 'X', params: {} } }),
      ],
    });
    renderCanvas();
  });

  it('rejects missing source/target and self-loops', () => {
    expect(valid({ source: null, target: 'dst' })).toBe(false);
    expect(valid({ source: 'src', target: null })).toBe(false);
    expect(valid({ source: 'src', target: 'src' })).toBe(false);
  });

  it('rejects connections to/from note nodes', () => {
    expect(valid({ source: 'note', target: 'dst', sourceHandle: 'out', targetHandle: 'in' })).toBe(false);
    expect(valid({ source: 'src', target: 'note', sourceHandle: 'out', targetHandle: 'in' })).toBe(false);
  });

  it('allows a trigger source only into a __trigger handle', () => {
    expect(valid({ source: 'src', target: 'dst', sourceHandle: 'trigger', targetHandle: '__trigger' })).toBe(true);
    expect(valid({ source: 'src', target: 'dst', sourceHandle: 'trigger', targetHandle: 'in' })).toBe(false);
  });

  it('validates compatible data types via isValidConnection', () => {
    expect(valid({ source: 'src', target: 'dst', sourceHandle: 'out', targetHandle: 'in' })).toBe(true);
  });

  it('allows the connection when a node has no definition', () => {
    expect(valid({ source: 'nodef', target: 'dst', sourceHandle: 'out', targetHandle: 'in' })).toBe(true);
  });

  it('allows the connection when source/target nodes are not found (handles present)', () => {
    // Both handles set, not a trigger, but neither node id exists in the tab.
    expect(valid({ source: 'ghost1', target: 'ghost2', sourceHandle: 'out', targetHandle: 'in' })).toBe(true);
  });

  it('allows the connection when the named ports are not found on the definitions', () => {
    expect(valid({ source: 'src', target: 'dst', sourceHandle: 'nope', targetHandle: 'in' })).toBe(true);
  });

  it('allows when handles are absent (final return true)', () => {
    expect(valid({ source: 'src', target: 'dst', sourceHandle: null, targetHandle: null })).toBe(true);
  });
});

// ── onConnectStart / onConnectEnd ───────────────────────────────────────────

describe('onConnectStart / onConnectEnd', () => {
  it('sets draggingSourceType from the source port and clears it on end', () => {
    setTab({ nodes: [node('a')] });
    renderCanvas();
    act(() => {
      captured.rf.onConnectStart(null, { nodeId: 'a', handleId: 'out', handleType: 'source' });
    });
    expect(useUIStore.getState().draggingSourceType).toBe('TENSOR');
    act(() => captured.rf.onConnectEnd());
    expect(useUIStore.getState().draggingSourceType).toBeNull();
  });

  it('ignores connect-start that is not a source handle, or has a missing node/output', () => {
    setTab({ nodes: [node('a')] });
    renderCanvas();
    // Not a source handle -> no change.
    act(() => captured.rf.onConnectStart(null, { nodeId: 'a', handleId: 'out', handleType: 'target' }));
    expect(useUIStore.getState().draggingSourceType).toBeNull();
    // Unknown node id.
    act(() => captured.rf.onConnectStart(null, { nodeId: 'ghost', handleId: 'out', handleType: 'source' }));
    expect(useUIStore.getState().draggingSourceType).toBeNull();
    // Known node but unknown output name.
    act(() => captured.rf.onConnectStart(null, { nodeId: 'a', handleId: 'nope', handleType: 'source' }));
    expect(useUIStore.getState().draggingSourceType).toBeNull();
  });
});

// ── Reconnect handlers ──────────────────────────────────────────────────────

describe('reconnect handlers', () => {
  const oldEdge: Edge = { id: 'e1', source: 'a', target: 'b', sourceHandle: 'out', targetHandle: 'in' };

  it('replaces an edge on a completed reconnect', () => {
    setTab({ nodes: [node('a'), node('b'), node('c')], edges: [oldEdge] });
    renderCanvas();
    act(() => captured.rf.onReconnectStart(null, oldEdge));
    act(() =>
      captured.rf.onReconnect(oldEdge, { source: 'a', target: 'c', sourceHandle: 'out', targetHandle: 'in' }),
    );
    const edges = activeTab().edges;
    expect(edges).toHaveLength(1);
    expect(edges[0].target).toBe('c');
    expect(edges[0].id).toBe('e1');
  });

  it('reconnect with null handles falls back to undefined handles', () => {
    setTab({ nodes: [node('a'), node('b'), node('c')], edges: [oldEdge] });
    renderCanvas();
    act(() => captured.rf.onReconnect(oldEdge, { source: 'a', target: 'c', sourceHandle: null, targetHandle: null }));
    const edges = activeTab().edges;
    expect(edges[0].sourceHandle).toBeUndefined();
    expect(edges[0].targetHandle).toBeUndefined();
  });

  it('deletes the edge when reconnect ends on empty space (ref still matches)', () => {
    setTab({ nodes: [node('a'), node('b')], edges: [oldEdge] });
    renderCanvas();
    act(() => captured.rf.onReconnectStart(null, oldEdge));
    act(() => captured.rf.onReconnectEnd(null, oldEdge));
    expect(activeTab().edges).toHaveLength(0);
  });

  it('does NOT delete on reconnect end when the edge was already reconnected (ref cleared)', () => {
    setTab({ nodes: [node('a'), node('b'), node('c')], edges: [oldEdge] });
    renderCanvas();
    act(() => captured.rf.onReconnectStart(null, oldEdge));
    act(() => captured.rf.onReconnect(oldEdge, { source: 'a', target: 'c', sourceHandle: 'out', targetHandle: 'in' }));
    // ref was cleared by onReconnect; end should be a no-op delete.
    act(() => captured.rf.onReconnectEnd(null, oldEdge));
    expect(activeTab().edges).toHaveLength(1);
  });

  it('onReconnect bails out when the active tab cannot be found', () => {
    setTab({ nodes: [node('a'), node('b'), node('c')], edges: [oldEdge] });
    const { unmount } = renderCanvas();
    const handler = captured.rf.onReconnect;
    // Unmount first so making the store invalid does not re-render the canvas
    // (its render reads activeTab via a non-null assertion).
    unmount();
    useTabStore.setState({ activeTabId: 'no-such-tab' });
    // The handler reads getState() fresh; with no matching tab it returns early.
    expect(() =>
      handler(oldEdge, { source: 'a', target: 'c', sourceHandle: 'out', targetHandle: 'in' }),
    ).not.toThrow();
  });

  it('onReconnectEnd bails out when the active tab cannot be found', () => {
    setTab({ nodes: [node('a'), node('b')], edges: [oldEdge] });
    const { unmount } = renderCanvas();
    const start = captured.rf.onReconnectStart;
    const end = captured.rf.onReconnectEnd;
    // Mark the edge as reconnecting so the ref matches and we reach the
    // tab lookup inside onReconnectEnd.
    act(() => start(null, oldEdge));
    unmount();
    useTabStore.setState({ activeTabId: 'no-such-tab' });
    expect(() => end(null, oldEdge)).not.toThrow();
  });
});

// ── Node / edge / pane click handlers ───────────────────────────────────────

describe('click handlers', () => {
  it('selects a node on click', () => {
    setTab({ nodes: [node('a')] });
    renderCanvas();
    act(() => captured.rf.onNodeClick({} as any, { id: 'a' }));
    expect(activeTab().selectedNodeId).toBe('a');
  });

  it('clears selection and menus on pane click', () => {
    setTab({ nodes: [node('a')] });
    renderCanvas();
    act(() => captured.rf.onNodeClick({} as any, { id: 'a' }));
    act(() => captured.rf.onPaneClick());
    expect(activeTab().selectedNodeId).toBeNull();
  });

  it('opens a data tooltip on edge click when a summary exists', () => {
    setTab({
      nodes: [node('a', { data: { label: 'A', type: 'Linear', params: {}, definition: makeDef() } }), node('b')],
      edges: [{ id: 'e1', source: 'a', target: 'b', sourceHandle: 'out' }],
      outputSummaries: { a: { out: { type: 'Tensor', shape: [2] } } },
    });
    renderCanvas();
    act(() =>
      captured.rf.onEdgeClick({ clientX: 100, clientY: 100 } as any, {
        id: 'e1', source: 'a', target: 'b', sourceHandle: 'out',
      }),
    );
    // EdgeDataTooltip renders the source -> target title.
    expect(screen.getByText(/A/)).toBeTruthy();
    expect(screen.getByText('Type')).toBeTruthy();
    // Pressing Escape fires the tooltip's onClose -> setEdgeTooltip(null).
    act(() => {
      fireEvent.keyDown(document, { key: 'Escape' });
    });
    expect(screen.queryByText('Type')).toBeNull();
  });

  it('uses id-slice fallbacks for labels when nodes are missing/unlabeled', () => {
    setTab({
      nodes: [],
      edges: [{ id: 'e1', source: 'aaaaaaaaaaaa', target: 'bbbbbbbbbbbb', sourceHandle: 'out' }],
      outputSummaries: { aaaaaaaaaaaa: { out: { type: 'Tensor' } } },
    });
    renderCanvas();
    act(() =>
      captured.rf.onEdgeClick({ clientX: 10, clientY: 10 } as any, {
        id: 'e1', source: 'aaaaaaaaaaaa', target: 'bbbbbbbbbbbb', sourceHandle: 'out',
      }),
    );
    // Labels fall back to the first 8 chars of the ids.
    expect(screen.getByText(/aaaaaaaa/)).toBeTruthy();
  });

  it('closes (no tooltip) on edge click when no summary is available', () => {
    setTab({
      nodes: [node('a'), node('b')],
      edges: [{ id: 'e1', source: 'a', target: 'b', sourceHandle: 'out' }],
      outputSummaries: {},
    });
    renderCanvas();
    act(() =>
      captured.rf.onEdgeClick({ clientX: 5, clientY: 5 } as any, {
        id: 'e1', source: 'a', target: 'b', sourceHandle: 'out',
      }),
    );
    expect(screen.queryByText('Type')).toBeNull();
  });

  it('treats an edge with no sourceHandle as having an empty-string handle', () => {
    setTab({
      nodes: [node('a'), node('b')],
      edges: [{ id: 'e1', source: 'a', target: 'b' }],
      outputSummaries: { a: { '': { type: 'Scalar' } } },
    });
    renderCanvas();
    act(() =>
      captured.rf.onEdgeClick({ clientX: 5, clientY: 5 } as any, {
        id: 'e1', source: 'a', target: 'b', sourceHandle: null,
      }),
    );
    expect(screen.getByText('Type')).toBeTruthy();
  });

  it('onMoveStart / onMoveEnd toggle canvas panning', () => {
    renderCanvas();
    act(() => captured.rf.onMoveStart());
    expect(useUIStore.getState().isCanvasPanning).toBe(true);
    act(() => captured.rf.onMoveEnd());
    expect(useUIStore.getState().isCanvasPanning).toBe(false);
  });
});

// ── Context menus ───────────────────────────────────────────────────────────

describe('context menus', () => {
  it('opens a node context menu and selects the node', () => {
    setTab({ nodes: [node('a')] });
    renderCanvas();
    act(() =>
      captured.rf.onNodeContextMenu({ preventDefault: vi.fn(), clientX: 30, clientY: 40 } as any, { id: 'a' }),
    );
    expect(activeTab().selectedNodeId).toBe('a');
    // Node menu shows Rename/Duplicate/Delete (localized).
    expect(screen.getByText(useI18n.getState().t('contextMenu.rename'))).toBeTruthy();
    expect(screen.getByText(useI18n.getState().t('contextMenu.delete'))).toBeTruthy();
  });

  it('renders the note menu variant for note nodes', () => {
    setTab({
      nodes: [node('n', { type: 'noteNode', data: { label: 'n', type: 'note', params: {}, boundToNodeId: null } })],
    });
    renderCanvas();
    act(() =>
      captured.rf.onNodeContextMenu({ preventDefault: vi.fn(), clientX: 1, clientY: 1 } as any, { id: 'n' }),
    );
    // Note menu has a Bind item (unbound note).
    expect(screen.getByText(useI18n.getState().t('note.bind'))).toBeTruthy();
  });

  it('closes the node context menu via its onClose', () => {
    setTab({ nodes: [node('a')] });
    renderCanvas();
    act(() =>
      captured.rf.onNodeContextMenu({ preventDefault: vi.fn(), clientX: 1, clientY: 1 } as any, { id: 'a' }),
    );
    const deleteBtn = screen.getByText(useI18n.getState().t('contextMenu.delete'));
    expect(deleteBtn).toBeTruthy();
    // Clicking a menu item invokes its action and then onClose -> menu unmounts.
    act(() => fireEvent.click(deleteBtn));
    expect(screen.queryByText(useI18n.getState().t('contextMenu.delete'))).toBeNull();
    // The delete action removed node 'a'.
    expect(activeTab().nodes).toHaveLength(0);
  });

  it('opens a pane context menu via screenToFlowPosition', () => {
    renderCanvas();
    act(() =>
      captured.rf.onPaneContextMenu({ preventDefault: vi.fn(), clientX: 7, clientY: 9 } as any),
    );
    expect(screen.getByTestId('pane-menu')).toBeTruthy();
    // Closing it removes the menu.
    act(() => fireEvent.click(screen.getByTestId('pane-menu')));
    expect(screen.queryByTestId('pane-menu')).toBeNull();
  });
});

// ── handleRename (dialog prompt) ────────────────────────────────────────────

describe('handleRename', () => {
  it('renames the node when the prompt resolves a non-empty value', async () => {
    setTab({ nodes: [node('a', { data: { label: 'Old', type: 'Linear', params: {}, definition: makeDef() } })] });
    renderCanvas();
    act(() =>
      captured.rf.onNodeContextMenu({ preventDefault: vi.fn(), clientX: 1, clientY: 1 } as any, { id: 'a' }),
    );
    const renameBtn = screen.getByText(useI18n.getState().t('contextMenu.rename'));
    await act(async () => {
      fireEvent.click(renameBtn);
    });
    // The dialog store now holds an open prompt; resolve it with a new label.
    expect(useDialogStore.getState().active?.kind).toBe('prompt');
    await act(async () => {
      useDialogStore.getState().close('  Renamed  ');
    });
    expect(activeTab().nodes[0].data.label).toBe('Renamed');
  });

  it('does not rename when the prompt is cancelled (null)', async () => {
    setTab({ nodes: [node('a', { data: { label: 'Old', type: 'Linear', params: {}, definition: makeDef() } })] });
    renderCanvas();
    act(() =>
      captured.rf.onNodeContextMenu({ preventDefault: vi.fn(), clientX: 1, clientY: 1 } as any, { id: 'a' }),
    );
    await act(async () => {
      fireEvent.click(screen.getByText(useI18n.getState().t('contextMenu.rename')));
    });
    await act(async () => {
      useDialogStore.getState().close(null);
    });
    expect(activeTab().nodes[0].data.label).toBe('Old');
  });

  it('does not rename when the prompt resolves only whitespace', async () => {
    setTab({ nodes: [node('a', { data: { label: 'Old', type: 'Linear', params: {}, definition: makeDef() } })] });
    renderCanvas();
    act(() =>
      captured.rf.onNodeContextMenu({ preventDefault: vi.fn(), clientX: 1, clientY: 1 } as any, { id: 'a' }),
    );
    await act(async () => {
      fireEvent.click(screen.getByText(useI18n.getState().t('contextMenu.rename')));
    });
    await act(async () => {
      useDialogStore.getState().close('   ');
    });
    expect(activeTab().nodes[0].data.label).toBe('Old');
  });

  it('uses an empty default when renaming a node that no longer exists', async () => {
    setTab({ nodes: [node('a')] });
    renderCanvas();
    act(() =>
      captured.rf.onNodeContextMenu({ preventDefault: vi.fn(), clientX: 1, clientY: 1 } as any, { id: 'a' }),
    );
    // Remove the node before resolving the rename prompt; flush the re-render
    // so the menu rebinds handleRename to the now-empty node list.
    act(() => {
      setTab({ nodes: [] });
    });
    await act(async () => {
      fireEvent.click(screen.getByText(useI18n.getState().t('contextMenu.rename')));
    });
    expect(useDialogStore.getState().active?.kind).toBe('prompt');
    // defaultValue should be '' since node is gone.
    expect((useDialogStore.getState().active as any).defaultValue).toBe('');
    await act(async () => {
      useDialogStore.getState().close('whatever');
    });
  });
});

// ── DnD passthrough ─────────────────────────────────────────────────────────

describe('drag-and-drop passthrough', () => {
  it('onDragOver / onDrop come from useDragAndDrop and add a node on drop', () => {
    renderCanvas();
    const over = { preventDefault: vi.fn(), dataTransfer: { dropEffect: '' } };
    act(() => captured.rf.onDragOver(over as any));
    expect(over.preventDefault).toHaveBeenCalled();

    const drop = {
      preventDefault: vi.fn(),
      clientX: 0,
      clientY: 0,
      dataTransfer: { getData: (k: string) => (k === 'application/codefyui-node' ? 'Linear' : '') },
    };
    act(() => captured.rf.onDrop(drop as any));
    expect(activeTab().nodes.some((n) => n.data.type === 'Linear')).toBe(true);
  });
});

// ── Double-click pane -> quick search (the timer + DOM listener effect) ──────

describe('pane double-click quick search', () => {
  it('opens quick search when double-clicking the pane (not inside a node)', () => {
    vi.useFakeTimers();
    renderCanvas();
    // Flush the 100ms setTimeout that attaches the dblclick listener.
    act(() => {
      vi.advanceTimersByTime(100);
    });
    const pane = screen.getByTestId('pane');
    act(() => {
      fireEvent.dblClick(pane, { clientX: 12, clientY: 34 });
    });
    expect(screen.getByTestId('quick-search')).toBeTruthy();
  });

  it('ignores a double-click that originated inside a node', () => {
    vi.useFakeTimers();
    renderCanvas();
    act(() => {
      vi.advanceTimersByTime(100);
    });
    const pane = screen.getByTestId('pane');
    // Build a child marked as a react-flow node; dispatch a dblclick whose
    // target.closest('.react-flow__node') is truthy.
    const nodeEl = document.createElement('div');
    nodeEl.className = 'react-flow__node';
    pane.appendChild(nodeEl);
    act(() => {
      fireEvent.dblClick(nodeEl, { clientX: 1, clientY: 1 });
    });
    expect(screen.queryByTestId('quick-search')).toBeNull();
  });

  it('closes quick search via its onClose', () => {
    vi.useFakeTimers();
    renderCanvas();
    act(() => vi.advanceTimersByTime(100));
    act(() => fireEvent.dblClick(screen.getByTestId('pane'), { clientX: 1, clientY: 1 }));
    const qs = screen.getByTestId('quick-search');
    act(() => fireEvent.click(qs));
    expect(screen.queryByTestId('quick-search')).toBeNull();
  });

  it('handles the pane being absent at attach and cleanup time (both `if (pane)` false)', () => {
    vi.useFakeTimers();
    renderPane.value = false; // stub renders no .react-flow__pane
    const { unmount } = renderCanvas();
    // Attach-time: the timeout callback finds no pane -> skips addEventListener.
    act(() => vi.advanceTimersByTime(100));
    expect(screen.queryByTestId('pane')).toBeNull();
    expect(screen.queryByTestId('quick-search')).toBeNull();
    // Cleanup-time: unmount with no pane -> skips removeEventListener.
    expect(() => unmount()).not.toThrow();
  });
});

// ── Grid-snap effect ────────────────────────────────────────────────────────

describe('grid-snap effect', () => {
  it('snaps existing node positions to the grid when grid snap is enabled at mount', () => {
    // gridSnapEnabled true + an off-grid node -> effect rounds positions to 24.
    useUIStore.setState({ gridSnapEnabled: true });
    setTab({ nodes: [node('a', { position: { x: 10, y: 50 } })] });
    renderCanvas();
    const n = activeTab().nodes[0];
    expect(n.position.x).toBe(0); // round(10/24)*24 = 0
    expect(n.position.y).toBe(48); // round(50/24)*24 = 48
  });

  it('does not reposition nodes that are already on the grid (changed === false)', () => {
    useUIStore.setState({ gridSnapEnabled: true });
    setTab({ nodes: [node('a', { position: { x: 24, y: 48 } })] });
    renderCanvas();
    const n = activeTab().nodes[0];
    expect(n.position).toEqual({ x: 24, y: 48 });
  });

  it('does nothing when grid snap is disabled', () => {
    useUIStore.setState({ gridSnapEnabled: false });
    setTab({ nodes: [node('a', { position: { x: 10, y: 50 } })] });
    renderCanvas();
    expect(activeTab().nodes[0].position).toEqual({ x: 10, y: 50 });
  });
});

// ── store-driven onNodesChange / onEdgesChange passthrough ───────────────────

describe('change handlers passthrough', () => {
  it('forwards node changes to the store (onNodesChange)', () => {
    setTab({ nodes: [node('a', { position: { x: 0, y: 0 } })] });
    renderCanvas();
    act(() =>
      captured.rf.onNodesChange([{ id: 'a', type: 'position', position: { x: 5, y: 6 } }]),
    );
    expect(activeTab().nodes[0].position).toEqual({ x: 5, y: 6 });
  });

  it('forwards edge changes to the store (onEdgesChange)', () => {
    setTab({
      nodes: [node('a'), node('b')],
      edges: [{ id: 'e1', source: 'a', target: 'b' }],
    });
    renderCanvas();
    act(() => captured.rf.onEdgesChange([{ id: 'e1', type: 'remove' }]));
    expect(activeTab().edges).toHaveLength(0);
  });
});
