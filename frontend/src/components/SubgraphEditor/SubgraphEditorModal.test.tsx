import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, within, act } from '@testing-library/react';

import { useTabStore } from '../../store/tabStore';
import { useToastStore } from '../../store/toastStore';
import { useI18n } from '../../i18n';

// ──────────────────────────────────────────────────────────────────────────
// Controllable @xyflow/react mock.
//
// We keep ReactFlowProvider / Handle / Background / Controls / applyNode*
// REAL so children still render, but replace <ReactFlow> with a stub that
// (a) captures the props the component passes (so tests can invoke the canvas
// handlers — onConnect, onNodesChange, isValidConnection, onNodesDelete, …
// which are otherwise awkward to trigger through real DnD in jsdom), and
// (b) renders its `children` plus a node list so the JSX still mounts.
//
// `useReactFlow` is overridden so screenToFlowPosition is deterministic and
// fitView is a recorded no-op (jsdom has no real flow viewport).
// ──────────────────────────────────────────────────────────────────────────

let lastFlowProps: any = null;
const fitViewSpy = vi.fn();
const screenToFlowPositionSpy = vi.fn((p: { x: number; y: number }) => ({ x: p.x, y: p.y }));

vi.mock('@xyflow/react', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@xyflow/react')>();
  const React = await import('react');
  return {
    ...actual,
    useReactFlow: () => ({
      screenToFlowPosition: screenToFlowPositionSpy,
      fitView: fitViewSpy,
    }),
    ReactFlow: (props: any) => {
      lastFlowProps = props;
      return React.createElement(
        'div',
        { 'data-testid': 'reactflow-stub' },
        React.createElement(
          'div',
          { 'data-testid': 'rf-node-count' },
          String((props.nodes ?? []).length),
        ),
        props.children,
      );
    },
  };
});

import { SubgraphEditorModal } from './SubgraphEditorModal';

// ── Helpers ────────────────────────────────────────────────────────────────

/** A non-empty, valid GraphSpec v2 with Input → Linear → Output. */
function validGraphJson(): string {
  return JSON.stringify({
    version: 2,
    nodes: [
      {
        id: 'in1',
        type: 'Input',
        ports: [{ id: 'ip1', name: 'x' }],
        position: { x: 0, y: 0 },
      },
      {
        id: 'lin1',
        type: 'Linear',
        params: { in_features: 512, out_features: 10 },
        position: { x: 0, y: 100 },
      },
      {
        id: 'merge1',
        type: 'Concat',
        params: { dim: 1 },
        position: { x: 0, y: 200 },
      },
      {
        id: 'out1',
        type: 'Output',
        ports: [{ id: 'op1', name: 'y' }],
        position: { x: 0, y: 300 },
      },
    ],
    edges: [
      { id: 'e1', source: 'in1', sourceHandle: 'ip1', target: 'lin1', targetHandle: null },
      { id: 'e2', source: 'lin1', sourceHandle: null, target: 'merge1', targetHandle: null },
      { id: 'e3', source: 'merge1', sourceHandle: null, target: 'out1', targetHandle: 'op1' },
    ],
  });
}

/**
 * Install a tab whose active node carries `layersJson` in params.layers, and
 * open the subgraph modal on it. Returns the node id.
 */
function setupOpenModal(layersJson: string | undefined, opts?: { paramsUndefined?: boolean }): string {
  const tabId = 't1';
  const nodeId = 'sg-node';
  const params = opts?.paramsUndefined
    ? undefined
    : layersJson === undefined
      ? {}
      : { layers: layersJson };
  useTabStore.setState({
    activeTabId: tabId,
    tabs: [
      {
        id: tabId,
        name: 'Tab 1',
        nodes: [
          {
            id: nodeId,
            type: 'baseNode',
            position: { x: 0, y: 0 },
            data: {
              label: 'SequentialModel',
              type: 'SequentialModel',
              params,
              executionStatus: 'idle',
            },
          },
        ] as any,
        edges: [],
        selectedNodeId: null,
        presetModalNodeId: null,
        subgraphModalNodeId: nodeId,
        undoStack: [],
        redoStack: [],
        dirtyNodeIds: new Set(),
        status: 'idle',
        logs: [],
        ws: { disconnect: vi.fn() } as any,
        outputSummaries: {},
        recordOutputs: true,
        lastRunId: null,
        activeSegment: null,
        segmentGroups: [],
        verboseMode: false,
        graphId: 'g1',
        weightsPersistent: true,
        backwardMode: false,
        autoBackward: false,
      },
    ] as any,
  });
  return nodeId;
}

describe('SubgraphEditorModal', () => {
  beforeEach(() => {
    useI18n.setState({ locale: 'en' });
    useToastStore.setState({ toasts: [] });
    lastFlowProps = null;
    fitViewSpy.mockClear();
    screenToFlowPositionSpy.mockClear();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.runOnlyPendingTimers();
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.clearAllMocks();
  });

  // ── Main export gating ────────────────────────────────────────────────────

  it('returns null when there is no subgraphModalNodeId', () => {
    setupOpenModal(validGraphJson());
    // Clear the modal node id → main export early-returns null.
    useTabStore.setState({
      tabs: useTabStore.getState().tabs.map((t) => ({ ...t, subgraphModalNodeId: null })) as any,
    });
    const { container } = render(<SubgraphEditorModal />);
    expect(container.firstChild).toBeNull();
  });

  it('returns null when the referenced node is not found', () => {
    setupOpenModal(validGraphJson());
    // Point modal at a node id that does not exist.
    useTabStore.setState({
      tabs: useTabStore.getState().tabs.map((t) => ({ ...t, subgraphModalNodeId: 'missing' })) as any,
    });
    const { container } = render(<SubgraphEditorModal />);
    expect(container.firstChild).toBeNull();
  });

  it('renders the editor with the title and layer count for a valid graph', () => {
    setupOpenModal(validGraphJson());
    render(<SubgraphEditorModal />);
    expect(screen.getByText('Model Architecture Editor')).toBeTruthy();
    // 4 nodes in the valid graph.
    expect(screen.getByText('4 layers')).toBeTruthy();
  });

  it('falls back to an empty graph (Input+Output) when params.layers is missing', () => {
    // node.data.params exists but has no `layers` → `?? '{}'` → graphToFlow
    // returns emptyGraph (0 spec nodes) → initial uses emptyGraph() (2 nodes).
    setupOpenModal(undefined);
    render(<SubgraphEditorModal />);
    expect(screen.getByText('2 layers')).toBeTruthy();
  });

  it('falls back to {} when node.data.params is undefined', () => {
    // Covers `node.data.params?.layers` optional chaining + `?? '{}'`.
    setupOpenModal(undefined, { paramsUndefined: true });
    render(<SubgraphEditorModal />);
    expect(screen.getByText('2 layers')).toBeTruthy();
  });

  it('uses emptyGraph when graphToFlow parses to zero nodes (initial useMemo)', () => {
    // A *valid* version-2 spec with an empty node list → graphToFlow returns
    // { nodes: [], edges: [] } (length 0) → `if (parsed.nodes.length === 0)`
    // true branch → emptyGraph() (2 boundary nodes).
    setupOpenModal(JSON.stringify({ version: 2, nodes: [], edges: [] }));
    render(<SubgraphEditorModal />);
    expect(screen.getByText('2 layers')).toBeTruthy();
  });

  // ── Palette: search filter, grouping, category labels ─────────────────────

  it('filters the layer palette via the search box and shows the Merge category label', () => {
    setupOpenModal(validGraphJson());
    render(<SubgraphEditorModal />);
    const search = screen.getByPlaceholderText('Search layers...') as HTMLInputElement;

    // Default: all categories visible, including the localized Merge group.
    expect(screen.getByText('Merge')).toBeTruthy();
    expect(screen.getAllByText('Conv2d').length).toBeGreaterThan(0);

    // Filter to "conv" → only Conv* remain, Merge group gone.
    fireEvent.change(search, { target: { value: 'conv' } });
    expect(screen.getAllByText('Conv2d').length).toBeGreaterThan(0);
    expect(screen.queryByText('Add')).toBeNull();

    // Clearing returns to the full list (search.trim() falsy branch).
    fireEvent.change(search, { target: { value: '   ' } });
    expect(screen.getByText('Merge')).toBeTruthy();
  });

  it('shows a parameter-count badge for layers with params and hovers a palette item', () => {
    setupOpenModal(validGraphJson());
    render(<SubgraphEditorModal />);
    const search = screen.getByPlaceholderText('Search layers...');
    // Linear has 2 params → "2p" badge.
    fireEvent.change(search, { target: { value: 'Linear' } });
    expect(screen.getByText('2p')).toBeTruthy();

    // The draggable palette item is the <span> label's parent div ("Linear"
    // also appears as the category header, so pick the draggable ancestor).
    const linearLabel = screen
      .getAllByText('Linear')
      .find((el) => el.tagName === 'SPAN')!;
    const linearItem = linearLabel.closest('[draggable]') as HTMLElement;
    // Hover on/off toggles background (covers onMouseEnter / onMouseLeave).
    fireEvent.mouseEnter(linearItem);
    expect(linearItem.style.background).toContain('rgb(42, 42, 42)');
    fireEvent.mouseLeave(linearItem);
    expect(linearItem.style.background).toBe('transparent');

    // ReLU has zero params → no badge.
    fireEvent.change(search, { target: { value: 'ReLU' } });
    expect(screen.queryByText('0p')).toBeNull();
  });

  it('LayerPaletteItem.handleDragStart sets the drag payload and effect', () => {
    setupOpenModal(validGraphJson());
    render(<SubgraphEditorModal />);
    const search = screen.getByPlaceholderText('Search layers...');
    fireEvent.change(search, { target: { value: 'Flatten' } });
    const item = screen.getByText('Flatten').closest('div')!;

    const setData = vi.fn();
    const dt: any = { setData, effectAllowed: '' };
    fireEvent.dragStart(item, { dataTransfer: dt });
    expect(setData).toHaveBeenCalledWith('application/subgraph-layer', 'Flatten');
    expect(dt.effectAllowed).toBe('move');
  });

  // ── Canvas handlers via captured ReactFlow props ──────────────────────────

  it('onNodesChange / onEdgesChange apply changes through the store setters', () => {
    setupOpenModal(validGraphJson());
    render(<SubgraphEditorModal />);
    expect(lastFlowProps).toBeTruthy();

    // Remove the Linear node via a node change.
    act(() => {
      lastFlowProps.onNodesChange([{ type: 'remove', id: 'lin1' }]);
    });
    // 4 → 3 layers.
    expect(screen.getByText('3 layers')).toBeTruthy();

    // Remove an edge via an edge change (just exercises the callback).
    act(() => {
      lastFlowProps.onEdgesChange([{ type: 'remove', id: 'e1' }]);
    });
  });

  it('onNodeClick selects a layer node → ParamEditor; onPaneClick clears selection', () => {
    setupOpenModal(validGraphJson());
    render(<SubgraphEditorModal />);

    // Select the Linear (plain) node → ParamEditor with its params.
    act(() => {
      lastFlowProps.onNodeClick({}, { id: 'lin1' });
    });
    expect(screen.getByText('in_features')).toBeTruthy();
    expect(screen.getByText('out_features')).toBeTruthy();

    // Clearing selection shows the "No parameters" placeholder.
    act(() => {
      lastFlowProps.onPaneClick();
    });
    // Placeholder appears in the right panel.
    expect(screen.getAllByText('No parameters').length).toBeGreaterThan(0);
  });

  it('selecting a boundary node renders the PortListEditor', () => {
    setupOpenModal(validGraphJson());
    render(<SubgraphEditorModal />);
    act(() => {
      lastFlowProps.onNodeClick({}, { id: 'in1' });
    });
    // PortListEditor header: "Input — Ports".
    expect(screen.getByText(/Ports/)).toBeTruthy();
    expect(screen.getByDisplayValue('x')).toBeTruthy();
  });

  it('onConnect appends an edge with default styling and null handles fall back to undefined', () => {
    setupOpenModal(validGraphJson());
    render(<SubgraphEditorModal />);
    act(() => {
      // null source/target handles exercise the `?? undefined` fallbacks.
      lastFlowProps.onConnect({
        source: 'in1',
        sourceHandle: null,
        target: 'lin1',
        targetHandle: null,
      });
    });
    // Edge added to the flow → reflected in captured props on re-render.
    const targets = (lastFlowProps.edges as any[]).filter((e) => e.target === 'lin1');
    expect(targets.length).toBeGreaterThanOrEqual(2);
    // The newly-added edge is the only one whose ids are NOT the seed edges.
    const added = targets.find((e) => !['e1', 'e2', 'e3'].includes(e.id));
    expect(added.sourceHandle).toBeUndefined();
    expect(added.targetHandle).toBeUndefined();
    expect(added.style.stroke).toBe('#555');
  });

  it('onConnect preserves explicit handles when provided', () => {
    setupOpenModal(validGraphJson());
    render(<SubgraphEditorModal />);
    act(() => {
      lastFlowProps.onConnect({
        source: 'lin1',
        sourceHandle: 'sh',
        target: 'merge1',
        targetHandle: 'th',
      });
    });
    const added = (lastFlowProps.edges as any[]).find(
      (e) => e.sourceHandle === 'sh' && e.targetHandle === 'th',
    );
    expect(added).toBeTruthy();
  });

  it('isValidConnection covers all branches', () => {
    setupOpenModal(validGraphJson());
    render(<SubgraphEditorModal />);
    const isValid = lastFlowProps.isValidConnection as (c: any) => boolean;

    // target not found → false.
    expect(isValid({ source: 'in1', target: 'nope' })).toBe(false);

    // plain layer 'lin1' already has 1 incoming (e1) → false.
    expect(isValid({ source: 'in1', target: 'lin1' })).toBe(false);

    // merge node 'merge1' is exempt from the single-incoming rule → true.
    expect(isValid({ source: 'lin1', target: 'merge1' })).toBe(true);

    // A freshly-dropped plain layer with NO incoming edges → allowed (covers
    // the `existing.length >= 1` false path for non-merge/non-boundary nodes).
    act(() => {
      lastFlowProps.onDrop({
        preventDefault: vi.fn(),
        clientX: 9,
        clientY: 9,
        dataTransfer: { getData: () => 'ReLU' },
      });
    });
    const fresh = (lastFlowProps.nodes as any[]).find((n) => n.data.layerType === 'ReLU');
    const isValid2 = lastFlowProps.isValidConnection as (c: any) => boolean;
    expect(isValid2({ source: 'in1', target: fresh.id })).toBe(true);

    // boundary Output 'out1' with a fresh handle → true.
    expect(isValid({ source: 'merge1', target: 'out1', targetHandle: 'free' })).toBe(true);

    // Output port that already has an incoming edge for that handle → false.
    expect(isValid({ source: 'merge1', target: 'out1', targetHandle: 'op1' })).toBe(false);
  });

  it('onNodesDelete prunes nodes, their edges, and clears selection', () => {
    setupOpenModal(validGraphJson());
    render(<SubgraphEditorModal />);

    // Select merge1 first so the selection-clearing branch runs.
    act(() => {
      lastFlowProps.onNodeClick({}, { id: 'merge1' });
    });
    act(() => {
      lastFlowProps.onNodesDelete([{ id: 'merge1' }]);
    });
    // 4 → 3 nodes.
    expect(screen.getByText('3 layers')).toBeTruthy();
    // Selection cleared → placeholder shown.
    expect(screen.getAllByText('No parameters').length).toBeGreaterThan(0);
  });

  it('onNodesDelete keeps selection when a different node is deleted', () => {
    setupOpenModal(validGraphJson());
    render(<SubgraphEditorModal />);
    act(() => {
      lastFlowProps.onNodeClick({}, { id: 'lin1' });
    });
    // Delete a node that is NOT selected → selectedNodeId kept (ternary false).
    act(() => {
      lastFlowProps.onNodesDelete([{ id: 'merge1' }]);
    });
    // ParamEditor for lin1 still present.
    expect(screen.getByText('in_features')).toBeTruthy();
  });

  // ── Drag & drop onto the canvas ───────────────────────────────────────────

  it('handleDragOver prevents default and sets the drop effect', () => {
    setupOpenModal(validGraphJson());
    render(<SubgraphEditorModal />);
    const preventDefault = vi.fn();
    const dt: any = { dropEffect: '' };
    act(() => {
      lastFlowProps.onDragOver({ preventDefault, dataTransfer: dt });
    });
    expect(preventDefault).toHaveBeenCalled();
    expect(dt.dropEffect).toBe('move');
  });

  it('handleDrop with a known layer type adds a node at the mapped position', () => {
    setupOpenModal(validGraphJson());
    render(<SubgraphEditorModal />);
    const preventDefault = vi.fn();
    act(() => {
      lastFlowProps.onDrop({
        preventDefault,
        clientX: 120,
        clientY: 80,
        dataTransfer: { getData: () => 'Conv2d' },
      });
    });
    expect(preventDefault).toHaveBeenCalled();
    expect(screenToFlowPositionSpy).toHaveBeenCalledWith({ x: 120, y: 80 });
    // 4 → 5 layers.
    expect(screen.getByText('5 layers')).toBeTruthy();
  });

  it('handleDrop with no layer type is a no-op', () => {
    setupOpenModal(validGraphJson());
    render(<SubgraphEditorModal />);
    act(() => {
      lastFlowProps.onDrop({
        preventDefault: vi.fn(),
        clientX: 0,
        clientY: 0,
        dataTransfer: { getData: () => '' },
      });
    });
    // unchanged.
    expect(screen.getByText('4 layers')).toBeTruthy();
    expect(screenToFlowPositionSpy).not.toHaveBeenCalled();
  });

  it('handleDrop with an unknown layer type does not add a node (addLayer early return)', () => {
    setupOpenModal(validGraphJson());
    render(<SubgraphEditorModal />);
    act(() => {
      lastFlowProps.onDrop({
        preventDefault: vi.fn(),
        clientX: 5,
        clientY: 5,
        dataTransfer: { getData: () => 'NotARealLayer' },
      });
    });
    // screenToFlowPosition is still called, but addLayer returns early.
    expect(screen.getByText('4 layers')).toBeTruthy();
  });

  // ── ParamEditor: param edits + delete ─────────────────────────────────────

  it('ParamEditor edits int, float, and renders for a param-less layer', () => {
    setupOpenModal(validGraphJson());
    render(<SubgraphEditorModal />);

    // Select Linear (int params).
    act(() => {
      lastFlowProps.onNodeClick({}, { id: 'lin1' });
    });
    const inFeatures = screen.getByDisplayValue('512') as HTMLInputElement;
    fireEvent.change(inFeatures, { target: { value: '256' } });
    expect(screen.getByDisplayValue('256')).toBeTruthy();

    // Select Concat (has int `dim`) then add a Dropout to exercise float.
    act(() => {
      lastFlowProps.onDrop({
        preventDefault: vi.fn(),
        clientX: 1,
        clientY: 1,
        dataTransfer: { getData: () => 'Dropout' },
      });
    });
    // Newly added node is last; find it via captured props.
    const dropoutNode = (lastFlowProps.nodes as any[]).find((n) => n.data.layerType === 'Dropout');
    act(() => {
      lastFlowProps.onNodeClick({}, { id: dropoutNode.id });
    });
    const pInput = screen.getByDisplayValue('0.5') as HTMLInputElement;
    fireEvent.change(pInput, { target: { value: '0.25' } });
    expect(screen.getByDisplayValue('0.25')).toBeTruthy();
  });

  it('ParamEditor shows "No parameters" for a layer with no params and deletes the layer', () => {
    setupOpenModal(validGraphJson());
    render(<SubgraphEditorModal />);

    // Add a ReLU (no params) and select it.
    act(() => {
      lastFlowProps.onDrop({
        preventDefault: vi.fn(),
        clientX: 2,
        clientY: 2,
        dataTransfer: { getData: () => 'ReLU' },
      });
    });
    const relu = (lastFlowProps.nodes as any[]).find((n) => n.data.layerType === 'ReLU');
    act(() => {
      lastFlowProps.onNodeClick({}, { id: relu.id });
    });
    // No-params message inside the ParamEditor.
    expect(screen.getAllByText('No parameters').length).toBeGreaterThan(0);

    // Delete via the ParamEditor button.
    fireEvent.click(screen.getByText('Delete'));
    // Back to 4 layers (added 1, deleted 1).
    expect(screen.getByText('4 layers')).toBeTruthy();
  });

  it('ParamEditor falls back to default when a param value is absent (?? p.default)', () => {
    // Linear node without explicit params → ParamEditor reads p.default.
    const json = JSON.stringify({
      version: 2,
      nodes: [
        { id: 'in1', type: 'Input', ports: [{ id: 'ip1', name: 'x' }], position: { x: 0, y: 0 } },
        { id: 'lin1', type: 'Linear', position: { x: 0, y: 100 } },
        { id: 'out1', type: 'Output', ports: [{ id: 'op1', name: 'y' }], position: { x: 0, y: 200 } },
      ],
      edges: [],
    });
    setupOpenModal(json);
    render(<SubgraphEditorModal />);
    act(() => {
      lastFlowProps.onNodeClick({}, { id: 'lin1' });
    });
    // Defaults: in_features=512, out_features=10.
    expect(screen.getByDisplayValue('512')).toBeTruthy();
    expect(screen.getByDisplayValue('10')).toBeTruthy();
  });

  // ── PortListEditor wiring (handleUpdatePorts / handleRemoveEdges) ──────────

  it('editing a port name through PortListEditor updates the node', () => {
    setupOpenModal(validGraphJson());
    render(<SubgraphEditorModal />);
    act(() => {
      lastFlowProps.onNodeClick({}, { id: 'in1' });
    });
    const portInput = screen.getByDisplayValue('x');
    fireEvent.change(portInput, { target: { value: 'inp' } });
    expect(screen.getByDisplayValue('inp')).toBeTruthy();
  });

  it('adding then removing an Input port also removes orphaned edges (handleRemoveEdges)', () => {
    setupOpenModal(validGraphJson());
    render(<SubgraphEditorModal />);
    act(() => {
      lastFlowProps.onNodeClick({}, { id: 'in1' });
    });
    // Add a port so removing the original (which has edge e1) is allowed.
    fireEvent.click(screen.getByText('+ Add port'));
    // Now remove the first port ('x') → its source-handle edge e1 is orphaned.
    const rows = screen.getAllByRole('textbox').filter((el) => (el as HTMLInputElement).value !== '');
    // The port rows contain the port-name inputs; pick the 'x' row's Remove btn.
    const xRow = screen.getByDisplayValue('x').closest('div')!;
    fireEvent.click(within(xRow).getByRole('button'));
    // e1 referenced in1→ip1; after removal it should be gone from edges.
    const hasE1 = (lastFlowProps.edges as any[]).some((e) => e.id === 'e1');
    expect(hasE1).toBe(false);
    expect(rows.length).toBeGreaterThan(0);
  });

  // ── Apply / Cancel / validation ───────────────────────────────────────────

  it('Apply on a valid graph serializes layers and closes the modal', () => {
    const nodeId = setupOpenModal(validGraphJson());
    render(<SubgraphEditorModal />);

    fireEvent.click(screen.getByText('Apply'));

    const tab = useTabStore.getState().tabs[0];
    // Modal closed.
    expect(tab.subgraphModalNodeId).toBeNull();
    // Layers updated on the node with serialized JSON containing our nodes.
    const node = tab.nodes.find((n) => n.id === nodeId)!;
    const saved = node.data.params!.layers as string;
    expect(saved).toContain('"version":2');
    expect(saved).toContain('Linear');
  });

  it('Apply on an invalid graph surfaces a validation toast and does NOT close', () => {
    // Graph with TWO Input nodes → validateGraph fails immediately.
    const badJson = JSON.stringify({
      version: 2,
      nodes: [
        { id: 'in1', type: 'Input', ports: [{ id: 'a', name: 'x' }], position: { x: 0, y: 0 } },
        { id: 'in2', type: 'Input', ports: [{ id: 'b', name: 'x2' }], position: { x: 0, y: 50 } },
        { id: 'out1', type: 'Output', ports: [{ id: 'c', name: 'y' }], position: { x: 0, y: 100 } },
      ],
      edges: [],
    });
    setupOpenModal(badJson);
    render(<SubgraphEditorModal />);

    fireEvent.click(screen.getByText('Apply'));

    const toasts = useToastStore.getState().toasts;
    expect(toasts.length).toBe(1);
    expect(toasts[0].type).toBe('error');
    // Still open.
    expect(useTabStore.getState().tabs[0].subgraphModalNodeId).not.toBeNull();
  });

  it('Cancel (footer button) closes the modal', () => {
    setupOpenModal(validGraphJson());
    render(<SubgraphEditorModal />);
    // There are two "Cancel" buttons (footer + would-be header ✕); footer text.
    fireEvent.click(screen.getAllByText('Cancel')[0]);
    expect(useTabStore.getState().tabs[0].subgraphModalNodeId).toBeNull();
  });

  it('the header ✕ button closes the modal', () => {
    setupOpenModal(validGraphJson());
    render(<SubgraphEditorModal />);
    fireEvent.click(screen.getByText('✕'));
    expect(useTabStore.getState().tabs[0].subgraphModalNodeId).toBeNull();
  });

  it('clicking the overlay backdrop closes the modal, clicking the panel does not', () => {
    setupOpenModal(validGraphJson());
    const { container } = render(<SubgraphEditorModal />);
    const overlay = container.querySelector('div')!;

    // Click on the inner panel (currentTarget !== target) → stays open.
    fireEvent.click(screen.getByText('Model Architecture Editor'));
    expect(useTabStore.getState().tabs[0].subgraphModalNodeId).not.toBeNull();

    // Click directly on the backdrop (target === currentTarget) → closes.
    fireEvent.click(overlay);
    expect(useTabStore.getState().tabs[0].subgraphModalNodeId).toBeNull();
  });

  // ── Snap toggle ───────────────────────────────────────────────────────────

  it('toggles grid snap on and off, snapping node positions on enable', () => {
    // Positions deliberately off-grid so snapping changes them (changed branch).
    const json = JSON.stringify({
      version: 2,
      nodes: [
        { id: 'in1', type: 'Input', ports: [{ id: 'ip1', name: 'x' }], position: { x: 13, y: 27 } },
        { id: 'out1', type: 'Output', ports: [{ id: 'op1', name: 'y' }], position: { x: 51, y: 99 } },
      ],
      edges: [],
    });
    setupOpenModal(json);
    render(<SubgraphEditorModal />);

    // Initially OFF.
    expect(screen.getByText('Snap: OFF')).toBeTruthy();
    // Turn ON → effect snaps positions to the 20px grid.
    fireEvent.click(screen.getByText('Snap: OFF'));
    expect(screen.getByText('Snap: ON')).toBeTruthy();
    const inNode = (lastFlowProps.nodes as any[]).find((n) => n.id === 'in1');
    expect(inNode.position).toEqual({ x: 20, y: 20 });
    // snapToGrid prop reflects the toggle.
    expect(lastFlowProps.snapToGrid).toBe(true);

    // Turn OFF again (covers the early `if (!snapEnabled) return;` on next run).
    fireEvent.click(screen.getByText('Snap: ON'));
    expect(screen.getByText('Snap: OFF')).toBeTruthy();
  });

  it('snap with already-aligned nodes leaves positions unchanged (unchanged branch)', () => {
    const json = JSON.stringify({
      version: 2,
      nodes: [
        { id: 'in1', type: 'Input', ports: [{ id: 'ip1', name: 'x' }], position: { x: 20, y: 40 } },
        { id: 'out1', type: 'Output', ports: [{ id: 'op1', name: 'y' }], position: { x: 60, y: 80 } },
      ],
      edges: [],
    });
    setupOpenModal(json);
    render(<SubgraphEditorModal />);
    fireEvent.click(screen.getByText('Snap: OFF'));
    const inNode = (lastFlowProps.nodes as any[]).find((n) => n.id === 'in1');
    // Unchanged → identical positions.
    expect(inNode.position).toEqual({ x: 20, y: 40 });
  });

  // ── Auto layout ───────────────────────────────────────────────────────────

  it('Auto Layout repositions nodes and schedules a fitView', () => {
    setupOpenModal(validGraphJson());
    render(<SubgraphEditorModal />);
    fitViewSpy.mockClear();
    fireEvent.click(screen.getByText('Auto Layout'));
    // setTimeout(fitView) fires.
    act(() => {
      vi.advanceTimersByTime(60);
    });
    expect(fitViewSpy).toHaveBeenCalledWith({ padding: 0.3 });
  });

  // ── Export ────────────────────────────────────────────────────────────────

  it('Export builds a JSON blob and triggers a download anchor', () => {
    setupOpenModal(validGraphJson());
    render(<SubgraphEditorModal />);

    const createUrl = vi
      .spyOn(URL, 'createObjectURL')
      .mockReturnValue('blob:mock');
    const revokeUrl = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    const clickSpy = vi.fn();
    const origCreate = document.createElement.bind(document);
    const createElSpy = vi
      .spyOn(document, 'createElement')
      .mockImplementation((tag: string) => {
        const el = origCreate(tag);
        if (tag === 'a') {
          (el as HTMLAnchorElement).click = clickSpy;
        }
        return el;
      });

    fireEvent.click(screen.getByText('Export'));

    expect(createUrl).toHaveBeenCalled();
    expect(clickSpy).toHaveBeenCalled();
    expect(revokeUrl).toHaveBeenCalledWith('blob:mock');

    createElSpy.mockRestore();
    createUrl.mockRestore();
    revokeUrl.mockRestore();
  });

  // ── Import: handleImport click + handleFileSelect branches ────────────────

  it('Import button clicks the hidden file input', () => {
    setupOpenModal(validGraphJson());
    const { container } = render(<SubgraphEditorModal />);
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    const clickSpy = vi.spyOn(fileInput, 'click').mockImplementation(() => {});
    fireEvent.click(screen.getByText('Import'));
    expect(clickSpy).toHaveBeenCalled();
  });

  it('handleFileSelect with no file selected is a no-op', () => {
    setupOpenModal(validGraphJson());
    const { container } = render(<SubgraphEditorModal />);
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    // No files → early return.
    fireEvent.change(fileInput, { target: { files: [] } });
    expect(screen.getByText('4 layers')).toBeTruthy();
  });

  /**
   * Drives handleFileSelect by stubbing FileReader so onload fires synchronously
   * with the provided text, then returns the modal's layer count text.
   */
  function importText(container: HTMLElement, text: string) {
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;

    class FR {
      onload: ((e: any) => void) | null = null;
      readAsText() {
        // Fire async-ish but within fake timers we call directly.
        this.onload?.({ target: { result: text } });
      }
    }
    const orig = globalThis.FileReader;
    (globalThis as any).FileReader = FR as any;

    const file = new File([text], 'arch.json', { type: 'application/json' });
    // jsdom File.text not used; FileReader stub returns our text regardless.
    act(() => {
      fireEvent.change(fileInput, { target: { files: [file] } });
    });
    (globalThis as any).FileReader = orig;
  }

  it('handleFileSelect imports a GraphSpec v2 file (graphspec branch)', () => {
    setupOpenModal(validGraphJson());
    const { container } = render(<SubgraphEditorModal />);
    const spec = JSON.stringify({
      version: 2,
      nodes: [
        { id: 'i', type: 'Input', ports: [{ id: 'ip', name: 'x' }], position: { x: 0, y: 0 } },
        { id: 'r', type: 'ReLU', position: { x: 0, y: 50 } },
        { id: 'o', type: 'Output', ports: [{ id: 'op', name: 'y' }], position: { x: 0, y: 100 } },
      ],
      edges: [],
    });
    importText(container, spec);
    act(() => vi.advanceTimersByTime(60));
    // Imported graph has 3 nodes.
    expect(screen.getByText('3 layers')).toBeTruthy();
    expect(fitViewSpy).toHaveBeenCalled();
  });

  it('handleFileSelect imports a main-editor workflow with layer nodes (workflow-layers branch)', () => {
    setupOpenModal(validGraphJson());
    const { container } = render(<SubgraphEditorModal />);
    const workflow = JSON.stringify({
      nodes: [
        { id: 'a', type: 'Conv2d', position: { x: 0, y: 0 }, data: { params: { in_channels: 3 } } },
        { id: 'b', type: 'ReLU', position: { x: 0, y: 50 }, data: {} },
      ],
      edges: [{ id: 'e', source: 'a', target: 'b' }],
    });
    importText(container, workflow);
    act(() => vi.advanceTimersByTime(60));
    // convertWorkflowToGraphSpec adds Input + Output around the 2 layers → 4.
    expect(screen.getByText('4 layers')).toBeTruthy();
  });

  it('handleFileSelect imports a workflow with an Activation node (function → layer mapping)', () => {
    setupOpenModal(validGraphJson());
    const { container } = render(<SubgraphEditorModal />);
    // An Activation node whose `function` maps via ACTIVATION_MAP makes
    // detection = workflow-layers and converts to a ReLU layer.
    const workflow = JSON.stringify({
      nodes: [
        { id: 'a', type: 'Activation', position: { x: 0, y: 0 }, data: { params: { function: 'relu' } } },
      ],
      edges: [],
    });
    importText(container, workflow);
    act(() => vi.advanceTimersByTime(60));
    // Input + ReLU + Output = 3 nodes.
    expect(screen.getByText('3 layers')).toBeTruthy();
  });

  it('handleFileSelect imports a single SequentialModel workflow (workflow-sequential, len 1)', () => {
    setupOpenModal(validGraphJson());
    const { container } = render(<SubgraphEditorModal />);
    const inner = JSON.stringify({
      version: 2,
      nodes: [
        { id: 'i', type: 'Input', ports: [{ id: 'ip', name: 'x' }], position: { x: 0, y: 0 } },
        { id: 'o', type: 'Output', ports: [{ id: 'op', name: 'y' }], position: { x: 0, y: 50 } },
      ],
      edges: [],
    });
    const workflow = JSON.stringify({
      nodes: [
        { id: 's1', type: 'SequentialModel', data: { label: 'Seq A', params: { layers: inner } } },
      ],
      edges: [],
    });
    importText(container, workflow);
    act(() => vi.advanceTimersByTime(60));
    // Imported inner graph has 2 nodes.
    expect(screen.getByText('2 layers')).toBeTruthy();
  });

  it('handleFileSelect with multiple SequentialModels opens the selector and imports a choice', () => {
    setupOpenModal(validGraphJson());
    const { container } = render(<SubgraphEditorModal />);
    const innerA = JSON.stringify({
      version: 2,
      nodes: [
        { id: 'i', type: 'Input', ports: [{ id: 'ip', name: 'x' }], position: { x: 0, y: 0 } },
        { id: 'o', type: 'Output', ports: [{ id: 'op', name: 'y' }], position: { x: 0, y: 50 } },
      ],
      edges: [],
    });
    const innerB = JSON.stringify({
      version: 2,
      nodes: [
        { id: 'i', type: 'Input', ports: [{ id: 'ip', name: 'x' }], position: { x: 0, y: 0 } },
        { id: 'r', type: 'ReLU', position: { x: 0, y: 50 } },
        { id: 'o', type: 'Output', ports: [{ id: 'op', name: 'y' }], position: { x: 0, y: 100 } },
      ],
      edges: [],
    });
    const workflow = JSON.stringify({
      nodes: [
        { id: 's1', type: 'SequentialModel', data: { label: 'Seq A', params: { layers: innerA } } },
        { id: 's2', type: 'SequentialModel', data: { label: 'Seq B', params: { layers: innerB } } },
      ],
      edges: [],
    });
    importText(container, workflow);

    // Selector dialog appears.
    const heading = screen.getByText('Select SequentialModel to Import');
    expect(heading).toBeTruthy();
    expect(screen.getByText('Seq A')).toBeTruthy();
    expect(screen.getByText('Seq B')).toBeTruthy();

    // Select the second model (covers setSelectedIdx) then import. The
    // selector panel is the heading's parent; scope to its Import button so we
    // don't hit the header's Import button.
    fireEvent.click(screen.getByText('Seq B'));
    const panel = heading.parentElement as HTMLElement;
    fireEvent.click(within(panel).getByText('Import'));
    act(() => vi.advanceTimersByTime(60));
    // Seq B has 3 nodes.
    expect(screen.getByText('3 layers')).toBeTruthy();
  });

  it('SequentialModelSelector: cancel button and backdrop both close it', () => {
    setupOpenModal(validGraphJson());
    const { container } = render(<SubgraphEditorModal />);
    const inner = JSON.stringify({ version: 2, nodes: [], edges: [] });
    const workflow = JSON.stringify({
      nodes: [
        { id: 's1', type: 'SequentialModel', data: { label: 'A', params: { layers: inner } } },
        { id: 's2', type: 'SequentialModel', data: { label: 'B', params: { layers: inner } } },
      ],
      edges: [],
    });
    importText(container, workflow);
    const heading1 = screen.getByText('Select SequentialModel to Import');
    expect(heading1).toBeTruthy();

    // Cancel button (scoped to the selector panel) closes the selector.
    const panel1 = heading1.parentElement as HTMLElement;
    fireEvent.click(within(panel1).getByText('Cancel'));
    expect(screen.queryByText('Select SequentialModel to Import')).toBeNull();

    // Reopen, then close via backdrop click.
    importText(container, workflow);
    const heading = screen.getByText('Select SequentialModel to Import');
    const panel = heading.parentElement as HTMLElement;
    const backdrop = panel.parentElement as HTMLElement; // panel → backdrop
    // Click the panel itself (stopPropagation) → stays open.
    fireEvent.click(panel);
    expect(screen.getByText('Select SequentialModel to Import')).toBeTruthy();
    // Click the backdrop → closes (onClick={onCancel}).
    fireEvent.click(backdrop);
    expect(screen.queryByText('Select SequentialModel to Import')).toBeNull();
  });

  it('handleSelectSequentialModel surfaces a toast when the chosen model is invalid', () => {
    setupOpenModal(validGraphJson());
    const { container } = render(<SubgraphEditorModal />);
    // Two models so the selector is shown; one has an EMPTY graph so importing
    // it throws "Empty or invalid graph" → caught → toast.
    const emptyInner = JSON.stringify({ version: 2, nodes: [], edges: [] });
    const goodInner = JSON.stringify({
      version: 2,
      nodes: [
        { id: 'i', type: 'Input', ports: [{ id: 'ip', name: 'x' }], position: { x: 0, y: 0 } },
        { id: 'o', type: 'Output', ports: [{ id: 'op', name: 'y' }], position: { x: 0, y: 50 } },
      ],
      edges: [],
    });
    const workflow = JSON.stringify({
      nodes: [
        { id: 's1', type: 'SequentialModel', data: { label: 'Empty', params: { layers: emptyInner } } },
        { id: 's2', type: 'SequentialModel', data: { label: 'Good', params: { layers: goodInner } } },
      ],
      edges: [],
    });
    importText(container, workflow);

    // First entry (Empty) is selected by default; import it → catch → toast.
    const heading = screen.getByText('Select SequentialModel to Import');
    const panel = heading.parentElement as HTMLElement;
    fireEvent.click(within(panel).getByText('Import'));
    const toasts = useToastStore.getState().toasts;
    expect(toasts.some((t) => t.type === 'error')).toBe(true);
    // Selector closed regardless.
    expect(screen.queryByText('Select SequentialModel to Import')).toBeNull();
  });

  it('handleFileSelect with unparseable JSON shows the import-fail toast (unknown branch)', () => {
    setupOpenModal(validGraphJson());
    const { container } = render(<SubgraphEditorModal />);
    importText(container, 'not-json-at-all');
    const toasts = useToastStore.getState().toasts;
    expect(toasts.length).toBe(1);
    expect(toasts[0].type).toBe('error');
  });

  it('handleFileSelect with valid JSON but no importable content throws noContent', () => {
    setupOpenModal(validGraphJson());
    const { container } = render(<SubgraphEditorModal />);
    // Valid JSON, has nodes/edges arrays but no known layer nodes and no
    // SequentialModel → detection = unknown → throws noContent → toast.
    const json = JSON.stringify({ nodes: [{ id: 'x', type: 'Weird' }], edges: [] });
    importText(container, json);
    const toasts = useToastStore.getState().toasts;
    expect(toasts.length).toBe(1);
    expect(toasts[0].type).toBe('error');
  });

  // ── Empty-graph overlay ───────────────────────────────────────────────────

  it('renders the empty-canvas hint when there are no nodes', () => {
    // Force a zero-node graph by deleting both boundary nodes after open.
    setupOpenModal(undefined); // emptyGraph → 2 nodes
    render(<SubgraphEditorModal />);
    act(() => {
      lastFlowProps.onNodesChange([{ type: 'remove', id: (lastFlowProps.nodes as any[])[0].id }]);
    });
    act(() => {
      lastFlowProps.onNodesChange([{ type: 'remove', id: (lastFlowProps.nodes as any[])[0].id }]);
    });
    expect(screen.getByText('Drag layers from the left panel to build your model')).toBeTruthy();
    expect(screen.getByText('0 layers')).toBeTruthy();
  });
});
