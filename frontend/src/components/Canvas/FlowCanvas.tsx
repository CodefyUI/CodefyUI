import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  ReactFlow,
  MiniMap,
  Background,
  Controls,
  BackgroundVariant,
  useReactFlow,
  type Node,
  type NodeTypes,
  type EdgeTypes,
  type OnConnect,
  type IsValidConnection,
  type Connection,
  type Edge,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { CANVAS_MIN_ZOOM, CATEGORY_COLORS } from '../../styles/theme';

import BaseNode from '../Nodes/BaseNode';
import PluginNodeBridge from '../Nodes/PluginNodeBridge';
import PresetNode from '../Nodes/PresetNode';
import SubgraphInstanceNode from '../Nodes/SubgraphInstanceNode';
import { StartNode } from '../Nodes/StartNode';
import NoteNode from '../Nodes/NoteNode';
import TokenizerVizNode from '../Nodes/TokenizerVizNode';
import EmbeddingScatterVizNode from '../Nodes/EmbeddingScatterVizNode';
import TextInputVizNode from '../Nodes/TextInputVizNode';
import EduSelfAttentionVizNode from '../Nodes/EduSelfAttentionVizNode';
import EduMultiHeadAttentionVizNode from '../Nodes/EduMultiHeadAttentionVizNode';
import AttentionHeatmapVizNode from '../Nodes/AttentionHeatmapVizNode';
import AttentionMaskVizNode from '../Nodes/AttentionMaskVizNode';
import EduCrossAttentionVizNode from '../Nodes/EduCrossAttentionVizNode';
import EduKNNVizNode from '../Nodes/EduKNNVizNode';
import { CustomConnectionLine } from './CustomConnectionLine';
import { EdgeLaneProvider } from './EdgeLaneContext';
import { SmartDataEdge } from './SmartDataEdge';
import { TriggerEdge } from './TriggerEdge';
import { EmptyCanvasOverlay } from './EmptyCanvasOverlay';
import { EdgeDataTooltip } from './EdgeDataTooltip';
import { QuickNodeSearch } from './QuickNodeSearch';
import {
  NodeContextMenu,
  useNodeContextMenuItems,
  useNoteContextMenuItems,
  type ContextMenuPosition,
} from '../ContextMenu/NodeContextMenu';
import { PaneContextMenu } from './PaneContextMenu';
import { SubgraphBreadcrumb } from './SubgraphBreadcrumb';
import { NoteBindingLines } from './NoteBindingLines';
import { SegmentBubble } from './SegmentBubble';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import { useDragAndDrop } from '../../hooks/useDragAndDrop';
import {
  isValidConnection,
  getPortColor,
  resolveDynamicInputs,
  resolveDynamicOutputs,
} from '../../utils';
import { computeDetachedEndpoint } from '../../utils/reconnect';
import { nodesBoundingBox } from '../../utils/autoLayout';
import { rememberViewport, recallViewport } from '../../utils/viewportMemory';
import { prompt } from '../../utils/dialog';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useI18n } from '../../i18n';
import type { OutputSummary } from '../../types';
import styles from './FlowCanvas.module.css';

const nodeTypes: NodeTypes = {
  baseNode: BaseNode,
  pluginNode: PluginNodeBridge,
  presetNode: PresetNode,
  subgraphNode: SubgraphInstanceNode,
  start: StartNode,
  noteNode: NoteNode,
  tokenizerNode: TokenizerVizNode,
  embeddingScatterNode: EmbeddingScatterVizNode,
  textInputNode: TextInputVizNode,
  eduSelfAttentionNode: EduSelfAttentionVizNode,
  eduMultiHeadAttentionNode: EduMultiHeadAttentionVizNode,
  attentionHeatmapNode: AttentionHeatmapVizNode,
  attentionMaskNode: AttentionMaskVizNode,
  eduCrossAttentionNode: EduCrossAttentionVizNode,
  eduKNNNode: EduKNNVizNode,
};

const edgeTypes: EdgeTypes = {
  default: SmartDataEdge,
  triggerEdge: TriggerEdge,
};

const minimapNodeColor = (node: any) => {
  // No note-category token exists in tokens.css (see migration report) —
  // kept literal; this is a decorative minimap dot, not chrome text.
  if (node.type === 'noteNode') return '#FFD700';
  const data = node.data as any;
  if (data?.isPreset) return 'var(--status-preset)';
  const category = data?.definition?.category ?? 'Utility';
  // Fallback used to be the raw, unlifted '#607D8B' — see the same fix in
  // BaseNode.tsx's headerColor.
  return CATEGORY_COLORS[category] ?? CATEGORY_COLORS.Utility;
};


export function FlowCanvas({ tabId }: { tabId?: string } = {}) {
  const activeTab = useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId)!);
  const activeTabId = useTabStore((s) => s.activeTabId);
  const onNodesChange = useTabStore((s) => s.onNodesChange);
  const onEdgesChange = useTabStore((s) => s.onEdgesChange);
  const storeOnConnect = useTabStore((s) => s.onConnect);
  const setSelectedNodeId = useTabStore((s) => s.setSelectedNodeId);
  const selectNodeExclusively = useTabStore((s) => s.selectNodeExclusively);
  const deleteNode = useTabStore((s) => s.deleteNode);
  const duplicateNode = useTabStore((s) => s.duplicateNode);
  const renameNode = useTabStore((s) => s.renameNode);
  const openNodeDetail = useTabStore((s) => s.openNodeDetail);
  const { t } = useI18n();
  const gridSnapEnabled = useUIStore((s) => s.gridSnapEnabled);
  const setCanvasPanning = useUIStore((s) => s.setCanvasPanning);
  const setNodes = useTabStore((s) => s.setNodes);
  const layoutFitRequest = useUIStore((s) => s.layoutFitRequest);
  const { screenToFlowPosition, fitBounds, getViewport, setViewport } =
    useReactFlow();

  const containerRef = useRef<HTMLDivElement>(null);
  const reactFlowId = useId();

  // Fit a flow-space box as an OVERVIEW: small boxes (a single selected node,
  // a two-node graph) are inflated to most of the viewport first, so the fit
  // never zooms in aggressively toward maxZoom.
  //
  // Instant, never animated: animated viewport transitions run on
  // requestAnimationFrame, which Chrome throttles to zero in occluded or
  // background windows — the animation then never applies at all. The
  // subgraph editor's post-layout fit and the Controls button are instant for
  // the same reason.
  const fitToBounds = useCallback(
    (bounds: { x: number; y: number; width: number; height: number }) => {
      const el = containerRef.current;
      if (!el || el.offsetWidth === 0) return;
      let { x, y, width, height } = bounds;
      const minW = el.offsetWidth * 0.85;
      const minH = el.offsetHeight * 0.85;
      if (width < minW) {
        x -= (minW - width) / 2;
        width = minW;
      }
      if (height < minH) {
        y -= (minH - height) / 2;
        height = minH;
      }
      fitBounds({ x, y, width, height }, { padding: 0.2 });
    },
    [fitBounds],
  );

  // ── Per-tab viewport handover (#125) ───────────────────────────────────────
  // One canvas now serves every tab, so switching tabs has to move the
  // viewport by hand: stash where the outgoing tab was looking, put the
  // incoming tab back where IT was. A tab being opened for the first time has
  // nothing stored, so it gets the same overview fit it used to get from its
  // own freshly-mounted provider.
  //
  // That first-visit fit is computed from the STORE's node positions rather
  // than asked of React Flow. `fitView()` needs measured nodes, and React Flow
  // has only just been handed the incoming tab's — so it would either fit
  // nothing or have to wait for measurement, during which the user stares at
  // the OUTGOING tab's viewport over the incoming tab's graph. `fitBounds`
  // over a box we can compute ourselves lands in the same tick, with no
  // intermediate wrong frame. Sizes fall back to the same defaults the
  // auto-layout fit uses, so an unmeasured node still contributes a box.
  //
  // A LAYOUT effect, not a passive one: the render that changes activeTabId
  // has already handed <ReactFlow> the incoming tab's nodes, so a passive
  // effect would let the browser paint one frame of the new graph under the
  // OUTGOING tab's pan/zoom before correcting it. useLayoutEffect runs before
  // that paint, and everything it needs is available there — `offsetWidth` is
  // read after the DOM is committed, and `getViewport` reads store state.
  //
  // Keyed on the store's activeTabId rather than the `tabId` prop so this
  // holds regardless of how the canvas is mounted, and skipped entirely on
  // first mount — the `fitView` prop on <ReactFlow> owns the initial viewport.
  // `previousTabRef` also makes the effect idempotent under StrictMode's
  // double invocation.
  const previousTabRef = useRef<string | null>(null);

  useLayoutEffect(() => {
    const previous = previousTabRef.current;
    if (previous === activeTabId) return;
    previousTabRef.current = activeTabId;
    if (previous === null) return;

    rememberViewport(previous, getViewport());
    const restored = recallViewport(activeTabId);
    if (restored) {
      setViewport(restored);
      return;
    }
    const incoming = useTabStore.getState().tabs.find((t) => t.id === activeTabId);
    const bounds = nodesBoundingBox((incoming?.nodes ?? []) as Node[]);
    // An empty tab has nothing to fit; leave the viewport where it is rather
    // than inventing a position for a blank canvas.
    if (bounds) fitToBounds(bounds);
  }, [activeTabId, getViewport, setViewport, fitToBounds]);

  // Snap all existing nodes to grid when grid snap is enabled
  useEffect(() => {
    if (!gridSnapEnabled) return;
    const GRID = 24;
    const snapped = activeTab.nodes.map((node) => ({
      ...node,
      position: {
        x: Math.round(node.position.x / GRID) * GRID,
        y: Math.round(node.position.y / GRID) * GRID,
      },
    }));
    const changed = snapped.some(
      (n, i) =>
        n.position.x !== activeTab.nodes[i].position.x ||
        n.position.y !== activeTab.nodes[i].position.y
    );
    if (changed) {
      setNodes(snapped);
    }
  }, [gridSnapEnabled]);

  // Re-fit the viewport after auto-layout. The request already carries the
  // laid-out bounding box (computed from store data), so this needs nothing
  // from React Flow's internal position sync — fitBounds sets the viewport
  // directly and immediately. (The queued fitView() from useReactFlow only
  // flushes on the next node change, and reading positions back via
  // getNodesBounds races the sync — both failure modes seen in e2e.) Since
  // #125 only the active tab's canvas is mounted, so the `tabId` check below
  // is belt-and-braces (a harness can still mount several); the one-shot
  // request is cleared either way so a remount can't replay it.
  useEffect(() => {
    if (!layoutFitRequest) return;
    const el = containerRef.current;
    if (!el || el.offsetWidth === 0) return;
    if (tabId !== undefined && tabId !== useTabStore.getState().activeTabId) return;
    fitToBounds(layoutFitRequest.bounds);
    useUIStore.getState().clearLayoutFit();
  }, [layoutFitRequest, fitToBounds, tabId]);

  const [quickSearch, setQuickSearch] = useState<{
    screen: { x: number; y: number };
    flow: { x: number; y: number };
  } | null>(null);

  const [contextMenu, setContextMenu] = useState<ContextMenuPosition | null>(null);
  const [paneMenu, setPaneMenu] = useState<{
    screen: { x: number; y: number };
    flow: { x: number; y: number };
  } | null>(null);
  const [edgeTooltip, setEdgeTooltip] = useState<{
    x: number; y: number;
    sourceLabel: string; targetLabel: string;
    portName: string; summary: OutputSummary;
    // Where the value on this edge comes from and where it lands. "View
    // stats" opens the CONSUMER's modal, because the port reads there as the
    // input it is — and the Stats tab resolves an input back to the producing
    // node's capture, so it is the same numbers either way (#129).
    sourceId: string; targetId: string;
  } | null>(null);

  const outputSummaries = useTabStore((s) => {
    const tab = s.tabs.find((t) => t.id === s.activeTabId);
    // every tab always has outputSummaries (required field, defaults to {}); the
    // ?? {} fallback only fires when tab is absent, which cannot happen here
    /* v8 ignore next -- @preserve */
    return tab?.outputSummaries ?? {};
  });

  const { onDragOver, onDrop } = useDragAndDrop();

  const handleConnect: OnConnect = useCallback(
    (connection) => {
      storeOnConnect(connection);

      if (connection.sourceHandle === 'trigger') {
        const { setEdges } = useTabStore.getState();
        const tab = useTabStore.getState().tabs.find(
          (t) => t.id === useTabStore.getState().activeTabId,
        );
        // FlowCanvas only renders with an active tab (the `activeTab` selector
        // at the top asserts it), so this lookup always finds it; the else
        // branch is never taken
        /* v8 ignore next -- @preserve */
        if (tab) {
          setEdges(
            tab.edges.map((e) =>
              e.source === connection.source &&
              e.target === connection.target &&
              e.sourceHandle === connection.sourceHandle
                ? {
                    ...e,
                    type: 'triggerEdge',
                    targetHandle: '__trigger',
                    data: { ...(e.data ?? {}), type: 'trigger' },
                  }
                : e,
            ),
          );
        }
        return; // skip the data-edge color logic
      }

      // Color the new edge by source port data type
      if (connection.source && connection.sourceHandle) {
        const defs = useNodeDefStore.getState().definitions;
        const currentTab = useTabStore.getState().tabs.find(
          (t) => t.id === useTabStore.getState().activeTabId,
        );
        const srcNode = currentTab?.nodes.find((n) => n.id === connection.source);
        if (srcNode) {
          // Flow nodes carry the xyflow component type in `.type` ('baseNode',
          // 'pluginNode', viz types, ...) and the real node type + definition
          // in `.data`, so resolve the source port from the node's own
          // definition (dynamic outputs included — Split's chunk_N ports).
          // Fall back to the registry keyed by the real node type when a node
          // has no inline definition.
          const data = srcNode.data;
          const definition =
            data?.definition ?? defs.find((d) => d.node_name === (data?.type ?? srcNode.type));
          const output = resolveDynamicOutputs(definition, data?.params).find(
            (o) => o.name === connection.sourceHandle,
          );
          if (output) {
            const color = getPortColor(output.data_type);
            const { setEdges } = useTabStore.getState();
            const tab = useTabStore.getState().tabs.find(
              (t) => t.id === useTabStore.getState().activeTabId,
            );
            // FlowCanvas only renders with an active tab (the `activeTab`
            // selector at the top asserts it), so reaching here means the
            // lookup above already succeeded; else never taken
            /* v8 ignore next -- @preserve */
            if (tab) {
              setEdges(
                tab.edges.map((e) =>
                  e.source === connection.source &&
                  e.sourceHandle === connection.sourceHandle &&
                  e.target === connection.target &&
                  e.targetHandle === connection.targetHandle
                    ? { ...e, style: { ...e.style, stroke: color, strokeWidth: 2 } }
                    : e,
                ),
              );
            }
          }
        }
      }
    },
    [storeOnConnect],
  );

  const handleIsValidConnection: IsValidConnection = useCallback(
    (edgeOrConnection) => {
      // `IsValidConnection` now receives `Edge | Connection`; both expose
      // source / target / sourceHandle / targetHandle, so we don't need to
      // narrow for the checks below.
      const { source, target, sourceHandle, targetHandle } = edgeOrConnection;
      if (!source || !target) return false;
      if (source === target) return false;

      // Notes cannot be connected
      const { tabs, activeTabId } = useTabStore.getState();
      const tab = tabs.find((t) => t.id === activeTabId)!;
      const sourceNode = tab.nodes.find((n) => n.id === source);
      const targetNode = tab.nodes.find((n) => n.id === target);
      if (sourceNode?.type === 'noteNode' || targetNode?.type === 'noteNode') return false;

      // Trigger connections (from Start node) are control-flow markers,
      // not data — they connect only to the __trigger handle on target nodes.
      if (sourceHandle === 'trigger') return targetHandle === '__trigger';

      if (sourceHandle && targetHandle) {
        if (!sourceNode || !targetNode) return true;

        const sourceDef = sourceNode.data.definition;
        const targetDef = targetNode.data.definition;
        if (!sourceDef || !targetDef) return true;

        // Live port sets, not the palette template: a script node's ports
        // and their types follow its params (core#131).
        const sourceOutput = resolveDynamicOutputs(sourceDef, sourceNode.data.params)
          .find((o) => o.name === sourceHandle);
        const targetInput = resolveDynamicInputs(targetDef, targetNode.data.params)
          .find((i) => i.name === targetHandle);
        if (!sourceOutput || !targetInput) return true;

        return isValidConnection(sourceOutput.data_type, targetInput.data_type);
      }

      return true;
    },
    []
  );

  const onConnectStart = useCallback(
    (_: any, params: { nodeId: string | null; handleId: string | null; handleType: string | null }) => {
      if (params.nodeId && params.handleId && params.handleType === 'source') {
        const { tabs, activeTabId } = useTabStore.getState();
        const tab = tabs.find((t) => t.id === activeTabId);
        const node = tab?.nodes.find((n) => n.id === params.nodeId);
        if (node) {
          const def = node.data.definition;
          const output = def?.outputs.find((o) => o.name === params.handleId);
          if (output) {
            useUIStore.getState().setDraggingSourceType(output.data_type);
          }
        }
      }
    },
    []
  );

  const onConnectEnd = useCallback(() => {
    useUIStore.getState().setDraggingSourceType(null);
  }, []);

  // Track which edge is being reconnected so we can delete it if dropped on empty space
  const reconnectingEdgeRef = useRef<string | null>(null);

  const onReconnectStart = useCallback((_: any, edge: Edge, handleType: 'source' | 'target') => {
    reconnectingEdgeRef.current = edge.id;
    // Mark the endpoint being detached so its handle shows the red warning
    // ring (dropping on empty space deletes the edge). Note handleType names
    // the end that STAYS connected — see computeDetachedEndpoint.
    useUIStore.getState().setReconnectingHandle(computeDetachedEndpoint(edge, handleType));
  }, []);

  const onReconnect = useCallback((oldEdge: Edge, newConnection: Connection) => {
    reconnectingEdgeRef.current = null;
    // onReconnectEnd always follows and clears too; clearing here as well
    // keeps the indicator lifecycle local to each handler.
    useUIStore.getState().setReconnectingHandle(null);
    // Replace old edge with new connection
    const { setEdges } = useTabStore.getState();
    const tab = useTabStore.getState().tabs.find(
      (t) => t.id === useTabStore.getState().activeTabId,
    );
    if (!tab) return;
    useTabStore.getState().pushUndoSnapshot();
    setEdges(
      tab.edges
        .filter((e) => e.id !== oldEdge.id)
        .concat({
          ...oldEdge,
          source: newConnection.source,
          target: newConnection.target,
          sourceHandle: newConnection.sourceHandle ?? undefined,
          targetHandle: newConnection.targetHandle ?? undefined,
        }),
    );
  }, []);

  const onReconnectEnd = useCallback((_: any, edge: Edge) => {
    // Always clear the red detach indicator — this fires after both outcomes
    // (edge rewired via onReconnect, or dropped on empty space and deleted).
    useUIStore.getState().setReconnectingHandle(null);
    // If the reconnect was not completed (dropped on empty space), delete the edge
    if (reconnectingEdgeRef.current === edge.id) {
      reconnectingEdgeRef.current = null;
      const { setEdges } = useTabStore.getState();
      const tab = useTabStore.getState().tabs.find(
        (t) => t.id === useTabStore.getState().activeTabId,
      );
      if (!tab) return;
      useTabStore.getState().pushUndoSnapshot();
      setEdges(tab.edges.filter((e) => e.id !== edge.id));
    }
  }, []);

  const handleNodeClick = useCallback(
    (_: React.MouseEvent, node: { id: string }) => {
      setSelectedNodeId(node.id);
    },
    [setSelectedNodeId]
  );

  const handleEdgeClick = useCallback(
    (event: React.MouseEvent, edge: Edge) => {
      const sourceId = edge.source;
      const sourceHandle = edge.sourceHandle ?? '';
      const nodeSummaries = outputSummaries[sourceId];
      if (!nodeSummaries || !nodeSummaries[sourceHandle]) {
        setEdgeTooltip(null);
        return;
      }
      const sourceNode = activeTab.nodes.find((n) => n.id === sourceId);
      const targetNode = activeTab.nodes.find((n) => n.id === edge.target);
      setEdgeTooltip({
        x: event.clientX + 8,
        y: event.clientY - 8,
        sourceLabel: sourceNode?.data.label ?? sourceId.slice(0, 8),
        targetLabel: targetNode?.data.label ?? edge.target.slice(0, 8),
        portName: sourceHandle,
        summary: nodeSummaries[sourceHandle],
        sourceId,
        targetId: edge.target,
      });
    },
    [outputSummaries, activeTab.nodes]
  );

  // Double-click on pane to open quick node search
  const screenToFlowRef = useRef(screenToFlowPosition);
  screenToFlowRef.current = screenToFlowPosition;
  const setQuickSearchRef = useRef(setQuickSearch);
  setQuickSearchRef.current = setQuickSearch;

  useEffect(() => {
    const container = containerRef.current;
    // containerRef is always bound to the root div, which renders unconditionally
    /* v8 ignore next -- @preserve */
    if (!container) return;

    const handler = (e: MouseEvent) => {
      // Ignore if the double-click originated inside a node (e.g. NoteNode editing)
      if ((e.target as HTMLElement).closest('.react-flow__node')) return;
      const flowPos = screenToFlowRef.current({ x: e.clientX, y: e.clientY });
      setQuickSearchRef.current({ screen: { x: e.clientX, y: e.clientY }, flow: flowPos });
    };
    // Wait for React Flow to mount, then attach directly to .react-flow__pane
    const timer = setTimeout(() => {
      const pane = container.querySelector('.react-flow__pane');
      if (pane) {
        pane.addEventListener('dblclick', handler as EventListener);
      }
    }, 100);
    return () => {
      clearTimeout(timer);
      const pane = container.querySelector('.react-flow__pane');
      if (pane) pane.removeEventListener('dblclick', handler as EventListener);
    };
  }, []);

  const handlePaneClick = useCallback(() => {
    selectNodeExclusively(null);
    setContextMenu(null);
    setPaneMenu(null);
    setEdgeTooltip(null);
    // quickSearch is closed by QuickNodeSearch's own outside-click handler
  }, [selectNodeExclusively]);

  const handlePaneContextMenu = useCallback(
    (event: MouseEvent | React.MouseEvent) => {
      event.preventDefault();
      const flowPos = screenToFlowPosition({ x: event.clientX, y: event.clientY });
      setPaneMenu({ screen: { x: event.clientX, y: event.clientY }, flow: flowPos });
    },
    [screenToFlowPosition],
  );

  const handleNodeContextMenu = useCallback(
    (event: React.MouseEvent, node: { id: string }) => {
      event.preventDefault();
      selectNodeExclusively(node.id);
      setContextMenu({ nodeId: node.id, x: event.clientX, y: event.clientY });
    },
    [selectNodeExclusively]
  );

  const handleRename = useCallback(
    async (nodeId: string) => {
      const node = activeTab.nodes.find((n) => n.id === nodeId);
      const currentLabel = node?.data.label ?? '';
      const newLabel = await prompt({
        title: t('contextMenu.rename.prompt'),
        defaultValue: currentLabel,
      });
      if (newLabel !== null && newLabel.trim()) {
        renameNode(nodeId, newLabel.trim());
      }
    },
    [activeTab.nodes, renameNode, t]
  );

  const nodeMenuItems = useNodeContextMenuItems(contextMenu?.nodeId ?? '', {
    onDelete: deleteNode,
    onRename: handleRename,
    onDuplicate: duplicateNode,
    onOpenDetails: openNodeDetail,
  });

  const noteMenuItems = useNoteContextMenuItems(contextMenu?.nodeId ?? '', {
    onDelete: deleteNode,
  });

  // Pick the right menu items based on node type
  const contextNode = activeTab.nodes.find((n) => n.id === contextMenu?.nodeId);
  const menuItems = contextNode?.type === 'noteNode' ? noteMenuItems : nodeMenuItems;

  const proOptions = useMemo(() => ({ hideAttribution: true }), []);
  const isEmpty = activeTab.nodes.length === 0;

  return (
    <div ref={containerRef} className={styles.canvas}>
      {isEmpty && <EmptyCanvasOverlay />}
      <EdgeLaneProvider edges={activeTab.edges} nodes={activeTab.nodes}>
        <ReactFlow
          id={reactFlowId}
          nodes={activeTab.nodes}
          edges={activeTab.edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={handleConnect}
          onConnectStart={onConnectStart}
          onConnectEnd={onConnectEnd}
          onReconnectStart={onReconnectStart}
          onReconnect={onReconnect}
          onReconnectEnd={onReconnectEnd}
          isValidConnection={handleIsValidConnection}
          connectionLineComponent={CustomConnectionLine}
          onNodeClick={handleNodeClick}
          onEdgeClick={handleEdgeClick}
          onNodeContextMenu={handleNodeContextMenu}
          onPaneContextMenu={handlePaneContextMenu}
          onPaneClick={handlePaneClick}
          onDragOver={onDragOver}
          onDrop={onDrop}
          onMoveStart={() => setCanvasPanning(true)}
          onMoveEnd={() => setCanvasPanning(false)}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          fitView
          minZoom={CANVAS_MIN_ZOOM}
          proOptions={proOptions}
          deleteKeyCode="Delete"
          multiSelectionKeyCode="Shift"
          style={{ background: 'var(--surface-canvas)' }}
          defaultEdgeOptions={{
            animated: false,
            style: { stroke: 'var(--wire)', strokeWidth: 2 },
          }}
          connectionLineStyle={{ stroke: 'var(--wire-active)', strokeWidth: 2 }}
          zoomOnDoubleClick={false}
          snapToGrid={gridSnapEnabled}
          snapGrid={[24, 24]}
        >
          <Background
            color="var(--border-subtle)"
            variant={BackgroundVariant.Dots}
            gap={24}
            size={1.5}
          />
          <SegmentBubble />
          <NoteBindingLines />
          <Controls />
          <MiniMap
            pannable
            zoomable
            position="bottom-right"
            nodeColor={minimapNodeColor}
            maskColor="var(--surface-scrim)"
            style={{ background: 'var(--surface-raised)' }}
          />
        </ReactFlow>
      </EdgeLaneProvider>

      <SubgraphBreadcrumb />

      {contextMenu && (
        <NodeContextMenu
          position={contextMenu}
          items={menuItems}
          onClose={() => setContextMenu(null)}
        />
      )}

      {paneMenu && (
        <PaneContextMenu
          screen={paneMenu.screen}
          flow={paneMenu.flow}
          onClose={() => setPaneMenu(null)}
        />
      )}

      {edgeTooltip && (
        <EdgeDataTooltip
          x={edgeTooltip.x}
          y={edgeTooltip.y}
          sourceLabel={edgeTooltip.sourceLabel}
          targetLabel={edgeTooltip.targetLabel}
          portName={edgeTooltip.portName}
          summary={edgeTooltip.summary}
          onClose={() => setEdgeTooltip(null)}
          onViewStats={
            // No run, no capture, nothing to summarise — so no link either.
            activeTab.lastRunId
              ? () => {
                  openNodeDetail(edgeTooltip.targetId, {
                    tab: 'stats',
                    port: `${edgeTooltip.sourceId}::${edgeTooltip.portName}`,
                  });
                  setEdgeTooltip(null);
                }
              : undefined
          }
        />
      )}

      {quickSearch && (
        <QuickNodeSearch
          screenPos={quickSearch.screen}
          flowPos={quickSearch.flow}
          onClose={() => setQuickSearch(null)}
        />
      )}
    </div>
  );
}
