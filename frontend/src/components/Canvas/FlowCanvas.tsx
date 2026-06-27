import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import {
  ReactFlow,
  MiniMap,
  Background,
  Controls,
  BackgroundVariant,
  useReactFlow,
  type NodeTypes,
  type EdgeTypes,
  type OnConnect,
  type IsValidConnection,
  type Connection,
  type Edge,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { CATEGORY_COLORS } from '../../styles/theme';

import BaseNode from '../Nodes/BaseNode';
import PluginNodeBridge from '../Nodes/PluginNodeBridge';
import PresetNode from '../Nodes/PresetNode';
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
import { NoteBindingLines } from './NoteBindingLines';
import { SegmentBubble } from './SegmentBubble';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import { useDragAndDrop } from '../../hooks/useDragAndDrop';
import { isValidConnection, getPortColor } from '../../utils';
import { prompt } from '../../utils/dialog';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useI18n } from '../../i18n';
import type { OutputSummary } from '../../types';
import styles from './FlowCanvas.module.css';

const nodeTypes: NodeTypes = {
  baseNode: BaseNode,
  pluginNode: PluginNodeBridge,
  presetNode: PresetNode,
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
  if (node.type === 'noteNode') return '#FFD700';
  const data = node.data as any;
  if (data?.isPreset) return '#D4A017';
  const category = data?.definition?.category ?? 'Utility';
  return CATEGORY_COLORS[category] ?? '#607D8B';
};


export function FlowCanvas() {
  const activeTab = useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId)!);
  const onNodesChange = useTabStore((s) => s.onNodesChange);
  const onEdgesChange = useTabStore((s) => s.onEdgesChange);
  const storeOnConnect = useTabStore((s) => s.onConnect);
  const setSelectedNodeId = useTabStore((s) => s.setSelectedNodeId);
  const deleteNode = useTabStore((s) => s.deleteNode);
  const duplicateNode = useTabStore((s) => s.duplicateNode);
  const renameNode = useTabStore((s) => s.renameNode);
  const { t } = useI18n();
  const gridSnapEnabled = useUIStore((s) => s.gridSnapEnabled);
  const setCanvasPanning = useUIStore((s) => s.setCanvasPanning);
  const setNodes = useTabStore((s) => s.setNodes);
  const { screenToFlowPosition } = useReactFlow();

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

  const containerRef = useRef<HTMLDivElement>(null);
  const reactFlowId = useId();

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
        // FlowCanvas only renders with an active tab (line 88 asserts it), so this
        // lookup always finds it; the else branch is never taken
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
          const def = defs.find((d) => d.node_name === srcNode.type);
          const output = def?.outputs.find((o) => o.name === connection.sourceHandle);
          if (output) {
            const color = getPortColor(output.data_type);
            const { setEdges } = useTabStore.getState();
            const tab = useTabStore.getState().tabs.find(
              (t) => t.id === useTabStore.getState().activeTabId,
            );
            // FlowCanvas only renders with an active tab (line 88 asserts it), so
            // reaching here means the lookup above already succeeded; else never taken
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

        const sourceOutput = sourceDef.outputs.find((o) => o.name === sourceHandle);
        const targetInput = targetDef.inputs.find((i) => i.name === targetHandle);
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

  const onReconnectStart = useCallback((_: any, edge: Edge) => {
    reconnectingEdgeRef.current = edge.id;
  }, []);

  const onReconnect = useCallback((oldEdge: Edge, newConnection: Connection) => {
    reconnectingEdgeRef.current = null;
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
    setSelectedNodeId(null);
    setContextMenu(null);
    setPaneMenu(null);
    setEdgeTooltip(null);
    // quickSearch is closed by QuickNodeSearch's own outside-click handler
  }, [setSelectedNodeId]);

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
      setSelectedNodeId(node.id);
      setContextMenu({ nodeId: node.id, x: event.clientX, y: event.clientY });
    },
    [setSelectedNodeId]
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
        proOptions={proOptions}
        deleteKeyCode="Delete"
        multiSelectionKeyCode="Shift"
        style={{ background: '#0a0a0a' }}
        defaultEdgeOptions={{
          animated: false,
          style: { stroke: '#555', strokeWidth: 2 },
        }}
        connectionLineStyle={{ stroke: '#888', strokeWidth: 2 }}
        zoomOnDoubleClick={false}
        snapToGrid={gridSnapEnabled}
        snapGrid={[24, 24]}
      >
        <Background
          color="#2a2a2a"
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
          maskColor="rgba(0,0,0,0.7)"
          style={{ background: '#1e1e1e' }}
        />
      </ReactFlow>

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
