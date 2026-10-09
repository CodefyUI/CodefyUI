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
  getViewportForBounds,
  useReactFlow,
  useStoreApi,
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
import { withNodeCardBoundaries } from '../Nodes/NodeCardBoundary';
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
import { triggerDropConnection } from './triggerDrop';
import type { FinalConnectionState } from '@xyflow/react';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import { isAnyModalOpen } from '../../store/modalState';
import { useDragAndDrop } from '../../hooks/useDragAndDrop';
import { useDeleteKey } from '../../hooks/useDeleteKey';
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

// Every card draws inside its own NodeCardBoundary: one that throws becomes a
// small box naming the node, and the rest of the canvas keeps working.
const nodeTypes: NodeTypes = withNodeCardBoundaries({
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
});

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

/**
 * React Flow's `onBeforeDelete` for this canvas: no deletion while a modal is
 * open.
 *
 * Delete is heard on `document`, and a modal panel's focus target is a
 * `tabIndex={-1}` div, not a field, so a Delete pressed over a panel the user
 * was reading destroyed the selection on this canvas behind it (#475).
 * Unbinding the key while a modal was up stopped that, but React Flow's own
 * binding then missed the release of the key that opened the panel, and the
 * first Delete after the panel closed did nothing (#491). So the deletion
 * itself is refused, asked at the moment of deleting, which also needs no
 * re-render when a modal opens or closes. The key is the canvas's own now
 * (`useDeleteKey`, #501), and it deletes through `deleteElements`, which
 * asks this.
 */
async function allowDeleteWithNoModalOpen(): Promise<boolean> {
  return !isAnyModalOpen();
}

/**
 * The wire `connection` would repeat, if one is already on the canvas
 * (#619). `ignoreId` is a wire being moved, which is not a copy of itself.
 *
 * A trigger lands on a card's one `__trigger` handle, so any trigger wire
 * between the same two nodes is the same wire, one saved without
 * `sourceHandle` included. A data wire is the same when both its ports are,
 * as the plugin `connect` op tests (`plugins/ops.ts`). A different source into
 * an input that already has one is fan-in, not a copy: that is how branches
 * merge, and the engine and the exported script both read the last source
 * that produced a value.
 */
function duplicateEdgeOf(
  edges: readonly Edge[],
  connection: Pick<Edge, 'source' | 'target' | 'sourceHandle' | 'targetHandle'>,
  ignoreId: string | null,
): Edge | undefined {
  const trigger = connection.targetHandle === '__trigger';
  return edges.find(
    (e) =>
      e.id !== ignoreId &&
      e.source === connection.source &&
      e.target === connection.target &&
      (trigger
        ? e.targetHandle === '__trigger'
        : (e.sourceHandle ?? '') === (connection.sourceHandle ?? '') &&
          (e.targetHandle ?? '') === (connection.targetHandle ?? '')),
  );
}

/*
 * ONLY-RENDER-VISIBLE — why `onlyRenderVisibleElements` is set below (#162).
 *
 * After #125 a 300-node drag still cost 49.3ms p95 against a 32ms budget, and
 * the residue was React Flow re-rendering all 300 node components on every
 * pointermove, including the ones nobody can see.
 *
 * WHERE IT PAYS, and where it does not. The saving is proportional to how much
 * of the graph is off-screen, so it is zero at the zoom `fitView` picks: a
 * fitted graph is on-screen BY CONSTRUCTION, every node passes the visibility
 * test, and the flag only adds the per-frame `getNodesInside` sweep. It pays at
 * the zoom someone actually works at — reading node titles and dragging wires,
 * where most of a large graph is outside the viewport. So a measurement taken
 * right after fit-view will show nothing; that is the geometry, not a broken
 * flag.
 *
 * WHY IT IS SAFE. Culling could lose things that are supposed to survive going
 * off-screen. Checked against @xyflow/react 12.10.1's own selectors rather than
 * assumed:
 *   - MiniMap draws from `s.nodes` (MiniMapNodes' `selectorNodeIds`), not from
 *     the visible set, so it stays complete.
 *   - An edge is kept while the box spanning its two endpoints overlaps the
 *     viewport at all (`isEdgeVisible`), so a wire with one endpoint
 *     off-screen still draws.
 *   - Box selection runs `getNodesInside` over the whole `nodeLookup`, and a
 *     node being dragged is force-rendered (`isVisible || node.dragging`), so
 *     selection and drag do not depend on a node having been rendered.
 *   - A node is force-rendered until it has been measured
 *     (`!node.internals.handleBounds`), so every node is laid out once and its
 *     size is known to layout and to the minimap even if it is never looked at.
 *
 * CULLING STARTS ONLY AFTER THE FIRST FRAME, and this is the part that fools a
 * measurement. A node is force-rendered until React Flow has measured it, and
 * measurement arrives through a ResizeObserver — which Chrome delivers only
 * when the tab gets a rendering opportunity. A hidden, occluded or throttled
 * tab gets none: `requestAnimationFrame` never fires, no node is ever
 * measured, and EVERY node stays mounted with the flag set and correct.
 * Verified in the built app: 320 nodes / 318 mounted before the first frame,
 * 320 measured and 39 mounted after one. So count mounted nodes only in a
 * foreground tab (or force a paint first) — a count taken in a background tab
 * says nothing about culling.
 *
 * WHAT A UNIT TEST CANNOT SEE. jsdom is the same story permanently: it gives
 * an unmeasured node zero area, and zero area passes the visibility test, so
 * all 300 nodes still render there. Edges are the opposite — unmeasured
 * endpoints make every edge test as off-screen — so the perf harness's
 * drag-frame improvement is the edge half under a condition no browser
 * reaches. It is an upper bound, not a prediction; the browser pass is what
 * settles the #162 number.
 *
 * The store side of the same mechanism is load-bearing: `onNodesChange` must
 * keep applying `dimensions` changes, or `measured` never reaches our nodes,
 * @xyflow drops `handleBounds` on the next commit and every node becomes
 * force-rendered again. `tabStore.test.ts` pins that.
 *
 * ONE KNOWN COST. A card that leaves the viewport unmounts, so state local to
 * it resets on the way back (the viz nodes' expanded toggle). NoteNode's
 * mid-edit text is the one case where that could lose data, and NoteNode
 * commits on unmount for exactly this reason — see the comment there.
 */
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
  // Only the request of the tab on screen; another tab's waits for its visit (#522).
  const layoutFitRequest = useUIStore((s) => s.layoutFitRequests[activeTabId]);
  const { screenToFlowPosition, getViewport, setViewport } = useReactFlow();
  const storeApi = useStoreApi();

  const containerRef = useRef<HTMLDivElement>(null);
  const reactFlowId = useId();

  // Fit a flow-space box as an OVERVIEW: small boxes (a single selected node,
  // a two-node graph) are inflated to most of the viewport first, so the fit
  // never zooms in aggressively toward maxZoom.
  //
  // Framed for the container's live size, through `getViewportForBounds`, not
  // by React Flow's `fitBounds`: that one uses the size React Flow's resize
  // observer last reported, which lags a commit that also resizes the canvas
  // — the config panel closing as a tab switch lands — and framed the graph
  // for the narrower canvas, small and at the left edge (#563).
  //
  // Instant, never animated: animated viewport transitions run on
  // requestAnimationFrame, which Chrome throttles to zero in occluded or
  // background windows — the animation then never applies at all. The
  // layers editor's post-layout fit and the Controls button are instant for
  // the same reason.
  //
  // A box worked out from nodes React Flow has not measured yet is an
  // estimate; such a fit is framed again from the measured sizes once React
  // Flow reports them (#622, `refitRef` below).
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
      const { minZoom, maxZoom } = storeApi.getState();
      const box = { x, y, width, height };
      void setViewport(
        getViewportForBounds(box, el.offsetWidth, el.offsetHeight, minZoom, maxZoom, 0.2),
      );
      // The nodes this fit is about are those whose position lies inside the
      // box asked for: every box over nodes that have no size yet covers
      // exactly what was just installed -- a whole graph, or a template
      // `insertGraph` placed clear of the graph already there -- so no caller
      // has to name them. While one of them has no size yet, the fit waits in
      // `refitRef` with the view it set (#622).
      const { tabs, activeTabId: fitTabId } = useTabStore.getState();
      const right = bounds.x + bounds.width;
      const bottom = bounds.y + bounds.height;
      const inside = (tabs.find((t) => t.id === fitTabId)?.nodes ?? []).filter(
        ({ position: p }) => p.x >= bounds.x && p.x <= right && p.y >= bounds.y && p.y <= bottom,
      );
      const unmeasured = inside.some((n) => !n.measured?.width || !n.measured?.height);
      const [vx, vy, zoom] = storeApi.getState().transform;
      refitRef.current = unmeasured
        ? {
            tabId: fitTabId,
            ids: new Set(inside.map((n) => n.id)),
            viewport: { x: vx, y: vy, zoom },
          }
        : null;
    },
    // The store from `useStoreApi` never changes, so this follows `setViewport` alone.
    [setViewport, storeApi],
  );

  // Frame a whole level of a tab: its nodes, or, when it has none, React
  // Flow's default view, where a canvas mounted for it starts (#622). Left at
  // the zoom of whatever was on screen before, an empty tab or block showed
  // the first nodes dropped into it huge.
  const frameLevel = useCallback(
    (nodes: Node[]) => {
      const bounds = nodesBoundingBox(nodes);
      if (bounds) fitToBounds(bounds);
      else void setViewport({ x: 0, y: 0, zoom: 1 });
    },
    [fitToBounds, setViewport],
  );

  // React Flow's own first fit, its `fitView` prop, only for a canvas that
  // mounts on a graph nobody has asked to frame. It waits for the first sizes:
  // on an empty canvas it framed the first node dropped there at full zoom,
  // and over a fit already asked for (a starter opened from the welcome
  // screen) it replaced that overview with a close-up.
  const [fitViewOnMount] = useState(
    () => activeTab.nodes.length > 0 && !useUIStore.getState().layoutFitRequests[activeTabId],
  );

  // ── Per-tab viewport handover (#125) ───────────────────────────────────────
  // One canvas now serves every tab, so switching tabs has to move the
  // viewport by hand: stash where the outgoing tab was looking, put the
  // incoming tab back where IT was. A tab being opened for the first time has
  // nothing stored, so it gets the same overview fit it used to get from its
  // own freshly-mounted provider. A tab with a fit waiting for it (#522) gets
  // that fit instead of either: the box was asked for after the view it had
  // was stored, so that view looks at where the nodes used to be.
  //
  // That first-visit fit is computed from the STORE's node positions rather
  // than asked of React Flow. `fitView()` needs measured nodes, and React Flow
  // has only just been handed the incoming tab's — so it would either fit
  // nothing or have to wait for measurement, during which the user stares at
  // the OUTGOING tab's viewport over the incoming tab's graph. A viewport
  // worked out by `getViewportForBounds` from a box we can compute ourselves
  // lands in the same tick, with no intermediate wrong frame. Sizes fall back
  // to the same defaults the auto-layout fit uses, so an unmeasured node
  // still contributes a box -- an estimate, framed again from the measured
  // sizes once React Flow reports them (#622).
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
    // A re-fit waiting on the outgoing tab's sizes is not this tab's (#622).
    refitRef.current = null;

    rememberViewport(previous, getViewport());
    const pending = useUIStore.getState().layoutFitRequests[activeTabId];
    if (pending) {
      fitToBounds(pending);
      useUIStore.getState().clearLayoutFit(activeTabId);
      return;
    }
    const restored = recallViewport(activeTabId);
    if (restored) {
      setViewport(restored);
      return;
    }
    const incoming = useTabStore.getState().tabs.find((t) => t.id === activeTabId);
    // An empty tab starts at the default view (#622, `frameLevel`).
    frameLevel((incoming?.nodes ?? []) as Node[]);
  }, [activeTabId, getViewport, setViewport, fitToBounds, frameLevel]);

  // ── Entering and leaving a block (#622) ────────────────────────────────────
  // A block opens on this same canvas, at the pan and zoom of the graph around
  // it, and collapse stores its nodes relative to the block's corner, so they
  // sit near the origin and often opened off screen: a blank canvas, without
  // even the empty-block note. So a change of level in the tab on screen
  // frames the level it lands on. Going in, that is the block's nodes,
  // framed again by their measured sizes (`refitRef`). Coming out, it is the
  // view the level had when it was left, kept against the frame that restores
  // that level, or a fit of the level when none was kept (the canvas mounted
  // inside the block). A layout effect, like the handover above, which owns a
  // tab switch: no frame is painted with the wrong level under the old view.
  const levelRef = useRef<{ tabId: string; stack: readonly object[] } | null>(null);
  // Made once, not on every render: this canvas renders on every drag frame.
  const [levelViews] = useState(() => new WeakMap<object, { x: number; y: number; zoom: number }>());
  const depth = activeTab.subgraphStack?.length ?? 0;

  useLayoutEffect(() => {
    const tab = useTabStore.getState().tabs.find((t) => t.id === activeTabId);
    const stack = tab?.subgraphStack ?? [];
    const last = levelRef.current;
    levelRef.current = { tabId: activeTabId, stack };
    if (!last || last.tabId !== activeTabId || last.stack.length === stack.length) return;
    refitRef.current = null;
    if (stack.length > last.stack.length) {
      levelViews.set(stack[last.stack.length], getViewport());
    } else {
      const kept = levelViews.get(last.stack[stack.length]);
      if (kept) {
        void setViewport(kept);
        return;
      }
    }
    frameLevel((tab?.nodes ?? []) as Node[]);
  }, [activeTabId, depth, getViewport, setViewport, frameLevel, levelViews]);

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
  // from React Flow's internal position sync — getViewportForBounds gives the
  // viewport, set at once. (The queued fitView() from useReactFlow only
  // flushes on the next node change, and reading positions back via
  // getNodesBounds races the sync — both failure modes seen in e2e.) Each
  // request names its tab, and this takes only the request of the tab on
  // screen: a request for a tab in the background stays pending until the
  // handover above brings that tab forward (#522). Since #125 only the active
  // tab's canvas is mounted, so the `tabId` check below is belt-and-braces (a
  // harness can still mount several); the one-shot request is cleared once
  // fitted so a remount can't replay it.
  // Read again from the store: the handover may have fitted and cleared the
  // request this render subscribed to.
  useEffect(() => {
    if (!layoutFitRequest) return;
    const el = containerRef.current;
    if (!el || el.offsetWidth === 0) return;
    if (tabId !== undefined && tabId !== activeTabId) return;
    const pending = useUIStore.getState().layoutFitRequests[activeTabId];
    if (!pending) return;
    fitToBounds(pending);
    useUIStore.getState().clearLayoutFit(activeTabId);
  }, [layoutFitRequest, fitToBounds, tabId, activeTabId]);

  // ── Framing again once React Flow has measured (#622) ──────────────────────
  // A graph that was just installed has no sizes in the store: React Flow
  // measures each node through a ResizeObserver a frame later, and only its
  // `dimensions` change writes `measured` (see ONLY-RENDER-VISIBLE above).
  // Until then a node counts at the layout fallback size, so a fit made at
  // once framed an 873 px note as an 80 px card and cut it off at the bottom.
  // `fitToBounds` holds such a fit here, and the effect below frames its nodes
  // again the first time all of them have a size. Once: the re-fit is
  // disarmed as soon as that is decided, and the view moves only if it is
  // still where the fit left it, so a pan or zoom made in between stays and a
  // node resized or dragged later never moves the view. A tab switch drops it,
  // and every fit replaces it.
  const refitRef = useRef<{
    tabId: string;
    ids: ReadonlySet<string>;
    viewport: { x: number; y: number; zoom: number };
  } | null>(null);

  useEffect(() => {
    const pending = refitRef.current;
    if (!pending || pending.tabId !== activeTabId) return;
    const nodes = activeTab.nodes.filter((n) => pending.ids.has(n.id));
    if (nodes.some((n) => !n.measured?.width || !n.measured?.height)) return;
    refitRef.current = null;
    const now = getViewport();
    const moved =
      Math.abs(now.x - pending.viewport.x) > 0.01 ||
      Math.abs(now.y - pending.viewport.y) > 0.01 ||
      Math.abs(now.zoom - pending.viewport.zoom) > 1e-4;
    const bounds = nodesBoundingBox(nodes as Node[]);
    if (!moved && bounds) fitToBounds(bounds);
  }, [activeTab.nodes, activeTabId, getViewport, fitToBounds]);

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
  // Delete, in place of React Flow's own binding (#501).
  useDeleteKey();

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

  // The edge being reconnected, if any. The validity check leaves it out, so
  // the wire in hand is not taken for a copy of itself (#619).
  const reconnectingEdgeRef = useRef<string | null>(null);

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

      // No wire is added twice (#619): not by a snap or release on a handle,
      // a click-to-connect, a moved wire dropped where the same wire already
      // runs, or onConnectEnd's body drop, all of which ask this first. A
      // moved wire refused here is then removed, by onReconnectEnd.
      if (duplicateEdgeOf(tab.edges, edgeOrConnection, reconnectingEdgeRef.current)) return false;

      // Trigger connections (from Start node) are control-flow markers,
      // not data — they connect only to the __trigger handle on target nodes,
      // and that handle takes nothing else. It is hidden but still a snap
      // target inside React Flow's connection radius, so a data wire released
      // near a card's top-left corner was saved onto it, and validation and
      // export then refused the graph (#551). This runs ahead of the port
      // lookups below, which allow what they cannot find.
      //
      // A Start node's wire that names no handle is its trigger too: a trigger
      // edge saved without `sourceHandle` loads with none, and React Flow
      // asks about moving it with none.
      const fromTrigger =
        sourceHandle === 'trigger' || (!sourceHandle && sourceNode?.type === 'start');
      if (fromTrigger || targetHandle === '__trigger') {
        return fromTrigger && targetHandle === '__trigger';
      }

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

  // reconnectingEdgeRef, declared above for the validity check (#619), is set
  // here: onReconnectEnd deletes that edge when it is dropped where it
  // connects to nothing. A reconnect that lands clears it in onReconnect,
  // whether React Flow connected the drop or onConnectEnd moved a trigger wire
  // onto a card's body.
  const onReconnectStart = useCallback((_: any, edge: Edge, handleType: 'source' | 'target') => {
    reconnectingEdgeRef.current = edge.id;
    // Mark the endpoint being detached so its handle shows the red warning
    // ring (dropping on empty space deletes the edge). Note handleType names
    // the end that STAYS connected — see computeDetachedEndpoint.
    useUIStore.getState().setReconnectingHandle(computeDetachedEndpoint(edge, handleType));
  }, []);

  const onReconnect = useCallback((oldEdge: Edge, newConnection: Connection) => {
    // Replace old edge with new connection
    const { setEdges } = useTabStore.getState();
    const tab = useTabStore.getState().tabs.find(
      (t) => t.id === useTabStore.getState().activeTabId,
    );
    // A wire moved where the same wire already runs is refused, as the
    // validity check refuses it (#619): the reconnect is left unfinished, and
    // onReconnectEnd removes the wire. That check comes first for every drop
    // React Flow connects and for onConnectEnd's body drop; this second guard
    // keeps a copy from being stacked if a drop ever lands here unasked.
    if (duplicateEdgeOf(tab?.edges ?? [], newConnection, oldEdge.id)) return;
    reconnectingEdgeRef.current = null;
    // onReconnectEnd always follows and clears too; clearing here as well
    // keeps the indicator lifecycle local to each handler.
    useUIStore.getState().setReconnectingHandle(null);
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

  const onConnectEnd = useCallback(
    (event: MouseEvent | TouchEvent, state: FinalConnectionState) => {
      const ui = useUIStore.getState();
      const draggedTrigger = ui.draggingSourceType === 'TRIGGER';
      ui.setDraggingSourceType(null);
      // While a trigger is dragged every card glows as its drop target, but
      // React Flow connects only near a card's top-left `__trigger` diamond,
      // so a trigger released anywhere else on a card is connected here. A
      // trigger wire being moved lands the same way: React Flow ends a
      // reconnect here too, just before onReconnectEnd would delete the wire.
      if (!draggedTrigger) return;
      const { tabs, activeTabId } = useTabStore.getState();
      const { edges } = tabs.find((t) => t.id === activeTabId)!;
      // Still set when the wire was moved and React Flow did not connect it.
      const moving = reconnectingEdgeRef.current;
      // The wire in hand is not one its Start node already has on a card, so
      // it can be dropped back on its own card.
      const others = moving === null ? edges : edges.filter((e) => e.id !== moving);
      const connection = triggerDropConnection(event, state, others);
      if (!connection || !handleIsValidConnection(connection)) return;
      if (moving === null) {
        handleConnect(connection);
        return;
      }
      const moved = edges.find((e) => e.id === moving);
      if (moved) onReconnect(moved, connection);
    },
    [handleConnect, handleIsValidConnection, onReconnect],
  );

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
    // quickSearch is closed by QuickNodeSearch's own dismissal effect, which
    // listens in the capture phase precisely because this pane swallows
    // mousedown before it can bubble to the document.
  }, [selectNodeExclusively]);

  // A Shift+press on the empty canvas starts React Flow's box selection, but
  // nothing in React Flow cancels the browser's own reading of it -- extend
  // the page's text selection from the last click to here -- so Chrome
  // highlighted the sidebar and the tab bar. Ctrl+C and Ctrl+V yield to
  // selected page text, so the nodes the box then selected could not be
  // copied either (#506). Cancelling the press cancels the focus change it
  // would have made as well, and that is done by hand: focus leaves the field
  // it was in (the palette search box, say), or Delete and the shortcuts
  // would stay with it. Presses on a node or an edge are React Flow's, which
  // cancels them itself; a plain press pans.
  const handleCanvasMouseDown = useCallback((event: React.MouseEvent) => {
    if (!event.shiftKey || event.button !== 0) return;
    if (!(event.target as Element).classList?.contains('react-flow__pane')) return;
    event.preventDefault();
    window.getSelection()?.removeAllRanges();
    (document.activeElement as HTMLElement | null)?.blur?.();
  }, []);

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
    <div ref={containerRef} className={styles.canvas} onMouseDown={handleCanvasMouseDown}>
      {isEmpty && <EmptyCanvasOverlay onDragOver={onDragOver} onDrop={onDrop} />}
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
          // Only when the canvas mounts on a graph (#622, `fitViewOnMount`).
          fitView={fitViewOnMount}
          // Skip the node components the viewport cannot show (#162). See
          // ONLY-RENDER-VISIBLE above the component for why this is safe and
          // where it does and does not pay.
          onlyRenderVisibleElements
          minZoom={CANVAS_MIN_ZOOM}
          proOptions={proOptions}
          // Off: `useDeleteKey` above handles Delete (#501).
          deleteKeyCode={null}
          onBeforeDelete={allowDeleteWithNoModalOpen}
          multiSelectionKeyCode="Shift"
          style={{ background: 'var(--surface-canvas)' }}
          defaultEdgeOptions={{
            animated: false,
            style: { stroke: 'var(--wire)', strokeWidth: 2 },
            // A wire is picked up by its target end only, as dragging a
            // connected input does. React Flow's anchor at the source end lies
            // just outside the output port, under the node layer, so a press
            // just beside a Start node's diamond or an output's dot took the
            // wire, and a release on the empty canvas deleted it (#593).
            reconnectable: 'target',
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
