import { create } from 'zustand';
import { applyNodeChanges, applyEdgeChanges } from '@xyflow/react';
import type { Node, Edge, NodeChange, EdgeChange, Connection } from '@xyflow/react';
import {
  generateId,
  buildFlowNode,
  isBypassable,
  resolveDynamicInputs,
  resolveDynamicOutputs,
  resolveSerializedEdges,
  resolveSerializedNodes,
} from '../utils';
import { useNodeDefStore } from './nodeDefStore';
import { forgetViewport } from '../utils/viewportMemory';
import { idbAvailable } from '../utils/idb';
import { readSnapshot, writeSnapshot } from './tabPersistence';
import { autoLayout, autoLayoutWithTargets, nodesBoundingBox, type LayoutMode } from '../utils/autoLayout';
import type { NodeData, NodeDefinition, PresetDefinition, ExecutionStatus, OutputSummary, NodeProgress, SegmentGroup, SubgraphDefinition } from '../types';
import {
  collapseSelection,
  definitionFromCanvas,
  expandInstance,
  instanceDefinition,
  normalizeSubgraphs,
  pruneStaleBoundaryEdges,
  reachableSubgraphIds,
  refreshInstances,
  sameSubgraphs,
  subgraphIdOf,
  type CollapseResult,
} from '../utils/subgraph';
import { ExecutionWebSocket } from '../api/ws';
import { useToastStore } from './toastStore';
import { useUIStore } from './uiStore';
import { useI18n, type TranslationKey } from '../i18n';
import { useProjectStore } from './projectStore';

// ── Per-tab state ──

/**
 * What a log entry carries, mirroring the backend's `output_kind` (#117).
 * A plain string enum, not a closed union of everything the app will ever
 * render, so a later node pack can add its own kind; unknown kinds are
 * ignored by the panel rather than rendered wrong.
 */
export type LogKind = 'text' | 'image' | 'progress' | 'chart' | 'video';

/** Base64 media payload of a `kind: 'image'` entry. */
export interface LogImagePayload {
  /** Image subtype for the data URL (`png`, `svg+xml`, ...). */
  format: string;
  encoding: string;
  data: string;
  /** Output port the image came from, when the backend named one. */
  port?: string;
}

/**
 * Reference payload of a `kind: 'video'` entry (#310). Never the bytes —
 * one node_status event is capped at 128 KB, so the backend writes the
 * file under its media dir and ships where to find it; `url` is served by
 * `/api/media` with a real Content-Type (GETs are unauthenticated reads).
 */
export interface LogVideoPayload {
  /** POSIX-style path relative to the backend's media directory. */
  path: string;
  /** Same-origin URL a <video>/<img> element can point at directly. */
  url: string;
  format: string;
  fps?: number;
  frames?: number;
  width?: number;
  height?: number;
  bytes?: number;
  /** Output port the clip came from, when the backend named one. */
  port?: string;
}

/**
 * Chart spec of a `kind: 'chart'` entry (#130) — the payload a port declaring
 * `media=MEDIA_CHART` produced, verbatim.
 *
 * Deliberately NOT a discriminated union over `kind`: the backend's kinds are
 * open strings, so a pack can ship a fifth one, and a union would make that
 * payload unrepresentable rather than merely unrendered. `ChartView` switches
 * on `kind` and renders nothing for one it does not know.
 *
 * Every number here is finite by the producer's contract — run events are
 * serialised with `allow_nan=False`, so a NaN would already have become
 * `null` upstream. See `plugins/stats/README.md`.
 */
export interface LogChartPayload {
  kind: string;
  title?: string;
  x_label?: string;
  y_label?: string;
  /** Downsampling / substitution notice, shown under the title. */
  note?: string;
  /** `kind: 'bar'` */
  bars?: { label: string; value: number }[];
  /** `kind: 'line'` — each series' points are `[x, y]` pairs. */
  series?: { name: string; points: [number, number][] }[];
  /** `kind: 'scatter'` */
  points?: { x: number; y: number; label?: string; cluster?: number }[];
  /** `kind: 'heatmap'` */
  matrix?: number[][];
  row_labels?: string[];
  col_labels?: string[];
  /** Colour-scale bounds — the scale to read against, not the data's extent. */
  vmin?: number;
  vmax?: number;
  colormap?: string;
  /** Output port the chart came from, when the backend named one. */
  port?: string;
}

export interface LogEntry {
  timestamp: number;
  nodeId?: string;
  message: string;
  type: 'info' | 'error' | 'success';
  /**
   * Structured payload kind (#117). Absent means a plain text message
   * (an app-generated line such as "Execution started"). Before #117
   * images and progress events were smuggled through `message` as
   * `__IMAGE__:` / `__PROGRESS__:` prefixed strings.
   */
  kind?: LogKind;
  /** Set when `kind === 'image'`. */
  image?: LogImagePayload;
  /** Set when `kind === 'progress'` — the raw progress event. */
  progress?: Record<string, any>;
  /** Set when `kind === 'chart'` (#130) — the spec to draw. */
  chart?: LogChartPayload;
  /** Set when `kind === 'video'` (#310) — the clip reference to play. */
  video?: LogVideoPayload;
}

interface UndoSnapshot {
  nodes: Node<NodeData>[];
  edges: Edge[];
  /**
   * Subgraph definitions (core#137). Collapse and expand change the graph
   * AND the definition list in one commit, so a snapshot that carried only
   * nodes/edges would undo a collapse into a canvas holding an instance node
   * whose definition had been taken away.
   */
  subgraphs: SubgraphDefinition[];
}

/**
 * One level of "inside a subgraph" (core#137).
 *
 * Entering swaps the definition's contents into `tab.nodes`/`tab.edges` and
 * stashes what was there. Every existing canvas action -- drag, connect,
 * delete, undo -- then works inside a block with no changes at all, because
 * from their point of view nothing happened. The undo stacks travel with the
 * frame so an undo inside a block can never reach past its own boundary.
 */
interface SubgraphFrame {
  subgraphId: string;
  nodes: Node<NodeData>[];
  edges: Edge[];
  undoStack: UndoSnapshot[];
  redoStack: UndoSnapshot[];
  selectedNodeId: string | null;
  /**
   * The definition list as it stood on ENTRY.
   *
   * Leaving a block pushes ONE undo snapshot describing the whole visit, and
   * a snapshot restores `subgraphs` as well as nodes and edges -- so it needs
   * the pre-entry list, not the post-edit one. It is also what "did anything
   * actually change in there?" is answered against, which is how entering and
   * leaving without editing manages to push nothing at all.
   */
  subgraphs: SubgraphDefinition[];
}

const MAX_UNDO = 50;

/**
 * Batched execution updates: tabId -> nodeId -> patch (#125).
 *
 * The patch type itself lives in `nodeUpdateQueue`, which owns the frame
 * coalescing; the store only knows how to apply a batch. Imported as a type
 * so the module graph stays one-directional (queue -> store) at runtime.
 */
export type PendingNodeUpdates = Map<
  string,
  Map<string, import('./nodeUpdateQueue').PendingNodePatch>
>;

export interface TabState {
  id: string;
  name: string;
  // Graph-level metadata carried through save/load (distinct from the tab
  // label `name`). `description` round-trips to the saved file;
  // `currentGraphFile` is the sanitized stem of the saved graph this tab is
  // bound to (set on load and on save), used to skip the overwrite warning
  // when re-saving the same graph.
  description: string;
  currentGraphFile: string | null;
  // Project directory (absolute path) this tab's bound graph was last saved
  // into or loaded from. `null` for a tab never touched by a project save
  // (e.g. a brand-new tab, or one opened before any project was resolved).
  // Used by saveActiveGraph's cross-project refusal guard (ID10).
  projectOrigin: string | null;
  // True when the loaded graph's format_version is NEWER than this build
  // understands (ID8). The editor opens it read-only -- saveActiveGraph
  // refuses to write it -- so an older CodefyUI build can never
  // destructively down-save a newer file.
  readOnly: boolean;
  // flow
  nodes: Node<NodeData>[];
  edges: Edge[];
  /**
   * Subgraph definitions local to this graph (core#137). Instances reference
   * one by id, so two instances of a definition are the SAME block: editing
   * it changes both, which is the reuse a flattened preset cannot offer.
   */
  subgraphs: SubgraphDefinition[];
  /** Non-empty while the canvas is showing a subgraph's insides. */
  subgraphStack: SubgraphFrame[];
  selectedNodeId: string | null;
  presetModalNodeId: string | null;
  /**
   * Node whose *layers* editor is open, or null (core#199).
   *
   * Unrelated to `subgraphs` / `subgraphStack` above, which are graph nesting
   * (core#137). This one drives `components/LayersEditor`, a modal that edits
   * one SequentialModel node's `layers` JSON param — a list of neural-network
   * layers, not a nested graph. The two features shared the word "subgraph"
   * until the rename.
   */
  layersModalNodeId: string | null;
  /**
   * Node whose detail modal (#127) is open, or null. Transient like the other
   * two modal ids above — never persisted, so a reload lands on the canvas
   * rather than on top of a modal the user has no memory of opening.
   */
  nodeDetailNodeId: string | null;
  /**
   * Tab id the detail modal should open on, or null for its default (#129).
   * Set by the edge tooltip's "View stats" link; every other entry point
   * writes null, so a deep link never becomes sticky.
   */
  nodeDetailTab: string | null;
  /**
   * Port the modal was opened *about*, as `nodeId::port`, or null. Handed to
   * every tab so a list-of-ports tab can scroll to the one that was asked for.
   */
  nodeDetailPort: string | null;
  /**
   * Bumped on every `openNodeDetail` call. The modal watches it so a SECOND
   * deep link into the node it is already showing still lands: following
   * "View stats" from another edge into the same consumer changes neither the
   * node id nor the requested tab, and an effect keyed on those alone would
   * see nothing happen and leave the user on whatever tab they had wandered
   * to (#129).
   */
  nodeDetailRequest: number;
  // undo/redo
  undoStack: UndoSnapshot[];
  redoStack: UndoSnapshot[];
  // dirty tracking for partial re-execution
  dirtyNodeIds: Set<string>;
  // execution
  status: ExecutionStatus;
  logs: LogEntry[];
  ws: ExecutionWebSocket;
  // output summaries per node (for edge inspection)
  outputSummaries: Record<string, Record<string, OutputSummary>>;
  // Teaching Inspector state
  recordOutputs: boolean;
  lastRunId: string | null;
  // #121: highest event cursor this tab has rendered for `lastRunId`. Used
  // to resume an attach after a dropped socket without replaying history
  // that is already on screen. In-memory only — a page reload starts with
  // an empty log panel, so it re-attaches from 0 and replays everything.
  lastRunCursor: number;
  activeSegment: SegmentGroup | null;
  segmentGroups: SegmentGroup[];
  // A1: verbose / step-trace mode
  verboseMode: boolean;
  // A2: per-node weight persistence — graphId is a stable per-tab UUID
  // sent to the backend so NodeStateStore can key persistent layer weights
  // even if the tab is renamed or the user closes/reopens the workspace.
  graphId: string;
  weightsPersistent: boolean;
  // A3: gradient capture
  backwardMode: boolean;
  autoBackward: boolean;
  // core#134: reproducibility. `null` means "no seed" — the run uses
  // whatever entropy torch picks, which is the historical behaviour and
  // stays the default. A number makes the run reproducible: every node is
  // seeded from it and the engine executes serially.
  seed: number | null;
  deterministic: boolean;
}

function createTabState(id: string, name: string): TabState {
  return {
    id,
    name,
    description: '',
    currentGraphFile: null,
    projectOrigin: null,
    readOnly: false,
    nodes: [],
    edges: [],
    subgraphs: [],
    subgraphStack: [],
    selectedNodeId: null,
    presetModalNodeId: null,
    layersModalNodeId: null,
    nodeDetailNodeId: null,
    nodeDetailTab: null,
    nodeDetailPort: null,
    nodeDetailRequest: 0,
    undoStack: [],
    redoStack: [],
    dirtyNodeIds: new Set(),
    status: 'idle',
    logs: [],
    ws: new ExecutionWebSocket(),
    outputSummaries: {},
    recordOutputs: true,
    lastRunId: null,
    lastRunCursor: 0,
    activeSegment: null,
    segmentGroups: [],
    verboseMode: false,
    graphId:
      typeof crypto !== 'undefined' && 'randomUUID' in crypto
        ? crypto.randomUUID()
        : `graph-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
    weightsPersistent: true,
    backwardMode: false,
    autoBackward: false,
    seed: null,
    deterministic: false,
  };
}

// ── Store ──

interface TabStoreState {
  tabs: TabState[];
  activeTabId: string;

  // tab management
  addTab: (name?: string) => void;
  removeTab: (id: string) => void;
  setActiveTab: (id: string) => void;
  renameTab: (id: string, name: string) => void;
  // graph-level metadata (active tab)
  setDescription: (description: string) => void;
  setCurrentGraphFile: (file: string | null) => void;
  setTabReadOnly: (v: boolean) => void;
  // Per-project persistence scoping (ID10)
  rehydrateForProject: (projectId: string | null) => void;
  stampActiveTabProject: (projectId: string | null) => void;

  // flow actions (operate on active tab)
  setNodes: (nodes: Node<NodeData>[]) => void;
  setEdges: (edges: Edge[]) => void;
  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection) => void;
  addNode: (definition: NodeDefinition, position: { x: number; y: number }) => void;
  addPresetNode: (preset: PresetDefinition, position: { x: number; y: number }) => void;
  updateNodeParams: (nodeId: string, params: Record<string, any>) => void;
  updatePresetInternalParam: (nodeId: string, internalNodeId: string, paramName: string, value: any) => void;
  setSelectedNodeId: (id: string | null) => void;
  selectNodeExclusively: (id: string | null) => void;
  openPresetModal: (id: string) => void;
  closePresetModal: () => void;
  /* The layers editor (core#199) — one node's `layers` param. Not nesting;
     the graph-nesting actions are `enterSubgraph` / `exitSubgraph` / friends
     further down. */
  openLayersModal: (id: string) => void;
  closeLayersModal: () => void;
  openNodeDetail: (id: string, target?: { tab?: string; port?: string }) => void;
  closeNodeDetail: () => void;
  updateNodeLayers: (nodeId: string, layersJson: string) => void;
  setNodeExecutionStatus: (nodeId: string, status: NodeData['executionStatus'], error?: string) => void;
  clearExecutionStatus: () => void;
  clear: () => void;
  getSerializedGraph: () => {
    nodes: any[];
    edges: any[];
    // None of these are optional: the serializer always answers with a list,
    // so callers do not each have to re-establish that (core#137 review
    // round 1 for `subgraphs`; round 2 for the other two, which were the
    // same latent trap one field over -- `presets` in particular became
    // load-bearing once a preset could live inside a block, and an
    // `| undefined` on it is how a caller ends up quietly skipping it).
    presets: import('../types').PresetDefinition[];
    segmentGroups: SegmentGroup[];
    subgraphs: SubgraphDefinition[];
  };
  /**
   * Collapse the canvas selection into one subgraph instance (core#137).
   *
   * ONE undo step: the whole next state -- nodes, edges and the definition
   * list -- is computed first and committed in a single `set`.
   */
  /** Replace the definition list wholesale (load / import). */
  setSubgraphs: (subgraphs: SubgraphDefinition[]) => void;
  collapseSelectionToSubgraph: (name?: string) => CollapseResult;
  /** Put an instance's definition back on the canvas. One undo step. */
  expandSubgraphInstance: (nodeId: string) => boolean;
  /** Open an instance's definition on the canvas (breadcrumb push). */
  enterSubgraph: (nodeId: string) => boolean;
  /**
   * Leave the innermost sub-canvas, writing the edit back. One undo step,
   * but only when the definitions actually changed while inside.
   */
  exitSubgraph: () => void;
  /**
   * Jump straight back to the top level from any depth, writing every level's
   * edit back on the way. Driven by the breadcrumb's "Main" button, which is
   * its only caller -- saving and running do NOT come through here, they use
   * the pure `flushSubgraphEditing` and leave the user where they are
   * standing. Like `exitSubgraph`, one undo step for the whole exit.
   */
  exitAllSubgraphs: () => void;
  /** Rename the subgraph currently being edited. */
  renameSubgraph: (subgraphId: string, name: string) => void;
  deleteNode: (nodeId: string) => void;
  duplicateNode: (nodeId: string) => void;
  renameNode: (nodeId: string, newLabel: string) => void;
  applyLayout: (mode: LayoutMode) => void;
  /** Toggle ComfyUI-style bypass on one node (core#128). */
  toggleNodeBypass: (nodeId: string) => void;
  /**
   * Toggle bypass across the canvas selection. Returns false when nothing in
   * the selection can be bypassed, which is what lets Ctrl+B fall through to
   * the sidebar shortcut instead of doing nothing at all.
   */
  toggleBypassForSelection: () => boolean;
  /**
   * Merge a template's nodes/edges into the active tab (core#128).
   * Paste-style: fresh ids, placed clear of what is already on the canvas,
   * one undo step for the whole insertion.
   *
   * `subgraphs` (core#137) are the template's own definitions. Definition
   * ids are NOT remapped the way node ids are -- an id is what an instance
   * node names -- so a collision is resolved the way paste resolves it: the
   * definition already in this tab wins.
   */
  insertGraph: (
    nodes: Node<NodeData>[],
    edges: Edge[],
    subgraphs?: SubgraphDefinition[],
  ) => void;

  // note actions
  addNote: (kind: 'text' | 'image', position: { x: number; y: number }) => void;
  updateNoteData: (nodeId: string, updates: Partial<Pick<NodeData, 'noteContent' | 'noteColor' | 'boundToNodeId' | 'boundOffset' | 'noteWidth' | 'noteHeight'>>) => void;
  bindNoteToNode: (noteId: string, targetNodeId: string) => void;
  bindNoteToNearest: (noteId: string) => void;
  unbindNote: (noteId: string) => void;

  // undo/redo
  pushUndoSnapshot: () => void;
  undo: () => void;
  redo: () => void;

  // clipboard (copy/paste)
  clipboard: {
    nodes: Node<NodeData>[];
    edges: Edge[];
    /**
     * Definitions for any subgraph instance in the copied block (core#137).
     * Without them a paste into ANOTHER tab lands an instance whose
     * definition does not exist there -- a node the canvas can draw and the
     * server refuses to run.
     */
    subgraphs?: SubgraphDefinition[];
  } | null;
  copySelectedNodes: () => void;
  pasteNodes: () => void;

  // dirty tracking for partial re-execution
  markDirty: (nodeId: string) => void;
  clearDirty: () => void;
  getDirtyWithDownstream: () => string[];

  // execution actions (operate on active tab)
  setStatus: (s: ExecutionStatus) => void;
  addLog: (entry: Omit<LogEntry, 'timestamp'>) => void;
  clearLogs: () => void;

  // helpers
  getActiveTab: () => TabState;
  getTab: (id: string) => TabState | undefined;

  // execution actions for specific tab (used by WS handlers)
  applyTabNodeUpdates: (updates: PendingNodeUpdates) => void;
  setTabNodeExecutionStatus: (tabId: string, nodeId: string, status: NodeData['executionStatus'], error?: string) => void;
  setTabNodeProgress: (tabId: string, nodeId: string, progress: NodeProgress) => void;
  setTabOutputSummary: (tabId: string, nodeId: string, summary: Record<string, OutputSummary>) => void;
  clearOutputSummaries: () => void;
  setTabStatus: (tabId: string, s: ExecutionStatus) => void;
  addTabLog: (tabId: string, entry: Omit<LogEntry, 'timestamp'>) => void;

  // Teaching Inspector actions
  toggleRecord: () => void;
  setLastRunId: (tabId: string, runId: string | null) => void;
  setLastRunCursor: (tabId: string, cursor: number) => void;
  setActiveSegment: (segment: SegmentGroup | null) => void;
  addSegmentGroup: (segment: SegmentGroup) => void;
  removeSegmentGroup: (id: string) => void;
  setSegmentGroups: (segments: SegmentGroup[]) => void;
  // A1/A2/A3 toggles
  toggleVerbose: () => void;
  togglePersistWeights: () => void;
  toggleBackward: () => void;
  toggleAutoBackward: () => void;
  /** core#134: `null` clears the seed; a number makes the run reproducible. */
  setSeed: (seed: number | null) => void;
  toggleDeterministic: () => void;
}

function updateTab(tabs: TabState[], tabId: string, updater: (tab: TabState) => Partial<TabState>): TabState[] {
  return tabs.map((tab) => (tab.id === tabId ? { ...tab, ...updater(tab) } : tab));
}

/**
 * Point React Flow's own per-node `.selected` flag at exactly `id`,
 * deselecting every other node -- UNLESS `id` already names a member of an
 * existing multi-selection (two or more `.selected` nodes), in which case
 * the whole selection is left alone (#167).
 *
 * Used by the two PROGRAMMATIC selection paths, the ones React Flow itself
 * has no opinion about: `selectNodeExclusively` (a right-click, the
 * ResultsPanel's "click to highlight", or any other store-driven "make this
 * the selection") and `openNodeDetail` (the detail modal's arrow keys). That
 * makes `selectedNodeId` and `.selected` agree by construction on those two
 * paths. Deliberately NOT used by `setSelectedNodeId` (the plain-click
 * path) -- see its own comment for why running this on a click would be
 * actively wrong, not just redundant.
 *
 * The multi-selection exception exists because `.selected` is not only the
 * Delete-key target -- it is the app's OWN multi-selection, read by four
 * bulk operations (`collapseSelectionToSubgraph`, `toggleBypassForSelection`,
 * selection-scoped auto-layout, `copySelectedNodes`). Without it, right-
 * clicking a node inside a box-selection collapsed the selection down to
 * that one node before the context menu could read it -- which silently
 * removed `NodeContextMenu`'s only entry point to "Collapse to subgraph"
 * (#198) whenever it was opened via right-click. A lone selected node, or a
 * target outside the current selection, still narrows normally -- so the
 * modal's arrow-key target and a right-clicked non-member are unaffected,
 * which is what fixed #167 in the first place.
 */
function selectOnlyNode(nodes: Node<NodeData>[], id: string | null): Node<NodeData>[] {
  if (id !== null) {
    const selected = nodes.filter((n) => n.selected);
    if (selected.length >= 2 && selected.some((n) => n.id === id)) return nodes;
  }
  return nodes.map((n) => {
    const isTarget = n.id === id;
    return n.selected === isTarget ? n : { ...n, selected: isTarget };
  });
}

/**
 * Land a block of foreign nodes plus the definitions they name (core#137).
 *
 * Shared by paste and template insert, which are the same operation with
 * different sources -- and which had already drifted apart once, so one
 * implementation is the point.
 *
 * Node ids are remapped by the caller; DEFINITION ids are not, because an id
 * is what an instance node names. So a collision has to be resolved rather
 * than avoided, and the local definition wins: it is the one this tab's
 * other instances are sharing, and replacing it would edit them from a
 * clipboard. That decision then has a consequence the incoming nodes must
 * follow -- an instance cloned with the FOREIGN definition would keep the
 * foreign label and rendered ports while resolving to the local block, so
 * the canvas paints handles the block does not have. Re-render every
 * incoming instance from the definition that actually won.
 */
function mergeIncomingSubgraphs(
  existing: SubgraphDefinition[],
  incomingRaw: SubgraphDefinition[],
  incomingNodes: Node<NodeData>[],
): { subgraphs: SubgraphDefinition[]; nodes: Node<NodeData>[] } {
  // The second door for a list the store did not build: `insertGraph` gets
  // an example's definitions straight from the fetched file without passing
  // `setSubgraphs`, and paste reads the system clipboard. `existing` is
  // already normalized, by whichever door it came through.
  const incoming = normalizeSubgraphs(incomingRaw);
  const subgraphs = incoming.length
    ? [
        ...existing,
        ...incoming.filter((d) => !existing.some((e) => e.id === d.id)),
      ]
    : existing;
  const byId = new Map(subgraphs.map((d) => [d.id, d]));
  const nodes = incomingNodes.map((n) => {
    const sid = subgraphIdOf(n.data?.type);
    if (sid === null) return n;
    const definition = byId.get(sid);
    // A reference to a definition nobody carries is left exactly as it is:
    // it is already broken, and rewriting it would only hide that.
    if (!definition) return n;
    return {
      ...n,
      data: {
        ...n.data,
        label: definition.name || definition.id,
        definition: instanceDefinition(definition),
        subgraphId: definition.id,
      },
    };
  });
  return { subgraphs, nodes };
}

/**
 * The tab as it would be with every sub-canvas closed (core#137).
 *
 * Pure -- it does not touch the store. Save, autosave and Run all go through
 * it, so a graph is written and executed identically whether the user is at
 * the top level or three blocks deep. Returns the SAME object when nothing
 * is open, so the persistence cache's identity compare still hits.
 */
export function flushSubgraphEditing(tab: TabState): TabState {
  // Optional-chained: tests and older persisted records build tab objects
  // without the field, and a save path is the wrong place to throw.
  if (!tab.subgraphStack?.length) return tab;
  let nodes = tab.nodes;
  let edges = tab.edges;
  let subgraphs = tab.subgraphs;
  for (let level = tab.subgraphStack.length - 1; level >= 0; level -= 1) {
    const frame = tab.subgraphStack[level];
    const definition = subgraphs.find((d) => d.id === frame.subgraphId);
    if (definition) {
      const updated = definitionFromCanvas(definition, nodes, edges);
      subgraphs = subgraphs.map((d) => (d.id === updated.id ? updated : d));
      nodes = refreshInstances(frame.nodes, updated);
      edges = pruneStaleBoundaryEdges(nodes, frame.edges, subgraphs);
    } else {
      nodes = frame.nodes;
      edges = frame.edges;
    }
  }
  return { ...tab, nodes, edges, subgraphs, subgraphStack: [] };
}

/**
 * The undo/redo stacks to restore when a sub-canvas closes (core#137 review).
 *
 * Leaving a block puts the OUTER history back -- an undo inside a block must
 * never reach past its own boundary, so the inner stack is thrown away. But
 * the visit also COMMITS the edited definition into the graph, and for a long
 * time it committed it with no undo entry behind it. The next Ctrl+Z then
 * skipped straight to whatever the user had done before entering, undid THAT,
 * and silently took the block edit along with it (a snapshot restores
 * `subgraphs` too). One button press, two changes reverted, neither of them
 * the one the user was looking at.
 *
 * So: if the definitions actually changed in there, push exactly one snapshot
 * of the state as it was on ENTRY. Leaving a block is then one undoable step,
 * the same as collapse and expand already were, and the SECOND undo is what
 * reaches the outer edit.
 *
 * `sameSubgraphs` rather than an identity check because the exit path rebuilds
 * the definition from the canvas unconditionally: `definitionFromCanvas`
 * re-derives positions and re-serializes every edge, so it returns a fresh
 * object even when nothing moved. Without a structural compare, merely LOOKING
 * inside a block would cost the user a phantom undo step.
 */
function closeFrameHistory(
  frame: SubgraphFrame,
  nextSubgraphs: SubgraphDefinition[],
): Pick<TabState, 'undoStack' | 'redoStack'> {
  if (sameSubgraphs(frame.subgraphs, nextSubgraphs)) {
    return { undoStack: frame.undoStack, redoStack: frame.redoStack };
  }
  const snapshot: UndoSnapshot = {
    nodes: [...frame.nodes],
    edges: [...frame.edges],
    subgraphs: [...frame.subgraphs],
  };
  return {
    undoStack: [...frame.undoStack.slice(-(MAX_UNDO - 1)), snapshot],
    // Same reason `pushUndoSnapshot` clears it: the redo entries describe a
    // future that branched off before this change and can no longer be
    // reached from here.
    redoStack: [],
  };
}

const NO_STALE_EDGES: ReadonlySet<string> = new Set<string>();

/**
 * Ids of edges a param edit would leave hanging off a handle that no longer
 * exists (core#131).
 *
 * Only ports the resolver itself produced are candidates, so trigger handles
 * and preset ports — which this knows nothing about — are never touched. For
 * every node whose ports do NOT depend on params the two resolvers hand back
 * the definition's own arrays, so the identity check below settles it in two
 * comparisons; the set work only runs for the handful of nodes the two
 * resolvers actually expand (Split, PythonScript, ComposeTransform today).
 */
function staleEdges(
  node: Node<NodeData> | undefined,
  nextParams: Record<string, unknown>,
  edges: Edge[],
): ReadonlySet<string> {
  const def = node?.data.definition;
  if (!node || !def) return NO_STALE_EDGES;

  const inputsBefore = resolveDynamicInputs(def, node.data.params);
  const inputsAfter = resolveDynamicInputs(def, nextParams);
  const outputsBefore = resolveDynamicOutputs(def, node.data.params);
  const outputsAfter = resolveDynamicOutputs(def, nextParams);
  if (inputsBefore === inputsAfter && outputsBefore === outputsAfter) {
    return NO_STALE_EDGES;
  }

  const names = (ports: { name: string }[]) => new Set(ports.map((p) => p.name));
  const hadInput = names(inputsBefore);
  const hasInput = names(inputsAfter);
  const hadOutput = names(outputsBefore);
  const hasOutput = names(outputsAfter);

  const stale = new Set<string>();
  for (const edge of edges) {
    const from = edge.sourceHandle;
    const to = edge.targetHandle;
    if (edge.source === node.id && from && hadOutput.has(from) && !hasOutput.has(from)) {
      stale.add(edge.id);
    }
    if (edge.target === node.id && to && hadInput.has(to) && !hasInput.has(to)) {
      stale.add(edge.id);
    }
  }
  return stale;
}

// ── Serialization helpers ──
//
// Node positions serialize as integers so drag micro-movements don't produce
// noisy floating-point diffs in saved / exported graph JSON. Loading still
// tolerates floats.
function roundPosition(p: { x: number; y: number } | undefined): { x: number; y: number } | undefined {
  // Tolerate a missing position (some callers/tests build nodes without one);
  // the previous serializer passed `n.position` through verbatim.
  if (!p) return p;
  return { x: Math.round(p.x), y: Math.round(p.y) };
}

// Like roundPosition but preserves an explicit `null` (a note's boundOffset is
// `{x,y} | null`, where null means "unbound"). Rounds sub-pixel drag offsets
// so serialized note data doesn't carry noisy floats.
function roundOffset(
  p: { x: number; y: number } | null | undefined,
): { x: number; y: number } | null | undefined {
  if (!p) return p;
  return { x: Math.round(p.x), y: Math.round(p.y) };
}

// Round an optional pixel dimension (note width/height), tolerating undefined.
function roundDimension(v: number | undefined): number | undefined {
  return typeof v === 'number' ? Math.round(v) : v;
}

// ── Bypass (core#128) ──
//
// One patch builder for both entry points (single node, whole selection) so
// the node write and the dirty marking always land in the SAME commit — two
// separate `set` calls would re-render the canvas twice per keypress.
function bypassPatch(
  tab: TabState,
  ids: ReadonlySet<string>,
  bypassed: boolean,
): Partial<TabState> {
  const dirtyNodeIds = new Set(tab.dirtyNodeIds);
  for (const id of ids) dirtyNodeIds.add(id);
  return {
    nodes: tab.nodes.map((n) =>
      ids.has(n.id) ? { ...n, data: { ...n.data, bypassed } } : n,
    ),
    dirtyNodeIds,
  };
}

/** Gap between the existing graph and an inserted template, in flow pixels. */
const INSERT_GAP = 96;

/**
 * Where to drop an inserted template so it lands clear of the current graph:
 * left-aligned with what is already there, one gap below its lowest node. The
 * template keeps its own internal layout — only the whole block moves.
 */
function insertionOffset(
  existing: Node<NodeData>[],
  incoming: Node<NodeData>[],
): { x: number; y: number } {
  const target = nodesBoundingBox(existing as Node[]);
  const source = nodesBoundingBox(incoming as Node[]);
  // An empty canvas (or a template with nothing in it) needs no move at all.
  if (!target || !source) return { x: 0, y: 0 };
  return {
    x: target.x - source.x,
    y: target.y + target.height + INSERT_GAP - source.y,
  };
}

// Replace every SECRET-typed param value with '' so secrets (e.g. an LLM API
// key typed into the canvas) never reach a saved file or exported JSON. The
// node definition (attached by buildFlowNode / resolveSerializedNodes) tells
// us which params are secret. The backend save endpoint re-scrubs as
// defense-in-depth; this is the primary strip.
function stripSecretParams(
  params: Record<string, any>,
  definition: NodeDefinition | undefined,
): Record<string, any> {
  if (!params) return params;
  const secretNames = (definition?.params ?? [])
    .filter((p) => p.param_type === 'secret')
    .map((p) => p.name);
  if (secretNames.length === 0) return params;
  const cleaned = { ...params };
  for (const name of secretNames) {
    if (name in cleaned) cleaned[name] = '';
  }
  return cleaned;
}

// Blank SECRET-typed values embedded in a preset node's `internalParams`.
// A preset's `exposed_params` carry each exposed param's `param_def`, so a
// param exposed as `secret` (an OLD preset created before secrets were
// withheld from presets — see backend routes_presets) pins the exact
// (internal_node, param_name) slot to blank. New presets expose no secret at
// all, so nothing matches and internalParams passes through untouched. Only
// the secret slots are blanked; every other inner override persists. Returns
// the same reference when there is nothing to strip, so callers can cheaply
// detect a no-op.
function stripSecretInternalParams(
  internalParams: Record<string, Record<string, any>> | undefined,
  preset: PresetDefinition | undefined,
): Record<string, Record<string, any>> | undefined {
  if (!internalParams || !preset) return internalParams;
  const secretSlots = preset.exposed_params.filter(
    (ep) => ep.param_def?.param_type === 'secret',
  );
  if (secretSlots.length === 0) return internalParams;
  let cleaned: Record<string, Record<string, any>> | null = null;
  for (const ep of secretSlots) {
    const inner = internalParams[ep.internal_node];
    if (inner && ep.param_name in inner) {
      if (cleaned === null) cleaned = { ...internalParams };
      cleaned[ep.internal_node] = { ...inner, [ep.param_name]: '' };
    }
  }
  return cleaned ?? internalParams;
}

// Blank every SECRET-typed value inside SUBGRAPH DEFINITIONS (core#137).
//
// A node that lives inside a collapsed block is an ordinary node holding an
// ordinary API key; nothing about being in a block makes it less of a secret.
// Two things make this a separate function rather than a reuse of the node
// strip above:
//
//  - a definition's entries are the SERIALIZED shape (`type` at the top
//    level, no attached `data.definition`), so the param schema has to be
//    resolved through the node registry the way `resolveSerializedNodes`
//    does, and a preset among them through the preset registry;
//  - `subgraphs` is a FLAT list — a block inside a block is a
//    `subgraph:<id>` REFERENCE into this same list — so one pass over the
//    list reaches every node at every depth, with no recursion to get wrong.
//
// Returns the SAME array (and the same definition objects) when there is
// nothing to strip, so the persistence record cache stays a pointer compare.
function stripSubgraphSecrets(
  subgraphs: SubgraphDefinition[],
): SubgraphDefinition[] {
  if (!subgraphs.length) return subgraphs;
  const { definitions, presets } = useNodeDefStore.getState();
  const defByName = new Map(definitions.map((d) => [d.node_name, d]));
  const presetByName = new Map(presets.map((p) => [p.preset_name, p]));
  let listChanged = false;
  const next = subgraphs.map((definition) => {
    let changed = false;
    const nodes = definition.nodes.map((raw: any) => {
      const type: string = raw?.type ?? '';
      const data = raw?.data;
      if (!data) return raw;
      const params = stripSecretParams(data.params, defByName.get(type));
      const internalParams = type.startsWith('preset:')
        ? stripSecretInternalParams(
            data.internalParams,
            presetByName.get(type.slice('preset:'.length)),
          )
        : data.internalParams;
      if (params === data.params && internalParams === data.internalParams) {
        return raw;
      }
      changed = true;
      return {
        ...raw,
        data: {
          ...data,
          ...(params !== undefined ? { params } : {}),
          ...(internalParams !== undefined ? { internalParams } : {}),
        },
      };
    });
    if (!changed) return definition;
    listChanged = true;
    return { ...definition, nodes };
  });
  return listChanged ? next : subgraphs;
}

// Return a copy of `nodes` with every SECRET-typed value blanked in both
// `data.params` (via the node definition) and, for preset nodes,
// `data.internalParams` (via the preset's exposed_params). Nodes with no
// secret are returned by identity so persistence stays cheap. Used before
// writing to localStorage so a typed API key never survives a page refresh —
// honouring the field's "Session only" promise (a refresh drops typed keys).
function stripNodeSecretsForPersist(
  nodes: Node<NodeData>[],
): Node<NodeData>[] {
  return nodes.map((n) => {
    const params = stripSecretParams(n.data.params, n.data.definition);
    const internalParams = n.data.isPreset
      ? stripSecretInternalParams(n.data.internalParams, n.data.presetDefinition)
      : n.data.internalParams;
    if (params === n.data.params && internalParams === n.data.internalParams) {
      return n;
    }
    return {
      ...n,
      data: {
        ...n.data,
        params,
        ...(n.data.isPreset ? { internalParams } : {}),
      },
    };
  });
}

// ── Persistence ──
//
// Two tiers since #125. IndexedDB is the store of record: one record per tab,
// structured-cloned rather than stringified, with no practical size limit.
// localStorage is what the app used before, and stays on as (a) the source
// the first IndexedDB load migrates FROM, and (b) the write target when
// IndexedDB is unavailable — private-browsing modes, sandboxed frames, and
// jsdom, which has none at all.
//
// Both tiers key off the same project-scoped `_storageKey()`, and both store
// the same `PersistedTab` shape, so the migration is a straight copy and a
// downgrade still finds the last blob localStorage held.

const STORAGE_KEY_BASE = 'codefyui-tabs';

// Persistence key is scoped to the active project so `--project B` never
// resurrects project A's tabs (ID10). Non-project mode keeps the bare base key
// -> byte-for-byte unchanged.
function _storageKey(): string {
  const pid = useProjectStore.getState().projectDir;
  return pid ? `${STORAGE_KEY_BASE}::${pid}` : STORAGE_KEY_BASE;
}

/**
 * One tab as it is written to storage. Exported because `tabPersistence`
 * stores these as individual IndexedDB records (#125) and needs the shape.
 */
export interface PersistedTab {
  id: string;
  name: string;
  description?: string;
  currentGraphFile?: string | null;
  projectOrigin?: string | null;
  readOnly?: boolean;
  nodes: Node<NodeData>[];
  edges: Edge[];
  segmentGroups?: SegmentGroup[];
  /**
   * Subgraph definitions (core#137). Absent on a graph that has none, so a
   * workspace nobody has collapsed anything in persists byte-identically.
   *
   * The EDITING STACK is deliberately not persisted: a reload puts you back
   * at the top level of the graph, with every block edit already folded
   * into its definition by flushSubgraphEditing.
   */
  subgraphs?: SubgraphDefinition[];
  recordOutputs?: boolean;
  /**
   * #121: the run this tab was watching, persisted ONLY while it might
   * still be alive (see saveTabs). On the next load the hook asks the
   * server whether it really is, and re-attaches if so — which is how
   * "close the tab, reopen it, the training is still going" works.
   */
  lastRunId?: string;
  verboseMode?: boolean;
  seed?: number | null;
  deterministic?: boolean;
  graphId?: string;
  weightsPersistent?: boolean;
  backwardMode?: boolean;
  autoBackward?: boolean;
}

// Throttle the user-facing persistence warnings so a long editing session
// doesn't burst N toasts while storage stays broken. One per minute is plenty
// to surface "your work isn't being saved". Keyed by message, so a failing
// READ and a failing WRITE cannot silence each other — they are different
// problems with different advice.
const _lastPersistenceWarn = new Map<string, number>();

function warnPersistence(messageKey: TranslationKey): void {
  const now = Date.now();
  if (now - (_lastPersistenceWarn.get(messageKey) ?? 0) <= 60_000) return;
  // Opened before the toast is attempted, so a throwing i18n/toast layer
  // cannot turn this into a per-save retry loop.
  _lastPersistenceWarn.set(messageKey, now);
  try {
    useToastStore.getState().addToast(useI18n.getState().t(messageKey), 'error');
  } catch {
    /* toast/i18n not initialised yet — nothing useful to do here */
  }
}

/** Build the storage record for one tab. */
function buildPersistedTab(input: TabState): PersistedTab {
  // Autosave stores the whole graph, never the sub-canvas the user is looking
  // at: a refresh mid-edit must not turn a block's insides into the tab.
  const t = flushSubgraphEditing(input);
  return {
    id: t.id,
    name: t.name,
    description: t.description,
    currentGraphFile: t.currentGraphFile,
    // Only persisted when set, so non-project localStorage is byte-identical.
    ...(t.projectOrigin != null ? { projectOrigin: t.projectOrigin } : {}),
    // Only persisted when true, so an editable graph's localStorage shape
    // stays byte-identical to before this task.
    ...(t.readOnly ? { readOnly: true } : {}),
    // Never persist SECRET param values (typed API keys): they must not
    // survive a page refresh — the field is "Session only". Applied to
    // BOTH places this record carries nodes; stripping only the top level
    // means the promise above holds or not depending on whether the user
    // happened to collapse the node holding the key into a block.
    nodes: stripNodeSecretsForPersist(t.nodes),
    edges: t.edges,
    segmentGroups: t.segmentGroups,
    // Optional-chained like the flush above: a tab object built before this
    // field existed (a test double, an older persisted record) must still
    // persist rather than throw on the autosave path.
    ...(t.subgraphs?.length
      ? { subgraphs: stripSubgraphSecrets(t.subgraphs) }
      : {}),
    // Only while the run might still be in flight, so a finished run's
    // id never survives a reload: the Inspector's captured outputs live
    // in a process-lifetime store, and pointing it at a run whose
    // outputs are gone would replace "not run yet" with an empty view.
    // Keeps localStorage byte-identical for an idle tab, too.
    ...(t.status === 'running' && t.lastRunId ? { lastRunId: t.lastRunId } : {}),
    recordOutputs: t.recordOutputs,
    verboseMode: t.verboseMode,
    seed: t.seed,
    deterministic: t.deterministic,
    graphId: t.graphId,
    weightsPersistent: t.weightsPersistent,
    backwardMode: t.backwardMode,
    autoBackward: t.autoBackward,
  };
}

// Last record built per tab, with the exact inputs it was built from (#125).
//
// `buildPersistedTab` walks every node (secret stripping), so before this
// cache a five-tab workspace re-walked all five graphs every 250ms because
// ONE of them moved a node. Every field a record reads is either an array the
// store replaces immutably on change — so a reference compare is exact — or a
// scalar, and the scalars are folded into one signature string. A cache hit
// returns the SAME record object, which is also how `tabPersistence`
// recognises a tab it has already made durable and skips writing it.
interface TabRecordCacheEntry {
  nodes: Node<NodeData>[];
  edges: Edge[];
  segmentGroups: SegmentGroup[];
  subgraphs: SubgraphDefinition[];
  subgraphStack: TabState['subgraphStack'];
  scalars: string;
  record: PersistedTab;
}
let _recordCache = new Map<string, TabRecordCacheEntry>();

// JSON, not a joined string: `name`, `description` and the file/project paths
// are free text, and any separator character they could contain would let two
// different tabs produce the same signature — a save that never happens.
function scalarSignature(t: TabState): string {
  return JSON.stringify([
    t.name,
    t.description,
    t.currentGraphFile ?? '',
    t.projectOrigin ?? '',
    t.readOnly ? '1' : '0',
    t.status,
    t.lastRunId ?? '',
    t.recordOutputs ? '1' : '0',
    t.verboseMode ? '1' : '0',
    t.graphId,
    t.weightsPersistent ? '1' : '0',
    t.backwardMode ? '1' : '0',
    t.autoBackward ? '1' : '0',
    t.seed ?? '',
    t.deterministic ? '1' : '0',
  ]);
}

function persistedTabsFor(tabs: TabState[]): PersistedTab[] {
  const next = new Map<string, TabRecordCacheEntry>();
  const records = tabs.map((t) => {
    const scalars = scalarSignature(t);
    const cached = _recordCache.get(t.id);
    const entry: TabRecordCacheEntry =
      cached &&
      cached.nodes === t.nodes &&
      cached.edges === t.edges &&
      cached.segmentGroups === t.segmentGroups &&
      cached.subgraphs === t.subgraphs &&
      cached.subgraphStack === t.subgraphStack &&
      cached.scalars === scalars
        ? cached
        : {
            nodes: t.nodes,
            edges: t.edges,
            segmentGroups: t.segmentGroups,
            subgraphs: t.subgraphs,
            subgraphStack: t.subgraphStack,
            scalars,
            record: buildPersistedTab(t),
          };
    next.set(t.id, entry);
    return entry.record;
  });
  // Rebuilt rather than mutated so a closed tab's entry cannot linger.
  _recordCache = next;
  return records;
}

/** The pre-#125 write path: one JSON blob under the scoped key. */
function saveTabsToLocalStorage(records: PersistedTab[], activeTabId: string) {
  try {
    localStorage.setItem(
      _storageKey(),
      JSON.stringify({ activeTabId, tabs: records }),
    );
  } catch {
    // QuotaExceededError / SecurityError / private mode etc. The README
    // promises auto-save; failing silently lets the user lose work without
    // realising.
    warnPersistence('persistence.quotaError');
  }
}

// IndexedDB writes are asynchronous while the debounce that triggers them is
// not, so saves are chained rather than fired concurrently. Two overlapping
// writes would still both name the whole tab set (last one wins), but the
// durability bookkeeping in `tabPersistence` assumes it sees them in order.
let _idbWriteChain: Promise<void> = Promise.resolve();

// #164 follow-up: `saveTabs` retries IndexedDB on EVERY autosave, so a
// persistently broken database re-enters the catch below indefinitely --
// unlike `persistence.quotaError`, which only fires when the fallback write
// ALSO fails, and so is rare. `warnPersistence`'s own throttle is 60s, and
// error toasts never auto-dismiss, so the throttle alone would leave one new
// undismissed toast piling up every minute for the rest of the session. The
// downgrade is a one-time state transition, not a recurring event, so this
// latches it to one toast for the session regardless of how long the
// database stays broken -- deliberately never reset, even if a later save
// happens to succeed (a flaky database earns one warning, not a fresh one
// every time it flickers).
let _downgradeWarned = false;

function saveTabs(tabs: TabState[], activeTabId: string) {
  const records = persistedTabsFor(tabs);
  if (!idbAvailable()) {
    saveTabsToLocalStorage(records, activeTabId);
    return;
  }
  const scope = _storageKey();
  _idbWriteChain = _idbWriteChain
    .then(() => writeSnapshot(scope, records, activeTabId))
    .catch(() => {
      // IndexedDB was there a moment ago and is not now (a version upgrade
      // from another tab, a corrupted database, a browser that revoked
      // storage). Autosave is a promise to the user, so fall back to the
      // tier that may still work rather than dropping the save.
      saveTabsToLocalStorage(records, activeTabId);
      // The fallback above is silent by design (it has its own quotaError
      // notice for when IT fails), but the DOWNGRADE itself must not be
      // (#164): the user just quietly lost the generous IndexedDB ceiling
      // for the 5MB localStorage one, with nothing said about it until a
      // future save fails for real. A distinct key from quotaError /
      // storageUnavailable, so a read failure and a write failure never
      // suppress each other's warning. Latched (see `_downgradeWarned`)
      // rather than left to warnPersistence's 60s throttle alone, or a
      // database that stays broken would re-toast every minute forever.
      if (!_downgradeWarned) {
        _downgradeWarned = true;
        warnPersistence('persistence.downgraded');
      }
    });
}

/** Rebuild one `TabState` from its record, over a base carrying live fields. */
function tabFromPersisted(t: PersistedTab, base: TabState): TabState {
  return {
    ...base,
    name: t.name,
    description: t.description ?? '',
    currentGraphFile: t.currentGraphFile ?? null,
    projectOrigin: t.projectOrigin ?? null,
    readOnly: t.readOnly ?? false,
    nodes: t.nodes ?? [],
    edges: t.edges ?? [],
    segmentGroups: Array.isArray(t.segmentGroups) ? t.segmentGroups : [],
    // localStorage is user-editable and IndexedDB records outlive format
    // changes, so a restored record gets the same coercion an imported file
    // gets -- the alternative is a workspace that throws on every autosave
    // from the moment it is reopened.
    subgraphs: normalizeSubgraphs(t.subgraphs),
    // Never restored from disk: a reload lands at the top level.
    subgraphStack: [],
    lastRunId: typeof t.lastRunId === 'string' ? t.lastRunId : null,
    recordOutputs: t.recordOutputs ?? true,
    verboseMode: t.verboseMode ?? false,
    seed: t.seed ?? null,
    deterministic: t.deterministic ?? false,
    // Preserve persisted graphId — required so backend NodeStateStore
    // keeps weights linked to this tab across sessions. Falls back to
    // the freshly generated UUID for legacy tabs.
    graphId: t.graphId ?? base.graphId,
    weightsPersistent: t.weightsPersistent ?? true,
    backwardMode: t.backwardMode ?? false,
    autoBackward: t.autoBackward ?? false,
  };
}

function loadTabs(): { tabs: TabState[]; activeTabId: string } {
  try {
    const raw = localStorage.getItem(_storageKey());
    if (raw) {
      const data = JSON.parse(raw);
      if (Array.isArray(data.tabs) && data.tabs.length > 0) {
        const tabs: TabState[] = data.tabs.map((t: PersistedTab) =>
          tabFromPersisted(t, createTabState(t.id, t.name)),
        );
        const activeTabId = tabs.some((t) => t.id === data.activeTabId)
          ? data.activeTabId
          : tabs[0].id;
        return { tabs, activeTabId };
      }
    }
  } catch {
    // Corrupted data — fall through to default
  }
  const id = generateId();
  return { tabs: [createTabState(id, 'Tab 1')], activeTabId: id };
}

const initialState = loadTabs();

export const useTabStore = create<TabStoreState>((set, get) => ({
  tabs: initialState.tabs,
  activeTabId: initialState.activeTabId,

  // ── Tab management ──

  addTab: (name) => {
    const id = generateId();
    const tabCount = get().tabs.length;
    set({
      tabs: [...get().tabs, createTabState(id, name ?? `Tab ${tabCount + 1}`)],
      activeTabId: id,
    });
  },

  removeTab: (id) => {
    const { tabs, activeTabId } = get();
    if (tabs.length <= 1) return;

    const tab = tabs.find((t) => t.id === id);
    if (tab) tab.ws.disconnect();
    // The shared canvas keeps each tab's pan/zoom keyed by id (#125); a
    // closed tab's entry would otherwise outlive it for the whole session.
    forgetViewport(id);

    const remaining = tabs.filter((t) => t.id !== id);
    const newActive = activeTabId === id
      ? remaining[Math.min(tabs.findIndex((t) => t.id === id), remaining.length - 1)].id
      : activeTabId;
    set({ tabs: remaining, activeTabId: newActive });
  },

  setActiveTab: (id) => set({ activeTabId: id }),

  renameTab: (id, name) =>
    set({ tabs: updateTab(get().tabs, id, () => ({ name })) }),

  setDescription: (description) =>
    set({ tabs: updateTab(get().tabs, get().activeTabId, () => ({ description })) }),

  setCurrentGraphFile: (file) =>
    set({ tabs: updateTab(get().tabs, get().activeTabId, () => ({ currentGraphFile: file })) }),

  setTabReadOnly: (v) =>
    set({ tabs: updateTab(get().tabs, get().activeTabId, () => ({ readOnly: v })) }),

  rehydrateForProject: (projectId) => {
    // Non-project mode keeps the import-time base-key tabs untouched — its
    // scope was already hydrated at import, and re-reading the base key here
    // would throw away edits made since.
    if (projectId !== null) {
      const loaded = loadTabs(); // reads the now-scoped key
      set({ tabs: loaded.tabs, activeTabId: loaded.activeTabId });
      // The synchronous read above only sees localStorage. IndexedDB holds
      // this project's real tabs (and receives them if this is the first load
      // after the upgrade); it answers a round-trip later and wins.
      _startHydration();
    }
  },
  stampActiveTabProject: (projectId) =>
    set({ tabs: updateTab(get().tabs, get().activeTabId, () => ({ projectOrigin: projectId })) }),

  // ── Helpers ──

  getActiveTab: () => {
    const { tabs, activeTabId } = get();
    return tabs.find((t) => t.id === activeTabId)!;
  },

  getTab: (id) => get().tabs.find((t) => t.id === id),

  // ── Flow actions (active tab) ──

  setNodes: (nodes) =>
    set({ tabs: updateTab(get().tabs, get().activeTabId, () => ({ nodes })) }),

  setEdges: (edges) =>
    set({ tabs: updateTab(get().tabs, get().activeTabId, () => ({ edges })) }),

  onNodesChange: (changes) => {
    // Snapshot at drag start for undo (not every pixel)
    const hasDragStart = changes.some(
      (c) => c.type === 'position' && (c as any).dragging === true
    );
    if (hasDragStart) {
      // Check if we already snapshotted for this drag session
      const tab = get().getActiveTab();
      const wasDragging = tab.nodes.some((n) => n.dragging);
      if (!wasDragging) {
        get().pushUndoSnapshot();
      }
    }
    // Snapshot on node removal via Delete key
    const hasRemove = changes.some((c) => c.type === 'remove');
    if (hasRemove) {
      get().pushUndoSnapshot();
    }
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => {
        let updatedNodes = applyNodeChanges(changes, tab.nodes) as Node<NodeData>[];

        // Collect IDs of nodes that had position changes (not notes)
        // Narrow via type predicate so `id` is safely accessible — the
        // `NodeChange` union includes `NodeAddChange` which lacks `id`.
        const posChanges = changes.filter(
          (c): c is Extract<NodeChange, { type: 'position' }> =>
            c.type === 'position' && (c as { position?: unknown }).position != null,
        );
        // Bound notes ride along with their parent, which costs up to two more
        // passes over the WHOLE array on every pointer move of a drag — and
        // `map` allocates a fresh array even when it returns every element
        // unchanged. Most graphs (certainly the large ones that make a drag
        // frame expensive) have no bound note at all, so check once and skip
        // both passes: `some` short-circuits and allocates nothing (#125).
        const hasBoundNotes =
          posChanges.length > 0 &&
          updatedNodes.some(
            (n) => n.type === 'noteNode' && n.data.boundToNodeId && n.data.boundOffset,
          );
        if (hasBoundNotes) {
          const movedIds = new Set(posChanges.map((c) => c.id));

          // 1) If a bound note was dragged, update its offset relative to parent
          updatedNodes = updatedNodes.map((n) => {
            if (n.type !== 'noteNode' || !n.data.boundToNodeId || !n.data.boundOffset) return n;
            if (!movedIds.has(n.id)) return n;
            // Note itself was moved — recalculate offset
            const parent = updatedNodes.find((p) => p.id === n.data.boundToNodeId);
            if (!parent) return n;
            return {
              ...n,
              data: {
                ...n.data,
                boundOffset: {
                  x: n.position.x - parent.position.x,
                  y: n.position.y - parent.position.y,
                },
              },
            };
          });

          // 2) If a computational node moved, reposition all its bound notes
          const movedComputational = new Set(
            [...movedIds].filter((id) => {
              const node = updatedNodes.find((n) => n.id === id);
              return node && node.type !== 'noteNode';
            })
          );
          if (movedComputational.size > 0) {
            updatedNodes = updatedNodes.map((n) => {
              if (n.type !== 'noteNode' || !n.data.boundToNodeId || !n.data.boundOffset) return n;
              if (!movedComputational.has(n.data.boundToNodeId)) return n;
              // Skip if the note itself was also moved (user is dragging the note)
              if (movedIds.has(n.id)) return n;
              const parent = updatedNodes.find((p) => p.id === n.data.boundToNodeId);
              // boundToNodeId was just confirmed present in updatedNodes above
              /* v8 ignore start */
              if (!parent) return n;
              /* v8 ignore stop */
              return {
                ...n,
                position: {
                  x: parent.position.x + n.data.boundOffset.x,
                  y: parent.position.y + n.data.boundOffset.y,
                },
              };
            });
          }
        }

        // When a node is removed, unbind notes that were bound to it — and
        // close a detail modal that was showing it.
        //
        // The modal has to be closed HERE, not only in `deleteNode`: React
        // Flow's own Delete key never calls that action, it emits a `remove`
        // change straight into this reducer. Left stale, `nodeDetailNodeId`
        // points at a node that no longer exists — harmless while it renders
        // nothing, but an undo that restores the node would pop the modal
        // back open on its own.
        //
        // `selectedNodeId` gets the same treatment (#167): it is the other
        // half of the same desync the Delete key exposed, just read instead
        // of written. Left stale, it would name a node that no longer
        // exists to every reader of the field (bypass/copy fallbacks,
        // future callers) even though React Flow itself has no opinion left
        // — nothing is `.selected` once its node is gone.
        let nodeDetailNodeId = tab.nodeDetailNodeId;
        let selectedNodeId = tab.selectedNodeId;
        if (hasRemove) {
          const removedIds = new Set(
            changes.filter((c) => c.type === 'remove').map((c) => c.id)
          );
          updatedNodes = updatedNodes.map((n) => {
            if (n.type !== 'noteNode' || !n.data.boundToNodeId) return n;
            if (!removedIds.has(n.data.boundToNodeId)) return n;
            return { ...n, data: { ...n.data, boundToNodeId: null, boundOffset: null } };
          });
          if (nodeDetailNodeId !== null && removedIds.has(nodeDetailNodeId)) {
            nodeDetailNodeId = null;
          }
          if (selectedNodeId !== null && removedIds.has(selectedNodeId)) {
            selectedNodeId = null;
          }
        }

        return { nodes: updatedNodes, nodeDetailNodeId, selectedNodeId };
      }),
    });
  },

  onEdgesChange: (changes) => {
    const hasRemove = changes.some((c) => c.type === 'remove');
    if (hasRemove) {
      get().pushUndoSnapshot();
    }
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        edges: applyEdgeChanges(changes, tab.edges),
      })),
    });
  },

  onConnect: (connection) => {
    get().pushUndoSnapshot();
    const edge: Edge = {
      id: generateId(),
      source: connection.source,
      target: connection.target,
      sourceHandle: connection.sourceHandle ?? undefined,
      targetHandle: connection.targetHandle ?? undefined,
      animated: false,
      style: { stroke: '#555', strokeWidth: 2 },
    };
    if (connection.target) get().markDirty(connection.target);
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        edges: [...tab.edges, edge],
      })),
    });
  },

  addNode: (definition, position) => {
    get().pushUndoSnapshot();
    const node = buildFlowNode(definition, position);
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        nodes: [...tab.nodes, node],
      })),
    });
  },

  addPresetNode: (preset, position) => {
    get().pushUndoSnapshot();
    const internalParams: Record<string, Record<string, any>> = {};
    for (const n of preset.nodes) {
      internalParams[n.id] = { ...n.params };
    }
    const definition: NodeDefinition = {
      node_name: preset.preset_name,
      category: preset.category,
      description: preset.description,
      inputs: preset.exposed_inputs.map((p) => ({
        name: p.name,
        data_type: p.data_type,
        description: p.description,
        optional: false,
      })),
      outputs: preset.exposed_outputs.map((p) => ({
        name: p.name,
        data_type: p.data_type,
        description: p.description,
        optional: false,
      })),
      params: [],
    };
    const node: Node<NodeData> = {
      id: generateId(),
      type: 'presetNode',
      position,
      data: {
        label: preset.preset_name,
        type: `preset:${preset.preset_name}`,
        params: {},
        definition,
        isPreset: true,
        presetDefinition: preset,
        internalParams,
        executionStatus: 'idle',
      },
    };
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        nodes: [...tab.nodes, node],
      })),
    });
  },

  updateNodeParams: (nodeId, params) => {
    get().markDirty(nodeId);

    // Deleting edges is not something a param edit is expected to do, so it
    // gets its own undo entry BEFORE the write: dropping a script from 8
    // ports to 1 destroys up to 7 edges, and without this Ctrl+Z would skip
    // straight past their deletion to whatever was undoable before it.
    // Computed here, ahead of the write, because `pushUndoSnapshot` captures
    // the CURRENT tab and must see the edges intact.
    {
      const tab = get().getActiveTab();
      const node = tab.nodes.find((n) => n.id === nodeId);
      const merged = { ...(node?.data.params ?? {}), ...params };
      if (staleEdges(node, merged, tab.edges).size > 0) {
        get().pushUndoSnapshot();
      }
    }

    const orphaned = new Set<string>();
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => {
        const node = tab.nodes.find((n) => n.id === nodeId);
        const nextParams = { ...(node?.data.params ?? {}), ...params };
        const nodes = tab.nodes.map((n) =>
          n.id === nodeId
            ? { ...n, data: { ...n.data, params: nextParams } }
            : n
        );
        // A param can change the node's own PORT SET (Split's `chunks`,
        // PythonScript's `input_ports`/`output_ports`, ComposeTransform's
        // `steps`). An edge left hanging off a handle that no longer renders
        // is invisible on the canvas but very much alive in the graph JSON,
        // and the backend validator rejects the whole run for it. Drop those
        // edges here — the one choke point every param edit from every
        // surface goes through.
        const stale = staleEdges(node, nextParams, tab.edges);
        if (stale.size === 0) return { nodes };
        for (const edge of tab.edges) {
          // A consumer that just lost its input is dirty in its own right:
          // once the edge is gone the dirty walk cannot reach it from here.
          if (stale.has(edge.id) && edge.target !== nodeId) orphaned.add(edge.target);
        }
        return { nodes, edges: tab.edges.filter((e) => !stale.has(e.id)) };
      }),
    });
    for (const target of orphaned) get().markDirty(target);
  },

  updatePresetInternalParam: (nodeId, internalNodeId, paramName, value) =>
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        nodes: tab.nodes.map((n) => {
          if (n.id !== nodeId) return n;
          const prev = n.data.internalParams ?? {};
          return {
            ...n,
            data: {
              ...n.data,
              internalParams: {
                ...prev,
                [internalNodeId]: {
                  ...prev[internalNodeId],
                  [paramName]: value,
                },
              },
            },
          };
        }),
      })),
    }),

  // The plain-click path (FlowCanvas#handleNodeClick) ONLY. Deliberately
  // does NOT sync `.selected` -- contrast `selectNodeExclusively` below.
  //
  // React Flow already applies every click's FULL selection effect to
  // `.selected`, via its own `onNodesChange` dispatch, before `onNodeClick`
  // (and therefore this action) ever runs -- verified against the installed
  // `@xyflow/react` source for all three cases: a plain click (replaces the
  // selection), a shift+click that ADDS to one, and a shift+click that
  // REMOVES a member from one.
  //
  // Re-deriving `.selected` from just `id` here would fight that -- and for
  // a shift+click REMOVAL specifically, it would get it backwards rather
  // than merely redundant: the node the user just removed is, by the time
  // this runs, the one node in the array that is NOT `.selected`, which is
  // indistinguishable from "a stale click landed on a node outside the
  // current selection" (the exact shape `selectNodeExclusively` exists to
  // correct for a right-click). `selectOnlyNode` cannot tell those two
  // apart from the nodes array alone, so it would re-select the node the
  // user just removed and deselect the rest -- the opposite of the click
  // (#167 follow-up). Staying raw for every click side-steps the ambiguity
  // entirely: a click's own selection effect is never touched twice.
  setSelectedNodeId: (id) =>
    set({ tabs: updateTab(get().tabs, get().activeTabId, () => ({ selectedNodeId: id })) }),

  // Right-click, the ResultsPanel's "click to highlight", the pane-click
  // clear, and anything else that is NOT a plain click and so gets no help
  // from React Flow's own selection handling (#167). Goes through
  // `selectOnlyNode` so `.selected` agrees with `selectedNodeId` by
  // construction -- see that helper's comment for the multi-selection
  // exception, and `setSelectedNodeId` above for why the click path is a
  // separate action rather than sharing this one.
  selectNodeExclusively: (id) =>
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        selectedNodeId: id,
        nodes: selectOnlyNode(tab.nodes, id),
      })),
    }),

  openPresetModal: (id) =>
    set({ tabs: updateTab(get().tabs, get().activeTabId, () => ({ presetModalNodeId: id })) }),

  closePresetModal: () =>
    set({ tabs: updateTab(get().tabs, get().activeTabId, () => ({ presetModalNodeId: null })) }),

  openLayersModal: (id) =>
    set({ tabs: updateTab(get().tabs, get().activeTabId, () => ({ layersModalNodeId: id })) }),

  closeLayersModal: () =>
    set({ tabs: updateTab(get().tabs, get().activeTabId, () => ({ layersModalNodeId: null })) }),

  // Opening the detail modal also selects the node, in ONE commit. Every entry
  // point (double-click, context menu, Enter, the modal's own prev/next
  // arrows) therefore leaves the canvas, the config panel and the Inspector
  // pointing at the same node the modal is showing — which is what makes
  // stepping through a graph with the arrow keys read as a walkthrough.
  // `target` deep-links a tab and a port (#129: the edge tooltip's "View
  // stats"). Both are written unconditionally — an omitted target CLEARS
  // them, so opening a node normally after following a deep link lands on the
  // default tab rather than wherever the link went.
  openNodeDetail: (id, target) =>
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        nodeDetailNodeId: id,
        selectedNodeId: id,
        nodes: selectOnlyNode(tab.nodes, id),
        nodeDetailTab: target?.tab ?? null,
        nodeDetailPort: target?.port ?? null,
        nodeDetailRequest: tab.nodeDetailRequest + 1,
      })),
    }),

  closeNodeDetail: () =>
    set({
      tabs: updateTab(get().tabs, get().activeTabId, () => ({
        nodeDetailNodeId: null,
        nodeDetailTab: null,
        nodeDetailPort: null,
      })),
    }),

  updateNodeLayers: (nodeId, layersJson) =>
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        nodes: tab.nodes.map((n) =>
          n.id === nodeId
            ? { ...n, data: { ...n.data, params: { ...n.data.params, layers: layersJson } } }
            : n
        ),
      })),
    }),

  setNodeExecutionStatus: (nodeId, status, error) =>
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        nodes: tab.nodes.map((n) =>
          n.id === nodeId
            ? { ...n, data: { ...n.data, executionStatus: status, error } }
            : n
        ),
      })),
    }),

  clearExecutionStatus: () =>
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        nodes: tab.nodes.map((n) => ({
          ...n,
          data: { ...n.data, executionStatus: 'idle' as const, error: undefined },
        })),
      })),
    }),

  clear: () => {
    // Flush the sub-canvas editing stack BEFORE snapshotting, exactly as
    // `getSerializedGraph` and `buildPersistedTab` do. Without it the
    // snapshot captured whatever block the user happened to be standing
    // inside, and the outer graph -- which lived nowhere else -- was gone
    // for good: one undo after File > Clear replaced the user's graph with
    // the block's insides. `File > Clear` is in a menu with no
    // `subgraphStack` gating, so "the user is inside a block" is not a
    // state this action can assume away.
    const active = get().getActiveTab();
    const flushed = flushSubgraphEditing(active);
    // Identity: `flushSubgraphEditing` returns its argument untouched when
    // there is no stack, so this is a no-op from the top level.
    if (flushed !== active) {
      set({
        tabs: updateTab(get().tabs, get().activeTabId, () => ({
          nodes: flushed.nodes,
          edges: flushed.edges,
          subgraphs: flushed.subgraphs,
          subgraphStack: [],
        })),
      });
    }
    get().pushUndoSnapshot();
    set({
      tabs: updateTab(get().tabs, get().activeTabId, () => ({
        nodes: [],
        edges: [],
        // Definitions belong to the graph that was cleared, and the editing
        // stack points into nodes that no longer exist.
        subgraphs: [],
        subgraphStack: [],
        selectedNodeId: null,
        presetModalNodeId: null,
        layersModalNodeId: null,
        nodeDetailNodeId: null,
        // A cleared canvas is a fresh, unbound graph. Drop the metadata tied
        // to the previously-open graph so the next save doesn't silently
        // overwrite that file with stale description / segment overlays.
        description: '',
        currentGraphFile: null,
        segmentGroups: [],
        activeSegment: null,
        // An empty graph is trivially current-format -- never leave a tab
        // that became read-only from a newer-format load stuck refusing
        // Save forever after Clear (ID8 fast-follow, task 16 review
        // Adjudication B / Important finding 1).
        readOnly: false,
      })),
    });
  },

  getSerializedGraph: () => {
    // A tab whose canvas is showing a subgraph's insides still SAVES and
    // RUNS the whole graph: flatten the editing stack first, so what is
    // serialized never depends on where the user happens to be standing.
    const tab = flushSubgraphEditing(get().getActiveTab());
    // Computed AFTER the flush, so a block the user is editing right now --
    // whose instance is back on the canvas only because the flush put it
    // there -- is never mistaken for an orphan and dropped mid-edit.
    //
    // Optional-chained like the flush and the persist path: tests and older
    // persisted records build tab objects with no `subgraphs` field at all,
    // and serialization -- which every Run goes through -- is the last place
    // that should throw over a missing optional.
    const liveSubgraphIds = reachableSubgraphIds(tab.nodes, tab.subgraphs ?? []);
    const subgraphs = stripSubgraphSecrets(
      (tab.subgraphs ?? []).filter((d) => liveSubgraphIds.has(d.id)),
    );
    const presets: import('../types').PresetDefinition[] = [];
    const seenPresets = new Set<string>();

    const nodes = tab.nodes.map((n) => {
      // Note nodes: serialize with note-specific fields
      if (n.type === 'noteNode') {
        return {
          id: n.id,
          type: 'note',
          position: roundPosition(n.position),
          data: {
            noteKind: n.data.noteKind,
            noteContent: n.data.noteContent,
            noteColor: n.data.noteColor,
            boundToNodeId: n.data.boundToNodeId,
            boundOffset: roundOffset(n.data.boundOffset),
            noteWidth: roundDimension(n.data.noteWidth),
            noteHeight: roundDimension(n.data.noteHeight),
          },
        };
      }

      if (n.data.isPreset && n.data.presetDefinition) {
        const name = n.data.presetDefinition.preset_name;
        if (!seenPresets.has(name)) {
          seenPresets.add(name);
          presets.push(n.data.presetDefinition);
        }
      }
      return {
        id: n.id,
        type: n.data.type,
        position: roundPosition(n.position),
        data: {
          params: stripSecretParams(n.data.params, n.data.definition),
          ...(n.data.isPreset
            ? { internalParams: stripSecretInternalParams(n.data.internalParams, n.data.presetDefinition) }
            : {}),
          // Written only when muted (core#128), so a graph nobody has
          // bypassed anything in serializes byte-identically to before.
          ...(n.data.bypassed ? { bypassed: true } : {}),
        },
      };
    });

    // A preset node collapsed INTO a block is still a preset node: the
    // definition keeps it as `preset:<name>` + `internalParams`, and without
    // its portable definition travelling in `presets[]` the export 400s with
    // `Unknown preset` and the save writes a file nothing can run. The
    // canvas walk above cannot see it -- the node is not on the canvas any
    // more -- and a definition entry carries only the type STRING, so the
    // definition is resolved through the registry the way
    // `resolveSerializedNodes` resolves it when the file is reopened.
    const knownPresets = new Map(
      useNodeDefStore.getState().presets.map((p) => [p.preset_name, p]),
    );
    for (const definition of subgraphs) {
      for (const raw of definition.nodes as { type?: unknown }[]) {
        const type = typeof raw?.type === 'string' ? raw.type : '';
        if (!type.startsWith('preset:')) continue;
        const name = type.slice('preset:'.length);
        if (seenPresets.has(name)) continue;
        const portable = knownPresets.get(name);
        // An unknown name is left out rather than invented: it is already
        // broken, and the backend names it (`Unknown preset: <name>`).
        if (!portable) continue;
        seenPresets.add(name);
        presets.push(portable);
      }
    }

    return {
      nodes,
      edges: tab.edges.map((e) => {
        const isTrigger = e.type === 'triggerEdge' || (e.data as any)?.type === 'trigger';
        return {
          id: e.id,
          source: e.source,
          target: e.target,
          sourceHandle: e.sourceHandle ?? '',
          targetHandle: e.targetHandle ?? '',
          ...(isTrigger ? { type: 'trigger' } : {}),
        };
      }),
      presets,
      segmentGroups: tab.segmentGroups,
      // Only the definitions something can still reach (core#137 review).
      //
      // Deleting the last instance of a block does NOT drop its definition,
      // and deliberately so -- the definition has to stay in memory or
      // undoing the delete would bring back an instance whose block is gone.
      // But there are several ways to delete a node (deleteNode, the Delete
      // key through onNodesChange, clear) and pruning at each of them is how
      // you end up with the invariant true on some paths and false on
      // others. Pruning HERE, at the one boundary every save and every run
      // passes through, covers all of them at once.
      //
      // Transitive, because a definition can hold an instance of another:
      // a block reachable only from INSIDE a reachable block is still live.
      //
      // Always an ARRAY, never undefined. `createTabState` and
      // `tabFromPersisted` -- the only two places a tab object is built --
      // give every real tab a `subgraphs` list, so the coalesce only covers
      // hand-built tab doubles in tests. A public serializer that answers
      // `undefined` because some test object was malformed pushes an
      // `| undefined` into the types of every caller, which is exactly how
      // `Toolbar.handleExportSubgraph` ended up dereferencing a
      // possibly-undefined list.
      //
      // SECRET params inside a definition are blanked by the same rule the
      // top-level nodes above follow: a key does not stop being a key
      // because the node holding it was collapsed into a block.
      subgraphs,
    };
  },

  // ── Subgraphs (core#137) ──
  //
  // Every action here computes the whole next state and commits it with ONE
  // `set`, after ONE `pushUndoSnapshot()`. That is the only way this store
  // makes a multi-part change a single undo step, and it is what the
  // acceptance criterion "undo restores across collapse/expand" needs.
  //
  // readOnly policy for the six actions below -- deliberate, not accidental:
  //
  //   collapseSelectionToSubgraph, expandSubgraphInstance, renameSubgraph
  //     User-initiated MUTATIONS of the graph. Guarded. Two of the three
  //     were guarded and one was not, which is worse than guarding none:
  //     inconsistent guards advertise a protection that does not exist.
  //
  //   enterSubgraph, exitSubgraph
  //     NAVIGATION. Never guarded. Blocking entry would stop a user READING
  //     a block on a graph they are allowed to open, and blocking exit would
  //     trap them inside one with no way back to the top level.
  //
  //   setSubgraphs
  //     DESERIALIZATION, called by the load path. Never guarded -- guarding
  //     it would make a read-only graph fail to open at all, since readOnly
  //     is set by the very load that then has to install the definitions.
  //
  // Note this is a UI-affordance gate, matching how `readOnly` works
  // everywhere else in this store (a save-time gate; `deleteNode` and
  // `updateNodeParams` are not guarded either). It is not a security
  // boundary, and nothing here should be mistaken for one.

  // CONTRACT, because this is a trap otherwise: call this only AFTER
  // `setNodes`/`setEdges` have installed the graph these definitions belong
  // to. It drops `subgraphStack` without putting a canvas back, so calling
  // it while the user is inside a block leaves the block's INSIDES on screen
  // as if they were the whole graph. All three callers (`Toolbar`'s load and
  // import, `openExample`) set nodes and edges first, so no path reaches
  // that today.
  //
  // Deliberately does NOT flush the stack itself, which would look like the
  // safer choice and is not: `flushSubgraphEditing` writes whatever is on
  // the canvas back into the definition being edited, and by the time this
  // runs the canvas is already the NEW graph -- it would overwrite the
  // incoming definition with the incoming top level.
  setSubgraphs: (subgraphs) =>
    set({
      tabs: updateTab(get().tabs, get().activeTabId, () => ({
        // Every caller is a document reader handing over a list parsed out
        // of a file, and none of them validates the entries -- see
        // `normalizeSubgraphs`. This is the one door they all come through,
        // so the shape is settled here rather than in the four places that
        // later walk it.
        subgraphs: normalizeSubgraphs(subgraphs),
        // A load replaces the graph, so any open sub-canvas was editing a
        // definition that is gone.
        subgraphStack: [],
      })),
    }),

  collapseSelectionToSubgraph: (name) => {
    const tab = get().getActiveTab();
    if (tab.readOnly) {
      return { ok: false, reason: 'read-only', blockers: [] } as CollapseResult;
    }
    const selectedIds = tab.nodes.filter((n) => n.selected).map((n) => n.id);
    const result = collapseSelection(
      tab.nodes,
      tab.edges,
      tab.subgraphs,
      selectedIds,
      { name },
    );
    if (!result.ok) return result;
    // The ids that just moved off the canvas and into the definition. From
    // the rest of the tab's point of view they are GONE, so everything
    // `deleteNode` cleans up when a node disappears has to be cleaned up
    // here too -- a Teaching Inspector segment can never resolve a path once
    // an endpoint lives inside a block, and it is persisted through save, so
    // leaving it behind writes a permanently broken segment to disk.
    const swallowed = new Set<string>(
      result.definition.nodes.map((n: { id: unknown }) => String(n.id)),
    );
    const nodes = result.nodes.map((n) =>
      n.type === 'noteNode' && n.data.boundToNodeId && swallowed.has(n.data.boundToNodeId)
        ? { ...n, data: { ...n.data, boundToNodeId: null, boundOffset: null } }
        : n,
    );
    get().pushUndoSnapshot();
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (t) => ({
        nodes,
        edges: result.edges,
        subgraphs: result.subgraphs,
        selectedNodeId: result.instanceId,
        segmentGroups: t.segmentGroups.filter(
          (s) => !swallowed.has(s.headNodeId) && !swallowed.has(s.tailNodeId),
        ),
        activeSegment:
          t.activeSegment &&
          (swallowed.has(t.activeSegment.headNodeId) ||
            swallowed.has(t.activeSegment.tailNodeId))
            ? null
            : t.activeSegment,
        // The last thing on `deleteNode`'s cleanup list: a detail modal
        // showing a node that just moved inside a block has nothing left to
        // render on this canvas.
        nodeDetailNodeId:
          t.nodeDetailNodeId && swallowed.has(t.nodeDetailNodeId)
            ? null
            : t.nodeDetailNodeId,
        // A collapsed block is one node again, so everything it contained
        // stops being a candidate for partial re-execution under its old id.
        dirtyNodeIds: new Set([result.instanceId]),
      })),
    });
    return result;
  },

  expandSubgraphInstance: (nodeId) => {
    const tab = get().getActiveTab();
    if (tab.readOnly) return false;
    const defs = useNodeDefStore.getState().definitions;
    const presets = useNodeDefStore.getState().presets;
    const result = expandInstance(
      tab.nodes,
      tab.edges,
      tab.subgraphs,
      nodeId,
      (raw) => resolveSerializedNodes(raw, defs, presets, tab.subgraphs),
    );
    if (!result.ok) return false;
    // The INSTANCE node is the one that disappears here -- the exact mirror
    // of collapse, where the members disappear. Everything `deleteNode`
    // cleans up when a node vanishes has to be cleaned up on both sides of
    // that mirror, or expanding a block leaves a Teaching Inspector segment
    // that can never resolve a path (and `segmentGroups` is persisted, so
    // the broken segment is written to disk) plus a note bound to an id
    // nothing wears any more.
    const nodes = result.nodes.map((n) =>
      n.type === 'noteNode' && n.data.boundToNodeId === nodeId
        ? { ...n, data: { ...n.data, boundToNodeId: null, boundOffset: null } }
        : n,
    );
    get().pushUndoSnapshot();
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (t) => ({
        nodes,
        edges: result.edges,
        subgraphs: result.subgraphs,
        selectedNodeId: null,
        segmentGroups: t.segmentGroups.filter(
          (s) => s.headNodeId !== nodeId && s.tailNodeId !== nodeId,
        ),
        activeSegment:
          t.activeSegment &&
          (t.activeSegment.headNodeId === nodeId ||
            t.activeSegment.tailNodeId === nodeId)
            ? null
            : t.activeSegment,
        nodeDetailNodeId:
          t.nodeDetailNodeId === nodeId ? null : t.nodeDetailNodeId,
        dirtyNodeIds: new Set(result.restoredIds),
      })),
    });
    return true;
  },

  enterSubgraph: (nodeId) => {
    const tab = get().getActiveTab();
    const instance = tab.nodes.find((n) => n.id === nodeId);
    const subgraphId = subgraphIdOf(instance?.data?.type);
    const definition = tab.subgraphs.find((d) => d.id === subgraphId);
    if (!instance || !definition) return false;
    const defs = useNodeDefStore.getState().definitions;
    const presets = useNodeDefStore.getState().presets;
    const resolved = resolveSerializedNodes(
      definition.nodes,
      defs,
      presets,
      tab.subgraphs,
    );
    const innerEdges = resolveSerializedEdges(definition.edges, resolved);
    // Lay the block out on FIRST entry when it has no positions to open at.
    //
    // A definition built by collapse always carries positions, but one that
    // came from an externally-authored project, a hand-written file, or a
    // backend merge that never opened the sub-canvas may not -- and
    // `resolveSerializedNodes` defaults a missing position to {0,0}, so
    // without this every node opens stacked on the origin, unreadable and
    // unpickable. The backend deliberately does not flag `layout_missing`
    // for a sub-canvas on the grounds that the frontend lays one out on
    // entry; this is the code that makes that true.
    //
    // The check is against the RAW definition entries, not the resolved
    // nodes: by then the missing positions have already been defaulted to
    // {0,0}, which is indistinguishable from a block genuinely authored at
    // the origin. A single positionless node triggers a full relayout rather
    // than a partial one -- half a layout is not a layout.
    const needsLayout =
      definition.nodes.length > 0 &&
      definition.nodes.some((raw: { position?: { x?: unknown; y?: unknown } }) =>
        !(
          typeof raw?.position?.x === 'number' &&
          Number.isFinite(raw.position.x) &&
          typeof raw?.position?.y === 'number' &&
          Number.isFinite(raw.position.y)
        ),
      );
    const inner = needsLayout
      ? (autoLayout(resolved, innerEdges, 'all') as Node<NodeData>[])
      : resolved;
    const frame: SubgraphFrame = {
      subgraphId: definition.id,
      nodes: tab.nodes,
      edges: tab.edges,
      undoStack: tab.undoStack,
      redoStack: tab.redoStack,
      selectedNodeId: tab.selectedNodeId,
      subgraphs: tab.subgraphs,
    };
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (t) => ({
        subgraphStack: [...t.subgraphStack, frame],
        nodes: inner,
        edges: innerEdges,
        // A fresh history per level: an undo inside a block must never reach
        // past its own boundary and start rewriting the graph that contains
        // it. The outer stacks are restored on the way back out.
        undoStack: [],
        redoStack: [],
        selectedNodeId: null,
      })),
    });
    return true;
  },

  exitSubgraph: () => {
    const tab = get().getActiveTab();
    if (tab.subgraphStack.length === 0) return;
    const frame = tab.subgraphStack[tab.subgraphStack.length - 1];
    const definition = tab.subgraphs.find((d) => d.id === frame.subgraphId);
    let nodes = frame.nodes;
    let edges = frame.edges;
    let subgraphs = tab.subgraphs;
    if (definition) {
      const updated = definitionFromCanvas(definition, tab.nodes, tab.edges);
      subgraphs = subgraphs.map((d) => (d.id === updated.id ? updated : d));
      // Instances render their ports FROM the interface, so refreshing them
      // here is what makes an edit to one definition show up on every
      // instance of it -- the reuse the whole feature exists for.
      nodes = refreshInstances(frame.nodes, updated);
      edges = pruneStaleBoundaryEdges(nodes, frame.edges, subgraphs);
    }
    const history = closeFrameHistory(frame, subgraphs);
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (t) => ({
        subgraphStack: t.subgraphStack.slice(0, -1),
        nodes,
        edges,
        subgraphs,
        ...history,
        selectedNodeId: frame.selectedNodeId,
        dirtyNodeIds: new Set(
          nodes
            .filter((n) => subgraphIdOf(n.data?.type) === frame.subgraphId)
            .map((n) => n.id),
        ),
      })),
    });
  },

  exitAllSubgraphs: () => {
    const tab = get().getActiveTab();
    if (tab.subgraphStack.length === 0) return;
    const flushed = flushSubgraphEditing(tab);
    // ONE snapshot for the whole exit, not one per level: the user pressed a
    // single button, so a single Ctrl+Z is what they expect to reverse it.
    const history = closeFrameHistory(tab.subgraphStack[0], flushed.subgraphs);
    set({
      tabs: updateTab(get().tabs, get().activeTabId, () => ({
        nodes: flushed.nodes,
        edges: flushed.edges,
        subgraphs: flushed.subgraphs,
        subgraphStack: [],
        ...history,
        selectedNodeId: null,
      })),
    });
  },

  renameSubgraph: (subgraphId, name) => {
    const trimmed = name.trim();
    if (!trimmed) return;
    const tab = get().getActiveTab();
    if (tab.readOnly) return;
    if (!tab.subgraphs.some((d) => d.id === subgraphId)) return;
    get().pushUndoSnapshot();
    const subgraphs = tab.subgraphs.map((d) =>
      d.id === subgraphId ? { ...d, name: trimmed } : d,
    );
    const renamed = subgraphs.find((d) => d.id === subgraphId)!;
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (t) => ({
        subgraphs,
        nodes: refreshInstances(t.nodes, renamed),
        subgraphStack: t.subgraphStack.map((frame) =>
          frame.subgraphId === subgraphId
            ? { ...frame, nodes: refreshInstances(frame.nodes, renamed) }
            : frame,
        ),
      })),
    });
  },

  deleteNode: (nodeId) => {
    get().pushUndoSnapshot();
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        nodes: tab.nodes
          .filter((n) => n.id !== nodeId)
          // Unbind notes that were bound to the deleted node
          .map((n) =>
            n.type === 'noteNode' && n.data.boundToNodeId === nodeId
              ? { ...n, data: { ...n.data, boundToNodeId: null, boundOffset: null } }
              : n
          ),
        edges: tab.edges.filter((e) => e.source !== nodeId && e.target !== nodeId),
        selectedNodeId: tab.selectedNodeId === nodeId ? null : tab.selectedNodeId,
        // A detail modal showing the node that just vanished has nothing left
        // to render, so close it rather than leave an empty shell on screen.
        nodeDetailNodeId:
          tab.nodeDetailNodeId === nodeId ? null : tab.nodeDetailNodeId,
        // Drop any Teaching Inspector segment whose head/tail was this node —
        // it can never resolve a path once an endpoint is gone.
        segmentGroups: tab.segmentGroups.filter(
          (s) => s.headNodeId !== nodeId && s.tailNodeId !== nodeId,
        ),
        activeSegment:
          tab.activeSegment &&
          (tab.activeSegment.headNodeId === nodeId ||
            tab.activeSegment.tailNodeId === nodeId)
            ? null
            : tab.activeSegment,
      })),
    });
  },

  duplicateNode: (nodeId) => {
    get().pushUndoSnapshot();
    const tab = get().getActiveTab();
    const original = tab.nodes.find((n) => n.id === nodeId);
    if (!original) return;
    const newNode: Node<NodeData> = {
      ...original,
      id: generateId(),
      position: { x: original.position.x + 40, y: original.position.y + 40 },
      selected: false,
      data: { ...original.data, executionStatus: 'idle', error: undefined },
    };
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (t) => ({
        nodes: [...t.nodes, newNode],
      })),
    });
  },

  renameNode: (nodeId, newLabel) => {
    get().pushUndoSnapshot();
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        nodes: tab.nodes.map((n) =>
          n.id === nodeId ? { ...n, data: { ...n.data, label: newLabel } } : n
        ),
      })),
    });
  },

  // ── Bypass (core#128) ──

  toggleNodeBypass: (nodeId) => {
    const tab = get().getActiveTab();
    const node = tab.nodes.find((n) => n.id === nodeId);
    if (!node || !isBypassable(node)) return;
    get().pushUndoSnapshot();
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (t) =>
        bypassPatch(t, new Set([nodeId]), !node.data.bypassed),
      ),
    });
  },

  toggleBypassForSelection: () => {
    const tab = get().getActiveTab();
    // React Flow's own multi-selection first. A plain click sets BOTH
    // `selected` and `selectedNodeId`, so the fallback only matters for
    // selections made programmatically (a modal, a test, a plugin).
    const marked = tab.nodes.filter((n) => n.selected);
    const pool = marked.length > 0
      ? marked
      : tab.nodes.filter((n) => n.id === tab.selectedNodeId);
    const targets = pool.filter(isBypassable);
    if (targets.length === 0) return false;
    // A mixed selection becomes uniform: mute everything unless all of it is
    // already muted, in which case the whole selection comes back. Same rule
    // ComfyUI applies, and it makes the shortcut its own inverse.
    const bypassed = targets.some((n) => !n.data.bypassed);
    const ids = new Set(targets.map((n) => n.id));
    get().pushUndoSnapshot();
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (t) =>
        bypassPatch(t, ids, bypassed),
      ),
    });
    return true;
  },

  // ── Template insertion (core#128) ──

  insertGraph: (incomingNodes, incomingEdges, incomingSubgraphs = []) => {
    if (incomingNodes.length === 0) return;
    const tab = get().getActiveTab();
    get().pushUndoSnapshot();

    // Every incoming id is remapped, unconditionally. A template ships
    // whatever ids its author saved ("node_1", "conv"), and two templates —
    // or a template and the current graph — routinely collide; reusing an
    // incoming id would silently replace the node already wearing it.
    const idMap = new Map<string, string>();
    for (const node of incomingNodes) idMap.set(node.id, generateId());

    const offset = insertionOffset(tab.nodes, incomingNodes);

    const newNodes: Node<NodeData>[] = incomingNodes.map((node) => {
      const data: NodeData = { ...node.data, executionStatus: 'idle', error: undefined };
      if (node.type === 'noteNode' && data.boundToNodeId) {
        const remapped = idMap.get(data.boundToNodeId);
        data.boundToNodeId = remapped ?? null;
        if (!remapped) data.boundOffset = null;
      }
      return {
        ...node,
        id: idMap.get(node.id)!,
        position: {
          x: node.position.x + offset.x,
          y: node.position.y + offset.y,
        },
        selected: true,
        data,
      };
    });

    const newEdges: Edge[] = incomingEdges
      // An edge naming a node the template did not ship would dangle over the
      // existing graph, so it is dropped rather than remapped to nothing.
      .filter((e) => idMap.has(e.source) && idMap.has(e.target))
      .map((e) => ({
        ...e,
        id: generateId(),
        source: idMap.get(e.source)!,
        target: idMap.get(e.target)!,
      }));

    // Selecting exactly what was inserted (and nothing else) is what makes
    // "insert, then drag/lay-out the new block" work as one gesture — the
    // same thing paste does.
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (t) => {
        const merged = mergeIncomingSubgraphs(
          t.subgraphs,
          incomingSubgraphs,
          newNodes,
        );
        return {
          nodes: [
            ...t.nodes.map((n) => ({ ...n, selected: false })),
            ...merged.nodes,
          ],
          edges: [...t.edges, ...newEdges],
          subgraphs: merged.subgraphs,
        };
      }),
    });

    // The block is placed below the existing graph, which on a canvas the
    // user has panned or zoomed into can be entirely off-screen — an insert
    // that looks like it did nothing. Reuse auto-layout's one-shot fit
    // request so the viewport lands on what just arrived.
    const inserted = nodesBoundingBox(newNodes as Node[]);
    if (inserted && inserted.width > 0 && inserted.height > 0) {
      useUIStore.getState().requestLayoutFit(inserted);
    }
  },

  // ── Note actions ──

  addNote: (kind, position) => {
    get().pushUndoSnapshot();
    const node: Node<NodeData> = {
      id: generateId(),
      type: 'noteNode',
      position,
      data: {
        label: 'Note',
        type: 'note',
        params: {},
        noteKind: kind,
        noteContent: '',
        noteColor: '#3d3d1a',
        boundToNodeId: null,
        boundOffset: null,
        noteWidth: 200,
        noteHeight: kind === 'image' ? 150 : undefined,
      },
    };
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        nodes: [...tab.nodes, node],
      })),
    });
  },

  updateNoteData: (nodeId, updates) => {
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        nodes: tab.nodes.map((n) =>
          n.id === nodeId ? { ...n, data: { ...n.data, ...updates } } : n
        ),
      })),
    });
  },

  bindNoteToNode: (noteId, targetNodeId) => {
    get().pushUndoSnapshot();
    const tab = get().getActiveTab();
    const note = tab.nodes.find((n) => n.id === noteId);
    const target = tab.nodes.find((n) => n.id === targetNodeId);
    if (!note || !target) return;
    const offset = {
      x: note.position.x - target.position.x,
      y: note.position.y - target.position.y,
    };
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (t) => ({
        nodes: t.nodes.map((n) =>
          n.id === noteId
            ? { ...n, data: { ...n.data, boundToNodeId: targetNodeId, boundOffset: offset } }
            : n
        ),
      })),
    });
  },

  bindNoteToNearest: (noteId) => {
    const tab = get().getActiveTab();
    const note = tab.nodes.find((n) => n.id === noteId);
    if (!note) return;
    const cx = note.position.x + (note.measured?.width ?? 200) / 2;
    const cy = note.position.y + (note.measured?.height ?? 80) / 2;
    let bestId: string | null = null;
    let bestDist = Infinity;
    for (const n of tab.nodes) {
      if (n.type === 'noteNode') continue;
      const nx = n.position.x + (n.measured?.width ?? 200) / 2;
      const ny = n.position.y + (n.measured?.height ?? 80) / 2;
      const d = (cx - nx) ** 2 + (cy - ny) ** 2;
      if (d < bestDist) {
        bestDist = d;
        bestId = n.id;
      }
    }
    if (bestId) get().bindNoteToNode(noteId, bestId);
  },

  unbindNote: (noteId) => {
    get().pushUndoSnapshot();
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        nodes: tab.nodes.map((n) =>
          n.id === noteId
            ? { ...n, data: { ...n.data, boundToNodeId: null, boundOffset: null } }
            : n
        ),
      })),
    });
  },

  applyLayout: (mode) => {
    const tabId = get().activeTabId;
    if (!tabId) return;
    const activeTab = get().getActiveTab();
    const selectedIds = new Set(
      activeTab.nodes.filter((n) => n.selected).map((n) => n.id),
    );
    const { nodes: laidNodes, targetIds } = autoLayoutWithTargets(
      activeTab.nodes as Node[],
      activeTab.edges as Edge[],
      mode,
      selectedIds,
    );
    const newNodes = laidNodes as Node<NodeData>[];
    get().pushUndoSnapshot();
    set((state) => ({
      tabs: state.tabs.map((tab) =>
        tab.id === tabId ? { ...tab, nodes: newNodes } : tab,
      ),
    }));
    // Ask the canvas to re-fit the viewport to what was laid out (scoped so
    // "selected" mode focuses the selection). Bound notes follow their parents
    // during layout, so include them or they could end up cropped off-screen.
    // The request carries the bounding box computed from the fresh store nodes
    // so the canvas never races React Flow's internal position sync.
    if (targetIds.size > 0) {
      const fitIds = new Set(targetIds);
      for (const n of newNodes) {
        const boundTo = n.data.boundToNodeId;
        if (n.type === 'noteNode' && boundTo && fitIds.has(boundTo)) {
          fitIds.add(n.id);
        }
      }
      const bounds = nodesBoundingBox(
        newNodes.filter((n) => fitIds.has(n.id)) as Node[],
      );
      if (bounds && bounds.width > 0 && bounds.height > 0) {
        useUIStore.getState().requestLayoutFit(bounds);
      }
    }
    // Warn if there are unbound notes on the canvas
    const hasUnboundNotes = newNodes.some(
      (n) => n.type === 'noteNode' && !n.data.boundToNodeId
    );
    if (hasUnboundNotes) {
      useToastStore.getState().addToast(
        useI18n.getState().t('note.layoutWarning'),
        'warning',
      );
    }
  },

  // ── Undo/Redo ──
  //
  // Snapshots are SHALLOW array copies, not deep clones (#125).
  //
  // Why the deep clone was unnecessary: every writer in this store replaces
  // what it changes rather than mutating it — `nodes.map(n => ({...n, data:
  // {...n.data, ...}}))` and friends, without exception — and React Flow does
  // the same with the arrays we hand it (`applyNodeChanges` builds a new
  // array, shallow-copies only the elements a change touches, and never
  // writes through `element.data`). So a node object reachable from a
  // snapshot can never be modified in place; the only way its contents can
  // change is by the live array being replaced, which leaves the snapshot's
  // array pointing at the old objects. Copying the ARRAY is what the
  // snapshot needs, and that is all it needs.
  //
  // Why it was actively harmful: at ~1 KB of JSON per node, a 300-node graph
  // paid a full stringify + parse on every drag start and every delete —
  // on the interaction's critical path. It was also lossy (JSON drops
  // `undefined`-valued keys such as `data.error`) and it detached each node's
  // `data.definition` from the shared registry object. And because deep
  // clones fail React Flow's `userNode === internals.userNode` identity
  // check, every undo forced it to re-adopt all N nodes; shallow snapshots
  // restore untouched nodes by identity.
  //
  // A fresh `[...]` rather than the bare reference: the arrays are never
  // mutated, so aliasing is harmless today, but `undo()` promotes a
  // snapshot's array to being the live one and a copy costs a pointer memcpy.

  pushUndoSnapshot: () => {
    const tab = get().getActiveTab();
    const snapshot: UndoSnapshot = {
      nodes: [...tab.nodes],
      edges: [...tab.edges],
      subgraphs: [...tab.subgraphs],
    };
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (t) => ({
        undoStack: [...t.undoStack.slice(-(MAX_UNDO - 1)), snapshot],
        redoStack: [],
      })),
    });
  },

  undo: () => {
    const tab = get().getActiveTab();
    if (tab.undoStack.length === 0) return;
    const current: UndoSnapshot = {
      nodes: [...tab.nodes],
      edges: [...tab.edges],
      subgraphs: [...tab.subgraphs],
    };
    const prev = tab.undoStack[tab.undoStack.length - 1];
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (t) => ({
        nodes: prev.nodes,
        edges: prev.edges,
        subgraphs: prev.subgraphs,
        undoStack: t.undoStack.slice(0, -1),
        redoStack: [...t.redoStack, current],
      })),
    });
  },

  redo: () => {
    const tab = get().getActiveTab();
    if (tab.redoStack.length === 0) return;
    const current: UndoSnapshot = {
      nodes: [...tab.nodes],
      edges: [...tab.edges],
      subgraphs: [...tab.subgraphs],
    };
    const next = tab.redoStack[tab.redoStack.length - 1];
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (t) => ({
        nodes: next.nodes,
        edges: next.edges,
        subgraphs: next.subgraphs,
        redoStack: t.redoStack.slice(0, -1),
        undoStack: [...t.undoStack, current],
      })),
    });
  },

  // ── Clipboard (copy/paste) ──

  clipboard: null,

  copySelectedNodes: () => {
    const tab = get().getActiveTab();
    const selected = tab.nodes.filter((n) => n.selected);
    if (selected.length === 0) return;
    const selectedIds = new Set(selected.map((n) => n.id));
    const internalEdges = tab.edges.filter(
      (e) => selectedIds.has(e.source) && selectedIds.has(e.target)
    );
    // TRANSITIVE (core#137 review): a copied instance's definition can itself
    // contain instances of other definitions, to any depth. Collecting only
    // the ids the SELECTED CANVAS NODES name leaves those behind, and a paste
    // into another tab then lands a definition referencing a `subgraph:<id>`
    // that does not exist there -- a node the canvas happily draws and the
    // server refuses to run.
    const copiedSubgraphIds = reachableSubgraphIds(selected, tab.subgraphs);
    set({
      clipboard: {
        nodes: JSON.parse(JSON.stringify(selected)),
        edges: JSON.parse(JSON.stringify(internalEdges)),
        subgraphs: JSON.parse(
          JSON.stringify(
            tab.subgraphs.filter((d) => copiedSubgraphIds.has(d.id)),
          ),
        ),
      },
    });
  },

  pasteNodes: () => {
    const { clipboard } = get();
    if (!clipboard || clipboard.nodes.length === 0) return;
    get().pushUndoSnapshot();

    const idMap = new Map<string, string>();
    clipboard.nodes.forEach((n) => idMap.set(n.id, generateId()));

    const newNodes: Node<NodeData>[] = clipboard.nodes.map((n) => {
      const cloned = JSON.parse(JSON.stringify(n));
      const data = { ...cloned.data, executionStatus: 'idle' as const, error: undefined };
      // Remap note binding: if bound parent was also copied, remap; otherwise clear
      if (cloned.type === 'noteNode' && data.boundToNodeId) {
        const remapped = idMap.get(data.boundToNodeId);
        data.boundToNodeId = remapped ?? null;
        if (!remapped) data.boundOffset = null;
      }
      return {
        ...cloned,
        id: idMap.get(n.id)!,
        position: { x: n.position.x + 50, y: n.position.y + 50 },
        selected: true,
        data,
      };
    });

    const newEdges: Edge[] = clipboard.edges.map((e) => ({
      ...JSON.parse(JSON.stringify(e)),
      id: generateId(),
      source: idMap.get(e.source) ?? e.source,
      target: idMap.get(e.target) ?? e.target,
    }));

    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => {
        const merged = mergeIncomingSubgraphs(
          tab.subgraphs,
          clipboard.subgraphs ?? [],
          newNodes,
        );
        return {
          nodes: [
            ...tab.nodes.map((n) => ({ ...n, selected: false })),
            ...merged.nodes,
          ],
          edges: [...tab.edges, ...newEdges],
          subgraphs: merged.subgraphs,
        };
      }),
    });
  },

  // ── Dirty tracking (partial re-execution) ──

  markDirty: (nodeId) =>
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => {
        const next = new Set(tab.dirtyNodeIds);
        next.add(nodeId);
        return { dirtyNodeIds: next };
      }),
    }),

  clearDirty: () =>
    set({
      tabs: updateTab(get().tabs, get().activeTabId, () => ({
        dirtyNodeIds: new Set(),
      })),
    }),

  getDirtyWithDownstream: () => {
    const tab = get().getActiveTab();
    if (tab.dirtyNodeIds.size === 0) return [];

    // Build adjacency: source -> targets
    const adj = new Map<string, string[]>();
    for (const edge of tab.edges) {
      if (!adj.has(edge.source)) adj.set(edge.source, []);
      adj.get(edge.source)!.push(edge.target);
    }

    // BFS from all dirty nodes
    const result = new Set<string>(tab.dirtyNodeIds);
    const queue = [...tab.dirtyNodeIds];
    while (queue.length > 0) {
      const nid = queue.shift()!;
      for (const downstream of adj.get(nid) ?? []) {
        if (!result.has(downstream)) {
          result.add(downstream);
          queue.push(downstream);
        }
      }
    }
    // Bypassed nodes are dropped from the RESULT but not from the walk
    // (core#128). This list becomes the backend's `changed_nodes` — a
    // force-re-execute hint — and a bypassed node is never executed, so
    // naming one says nothing.
    //
    // `markDirty` upstream of this stays unconditional, and that is REQUIRED,
    // not merely conservative: a param can change a bypassed node's own PORT
    // SET (define_outputs_dynamic — Split's `chunks` is the live example), so
    // editing a muted node's params can change which input each output
    // forwards, and therefore what its consumers receive. Skipping the mark
    // would leave those consumers on cached values computed from the old
    // pass-through. Dirtiness travelling THROUGH the node is the same story,
    // which is why the BFS above is unfiltered.
    const bypassed = new Set(
      tab.nodes.filter((n) => n.data.bypassed).map((n) => n.id),
    );
    return [...result].filter((id) => !bypassed.has(id));
  },

  // ── Execution actions (active tab) ──

  setStatus: (s) =>
    set({ tabs: updateTab(get().tabs, get().activeTabId, () => ({ status: s })) }),

  addLog: (entry) =>
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        logs: [...tab.logs, { ...entry, timestamp: Date.now() }],
      })),
    }),

  clearLogs: () =>
    set({ tabs: updateTab(get().tabs, get().activeTabId, () => ({ logs: [] })) }),

  // ── Tab-specific execution actions (WS handlers target a specific tab) ──

  // The single write path for streamed execution state (#125). One call
  // rebuilds each affected tab's nodes array exactly once, no matter how
  // many nodes or how many events the batch covers, and leaves untouched
  // nodes referentially identical so React Flow re-diffs only what moved.
  // Live runs reach it through `nodeUpdateQueue`, which buffers a frame's
  // worth of events into one batch.
  applyTabNodeUpdates: (updates) =>
    set((state) => {
      let anyTabChanged = false;
      const tabs = state.tabs.map((tab) => {
        const patches = updates.get(tab.id);
        if (!patches || patches.size === 0) return tab;
        let changed = false;
        const nodes = tab.nodes.map((n) => {
          const patch = patches.get(n.id);
          if (!patch) return n;
          changed = true;
          const data = { ...n.data };
          if (patch.status) {
            data.executionStatus = patch.status.executionStatus;
            data.error = patch.status.error;
          }
          if (patch.progress) data.progress = patch.progress;
          return { ...n, data };
        });
        if (!changed) return tab;
        anyTabChanged = true;
        return { ...tab, nodes };
      });
      // A batch naming only stale tabs/nodes (a run whose graph was edited
      // mid-flight) leaves `tabs` out of the patch entirely, so the array
      // reference every selector reads is preserved. Zustand still runs its
      // listeners — an empty patch is not a no-op at that level — but no
      // selector sees a change, so nothing re-renders and React Flow is not
      // handed a new nodes array to diff.
      return anyTabChanged ? { tabs } : {};
    }),

  // Immediate single-node forms over the same applier. Nothing in the live
  // app calls them any more — the WS handler batches through
  // `nodeUpdateQueue` — and they are kept deliberately, for two callers that
  // are not the live path: the perf harness drives them as its pre-#125
  // comparison (writing straight through, once per event, which is what these
  // measure), and tests that assert on one node's outcome should not have to
  // build a nested Map to set it up. Retire them only if both go away.
  setTabNodeExecutionStatus: (tabId, nodeId, status, error) =>
    get().applyTabNodeUpdates(
      new Map([[tabId, new Map([[nodeId, { status: { executionStatus: status, error } }]])]]),
    ),

  setTabNodeProgress: (tabId, nodeId, progress) =>
    get().applyTabNodeUpdates(new Map([[tabId, new Map([[nodeId, { progress }]])]])),

  setTabOutputSummary: (tabId, nodeId, summary) =>
    set({
      tabs: updateTab(get().tabs, tabId, (tab) => ({
        outputSummaries: { ...tab.outputSummaries, [nodeId]: summary },
      })),
    }),

  clearOutputSummaries: () =>
    set({ tabs: updateTab(get().tabs, get().activeTabId, () => ({ outputSummaries: {} })) }),

  setTabStatus: (tabId, s) =>
    set({ tabs: updateTab(get().tabs, tabId, () => ({ status: s })) }),

  addTabLog: (tabId, entry) =>
    set({
      tabs: updateTab(get().tabs, tabId, (tab) => ({
        logs: [...tab.logs, { ...entry, timestamp: Date.now() }],
      })),
    }),

  // ── Teaching Inspector actions ──

  toggleRecord: () =>
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        recordOutputs: !tab.recordOutputs,
      })),
    }),

  // Both setters below are called on EVERY WebSocket frame (#121 tracks the
  // run and cursor each one carries), so both short-circuit to an untouched
  // `tabs` array when there is nothing to change. `updateTab` maps, and a
  // map always yields a new array and a new tab object — which every
  // component selecting on those would see as a change, once per frame, for
  // no reason.
  setLastRunId: (tabId, runId) =>
    set((state) => {
      const tab = state.tabs.find((t) => t.id === tabId);
      if (!tab || tab.lastRunId === runId) return {};
      // A new run always starts the cursor over: cursors are per run, so
      // carrying the previous one forward would make a resume skip the
      // beginning of the new run's log.
      return {
        tabs: updateTab(state.tabs, tabId, () => ({
          lastRunId: runId,
          lastRunCursor: 0,
        })),
      };
    }),

  setLastRunCursor: (tabId, cursor) =>
    set((state) => {
      // Monotonic: frames arrive in cursor order, but a stale frame from a
      // previous attachment must never rewind the resume point.
      const tab = state.tabs.find((t) => t.id === tabId);
      if (!tab || cursor <= tab.lastRunCursor) return {};
      return { tabs: updateTab(state.tabs, tabId, () => ({ lastRunCursor: cursor })) };
    }),

  setActiveSegment: (segment) =>
    set({
      tabs: updateTab(get().tabs, get().activeTabId, () => ({ activeSegment: segment })),
    }),

  addSegmentGroup: (segment) =>
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        segmentGroups: [...tab.segmentGroups.filter((s) => s.id !== segment.id), segment],
      })),
    }),

  removeSegmentGroup: (id) =>
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        segmentGroups: tab.segmentGroups.filter((s) => s.id !== id),
        activeSegment: tab.activeSegment?.id === id ? null : tab.activeSegment,
      })),
    }),

  setSegmentGroups: (segments) =>
    set({
      tabs: updateTab(get().tabs, get().activeTabId, () => ({
        segmentGroups: segments,
      })),
    }),

  // ── Educational toggles (A1/A2/A3) ──

  toggleVerbose: () =>
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        verboseMode: !tab.verboseMode,
      })),
    }),

  togglePersistWeights: () =>
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        weightsPersistent: !tab.weightsPersistent,
      })),
    }),

  toggleBackward: () =>
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        backwardMode: !tab.backwardMode,
      })),
    }),

  toggleAutoBackward: () =>
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        autoBackward: !tab.autoBackward,
      })),
    }),

  // ── Reproducibility (core#134) ──

  setSeed: (seed) =>
    set({
      tabs: updateTab(get().tabs, get().activeTabId, () => ({
        // Normalised here rather than at each call site so every entry point
        // agrees on what "no seed" is: null. A cleared field and a
        // half-typed one that parsed to NaN mean the same thing to a run.
        seed: seed === null || Number.isNaN(seed) ? null : Math.trunc(seed),
      })),
    }),

  toggleDeterministic: () =>
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        deterministic: !tab.deterministic,
      })),
    }),
}));

// ── Loading the IndexedDB tier ──
//
// `loadTabs()` above runs at import and reads localStorage, because the store
// has to exist before React renders and IndexedDB cannot be read
// synchronously. That result is a placeholder in the IndexedDB era: whatever
// the pre-#125 build last wrote, which may be nothing, and which is not
// updated any more once IndexedDB takes over. Hydration below is what makes
// the real state appear, one IndexedDB round-trip later.

/** What a hydration attempt did, for tests and for the caller's logging. */
export type HydrationOutcome =
  /** No IndexedDB here (jsdom, private mode, sandboxed frame). */
  | 'unavailable'
  /** IndexedDB held this scope's tabs; the store now shows them. */
  | 'loaded'
  /** Nothing in IndexedDB yet; localStorage's tabs were copied across. */
  | 'migrated'
  /** Nothing in either tier; the default tab was written as the seed. */
  | 'seeded'
  /** The storage scope moved while reading; the result was thrown away. */
  | 'superseded'
  /** IndexedDB is present but unusable; localStorage stays authoritative. */
  | 'failed';

// How many hydrations are in flight. A debounced save that fires inside that
// window would write the PRE-hydration tabs over the records being read — the
// classic rehydrate/autosave race — so it defers instead. A counter rather
// than a flag because the import-time hydration can still be running when a
// resolved project starts a second one, and the first to finish must not
// declare the window closed.
let _hydrationsInFlight = 0;

/**
 * Bring the tab tree in line with IndexedDB for the current storage scope,
 * migrating localStorage's contents on the first run that finds it empty.
 *
 * Called at import for the base scope and again once a project resolves, and
 * exported so tests can await it.
 *
 * Never rejects — but a failure here is NOT benign, and the toast it raises
 * is the only thing that says so. Once a workspace has migrated, localStorage
 * is frozen at migration-day content: it is still read, never written. So a
 * failed READ (Firefox private mode, a corrupted database, storage evicted
 * under pressure — `idbAvailable()` is true in all three) puts those old tabs
 * on screen, accepts edits against them, routes the saves to the localStorage
 * fallback, and lets the next healthy session load IndexedDB and discard the
 * lot. The user has to know to export before that happens.
 */
export async function hydrateTabsFromPersistence(): Promise<HydrationOutcome> {
  if (!idbAvailable()) return 'unavailable';
  const scope = _storageKey();
  _hydrationsInFlight += 1;
  try {
    const snapshot = await readSnapshot(scope);

    // The scope can move out from under this read. The base scope's hydration
    // starts at import; a resolved project switches the scope as soon as
    // /api/health answers, which can easily beat an IndexedDB round-trip.
    // Applying now would replace the project's graphs with the base ones —
    // and on the migrate branch it would write the project's tabs INTO the
    // base scope. Both are silent data loss, so a stale read is discarded.
    if (scope !== _storageKey()) return 'superseded';

    if (snapshot) {
      // Reuse the live tab object for any id already on screen. A tab's
      // record covers what is on disk; its socket, logs, undo stacks and
      // run cursor are session state that hydration has no business
      // recreating — and re-creating the socket would strand the WS
      // handlers `useGraphExecution` already attached to the old one.
      const existing = new Map(
        useTabStore.getState().tabs.map((t) => [t.id, t] as const),
      );
      const tabs = snapshot.tabs.map((rec) =>
        tabFromPersisted(rec, existing.get(rec.id) ?? createTabState(rec.id, rec.name)),
      );
      useTabStore.setState({ tabs, activeTabId: snapshot.activeTabId });
      return 'loaded';
    }

    // Nothing under this scope in IndexedDB. Whatever `loadTabs()` put in the
    // store IS the pre-#125 state (or a fresh default), so writing it across
    // is the migration — one copy, on the first load after upgrading.
    const hadLegacy = localStorage.getItem(scope) !== null;
    const state = useTabStore.getState();
    await writeSnapshot(scope, persistedTabsFor(state.tabs), state.activeTabId);
    // The legacy blob is deliberately left where it is: an in-place downgrade
    // to a build without this code still opens the graph it last saw, and the
    // load path above still reads it for one release. It stops being updated
    // from here on, so IndexedDB is the only tier that stays current.
    return hadLegacy ? 'migrated' : 'seeded';
  } catch {
    // See the note above: silence here reads to the user as "nothing to
    // restore" and costs them the session's work at the next healthy load.
    warnPersistence('persistence.storageUnavailable');
    return 'failed';
  } finally {
    _hydrationsInFlight -= 1;
  }
}

// The most recent hydration, so callers (and tests) can wait for the tab tree
// to be real rather than guessing at a timeout.
let _lastHydration: Promise<HydrationOutcome> = Promise.resolve('unavailable');

function _startHydration(): void {
  _lastHydration = hydrateTabsFromPersistence();
}

/**
 * Resolves once the newest hydration attempt has settled.
 *
 * No production caller, deliberately: acting on a bad outcome is
 * `hydrateTabsFromPersistence`'s own job (it raises the toast at the point of
 * failure, where the reason is still in scope), so nothing has to remember to
 * await this and check. It exists so tests can be deterministic about a step
 * that is otherwise only observable as "the tabs changed a bit later", and so
 * a future caller that genuinely needs to sequence against hydration has a
 * handle rather than a timeout.
 */
export function whenTabsHydrated(): Promise<HydrationOutcome> {
  return _lastHydration;
}

// ── Auto-save ──
//
// React Flow fires a state change per pointer event during a drag, and
// status updates from the backend stream tick it many times per run. Writing
// the whole tab tree on every tick wastes CPU and can stutter visibly on
// large graphs. We collapse a burst of changes into a single trailing-edge
// save; the actual `saveTabs` call reads fresh state at fire time so we never
// persist a stale snapshot.
const SAVE_DEBOUNCE_MS = 250;
let _saveTimer: ReturnType<typeof setTimeout> | null = null;

function _scheduleSave() {
  if (_saveTimer !== null) clearTimeout(_saveTimer);
  _saveTimer = setTimeout(() => {
    _saveTimer = null;
    if (_hydrationsInFlight > 0) {
      // Re-arm rather than save. Hydration always settles (it swallows its
      // own failures), so this defers by one window, not forever.
      _scheduleSave();
      return;
    }
    const s = useTabStore.getState();
    saveTabs(s.tabs, s.activeTabId);
  }, SAVE_DEBOUNCE_MS);
}

useTabStore.subscribe(() => {
  _scheduleSave();
});

// Start the IndexedDB read for the base scope immediately, so a non-project
// session has its real tabs before the first paint of the canvas rather than
// after a REST round-trip. Project mode repeats this from
// `rehydrateForProject` once the scope is actually known.
_startHydration();
