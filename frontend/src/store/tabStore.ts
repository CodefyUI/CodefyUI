import { create } from 'zustand';
import { applyNodeChanges, applyEdgeChanges } from '@xyflow/react';
import type { Node, Edge, NodeChange, EdgeChange, Connection } from '@xyflow/react';
import { generateId, buildFlowNode, isBypassable } from '../utils';
import { forgetViewport } from '../utils/viewportMemory';
import { idbAvailable } from '../utils/idb';
import { readSnapshot, writeSnapshot } from './tabPersistence';
import { autoLayoutWithTargets, nodesBoundingBox, type LayoutMode } from '../utils/autoLayout';
import type { NodeData, NodeDefinition, PresetDefinition, ExecutionStatus, OutputSummary, NodeProgress, SegmentGroup } from '../types';
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
export type LogKind = 'text' | 'image' | 'progress' | 'chart';

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
}

interface UndoSnapshot {
  nodes: Node<NodeData>[];
  edges: Edge[];
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
  selectedNodeId: string | null;
  presetModalNodeId: string | null;
  subgraphModalNodeId: string | null;
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
    selectedNodeId: null,
    presetModalNodeId: null,
    subgraphModalNodeId: null,
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
  openPresetModal: (id: string) => void;
  closePresetModal: () => void;
  openSubgraphModal: (id: string) => void;
  closeSubgraphModal: () => void;
  openNodeDetail: (id: string, target?: { tab?: string; port?: string }) => void;
  closeNodeDetail: () => void;
  updateSubgraphLayers: (nodeId: string, layersJson: string) => void;
  setNodeExecutionStatus: (nodeId: string, status: NodeData['executionStatus'], error?: string) => void;
  clearExecutionStatus: () => void;
  clear: () => void;
  getSerializedGraph: () => {
    nodes: any[];
    edges: any[];
    presets?: import('../types').PresetDefinition[];
    segmentGroups?: SegmentGroup[];
  };
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
   */
  insertGraph: (nodes: Node<NodeData>[], edges: Edge[]) => void;

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
  clipboard: { nodes: Node<NodeData>[]; edges: Edge[] } | null;
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
}

function updateTab(tabs: TabState[], tabId: string, updater: (tab: TabState) => Partial<TabState>): TabState[] {
  return tabs.map((tab) => (tab.id === tabId ? { ...tab, ...updater(tab) } : tab));
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
  recordOutputs?: boolean;
  /**
   * #121: the run this tab was watching, persisted ONLY while it might
   * still be alive (see saveTabs). On the next load the hook asks the
   * server whether it really is, and re-attaches if so — which is how
   * "close the tab, reopen it, the training is still going" works.
   */
  lastRunId?: string;
  verboseMode?: boolean;
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
function buildPersistedTab(t: TabState): PersistedTab {
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
    // survive a page refresh — the field is "Session only".
    nodes: stripNodeSecretsForPersist(t.nodes),
    edges: t.edges,
    segmentGroups: t.segmentGroups,
    // Only while the run might still be in flight, so a finished run's
    // id never survives a reload: the Inspector's captured outputs live
    // in a process-lifetime store, and pointing it at a run whose
    // outputs are gone would replace "not run yet" with an empty view.
    // Keeps localStorage byte-identical for an idle tab, too.
    ...(t.status === 'running' && t.lastRunId ? { lastRunId: t.lastRunId } : {}),
    recordOutputs: t.recordOutputs,
    verboseMode: t.verboseMode,
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
      cached.scalars === scalars
        ? cached
        : {
            nodes: t.nodes,
            edges: t.edges,
            segmentGroups: t.segmentGroups,
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
    lastRunId: typeof t.lastRunId === 'string' ? t.lastRunId : null,
    recordOutputs: t.recordOutputs ?? true,
    verboseMode: t.verboseMode ?? false,
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
        let nodeDetailNodeId = tab.nodeDetailNodeId;
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
        }

        return { nodes: updatedNodes, nodeDetailNodeId };
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
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        nodes: tab.nodes.map((n) =>
          n.id === nodeId
            ? { ...n, data: { ...n.data, params: { ...n.data.params, ...params } } }
            : n
        ),
      })),
    });
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

  setSelectedNodeId: (id) =>
    set({ tabs: updateTab(get().tabs, get().activeTabId, () => ({ selectedNodeId: id })) }),

  openPresetModal: (id) =>
    set({ tabs: updateTab(get().tabs, get().activeTabId, () => ({ presetModalNodeId: id })) }),

  closePresetModal: () =>
    set({ tabs: updateTab(get().tabs, get().activeTabId, () => ({ presetModalNodeId: null })) }),

  openSubgraphModal: (id) =>
    set({ tabs: updateTab(get().tabs, get().activeTabId, () => ({ subgraphModalNodeId: id })) }),

  closeSubgraphModal: () =>
    set({ tabs: updateTab(get().tabs, get().activeTabId, () => ({ subgraphModalNodeId: null })) }),

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

  updateSubgraphLayers: (nodeId, layersJson) =>
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
    get().pushUndoSnapshot();
    set({
      tabs: updateTab(get().tabs, get().activeTabId, () => ({
        nodes: [],
        edges: [],
        selectedNodeId: null,
        presetModalNodeId: null,
        subgraphModalNodeId: null,
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
    const tab = get().getActiveTab();
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
    };
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

  insertGraph: (incomingNodes, incomingEdges) => {
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
      tabs: updateTab(get().tabs, get().activeTabId, (t) => ({
        nodes: [...t.nodes.map((n) => ({ ...n, selected: false })), ...newNodes],
        edges: [...t.edges, ...newEdges],
      })),
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
    };
    const prev = tab.undoStack[tab.undoStack.length - 1];
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (t) => ({
        nodes: prev.nodes,
        edges: prev.edges,
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
    };
    const next = tab.redoStack[tab.redoStack.length - 1];
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (t) => ({
        nodes: next.nodes,
        edges: next.edges,
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
    set({
      clipboard: {
        nodes: JSON.parse(JSON.stringify(selected)),
        edges: JSON.parse(JSON.stringify(internalEdges)),
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
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        nodes: [
          ...tab.nodes.map((n) => ({ ...n, selected: false })),
          ...newNodes,
        ],
        edges: [...tab.edges, ...newEdges],
      })),
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
