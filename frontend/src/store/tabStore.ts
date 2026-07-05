import { create } from 'zustand';
import { applyNodeChanges, applyEdgeChanges } from '@xyflow/react';
import type { Node, Edge, NodeChange, EdgeChange, Connection } from '@xyflow/react';
import { generateId, buildFlowNode } from '../utils';
import { autoLayout, type LayoutMode } from '../utils/autoLayout';
import type { NodeData, NodeDefinition, PresetDefinition, ExecutionStatus, OutputSummary, NodeProgress, SegmentGroup } from '../types';
import { ExecutionWebSocket } from '../api/ws';
import { useToastStore } from './toastStore';
import { useI18n } from '../i18n';

// ── Per-tab state ──

export interface LogEntry {
  timestamp: number;
  nodeId?: string;
  message: string;
  type: 'info' | 'error' | 'success';
}

interface UndoSnapshot {
  nodes: Node<NodeData>[];
  edges: Edge[];
}

const MAX_UNDO = 50;

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
  // flow
  nodes: Node<NodeData>[];
  edges: Edge[];
  selectedNodeId: string | null;
  presetModalNodeId: string | null;
  subgraphModalNodeId: string | null;
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
    nodes: [],
    edges: [],
    selectedNodeId: null,
    presetModalNodeId: null,
    subgraphModalNodeId: null,
    undoStack: [],
    redoStack: [],
    dirtyNodeIds: new Set(),
    status: 'idle',
    logs: [],
    ws: new ExecutionWebSocket(),
    outputSummaries: {},
    recordOutputs: true,
    lastRunId: null,
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
  setTabNodeExecutionStatus: (tabId: string, nodeId: string, status: NodeData['executionStatus'], error?: string) => void;
  setTabNodeProgress: (tabId: string, nodeId: string, progress: NodeProgress) => void;
  setTabOutputSummary: (tabId: string, nodeId: string, summary: Record<string, OutputSummary>) => void;
  clearOutputSummaries: () => void;
  setTabStatus: (tabId: string, s: ExecutionStatus) => void;
  addTabLog: (tabId: string, entry: Omit<LogEntry, 'timestamp'>) => void;

  // Teaching Inspector actions
  toggleRecord: () => void;
  setLastRunId: (tabId: string, runId: string) => void;
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

// ── LocalStorage persistence ──

const STORAGE_KEY = 'codefyui-tabs';

interface PersistedTab {
  id: string;
  name: string;
  description?: string;
  currentGraphFile?: string | null;
  nodes: Node<NodeData>[];
  edges: Edge[];
  segmentGroups?: SegmentGroup[];
  recordOutputs?: boolean;
  verboseMode?: boolean;
  graphId?: string;
  weightsPersistent?: boolean;
  backwardMode?: boolean;
  autoBackward?: boolean;
}

// Throttle the user-facing quota toast so a big graph editing session
// doesn't burst N toasts when localStorage fills up. One per minute is
// plenty to surface "your work isn't being saved".
let _lastQuotaWarn = 0;

function saveTabs(tabs: TabState[], activeTabId: string) {
  try {
    const data: { tabs: PersistedTab[]; activeTabId: string } = {
      activeTabId,
      tabs: tabs.map((t) => ({
        id: t.id,
        name: t.name,
        description: t.description,
        currentGraphFile: t.currentGraphFile,
        nodes: t.nodes,
        edges: t.edges,
        segmentGroups: t.segmentGroups,
        recordOutputs: t.recordOutputs,
        verboseMode: t.verboseMode,
        graphId: t.graphId,
        weightsPersistent: t.weightsPersistent,
        backwardMode: t.backwardMode,
        autoBackward: t.autoBackward,
      })),
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch {
    // QuotaExceededError / SecurityError / private mode etc. The README
    // promises auto-save; failing silently lets the user lose work without
    // realising. Surface once per minute at most.
    const now = Date.now();
    if (now - _lastQuotaWarn > 60_000) {
      _lastQuotaWarn = now;
      try {
        const message = useI18n.getState().t('persistence.quotaError');
        useToastStore.getState().addToast(message, 'error');
      } catch {
        /* toast/i18n not initialised yet — nothing useful to do here */
      }
    }
  }
}

function loadTabs(): { tabs: TabState[]; activeTabId: string } {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const data = JSON.parse(raw);
      if (Array.isArray(data.tabs) && data.tabs.length > 0) {
        const tabs: TabState[] = data.tabs.map((t: PersistedTab) => {
          const base = createTabState(t.id, t.name);
          return {
            ...base,
            description: t.description ?? '',
            currentGraphFile: t.currentGraphFile ?? null,
            nodes: t.nodes ?? [],
            edges: t.edges ?? [],
            segmentGroups: Array.isArray(t.segmentGroups) ? t.segmentGroups : [],
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
        });
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
        if (posChanges.length > 0) {
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

        // When a node is removed, unbind notes that were bound to it
        if (hasRemove) {
          const removedIds = new Set(
            changes.filter((c) => c.type === 'remove').map((c) => c.id)
          );
          updatedNodes = updatedNodes.map((n) => {
            if (n.type !== 'noteNode' || !n.data.boundToNodeId) return n;
            if (!removedIds.has(n.data.boundToNodeId)) return n;
            return { ...n, data: { ...n.data, boundToNodeId: null, boundOffset: null } };
          });
        }

        return { nodes: updatedNodes };
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
            boundOffset: n.data.boundOffset,
            noteWidth: n.data.noteWidth,
            noteHeight: n.data.noteHeight,
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
          ...(n.data.isPreset ? { internalParams: n.data.internalParams } : {}),
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
    get().pushUndoSnapshot();
    set((state) => ({
      tabs: state.tabs.map((tab) => {
        if (tab.id !== tabId) return tab;
        const selectedIds = new Set(
          tab.nodes.filter((n) => n.selected).map((n) => n.id),
        );
        const newNodes = autoLayout(tab.nodes, tab.edges, mode, selectedIds) as Node<NodeData>[];
        return {
          ...tab,
          nodes: newNodes,
        };
      }),
    }));
    // Warn if there are unbound notes on the canvas
    const tab = get().getActiveTab();
    const hasUnboundNotes = tab.nodes.some(
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

  pushUndoSnapshot: () => {
    const tab = get().getActiveTab();
    const snapshot: UndoSnapshot = {
      nodes: JSON.parse(JSON.stringify(tab.nodes)),
      edges: JSON.parse(JSON.stringify(tab.edges)),
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
      nodes: JSON.parse(JSON.stringify(tab.nodes)),
      edges: JSON.parse(JSON.stringify(tab.edges)),
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
      nodes: JSON.parse(JSON.stringify(tab.nodes)),
      edges: JSON.parse(JSON.stringify(tab.edges)),
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
    return [...result];
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

  setTabNodeExecutionStatus: (tabId, nodeId, status, error) =>
    set({
      tabs: updateTab(get().tabs, tabId, (tab) => ({
        nodes: tab.nodes.map((n) =>
          n.id === nodeId
            ? { ...n, data: { ...n.data, executionStatus: status, error } }
            : n
        ),
      })),
    }),

  setTabNodeProgress: (tabId, nodeId, progress) =>
    set({
      tabs: updateTab(get().tabs, tabId, (tab) => ({
        nodes: tab.nodes.map((n) =>
          n.id === nodeId
            ? { ...n, data: { ...n.data, progress } }
            : n
        ),
      })),
    }),

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

  setLastRunId: (tabId, runId) =>
    set({
      tabs: updateTab(get().tabs, tabId, () => ({ lastRunId: runId })),
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

// ── Auto-save to localStorage ──
//
// React Flow fires a state change per pointer event during a drag, and
// status updates from the backend stream tick it many times per run. Writing
// (and JSON-stringifying) the whole tab tree on every tick wastes CPU and
// can stutter visibly on large graphs. We collapse a burst of changes into
// a single trailing-edge save; the actual `saveTabs` call reads fresh state
// at fire time so we never persist a stale snapshot.
const SAVE_DEBOUNCE_MS = 250;
let _saveTimer: ReturnType<typeof setTimeout> | null = null;

function _scheduleSave() {
  if (_saveTimer !== null) clearTimeout(_saveTimer);
  _saveTimer = setTimeout(() => {
    _saveTimer = null;
    const s = useTabStore.getState();
    saveTabs(s.tabs, s.activeTabId);
  }, SAVE_DEBOUNCE_MS);
}

useTabStore.subscribe(() => {
  _scheduleSave();
});
