import type { Node } from '@xyflow/react';
import type { NodeData, NodeDefinition } from '../types';

export function generateId(): string {
  return crypto.randomUUID();
}

/**
 * Client-side replica of the backend's ``_sanitize_name`` (routes_graph.py):
 * every char outside [alphanumeric, '-', '_'] becomes '_'. Two different
 * display names can therefore map to the same on-disk file. Python's
 * ``str.isalnum()`` is Unicode-aware (Chinese, accented Latin, etc. are kept),
 * so we mirror that with the Unicode letter/number classes rather than
 * ASCII-only ranges. The backend remains authoritative — this only powers the
 * pre-save overwrite warning, so an exotic-codepoint divergence at worst
 * misses or double-shows the warning; it never affects what is written.
 */
export function sanitizeGraphName(name: string): string {
  return Array.from(name)
    .map((ch) => (/[\p{L}\p{N}]/u.test(ch) || ch === '-' || ch === '_' ? ch : '_'))
    .join('');
}

/**
 * Detect whether saving under ``targetName`` would silently overwrite a
 * DIFFERENT existing graph. Returns the colliding graph's display name (for
 * the confirm dialog) or ``null`` when there is no collision. ``existing`` is
 * the ``/api/graph/list`` result (``file`` is the sanitized stem);
 * ``currentFile`` is the sanitized stem of the graph currently open in the tab
 * — re-saving the SAME graph is never treated as a collision.
 */
export function findGraphNameCollision(
  targetName: string,
  existing: { name: string; file: string }[],
  currentFile: string | null,
): string | null {
  const target = sanitizeGraphName(targetName);
  const hit = existing.find((g) => g.file === target && g.file !== currentFile);
  return hit ? hit.name : null;
}

/**
 * Frontend allowlist mapping NODE_NAME → custom xyflow node type. Nodes not
 * listed here render via the default `baseNode`. Backend stays UI-agnostic;
 * the renderer choice lives in the frontend so saved graphs round-trip without
 * baking a renderer hint into persistence.
 */
export const VIZ_NODE_TYPES: Record<string, string> = {
  Tokenizer: 'tokenizerNode',
  EmbeddingScatter: 'embeddingScatterNode',
  TextInput: 'textInputNode',
  'Edu-SelfAttention': 'eduSelfAttentionNode',
  'Edu-MultiHeadAttention': 'eduMultiHeadAttentionNode',
  AttentionHeatmap: 'attentionHeatmapNode',
  AttentionMask: 'attentionMaskNode',
  'Edu-CrossAttention': 'eduCrossAttentionNode',
  'Edu-KNN': 'eduKNNNode',
};

/**
 * Pick the xyflow node component for a (possibly namespaced) node type:
 * a first-party viz renderer if one is registered for the bare NODE_NAME, else
 * the plugin bridge for namespaced plugin nodes (`<plugin>:Name`), else the
 * default card. Routing plugin nodes through the bridge lets a plugin attach a
 * custom React renderer; with no renderer registered the bridge renders exactly
 * like `baseNode`.
 */
export function resolveNodeComponentType(nodeType: string): string {
  const bare = nodeType.includes(':')
    ? nodeType.slice(nodeType.lastIndexOf(':') + 1)
    : nodeType;
  if (VIZ_NODE_TYPES[bare]) return VIZ_NODE_TYPES[bare];
  return nodeType.includes(':') ? 'pluginNode' : 'baseNode';
}

export const DATA_TYPE_COLORS: Record<string, string> = {
  TENSOR: '#4CAF50',
  MODEL: '#2196F3',
  DATASET: '#FF9800',
  DATALOADER: '#9C27B0',
  OPTIMIZER: '#F44336',
  LOSS_FN: '#E91E63',
  SCALAR: '#00BCD4',
  STRING: '#8BC34A',
  IMAGE: '#FF5722',
  LIST: '#CDDC39',
  ANY: '#9E9E9E',
};

export function getPortColor(dataType: string): string {
  return DATA_TYPE_COLORS[dataType.toUpperCase()] ?? DATA_TYPE_COLORS['ANY'];
}

/**
 * Evaluate a param's ``visible_when`` rule against the current params on
 * the node. Returns true when the param should render. The default rule
 * (no ``visible_when``) is "always visible".
 *
 * Matching is shallow equality after string coercion — sufficient for
 * SELECT / INT / BOOL / FLOAT params, which cover every realistic use of
 * conditional visibility.
 */
export function isParamVisible(
  param: import('../types').ParamDefinition,
  params: Record<string, unknown> | undefined,
): boolean {
  const rule = param.visible_when;
  if (!rule) return true;
  const live = params ?? {};
  for (const [siblingName, expected] of Object.entries(rule)) {
    const actual = live[siblingName];
    if (String(actual) !== String(expected)) return false;
  }
  return true;
}

const SPLIT_MAX_CHUNKS = 32;

function bareName(qualifiedName: string): string {
  const idx = qualifiedName.lastIndexOf(':');
  return idx >= 0 ? qualifiedName.slice(idx + 1) : qualifiedName;
}

/**
 * Resolve a node's *live* output ports, expanding param-driven nodes whose
 * port count depends on a runtime parameter. Mirrors the backend
 * `BaseNode.define_outputs_dynamic` mechanism so palette template and live
 * canvas agree on what handles exist.
 *
 * For nodes whose ports don't depend on params (every node except Split as
 * of writing) this just returns `definition.outputs` verbatim. New
 * dynamic-port nodes added in the future should add a clause here.
 */
export function resolveDynamicOutputs(
  definition: import('../types').NodeDefinition | undefined,
  params: Record<string, unknown> | undefined,
): import('../types').PortDefinition[] {
  if (!definition) return [];
  const bare = bareName(definition.node_name);
  if (bare === 'Split') {
    const raw = params?.chunks;
    const parsed = typeof raw === 'number' ? raw : parseInt(String(raw ?? ''), 10);
    const chunks = Math.max(
      1,
      Math.min(SPLIT_MAX_CHUNKS, Number.isFinite(parsed) ? Math.floor(parsed) : 2),
    );
    return Array.from({ length: chunks }, (_, i) => ({
      name: `chunk_${i}`,
      data_type: 'TENSOR',
      description: `Chunk ${i} of ${chunks}`,
      optional: false,
    }));
  }
  return definition.outputs;
}

/**
 * Reconstruct full ReactFlow nodes from the minimal serialized graph format.
 * The serialized format (from getSerializedGraph / backend save) only stores:
 *   { id, type, position, data: { params, internalParams? } }
 * ReactFlow needs:
 *   { id, type: "baseNode"|"presetNode", position, data: { label, type, params, definition, executionStatus, ... } }
 */
export function resolveSerializedNodes(
  rawNodes: any[],
  definitions: import('../types').NodeDefinition[],
  presets: import('../types').PresetDefinition[],
): import('@xyflow/react').Node<import('../types').NodeData>[] {
  const defMap = new Map(definitions.map((d) => [d.node_name, d]));
  const presetMap = new Map(presets.map((p) => [p.preset_name, p]));

  return rawNodes.map((raw) => {
    const nodeType: string = raw.type ?? '';
    const position = raw.position ?? { x: 0, y: 0 };
    const params = raw.data?.params ?? {};

    // Note node
    if (nodeType === 'note') {
      return {
        id: raw.id,
        type: 'noteNode' as const,
        position,
        data: {
          label: 'Note',
          type: 'note',
          params: {},
          noteKind: raw.data?.noteKind ?? 'text',
          noteContent: raw.data?.noteContent ?? '',
          noteColor: raw.data?.noteColor ?? '#3d3d1a',
          boundToNodeId: raw.data?.boundToNodeId ?? null,
          boundOffset: raw.data?.boundOffset ?? null,
          noteWidth: raw.data?.noteWidth ?? 200,
          noteHeight: raw.data?.noteHeight,
        },
      };
    }

    // Preset node
    if (nodeType.startsWith('preset:')) {
      const presetName = nodeType.slice('preset:'.length);
      const preset = presetMap.get(presetName);
      const internalParams = raw.data?.internalParams ?? {};
      const definition: import('../types').NodeDefinition = preset
        ? {
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
          }
        : { node_name: presetName, category: 'Preset', description: '', inputs: [], outputs: [], params: [] };
      return {
        id: raw.id,
        type: 'presetNode',
        position,
        data: {
          label: presetName,
          type: nodeType,
          params,
          definition,
          isPreset: true,
          presetDefinition: preset,
          internalParams,
          executionStatus: 'idle' as const,
        },
      };
    }

    // Start node
    if (nodeType === 'Start') {
      return {
        id: raw.id,
        type: 'start',
        position,
        data: {
          label: 'Start',
          type: 'Start',
          params,
          definition: defMap.get('Start') ?? { node_name: 'Start', category: 'Control', description: '', inputs: [], outputs: [{ name: 'trigger', data_type: 'TRIGGER', description: '', optional: false }], params: [] },
          executionStatus: 'idle' as const,
        },
      };
    }

    // Regular node. resolveNodeComponentType handles namespaced plugin types
    // ("foundations:Edu-KNN") — first-party viz, else the plugin bridge, else base.
    const def = defMap.get(nodeType);
    return {
      id: raw.id,
      type: resolveNodeComponentType(nodeType),
      position,
      data: {
        label: raw.data?.label ?? nodeType,
        type: nodeType,
        params,
        definition: def ?? { node_name: nodeType, category: 'Utility', description: '', inputs: [], outputs: [], params: [] },
        executionStatus: 'idle' as const,
      },
    };
  });
}

export function resolveSerializedEdges(rawEdges: any[]): import('@xyflow/react').Edge[] {
  return rawEdges.map((e) => {
    const isTrigger = e.type === 'trigger' || e.sourceHandle === 'trigger';
    return {
      id: e.id ?? generateId(),
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle || undefined,
      targetHandle: isTrigger ? '__trigger' : (e.targetHandle || undefined),
      animated: false,
      ...(isTrigger
        ? { type: 'triggerEdge', data: { type: 'trigger' } }
        : { style: { stroke: '#555', strokeWidth: 2 } }),
    };
  });
}

export function buildFlowNode(
  definition: NodeDefinition,
  position: { x: number; y: number },
): Node<NodeData> {
  const defaultParams: Record<string, unknown> = {};
  for (const p of definition.params) {
    defaultParams[p.name] = p.default;
  }
  const name = definition.node_name;
  return {
    id: generateId(),
    type: name === 'Start' ? 'start' : resolveNodeComponentType(name),
    position,
    data: {
      label: name,
      type: name,
      params: defaultParams,
      definition,
      executionStatus: 'idle',
    },
  };
}

export function isValidConnection(sourceType: string, targetType: string): boolean {
  // Trigger type uses a dedicated __trigger handle, not regular data ports
  if (sourceType === 'TRIGGER' || targetType === 'TRIGGER') return false;
  if (sourceType === 'ANY' || targetType === 'ANY') return true;
  if (sourceType === targetType) return true;

  const compatibilityMap: Record<string, string[]> = {
    TENSOR: ['TENSOR', 'ANY'],
    MODEL: ['MODEL', 'ANY'],
    DATASET: ['DATASET', 'DATALOADER', 'ANY'],
    DATALOADER: ['DATALOADER', 'ANY'],
    OPTIMIZER: ['OPTIMIZER', 'ANY'],
    LOSS_FN: ['LOSS_FN', 'ANY'],
    SCALAR: ['SCALAR', 'ANY'],
    STRING: ['STRING', 'ANY'],
    IMAGE: ['IMAGE', 'TENSOR', 'ANY'],
  };

  const compatible = compatibilityMap[sourceType.toUpperCase()];
  return compatible ? compatible.includes(targetType.toUpperCase()) : false;
}
