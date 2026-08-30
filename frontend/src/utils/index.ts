import type { Node } from '@xyflow/react';
import type { NodeData, NodeDefinition } from '../types';
import {
  SUBGRAPH_TYPE_PREFIX,
  instanceDefinition,
  normalizeSubgraphs,
} from './subgraph';
import { generateId } from './ids';

export { generateId } from './ids';

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
  // NTFS and APFS are case-INSENSITIVE, so "My_Graph.json" and
  // "my_graph.json" are the same file on Windows/macOS even though the
  // backend's _sanitize_name preserves case. Compare lowercased stems so a
  // case-only collision still triggers the overwrite warning. A spurious
  // warning on case-sensitive Linux (where the two really are distinct files)
  // is far safer than a silent overwrite on the majority platform.
  const target = sanitizeGraphName(targetName).toLowerCase();
  const current = currentFile == null ? null : currentFile.toLowerCase();
  const hit = existing.find(
    (g) => g.file.toLowerCase() === target && g.file.toLowerCase() !== current,
  );
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

/**
 * xyflow COMPONENT types that cannot be bypassed (core#128).
 *
 * `noteNode` is an annotation with no execution to skip; `start` is the
 * trigger marker that defines an entry point (muting it would silently strip
 * the graph's only entry); `presetNode` expands into a sub-graph whose ports
 * come from the preset definition, so there is nothing for the engine's
 * pass-through rule to match on — the backend refuses it explicitly.
 */
const NON_BYPASSABLE_NODE_TYPES = new Set(['noteNode', 'start', 'presetNode']);

/**
 * GRAPH node types that cannot be bypassed, checked against `data.type`.
 *
 * These two render as ordinary `baseNode` cards, so the component-type set
 * above cannot see them. They declare the graph's I/O contract, which
 * `derive_contract` / `check_wiring` read from the RAW graph — muting one
 * would leave a published app advertising an input or output the run cannot
 * honour. The backend refuses them for the same reason; this keeps the
 * canvas from offering an action that can only end in a validation error.
 */
const NON_BYPASSABLE_GRAPH_TYPES = new Set(['GraphInput', 'GraphOutput']);

/** Whether the bypass toggle applies to this canvas node. */
export function isBypassable(
  node: { type?: string; data?: { type?: string } } | undefined | null,
): boolean {
  if (!node) return false;
  if (NON_BYPASSABLE_NODE_TYPES.has(node.type ?? '')) return false;
  return !NON_BYPASSABLE_GRAPH_TYPES.has(node.data?.type ?? '');
}

/**
 * Wire / port colour per data type.
 *
 * KEY ORDER MIRRORS the backend `DataType` enum in
 * `backend/app/core/node_base.py` (#197 item 4) — minus TRIGGER, which is
 * control flow and never a data port, so the canvas has no colour for it.
 * The order is not decoration: `SELECTABLE_DATA_TYPES` below is
 * `Object.keys` of this map and drives the PythonScript per-port type
 * dropdown, so a reader comparing the dropdown with the enum sees the same
 * sequence. Add a type in the position the enum puts it in; `index.test.ts`
 * pins the order against a transcription of the enum.
 */
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
  // Kept in the amber family next to DATASET's orange on purpose — a
  // transform chain is what feeds a dataset, and the two are read together —
  // but LIGHTER than the old '#FFC107' (#197 item 5). Those two ambers meet
  // at every train_transform / eval_transform port, and 14.5 dE00 apart is
  // close for a pair that is always drawn touching. Worse, most of that
  // distance was on the red-green axis: simulate deuteranopia and the old
  // amber fell to 6.1 dE00 from DATASET and 2.5 from LIST — the closest pair
  // in the whole type palette for a dichromat.
  //
  // '#FFE082' is Material Amber 200: the same hue (91 degrees in Lab against
  // the old 83 and DATASET's 68), sitting ~18 L* above DATASET instead of
  // ~9.5. That reads as 21.9 dE00 in normal vision and 12.6 simulated
  // deuteran / 16.6 protan, and TRANSFORM stops being any dichromat's
  // closest pair. Lightness is the axis every viewer keeps, which is why the
  // fix is a lighter amber rather than a different hue. (The palette's own
  // closest pair is unchanged at 8.2 dE00, OPTIMIZER/IMAGE; the contrast
  // gate's floor for this palette is 5.)
  //
  // The light-export twin (`--diagram-light-type-transform`, #b78901) does
  // NOT follow it lighter: that palette is drawn on white and is held to
  // 3:1 there, which any lighter amber fails. It stays the same hue as this
  // one, darkened, exactly as the light palette is meant to be.
  TRANSFORM: '#FFE082',
};

export function getPortColor(dataType: string): string {
  return DATA_TYPE_COLORS[dataType.toUpperCase()] ?? DATA_TYPE_COLORS['ANY'];
}

/**
 * Palette hexes this app has RETIRED, and the data type each one used to
 * stand for (core#325).
 *
 * An interactive connect bakes the type's colour into `edge.style.stroke`,
 * and an autosaved workspace stores the edge object as it stands — so a
 * graph last saved before a palette change keeps drawing that wire in the
 * old colour, next to port dots that are painted live from the new one. One
 * wire, two ambers.
 *
 * This is the record of which baked hexes are ours to correct. Anything not
 * listed here is somebody else's choice and is left alone. Append to it —
 * never edit an entry — when a value in `DATA_TYPE_COLORS` changes: the old
 * value goes in with the type it belonged to.
 *
 * Keys are compared case-insensitively; a graph may have been hand-edited.
 */
export const STALE_TYPE_STROKES: Record<string, string> = {
  // #197 item 5 lightened TRANSFORM to '#FFE082' to part it from DATASET.
  '#FFC107': 'TRANSFORM',
};

/**
 * Evaluate a param's ``visible_when`` rule against the current params on
 * the node. Returns true when the param should render. The default rule
 * (no ``visible_when``) is "always visible".
 *
 * Matching is shallow equality after string coercion — sufficient for
 * SELECT / INT / BOOL / FLOAT params, which cover every realistic use of
 * conditional visibility.
 *
 * An expected value may be an ARRAY, read as "any of these" (core#134).
 * Optimizer's `betas` belongs to four of the nine algorithms, and a
 * single-value rule could only ever name one of them. A scalar keeps its
 * original meaning, so no existing rule changes behaviour.
 */
export function isParamVisible(
  param: import('../types').ParamDefinition,
  params: Record<string, unknown> | undefined,
): boolean {
  const rule = param.visible_when;
  if (!rule) return true;
  const live = params ?? {};
  for (const [siblingName, expected] of Object.entries(rule)) {
    const actual = String(live[siblingName]);
    const accepted = Array.isArray(expected) ? expected : [expected];
    if (!accepted.some((candidate) => String(candidate) === actual)) return false;
  }
  return true;
}

const SPLIT_MAX_CHUNKS = 32;

/** Mirrors `python_script_node.MAX_PORTS`. */
export const SCRIPT_MAX_PORTS = 8;

/** Mirrors `compose_transform_node.MIN_STEPS` / `MAX_STEPS`. */
export const COMPOSE_MIN_STEPS = 2;
export const COMPOSE_MAX_STEPS = 8;

function bareName(qualifiedName: string): string {
  const idx = qualifiedName.lastIndexOf(':');
  return idx >= 0 ? qualifiedName.slice(idx + 1) : qualifiedName;
}

/** Clamp a param into `1..max`, tolerating strings, NaN and undefined. */
function clampCount(raw: unknown, fallback: number, max: number): number {
  const parsed = typeof raw === 'number' ? raw : parseInt(String(raw ?? ''), 10);
  return Math.max(1, Math.min(max, Number.isFinite(parsed) ? Math.floor(parsed) : fallback));
}

/**
 * Per-port data types from a comma-separated param, repeating the last entry.
 *
 * Mirrors `python_script_node.resolve_port_types`: a short list must not
 * leave newly-added ports untyped, and an unknown name falls back to the
 * default rather than colouring a handle after a type that does not exist.
 */
function resolvePortTypes(raw: unknown, count: number, fallback: string): string[] {
  const names = String(raw ?? '')
    .split(',')
    .map((part) => part.trim().toUpperCase())
    .filter(Boolean);
  return Array.from({ length: count }, (_, i) => {
    const name = names[i] ?? names[names.length - 1] ?? '';
    // DATA_TYPE_COLORS is the set of types the canvas can draw, and it
    // excludes TRIGGER — which is control flow, never a script's data port.
    return name && DATA_TYPE_COLORS[name] ? name : fallback;
  });
}

/**
 * Data types a per-port select may offer. Everything the canvas can draw,
 * which is every backend `DataType` except TRIGGER.
 */
export const SELECTABLE_DATA_TYPES: string[] = Object.keys(DATA_TYPE_COLORS);

/**
 * Resolve a node's *live* output ports, expanding param-driven nodes whose
 * port count depends on a runtime parameter. Mirrors the backend
 * `BaseNode.define_outputs_dynamic` mechanism so palette template and live
 * canvas agree on what handles exist.
 *
 * For nodes whose ports don't depend on params this returns
 * `definition.outputs` VERBATIM — the same array reference, which callers
 * (see `tabStore.updateNodeParams`) use as a cheap "nothing can have
 * changed" check. New dynamic-port nodes add a clause here, and must keep
 * that identity property for every node they do not handle.
 */
export function resolveDynamicOutputs(
  definition: import('../types').NodeDefinition | undefined,
  params: Record<string, unknown> | undefined,
): import('../types').PortDefinition[] {
  if (!definition) return [];
  const bare = bareName(definition.node_name);
  if (bare === 'Split') {
    const chunks = clampCount(params?.chunks, 2, SPLIT_MAX_CHUNKS);
    return Array.from({ length: chunks }, (_, i) => ({
      name: `chunk_${i}`,
      data_type: 'TENSOR',
      description: `Chunk ${i} of ${chunks}`,
      optional: false,
    }));
  }
  if (bare === 'PythonScript') {
    const count = clampCount(params?.output_ports, 1, SCRIPT_MAX_PORTS);
    const types = resolvePortTypes(params?.output_types, count, 'ANY');
    return Array.from({ length: count }, (_, i) => ({
      name: `out${i + 1}`,
      data_type: types[i],
      description: `return {'out${i + 1}': ...}`,
      optional: false,
    }));
  }
  return definition.outputs;
}

/**
 * Resolve a node's *live* input ports. The mirror of
 * `resolveDynamicOutputs`, for the backend's `define_inputs_dynamic`.
 *
 * PythonScript and ComposeTransform vary their inputs; everything else
 * returns `definition.inputs` verbatim (same reference — see above).
 */
export function resolveDynamicInputs(
  definition: import('../types').NodeDefinition | undefined,
  params: Record<string, unknown> | undefined,
): import('../types').PortDefinition[] {
  if (!definition) return [];
  const bare = bareName(definition.node_name);
  if (bare === 'ComposeTransform') {
    // `clampCount`'s floor is 1; ComposeTransform's is 2, because composing
    // one chain is what a plain edge already does.
    const count = Math.max(
      COMPOSE_MIN_STEPS,
      clampCount(params?.steps, COMPOSE_MIN_STEPS, COMPOSE_MAX_STEPS),
    );
    return Array.from({ length: count }, (_, i) => ({
      name: `step_${i + 1}`,
      data_type: 'TRANSFORM',
      description: `Chain to run at position ${i + 1}`,
      optional: true,
    }));
  }
  if (bare === 'PythonScript') {
    const count = clampCount(params?.input_ports, 1, SCRIPT_MAX_PORTS);
    const types = resolvePortTypes(params?.input_types, count, 'TENSOR');
    return Array.from({ length: count }, (_, i) => ({
      name: `in${i + 1}`,
      data_type: types[i],
      description: `inputs['in${i + 1}']`,
      optional: true,
    }));
  }
  return definition.inputs;
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
  subgraphs: import('../types').SubgraphDefinition[] = [],
): import('@xyflow/react').Node<import('../types').NodeData>[] {
  const defMap = new Map(definitions.map((d) => [d.node_name, d]));
  const presetMap = new Map(presets.map((p) => [p.preset_name, p]));
  // Normalized here as well as in `setSubgraphs`, because every reader of a
  // graph document resolves the nodes BEFORE it installs the definitions --
  // so this runs first, on the list exactly as it came out of the file. An
  // instance node whose definition is `{"id":"x"}` reaches
  // `instanceDefinition` below, which maps `interface.inputs`, and the whole
  // import dies on a file the server would have run.
  const subgraphMap = new Map(
    normalizeSubgraphs(subgraphs).map((sg) => [sg.id, sg]),
  );

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

    // Subgraph instance node (core#137). Its ports come from the shared
    // definition's interface, so an instance whose definition changed shows
    // the new boundary the moment the graph is re-resolved.
    if (nodeType.startsWith(SUBGRAPH_TYPE_PREFIX)) {
      const subgraphId = nodeType.slice(SUBGRAPH_TYPE_PREFIX.length);
      const definition = subgraphMap.get(subgraphId);
      return {
        id: raw.id,
        type: 'subgraphNode',
        position,
        data: {
          label: definition?.name || subgraphId,
          type: nodeType,
          params: {},
          definition: definition
            ? instanceDefinition(definition)
            : {
                node_name: subgraphId,
                category: 'Subgraph',
                description: '',
                inputs: [],
                outputs: [],
                params: [],
              },
          subgraphId,
          executionStatus: 'idle' as const,
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
        // Only set when muted, so a graph with no bypass carries no flag at
        // all — the same asymmetry getSerializedGraph writes with (core#128).
        ...(raw.data?.bypassed ? { bypassed: true } : {}),
      },
    };
  });
}

/** Stroke used when an edge's source port data type cannot be resolved. */
const DEFAULT_EDGE_STROKE = '#555';

/**
 * The data type an edge carries, read from its source node's definition
 * (dynamic outputs included, so Split's chunk_N ports resolve too). Null when
 * the node, its definition's output, or the handle is missing — e.g. a graph
 * referencing plugin nodes that are not loaded yet.
 */
function resolveEdgeDataType(
  rawEdge: any,
  nodeMap: Map<string, Node<NodeData>>,
): string | null {
  const sourceNode = nodeMap.get(rawEdge.source);
  const data = sourceNode?.data;
  if (!data || !rawEdge.sourceHandle) return null;
  const output = resolveDynamicOutputs(data.definition, data.params).find(
    (o) => o.name === rawEdge.sourceHandle,
  );
  return output ? output.data_type : null;
}

/**
 * Look up the per-data-type stroke for a serialized edge. Falls back to the
 * neutral gray whenever {@link resolveEdgeDataType} cannot answer.
 */
function resolveEdgeStroke(
  rawEdge: any,
  nodeMap: Map<string, Node<NodeData>>,
): string {
  const dataType = resolveEdgeDataType(rawEdge, nodeMap);
  return dataType ? getPortColor(dataType) : DEFAULT_EDGE_STROKE;
}

/**
 * Repaint edges whose baked stroke is a retired palette hex (core#325).
 *
 * A saved graph's edges come back with whatever colour was current when the
 * user drew them, so a palette change leaves old wires in the old colour
 * forever. This corrects exactly the wires it can prove are stale: the baked
 * stroke has to be a hex listed in {@link STALE_TYPE_STROKES}, AND the edge
 * has to still derive to the data type that hex belonged to. A wire wearing a
 * colour we never shipped, one whose type has since changed, one whose source
 * node cannot be resolved, and one with no baked stroke at all (every
 * committed example) are all left exactly as they are.
 *
 * Returns the input array unchanged when nothing is stale, which is the
 * common case — the autosave record cache compares edge arrays by reference.
 */
export function migrateStaleEdgeStrokes(
  edges: import('@xyflow/react').Edge[],
  nodes: Node<NodeData>[],
): import('@xyflow/react').Edge[] {
  const stale = edges.some(
    (e) => STALE_TYPE_STROKES[String((e.style as any)?.stroke).toUpperCase()],
  );
  if (!stale) return edges;
  const nodeMap = new Map(nodes.map((n) => [n.id, n]));
  return edges.map((e) => {
    const stroke = (e.style as { stroke?: string } | undefined)?.stroke;
    const wasType = stroke ? STALE_TYPE_STROKES[stroke.toUpperCase()] : undefined;
    if (!wasType) return e;
    const nowType = resolveEdgeDataType(e, nodeMap);
    if (nowType?.toUpperCase() !== wasType) return e;
    return { ...e, style: { ...e.style, stroke: getPortColor(wasType) } };
  });
}

/**
 * Reconstruct ReactFlow edges from the serialized graph format. Pass the
 * already-resolved nodes (from {@link resolveSerializedNodes}) so value edges
 * regain the same per-data-type stroke color that live-created edges get;
 * without them every value edge falls back to the neutral gray. Trigger
 * edges keep their dedicated type/handle and take no inline style.
 */
export function resolveSerializedEdges(
  rawEdges: any[],
  nodes?: Node<NodeData>[],
): import('@xyflow/react').Edge[] {
  const nodeMap = new Map((nodes ?? []).map((n) => [n.id, n]));
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
        : { style: { stroke: resolveEdgeStroke(e, nodeMap), strokeWidth: 2 } }),
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

/**
 * Which target types each source type may feed, keyed by SOURCE.
 *
 * The relation is deliberately directional: `DATASET -> DATALOADER` is a
 * legal wiring, the reverse is not. A source with no row here connects to
 * nothing beyond what `isValidConnection`'s early returns already allow, so
 * every type the palette can offer needs a row — an omission silently
 * downgrades a type to "same-type and ANY only" instead of failing loudly.
 * `SELECTABLE_DATA_TYPES` is the list that has to be covered; the test suite
 * pins that, since the rows whose only entries are the type itself plus ANY
 * are invisible to `isValidConnection`'s behaviour.
 */
export const DATA_TYPE_COMPATIBILITY: Record<string, string[]> = {
  TENSOR: ['TENSOR', 'ANY'],
  MODEL: ['MODEL', 'ANY'],
  DATASET: ['DATASET', 'DATALOADER', 'ANY'],
  DATALOADER: ['DATALOADER', 'ANY'],
  OPTIMIZER: ['OPTIMIZER', 'ANY'],
  LOSS_FN: ['LOSS_FN', 'ANY'],
  SCALAR: ['SCALAR', 'ANY'],
  STRING: ['STRING', 'ANY'],
  IMAGE: ['IMAGE', 'TENSOR', 'ANY'],
  LIST: ['LIST', 'ANY'],
  TRANSFORM: ['TRANSFORM', 'ANY'],
};

export function isValidConnection(sourceType: string, targetType: string): boolean {
  // Trigger type uses a dedicated __trigger handle, not regular data ports
  if (sourceType === 'TRIGGER' || targetType === 'TRIGGER') return false;
  if (sourceType === 'ANY' || targetType === 'ANY') return true;
  if (sourceType === targetType) return true;

  const compatible = DATA_TYPE_COMPATIBILITY[sourceType.toUpperCase()];
  return compatible ? compatible.includes(targetType.toUpperCase()) : false;
}
