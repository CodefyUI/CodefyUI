import type { Node as FlowNode } from '@xyflow/react';

export interface PortDefinition {
  name: string;
  data_type: string;
  description: string;
  optional: boolean;
}

export interface ParamDefinition {
  name: string;
  param_type:
    | 'int' | 'float' | 'string' | 'bool' | 'select'
    | 'model_file' | 'image_file' | 'tensor_grid' | 'secret'
    /** Tabular / plain-data upload (CSV, TSV, TXT, JSON); same widget as image_file. */
    | 'data_file'
    /** Multi-line Python source; rendered with a code editor (core#131). */
    | 'code';
  default: any;
  description: string;
  options: string[];
  min_value: number | null;
  max_value: number | null;
  /**
   * Conditional visibility — the param renders only when every key in
   * this dict matches the live value of the corresponding sibling param.
   * Example: ``{ preset: "Custom" }`` on Conv2dExplicit's ``weights`` means
   * the editor only shows when ``preset === "Custom"``. ``null`` /
   * undefined means always visible.
   *
   * A value may be an ARRAY, read as "any of these" (core#134):
   * ``{ type: ["Adam", "AdamW"] }`` on Optimizer's ``amsgrad``.
   */
  visible_when?: Record<string, unknown> | null;
  /**
   * Two-tier parameter UI (core#134). An advanced param is fully real —
   * persisted, exported, identical in every way — it is simply collected
   * behind a collapsed "Advanced" section so the default view of a node
   * stays readable. Absent means basic, which is what a plugin built
   * against an older host sends.
   */
  advanced?: boolean;
}

export interface SegmentGroup {
  id: string;
  headNodeId: string;
  tailNodeId: string;
}

/**
 * One boundary port of a subgraph (core#137).
 *
 * `port` is the handle the INSTANCE node shows on the canvas; `innerNode` /
 * `innerPort` say which node inside the definition it stands for. The port
 * name is only a label -- validation and execution type-check the inner port.
 */
export interface SubgraphPort {
  port: string;
  innerNode: string;
  innerPort: string;
  data_type: string;
}

export interface SubgraphInterface {
  inputs: SubgraphPort[];
  outputs: SubgraphPort[];
  /**
   * Inner nodes an incoming trigger edge fans out to. Collapse records
   * exactly the inner nodes that were triggered before it ran, so the
   * expanded graph keeps the same entry points -- which is what makes
   * collapse and expand produce the same run.
   */
  triggerTargets: string[];
}

/**
 * A reusable block of graph, local to one graph file (v1).
 *
 * Instances reference it by id and carry NO overrides: two instances of one
 * definition are identical blocks, and editing the definition changes both.
 * Per-instance parameter overrides are deliberately out of scope for v1.
 */
export interface SubgraphDefinition {
  id: string;
  name: string;
  description: string;
  /** Serialized (not React Flow) nodes -- the same shape a graph file uses. */
  nodes: any[];
  edges: any[];
  interface: SubgraphInterface;
}

export interface NodeDefinition {
  node_name: string;
  category: string;
  description: string;
  inputs: PortDefinition[];
  outputs: PortDefinition[];
  params: ParamDefinition[];
}

export interface PresetDefinition {
  preset_name: string;
  category: string;
  description: string;
  tags: string[];
  nodes: { id: string; type: string; params: Record<string, any> }[];
  edges: { source: string; sourceHandle: string; target: string; targetHandle: string }[];
  exposed_inputs: { name: string; internal_node: string; internal_port: string; data_type: string; description: string }[];
  exposed_outputs: { name: string; internal_node: string; internal_port: string; data_type: string; description: string }[];
  exposed_params: { internal_node: string; param_name: string; display_name: string; group: string; param_def: ParamDefinition | null }[];
}

export interface NodeProgress {
  event: string;
  epoch?: number;
  total_epochs?: number;
  loss?: number;
  losses?: number[];
  [key: string]: unknown;
}

export interface OutputSummary {
  type: string;
  shape?: number[];
  dtype?: string;
  min?: number;
  max?: number;
  mean?: number;
  value?: any;
  class?: string;
  params?: number;
  trainable?: number;
  repr?: string;
  /** For LIST outputs: total length (regardless of whether values are inlined). */
  length?: number;
  /**
   * For short LIST outputs (≤256 primitive items, or 2-tuples of numbers),
   * the actual values are embedded so per-node visualizations can render
   * without a separate REST fetch.
   */
  values?: unknown[];
  // Set by the backend when the string value is a file under MODELS_DIR;
  // holds the path relative to MODELS_DIR so the frontend can download it.
  download_path?: string;
}

export interface NodeData {
  label: string;
  type: string;
  params: Record<string, any>;
  definition?: NodeDefinition;
  executionStatus?: ExecutionStatus;
  error?: string;
  progress?: NodeProgress;
  isPreset?: boolean;
  presetDefinition?: PresetDefinition;
  internalParams?: Record<string, Record<string, any>>;
  /**
   * ComfyUI-style bypass (core#128). A bypassed node is not executed: the
   * engine removes it and forwards, per output port, the value that arrived
   * on the first type-compatible input. Serialized with the graph, so a saved
   * file remembers what was muted.
   */
  bypassed?: boolean;
  // Note-specific fields (present only when node.type === 'noteNode')
  noteKind?: 'text' | 'image';
  noteContent?: string;
  noteColor?: string;
  boundToNodeId?: string | null;
  boundOffset?: { x: number; y: number } | null;
  noteWidth?: number;
  noteHeight?: number;
  [key: string]: unknown;
}

// 'interrupted' (core#122): the node polled context.should_stop() inside its
// own loop, saved what it had and returned PARTIAL outputs. Distinct from
// 'error' (it did not fail) and from 'completed' (it did not finish), and
// distinct from a node that never ran at all — which is what it would look
// like without an entry here.
export type ExecutionStatus =
  | 'idle'
  | 'running'
  | 'completed'
  | 'error'
  | 'skipped'
  | 'cached'
  | 'interrupted';

// @xyflow/react v12 expects the generic to be a full Node type (not the data
// payload). Use this alias wherever a component types its props or a store
// holds nodes. The empty string fallback keeps the `type` slot string-assignable.
export type AppNode = FlowNode<NodeData, string | undefined>;

export interface GraphSaveData {
  nodes: any[];
  edges: any[];
  name: string;
  description: string;
  presets?: PresetDefinition[];
  segmentGroups?: SegmentGroup[];
  subgraphs?: SubgraphDefinition[];
}

// Teaching Inspector: full-value responses from /api/execution/outputs
export interface TensorOutput {
  type: 'tensor';
  run_id: string;
  node_id: string;
  port: string;
  full_shape: number[];
  dtype: string;
  slice: string;
  sliced_shape: number[];
  values: unknown;
  truncated: boolean;
  min?: number;
  max?: number;
  mean?: number;
}

export interface ModelOutput {
  type: 'model';
  run_id: string;
  node_id: string;
  port: string;
  class: string;
  params: number;
  trainable: number;
  repr: string;
}

export interface ScalarOutput {
  type: 'scalar';
  run_id: string;
  node_id: string;
  port: string;
  value: number | boolean;
}

export interface StringOutput {
  type: 'string';
  run_id: string;
  node_id: string;
  port: string;
  value: string;
}

export interface ListOutput {
  type: 'list';
  run_id: string;
  node_id: string;
  port: string;
  length: number;
  /** Populated when the list is small and primitive (≤1024 strings/numbers/bools or 2-tuples of numbers). */
  values?: unknown[];
  repr?: string;
}

export interface GenericOutput {
  type: string;
  run_id: string;
  node_id: string;
  port: string;
  repr?: string;
  length?: number;
}

export type OutputData =
  | TensorOutput
  | ModelOutput
  | ScalarOutput
  | StringOutput
  | ListOutput
  | GenericOutput;

export interface RunOutputRef {
  node_id: string;
  port: string;
  type: string;
  full_shape: number[] | null;
}
