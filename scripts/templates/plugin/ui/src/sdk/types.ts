// CodefyUI plugin SDK — type contract (mirrors the host's plugin API).
// Generated from CodefyUI's frontend/src/plugins/contract.ts — do not edit
// by hand; refresh it when you target a newer CodefyUI release.

export type ToastType = 'success' | 'error' | 'info' | 'warning';

export interface PortDefinition {
  name: string;
  data_type: string;
  description: string;
  optional: boolean;
}

export interface ParamDefinition {
  name: string;
  param_type:
    | 'int' | 'float' | 'string' | 'bool'
    | 'select' | 'model_file' | 'image_file' | 'tensor_grid';
  default: unknown;
  description: string;
  options: string[];
  min_value: number | null;
  max_value: number | null;
  visible_when?: Record<string, unknown> | null;
}

export interface NodeDefinition {
  node_name: string;
  category: string;
  description: string;
  inputs: PortDefinition[];
  outputs: PortDefinition[];
  params: ParamDefinition[];
}

/** A batch operation accepted by `api.graph.applyOperations`. Field names are exact. */
export type GraphOp =
  | { op: 'add_node'; node_type: string; ref?: string;
      params?: Record<string, unknown>; position?: { x: number; y: number } }
  | { op: 'connect'; source: string; source_handle: string;
      target: string; target_handle: string }
  | { op: 'set_params'; node_id: string; params: Record<string, unknown> }
  | { op: 'remove_node'; node_id: string }
  | { op: 'remove_edge'; source: string; target: string;
      source_handle?: string; target_handle?: string }
  | { op: 'clear_graph' }
  | { op: 'auto_layout' };

export interface OpResult {
  index: number;
  ok: boolean;
  error?: string;
  node_id?: string;
}

export interface ApplyResult {
  results: OpResult[];
  refs: Record<string, string>;
  node_count: number;
  edge_count: number;
}

export interface SerializedGraphNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: { params: Record<string, unknown>; [key: string]: unknown };
}

export interface SerializedGraphEdge {
  id: string;
  source: string;
  target: string;
  sourceHandle: string;
  targetHandle: string;
}

export interface SerializedGraph {
  nodes: SerializedGraphNode[];
  edges: SerializedGraphEdge[];
  presets?: unknown[];
  segmentGroups?: unknown[];
  name?: string;
  description?: string;
  [key: string]: unknown;
}

export interface NodeRenderContext {
  node: {
    id: string;
    type: string;
    params: Record<string, unknown>;
    definition?: unknown;
  };
}

/**
 * Imperative renderer for a node's card body. The SDK's `defineNodeRenderer`
 * adapts a React component to this shape; the host calls mount once, update on
 * param changes, and unmount when the node is removed.
 */
export interface PluginNodeRenderer {
  mount(container: HTMLElement, ctx: NodeRenderContext): void;
  update?(container: HTMLElement, ctx: NodeRenderContext): void;
  unmount?(container: HTMLElement): void;
}

/** The object the editor hands every plugin frontend at activation. */
export interface CodefyUIPluginAPI {
  apiVersion: number;
  pluginId: string;
  ui: {
    /** Create (or reuse) a container `<div>` in the editor's widget stack. */
    addFloatingWidget(opts: { id: string }): HTMLElement;
    toast(message: string, type?: ToastType): void;
  };
  graph: {
    getGraph(): SerializedGraph;
    getNodeDefinitions(): NodeDefinition[];
    /** Synchronous — returns the result directly, committed as one undo step. */
    applyOperations(ops: GraphOp[]): ApplyResult;
    onGraphChanged(cb: () => void): () => void;
  };
  /** Custom node renderers — requires apiVersion >= 2. */
  nodes: {
    registerRenderer(nodeType: string, renderer: PluginNodeRenderer): () => void;
  };
  http: {
    /** Browser `fetch`, with the CodefyUI session token attached. */
    fetch(url: string, init?: RequestInit): Promise<Response>;
  };
  storage: {
    get(key: string): string | null;
    set(key: string, value: string): void;
    remove(key: string): void;
  };
}

/** The default-export contract: the editor calls this once with the API. */
export type ActivateFn = (api: CodefyUIPluginAPI) => void;
