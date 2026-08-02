/**
 * Frozen snapshot of the apiVersion 2 contract, and a compile-time proof that
 * the host still satisfies it.
 *
 * apiVersion 3 (#132) was specified as purely additive, and "purely additive"
 * is a claim that decays the moment somebody tightens a type or renames a
 * field two releases from now. A v2 plugin — Graph Copilot among them —
 * vendored its own copy of the contract below and declares
 * `activate(api: CodefyUIPluginAPI)` against it. That plugin keeps working
 * exactly as long as the host's real API object stays ASSIGNABLE to the v2
 * interface, which is what this file asserts.
 *
 * The body below is `contract.ts` as it stood at apiVersion 2, verbatim. It is
 * a historical record: DO NOT edit it to make a change compile. If a change
 * makes this file fail, that change is breaking for every installed v2 plugin
 * and needs an apiVersion bump and a migration note, not a patched snapshot.
 *
 * (`api.apiVersion` is `number` here, exactly as v2 published it, so a plugin
 * could defensively feature-check. The host narrowing it to a literal `3` is
 * assignable to `number` and therefore not a break.)
 */
import type { CodefyUIPluginAPI as HostApi } from './api';

/** Compile error unless the argument resolves to exactly `true`. */
type Expect<T extends true> = T;
type Extends<A, B> = A extends B ? true : false;

/* ── BEGIN frozen apiVersion 2 contract ─────────────────────────────────── */

type ToastType = 'success' | 'error' | 'info' | 'warning';

interface PortDefinition {
  name: string;
  data_type: string;
  description: string;
  optional: boolean;
}

interface ParamDefinition {
  name: string;
  param_type:
    | 'int' | 'float' | 'string' | 'bool'
    | 'select' | 'model_file' | 'image_file' | 'tensor_grid' | 'secret'
    | 'code';
  default: unknown;
  description: string;
  options: string[];
  min_value: number | null;
  max_value: number | null;
  visible_when?: Record<string, unknown> | null;
}

interface NodeDefinition {
  node_name: string;
  category: string;
  description: string;
  inputs: PortDefinition[];
  outputs: PortDefinition[];
  params: ParamDefinition[];
}

type GraphOp =
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

interface OpResult {
  index: number;
  ok: boolean;
  error?: string;
  node_id?: string;
}

interface ApplyResult {
  results: OpResult[];
  refs: Record<string, string>;
  node_count: number;
  edge_count: number;
}

interface SerializedGraphNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: { params: Record<string, unknown>; [key: string]: unknown };
}

interface SerializedGraphEdge {
  id: string;
  source: string;
  target: string;
  sourceHandle: string;
  targetHandle: string;
}

interface SerializedGraph {
  nodes: SerializedGraphNode[];
  edges: SerializedGraphEdge[];
  presets?: unknown[];
  segmentGroups?: unknown[];
  name?: string;
  description?: string;
  [key: string]: unknown;
}

interface NodeRenderContext {
  node: {
    id: string;
    type: string;
    params: Record<string, unknown>;
    definition?: unknown;
  };
}

interface PluginNodeRenderer {
  mount(container: HTMLElement, ctx: NodeRenderContext): void;
  update?(container: HTMLElement, ctx: NodeRenderContext): void;
  unmount?(container: HTMLElement): void;
}

interface CodefyUIPluginAPI {
  apiVersion: number;
  pluginId: string;
  ui: {
    addFloatingWidget(opts: { id: string }): HTMLElement;
    toast(message: string, type?: ToastType): void;
  };
  graph: {
    getGraph(): SerializedGraph;
    getNodeDefinitions(): NodeDefinition[];
    applyOperations(ops: GraphOp[]): ApplyResult;
    onGraphChanged(cb: () => void): () => void;
  };
  nodes: {
    registerRenderer(nodeType: string, renderer: PluginNodeRenderer): () => void;
  };
  http: {
    fetch(url: string, init?: RequestInit): Promise<Response>;
  };
  storage: {
    get(key: string): string | null;
    set(key: string, value: string): void;
    remove(key: string): void;
  };
}

type ActivateFn = (api: CodefyUIPluginAPI) => void;

/* ── END frozen apiVersion 2 contract ───────────────────────────────────── */

/**
 * The host's real API object still satisfies the v2 interface — every member
 * a v2 plugin can reach is present, with a compatible type.
 */
export type _V2SurfaceIntact = Expect<Extends<HostApi, CodefyUIPluginAPI>>;

/**
 * ...so a v2 plugin's exported `activate` is still callable with what the
 * host hands it, which is the thing that actually has to keep working.
 *
 * (Parameters are contravariant, so this direction is the one that models the
 * real call `mod.default(buildPluginAPI(...))` — a v2 `ActivateFn` used where
 * the host expects `(api: HostApi) => void`.)
 */
export type _V2ActivateStillCallable = Expect<
  Extends<ActivateFn, (api: HostApi) => void>
>;
