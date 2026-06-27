/**
 * React bindings for the CodefyUI plugin API.
 *
 * `defineTool` turns a React component into the editor's `activate(api)` default
 * export — mounting it into a floating widget and providing the API to the whole
 * subtree via context, so the hooks below work anywhere inside your component.
 *
 * React is bundled with your plugin (peer of your `ui/`), so your components run
 * on their own React root inside the editor.
 */
import React from 'react';
import { createRoot, type Root } from 'react-dom/client';
import type {
  ActivateFn, ApplyResult, CodefyUIPluginAPI, GraphOp,
  NodeDefinition, NodeRenderContext, PluginNodeRenderer,
  SerializedGraph, ToastType,
} from './types';

const ApiContext = React.createContext<CodefyUIPluginAPI | null>(null);

/** The plugin API, from the nearest `defineTool` provider. Throws if absent. */
export function useCodefyUI(): CodefyUIPluginAPI {
  const api = React.useContext(ApiContext);
  if (!api) {
    throw new Error(
      'CodefyUI hooks must be used inside a component mounted by defineTool().',
    );
  }
  return api;
}

export interface ToolOptions {
  /** Unique-per-plugin widget id. */
  id: string;
  /** Accessible label for the widget container. */
  title?: string;
}

/**
 * Wrap a React component as a plugin entry point. Use it as your default export:
 *
 * ```tsx
 * export default defineTool({ id: 'my-panel', title: 'My Panel' }, MyPanel);
 * ```
 */
export function mountTool(
  api: CodefyUIPluginAPI,
  opts: ToolOptions,
  Component: React.ComponentType<{ api: CodefyUIPluginAPI }>,
): void {
  const el = api.ui.addFloatingWidget({ id: opts.id });
  if (opts.title) el.setAttribute('aria-label', opts.title);
  createRoot(el).render(
    <React.StrictMode>
      <ApiContext.Provider value={api}>
        <Component api={api} />
      </ApiContext.Provider>
    </React.StrictMode>,
  );
}

export function defineTool(
  opts: ToolOptions,
  Component: React.ComponentType<{ api: CodefyUIPluginAPI }>,
): ActivateFn {
  return (api) => mountTool(api, opts, Component);
}

/**
 * Adapt a React component into a node-body renderer for
 * `api.nodes.registerRenderer`. The component receives the live node context
 * (id, type, params) and re-renders whenever the node's params change.
 *
 * ```tsx
 * api.nodes.registerRenderer('my_plugin:MyNode', defineNodeRenderer(MyBody));
 * ```
 */
export function defineNodeRenderer(
  Component: React.ComponentType<{ node: NodeRenderContext['node'] }>,
): PluginNodeRenderer {
  const roots = new WeakMap<HTMLElement, Root>();
  const render = (el: HTMLElement, ctx: NodeRenderContext) => {
    let root = roots.get(el);
    if (!root) {
      root = createRoot(el);
      roots.set(el, root);
    }
    root.render(<Component node={ctx.node} />);
  };
  return {
    mount: render,
    update: render,
    unmount: (el) => {
      const root = roots.get(el);
      if (root) {
        root.unmount();
        roots.delete(el);
      }
    },
  };
}

/** Live serialized graph; the component re-renders whenever the graph changes. */
export function useGraph(): SerializedGraph {
  const api = useCodefyUI();
  const [graph, setGraph] = React.useState<SerializedGraph>(() => api.graph.getGraph());
  React.useEffect(
    () => api.graph.onGraphChanged(() => setGraph(api.graph.getGraph())),
    [api],
  );
  return graph;
}

/** The node palette (types, ports, params). Stable for the session. */
export function useNodeDefinitions(): NodeDefinition[] {
  const api = useCodefyUI();
  return React.useMemo(() => api.graph.getNodeDefinitions(), [api]);
}

/** Run your own callback whenever the graph changes. */
export function useGraphChanged(cb: () => void): void {
  const api = useCodefyUI();
  const ref = React.useRef(cb);
  ref.current = cb;
  React.useEffect(() => api.graph.onGraphChanged(() => ref.current()), [api]);
}

/** A stable function to apply a batch of graph operations. */
export function useApplyOperations(): (ops: GraphOp[]) => ApplyResult {
  const api = useCodefyUI();
  return React.useCallback((ops) => api.graph.applyOperations(ops), [api]);
}

/** A stable toast notifier. */
export function useToast(): (message: string, type?: ToastType) => void {
  const api = useCodefyUI();
  return React.useCallback((message, type) => api.ui.toast(message, type), [api]);
}

/** Session-aware fetch bound to the CodefyUI backend (relative paths only). */
export function useCodefyFetch(): (url: string, init?: RequestInit) => Promise<Response> {
  const api = useCodefyUI();
  return React.useCallback((url, init) => api.http.fetch(url, init), [api]);
}

/** Persistent, plugin-namespaced string state — like useState backed by storage. */
export function useStorage(key: string, initial = ''): [string, (v: string) => void] {
  const api = useCodefyUI();
  const [value, setValue] = React.useState<string>(() => api.storage.get(key) ?? initial);
  const set = React.useCallback((v: string) => {
    api.storage.set(key, v);
    setValue(v);
  }, [api, key]);
  return [value, set];
}
