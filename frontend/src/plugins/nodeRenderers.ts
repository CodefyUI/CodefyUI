/**
 * Registry of plugin-provided custom node renderers.
 *
 * A plugin calls `api.nodes.registerRenderer(nodeType, renderer)` inside its
 * `activate()`. `PluginNodeBridge` looks the renderer up by the node's
 * (namespaced) type and mounts it into the node card's body slot. The registry
 * is observable (subscribe + snapshot) so a renderer registered *after* nodes
 * are already on the canvas still appears, and re-registering the same type
 * overwrites — so a dev hot-reload doesn't accumulate stale renderers.
 *
 * The renderer is framework-agnostic (imperative mount/update/unmount), so the
 * host stays decoupled from whichever React the plugin bundles. The SDK's
 * `defineNodeRenderer` adapts a React component to this shape.
 */
export interface PluginNodeContext {
  node: {
    id: string;
    type: string;
    params: Record<string, unknown>;
    definition?: unknown;
  };
}

export interface PluginNodeRenderer {
  mount(container: HTMLElement, ctx: PluginNodeContext): void;
  update?(container: HTMLElement, ctx: PluginNodeContext): void;
  unmount?(container: HTMLElement): void;
}

const renderers = new Map<string, PluginNodeRenderer>();
const listeners = new Set<() => void>();

function notify(): void {
  listeners.forEach((l) => l());
}

/** Register a renderer for a (namespaced) node type. Returns an unregister fn. */
export function registerNodeRenderer(
  nodeType: string,
  renderer: PluginNodeRenderer,
): () => void {
  renderers.set(nodeType, renderer);
  notify();
  return () => {
    if (renderers.get(nodeType) === renderer) {
      renderers.delete(nodeType);
      notify();
    }
  };
}

export function getNodeRenderer(nodeType: string): PluginNodeRenderer | undefined {
  return renderers.get(nodeType);
}

export function subscribeNodeRenderers(cb: () => void): () => void {
  listeners.add(cb);
  return () => { listeners.delete(cb); };
}

/** Test helper — drop all registered renderers. */
export function _clearNodeRenderers(): void {
  renderers.clear();
  notify();
}
