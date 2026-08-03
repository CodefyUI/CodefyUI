/**
 * Registry of plugin-provided dock panels (`api.ui.addPanel`, apiVersion 3).
 *
 * ── The one design decision worth reading ────────────────────────────────
 *
 * The published contract says a panel's element "persists across tab
 * switches". The host's dock does the opposite: `ResultsPanel` renders its
 * tab bodies with `{panelTab === 'x' && ...}`, so switching tabs UNMOUNTS the
 * previous body outright, and the Node Detail Modal (#127/#129) does the same.
 * That is not an accident to work around — it is the mounting discipline the
 * perf work leans on, and a plugin panel that forced the dock to keep every
 * tab alive would quietly undo it for everyone.
 *
 * Both hold at once because the ELEMENT and its PLACEMENT are separated:
 *
 *   - The element is created here, at `addPanel` time, and owned by this
 *     module — outside React entirely. Its identity never changes, so a
 *     plugin can `createRoot(el)` once and keep rendering into it forever.
 *   - The dock mounts a thin adapter for the ACTIVE plugin tab only, which
 *     `appendChild`s that element on mount and `remove()`s it on unmount.
 *     Removing detaches; it never destroys. Nothing of an inactive plugin
 *     panel is mounted, rendered, or laid out.
 *
 * So the dock keeps exactly one panel body mounted, as before, while the
 * plugin keeps a stable element across switches.
 *
 * A detached element is still live JavaScript, though, and a plugin
 * re-rendering into one every frame would burn CPU for pixels that do not
 * exist. `onShow`/`onHide` (fired by the adapter, isolated per callback) are
 * how a plugin cooperates with the same optimisation the host observes.
 */

export type PluginPanelDock = 'bottom' | 'right';

/** What a plugin passes to `api.ui.addPanel`. */
export interface PluginPanelOptions {
  /** Unique within the plugin. Namespaced with the plugin id by the host. */
  id: string;
  /** Tab label (bottom dock) or section heading (right dock). */
  title: string;
  /** Optional short glyph shown before the title. */
  icon?: string;
  /** Where the panel lives. Defaults to `'bottom'`. */
  dock?: PluginPanelDock;
  /** Called after the element is attached to the document. */
  onShow?: () => void;
  /** Called after the element is detached. */
  onHide?: () => void;
}

/** A registered panel, as the host chrome sees it. */
export interface PluginPanel {
  /** Host-wide id: `${pluginId}:${options.id}`. */
  key: string;
  pluginId: string;
  /** The plugin's own id for this panel, unprefixed. */
  localId: string;
  title: string;
  icon?: string;
  dock: PluginPanelDock;
  /** The host-owned container. Stable for the lifetime of the panel. */
  element: HTMLElement;
  onShow?: () => void;
  onHide?: () => void;
}

const panels = new Map<string, PluginPanel>();
const listeners = new Set<() => void>();

/**
 * Insertion-ordered snapshot, rebuilt only when the registry changes.
 *
 * `useSyncExternalStore` compares snapshots by identity and loops forever if
 * `getSnapshot` allocates a fresh array every call, so the array is cached.
 */
let snapshot: PluginPanel[] = [];

function panelKey(pluginId: string, localId: string): string {
  return `${pluginId}:${localId}`;
}

function notify(): void {
  snapshot = Array.from(panels.values());
  listeners.forEach((l) => l());
}

/**
 * Register (or update) a panel and return its stable container element.
 *
 * Re-registering the same id returns the SAME element and updates the
 * presentation — matching `addFloatingWidget`, and keeping a plugin that
 * calls `addPanel` twice from silently losing whatever it rendered.
 */
export function registerPluginPanel(
  pluginId: string,
  opts: PluginPanelOptions,
): HTMLElement {
  const key = panelKey(pluginId, opts.id);
  const existing = panels.get(key);
  const element = existing?.element ?? document.createElement('div');
  if (!existing) {
    element.id = `plugin-panel-${key}`;
    element.dataset.pluginId = pluginId;
  }
  panels.set(key, {
    key,
    pluginId,
    localId: opts.id,
    title: opts.title,
    icon: opts.icon,
    dock: opts.dock ?? 'bottom',
    element,
    onShow: opts.onShow,
    onHide: opts.onHide,
  });
  notify();
  return element;
}

/**
 * Drop one panel.
 *
 * The element is detached from wherever the host had it so an unloading
 * plugin cannot leave DOM behind, but it is NOT emptied: the plugin's own
 * teardown (a React root's `unmount()`, say) still needs its children.
 */
export function removePluginPanel(pluginId: string, localId: string): void {
  const key = panelKey(pluginId, localId);
  const panel = panels.get(key);
  if (!panel) return;
  panels.delete(key);
  panel.element.remove();
  notify();
}

/** Drop every panel a plugin registered. Used by the host's teardown. */
export function removePluginPanelsFor(pluginId: string): void {
  let changed = false;
  for (const panel of Array.from(panels.values())) {
    if (panel.pluginId !== pluginId) continue;
    panels.delete(panel.key);
    panel.element.remove();
    changed = true;
  }
  if (changed) notify();
}

export function getPluginPanel(key: string): PluginPanel | undefined {
  return panels.get(key);
}

/** Every registered panel, in registration order. Stable between changes. */
export function getPluginPanels(): PluginPanel[] {
  return snapshot;
}

export function subscribePluginPanels(cb: () => void): () => void {
  listeners.add(cb);
  return () => { listeners.delete(cb); };
}

/**
 * Tell a panel it became visible, or stopped being.
 *
 * A plugin callback that throws is contained here: the host is mid-mount or
 * mid-unmount when it runs, and an exception escaping would take the dock
 * down with it.
 */
export function notifyPanelVisibility(panel: PluginPanel, visible: boolean): void {
  const cb = visible ? panel.onShow : panel.onHide;
  if (!cb) return;
  try {
    cb();
  } catch (err) {
    console.warn(
      `[plugins] panel '${panel.key}' ${visible ? 'onShow' : 'onHide'} failed:`,
      err,
    );
  }
}

/** Test helper — drop all panels. */
export function _clearPluginPanels(): void {
  for (const panel of panels.values()) panel.element.remove();
  panels.clear();
  notify();
}
