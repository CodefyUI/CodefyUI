/**
 * Registry of plugin-provided toolbar buttons (`api.ui.addToolbarButton`,
 * apiVersion 3).
 *
 * Placement is the host's, not the plugin's: buttons land in one group at the
 * right of the toolbar, in registration order, and the host decides how many
 * of them the current window width can show (`PluginToolbarButtons`). There is
 * deliberately no way for a plugin to ask for a position — the toolbar already
 * wraps under pressure, and letting ten plugins each claim a spot next to Run
 * is how a toolbar stops being usable.
 *
 * `onClick` runs inside a try/catch at the call site for the same reason every
 * other plugin callback does: a throwing plugin must not take a host click
 * handler with it.
 */

/** What a plugin passes to `api.ui.addToolbarButton`. */
export interface PluginToolbarButtonOptions {
  /** Unique within the plugin. Namespaced with the plugin id by the host. */
  id: string;
  /** Short glyph or one or two characters — the toolbar has room for a glyph. */
  icon: string;
  /** Hover/accessible text. Required: an icon alone is not a label. */
  tooltip: string;
  onClick: () => void;
}

/** A registered button, as the toolbar sees it. */
export interface PluginToolbarButton {
  /** Host-wide id: `${pluginId}:${options.id}`. */
  key: string;
  pluginId: string;
  localId: string;
  icon: string;
  tooltip: string;
  onClick: () => void;
}

const buttons = new Map<string, PluginToolbarButton>();
const listeners = new Set<() => void>();

/** Cached snapshot — see the note in `panels.ts`. */
let snapshot: PluginToolbarButton[] = [];

function buttonKey(pluginId: string, localId: string): string {
  return `${pluginId}:${localId}`;
}

function notify(): void {
  snapshot = Array.from(buttons.values());
  listeners.forEach((l) => l());
}

/** Register (or replace) a button. Returns a remove function. */
export function registerPluginToolbarButton(
  pluginId: string,
  opts: PluginToolbarButtonOptions,
): () => void {
  const key = buttonKey(pluginId, opts.id);
  buttons.set(key, {
    key,
    pluginId,
    localId: opts.id,
    icon: opts.icon,
    tooltip: opts.tooltip,
    onClick: opts.onClick,
  });
  notify();
  return () => removePluginToolbarButton(pluginId, opts.id);
}

export function removePluginToolbarButton(pluginId: string, localId: string): void {
  if (buttons.delete(buttonKey(pluginId, localId))) notify();
}

/** Drop every button a plugin registered. Used by the host's teardown. */
export function removePluginToolbarButtonsFor(pluginId: string): void {
  let changed = false;
  for (const button of Array.from(buttons.values())) {
    if (button.pluginId !== pluginId) continue;
    buttons.delete(button.key);
    changed = true;
  }
  if (changed) notify();
}

/** Every registered button, in registration order. Stable between changes. */
export function getPluginToolbarButtons(): PluginToolbarButton[] {
  return snapshot;
}

export function subscribePluginToolbarButtons(cb: () => void): () => void {
  listeners.add(cb);
  return () => { listeners.delete(cb); };
}

/** Invoke a button's handler, containing a plugin-side throw. */
export function invokePluginToolbarButton(button: PluginToolbarButton): void {
  try {
    button.onClick();
  } catch (err) {
    console.warn(`[plugins] toolbar button '${button.key}' onClick failed:`, err);
  }
}

/** Test helper — drop all buttons. */
export function _clearPluginToolbarButtons(): void {
  buttons.clear();
  notify();
}
