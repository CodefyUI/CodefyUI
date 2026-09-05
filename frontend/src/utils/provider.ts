import type { PluginCatalogEntry } from '../api/rest';

/**
 * The one place that answers "which plugin did this come from?".
 *
 * A node definition carries `provider` (`builtin`, `custom`, or
 * `plugin:<id>`; see `types/index.ts`) and an example carries the same shape
 * in `source`. Four surfaces read it -- the palette tooltip, the palette
 * search, the canvas quick search and the gallery's detail pane -- and every
 * one of them wants the same two answers, so the parse lives here rather than
 * four times over -- the gallery's detail pane included, which used to keep a
 * second copy of the prefix test for a gate this module now answers.
 *
 * Both functions are PURE, and deliberately take the catalog index as an
 * argument instead of reading `usePluginStore` themselves. A component that
 * names a plugin has to re-render when the catalog lands -- `byId` is rebuilt
 * asynchronously after boot -- and that only happens if the component itself
 * subscribes with `usePluginStore((s) => s.byId)`. A helper reading
 * `getState()` would render the id once and never correct itself.
 *
 * ── Why an unknown id still gets a line ────────────────────────────────
 * `byId` is empty in three ordinary states: before the boot fetch resolves,
 * on a server with no Plugin Center (a 404 clears the catalog), and after a
 * network error. Suppressing the provenance until the catalog answers would
 * make the line flicker in on every page load; showing the id instead is a
 * sentence that is already true -- the node type is `edu:FilterRows` and the
 * line says it came from `edu` -- and it improves by itself the moment the
 * catalog arrives.
 */

/** Plugin catalog rows keyed by id -- `usePluginStore`'s `byId` slice. */
export type PluginIndex = Record<string, PluginCatalogEntry>;

const PLUGIN_PREFIX = 'plugin:';

/**
 * The plugin id inside a `provider` / `source` value, or null.
 *
 * `plugin:` with nothing after it is null rather than an empty string: it
 * names no plugin, and an empty name in a sentence reads as a bug.
 */
export function pluginIdOf(provider: string | undefined | null): string | null {
  if (typeof provider !== 'string' || !provider.startsWith(PLUGIN_PREFIX)) return null;
  const id = provider.slice(PLUGIN_PREFIX.length);
  return id === '' ? null : id;
}

/**
 * What to call the plugin a node or an example came from, or null when it
 * came from no plugin at all.
 *
 * Three answers in order: the catalog's human name, the bare id, null.
 * `hasOwnProperty` rather than a plain index read because `byId` is built from
 * parsed JSON -- a plugin id of `constructor` would otherwise resolve to
 * `Object.prototype.constructor` and put the word "Object" in the sentence.
 */
export function pluginNameOf(
  byId: PluginIndex,
  provider: string | undefined | null,
): string | null {
  const id = pluginIdOf(provider);
  if (id === null) return null;
  const entry = Object.prototype.hasOwnProperty.call(byId, id) ? byId[id] : undefined;
  return entry?.name || id;
}
