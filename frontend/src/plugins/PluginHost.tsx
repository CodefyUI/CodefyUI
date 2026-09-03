/**
 * Loads installed plugins' frontend bundles and hosts their floating
 * widgets in a fixed bottom-right stack.
 *
 * Boot activation is once per page load (module-level guard — React
 * StrictMode double-mounts effects in dev). A plugin that throws during import
 * or activate() is reported and skipped; it cannot break the app or other
 * plugins.
 *
 * After boot, `reloadPluginFrontends()` re-activates on demand: the Plugin
 * Center calls it once an install, update, uninstall, enable or disable has
 * settled, so the new UI appears without a page reload. Boot, reload and the
 * dev poller below all run on ONE promise chain, because the registries they
 * touch (`cleanups`, `activatedIds`) are module-wide: two loads overlapping
 * would tear down half of one and activate two copies of the other.
 *
 * Dev hot-reload: when a linked (source_kind="local") plugin is present, the
 * host polls /api/plugins/generation and, on a bump, reloads the frontends
 * with a cache-busted import, so a linked plugin's frontend edits appear
 * without a manual browser refresh. Production installs (no linked plugin)
 * never poll; there, re-entry is explicit from the Plugin Center.
 */
import { useEffect, useRef } from 'react';
import { useI18n } from '../i18n';
import { useNodeDefStore } from '../store/nodeDefStore';
import { useToastStore } from '../store/toastStore';
import { buildPluginAPI } from './api';
import { removePluginPanelsFor } from './panels';
import { removePluginToolbarButtonsFor } from './toolbarButtons';
import styles from './PluginHost.module.css';

interface PluginListItem {
  id: string;
  enabled: boolean;
  source_kind?: string;
  frontend_entry: string | null;
}

/** PluginListItem narrowed to entries that are ready to activate. */
interface ActivatablePlugin {
  id: string;
  enabled: true;
  frontend_entry: string;
}

type Importer = (url: string) => Promise<{ default?: unknown }>;
type GetContainer = (pluginId: string, widgetId: string) => HTMLElement;

const IMPORT_TIMEOUT_MS = 10000;
const DEV_POLL_MS = 1500;

/** The real thing, named so tests can substitute it at every entry point. */
const dynamicImporter: Importer = (url) => import(/* @vite-ignore */ url);

let hostStarted = false;
let stackEl: HTMLElement | null = null;
let cleanups: Array<() => void> = [];
let pollTimer: ReturnType<typeof setInterval> | null = null;
/** Ids activated by the last load — what a teardown has to unregister. */
let activatedIds: string[] = [];

/**
 * The single queue every activation runs on.
 *
 * `loadPluginFrontends` is not atomic: it awaits a fetch, then up to 15 s of
 * `waitForNodeDefinitions()`, then one import per plugin. An install that
 * settles inside any of those gaps would otherwise start a second load that
 * unloads nothing (the first has not registered anything yet) and then
 * activates every plugin a second time: duplicate dock tabs, duplicate toolbar
 * buttons, two createRoot() calls on one node.
 *
 * A rejected task must not poison the queue, so the chain stored for the next
 * caller is the swallowed one, while the caller still gets the real promise.
 */
let chain: Promise<unknown> = Promise.resolve();

function serialized<T>(task: () => Promise<T>): Promise<T> {
  // Both handlers are `task`: the next activation runs whether the previous
  // one resolved or threw.
  const next = chain.then(task, task);
  chain = next.catch(() => undefined);
  return next;
}

function widgetContainer(pluginId: string, widgetId: string): HTMLElement {
  const host = stackEl ?? document.body;
  const domId = `plugin-widget-${pluginId}-${widgetId}`;
  const existing = document.getElementById(domId);
  if (existing) return existing;
  const el = document.createElement('div');
  el.id = domId;
  host.appendChild(el);
  return el;
}

/**
 * Run tracked cleanups and remove plugin-created widget DOM. Called before a
 * dev re-activation so subscriptions don't accumulate and a plugin's
 * createRoot() isn't invoked twice on the same (already-rooted) node.
 *
 * Everything `buildPluginAPI` hands out that outlives the call — graph
 * subscriptions, node renderers, and since #132 dock panels, toolbar buttons
 * and execution-event subscriptions — registers its own undo here, so a
 * re-activation starts from an empty registry rather than a second copy of
 * every tab and button. The sweep below is the belt to that braces: it drops
 * anything left registered for a plugin whose cleanup threw or was never
 * tracked, so one misbehaving plugin cannot strand a tab in the dock.
 *
 * Every step is isolated, including the sweep. A belt that can itself abort is
 * not a belt: an exception out of one plugin's sweep used to propagate through
 * this loop, skipping every LATER plugin's sweep and the widget-stack clear
 * below it. That never bit because the tracked cleanups had already emptied
 * the registries by the time the sweep ran — so the second line of defence was
 * only ever correct by accident, and would have failed in exactly the case it
 * exists for. The registries are exception-safe on their own now (see
 * `detachElement` in `panels.ts`); this catch is what keeps that true when a
 * registry's own `notify()` reaches a host subscriber that throws.
 */
function teardownPlugins(pluginIds: string[] = []): void {
  for (const fn of cleanups) {
    try {
      fn();
    } catch (err) {
      console.warn('[plugins] cleanup failed:', err);
    }
  }
  cleanups = [];
  for (const id of pluginIds) {
    try {
      removePluginPanelsFor(id);
    } catch (err) {
      console.warn(`[plugins] panel sweep for '${id}' failed:`, err);
    }
    try {
      removePluginToolbarButtonsFor(id);
    } catch (err) {
      console.warn(`[plugins] toolbar sweep for '${id}' failed:`, err);
    }
  }
  if (stackEl) {
    while (stackEl.firstChild) stackEl.removeChild(stackEl.firstChild);
  }
}

/** Wait (bounded) for node definitions so plugins see a usable catalog. */
async function waitForNodeDefinitions(timeoutMs = 15000): Promise<void> {
  const start = Date.now();
  while (useNodeDefStore.getState().definitions.length === 0) {
    if (Date.now() - start > timeoutMs) return;
    await new Promise((r) => setTimeout(r, 250));
  }
}

export async function loadPluginFrontends(
  getContainer: GetContainer = widgetContainer,
  importer: Importer = dynamicImporter,
): Promise<string[]> {
  let plugins: unknown;
  try {
    const res = await fetch('/api/plugins');
    if (!res.ok) return [];
    plugins = await res.json();
  } catch {
    return [];
  }

  if (!Array.isArray(plugins)) return [];

  const activatable = plugins.filter(
    (p): p is ActivatablePlugin =>
      !!p && typeof p === 'object'
      && typeof (p as PluginListItem).id === 'string'
      && (p as PluginListItem).enabled === true
      && typeof (p as PluginListItem).frontend_entry === 'string',
  );
  if (activatable.length === 0) return [];

  await waitForNodeDefinitions();

  const activated: string[] = [];
  for (const p of activatable) {
    try {
      const mod = await Promise.race([
        importer(p.frontend_entry),
        new Promise<never>((_, reject) =>
          setTimeout(
            () => reject(new Error(`import timed out after ${IMPORT_TIMEOUT_MS}ms`)),
            IMPORT_TIMEOUT_MS,
          ),
        ),
      ]);
      if (typeof mod.default !== 'function') {
        throw new Error('frontend entry has no default export function');
      }
      mod.default(buildPluginAPI(
        p.id,
        (widgetId) => getContainer(p.id, widgetId),
        (fn) => cleanups.push(fn),
      ));
      activated.push(p.id);
    } catch (err) {
      console.warn(`[plugins] failed to activate '${p.id}' frontend:`, err);
      useToastStore.getState().addToast(
        useI18n.getState().t('pluginCenter.toast.frontendFailed', { plugin: p.id }),
        'error',
      );
    }
  }
  activatedIds = activated;
  return activated;
}

/**
 * Tear every activated plugin frontend down. Call it on the chain (see
 * `reloadPluginFrontends`) rather than on its own: an unload that lands in the
 * middle of a load takes the cleanups registered so far with it.
 */
export function unloadPluginFrontends(): void {
  teardownPlugins(activatedIds);
  activatedIds = [];
}

async function fetchGeneration(): Promise<number | null> {
  try {
    const res = await fetch('/api/plugins/generation');
    if (!res.ok) return null;
    const data = await res.json();
    return typeof data?.generation === 'number' ? data.generation : null;
  } catch {
    return null;
  }
}

/**
 * The boot activation: one load, on the chain, with nothing torn down first
 * (nothing is up yet). Kept separate from `reloadPluginFrontends` because boot
 * must not cache-bust: the very first import of a bundle should hit the HTTP
 * cache like any other module.
 *
 * The parameters exist so the host's tests can drive the boot path the mount
 * effect drives; production always calls it with none.
 */
export async function startPluginFrontends(
  getContainer: GetContainer = widgetContainer,
  importer: Importer = dynamicImporter,
): Promise<string[]> {
  return serialized(() => loadPluginFrontends(getContainer, importer));
}

/**
 * Tear down and re-activate every plugin frontend. The Plugin Center calls
 * this after an install, update, uninstall, enable or disable has settled.
 *
 * Queued behind whatever is already activating, so a reload that arrives
 * mid-boot runs after it rather than alongside it.
 *
 * The import is cache-busted with the reload generation because the browser
 * keeps its ESM module registry keyed by URL: without a fresh query a rebuilt
 * bundle would re-run the module instance already in memory. `Date.now()` is
 * the fallback for a server too old to serve /api/plugins/generation: a
 * needlessly fresh import is a far smaller cost than a stale one.
 *
 * Whatever the next /api/plugins answer omits (uninstalled, or disabled) is
 * simply not activated again; its panels, buttons and widgets went with the
 * teardown.
 */
export async function reloadPluginFrontends(
  getContainer: GetContainer = widgetContainer,
  importer: Importer = dynamicImporter,
): Promise<string[]> {
  return serialized(async () => {
    unloadPluginFrontends();
    const gen = await fetchGeneration();
    const version = gen ?? Date.now();
    return loadPluginFrontends(getContainer, (url) => importer(`${url}?v=${version}`));
  });
}

/**
 * Dev-only: if a linked (local) plugin is installed, poll the reload
 * generation and re-activate plugin frontends whenever it bumps, so a linked
 * plugin's frontend edits land without a browser refresh. The re-activation
 * (and its cache-buster) is `reloadPluginFrontends` above.
 *
 * Armed once, at boot, from the plugin list as it was then: a machine with no
 * linked plugin never polls, which is every production install.
 */
async function maybeStartDevHotReload(): Promise<void> {
  if (pollTimer !== null) return;

  let plugins: PluginListItem[];
  try {
    const res = await fetch('/api/plugins');
    if (!res.ok) return;
    const data = await res.json();
    if (!Array.isArray(data)) return;
    plugins = data;
  } catch {
    return;
  }

  const hasLocal = plugins.some(
    (p) => p && p.source_kind === 'local' && p.enabled === true,
  );
  if (!hasLocal) return;

  let lastGen = await fetchGeneration();
  if (lastGen === null) return;

  pollTimer = setInterval(() => {
    void (async () => {
      const gen = await fetchGeneration();
      if (gen === null || gen === lastGen) return;
      lastGen = gen;
      // The reload reads the generation again for its cache-buster. That
      // second read is deliberate: it is the one the import is stamped with,
      // so a build that lands between the tick and the teardown is picked up
      // instead of being pinned to a number that is already old.
      await reloadPluginFrontends();
      useToastStore.getState().addToast(
        useI18n.getState().t('pluginCenter.toast.frontendsReloaded'), 'info',
      );
    })();
  }, DEV_POLL_MS);
}

export function PluginHost() {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    stackEl = ref.current;
    if (hostStarted) return;
    hostStarted = true;
    void startPluginFrontends().then(() => maybeStartDevHotReload());
    return () => { stackEl = null; };
  }, []);

  return <div ref={ref} className={styles.stack} data-testid="plugin-widget-stack" />;
}

/**
 * Test helper: put the module back to its pre-boot state, meaning the boot
 * guard, the dev poller, the activation queue and everything a load leaves
 * behind.
 */
export function _resetPluginHostForTesting(): void {
  hostStarted = false;
  if (pollTimer !== null) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  stackEl = null;
  cleanups = [];
  activatedIds = [];
  chain = Promise.resolve();
}
