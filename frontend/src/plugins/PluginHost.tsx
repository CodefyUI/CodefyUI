/**
 * Loads installed plugins' frontend bundles and hosts their floating
 * widgets in a fixed bottom-right stack.
 *
 * Activation is once per page load (module-level guard — React StrictMode
 * double-mounts effects in dev). A plugin that throws during import or
 * activate() is reported and skipped; it cannot break the app or other
 * plugins.
 *
 * Dev hot-reload: when a linked (source_kind="local") plugin is present, the
 * host polls /api/plugins/generation and, on a bump, tears down and
 * re-activates plugin frontends with a cache-busted import — so a linked
 * plugin's frontend edits appear without a manual browser refresh. Production
 * installs (no linked plugin) never poll.
 */
import { useEffect, useRef } from 'react';
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

const IMPORT_TIMEOUT_MS = 10000;
const DEV_POLL_MS = 1500;

let hostStarted = false;
let stackEl: HTMLElement | null = null;
let cleanups: Array<() => void> = [];
let pollTimer: ReturnType<typeof setInterval> | null = null;
/** Ids activated by the last load — what a teardown has to unregister. */
let activatedIds: string[] = [];

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
  getContainer: (pluginId: string, widgetId: string) => HTMLElement
    = widgetContainer,
  importer: Importer = (url) => import(/* @vite-ignore */ url),
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
        `Plugin "${p.id}" UI failed to load`, 'error',
      );
    }
  }
  activatedIds = activated;
  return activated;
}

/**
 * Tear every activated plugin frontend down. Exported for the host's own
 * tests and for a future explicit unload path; the dev hot-reload below is
 * the only production caller today.
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
 * Dev-only: if a linked (local) plugin is installed, poll the reload
 * generation and re-activate plugin frontends whenever it bumps. The bundle is
 * re-imported with a `?v=<generation>` cache-buster (the browser keeps the ESM
 * module registry keyed by URL, so the query bump is required even though the
 * server already sends Cache-Control: no-cache). No-ops in production.
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
      unloadPluginFrontends();
      await loadPluginFrontends(
        widgetContainer,
        (url) => import(/* @vite-ignore */ `${url}?v=${gen}`),
      );
      useToastStore.getState().addToast('Plugin frontends reloaded', 'info');
    })();
  }, DEV_POLL_MS);
}

export function PluginHost() {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    stackEl = ref.current;
    if (hostStarted) return;
    hostStarted = true;
    void loadPluginFrontends().then(() => maybeStartDevHotReload());
    return () => { stackEl = null; };
  }, []);

  return <div ref={ref} className={styles.stack} data-testid="plugin-widget-stack" />;
}
