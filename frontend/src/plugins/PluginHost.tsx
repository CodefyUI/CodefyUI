/**
 * Loads installed plugins' frontend bundles and hosts their floating
 * widgets in a fixed bottom-right stack.
 *
 * Activation is once per page load (module-level guard — React StrictMode
 * double-mounts effects in dev). A plugin that throws during import or
 * activate() is reported and skipped; it cannot break the app or other
 * plugins.
 */
import { useEffect, useRef } from 'react';
import { useNodeDefStore } from '../store/nodeDefStore';
import { useToastStore } from '../store/toastStore';
import { buildPluginAPI } from './api';
import styles from './PluginHost.module.css';

interface PluginListItem {
  id: string;
  enabled: boolean;
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

let hostStarted = false;
let stackEl: HTMLElement | null = null;

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
      mod.default(buildPluginAPI(p.id, (widgetId) => getContainer(p.id, widgetId)));
      activated.push(p.id);
    } catch (err) {
      console.warn(`[plugins] failed to activate '${p.id}' frontend:`, err);
      useToastStore.getState().addToast(
        `Plugin "${p.id}" UI failed to load`, 'error',
      );
    }
  }
  return activated;
}

export function PluginHost() {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    stackEl = ref.current;
    if (hostStarted) return;
    hostStarted = true;
    void loadPluginFrontends();
    return () => { stackEl = null; };
  }, []);

  return <div ref={ref} className={styles.stack} data-testid="plugin-widget-stack" />;
}
