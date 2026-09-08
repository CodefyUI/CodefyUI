import { defineConfig, type Rollup } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

/**
 * Chunk layout.
 *
 * The entry bundle used to be one 1.48 MB file, past Rollup's 500 kB warning.
 * It is cut along lines that change at different rates, so a release only
 * re-downloads what changed and no single chunk can drift past the warning
 * unnoticed:
 *   react     react, react-dom, scheduler, zustand (moves on a React bump)
 *   flow      React Flow, dagre and d3 (moves on a React Flow bump)
 *   vendor    any other package the entry reaches statically (none today)
 *   i18n      the string tables
 *   app-core  stores, API clients, utils, hooks, plugin host: the non-visual
 *             layer every component imports
 *   index     the components, App and main
 * Chunk edges are all one-way, so chunks evaluate in a fixed order:
 *   index    -> react, flow, i18n, app-core  (dynamic: katexRenderer, codemirrorSetup)
 *   app-core -> react, flow, i18n
 *   i18n     -> react
 *   flow     -> react
 * `onLog` below fails the build if an import ever closes a cycle.
 *
 * CodeMirror (shared/codemirrorSetup.ts) and KaTeX (shared/katexRenderer.tsx)
 * are reached only through a dynamic import() and are deliberately NOT named
 * here: a manual chunk is loaded eagerly, so naming them would undo the lazy
 * loading. `eagerIds` keeps any module that is reachable only through an
 * import() out of the eager chunks without a hand-maintained list, and
 * LAZY_ONLY turns a static import of one of those packages into a build
 * error instead of a silent 260 kB regression.
 *
 * Rollup pulls every UNASSIGNED static dependency into the first manual chunk
 * that walks it. That is why the packages are assigned explicitly: left open,
 * zustand was walked into app-core, and since every store imports useI18n
 * while i18n imports zustand, Rollup reported a circular chunk.
 */
type ChunkMeta = Parameters<Rollup.GetManualChunk>[1];

/** Packages that exist only behind a dynamic import(). */
const LAZY_ONLY = /\/node_modules\/(katex|react-katex|@codemirror|@lezer)\//;
const REACT_RUNTIME = /\/node_modules\/(react|react-dom|scheduler|zustand|use-sync-external-store)\//;
const FLOW_RUNTIME = /\/node_modules\/(@xyflow\/[^/]+|@dagrejs\/[^/]+|d3-[^/]+|classcat)\//;
const APP_CORE = /\/src\/(store|utils|api|hooks|plugins)\//;
const STYLESHEET = /\.css(\?|$)/;

const eagerCache = new WeakMap<ChunkMeta, Set<string>>();

/** Module ids reachable from the entry through static imports alone. */
function eagerIds(meta: ChunkMeta): Set<string> {
  const cached = eagerCache.get(meta);
  if (cached) return cached;
  const ids = new Set<string>();
  const queue = [...meta.getModuleIds()].filter((id) => meta.getModuleInfo(id)?.isEntry);
  if (queue.length === 0) throw new Error('manualChunks: no entry module found');
  for (const id of queue) {
    if (ids.has(id)) continue;
    ids.add(id);
    queue.push(...(meta.getModuleInfo(id)?.importedIds ?? []));
  }
  eagerCache.set(meta, ids);
  return ids;
}

function manualChunks(id: string, meta: ChunkMeta): string | undefined {
  if (id.includes('/node_modules/')) {
    if (!eagerIds(meta).has(id)) return undefined;
    // Loud, at build time, when a static import undoes one of the two lazy
    // boundaries. Checked before the stylesheet rule so that re-adding
    // `import 'katex/dist/katex.min.css'` to main.tsx is caught as well.
    if (LAZY_ONLY.test(id)) {
      throw new Error(`${id} is reached through a static import; it must stay behind its dynamic import()`);
    }
    // A stylesheet stays with the module that imports it (Rollup's default).
    // App.css and @xyflow/react/dist/style.css set the same properties on the
    // same selectors at equal specificity (edge stroke, selection, minimap,
    // controls), so their order inside index.css is part of the rendered
    // result; a flow.css linked ahead of index.css would flip those rules.
    if (STYLESHEET.test(id)) return undefined;
    if (REACT_RUNTIME.test(id)) return 'react';
    if (FLOW_RUNTIME.test(id)) return 'flow';
    return 'vendor';
  }
  if (STYLESHEET.test(id)) return undefined;
  if (id.includes('/src/i18n/')) return 'i18n';
  // Only the statically reachable part of the layer: a store or API client
  // that later moves behind a lazy tab must not be dragged back by this rule.
  if (APP_CORE.test(id) && eagerIds(meta).has(id)) return 'app-core';
  return undefined;
}

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    rollupOptions: {
      output: { manualChunks },
      // A chunk cycle lets a module read an import before its chunk has
      // evaluated, and Rollup only warns about it. Fail the build instead, so
      // an import that closes a cycle is caught by `pnpm build` rather than
      // by a blank page after `cdui update`.
      onLog(level, log, handler) {
        if (log.code === 'CIRCULAR_CHUNK') {
          throw new Error(`manualChunks produced a chunk cycle: ${log.message}`);
        }
        handler(level, log);
      },
    },
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/plugins': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
