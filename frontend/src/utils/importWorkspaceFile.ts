import { useI18n } from '../i18n';
import { useNodeDefStore } from '../store/nodeDefStore';
import {
  tabHasContent,
  useTabStore,
  whenTabsHydrated,
  type GraphDocument,
} from '../store/tabStore';
import { useToastStore } from '../store/toastStore';
import { useUIStore } from '../store/uiStore';
import { resolveUnboundDocument } from './openExample';
import { SUBGRAPH_TYPE_PREFIX } from './subgraph';
import type { ParsedWorkspace, ParsedWorkspaceTab } from './workspaceFile';
import { MAX_WORKSPACE_TABS } from './workspaceLimits';

/**
 * Applying a parsed `.cduiworkspace` file to the stores.
 *
 * The tabs are ADDED beside the ones already open; nothing that holds a graph
 * is replaced. Every entry is handled on its own, so one that cannot open
 * never sinks the rest, and the results are positional -- the way
 * `api.workspace.openGraphs` answers.
 */

export type WorkspaceSkipReason = 'invalid_graph' | 'too_many_tabs';

export type WorkspaceEntryResult = { tabId: string } | { skipped: WorkspaceSkipReason };

export interface WorkspaceImportResult {
  /** Positional: `results[i]` answers `workspace.tabs[i]`. */
  results: WorkspaceEntryResult[];
  imported: number;
}

/** File contents that passed the raw checks: an object with two lists. */
type RawGraph = Record<string, unknown> & { nodes: unknown[]; edges: unknown[] };

function asRawGraph(value: unknown): RawGraph | null {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return null;
  const graph = value as Record<string, unknown>;
  return Array.isArray(graph.nodes) && Array.isArray(graph.edges) ? (graph as RawGraph) : null;
}

type EntryOutcome =
  | { tabId: string; graph: RawGraph; tooNew: boolean; formatVersion: unknown }
  | { skipped: WorkspaceSkipReason };

/**
 * Open one entry as a background tab.
 *
 * `liveTabs` is how many tabs will still be open when the import ends, which
 * is what the 32-tab limit is about.
 */
function openEntry(entry: ParsedWorkspaceTab, liveTabs: number): EntryOutcome {
  // The raw checks come FIRST, before anything reads the graph: reading it
  // merges the presets it carries into the palette (`resolveExample`), and an
  // entry that is refused must not leave its presets behind (#401).
  const graph = asRawGraph(entry.graph);
  if (graph === null) return { skipped: 'invalid_graph' };
  // No per-graph size cap, unlike `api.workspace.openGraphs`: the export end
  // of this format measures nothing, so a cap here would refuse a tab this
  // build had just written -- and an image note embeds its PNG as a data URL,
  // which is how a single tab passes 8 MiB. The 64 MiB cap on the FILE is
  // what bounds the input.
  if (liveTabs >= MAX_WORKSPACE_TABS) return { skipped: 'too_many_tabs' };

  let doc: GraphDocument;
  try {
    // Unbound (`boundFile: null`, `name: null`): the first Save asks for a
    // name, and the 409 `graph_exists` guard applies to it like any other.
    doc = resolveUnboundDocument(graph);
  } catch {
    // The reader throws before it merges anything, so nothing was left behind.
    return { skipped: 'invalid_graph' };
  }

  const store = useTabStore.getState();
  // A blank or non-string title falls back to the default tab name.
  const title =
    typeof entry.title === 'string' && entry.title.trim() ? entry.title.trim() : undefined;
  // `createTab` mints the `graphId`. A copied one would make two tabs share
  // trained weights on the server, so the file never carries one.
  const tabId = store.createTab({ title, activate: false });
  // `loadGraphDocumentInto` makes the read-only verdict and has already
  // written it to the tab. `tooNew` is carried out of here only so the notice
  // can be raised once per format version instead of once per tab.
  const tooNew = store.loadGraphDocumentInto(tabId, doc);
  store.setTabRunSettings(tabId, entry.run);
  // Project mode stamps nothing, exactly like a freshly opened example
  // (`openExample.ts`): `projectOrigin` stays null until the first Save.
  return { tabId, graph, tooNew, formatVersion: doc.formatVersion };
}

/** Node types that never come from the node catalog, so are never "missing". */
const NOT_FROM_THE_CATALOG = new Set(['note', 'Start']);

function collectNodeTypes(rawNodes: unknown, into: Set<string>): void {
  if (!Array.isArray(rawNodes)) return;
  for (const raw of rawNodes) {
    const type = (raw as { type?: unknown } | null)?.type;
    if (typeof type !== 'string' || type === '') continue;
    if (NOT_FROM_THE_CATALOG.has(type)) continue;
    if (type.startsWith(SUBGRAPH_TYPE_PREFIX) || type.startsWith('preset:')) continue;
    into.add(type);
  }
}

/**
 * Types the imported graphs use that this install does not have, distinct and
 * in first-seen order. Exact match on `node_name`, which is the lookup
 * `resolveSerializedNodes` makes before it falls back to a stub definition.
 */
function missingNodeTypes(graphs: RawGraph[]): string[] {
  const known = new Set(useNodeDefStore.getState().definitions.map((d) => d.node_name));
  // An empty catalog has not loaded (or failed to): every type would read as
  // missing, which is not what the user should be told.
  if (known.size === 0) return [];
  const used = new Set<string>();
  for (const graph of graphs) {
    collectNodeTypes(graph.nodes, used);
    if (Array.isArray(graph.subgraphs)) {
      for (const definition of graph.subgraphs) {
        collectNodeTypes((definition as { nodes?: unknown } | null)?.nodes, used);
      }
    }
  }
  return [...used].filter((type) => !known.has(type));
}

/** Step 9: one result toast, then the notices that qualify it. */
function report(
  results: WorkspaceEntryResult[],
  newerVersions: unknown[],
  openedGraphs: RawGraph[],
): void {
  const t = useI18n.getState().t;
  const addToast = useToastStore.getState().addToast;
  const skipped = results.flatMap((r) => ('skipped' in r ? [r.skipped] : []));
  const total = results.length;
  const imported = total - skipped.length;

  const labels: Record<WorkspaceSkipReason, string> = {
    invalid_graph: t('workspace.import.reason.invalidGraph'),
    too_many_tabs: t('workspace.import.reason.tooManyTabs', { max: MAX_WORKSPACE_TABS }),
  };
  const reasons = [...new Set(skipped)].map((reason) => labels[reason]).join(', ');

  if (imported === 0) {
    addToast(t('workspace.import.none', { reasons }), 'error');
  } else if (imported === total) {
    addToast(t('workspace.import.success', { count: imported }), 'success');
  } else {
    addToast(t('workspace.import.partial', { imported, total, reasons }), 'warning');
  }

  // The existing wording, once per format version rather than once per tab.
  for (const version of newerVersions) {
    addToast(
      t('project.readOnly.loadNotice', { version: version as string | number }),
      'warning',
    );
  }

  const missing = missingNodeTypes(openedGraphs);
  if (missing.length > 0) {
    const named = missing.slice(0, 3).join(', ');
    const types =
      missing.length > 3
        ? `${named}, ${t('workspace.import.moreTypes', { count: missing.length - 3 })}`
        : named;
    addToast(t('workspace.import.missingTypes', { types }), 'warning');
  }
}

/**
 * Import a parsed workspace. Raises its own toasts; resolves to what happened
 * to each entry.
 */
export async function importWorkspaceFile(
  workspace: ParsedWorkspace,
): Promise<WorkspaceImportResult> {
  // Step 3. Hydration writes `{tabs, activeTabId}` wholesale, so an import
  // that landed before it settled would be overwritten. Nothing below awaits
  // again, so nothing can interleave with the steps that follow.
  await whenTabsHydrated();

  // Step 4. "Lone empty tab": exactly one tab, empty, not transient, not running.
  // Read AFTER the wait: before it, the one tab is only the boot placeholder.
  const before = useTabStore.getState().tabs;
  const lone = before.length === 1 ? before[0] : null;
  const loneEmptyTabId =
    lone !== null && !tabHasContent(lone) && !lone.transient && lone.status !== 'running'
      ? lone.id
      : null;

  // Step 5. In file order, each entry on its own.
  const results: WorkspaceEntryResult[] = [];
  const openedGraphs: RawGraph[] = [];
  const newerVersions: unknown[] = [];
  for (const entry of workspace.tabs) {
    // The lone empty tab is closed in step 7, so it does not count: with it,
    // a 32-tab export could never round-trip into a fresh browser.
    const liveTabs = useTabStore.getState().tabs.length - (loneEmptyTabId === null ? 0 : 1);
    const outcome = openEntry(entry, liveTabs);
    if ('skipped' in outcome) {
      results.push(outcome);
      continue;
    }
    results.push({ tabId: outcome.tabId });
    openedGraphs.push(outcome.graph);
    if (outcome.tooNew && !newerVersions.includes(outcome.formatVersion)) {
      newerVersions.push(outcome.formatVersion);
    }
  }

  const openedIds = results.flatMap((r) => ('tabId' in r ? [r.tabId] : []));
  if (openedIds.length > 0) {
    // Step 6. The tab that was active, if it made it; else the first one.
    // `active` indexes the FILE, refused entries included, and `results` is
    // positional against it -- so a refusal earlier in the file shifts nothing.
    const wanted = workspace.active === null ? undefined : results[workspace.active];
    const activeId = wanted !== undefined && 'tabId' in wanted ? wanted.tabId : openedIds[0];
    useTabStore.getState().setActiveTab(activeId);
    // Step 7. Only now that something has taken its place.
    if (loneEmptyTabId !== null) useTabStore.getState().removeTab(loneEmptyTabId);
    // Step 8. Through the stores, so each one is persisted the normal way --
    // which means `localStorage.setItem`, and that throws where storage is
    // blocked or full. Wrapped because the tabs are already open by now: left
    // to escape, the router would report a finished import as "Import failed"
    // and the user would import the same file again, ending up with every tab
    // twice. A preference that could not be stored is simply not applied.
    try {
      const { locale, ...ui } = workspace.preferences;
      useUIStore.getState().applyPreferences(ui);
      if (locale !== undefined) useI18n.getState().setLocale(locale);
    } catch {
      // No toast: the import itself succeeded, and `report` says so below.
    }
  }

  report(results, newerVersions, openedGraphs);
  return { results, imported: openedIds.length };
}
