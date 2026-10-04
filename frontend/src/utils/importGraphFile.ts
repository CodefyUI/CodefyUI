import type { Node } from '@xyflow/react';
import { useNodeDefStore } from '../store/nodeDefStore';
import { tabHasContent, useTabStore, whenTabsHydrated, type TabState } from '../store/tabStore';
import { useToastStore } from '../store/toastStore';
import { useUIStore } from '../store/uiStore';
import { useI18n, type TranslationKey } from '../i18n';
import type { PresetDefinition, SubgraphDefinition } from '../types';
import { resolveSerializedNodes, resolveSerializedEdges } from '.';
import { nodesBoundingBox } from './autoLayout';
import { readGraphDevice } from './graphSettings';
import {
  effectivePresets,
  mergeUnknownPresetsIntoPalette,
  withPresetDefaults,
} from './presetOwnership';
import { importWorkspaceFile } from './importWorkspaceFile';
import {
  WORKSPACE_EXTENSION,
  isWorkspaceFile,
  parseWorkspaceFile,
  type WorkspaceParseFailure,
} from './workspaceFile';
import { MAX_WORKSPACE_FILE_BYTES, MAX_WORKSPACE_TABS } from './workspaceLimits';

/**
 * Reading a graph out of a file the user picked off their own disk.
 *
 * Another door onto `loadGraphDocumentInto`, and the odd one out: every other
 * reader gets its payload from the server, so the bytes have at least been
 * through a route that knows what a graph is. Here they have not. The file
 * is whatever was sitting on the disk under a `.json` name, which is why
 * this is the only door that checks the payload is a graph at all before
 * touching the canvas.
 *
 * Extracted from `Toolbar.handleImportFile` so the Graphs panel's import and
 * the toolbar's are one function rather than two. The DOM stays with the
 * caller: this takes a `File`, not an input event, because there is nothing
 * about reading a graph that needs to know an `<input type="file">` was
 * involved.
 *
 * `importFile` at the foot of the file is the live door the UI calls: it reads
 * the picked file once and routes by CONTENT between the graph opener below
 * and the `.cduiworkspace` importer, which adds tabs. Neither replaces work: a
 * graph goes into the active tab only when that tab is empty, and into a tab
 * of its own otherwise (#550). `importGraphFile` stays exported for its own
 * contract (a graph file, placed the same way, reporting its own failures)
 * and for the tests that pin it.
 */

/**
 * Does this parsed JSON claim to be a graph?
 *
 * `{}` used to pass every check the import made -- `data.nodes ?? []` is an
 * empty array, and an empty array is an array -- so picking a `package.json`,
 * a settings file, or anything else ending in `.json` REPLACED the canvas
 * with nothing, through an install that pushes no undo frame. The `nodes` key
 * is the cheapest thing that tells a graph from a JSON file that merely
 * parses: a graph that genuinely holds no nodes still writes `"nodes": []`,
 * because that is what the serializer emits.
 *
 * Returns a plain boolean rather than a type predicate on purpose: the caller
 * reads the payload as the untyped file contents it is, and a predicate here
 * would narrow it to the one key this function looked at.
 */
function looksLikeGraph(data: unknown): boolean {
  return (
    data !== null &&
    typeof data === 'object' &&
    !Array.isArray(data) &&
    'nodes' in data
  );
}

function reportImportFailure(message: string): false {
  useToastStore.getState().addToast(
    useI18n.getState().t('toolbar.import.fail', { error: message }),
    'error',
  );
  return false;
}

function refuse(message: string): false {
  useToastStore.getState().addToast(message, 'error');
  return false;
}

/** The picked file's text. Rejects with what the reader said went wrong. */
function readText(file: File): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    // A directory picked on Linux, a file deleted between the dialog and the
    // read, an unreadable network share: `readAsText` answers all of them by
    // firing `error` and never firing `load`. Without this the import simply
    // did nothing at all -- no canvas change, no message, nothing to tell
    // the user their click had been received.
    reader.onerror = () => {
      reject(new Error(reader.error?.message ?? 'Could not read the file'));
    };
    reader.onload = (e) => resolve(e.target?.result as string);
    reader.readAsText(file);
  });
}

/**
 * Read one picked file ONCE and parse it.
 *
 * Resolves to the parsed JSON inside a wrapper -- `null` is a legal JSON
 * document, so the wrapper is what tells "parsed to null" from "could not be
 * read" -- or to null after reporting why.
 */
async function readJsonFile(file: File): Promise<{ data: unknown } | null> {
  try {
    return { data: JSON.parse(await readText(file)) };
  } catch (err) {
    reportImportFailure((err as Error).message);
    return null;
  }
}

/**
 * May an import fill this tab instead of opening one of its own?
 *
 * Only an empty one, by the rule the tab's close button applies before it lets
 * a tab go without asking (`tabHasContent`): any node counts, a note included,
 * and so does a graph waiting outside an open block -- which the empty-canvas
 * overlay, looking only at the level on screen, calls empty. And only one the
 * user opened. A plugin's tab goes on saying "Opened by <plugin>", on hover and
 * in its accessible name, whatever graph is put in it, and one it opened for
 * this session only is gone after a reload, the import with it. Not one that
 * is running, either. The `.cduiworkspace` importer picks the lone empty tab
 * it closes by the same rule, less the `source` check: a closed tab keeps no
 * label.
 */
function canFill(tab: TabState | undefined): tab is TabState {
  return (
    tab !== undefined &&
    !tabHasContent(tab) &&
    !tab.source &&
    !tab.transient &&
    tab.status !== 'running'
  );
}

/**
 * What a new tab holding this graph is called: the name the graph carries,
 * else the picked file's, else the store's own `Tab N`. Graphs are often
 * handed out under one file name -- a `starter.json` per chapter -- so the
 * name inside the file is the one that tells two tabs apart.
 */
function newTabTitle(data: { name?: unknown }, fileName: string): string | undefined {
  if (typeof data.name === 'string' && data.name.trim()) return data.name.trim();
  const stem = fileName.replace(/\.[^.]*$/, '').trim();
  return stem === '' ? undefined : stem;
}

/**
 * Ask for the one-shot view fit once the canvas has been measured at the size
 * a tab switch gave it.
 *
 * Two frames, not one: a frame's animation callbacks run before its layout,
 * and React Flow's resize observer reports the canvas's new size after it --
 * a request made sooner is consumed against the old size and frames the graph
 * exactly as wrongly as the switch did. Dropped when the tab is no longer the
 * one on screen, because the request names no tab and the canvas on screen
 * consumes it.
 */
function fitOnceMeasured(
  tabId: string,
  bounds: NonNullable<ReturnType<typeof nodesBoundingBox>>,
): void {
  requestAnimationFrame(() =>
    requestAnimationFrame(() => {
      if (useTabStore.getState().activeTabId === tabId) {
        useUIStore.getState().requestLayoutFit(bounds);
      }
    }),
  );
}

/**
 * Open already-parsed JSON as a graph: into the active tab when that tab is
 * empty, and into a new tab, made active, in every other case -- no tab open
 * at all included (#550). Never over work: the install pushes no undo step and
 * autosave writes it within 250 ms, so a graph this replaced was a graph lost.
 *
 * Every refusal is decided before the first write, so a file that is turned
 * away leaves no tab behind and nothing in the palette. Reports its own
 * failure; answers whether a graph was opened.
 */
async function openGraphData(input: unknown, fileName: string): Promise<boolean> {
  const t = useI18n.getState().t;
  const addToast = useToastStore.getState().addToast;
  try {
    if (!looksLikeGraph(input)) throw new Error('Not a graph file');
    // The untyped file contents, read as what they are.
    const data = input as any;
    const rawNodes = data.nodes ?? [];
    const edges = data.edges ?? [];
    if (!Array.isArray(rawNodes) || !Array.isArray(edges)) {
      throw new Error('Invalid graph format');
    }

    // Hydration writes `{tabs, activeTabId}` wholesale. Asked before it has
    // settled, "is the active tab empty?" is answered by the placeholder the
    // editor boots with, and the graph -- in that tab or beside it -- is then
    // overwritten by the saved tabs. Nothing below awaits again, so nothing
    // can come between this decision and the writes that act on it.
    await whenTabsHydrated();
    const { tabs, activeTabId, getTab } = useTabStore.getState();
    const active = getTab(activeTabId);
    const fillId = canFill(active) ? active.id : null;
    // Only a new tab counts against the limit: filling an empty tab adds none.
    if (fillId === null && tabs.length >= MAX_WORKSPACE_TABS) {
      return refuse(t('graphs.import.tabLimit', { max: MAX_WORKSPACE_TABS }));
    }

    const store = useNodeDefStore.getState();
    const importedPresets: PresetDefinition[] = withPresetDefaults(data.presets);
    const mergedPresets = mergeUnknownPresetsIntoPalette(store.presets, importedPresets);
    const resolvingPresets = effectivePresets(importedPresets, store.presets);
    const importedSubgraphs: SubgraphDefinition[] = Array.isArray(data.subgraphs)
      ? data.subgraphs
      : [];
    const resolvedNodes = resolveSerializedNodes(
      rawNodes,
      store.definitions,
      resolvingPresets,
      importedSubgraphs,
    );
    const resolvedEdges = resolveSerializedEdges(edges, resolvedNodes);
    // Made only once the graph has been read, so a file the resolver throws
    // on leaves no empty tab behind -- the order the Graphs panel's row open
    // keeps for the same reason. Not stamped with the project: an imported
    // graph is not one of its files, as an opened example is not.
    const tabId =
      fillId ?? useTabStore.getState().createTab({ title: newTabTitle(data, fileName) });
    // The same one-call install the saved-graph readers end on --
    // `resolveSavedGraph` / `readSavedGraphDocument` in
    // `utils/openSavedGraph.ts` build a document and hand it to
    // `loadGraphDocument*` (#200 items 4 and 8). That it is one call is
    // the point: the format-version gate (ID8 fast-follow) runs inside
    // it, so importing a newer-format file opens it read-only and
    // importing an ordinary file into a previously read-only tab clears
    // the stale flag -- neither is a line a reader of a document can
    // forget to write any more.
    const tooNew = useTabStore.getState().loadGraphDocumentInto(tabId, {
      nodes: resolvedNodes,
      edges: resolvedEdges,
      // An imported file is a fresh, unsaved graph — not bound to any
      // saved file yet, so the next save always runs the overwrite
      // check (#200 item 9 moved this into the install; it used to be
      // an assignment after it).
      boundFile: null,
      presets: importedPresets,
      subgraphs: importedSubgraphs,
      segmentGroups: Array.isArray(data.segmentGroups) ? data.segmentGroups : [],
      description: typeof data.description === 'string' ? data.description : '',
      device: readGraphDevice(data.settings),
      formatVersion: data.format_version,
    });
    // Neither tab can be trusted to frame the graph on its own. A filled tab
    // is the one on screen and keeps the view it had, which after a pan can
    // show nothing of the graph that just arrived. A new tab gets the canvas's
    // first-visit fit, which uses the canvas size React Flow last measured:
    // when the switch itself resizes the canvas -- the config panel of a node
    // selected in the tab before closes -- that size is stale, and the graph
    // is framed for the narrow canvas, small and at the left edge. So both ask
    // for the one-shot fit an insert asks for. Nothing to fit for a graph with
    // no nodes, as on a first visit.
    const bounds = nodesBoundingBox(resolvedNodes as Node[]);
    if (bounds !== null) {
      if (fillId !== null) useUIStore.getState().requestLayoutFit(bounds);
      else fitOnceMeasured(tabId, bounds);
    }
    if (tooNew) {
      addToast(t('project.readOnly.loadNotice', { version: data.format_version }), 'warning');
    }
    if (importedPresets.length > 0) {
      useNodeDefStore.setState({ presets: mergedPresets });
    }
    return true;
  } catch (err) {
    return reportImportFailure((err as Error).message);
  }
}

/**
 * Read one picked file and open the graph it holds: into the active tab when
 * that tab is empty, otherwise into a tab of its own.
 *
 * Never throws and never rejects: everything that can go wrong with a file
 * off the disk is reported as a toast, and the graphs already open are left
 * alone. Resolves to whether a graph was opened.
 */
export async function importGraphFile(file: File): Promise<boolean> {
  const read = await readJsonFile(file);
  return read === null ? false : openGraphData(read.data, file.name);
}

/** Why a file that claims to be a workspace was refused whole. */
const WORKSPACE_REFUSALS: Record<WorkspaceParseFailure, TranslationKey> = {
  // Unreachable from here -- `isWorkspaceFile` said yes before the parse --
  // but the exhaustive `Record` requires it.
  not_workspace: 'workspace.import.invalid',
  invalid: 'workspace.import.invalid',
  too_new: 'workspace.import.tooNew',
};

/**
 * The Graphs panel's one Import door.
 *
 * Reads the picked file ONCE and lets the CONTENT decide: a
 * `codefyui-workspace` file is added as tabs beside the ones already open,
 * and anything else takes the graph path above -- the active tab when it is
 * empty, a tab of its own when it is not. The extension decides nothing.
 *
 * Never throws and never rejects. Resolves to whether anything was installed.
 */
export async function importFile(file: File): Promise<boolean> {
  const t = useI18n.getState().t;
  const tooLarge = () =>
    refuse(t('workspace.import.fileTooLarge', { mib: MAX_WORKSPACE_FILE_BYTES / 1024 / 1024 }));
  const oversize = file.size > MAX_WORKSPACE_FILE_BYTES;
  // Before the read, the name is all there is to go on: a file that SAYS it
  // is a workspace and is over the cap is never read into memory.
  if (oversize && file.name.toLowerCase().endsWith(WORKSPACE_EXTENSION)) return tooLarge();

  const read = await readJsonFile(file);
  if (read === null) return false;
  if (!isWorkspaceFile(read.data)) return openGraphData(read.data, file.name);

  // A workspace under another name: refused as soon as the content shows what
  // it is, and before anything is created from it.
  if (oversize) return tooLarge();
  const parsed = parseWorkspaceFile(read.data);
  if (!parsed.ok) return refuse(t(WORKSPACE_REFUSALS[parsed.reason]));
  try {
    return (await importWorkspaceFile(parsed.workspace)).imported > 0;
  } catch (err) {
    // The importer's three store calls per entry sit outside its own try, so
    // a rejection can reach here. The panel makes the call with `void`: left
    // to escape, it would be an unhandled rejection and the user would be
    // told nothing at all.
    return reportImportFailure((err as Error).message);
  }
}
