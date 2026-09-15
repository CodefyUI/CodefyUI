import type { Node } from '@xyflow/react';
import { loadGraph } from '../api/rest';
import { useNodeDefStore } from '../store/nodeDefStore';
import { useProjectStore } from '../store/projectStore';
import { useTabStore } from '../store/tabStore';
import { useToastStore } from '../store/toastStore';
import { useI18n } from '../i18n';
import type { GraphDocument } from '../store/tabStore';
import type {
  NodeData,
  PresetDefinition,
  SegmentGroup,
  SubgraphDefinition,
} from '../types';
import { resolveSerializedNodes, resolveSerializedEdges } from '.';
import { autoLayout, stackUnboundNotes } from './autoLayout';
import { readGraphDevice } from './graphSettings';

/**
 * Reading a SAVED GRAPH -- one of the project's own files -- into a document.
 *
 * The fourth door onto `loadGraphDocument`, and the one the Source Control
 * tab needs: when a discard puts an older version of `graphs/foo.graph.json`
 * back on disk, the tab holding it is showing something that no longer
 * exists, and the offer to reload it has to read the file exactly the way
 * opening one does. Both read it through this file rather than each writing
 * the steps out, so the two can never drift: the same preset merge, the same
 * subgraph resolution, the same layout pass for a project graph whose layout
 * file is missing.
 *
 * `openExample.ts` is the sibling for examples and templates; this file is
 * for files the user owns. The difference that matters is the binding: an
 * example is bound to nothing, and a saved graph is bound to itself.
 */

const BASE_URL = '/api';

/**
 * The graph file has been deleted, moved, or renamed since the tab opened it.
 *
 * Its own class because it is not a failure to report the way a 500 is: the
 * tab that was showing the graph is still valid, its author simply has
 * nothing to reload it from, and the caller says so in a sentence instead of
 * an error line.
 */
export class GraphMissingError extends Error {
  // A field rather than an assignment in the constructor, so it survives the
  // minified build's `new.target.name` (the reason `ApiError` spells its own
  // name out).
  override name = 'GraphMissingError';

  constructor(readonly file: string) {
    super(`No saved graph named ${file}`);
  }
}

/**
 * A saved graph as it arrives off the wire.
 *
 * Every field optional and none of them trusted: this is a file on disk,
 * possibly written by an older build, possibly hand-edited. `format_version`
 * stays `unknown` because the read-only verdict is
 * `loadGraphDocumentInto`'s to make, not this reader's.
 */
export interface SavedGraphPayload {
  nodes?: unknown[];
  edges?: unknown[];
  presets?: PresetDefinition[];
  subgraphs?: SubgraphDefinition[];
  segmentGroups?: SegmentGroup[];
  description?: string;
  /** The file's `settings` block, read through `readGraphDevice`. */
  settings?: unknown;
  /**
   * Project mode: `layout/<name>.layout.json` was missing or did not cover
   * every node, so the positions have to be computed before the graph
   * reaches the canvas.
   */
  layout_missing?: boolean;
  format_version?: unknown;
}

/**
 * Turn an already-fetched saved graph into the document to install.
 *
 * Split from the fetch the way `resolveExample` is, and for the caller the
 * Toolbar still is: it reads the file through `rest.loadGraph`, and what it
 * needed extracting was this -- the forty lines between the response and the
 * one `loadGraphDocument` call, which are the part the reload path must
 * repeat exactly.
 *
 * Not pure, and deliberately so: a saved graph may carry presets the running
 * server has never seen, and its nodes only resolve against them, so the
 * unknown ones are merged into the node-def store here (by name, never
 * clobbering an installed definition) exactly as `resolveExample` does.
 *
 * `boundFile` is the caller's binding decision -- the file a later plain
 * Save overwrites in place, or null for "ask where this should go". It is
 * the one thing not read out of the document, which is why `GraphDocument`
 * requires it.
 */
export function resolveSavedGraph(
  data: SavedGraphPayload,
  boundFile: string | null,
): GraphDocument {
  const store = useNodeDefStore.getState();
  const savedPresets = Array.isArray(data.presets) ? data.presets : [];
  const mergedPresets = [...store.presets];
  for (const p of savedPresets) {
    if (!mergedPresets.some((ep) => ep.preset_name === p.preset_name)) {
      mergedPresets.push(p);
    }
  }
  const loadedSubgraphs: SubgraphDefinition[] = Array.isArray(data.subgraphs)
    ? data.subgraphs
    : [];
  const resolvedNodes = resolveSerializedNodes(
    data.nodes ?? [],
    store.definitions,
    mergedPresets,
    loadedSubgraphs,
  );
  const resolvedEdges = resolveSerializedEdges(data.edges ?? [], resolvedNodes);
  // Missing/incomplete layout (project mode): dagre-lay-out ALL nodes
  // directly -- NOT via applyLayout, which pushes an undo snapshot and a
  // toast -- then deterministically place unbound notes. The next save
  // persists the computed layout (spec 6.3). Laid out BEFORE the install, so
  // the graph reaches the canvas already positioned.
  const laidOutNodes = data.layout_missing
    ? (stackUnboundNotes(
        autoLayout(resolvedNodes, resolvedEdges, 'all'),
      ) as Node<NodeData>[])
    : resolvedNodes;
  if (savedPresets.length > 0) {
    useNodeDefStore.setState({ presets: mergedPresets });
  }
  return {
    nodes: laidOutNodes,
    edges: resolvedEdges,
    boundFile,
    subgraphs: loadedSubgraphs,
    segmentGroups: Array.isArray(data.segmentGroups) ? data.segmentGroups : [],
    description: typeof data.description === 'string' ? data.description : '',
    device: readGraphDevice(data.settings),
    formatVersion: data.format_version,
    // `name` is deliberately absent: a saved graph is bound to its file by
    // `currentGraphFile`, not by the tab label, so a load must not rename a
    // tab the user named.
  };
}

/**
 * What opening a saved graph does to the tab it lands in.
 *
 * - `canvas` — replace what is on this canvas and bind the tab to NOTHING,
 *   so the next Save asks where the result should go. Overwriting live work
 *   is the whole of this path, which is why its callers confirm first.
 * - `bind` — the original Load: replace the canvas AND bind the tab to the
 *   file, so Save writes straight back over it.
 *
 * The two used to be one action (always `bind`), which meant opening a saved
 * graph to look at it silently took over where the tab saves.
 */
export type SavedGraphTarget = 'canvas' | 'bind';

/**
 * Open a saved graph into the ACTIVE tab.
 *
 * Extracted from `Toolbar.handleLoadGraph` so the Graphs panel opens a row
 * exactly the way the toolbar menu opened one -- not "the same idea", the
 * same function, down to which toast a failure produces.
 *
 * Deliberately does NOT confirm. The two callers disagree about when to ask:
 * a menu item the user aimed at is not a row they scrolled past, so the
 * question belongs to whoever drew the thing that was clicked. Everything
 * after the click is the same, and that is what lives here.
 *
 * `file` is the sanitized file stem, and `bind` adopts it so a later Save
 * overwrites the file in place with no overwrite warning; `canvas`
 * deliberately does not, which is what makes it safe to drop a saved graph
 * onto a canvas you are still working in -- the next Save asks for a name
 * instead of eating the original. The binding is part of installing the
 * document (#200 item 9), not a line after it: it says which file the graph
 * on screen writes to, so the two must never be set apart.
 *
 * Never throws: a failed read surfaces as a toast and leaves the graph
 * alone. Returns whether the graph was installed, for callers that want to
 * react.
 */
export async function openSavedGraph(
  // The whole list row, not just its `file`: the callers hold one, and the
  // pair is what stops `name` (the title inside the file) being mistaken for
  // the name the routes address it by.
  { file }: { name: string; file: string },
  target: SavedGraphTarget,
): Promise<boolean> {
  const t = useI18n.getState().t;
  const addToast = useToastStore.getState().addToast;
  try {
    // Read through `rest.loadGraph`, which is the call the toolbar's Load
    // always made. `readSavedGraphDocument` below is the sibling for the
    // reload path, and the only reason it exists is that it has to tell a
    // deleted file apart from a broken server; an open has one answer for
    // both, so it has no use for the distinction.
    const doc = resolveSavedGraph(await loadGraph(file), target === 'bind' ? file : null);
    // One call, not six (#200 items 4 and 8): the whole document lands in a
    // single store update, so no subscriber sees the new nodes beside the
    // old definitions, and the read-only gate is the action's own return
    // value rather than a line each reader has to remember -- which is what
    // the third reader of a document, `openExample`, did not.
    const tooNew = useTabStore.getState().loadGraphDocument(doc);
    if (tooNew) {
      addToast(
        t('project.readOnly.loadNotice', {
          version: doc.formatVersion as string | number,
        }),
        'warning',
      );
    }
    const projectDir = useProjectStore.getState().projectDir;
    if (projectDir !== null) useTabStore.getState().stampActiveTabProject(projectDir);
    return true;
  } catch (e) {
    addToast(t('toolbar.load.fail', { error: (e as Error).message }), 'error');
    return false;
  }
}

/**
 * Read one saved graph off the server and resolve it.
 *
 * Fetched here rather than through `rest.loadGraph` for one reason: that
 * function throws a plain `Error` carrying the status TEXT, so a caller
 * cannot tell "the file is gone" from "the server broke". The reload path
 * has to tell them apart -- a graph deleted by the commit being reloaded is
 * a sentence, and a 500 is an error line -- so the 404 becomes
 * `GraphMissingError` here. `loadGraph` itself is untouched; `openSavedGraph`
 * still goes through it and still shows what it always showed.
 */
export async function readSavedGraphDocument(
  file: string,
  boundFile: string | null,
): Promise<GraphDocument> {
  const res = await fetch(`${BASE_URL}/graph/load/${encodeURIComponent(file)}`);
  if (res.status === 404) throw new GraphMissingError(file);
  // The same message `loadGraph` produces, so a failure that is not a 404
  // still reads the way every other load failure in the app reads.
  if (!res.ok) throw new Error(`Load failed: ${res.statusText}`);
  return resolveSavedGraph((await res.json()) as SavedGraphPayload, boundFile);
}

/**
 * Re-read the file a tab is bound to and install it into THAT tab.
 *
 * Addressed by tab id, not "the active tab": the caller is the Source
 * Control tab reacting to a write that changed several files at once, and
 * the tabs it reloads are usually not the one in front of the user.
 *
 * The tab stays bound to the file it is being reloaded from -- that is the
 * whole point of the reload -- and keeps its label. A tab closed between the
 * offer and the click is a no-op rather than an error: the store's update
 * walks the open tabs and finds nothing to write.
 *
 * Throws `GraphMissingError` when the file is no longer on disk, which the
 * caller reports as "it is gone" while leaving the tab exactly as it is:
 * whatever the graph on screen is, it is the only copy left.
 *
 * Returns `loadGraphDocumentInto`'s verdict -- true when the file was
 * written by a newer CodefyUI, which has just put the tab into read-only.
 */
export async function reloadTabFromDisk(
  tabId: string,
  file: string,
): Promise<boolean> {
  const doc = await readSavedGraphDocument(file, file);
  return useTabStore.getState().loadGraphDocumentInto(tabId, doc);
}
