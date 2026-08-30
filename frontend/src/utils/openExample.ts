import type { Edge, Node } from '@xyflow/react';
import { loadExample } from '../api/rest';
import { useNodeDefStore } from '../store/nodeDefStore';
import { useTabStore } from '../store/tabStore';
import { useToastStore } from '../store/toastStore';
import { useI18n } from '../i18n';
import type { NodeData, SegmentGroup, SubgraphDefinition } from '../types';
import { resolveSerializedNodes, resolveSerializedEdges } from '.';
import { isFormatTooNew } from './formatVersion';

/**
 * A fetched example, resolved into live canvas nodes and edges.
 *
 * Exported since #341: `api.workspace.openGraphs` reads a plugin's graph
 * through this same door, so a graph handed over by an agent is normalised
 * exactly the way a file or an example is -- unknown presets merged, block
 * definitions honoured, `format_version` carried through to the read-only
 * verdict.
 */
export interface ResolvedExample {
  nodes: Node<NodeData>[];
  edges: Edge[];
  /** The example's own name, trimmed; null when it ships without one. */
  name: string | null;
  /**
   * Subgraph definitions the example ships (core#137).
   *
   * An example file has exactly the shape `Export -> JSON` produces, so it
   * can carry collapsed blocks. Read them or the instance nodes land with
   * nothing inside: the card renders "definition missing" and any run fails
   * `Unknown subgraph`.
   */
  subgraphs: SubgraphDefinition[];
  /** Teaching Inspector segment overlays the example ships; often empty. */
  segmentGroups: SegmentGroup[];
  /**
   * The example's own description, or '' when it ships without one (#200
   * item 4). Carried so opening an example REPLACES the tab's description
   * instead of leaving the previous graph's on it -- `description` is
   * persisted through save, so a leftover gets written to disk.
   */
  description: string;
  /**
   * The example's raw `format_version` (#200 item 4). An example -- or a
   * plugin-shipped template -- written by a newer CodefyUI must open
   * read-only here exactly as it does on the Toolbar's load and import
   * paths; the verdict is `loadGraphDocument`'s to make.
   */
  formatVersion: unknown;
}

/**
 * Fetch one example and turn it into canvas-ready nodes/edges.
 *
 * Every gallery surface goes through here, so an example resolves the same
 * way whether it is opened, opened in a new tab, or inserted (core#128).
 * Throws on a failed load; the three callers below each decide what that
 * means for the graph in front of the user.
 */
async function fetchResolvedExample(path: string): Promise<ResolvedExample> {
  return resolveExample(await loadExample(path));
}

/**
 * Resolve an already-fetched example payload.
 *
 * Split out of the fetch for `insertExample` (#200 item 10), which has to
 * read `format_version` off the RAW payload and refuse before anything is
 * resolved: resolution is not a pure function -- it merges the example's
 * unknown presets into the node-def store -- so a refusal decided after it
 * would already have left the palette holding a newer build's presets.
 */
export function resolveExample(data: any): ResolvedExample {
  const store = useNodeDefStore.getState();
  // An example may ship presets the running server has never seen. Merge the
  // unknown ones in by name so its nodes resolve, without clobbering the
  // installed definitions of same-named presets.
  const importedPresets = Array.isArray(data.presets) ? data.presets : [];
  const mergedPresets = [...store.presets];
  for (const p of importedPresets) {
    if (!mergedPresets.some((ep) => ep.preset_name === p.preset_name)) {
      mergedPresets.push(p);
    }
  }

  // Passed into the resolver, not just carried alongside it: an instance
  // node's rendered ports and label come from its definition's interface,
  // so resolving without them draws an empty box for a block that is right
  // there in the file.
  const subgraphs: SubgraphDefinition[] = Array.isArray(data.subgraphs)
    ? data.subgraphs
    : [];
  const nodes = resolveSerializedNodes(
    data.nodes ?? [],
    store.definitions,
    mergedPresets,
    subgraphs,
  );
  const edges = resolveSerializedEdges(data.edges ?? [], nodes);
  if (importedPresets.length > 0) {
    useNodeDefStore.setState({ presets: mergedPresets });
  }

  const name =
    typeof data.name === 'string' && data.name.trim() ? data.name.trim() : null;
  const segmentGroups: SegmentGroup[] = Array.isArray(data.segmentGroups)
    ? data.segmentGroups
    : [];
  const description =
    typeof data.description === 'string' ? data.description : '';
  return {
    nodes,
    edges,
    name,
    subgraphs,
    segmentGroups,
    description,
    formatVersion: data.format_version,
  };
}

/**
 * Replace the active tab's graph with a resolved example.
 *
 * One call, because this is the third door onto the same state and the two
 * in `Toolbar` were the reference (#200 items 4 and 8). Hand-sequencing the
 * setters here is what lost `description` and the format-version gate on
 * this path alone; `loadGraphDocument` installs the whole document in one
 * update, so there is nothing left to forget and no order to get wrong.
 *
 * The name is passed through rather than applied by a separate `renameTab`:
 * mirroring the example's name onto the tab is what gives saves, exports and
 * the script header a meaningful name out of the box.
 */
function applyToActiveTab(example: ResolvedExample): void {
  const openedReadOnly = useTabStore.getState().loadGraphDocument({
    nodes: example.nodes,
    edges: example.edges,
    // An example is a TEMPLATE, not a file the user owns: it is bound to
    // nothing, so the first Save has to ask for a name (#200 item 9). This
    // reader used to leave the binding alone, which meant an example opened
    // into a tab holding `foo.json` inherited it -- and the next Save wrote
    // the example over foo.json without the overwrite prompt, because a
    // bound tab is exactly the case that prompt is skipped for.
    boundFile: null,
    subgraphs: example.subgraphs,
    segmentGroups: example.segmentGroups,
    name: example.name,
    description: example.description,
    formatVersion: example.formatVersion,
  });
  // Same notice the Toolbar readers show, for the same reason: read-only is
  // not a failure, and a user who is not told will read the refused Save as
  // one.
  if (openedReadOnly) {
    useToastStore.getState().addToast(
      useI18n.getState().t('project.readOnly.loadNotice', {
        version: example.formatVersion as string | number,
      }),
      'warning',
    );
  }
}

function reportLoadFailure(): false {
  useToastStore.getState().addToast(useI18n.getState().t('empty.loadError'), 'error');
  return false;
}

/**
 * Load a builtin/plugin example into the ACTIVE tab.
 *
 * Extracted from `EmptyCanvasOverlay` in #126 so the sidebar's Templates tab
 * opens an example exactly the way the empty-canvas gallery does — same
 * resolution, same preset merge, same tab rename, same error toast. The
 * template gallery modal (core#128) calls it too.
 *
 * Stores are read through `getState()` instead of hooks because this is a
 * plain function called from an event handler, not a component.
 *
 * Never throws: a failed load surfaces as a toast and leaves the graph alone.
 * Returns whether the example was applied, for callers that want to react.
 */
export async function openExample(path: string): Promise<boolean> {
  try {
    applyToActiveTab(await fetchResolvedExample(path));
    return true;
  } catch {
    return reportLoadFailure();
  }
}

/**
 * Open an example in a NEW tab, leaving the current graph untouched.
 *
 * The tab is created only after the fetch succeeds — a failed load must not
 * leave an empty tab behind as the sole evidence that anything happened.
 */
export async function openExampleInNewTab(path: string): Promise<boolean> {
  let example: ResolvedExample;
  try {
    example = await fetchResolvedExample(path);
  } catch {
    return reportLoadFailure();
  }
  useTabStore.getState().addTab();
  applyToActiveTab(example);
  return true;
}

/**
 * Merge an example into the CURRENT canvas, paste-style (core#128).
 *
 * Ids are remapped and the block is placed clear of the existing graph by
 * `insertGraph`, so inserting into a populated canvas can never clobber a
 * node that is already there. Unlike `openExample` this does not rename the
 * tab: the tab still holds the user's own graph, which the template joined.
 *
 * REFUSES a template written by a newer CodefyUI (#200 item 10). The other
 * two paths answer a too-new `format_version` by opening the graph read-only,
 * and that answer does not translate here: a merge has no separate document
 * to mark, and marking the tab read-only would punish the user's own graph
 * for what the template is. Read-only exists so an older build can never
 * write back fields it does not understand -- merging those fields INTO an
 * editable graph reaches the same place by a shorter road, since the next
 * save writes the result. So the merge does not happen at all, and the user
 * is told why rather than left wondering why nothing appeared.
 *
 * `at` is the flow-space point a DROP released on (#348). It is the only
 * difference between dragging an example out of the sidebar and clicking it:
 * everything above -- the id remap, the definition merge, the format gate --
 * is the same work either way, which is why there is one function and not
 * two.
 */
export async function insertExample(
  path: string,
  at?: { x: number; y: number },
): Promise<boolean> {
  try {
    // Read off the raw payload, BEFORE resolving: resolution merges the
    // example's unknown presets into the node-def store, so a refusal after
    // it would have already installed a newer build's presets.
    const data = await loadExample(path);
    const formatVersion = data.format_version;
    if (isFormatTooNew(formatVersion)) {
      useToastStore.getState().addToast(
        useI18n.getState().t('project.formatTooNew.insertRefused', {
          version: formatVersion as string | number,
        }),
        'error',
      );
      return false;
    }
    const example = resolveExample(data);
    if (example.nodes.length === 0) return false;
    useTabStore.getState().insertGraph(
      example.nodes,
      example.edges,
      example.subgraphs,
      at,
    );
    return true;
  } catch {
    return reportLoadFailure();
  }
}
