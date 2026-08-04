import type { Edge, Node } from '@xyflow/react';
import { loadExample } from '../api/rest';
import { useNodeDefStore } from '../store/nodeDefStore';
import { useTabStore } from '../store/tabStore';
import { useToastStore } from '../store/toastStore';
import { useI18n } from '../i18n';
import type { NodeData, SegmentGroup, SubgraphDefinition } from '../types';
import { resolveSerializedNodes, resolveSerializedEdges } from '.';

/** A fetched example, resolved into live canvas nodes and edges. */
interface ResolvedExample {
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
  const data = await loadExample(path);

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
  return { nodes, edges, name, subgraphs, segmentGroups };
}

/** Replace the active tab's graph with a resolved example. */
function applyToActiveTab(example: ResolvedExample): void {
  const tabs = useTabStore.getState();
  tabs.setNodes(example.nodes);
  tabs.setEdges(example.edges);
  // After the nodes: `setSubgraphs` clears the editing stack and nothing
  // else, so the canvas has to be in place first.
  tabs.setSubgraphs(example.subgraphs);
  // Unconditionally, even for an example that ships none. This REPLACES the
  // tab's graph, so a segment left over from the graph that was there names
  // head/tail ids nothing wears any more -- and `segmentGroups` is persisted
  // through save, so the broken segment would be written to disk. Both
  // Toolbar readers (`handleLoadGraph`, `handleImportFile`) already do this;
  // this is the third door onto the same state.
  tabs.setSegmentGroups(example.segmentGroups);
  // Mirror the example name onto the active tab so saves, exports, and the
  // script header all use a meaningful name out of the box.
  if (example.name) {
    tabs.renameTab(useTabStore.getState().activeTabId, example.name);
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
 */
export async function insertExample(path: string): Promise<boolean> {
  try {
    const example = await fetchResolvedExample(path);
    if (example.nodes.length === 0) return false;
    useTabStore.getState().insertGraph(
      example.nodes,
      example.edges,
      example.subgraphs,
    );
    return true;
  } catch {
    return reportLoadFailure();
  }
}
