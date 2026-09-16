import { useNodeDefStore } from '../store/nodeDefStore';
import { useTabStore } from '../store/tabStore';
import { useToastStore } from '../store/toastStore';
import { useI18n } from '../i18n';
import type { SubgraphDefinition } from '../types';
import { resolveSerializedNodes, resolveSerializedEdges } from '.';
import { readGraphDevice } from './graphSettings';

/**
 * Reading a graph out of a file the user picked off their own disk.
 *
 * Another door onto `loadGraphDocument`, and the odd one out: every other
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

/**
 * Read one picked file and install the graph it holds into the active tab.
 *
 * Never throws and never rejects: everything that can go wrong with a file
 * off the disk is reported as a toast, and the graph on screen is left
 * alone. Resolves to whether a graph was installed.
 */
export function importGraphFile(file: File): Promise<boolean> {
  const t = useI18n.getState().t;
  const addToast = useToastStore.getState().addToast;
  const fail = (message: string): false => {
    addToast(t('toolbar.import.fail', { error: message }), 'error');
    return false;
  };

  return new Promise<boolean>((resolve) => {
    const reader = new FileReader();
    // A directory picked on Linux, a file deleted between the dialog and the
    // read, an unreadable network share: `readAsText` answers all of them by
    // firing `error` and never firing `load`. Without this the import simply
    // did nothing at all -- no canvas change, no message, nothing to tell
    // the user their click had been received.
    reader.onerror = () => {
      resolve(fail(reader.error?.message ?? 'Could not read the file'));
    };
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target?.result as string);
        if (!looksLikeGraph(data)) throw new Error('Not a graph file');
        const rawNodes = data.nodes ?? [];
        const edges = data.edges ?? [];
        if (!Array.isArray(rawNodes) || !Array.isArray(edges)) {
          throw new Error('Invalid graph format');
        }
        const store = useNodeDefStore.getState();
        const importedPresets = Array.isArray(data.presets) ? data.presets : [];
        const mergedPresets = [...store.presets];
        for (const p of importedPresets) {
          if (!mergedPresets.some((ep) => ep.preset_name === p.preset_name)) {
            mergedPresets.push(p);
          }
        }
        const importedSubgraphs: SubgraphDefinition[] = Array.isArray(data.subgraphs)
          ? data.subgraphs
          : [];
        const resolvedNodes = resolveSerializedNodes(
          rawNodes,
          store.definitions,
          mergedPresets,
          importedSubgraphs,
        );
        const resolvedEdges = resolveSerializedEdges(edges, resolvedNodes);
        // The same one-call install the saved-graph readers end on --
        // `resolveSavedGraph` / `readSavedGraphDocument` in
        // `utils/openSavedGraph.ts` build a document and hand it to
        // `loadGraphDocument*` (#200 items 4 and 8). That it is one call is
        // the point: the format-version gate (ID8 fast-follow) runs inside
        // it, so importing a newer-format file opens it read-only and
        // importing an ordinary file into a previously read-only tab clears
        // the stale flag -- neither is a line a reader of a document can
        // forget to write any more.
        const tooNew = useTabStore.getState().loadGraphDocument({
          nodes: resolvedNodes,
          edges: resolvedEdges,
          // An imported file is a fresh, unsaved graph — not bound to any
          // saved file yet, so the next save always runs the overwrite
          // check (#200 item 9 moved this into the install; it used to be
          // an assignment after it).
          boundFile: null,
          subgraphs: importedSubgraphs,
          segmentGroups: Array.isArray(data.segmentGroups) ? data.segmentGroups : [],
          description: typeof data.description === 'string' ? data.description : '',
          device: readGraphDevice(data.settings),
          formatVersion: data.format_version,
        });
        if (tooNew) {
          addToast(t('project.readOnly.loadNotice', { version: data.format_version }), 'warning');
        }
        if (importedPresets.length > 0) {
          useNodeDefStore.setState({ presets: mergedPresets });
        }
        resolve(true);
      } catch (err) {
        resolve(fail((err as Error).message));
      }
    };
    reader.readAsText(file);
  });
}
