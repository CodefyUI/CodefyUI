import { loadExample } from '../api/rest';
import { useNodeDefStore } from '../store/nodeDefStore';
import { useTabStore } from '../store/tabStore';
import { useToastStore } from '../store/toastStore';
import { useI18n } from '../i18n';
import { resolveSerializedNodes, resolveSerializedEdges } from '.';

/**
 * Load a builtin/plugin example into the ACTIVE tab.
 *
 * Extracted from `EmptyCanvasOverlay` in #126 so the sidebar's Templates tab
 * opens an example exactly the way the empty-canvas gallery does — same
 * resolution, same preset merge, same tab rename, same error toast. Any future
 * gallery surface should call this rather than re-implement it.
 *
 * Stores are read through `getState()` instead of hooks because this is a
 * plain function called from an event handler, not a component.
 *
 * Never throws: a failed load surfaces as a toast and leaves the graph alone.
 * Returns whether the example was applied, for callers that want to react.
 */
export async function openExample(path: string): Promise<boolean> {
  const { t } = useI18n.getState();
  try {
    const data = await loadExample(path);
    const rawNodes = data.nodes ?? [];
    const edges = data.edges ?? [];

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

    const resolvedNodes = resolveSerializedNodes(rawNodes, store.definitions, mergedPresets);
    const resolvedEdges = resolveSerializedEdges(edges, resolvedNodes);
    const tabs = useTabStore.getState();
    tabs.setNodes(resolvedNodes);
    tabs.setEdges(resolvedEdges);

    // Mirror the example name onto the active tab so saves, exports, and the
    // script header all use a meaningful name out of the box.
    const exampleName = typeof data.name === 'string' && data.name.trim() ? data.name.trim() : null;
    if (exampleName) {
      tabs.renameTab(useTabStore.getState().activeTabId, exampleName);
    }

    if (importedPresets.length > 0) {
      useNodeDefStore.setState({ presets: mergedPresets });
    }
    return true;
  } catch {
    useToastStore.getState().addToast(t('empty.loadError'), 'error');
    return false;
  }
}
