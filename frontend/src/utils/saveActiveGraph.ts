import { saveGraph, listGraphs } from '../api/rest';
import { useTabStore } from '../store/tabStore';
import { useProjectStore } from '../store/projectStore';
import { useToastStore } from '../store/toastStore';
import { useI18n } from '../i18n';
import { confirm, prompt } from './dialog';
import { sanitizeGraphName, findGraphNameCollision } from './index';

/**
 * Save the active tab's graph.
 *
 * - Non-project mode: ALWAYS prompt for a name (legacy behavior, Toolbar.tsx).
 * - Project mode + bound + !saveAs: overwrite the bound file IN PLACE, no
 *   prompt (git is the undo -- ID9).
 * - Save As (or unbound gallery/import graphs): prompt + collision guard.
 *
 * Task 13 extends this with project-origin stamping + cross-project refusal.
 * Task 16 (ID8) refuses outright when the active tab is read-only (its graph
 * was loaded from a newer format_version than this build writes).
 */
export async function saveActiveGraph(opts: { saveAs?: boolean } = {}): Promise<void> {
  const t = useI18n.getState().t;
  const addToast = useToastStore.getState().addToast;
  const store = useTabStore.getState();
  const tab = store.tabs.find((tb) => tb.id === store.activeTabId);
  if (!tab) return;
  if (tab.readOnly) {
    // A graph written by a newer CodefyUI opens read-only so an older build
    // can never round-trip-drop fields it does not understand (ID8).
    useToastStore.getState().addToast(t('project.readOnly.saveBlocked'), 'error');
    return;
  }
  const projectDir = useProjectStore.getState().projectDir;
  const projectMode = projectDir !== null;

  // Cross-project footgun guard (ID10): a tab stamped with a DIFFERENT project
  // must never overwrite into the currently-open project.
  if (projectMode && tab.projectOrigin != null && tab.projectOrigin !== projectDir) {
    addToast(t('project.save.crossProjectRefused', { origin: tab.projectOrigin }), 'error');
    return;
  }

  const inPlace = projectMode && !!tab.currentGraphFile && !opts.saveAs;

  let targetName: string;
  if (inPlace) {
    targetName = tab.currentGraphFile as string;
  } else {
    const entered = await prompt({ title: t('toolbar.save.prompt'), placeholder: 'graph-name' });
    const trimmed = entered?.trim();
    if (!trimmed) return;
    let existing: { name: string; file: string }[] = [];
    try {
      const r = await listGraphs();
      if (Array.isArray(r)) existing = r;
    } catch {
      /* list unavailable -- proceed without the overwrite check */
    }
    const colliding = findGraphNameCollision(trimmed, existing, tab.currentGraphFile);
    if (colliding !== null) {
      const okConfirm = await confirm({
        title: t('toolbar.save.overwriteConfirm', { name: colliding }),
        confirmText: t('toolbar.save'),
        variant: 'danger',
      });
      if (!okConfirm) return;
    }
    targetName = trimmed;
  }

  try {
    const { nodes, edges, presets, segmentGroups } = store.getSerializedGraph();
    await saveGraph({
      nodes, edges, name: targetName,
      description: tab.description ?? '', presets, segmentGroups,
    });
    store.setCurrentGraphFile(sanitizeGraphName(targetName));
    if (projectMode) store.stampActiveTabProject(projectDir);
    addToast(t('toolbar.save.success', { name: targetName }), 'success');
  } catch (e) {
    addToast(t('toolbar.save.fail', { error: (e as Error).message }), 'error');
  }
}
