import { useI18n } from '../../i18n';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { isProjectNodeType, useProjectStore } from '../../store/projectStore';
import { useTabStore } from '../../store/tabStore';
import styles from './Projects.module.css';

/**
 * How many nodes of the active tab (its canvas and its blocks) are
 * `project:` types this server has not registered: the project's mods are
 * not loaded. A primitive, so the banner re-renders only when it changes.
 */
function useUnloadedProjectNodes(): number {
  const known = useNodeDefStore((s) => s.definitions);
  return useTabStore((s) => {
    const tab = s.tabs.find((t) => t.id === s.activeTabId);
    if (!tab) return 0;
    const types: string[] = [];
    for (const node of tab.nodes) {
      if (isProjectNodeType(node.data?.type)) types.push(node.data.type);
    }
    for (const block of tab.subgraphs ?? []) {
      for (const node of block.nodes ?? []) {
        if (isProjectNodeType(node?.type)) types.push(node.type);
      }
    }
    // Before the first node list arrives every type would look missing.
    if (types.length === 0 || known.length === 0) return 0;
    const names = new Set(known.map((d) => d.node_name));
    return types.filter((type) => !names.has(type)).length;
  });
}

/**
 * The tab banner (phase 4C): this graph uses nodes from the project's mods,
 * which are not loaded. The nodes are kept exactly as saved; the banner says
 * why running them will fail and offers the review.
 */
export function ModsNotLoadedBanner() {
  const { t } = useI18n();
  const count = useUnloadedProjectNodes();
  if (count === 0) return null;
  return (
    <div role="status" className={styles.modsBanner} data-testid="mods-not-loaded-banner">
      <span>{t('projects.mods.banner', { count })}</span>
      <button type="button" onClick={() => useProjectStore.getState().openModsReview()}>
        {t('projects.mods.review')}
      </button>
    </div>
  );
}
