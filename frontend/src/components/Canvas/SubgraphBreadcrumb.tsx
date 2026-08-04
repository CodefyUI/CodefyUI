import { useState } from 'react';

import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';
import styles from './SubgraphBreadcrumb.module.css';

/**
 * "Main > MyBlock" trail, shown only while a sub-canvas is open (core#137).
 *
 * The canvas below it is the SAME canvas: entering a subgraph swaps the
 * definition's contents into the tab's nodes/edges, so there is no second
 * editor to keep in sync. This bar is therefore the only thing telling the
 * user which level they are editing -- which is why it also carries the
 * definition's name and the way back out.
 */
export function SubgraphBreadcrumb() {
  const { t } = useI18n();
  const tab = useTabStore((s) => s.tabs.find((x) => x.id === s.activeTabId));
  const exitSubgraph = useTabStore((s) => s.exitSubgraph);
  const exitAllSubgraphs = useTabStore((s) => s.exitAllSubgraphs);
  const renameSubgraph = useTabStore((s) => s.renameSubgraph);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');

  const stack = tab?.subgraphStack ?? [];
  if (!tab || stack.length === 0) return null;

  const names = stack.map((frame) => {
    const definition = tab.subgraphs.find((d) => d.id === frame.subgraphId);
    return definition?.name || frame.subgraphId;
  });
  const currentId = stack[stack.length - 1].subgraphId;

  const commitRename = () => {
    setEditing(false);
    renameSubgraph(currentId, draft);
  };

  return (
    <div className={styles.bar} data-testid="subgraph-breadcrumb">
      <button
        type="button"
        className={styles.crumb}
        onClick={exitAllSubgraphs}
        title={t('subgraph.breadcrumb.exitAll')}
      >
        {t('subgraph.breadcrumb.root')}
      </button>
      {names.map((name, index) => (
        <span key={`${name}-${index}`} className={styles.segment}>
          <span className={styles.separator} aria-hidden="true">
            {'▸'}
          </span>
          {index === names.length - 1 && editing ? (
            <input
              className={styles.input}
              value={draft}
              autoFocus
              aria-label={t('subgraph.rename.label')}
              onChange={(e) => setDraft(e.target.value)}
              onBlur={commitRename}
              onKeyDown={(e) => {
                if (e.key === 'Enter') commitRename();
                if (e.key === 'Escape') setEditing(false);
              }}
            />
          ) : (
            <button
              type="button"
              className={`${styles.crumb} ${
                index === names.length - 1 ? styles.current : ''
              }`}
              onClick={() => {
                if (index === names.length - 1) {
                  setDraft(name);
                  setEditing(true);
                  return;
                }
                // Clicking an ancestor leaves every level below it, one exit
                // at a time so each definition is written back on the way up.
                for (let step = names.length - 1; step > index; step -= 1) {
                  exitSubgraph();
                }
              }}
              title={
                index === names.length - 1
                  ? t('subgraph.rename.hint')
                  : t('subgraph.breadcrumb.jump')
              }
            >
              {name}
            </button>
          )}
        </span>
      ))}
      <button
        type="button"
        className={styles.exit}
        onClick={exitSubgraph}
        data-testid="subgraph-exit"
      >
        {t('subgraph.breadcrumb.back')}
      </button>
    </div>
  );
}

export default SubgraphBreadcrumb;
