import { useState, useCallback } from 'react';
import { useTabStore, tabHasContent, tabNodeCount } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { confirm } from '../../utils/dialog';
import styles from './TabBar.module.css';

export function TabBar() {
  const tabs = useTabStore((s) => s.tabs);
  const activeTabId = useTabStore((s) => s.activeTabId);
  const addTab = useTabStore((s) => s.addTab);
  const removeTab = useTabStore((s) => s.removeTab);
  const setActiveTab = useTabStore((s) => s.setActiveTab);
  const renameTab = useTabStore((s) => s.renameTab);
  const { t } = useI18n();

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState('');

  const startRename = useCallback((id: string, currentName: string) => {
    setEditingId(id);
    setEditingName(currentName);
  }, []);

  const commitRename = useCallback(() => {
    if (editingId && editingName.trim()) {
      renameTab(editingId, editingName.trim());
    }
    setEditingId(null);
  }, [editingId, editingName, renameTab]);

  const handleClose = useCallback(
    async (e: React.MouseEvent, id: string) => {
      e.stopPropagation();
      // The close button only renders when tabs.length > 1
      /* v8 ignore start */
      if (tabs.length <= 1) return;
      /* v8 ignore stop */
      const tab = tabs.find((t) => t.id === id);
      // Only reachable if the tab vanished between render and click.
      /* v8 ignore start */
      if (!tab) return;
      /* v8 ignore stop */
      const hasContent = tabHasContent(tab);
      // The graph warning, reused by both branches below so a running tab with
      // a graph in it is still told what it is about to lose.
      const lossMessage = hasContent
        ? t('tabs.close.confirmMessage', { count: tabNodeCount(tab) })
        : undefined;
      if (tab.status === 'running') {
        // A running tab already asked before #331, and its question is the
        // stronger one (closing kills the run too). Chaining the content
        // confirm after it would mean two dialogs for one click, so the run
        // question just carries the graph warning as its body.
        const ok = await confirm({
          title: t('tabs.closeRunning'),
          message: lossMessage,
          variant: 'danger',
        });
        if (!ok) return;
      } else if (hasContent) {
        // #331: one misclick used to discard a whole graph with no undo --
        // `removeTab` drops the tab's undo/redo stacks along with it, so there
        // was nothing left to undo from. An empty tab still closes silently:
        // asking about nothing is the noise that trains people to click
        // through the dialog that matters.
        const ok = await confirm({
          title: t('tabs.close.confirmTitle', { name: tab.name }),
          message: lossMessage,
          confirmText: t('tabs.close.confirmButton'),
          variant: 'danger',
        });
        if (!ok) return;
      }
      removeTab(id);
    },
    [tabs, removeTab, t]
  );

  return (
    <div className={styles.bar}>
      <div className={styles.tabsScroll}>
        {tabs.map((tab) => {
          const isActive = tab.id === activeTabId;
          const isRunning = tab.status === 'running';
          const isEditing = editingId === tab.id;
          // A plugin-opened tab says so on hover rather than in the strip:
          // the tab is 180px wide and already carries a name, a running dot
          // and a close button.
          const sourceTitle = tab.source
            ? t('tabBar.sourceTitle', { plugin: tab.source.pluginId })
            : undefined;

          return (
            <div
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              onDoubleClick={() => startRename(tab.id, tab.name)}
              className={styles.tab}
              title={sourceTitle}
              style={{
                background: isActive ? 'var(--surface-raised)' : 'transparent',
                borderBottom: isActive ? '2px solid var(--accent)' : '2px solid transparent',
                color: isActive ? 'var(--text-primary)' : 'var(--text-muted)',
                // Left as numeric literals: TabBar.test.tsx pins the exact
                // string ('600'/'400') and isn't in this migration's file
                // list to update.
                fontWeight: isActive ? 600 : 400,
              }}
            >
              {/* Running indicator. #FFC107 already equals --status-running
                  exactly (no contrast/colour bug here); left as a literal
                  because TabBar.test.tsx pins the hex substring in
                  boxShadow and the test file is outside this migration's
                  scope. */}
              {isRunning && (
                <span
                  className={styles.runningDot}
                  style={{
                    background: '#FFC107',
                    boxShadow: '0 0 4px #FFC107',
                  }}
                />
              )}

              {/* Tab name */}
              {isEditing ? (
                <input
                  autoFocus
                  value={editingName}
                  onChange={(e) => setEditingName(e.target.value)}
                  onBlur={commitRename}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') commitRename();
                    if (e.key === 'Escape') setEditingId(null);
                  }}
                  onClick={(e) => e.stopPropagation()}
                  className={styles.editInput}
                />
              ) : (
                <span className={styles.tabName}>{tab.name}</span>
              )}

              {/* Read-only badge (#341). One word, after the name, so the
                  name is what gets the room and the badge is what gets
                  clipped last. `readOnly` means Save is refused and plugin
                  writes are refused -- it does not mean the tab is frozen,
                  so rename and close are untouched. */}
              {tab.readOnly && (
                <span className={styles.badge}>{t('tabBar.readOnly')}</span>
              )}

              {/* Close button */}
              {tabs.length > 1 && (
                <span
                  onClick={(e) => handleClose(e, tab.id)}
                  className={styles.closeBtn}
                >
                  ×
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* Add tab button */}
      <button type="button"
        onClick={() => addTab()}
        title={t('tabs.add')}
        className={styles.addBtn}
      >
        +
      </button>
    </div>
  );
}
