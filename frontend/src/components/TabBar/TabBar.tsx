import { useState, useCallback } from 'react';
import { useTabStore } from '../../store/tabStore';
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
      if (tab && tab.status === 'running') {
        const ok = await confirm({
          title: t('tabs.closeRunning'),
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

          return (
            <div
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              onDoubleClick={() => startRename(tab.id, tab.name)}
              className={styles.tab}
              style={{
                background: isActive ? '#1e1e1e' : 'transparent',
                borderBottom: isActive ? '2px solid #2196F3' : '2px solid transparent',
                color: isActive ? '#eee' : '#777',
                fontWeight: isActive ? 600 : 400,
              }}
            >
              {/* Running indicator */}
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
