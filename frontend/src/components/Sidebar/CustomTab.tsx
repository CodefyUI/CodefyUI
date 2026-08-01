import { useCallback, useEffect, useState } from 'react';
import {
  listCustomNodes,
  listPlugins,
  type CustomNodeInfo,
  type PluginSummary,
} from '../../api/rest';
import { useI18n } from '../../i18n';
import { CustomNodeManager } from '../CustomNodeManager/CustomNodeManager';
import { RefreshIcon } from '../shared/Icons';
import styles from './NodePalette.module.css';
import tabStyles from './CustomTab.module.css';

/**
 * Everything the user has added to this install: uploaded custom-node files
 * and installed plugin packs (#126).
 *
 * The custom-node rows are a read-only summary; enable/disable/upload/delete
 * stay in the existing `CustomNodeManager` modal, opened from here rather than
 * duplicated into a 250px column. Plugin packs are read-only by design — they
 * are installed and removed through the `cdui plugin` CLI, which writes the
 * lockfile on disk.
 *
 * Both lists are fetched together on mount, and re-fetched when the manager
 * modal closes (an upload or a toggle changes what belongs here).
 */
export function CustomTab() {
  const [customNodes, setCustomNodes] = useState<CustomNodeInfo[]>([]);
  const [plugins, setPlugins] = useState<PluginSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [managerOpen, setManagerOpen] = useState(false);
  const { t } = useI18n();

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    Promise.all([listCustomNodes(), listPlugins()])
      .then(([nodes, packs]) => {
        setCustomNodes(nodes);
        setPlugins(packs);
      })
      .catch((e: Error) => {
        setCustomNodes([]);
        setPlugins([]);
        setError(e.message);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const closeManager = useCallback(() => {
    setManagerOpen(false);
    load();
  }, [load]);

  return (
    <>
      <div className={styles.header}>
        <div className={styles.headerRow}>
          <div className={styles.headerTitle}>{t('sidebar.tab.custom')}</div>
          <button
            type="button"
            className={styles.toolbarButton}
            onClick={load}
            aria-label={t('sidebar.refresh')}
            title={t('sidebar.refresh')}
          >
            <RefreshIcon size={13} />
          </button>
        </div>
      </div>

      <div className={styles.panelBody}>
        {loading && <div className={styles.stateMessage}>{t('customNodes.loading')}</div>}

        {!loading && error && (
          <div className={styles.errorWrapper}>
            <div className={styles.errorText}>{t('customTab.loadFail', { error })}</div>
            <button type="button" onClick={load} className={styles.retryButton}>
              {t('palette.retry')}
            </button>
          </div>
        )}

        {!loading && !error && (
          <div className={styles.content}>
            {/* ── Custom nodes ── */}
            <div className={tabStyles.sectionHeader}>
              <span className={tabStyles.sectionTitle}>{t('customTab.section.nodes')}</span>
              <span className={tabStyles.sectionCount}>{customNodes.length}</span>
              <button
                type="button"
                className={tabStyles.manageButton}
                onClick={() => setManagerOpen(true)}
              >
                {t('customTab.manage')}
              </button>
            </div>

            {customNodes.length === 0 ? (
              <div className={tabStyles.sectionEmpty}>{t('customTab.nodes.empty')}</div>
            ) : (
              customNodes.map((file) => (
                <div
                  key={file.filename}
                  className={tabStyles.row}
                  data-disabled={!file.enabled}
                >
                  <div className={tabStyles.rowTitle}>
                    <span className={tabStyles.rowName}>{file.filename}</span>
                    <span
                      className={file.enabled ? tabStyles.chipOn : tabStyles.chipOff}
                    >
                      {file.enabled ? t('customNodes.enabled') : t('customNodes.disabled')}
                    </span>
                  </div>
                  {file.nodes.length > 0 && (
                    <div className={tabStyles.rowDesc}>{file.nodes.join(', ')}</div>
                  )}
                </div>
              ))
            )}

            {/* ── Plugin packs ── */}
            <div className={tabStyles.sectionHeader}>
              <span className={tabStyles.sectionTitle}>{t('customTab.section.plugins')}</span>
              <span className={tabStyles.sectionCount}>{plugins.length}</span>
            </div>

            {plugins.length === 0 ? (
              <div className={tabStyles.sectionEmpty}>
                <div>{t('customTab.plugins.empty')}</div>
                <div className={tabStyles.sectionHint}>{t('customTab.plugins.hint')}</div>
              </div>
            ) : (
              plugins.map((plugin) => (
                <div key={plugin.id} className={tabStyles.row} data-disabled={!plugin.enabled}>
                  <div className={tabStyles.rowTitle}>
                    <span className={tabStyles.rowName}>{plugin.name}</span>
                    {plugin.version && (
                      <span className={tabStyles.rowVersion}>v{plugin.version}</span>
                    )}
                    <span className={plugin.enabled ? tabStyles.chipOn : tabStyles.chipOff}>
                      {plugin.enabled ? t('customNodes.enabled') : t('customNodes.disabled')}
                    </span>
                  </div>
                  {plugin.description && (
                    <div className={tabStyles.rowDesc}>{plugin.description}</div>
                  )}
                  {plugin.nodes.length > 0 && (
                    <div className={tabStyles.rowMeta}>
                      {t('empty.nodeCount', { count: plugin.nodes.length })}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        )}
      </div>

      {managerOpen && <CustomNodeManager onClose={closeManager} />}
    </>
  );
}
