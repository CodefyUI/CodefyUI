import { useCallback, useEffect, useState } from 'react';
import {
  listCustomNodes,
  listPlugins,
  type CustomNodeInfo,
  type PackSummary,
  type PluginSummary,
} from '../../api/rest';
import { useI18n } from '../../i18n';
import { usePackStore } from '../../store/packStore';
import { useUIStore } from '../../store/uiStore';
import { CustomNodeManager } from '../CustomNodeManager/CustomNodeManager';
import { StatusPill } from '../PackCenter/PackCard';
import { catalogKey, localizedPackTitle } from '../../utils/packAvailability';
import { RefreshIcon } from '../shared/Icons';
import styles from './NodePalette.module.css';
import tabStyles from './CustomTab.module.css';

type PackStoreState = ReturnType<typeof usePackStore.getState>;

// Module-scope selectors, so each subscription compares the SAME function's
// output frame to frame. Only two, and neither moves while a download runs:
// `packs` is replaced when the catalog is re-read or a status changes, which
// is exactly when these rows have something new to say.
const selectPacks = (state: PackStoreState): PackSummary[] => state.packs;
const selectPacksUnsupported = (state: PackStoreState): boolean => state.unsupported;

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
 *
 * Optional packs sit between the two, and are the one section that owns no
 * fetch: the catalog is read once by the sidebar shell this tab lives in
 * (`usePackCatalogBootstrap`) and kept in `packStore`, because an install
 * outlives the tab that started it. The rows are a pure view of it, and the
 * only way to change anything is the Package Center itself.
 */
export function CustomTab() {
  const [customNodes, setCustomNodes] = useState<CustomNodeInfo[]>([]);
  const [plugins, setPlugins] = useState<PluginSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [managerOpen, setManagerOpen] = useState(false);
  const { t } = useI18n();

  const packs = usePackStore(selectPacks);
  const packsUnsupported = usePackStore(selectPacksUnsupported);
  const openPackCenter = useUIStore((s) => s.openPackCenter);

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

  // The Refresh button means "re-read everything this tab shows", and the
  // pack catalog is now part of that — it is also the retry for a boot read
  // that never arrived. Mount does NOT go through here: the shell has already
  // read the catalog by the time this tab can be selected.
  const refreshAll = useCallback(() => {
    load();
    void usePackStore.getState().refresh();
  }, [load]);

  /** A pack's name and one-line description, preferring this build's copy. */
  const packCopy = (pack: PackSummary): { title: string; desc: string } => {
    const descKey = catalogKey(pack.id, 'desc');
    return {
      // A one-entry index: this row IS the pack, and the shared rule is
      // what keeps this list, the node badges and the panel in agreement.
      title: localizedPackTitle(t, { [pack.id]: pack }, pack.id),
      desc: descKey !== null ? t(descKey) : pack.description,
    };
  };

  // A server with no Package Center answers with no catalog at all, so an
  // empty list and an unsupported server are the same sentence here. The
  // count is taken from the same list the rows are, so a stale catalog left
  // over from a previous server cannot make the header disagree with them.
  const visiblePacks = packsUnsupported ? [] : packs;

  return (
    <>
      <div className={styles.header}>
        <div className={styles.headerRow}>
          <div className={styles.headerTitle}>{t('sidebar.tab.custom')}</div>
          <button
            type="button"
            className={styles.toolbarButton}
            onClick={refreshAll}
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

            {/* ── Optional packs ── */}
            <div className={tabStyles.sectionHeader}>
              <span className={tabStyles.sectionTitle}>{t('customTab.section.packs')}</span>
              {/* Rows, like both sibling sections: a count beside a
                  section title says how much is in it. A "1" over three
                  listed packs reads as a list that half failed to load. */}
              <span className={tabStyles.sectionCount}>{visiblePacks.length}</span>
              <button
                type="button"
                className={tabStyles.manageButton}
                onClick={() => openPackCenter()}
              >
                {t('customTab.packs.open')}
              </button>
            </div>

            {visiblePacks.length === 0 ? (
              <div className={tabStyles.sectionEmpty}>
                <div>{t('customTab.packs.empty')}</div>
                <div className={tabStyles.sectionHint}>{t('customTab.packs.hint')}</div>
              </div>
            ) : (
              visiblePacks.map((pack) => {
                const copy = packCopy(pack);
                return (
                  <div key={pack.id} className={tabStyles.row}>
                    <div className={tabStyles.rowTitle}>
                      <span className={tabStyles.rowName}>{copy.title}</span>
                      <StatusPill status={pack.status} />
                    </div>
                    {copy.desc && <div className={tabStyles.rowDesc}>{copy.desc}</div>}
                  </div>
                );
              })
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
