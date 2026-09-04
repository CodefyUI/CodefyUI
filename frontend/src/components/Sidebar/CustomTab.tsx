import { useCallback, useEffect, useState } from 'react';
import {
  listCustomNodes,
  type CustomNodeInfo,
  type PackSummary,
  type PluginCatalogEntry,
} from '../../api/rest';
import { useI18n } from '../../i18n';
import { usePackStore } from '../../store/packStore';
import { usePluginStore } from '../../store/pluginStore';
import { useUIStore } from '../../store/uiStore';
import { CustomNodeManager } from '../CustomNodeManager/CustomNodeManager';
import { StatusPill } from '../PackCenter/PackCard';
import { isInstalledStatus } from '../PluginCenter/pluginStatus';
import { catalogKey, localizedPackTitle } from '../../utils/packAvailability';
import { Pill } from '../shared/Pill';
import { RefreshIcon } from '../shared/Icons';
import styles from './NodePalette.module.css';
import tabStyles from './CustomTab.module.css';

type PackStoreState = ReturnType<typeof usePackStore.getState>;
type PluginStoreState = ReturnType<typeof usePluginStore.getState>;

// Module-scope selectors, so each subscription compares the SAME function's
// output frame to frame, and one slice per selector so an install writing its
// log and its per-step bytes on every long-poll turn cannot re-render this
// list. `packs` and `plugins` are replaced when a catalog is re-read or a
// status changes, which is exactly when these rows have something new to say.
const selectPacks = (state: PackStoreState): PackSummary[] => state.packs;
const selectPacksUnsupported = (state: PackStoreState): boolean => state.unsupported;
const selectPlugins = (state: PluginStoreState): PluginCatalogEntry[] => state.plugins;
const selectPluginsLoading = (state: PluginStoreState): boolean => state.loading;
const selectPluginsLoaded = (state: PluginStoreState): boolean => state.loaded;
const selectPluginsError = (state: PluginStoreState): string | null => state.error;
const selectPluginsUnsupported = (state: PluginStoreState): boolean => state.unsupported;

/**
 * Everything the user has added to this install: uploaded custom-node files
 * and installed plugins (#126).
 *
 * The custom-node rows are a read-only summary; enable/disable/upload/delete
 * stay in the existing `CustomNodeManager` modal, opened from here rather than
 * duplicated into a 250px column. Custom nodes are the one thing this tab
 * still fetches for itself, on mount and again when that modal closes (an
 * upload or a toggle changes what belongs here).
 *
 * The other two sections own no fetch at all. Both catalogs are read once by
 * the sidebar shell this tab lives in (`usePackCatalogBootstrap`,
 * `usePluginCatalogBootstrap`) and kept in `packStore` / `pluginStore`,
 * because an install is a download that outlives the tab that started it.
 * These rows are a pure view of that state, and the only way to change
 * anything is the Package Center or the Plugin Center — which is why all
 * three section headers now offer the same shape of button.
 */
export function CustomTab() {
  const [customNodes, setCustomNodes] = useState<CustomNodeInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [managerOpen, setManagerOpen] = useState(false);
  const { t } = useI18n();

  const packs = usePackStore(selectPacks);
  const packsUnsupported = usePackStore(selectPacksUnsupported);
  const openPackCenter = useUIStore((s) => s.openPackCenter);

  const plugins = usePluginStore(selectPlugins);
  const pluginsLoading = usePluginStore(selectPluginsLoading);
  const pluginsLoaded = usePluginStore(selectPluginsLoaded);
  const pluginsError = usePluginStore(selectPluginsError);
  const pluginsUnsupported = usePluginStore(selectPluginsUnsupported);
  const openPluginCenter = useUIStore((s) => s.openPluginCenter);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    listCustomNodes()
      .then(setCustomNodes)
      .catch((e: Error) => {
        setCustomNodes([]);
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

  // "Re-read everything this tab shows", which is both catalogs as well as
  // the files — and the retry for a boot read that never arrived. Mount does
  // NOT go through here: the shell has already read both catalogs by the time
  // this tab can be selected.
  const refreshAll = useCallback(() => {
    load();
    void usePackStore.getState().refresh();
    void usePluginStore.getState().refresh();
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

  // The same rule for plugins, over the half of the catalog that is actually
  // HERE. The full catalog also lists what could be installed, and this
  // section has always answered "what have I got" — with a chip that says
  // enabled or disabled, which is not a sentence about a plugin nobody has
  // downloaded. The rest of the catalog is the Plugin Center's to show.
  const visiblePlugins = pluginsUnsupported
    ? []
    : plugins.filter((plugin) => isInstalledStatus(plugin.status));

  // One spinner and one error line for the tab, as before: the plugin catalog
  // was fetched here until the store took it over, and hiding half a tab
  // behind a state the other half is not in reads as a broken list. A REFRESH
  // over a catalog already on screen is not a load, hence `!pluginsLoaded` —
  // only a first read blanks the tab.
  const busy = loading || (pluginsLoading && !pluginsLoaded);
  // The plugin store's `error` is sticky until the next catalog lands, and it
  // belongs to a SHARED store: a refresh that fails inside the Plugin Center
  // would otherwise replace this whole tab -- custom nodes and packs included
  // -- with "Failed to load". Only a catalog that has never arrived is this
  // tab's business, which is the store's own rule for the rows it keeps.
  const failed = error ?? (pluginsLoaded ? null : pluginsError);

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
        {busy && <div className={styles.stateMessage}>{t('customNodes.loading')}</div>}

        {!busy && failed && (
          <div className={styles.errorWrapper}>
            <div className={styles.errorText}>
              {t('customTab.loadFail', { error: failed })}
            </div>
            <button type="button" onClick={refreshAll} className={styles.retryButton}>
              {t('palette.retry')}
            </button>
          </div>
        )}

        {!busy && !failed && (
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
                    <Pill tone={file.enabled ? 'success' : 'neutral'}>
                      {file.enabled ? t('customNodes.enabled') : t('customNodes.disabled')}
                    </Pill>
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

            {/* ── Plugins ── */}
            <div className={tabStyles.sectionHeader}>
              <span className={tabStyles.sectionTitle}>{t('customTab.section.plugins')}</span>
              <span className={tabStyles.sectionCount}>{visiblePlugins.length}</span>
              <button
                type="button"
                className={tabStyles.manageButton}
                onClick={() => openPluginCenter()}
              >
                {t('customTab.plugins.open')}
              </button>
            </div>

            {/* No hint under this one, unlike the packs section above it: the
                header button one line up IS the Plugin Center, so a sentence
                whose only content is that destination says the same thing
                twice in sixty pixels of column. The packs hint earns its line
                by saying what a pack is. */}
            {visiblePlugins.length === 0 ? (
              <div className={tabStyles.sectionEmpty}>
                <div>{t('customTab.plugins.empty')}</div>
              </div>
            ) : (
              visiblePlugins.map((plugin) => (
                <div key={plugin.id} className={tabStyles.row} data-disabled={!plugin.enabled}>
                  <div className={tabStyles.rowTitle}>
                    <span className={tabStyles.rowName}>{plugin.name}</span>
                    {plugin.version && (
                      <span className={tabStyles.rowVersion}>v{plugin.version}</span>
                    )}
                    {/* Enabled or not, rather than the panel's six-state
                        status: everything listed here is installed, so the
                        one thing left to say is whether it is switched on. */}
                    <Pill tone={plugin.enabled ? 'success' : 'neutral'}>
                      {plugin.enabled ? t('customNodes.enabled') : t('customNodes.disabled')}
                    </Pill>
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
