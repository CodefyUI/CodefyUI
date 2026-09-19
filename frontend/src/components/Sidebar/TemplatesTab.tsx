import { useCallback, useEffect, useMemo, useState } from 'react';
import { listExamples, type ExampleSummary } from '../../api/rest';
import { insertExample } from '../../utils/openExample';
import {
  useLocalizedExamples,
  exampleMatches,
  type LocalizedExample,
} from '../../utils/localizeExamples';
import { selectPluginsById, usePluginStore } from '../../store/pluginStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import {
  flattenExampleSections,
  groupExamplesBySection,
} from '../../utils/exampleSections';
import { pluginNameOf } from '../../utils/provider';
import { EXAMPLE_CATEGORY_FALLBACK, EXAMPLE_SECTION_COLORS } from '../../styles/theme';
import { RefreshIcon } from '../shared/Icons';
import { CategoryList } from './CategoryList';
import styles from './NodePalette.module.css';
import tabStyles from './TemplatesTab.module.css';

// ── Example item ──

/**
 * One example, as a drag source (#348).
 *
 * Both gestures INSERT: dragging drops the block where the pointer was
 * released, clicking puts it below the graph already on the canvas. Neither
 * replaces anything, and either is one Ctrl+Z from the canvas as it was.
 *
 * This used to call `openExample`, which routes to `loadGraphDocument` — an
 * action that swaps out the tab's nodes, edges, subgraphs, description and
 * save binding in one commit and, by design, pushes no undo frame. So a
 * stray click on a list of ~30 examples destroyed however long the user had
 * spent on the graph, with nothing to undo. Nothing about the sidebar
 * suggested that; the hint underneath it read "Click an example to open it".
 *
 * A `<div>` carrying button semantics rather than a real `<button>`, which is
 * what this was: `draggable` is reliable on a plain div in every engine (it
 * is how the Nodes and Presets tabs have always dragged), and is not on a
 * button — Firefox has ignored the attribute there for years. The semantics
 * a button was providing are restored explicitly, because clicking is a real
 * action here and the drag has to stay an enhancement rather than the only
 * way in: `role`, `tabIndex`, and Enter/Space activation.
 */
function ExampleItem({ example }: { example: LocalizedExample }) {
  const { t } = useI18n();
  const insert = () => void insertExample(example.path);
  const handleDragStart = (event: React.DragEvent) => {
    // The path, not the resolved graph: the drop handler does the fetch, so
    // opening the Templates tab does not pull ~30 example files off the
    // server just in case one of them gets dragged.
    event.dataTransfer.setData('application/codefyui-example', example.path);
    event.dataTransfer.effectAllowed = 'move';
  };
  return (
    <div
      role="button"
      tabIndex={0}
      draggable
      onDragStart={handleDragStart}
      className={tabStyles.exampleItem}
      onClick={insert}
      onKeyDown={(event) => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        // Space would scroll the panel out from under the list otherwise,
        // which is the default a real <button> was suppressing for us.
        event.preventDefault();
        insert();
      }}
      title={example.description}
    >
      <div className={tabStyles.exampleName}>{example.name}</div>
      {example.description && (
        <div className={tabStyles.exampleDesc}>{example.description}</div>
      )}
      <div className={tabStyles.exampleMeta}>
        {t('empty.nodeCount', { count: example.node_count })}
      </div>
    </div>
  );
}

// ── Templates tab ──

/**
 * Builtin and plugin-shipped examples, listed straight from
 * `GET /api/examples/list` (#126).
 *
 * Deliberately thumbnail-less: this is the always-available list view, and the
 * richer gallery is the empty-canvas overlay's job — and core#128's modal,
 * which the footer button below opens. All three group through
 * `utils/exampleSections` (#141), so the same example sits in the same place
 * wherever it is opened from; this tab and the modal also share
 * `insertExample` (#348) so an example joins the canvas the same way from
 * either.
 *
 * Examples are fetched per mount rather than cached in a store: the sidebar
 * only mounts this tab while it is the selected one, and the list changes
 * whenever a plugin is installed or enabled — a manual refresh button covers
 * the case where that happens while the tab is open.
 */
export function TemplatesTab() {
  const [examples, setExamples] = useState<ExampleSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const { t } = useI18n();
  const pluginsById = usePluginStore(selectPluginsById);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    listExamples()
      .then((all) => setExamples(all))
      .catch((e: Error) => {
        setExamples([]);
        setError(e.message);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const localized = useLocalizedExamples(examples);

  // Filter first, group second: a search that empties a section drops the
  // section with it rather than leaving a header over nothing.
  const groups = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    const filtered = q ? localized.filter((e) => exampleMatches(e, q)) : localized;
    const sections = groupExamplesBySection(filtered, (source) =>
      pluginNameOf(pluginsById, source),
    );
    return flattenExampleSections(sections, t);
  }, [localized, pluginsById, searchQuery, t]);

  // `CategoryList` addresses a group by its key alone — collapse state, the
  // jump index and the scroll target all hang off it — so the label and the
  // accent are looked up rather than passed down.
  const byKey = useMemo(() => new Map(groups.map((g) => [g.category, g])), [groups]);

  return (
    <>
      <div className={styles.header}>
        <div className={styles.headerRow}>
          <div className={styles.headerTitle}>{t('sidebar.tab.templates')}</div>
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
        <input
          type="text"
          placeholder={t('templates.search')}
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className={styles.searchInput}
        />
      </div>

      <div className={styles.panelBody}>
        {loading && <div className={styles.stateMessage}>{t('templates.loading')}</div>}

        {!loading && error && (
          <div className={styles.errorWrapper}>
            <div className={styles.errorText}>{t('templates.loadFail', { error })}</div>
            <button type="button" onClick={load} className={styles.retryButton}>
              {t('palette.retry')}
            </button>
          </div>
        )}

        {!loading && !error && (
          groups.length === 0 ? (
            <div className={styles.stateMessageMuted}>
              {searchQuery ? t('templates.noMatch') : t('templates.empty')}
            </div>
          ) : (
            <CategoryList
              groups={groups}
              itemKey={(example) => example.path}
              renderItem={(example) => <ExampleItem example={example} />}
              colorFor={(key) => {
                const section = byKey.get(key)?.sectionKey;
                return (
                  (section && EXAMPLE_SECTION_COLORS[section]) ?? EXAMPLE_CATEGORY_FALLBACK
                );
              }}
              labelFor={(key) => byKey.get(key)?.label ?? key}
            />
          )
        )}
      </div>

      <div className={styles.footer}>
        <button
          type="button"
          className={tabStyles.browseButton}
          onClick={() => useUIStore.getState().openTemplateGallery()}
          title={t('gallery.open.title')}
        >
          {t('gallery.browse')}
        </button>
      </div>
    </>
  );
}
