import { useCallback, useEffect, useMemo, useState } from 'react';
import { listExamples, type ExampleSummary } from '../../api/rest';
import { openExample } from '../../utils/openExample';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import { EXAMPLE_CATEGORY_COLORS, EXAMPLE_CATEGORY_FALLBACK } from '../../styles/theme';
import { RefreshIcon } from '../shared/Icons';
import { CategoryList, type CategoryGroup } from './CategoryList';
import styles from './NodePalette.module.css';
import tabStyles from './TemplatesTab.module.css';

/** Beginner-facing usage examples lead; the reference architectures (which are
 * illustrative rather than runnable starting points) sit at the bottom. Every
 * other category — builtin or plugin-shipped — sorts alphabetically between
 * them. */
const FIRST_CATEGORY = 'Usage_Example';
const LAST_CATEGORY = 'Model_Architecture';

function categoryRank(category: string): number {
  if (category === FIRST_CATEGORY) return 0;
  if (category === LAST_CATEGORY) return 2;
  return 1;
}

export function exampleCategoryLabel(category: string): string {
  return category.replace(/_/g, ' ');
}

/** Group the flat `/api/examples/list` payload by category, in display order. */
export function groupExamplesByCategory(
  examples: ExampleSummary[],
): CategoryGroup<ExampleSummary>[] {
  const byCategory = new Map<string, ExampleSummary[]>();
  for (const example of examples) {
    const bucket = byCategory.get(example.category);
    if (bucket) bucket.push(example);
    else byCategory.set(example.category, [example]);
  }
  return [...byCategory.entries()]
    .map(([category, items]) => ({ category, items }))
    .sort(
      (a, b) =>
        categoryRank(a.category) - categoryRank(b.category) ||
        a.category.localeCompare(b.category),
    );
}

// ── Example item ──

function ExampleItem({ example }: { example: ExampleSummary }) {
  const { t } = useI18n();
  return (
    <button
      type="button"
      className={tabStyles.exampleItem}
      onClick={() => void openExample(example.path)}
      title={example.description}
    >
      <div className={tabStyles.exampleName}>{example.name}</div>
      {example.description && (
        <div className={tabStyles.exampleDesc}>{example.description}</div>
      )}
      <div className={tabStyles.exampleMeta}>
        {t('empty.nodeCount', { count: example.node_count })}
      </div>
    </button>
  );
}

// ── Templates tab ──

/**
 * Builtin and plugin-shipped examples, listed straight from
 * `GET /api/examples/list` (#126).
 *
 * Deliberately thumbnail-less: this is the always-available list view, and the
 * richer gallery is the empty-canvas overlay's job — and core#128's modal,
 * which the footer button below opens. Both call the same `openExample`
 * helper this does, so an example opens the same way from every surface, and
 * both group by category through `groupExamplesByCategory` (exported for
 * exactly that reason) so the two never drift out of order.
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

  const groups = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    const filtered = q
      ? examples.filter(
          (e) =>
            e.name.toLowerCase().includes(q) ||
            e.description.toLowerCase().includes(q) ||
            e.category.toLowerCase().includes(q),
        )
      : examples;
    return groupExamplesByCategory(filtered);
  }, [examples, searchQuery]);

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
              colorFor={(category) =>
                EXAMPLE_CATEGORY_COLORS[category] ?? EXAMPLE_CATEGORY_FALLBACK}
              labelFor={exampleCategoryLabel}
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
        <div className={tabStyles.footerHint}>{t('templates.hint')}</div>
      </div>
    </>
  );
}
