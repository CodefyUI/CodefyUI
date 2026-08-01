import { Fragment, useCallback, useRef, useState, type ReactNode } from 'react';
import { useI18n } from '../../i18n';
import { CATEGORY_COLORS } from '../../styles/theme';
import { CollapseAllIcon, ExpandAllIcon } from '../shared/Icons';
import styles from './NodePalette.module.css';

export interface CategoryGroup<T> {
  category: string;
  items: T[];
}

interface CategoryListProps<T> {
  groups: CategoryGroup<T>[];
  itemKey: (item: T) => string;
  renderItem: (item: T) => ReactNode;
  /** Accent per category; defaults to the shared node-category palette. */
  colorFor?: (category: string) => string;
  /** Display name per category; defaults to the raw key. */
  labelFor?: (category: string) => string;
}

const defaultColorFor = (category: string) => CATEGORY_COLORS[category] ?? '#607D8B';

/**
 * The accordion body shared by every list-shaped sidebar tab (#126): one
 * collapsible section per category, an expand-all/collapse-all pair, and the
 * mini jump index pinned to the panel's right edge.
 *
 * Collapse state is tracked as the set of COLLAPSED categories rather than
 * expanded ones, so a category that appears later (a plugin reload, a cleared
 * search) starts expanded like every other — the pre-#126 default.
 *
 * The toolbar and jump index only appear with more than one category: with a
 * single section there is nothing to jump between and nothing that "all"
 * usefully means.
 */
export function CategoryList<T>({
  groups,
  itemKey,
  renderItem,
  colorFor = defaultColorFor,
  labelFor = (category) => category,
}: CategoryListProps<T>) {
  const [collapsed, setCollapsed] = useState<ReadonlySet<string>>(() => new Set());
  const sectionRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const { t } = useI18n();

  const toggle = useCallback((category: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (!next.delete(category)) next.add(category);
      return next;
    });
  }, []);

  const expandAll = useCallback(() => setCollapsed(new Set()), []);

  const collapseAll = useCallback(
    () => setCollapsed(new Set(groups.map((g) => g.category))),
    [groups],
  );

  // Jumping to a collapsed section would scroll to a header with nothing under
  // it, so the jump expands it first.
  const jumpTo = useCallback((category: string) => {
    setCollapsed((prev) => {
      if (!prev.has(category)) return prev;
      const next = new Set(prev);
      next.delete(category);
      return next;
    });
    sectionRefs.current[category]?.scrollIntoView({ block: 'start' });
  }, []);

  const showIndex = groups.length > 1;

  return (
    <div className={styles.listWrap}>
      {showIndex && (
        <div className={styles.listToolbar}>
          <button
            type="button"
            className={styles.toolbarButton}
            onClick={expandAll}
            aria-label={t('sidebar.expandAll')}
            title={t('sidebar.expandAll')}
          >
            <ExpandAllIcon size={13} />
          </button>
          <button
            type="button"
            className={styles.toolbarButton}
            onClick={collapseAll}
            aria-label={t('sidebar.collapseAll')}
            title={t('sidebar.collapseAll')}
          >
            <CollapseAllIcon size={13} />
          </button>
        </div>
      )}

      <div className={styles.listBody}>
        <div className={styles.content}>
          {groups.map(({ category, items }) => {
            const color = colorFor(category);
            const expanded = !collapsed.has(category);
            return (
              <div
                key={category}
                // React 19 ref cleanup: drop the entry when the section goes
                // away (a search that filters it out, a plugin unload) so the
                // map cannot accumulate detached nodes for the session.
                ref={(el) => {
                  sectionRefs.current[category] = el;
                  return () => {
                    delete sectionRefs.current[category];
                  };
                }}
                className={styles.categorySection}
              >
                <button
                  type="button"
                  onClick={() => toggle(category)}
                  className={styles.categoryButton}
                  style={{ borderBottom: `2px solid ${color}` }}
                  aria-expanded={expanded}
                >
                  <span className={styles.categoryDot} style={{ background: color }} />
                  <span className={styles.categoryName} style={{ color }}>
                    {labelFor(category)}
                  </span>
                  <span className={styles.categoryCount}>{items.length}</span>
                  <span className={styles.categoryChevron}>{expanded ? '▾' : '▸'}</span>
                </button>

                {expanded && (
                  <div className={styles.categoryNodes}>
                    {items.map((item) => (
                      <Fragment key={itemKey(item)}>{renderItem(item)}</Fragment>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {showIndex && (
          <nav className={styles.jumpList} aria-label={t('sidebar.jumpTo')}>
            {groups.map(({ category }) => (
              <button
                key={category}
                type="button"
                className={styles.jumpDot}
                onClick={() => jumpTo(category)}
                aria-label={labelFor(category)}
                title={labelFor(category)}
              >
                <span
                  className={styles.jumpDotInner}
                  style={{ background: colorFor(category) }}
                />
              </button>
            ))}
          </nav>
        )}
      </div>
    </div>
  );
}
