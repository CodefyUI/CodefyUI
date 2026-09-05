import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { listExamples, type ExampleSummary } from '../../api/rest';
import { insertExample, openExampleInNewTab } from '../../utils/openExample';
import { useDialogStore } from '../../store/dialogStore';
import { usePluginStore } from '../../store/pluginStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import { pluginNameOf, type PluginIndex } from '../../utils/provider';
import { EXAMPLE_CATEGORY_COLORS, EXAMPLE_CATEGORY_FALLBACK, mixColor, NODE_HEADER_TINT, SURFACE_RAISED } from '../../styles/theme';
// The sidebar's Templates tab (#126) already owns the category order every
// example surface is expected to share; importing it keeps the modal's grid
// and the tab's list in the same sequence by construction rather than by
// two copies of the same sort staying in step.
import {
  exampleCategoryLabel,
  groupExamplesByCategory,
} from '../Sidebar/TemplatesTab';
import styles from './TemplateGalleryModal.module.css';

// Module-scope, so the subscription compares the same function's output frame
// to frame, and narrow: an install running in the Plugin Center writes `job`
// and its log on every long-poll turn, none of which renames a plugin.
type PluginStoreState = ReturnType<typeof usePluginStore.getState>;
const selectPluginsById = (state: PluginStoreState): PluginIndex => state.byId;

/**
 * Whether an example came from a plugin, and which one — as an id.
 *
 * Deliberately still the raw id and still pure: it is the GATE the detail pane
 * asks (plugin, or built-in?), and the human name it prints comes from
 * `pluginNameOf` at the render site, which needs the catalog and a live
 * subscription to it. Keeping the two apart leaves this callable, and testable,
 * without a store.
 */
export function exampleSourceLabel(example: ExampleSummary): string | null {
  const source = example.source ?? '';
  if (source.startsWith('plugin:')) return source.slice('plugin:'.length);
  return null;
}

function matches(example: ExampleSummary, query: string): boolean {
  return (
    example.name.toLowerCase().includes(query) ||
    example.description.toLowerCase().includes(query) ||
    example.category.toLowerCase().includes(query) ||
    (example.source ?? '').toLowerCase().includes(query)
  );
}

/**
 * The full example browser (core#128).
 *
 * Reachable at any time — toolbar, sidebar Templates tab, empty-canvas
 * overlay — which is the whole point: before this, the ~30 shipped examples
 * were visible only on an empty canvas, so the moment you had a graph they
 * became unreachable.
 *
 * Two ways to take one: open it in a NEW tab (leaving the current graph
 * alone), or insert it into the CURRENT canvas, which remaps every incoming
 * id and drops the block clear of what is already there.
 *
 * Mounted once at the app root and driven by `uiStore.templateGalleryOpen`.
 */
export function TemplateGalleryModal() {
  const open = useUIStore((s) => s.templateGalleryOpen);
  if (!open) return null;
  return <TemplateGalleryBody />;
}

function TemplateGalleryBody() {
  const close = useUIStore((s) => s.closeTemplateGallery);
  const pluginsById = usePluginStore(selectPluginsById);
  const { t } = useI18n();

  const [examples, setExamples] = useState<ExampleSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [chosenPath, setChosenPath] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const surfaceRef = useRef<HTMLDivElement | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    listExamples()
      .then(setExamples)
      .catch((e: Error) => {
        setExamples([]);
        setError(e.message);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // Move focus onto the surface so the keyboard starts inside the modal
  // rather than wherever it was on the page behind, and hand it back on
  // close — the way the node detail modal (#127) does. This is NOT a focus
  // trap: Tab can still walk out into the page underneath. Left that way on
  // purpose, so the two modals behave identically; trapping is worth doing,
  // but as one change to both rather than a divergence here.
  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    const timer = setTimeout(() => surfaceRef.current?.focus(), 0);
    return () => {
      clearTimeout(timer);
      if (previouslyFocused && previouslyFocused.isConnected) {
        previouslyFocused.focus();
      }
    };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      // Something can sit ON TOP of the gallery — a confirm/prompt dialog, or
      // the shortcuts modal, which `?` opens over whatever is showing and
      // which has no Escape handler of its own. Closing the surface
      // UNDERNEATH the one the user is looking at is the bug this guards.
      // Same check `useKeyboardShortcuts` makes before answering Enter.
      if (useDialogStore.getState().active !== null) return;
      if (useUIStore.getState().shortcutsModalOpen) return;
      // The Package Center is the third surface that renders over this one.
      if (useUIStore.getState().packCenterOpen) return;
      e.preventDefault();
      close();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [close]);

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q ? examples.filter((e) => matches(e, q)) : examples;
  }, [examples, query]);

  const groups = useMemo(() => groupExamplesByCategory(visible), [visible]);

  // The detail pane always describes something as long as anything is
  // listed: a search that filters the chosen example away falls back to the
  // first remaining one rather than emptying the pane.
  const chosen =
    visible.find((e) => e.path === chosenPath) ?? visible[0] ?? null;
  const chosenSourceId = chosen === null ? null : exampleSourceLabel(chosen);

  const take = useCallback(
    async (run: () => Promise<boolean>) => {
      setBusy(true);
      try {
        // Closing regardless of outcome: a failure has already surfaced its
        // own toast, and leaving the modal up over it just hides the message.
        await run();
      } finally {
        setBusy(false);
        close();
      }
    },
    [close],
  );

  // `busy` gates this as well as the detail buttons: a double-click lands two
  // events, and the second must not start a second load (two tabs, or a tab
  // opened after the modal has already closed).
  const openInNewTab = useCallback(
    (path: string) => {
      if (busy) return;
      void take(() => openExampleInNewTab(path));
    },
    [busy, take],
  );

  return createPortal(
    <div
      className={styles.backdrop}
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) close();
      }}
    >
      <div
        ref={surfaceRef}
        className={styles.modal}
        role="dialog"
        aria-modal="true"
        aria-label={t('gallery.title')}
        tabIndex={-1}
      >
        <div className={styles.header}>
          <div className={styles.titleBlock}>
            <div className={styles.title}>{t('gallery.title')}</div>
            <div className={styles.subtitle}>{t('gallery.subtitle')}</div>
          </div>
          <input
            type="search"
            className={styles.search}
            placeholder={t('gallery.search')}
            aria-label={t('gallery.search')}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button
            type="button"
            className={styles.closeBtn}
            onClick={close}
            title={t('gallery.close')}
            aria-label={t('gallery.close')}
          >
            &#215;
          </button>
        </div>

        <div className={styles.body}>
          {/* A named <section> so the card list is one addressable region,
              distinct from the detail pane that repeats the chosen name. */}
          <section className={styles.grid} aria-label={t('gallery.list')}>
            {loading && <div className={styles.stateMessage}>{t('templates.loading')}</div>}

            {!loading && error && (
              <div className={styles.stateMessage}>
                <div className={styles.errorText}>
                  {t('templates.loadFail', { error })}
                </div>
                <button type="button" className={styles.retryBtn} onClick={load}>
                  {t('palette.retry')}
                </button>
              </div>
            )}

            {!loading && !error && groups.length === 0 && (
              <div className={styles.stateMessage}>
                {query ? t('templates.noMatch') : t('templates.empty')}
              </div>
            )}

            {!loading && !error &&
              groups.map(({ category, items }) => {
                const color =
                  EXAMPLE_CATEGORY_COLORS[category] ?? EXAMPLE_CATEGORY_FALLBACK;
                // Badge fill is the category hue mixed into --surface-raised
                // (the same tint node headers use), not an alpha wash over an
                // unknown backdrop — that pattern is what measured 2.24:1 on
                // these badges. See scripts/check-contrast.mjs section 8b.
                const chipFill = mixColor(SURFACE_RAISED, color, NODE_HEADER_TINT);
                return (
                  <section key={category} className={styles.section}>
                    <h3 className={styles.sectionTitle} style={{ color }}>
                      <span className={styles.sectionDot} style={{ background: color }} />
                      {exampleCategoryLabel(category)}
                      <span className={styles.sectionCount}>{items.length}</span>
                    </h3>
                    <div className={styles.cards}>
                      {items.map((example) => (
                        <button
                          key={example.path}
                          type="button"
                          className={`${styles.card} ${
                            chosen?.path === example.path ? styles.cardActive : ''
                          }`}
                          aria-pressed={chosen?.path === example.path}
                          onClick={() => setChosenPath(example.path)}
                          onDoubleClick={() => openInNewTab(example.path)}
                        >
                          <span className={styles.cardName}>{example.name}</span>
                          <span className={styles.cardDesc}>{example.description}</span>
                          <span className={styles.cardFooter}>
                            <span
                              className={styles.cardChip}
                              // Hue on the border (a graphic, 3:1) and in the
                              // fill; the label is text and takes the text tier.
                              // The hue on its own tint cannot reach 4.5:1.
                              style={{ borderColor: color, background: chipFill }}
                            >
                              {exampleCategoryLabel(category)}
                            </span>
                            <span className={styles.cardCount}>
                              {t('empty.nodeCount', { count: example.node_count })}
                            </span>
                          </span>
                        </button>
                      ))}
                    </div>
                  </section>
                );
              })}
          </section>

          <aside className={styles.detail} aria-label={t('gallery.detail')}>
            {chosen === null ? (
              <div className={styles.detailEmpty}>{t('gallery.detailEmpty')}</div>
            ) : (
              <>
                {/* Only the prose scrolls. The actions below stay pinned, so a
                    long description can never push "Open in new tab" out of
                    reach on a short window. */}
                <div className={styles.detailScroll}>
                  <div className={styles.detailName}>{chosen.name}</div>
                  <div className={styles.detailMeta}>
                    <span>{exampleCategoryLabel(chosen.category)}</span>
                    <span>{t('empty.nodeCount', { count: chosen.node_count })}</span>
                    <span>{t('gallery.edgeCount', { count: chosen.edge_count })}</span>
                  </div>
                  <div className={styles.detailSource}>
                    {/* The id is still the gate; the name is what a reader
                        sees, and falls back to that same id for as long as the
                        catalog has not answered. */}
                    {chosenSourceId
                      ? t('gallery.sourcePlugin', {
                          plugin: pluginNameOf(pluginsById, chosen.source) ?? chosenSourceId,
                        })
                      : t('gallery.sourceBuiltin')}
                  </div>
                  <p className={styles.detailDesc}>
                    {chosen.description || t('gallery.noDescription')}
                  </p>
                </div>
                <div className={styles.detailActions}>
                  <button
                    type="button"
                    className={styles.primaryBtn}
                    disabled={busy}
                    onClick={() => openInNewTab(chosen.path)}
                  >
                    {t('gallery.openNewTab')}
                  </button>
                  <button
                    type="button"
                    className={styles.secondaryBtn}
                    disabled={busy}
                    onClick={() => void take(() => insertExample(chosen.path))}
                  >
                    {t('gallery.insert')}
                  </button>
                  <div className={styles.detailHint}>{t('gallery.insertHint')}</div>
                </div>
              </>
            )}
          </aside>
        </div>
      </div>
    </div>,
    document.body,
  );
}
