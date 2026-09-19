import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { listExamples, type ExampleSummary } from '../../api/rest';
import { insertExample, openExampleInNewTab } from '../../utils/openExample';
import {
  useLocalizedExamples,
  exampleMatches,
  type LocalizedExample,
} from '../../utils/localizeExamples';
import { useDialogStore } from '../../store/dialogStore';
import { selectPluginsById, usePluginStore } from '../../store/pluginStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import { pluginNameOf } from '../../utils/provider';
import {
  EXAMPLE_CATEGORY_COLORS,
  EXAMPLE_CATEGORY_FALLBACK,
  EXAMPLE_SECTION_COLORS,
  mixColor,
  NODE_HEADER_TINT,
  SURFACE_RAISED,
} from '../../styles/theme';
// `utils/exampleSections` (#141) owns the one grouping every example surface
// shares, so the modal's grid, the sidebar's list and the empty-canvas overlay
// are in the same sequence by construction rather than by three copies of the
// same sort staying in step.
import {
  exampleCategoryLabel,
  exampleChipLabel,
  flattenExampleSections,
  groupExamplesBySection,
} from '../../utils/exampleSections';
import styles from './TemplateGalleryModal.module.css';

function matches(example: LocalizedExample, query: string): boolean {
  // Name, category and description (in both the displayed language and the
  // original English) are the shared rule; the gallery adds `source` because
  // it is the only surface that prints which pack an example came from.
  return exampleMatches(example, query) || (example.source ?? '').toLowerCase().includes(query);
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

  const localized = useLocalizedExamples(examples);

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q ? localized.filter((e) => matches(e, q)) : localized;
  }, [localized, query]);

  // Grouped after the search, so a query that empties a section takes the
  // section's heading with it.
  const groups = useMemo(
    () =>
      flattenExampleSections(
        groupExamplesBySection(visible, (source) => pluginNameOf(pluginsById, source)),
        t,
      ),
    [pluginsById, t, visible],
  );

  // The detail pane always describes something as long as anything is
  // listed: a search that filters the chosen example away falls back to the
  // first remaining one rather than emptying the pane.
  const chosen =
    visible.find((e) => e.path === chosenPath) ?? visible[0] ?? null;
  // One question, one answer: the NAME is both the gate (null means the
  // example is a built-in, or names no plugin at all) and the word the pane
  // prints. `pluginNameOf` already falls back to the bare id while the
  // catalog has not answered, so there is no second fallback to keep in step.
  const chosenPluginName =
    chosen === null ? null : pluginNameOf(pluginsById, chosen.source);

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
              groups.map(({ category, label, sectionKey, items }) => {
                const color =
                  EXAMPLE_SECTION_COLORS[sectionKey] ?? EXAMPLE_CATEGORY_FALLBACK;
                return (
                  <section key={category} className={styles.section}>
                    <h3 className={styles.sectionTitle} style={{ color }}>
                      <span className={styles.sectionDot} style={{ background: color }} />
                      {label}
                      <span className={styles.sectionCount}>{items.length}</span>
                    </h3>
                    <div className={styles.cards}>
                      {items.map((example) => {
                        // Per card, not per section: a section holds several
                        // categories now, and the chip is what says which.
                        const chipColor =
                          EXAMPLE_CATEGORY_COLORS[example.category] ?? EXAMPLE_CATEGORY_FALLBACK;
                        // Fill is the hue mixed into --surface-raised (the same
                        // tint node headers use), not an alpha wash over an
                        // unknown backdrop — that pattern is what measured
                        // 2.24:1 on these badges. See check-contrast.mjs 8b.
                        const chipFill = mixColor(SURFACE_RAISED, chipColor, NODE_HEADER_TINT);
                        return (
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
                              {/* No chip when the group heading right above is
                                  that same word (an architecture family). */}
                              {exampleChipLabel(example) !== label && (
                                <span
                                  className={styles.cardChip}
                                  // Hue on the border (a graphic, 3:1) and in the
                                  // fill; the label is text and takes the text
                                  // tier. The hue on its own tint cannot reach
                                  // 4.5:1.
                                  style={{ borderColor: chipColor, background: chipFill }}
                                >
                                  {exampleChipLabel(example)}
                                </span>
                              )}
                              <span className={styles.cardCount}>
                                {t('empty.nodeCount', { count: example.node_count })}
                              </span>
                            </span>
                          </button>
                        );
                      })}
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
                    {chosenPluginName === null
                      ? t('gallery.sourceBuiltin')
                      : t('gallery.sourcePlugin', { plugin: chosenPluginName })}
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
