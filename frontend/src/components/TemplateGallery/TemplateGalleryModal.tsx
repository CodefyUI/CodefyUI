import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { insertExample, openExampleInNewTab } from '../../utils/openExample';
import {
  useLocalizedExamples,
  exampleMatches,
  type LocalizedExample,
} from '../../utils/localizeExamples';
import { useDialogStore } from '../../store/dialogStore';
import { selectPluginsById, usePluginStore } from '../../store/pluginStore';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import { pluginNameOf } from '../../utils/provider';
import {
  EXAMPLE_CATEGORY_FALLBACK,
  EXAMPLE_SECTION_COLORS,
  mixColor,
  NODE_HEADER_TINT,
  SURFACE_RAISED,
} from '../../styles/theme';
import { useExampleCatalog } from '../../utils/useExampleCatalog';
import { ALL_MAIN_TAGS, MainTagChips, mainTagTabId } from '../shared/MainTagChips';
// `utils/exampleSections` (#141) owns the one grouping every example surface
// shares, so the modal's grid, the sidebar's list and the empty-canvas overlay
// are in the same sequence by construction rather than by three copies of the
// same sort staying in step.
import {
  cardTags,
  exampleCategoryLabel,
  flattenExampleSections,
  freeTagsOf,
  groupExamplesBySection,
  mainTagLabel,
  mainTagOf,
  tagLabel,
} from '../../utils/exampleSections';

/** A tag is offered in the filter row once at least this many examples carry
 * it; the rest sit behind "More tags" (phase 3 review D). A filter button that
 * leaves one card is a search, not a filter. */
export const TAG_FILTER_MIN_COUNT = 3;
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
  // Whether there is a graph for Insert to merge into; see the button.
  const hasActiveTab = useTabStore((s) => s.tabs.some((t) => t.id === s.activeTabId));
  const { t } = useI18n();

  const catalog = useExampleCatalog();
  const { examples, loading, error, reload: load } = catalog;
  const [query, setQuery] = useState('');
  // The modal browses everything, so it opens on All rather than on the
  // empty-canvas browser's remembered chip.
  const [mainTag, setMainTag] = useState<string>(ALL_MAIN_TAGS);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [tagsOpen, setTagsOpen] = useState(false);
  const [moreTagsOpen, setMoreTagsOpen] = useState(false);
  const chipsPrefix = useId();
  const gridId = useId();
  const tagsRegionId = useId();
  const [chosenPath, setChosenPath] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const surfaceRef = useRef<HTMLDivElement | null>(null);

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
      // A confirm/prompt dialog can sit ON TOP of the gallery. Closing the
      // surface UNDERNEATH the one the user is looking at is the bug this
      // guards. (The shortcuts sheet stops its own Escape before it gets
      // here; see the stack policy in `shared/ShortcutsModal.tsx`.)
      if (useDialogStore.getState().active !== null) return;
      // The Package Center is the third surface that renders over this one.
      if (useUIStore.getState().packCenterOpen) return;
      e.preventDefault();
      close();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [close]);

  const localized = useLocalizedExamples(examples);

  // A main tag the list no longer has (a pack was removed) reads as All.
  const order = useMemo(() => catalog.mainTags.map((entry) => entry.id), [catalog.mainTags]);
  const activeMainTag = mainTag === ALL_MAIN_TAGS || order.includes(mainTag) ? mainTag : ALL_MAIN_TAGS;

  // The three filters combine: main tag, then every selected tag (an
  // intersection), then the search text.
  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return localized.filter(
      (e) =>
        (activeMainTag === ALL_MAIN_TAGS || mainTagOf(e) === activeMainTag) &&
        selectedTags.every((tag) => freeTagsOf(e).includes(tag)) &&
        (!q || matches(e, q)),
    );
  }, [localized, query, activeMainTag, selectedTags]);

  // Grouped after filtering, so a filter that empties a section takes the
  // section's heading with it.
  const groups = useMemo(
    () =>
      flattenExampleSections(
        groupExamplesBySection(visible, (source) => pluginNameOf(pluginsById, source), order),
        t,
      ),
    [pluginsById, t, visible, order],
  );

  // The filter row: tags carried by at least TAG_FILTER_MIN_COUNT examples,
  // most used first, plus any selected tag so a selection never hides; the
  // rest behind "More tags".
  const frequentTags = catalog.tags.filter(
    (tag) => tag.count >= TAG_FILTER_MIN_COUNT || selectedTags.includes(tag.id),
  );
  const rareTags = catalog.tags.filter(
    (tag) => tag.count < TAG_FILTER_MIN_COUNT && !selectedTags.includes(tag.id),
  );
  const filtering = activeMainTag !== ALL_MAIN_TAGS || selectedTags.length > 0 || query.trim() !== '';
  const toggleTag = (tag: string) =>
    setSelectedTags((current) =>
      current.includes(tag) ? current.filter((x) => x !== tag) : [...current, tag],
    );
  const applyTag = (tag: string) => {
    setSelectedTags((current) => (current.includes(tag) ? current : [...current, tag]));
    setTagsOpen(true);
  };
  const clearFilters = () => {
    setMainTag(ALL_MAIN_TAGS);
    setSelectedTags([]);
    setQuery('');
  };

  // The detail pane always describes something as long as anything is
  // listed: a search that filters the chosen example away falls back to the
  // first card on screen rather than emptying the pane. The fallback reads
  // `groups` because the cards render in that sectioned order; `visible` is
  // in API order, and its head is usually a card further down (#678).
  const chosen =
    visible.find((e) => e.path === chosenPath) ?? groups[0]?.items[0] ?? null;
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
            className={`${styles.tagToggle} ${selectedTags.length > 0 ? styles.tagToggleActive : ''}`}
            aria-expanded={tagsOpen}
            aria-controls={tagsRegionId}
            onClick={() => setTagsOpen((open) => !open)}
          >
            {selectedTags.length > 0
              ? t('examples.filter.tagsCount', { count: selectedTags.length })
              : t('examples.filter.tags')}
          </button>
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

        {catalog.mainTags.length > 0 && (
          <div className={styles.filters}>
            <MainTagChips
              entries={catalog.mainTags}
              value={activeMainTag}
              onChange={setMainTag}
              idPrefix={chipsPrefix}
              panelId={gridId}
            />
            {tagsOpen && (
              <div id={tagsRegionId} className={styles.tagFilter} role="group" aria-label={t('examples.filter.tags')}>
                {frequentTags.map((tag) => (
                  <TagFilterButton key={tag.id} tag={tag.id} count={tag.count}
                    pressed={selectedTags.includes(tag.id)} onToggle={toggleTag} />
                ))}
                {moreTagsOpen &&
                  rareTags.map((tag) => (
                    <TagFilterButton key={tag.id} tag={tag.id} count={tag.count}
                      pressed={false} onToggle={toggleTag} />
                  ))}
                {rareTags.length > 0 && (
                  <button
                    type="button"
                    className={styles.moreTagsBtn}
                    aria-expanded={moreTagsOpen}
                    onClick={() => setMoreTagsOpen((open) => !open)}
                  >
                    {moreTagsOpen ? t('examples.filter.fewer') : t('examples.filter.more')}
                  </button>
                )}
              </div>
            )}
          </div>
        )}

        {/* The body is the panel the main-tag chips control: the cards and
            the detail pane both follow the selected chip. */}
        <div
          id={gridId}
          className={styles.body}
          {...(catalog.mainTags.length > 0
            ? { role: 'tabpanel', 'aria-labelledby': mainTagTabId(chipsPrefix, activeMainTag) }
            : {})}
        >
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

            {!loading && !error && groups.length === 0 && !filtering && (
              <div className={styles.stateMessage}>{t('templates.empty')}</div>
            )}

            {/* Nothing matches: say which filters are on, and offer to clear
                them all at once. */}
            {!loading && !error && groups.length === 0 && filtering && (
              <div className={styles.stateMessage}>
                <div className={styles.emptyTitle}>{t('examples.filter.emptyTitle')}</div>
                <ul className={styles.emptyFilters}>
                  {activeMainTag !== ALL_MAIN_TAGS && (
                    <li>{t('examples.filter.emptyMainTag', { name: mainTagLabel(activeMainTag, t) })}</li>
                  )}
                  {selectedTags.length > 0 && (
                    <li>{t('examples.filter.emptyTags', {
                      tags: selectedTags.map((tag) => tagLabel(tag, t)).join(', '),
                    })}</li>
                  )}
                  {query.trim() !== '' && (
                    <li>{t('examples.filter.emptyQuery', { query: query.trim() })}</li>
                  )}
                </ul>
                <button type="button" className={styles.retryBtn} onClick={clearFilters}>
                  {t('examples.filter.clear')}
                </button>
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
                        // The main tag's hue on the tag chips' border and in
                        // their fill; the label is text and takes the text
                        // tier (check-contrast.mjs 8b).
                        const hue =
                          EXAMPLE_SECTION_COLORS[mainTagOf(example)] ?? EXAMPLE_CATEGORY_FALLBACK;
                        const chipFill = mixColor(SURFACE_RAISED, hue, NODE_HEADER_TINT);
                        const { shown, more } = cardTags(example, 2);
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
                              {shown.map((tag) => (
                                <span
                                  key={tag}
                                  className={styles.cardChip}
                                  style={{ borderColor: hue, background: chipFill }}
                                >
                                  {tagLabel(tag, t)}
                                </span>
                              ))}
                              {more > 0 && (
                                <span
                                  className={styles.cardMore}
                                  title={t('examples.card.moreTagsTitle', {
                                    count: more,
                                    tags: freeTagsOf(example).slice(shown.length).map((tag) => tagLabel(tag, t)).join(', '),
                                  })}
                                >
                                  {t('examples.card.moreTags', { count: more })}
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
                    <span>{exampleCategoryLabel(chosen.category, t)}</span>
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
                  {/* Every tag, main tag first. Each one is a filter: the main
                      tag selects its chip, a free tag joins the tag filter. */}
                  <div className={styles.detailTagsTitle}>{t('gallery.tags')}</div>
                  <div className={styles.detailTags}>
                    <button
                      type="button"
                      className={styles.detailTag}
                      title={t('gallery.mainTagApply', { name: mainTagLabel(mainTagOf(chosen), t) })}
                      onClick={() => setMainTag(mainTagOf(chosen))}
                    >
                      {mainTagLabel(mainTagOf(chosen), t)}
                    </button>
                    {freeTagsOf(chosen).map((tag) => (
                      <button
                        key={tag}
                        type="button"
                        className={styles.detailTag}
                        aria-pressed={selectedTags.includes(tag)}
                        title={t('gallery.tagApply', { tag: tagLabel(tag, t) })}
                        onClick={() => applyTag(tag)}
                      >
                        {tagLabel(tag, t)}
                      </button>
                    ))}
                  </div>
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
                  {/* Insert merges into the graph the user is standing in.
                      With no tab open -- the welcome screen, which offers
                      this modal too -- there is no such graph: `insertGraph`
                      would map over an empty tab list and do nothing at all,
                      silently. Open in a new tab is the whole offer there. */}
                  {hasActiveTab && (
                    <>
                      <button
                        type="button"
                        className={styles.secondaryBtn}
                        disabled={busy}
                        onClick={() => void take(() => insertExample(chosen.path))}
                      >
                        {t('gallery.insert')}
                      </button>
                      <div className={styles.detailHint}>{t('gallery.insertHint')}</div>
                    </>
                  )}
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

/** One tag in the filter row: a toggle that names the tag and its count. */
function TagFilterButton({ tag, count, pressed, onToggle }: {
  tag: string;
  count: number;
  pressed: boolean;
  onToggle: (tag: string) => void;
}) {
  const { t } = useI18n();
  const label = tagLabel(tag, t);
  return (
    <button
      type="button"
      className={`${styles.tagFilterBtn} ${pressed ? styles.tagFilterBtnOn : ''}`}
      aria-pressed={pressed}
      aria-label={t('examples.filter.tagCount', { tag: label, count })}
      onClick={() => onToggle(tag)}
    >
      <span>{label}</span>
      <span className={styles.tagFilterCount} aria-hidden="true">{count}</span>
    </button>
  );
}
