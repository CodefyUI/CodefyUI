import { useId, useMemo, useState } from 'react';
import { useI18n, type TranslationKey } from '../../i18n';
import { selectPluginsById, usePluginStore } from '../../store/pluginStore';
import type { ExampleSummary } from '../../api/rest';
import {
  useLocalizedExamples,
  truncateToWidth,
  type LocalizedExample,
} from '../../utils/localizeExamples';
import {
  cardTags,
  groupExamplesBySection,
  mainTagLabel,
  mainTagOf,
  tagLabel,
} from '../../utils/exampleSections';
import { useExampleCatalog } from '../../utils/useExampleCatalog';
import { pluginNameOf } from '../../utils/provider';
import {
  EXAMPLE_CATEGORY_FALLBACK,
  EXAMPLE_SECTION_COLORS,
  SURFACE_RAISED,
  NODE_HEADER_TINT,
  mixColor,
} from '../../styles/theme';
import {
  ALL_MAIN_TAGS,
  MainTagChips,
  initialMainTag,
  mainTagTabId,
  rememberMainTag,
} from './MainTagChips';
import styles from './ExampleBrowser.module.css';

/** How much of a description a preset card shows, in Latin columns.
 *
 * The same 40 columns a description is allowed to be in the first place --
 * `MAX_DESCRIPTION_COLUMNS` in `backend/tests/test_example_descriptions.py`
 * and its Chinese half in `exampleLocales/zh-TW.test.ts`. Keep the three in
 * step: a cut wider than the rule can never fire, and a cut narrower than it
 * would clip text that obeys the rule.
 */
const CARD_DESC_COLUMNS = 40;

/** The tags under a card: at most two, the rest as "+n" with a tooltip that
 * names them. Each carries its text; the border repeats the main tag's hue. */
function CardTags({ example, t }: {
  example: ExampleSummary;
  t: (k: TranslationKey, vars?: Record<string, string | number>) => string;
}) {
  const { shown, more } = cardTags(example, 2);
  const hue = EXAMPLE_SECTION_COLORS[mainTagOf(example)] ?? EXAMPLE_CATEGORY_FALLBACK;
  // Fill is the hue tinted into the card surface and the border carries the
  // hue at full strength; the label takes the text tier (check-contrast 8b).
  const style = { background: mixColor(SURFACE_RAISED, hue, NODE_HEADER_TINT), borderColor: hue };
  const rest = cardTags(example, Number.MAX_SAFE_INTEGER).shown.slice(2);
  return (
    <>
      {shown.map((tag) => (
        <span key={tag} className={styles.difficultyBadge} style={style}>
          {tagLabel(tag, t)}
        </span>
      ))}
      {more > 0 && (
        <span
          className={styles.moreTags}
          title={t('examples.card.moreTagsTitle', {
            count: more,
            tags: rest.map((tag) => tagLabel(tag, t)).join(', '),
          })}
        >
          {t('examples.card.moreTags', { count: more })}
        </span>
      )}
    </>
  );
}

function hoverOn(e: React.MouseEvent<HTMLButtonElement>) {
  e.currentTarget.style.borderColor = 'var(--status-preset)';
  // rgb(224,169,43) is --status-preset's own rgb() — no glow token is
  // paired with it, so this stays a hand-tuned literal at the hue.
  e.currentTarget.style.boxShadow = '0 4px 16px rgba(224, 169, 43, 0.15)';
}

function hoverOff(e: React.MouseEvent<HTMLButtonElement>) {
  e.currentTarget.style.borderColor = 'var(--border-base)';
  e.currentTarget.style.boxShadow = 'none';
}

function renderCard(
  example: LocalizedExample,
  onClick: (e: ExampleSummary) => void,
  t: (k: TranslationKey, vars?: Record<string, string | number>) => string,
) {
  const shown = truncateToWidth(example.description, CARD_DESC_COLUMNS);
  return (
    <button type="button"
      key={example.path}
      onClick={() => onClick(example)}
      className={styles.presetCard}
      onMouseEnter={hoverOn}
      onMouseLeave={hoverOff}
    >
      <div className={styles.presetCardHeader}>
        <span className={styles.presetCardName}>{example.name}</span>
      </div>
      {/* The cut is 40 columns — the width a description is allowed to be at
          all, pinned by both halves of the length rule. So for everything
          that ships in this repo it never fires; it fires for a third-party
          pack that writes past the rule, which is when the tooltip earns its
          place (core#305). Measured in columns rather than code points so
          the Chinese card cuts in the same place on screen. The tooltip is
          set only when there is more to show. */}
      <div
        className={styles.presetCardDesc}
        {...(shown !== example.description ? { title: example.description } : {})}
      >
        {shown}
      </div>
      <div className={styles.presetCardFooter}>
        {/* Tags, not the folder name: "Usage Example" on a card said where
            the file sits, not what the example is for (phase 3). */}
        <CardTags example={example} t={t} />
        <span className={styles.nodeCount}>{t('empty.nodeCount', { count: example.node_count })}</span>
      </div>
    </button>
  );
}

/** The first card on every view: an empty canvas, as a choice beside the
 * examples rather than a separate control. */
function BlankCard({ onClick, t }: {
  onClick: () => void;
  t: (k: TranslationKey, vars?: Record<string, string | number>) => string;
}) {
  return (
    <button
      type="button"
      className={`${styles.presetCard} ${styles.blankCard}`}
      onClick={onClick}
      onMouseEnter={hoverOn}
      onMouseLeave={hoverOff}
    >
      <div className={styles.presetCardHeader}>
        <span className={styles.presetCardName}>{t('examples.blank.name')}</span>
      </div>
      <div className={styles.presetCardDesc}>{t('examples.blank.desc')}</div>
    </button>
  );
}

/** Where the chosen main tag is remembered, so the next new tab opens on it. */
export const EXAMPLE_BROWSER_STORAGE_KEY = 'codefyui.examples.mainTag';

export interface ExampleBrowserProps {
  /**
   * What clicking a card does. The two hosts answer it differently: the empty
   * canvas replaces the graph in front of the user (there is nothing there to
   * lose), while the welcome screen has no tab to replace and makes one first.
   */
  onPick: (example: ExampleSummary) => void;
  /** What the "Blank graph" card does: close the overlay over an empty
   * canvas, or open an empty tab from the welcome screen. */
  onBlank: () => void;
}

/**
 * The example gallery, as both empty states draw it.
 *
 * Shared by `EmptyCanvasOverlay` and `WelcomeScreen` so the two cannot drift:
 * the fetch, the localization, the main-tag chips, the grouping, the
 * description cut and the card are one implementation.
 *
 * Under the host's title sits a row of main-tag chips: All plus every main
 * tag that has examples, in the server's order (practical first). It opens on
 * the chip chosen last time (localStorage), else Quick start. A main tag shows
 * its cards by order; All shows every main tag as a titled section. The first
 * card is always "Blank graph".
 */
export function ExampleBrowser({ onPick, onBlank }: ExampleBrowserProps) {
  const { t } = useI18n();
  const pluginsById = usePluginStore(selectPluginsById);
  const panelId = useId();
  const idPrefix = useId();
  const catalog = useExampleCatalog();
  const [chosen, setChosen] = useState<string | null>(null);

  const localized = useLocalizedExamples(catalog.examples);
  const order = useMemo(() => catalog.mainTags.map((entry) => entry.id), [catalog.mainTags]);
  const sections = useMemo(
    () => groupExamplesBySection(localized, (source) => pluginNameOf(pluginsById, source), order),
    [localized, pluginsById, order],
  );

  if (catalog.loading) return <div className={styles.hint}>{t('empty.loading')}</div>;

  const selected =
    chosen !== null && (chosen === ALL_MAIN_TAGS || order.includes(chosen))
      ? chosen
      : initialMainTag(EXAMPLE_BROWSER_STORAGE_KEY, catalog.mainTags, 'quickstart');
  const select = (value: string) => {
    setChosen(value);
    rememberMainTag(EXAMPLE_BROWSER_STORAGE_KEY, value);
  };
  const shownSections =
    selected === ALL_MAIN_TAGS ? sections : sections.filter((section) => section.key === selected);
  const blank = <BlankCard key="__blank" onClick={onBlank} t={t} />;

  return (
    <>
      {catalog.mainTags.length > 0 && (
        <MainTagChips
          entries={catalog.mainTags}
          value={selected}
          onChange={select}
          idPrefix={idPrefix}
          panelId={panelId}
        />
      )}
      <div
        id={panelId}
        role={catalog.mainTags.length > 0 ? 'tabpanel' : undefined}
        aria-labelledby={catalog.mainTags.length > 0 ? mainTagTabId(idPrefix, selected) : undefined}
      >
        {shownSections.length === 0 && <div className={styles.quickStartGrid}>{blank}</div>}
        {/* With All, one title per main tag. With one main tag chosen the
            chip already says it, so only the family and pack sub-headers
            remain. The blank card leads the first grid either way. */}
        {shownSections.map((section, sectionIndex) => (
          <div key={section.key} className={styles.section}>
            {selected === ALL_MAIN_TAGS && (
              <div className={styles.sectionTitle}>{mainTagLabel(section.key, t)}</div>
            )}
            {section.subgroups.map((subgroup, subgroupIndex) => (
              <div key={subgroup.key} className={styles.subsection}>
                {subgroup.label !== null && (
                  <div className={styles.subsectionTitle}>{subgroup.label}</div>
                )}
                <div className={styles.quickStartGrid}>
                  {sectionIndex === 0 && subgroupIndex === 0 && blank}
                  {subgroup.items.map((example) => renderCard(example, onPick, t))}
                </div>
              </div>
            ))}
          </div>
        ))}
      </div>
    </>
  );
}
