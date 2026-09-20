import { useState, useEffect, useMemo } from 'react';
import { useI18n, type TranslationKey } from '../../i18n';
import { selectPluginsById, usePluginStore } from '../../store/pluginStore';
import { listExamples } from '../../api/rest';
import type { ExampleSummary } from '../../api/rest';
import {
  useLocalizedExamples,
  truncateToWidth,
  type LocalizedExample,
} from '../../utils/localizeExamples';
import { groupExamplesBySection, exampleChipLabel } from '../../utils/exampleSections';
import { pluginNameOf } from '../../utils/provider';
import {
  EXAMPLE_CATEGORY_COLORS,
  EXAMPLE_CATEGORY_FALLBACK,
  SURFACE_RAISED,
  NODE_HEADER_TINT,
  mixColor,
} from '../../styles/theme';
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

function renderCard(
  example: LocalizedExample,
  onClick: (e: ExampleSummary) => void,
  t: (k: TranslationKey, vars?: Record<string, string | number>) => string,
  heading: string | null,
) {
  const catColor = EXAMPLE_CATEGORY_COLORS[example.category] ?? EXAMPLE_CATEGORY_FALLBACK;
  // The family, for the architectures: they are all one category, so
  // "Model Architecture" on every card of a section already headed
  // "Model Architectures" said nothing twice over. And no chip at all when
  // the sub-header right above the card is that same word.
  const chipLabel = exampleChipLabel(example);
  const showChip = chipLabel !== heading;
  const shown = truncateToWidth(example.description, CARD_DESC_COLUMNS);
  return (
    <button type="button"
      key={example.path}
      onClick={() => onClick(example)}
      className={styles.presetCard}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = 'var(--status-preset)';
        // rgb(224,169,43) is --status-preset's own rgb() — no glow token is
        // paired with it, so this stays a hand-tuned literal at the hue.
        e.currentTarget.style.boxShadow = '0 4px 16px rgba(224, 169, 43, 0.15)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = 'var(--border-base)';
        e.currentTarget.style.boxShadow = 'none';
      }}
    >
      <div className={styles.presetCardHeader}>
        <span className={styles.presetCardName}>{example.name}</span>
      </div>
      {/* The cut is 40 columns — the width a description is allowed to be at
          all, pinned by both halves of the length rule. So for everything
          that ships in this repo it never fires: the card shows the whole
          line, and a GPU, a download or a pack named in that line is on the
          card rather than behind a cut. It fires for a third-party pack that
          writes past the rule, which is exactly when the tooltip below earns
          its place — this card was the only surface that showed a
          description with no way to read past the cut, while the sidebar's
          gallery tab has carried the full text all along (core#305).
          Measured in columns rather than code points so the Chinese card
          cuts in the same place on screen: 40 ideographs are 80 columns wide
          and would run off the card. The tooltip is set only when there is
          more to show, so a description that fits does not get one repeating
          itself. */}
      <div
        className={styles.presetCardDesc}
        {...(shown !== example.description ? { title: example.description } : {})}
      >
        {shown}
      </div>
      <div className={styles.presetCardFooter}>
        {showChip && (
          <span
            className={styles.difficultyBadge}
            // Fill is the hue tinted into the card surface and the border carries
            // the hue at full strength; the label takes the text tier. The old
            // `${catColor}22` wash with the hue as text measured 2.24:1.
            style={{
              background: mixColor(SURFACE_RAISED, catColor, NODE_HEADER_TINT),
              borderColor: catColor,
            }}
          >
            {chipLabel}
          </span>
        )}
        <span className={styles.nodeCount}>{t('empty.nodeCount', { count: example.node_count })}</span>
      </div>
    </button>
  );
}

export interface ExampleBrowserProps {
  /**
   * What clicking a card does. The two hosts answer it differently and that
   * is the ONLY thing they differ on: the empty canvas replaces the graph in
   * front of the user (there is nothing there to lose), while the welcome
   * screen has no tab to replace and has to make one first.
   */
  onPick: (example: ExampleSummary) => void;
}

/**
 * The sectioned example gallery, as both empty states draw it.
 *
 * Extracted from `EmptyCanvasOverlay` when the welcome screen appeared, so the
 * two cannot drift: the fetch, the localization, the section/family grouping,
 * the 80-column description cut and the card itself are one implementation.
 * What each host keeps for itself is its own framing -- heading, copy, and
 * where the "browse everything" button sits.
 *
 * Fetches on mount, per host. Two of these are never on screen at once (one
 * needs a tab, the other needs no tab), so nothing is fetched twice.
 */
export function ExampleBrowser({ onPick }: ExampleBrowserProps) {
  const { t } = useI18n();
  const pluginsById = usePluginStore(selectPluginsById);

  const [examples, setExamples] = useState<ExampleSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listExamples()
      .then((all) => setExamples(all))
      .catch(() => setExamples([]))
      .finally(() => setLoading(false));
  }, []);

  const localized = useLocalizedExamples(examples);
  const sections = useMemo(
    () =>
      groupExamplesBySection(localized, (source) => pluginNameOf(pluginsById, source)),
    [localized, pluginsById],
  );

  if (loading) return <div className={styles.hint}>{t('empty.loading')}</div>;

  // One section title, then a sub-header per architecture family and per pack
  // — the two groupings where the section name alone leaves fifteen cards in
  // one undifferentiated run. Every other section has a single unlabelled
  // sub-group and renders exactly as before.
  return (
    <>
      {sections.map((section) => (
        <div key={section.key} className={styles.section}>
          <div className={styles.sectionTitle}>{t(section.titleKey)}</div>
          {section.subgroups.map((subgroup) => (
            <div key={subgroup.key} className={styles.subsection}>
              {subgroup.label !== null && (
                <div className={styles.subsectionTitle}>{subgroup.label}</div>
              )}
              <div className={styles.quickStartGrid}>
                {subgroup.items.map((example) => renderCard(example, onPick, t, subgroup.label))}
              </div>
            </div>
          ))}
        </div>
      ))}
    </>
  );
}
