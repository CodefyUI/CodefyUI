import { useState, useEffect, useCallback, useMemo } from 'react';
import { useI18n, type TranslationKey } from '../../i18n';
import { openExample } from '../../utils/openExample';
import { selectPluginsById, usePluginStore } from '../../store/pluginStore';
import { useUIStore } from '../../store/uiStore';
import { listExamples } from '../../api/rest';
import type { ExampleSummary } from '../../api/rest';
import {
  useLocalizedExamples,
  truncateToWidth,
  type LocalizedExample,
} from '../../utils/localizeExamples';
import { groupExamplesBySection, exampleChipLabel } from '../../utils/exampleSections';
import { pluginNameOf } from '../../utils/provider';
import { EXAMPLE_CATEGORY_COLORS, EXAMPLE_CATEGORY_FALLBACK, SURFACE_RAISED, NODE_HEADER_TINT, mixColor } from '../../styles/theme';
import styles from './EmptyCanvasOverlay.module.css';

/** How much of a description a preset card shows, in Latin columns. */
const CARD_DESC_COLUMNS = 80;

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
      {/* The cut stays at 80 columns: every example's first line is written to
          say what the card has to say inside it, and the backend example suite
          asserts that a GPU, a download or a pack is named there. Measured in
          columns rather than code points so the Chinese card cuts in the same
          place on screen — 80 ideographs are 160 columns wide and would run
          off the card. What the card was missing is the REST of the
          description — this was the only place one appeared with no way to
          read past the cut, while the sidebar's gallery tab has carried the
          full text as a tooltip all along (core#305). Only when there is more
          to show, so a short description does not get a tooltip repeating
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

export function EmptyCanvasOverlay() {
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

  // Still the REPLACING reader, and the only one left (#348): this overlay
  // is shown when the canvas is empty, so there is nothing for a replace to
  // take. The sidebar's Templates tab and the gallery modal both insert
  // instead, because they are reachable while a graph is on screen.
  const handleClick = useCallback(
    (example: ExampleSummary) => void openExample(example.path),
    [],
  );

  return (
    <div className={styles.overlay}>
      <div className={styles.inner}>
        <div className={styles.title}>{t('empty.title')}</div>
        <div className={styles.subtitle}>{t('empty.subtitle')}</div>

        {/* The sections below stay the fast path; this is the way to the
            full, searchable list — the same modal the toolbar and the
            sidebar's Templates tab open (core#128). */}
        <button
          type="button"
          className={styles.browseButton}
          onClick={() => useUIStore.getState().openTemplateGallery()}
          title={t('gallery.open.title')}
        >
          {t('gallery.browse')}
        </button>

        {loading && (
          <div className={styles.hint}>{t('empty.loading')}</div>
        )}

        {/* One section title, then a sub-header per architecture family and
            per pack — the two groupings where the section name alone leaves
            fifteen cards in one undifferentiated run. Every other section has
            a single unlabelled sub-group and renders exactly as before. */}
        {!loading && sections.map((section) => (
          <div key={section.key} className={styles.section}>
            <div className={styles.sectionTitle}>{t(section.titleKey)}</div>
            {section.subgroups.map((subgroup) => (
              <div key={subgroup.key} className={styles.subsection}>
                {subgroup.label !== null && (
                  <div className={styles.subsectionTitle}>{subgroup.label}</div>
                )}
                <div className={styles.quickStartGrid}>
                  {subgroup.items.map((example) => renderCard(example, handleClick, t, subgroup.label))}
                </div>
              </div>
            ))}
          </div>
        ))}

        {/* No trailing "or drag a node from the left palette": the palette it
            points at is open on its default tab with its own pinned footer
            reading "Drag nodes onto the canvas", and that footer is the copy
            that has to stay -- a node item is drag-only, and the footer is
            still there after the canvas stops being empty. */}
      </div>
    </div>
  );
}
