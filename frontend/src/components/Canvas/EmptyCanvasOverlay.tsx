import { useState, useEffect, useCallback } from 'react';
import { useI18n, type TranslationKey } from '../../i18n';
import { openExample } from '../../utils/openExample';
import { useUIStore } from '../../store/uiStore';
import { listExamples } from '../../api/rest';
import type { ExampleSummary } from '../../api/rest';
import { EXAMPLE_CATEGORY_COLORS, EXAMPLE_CATEGORY_FALLBACK, SURFACE_RAISED, NODE_HEADER_TINT, mixColor } from '../../styles/theme';
import styles from './EmptyCanvasOverlay.module.css';

// Quick Start: pinned by path (paths are stable identifiers, robust to
// backend ordering), rendered in exactly this order.
const QUICK_START_PATHS: string[] = [
  'Usage_Example/CNN-MNIST/TrainCNN-MNIST',
  'Usage_Example/CNN-MNIST/InferenceCNN-MNIST',
  'Usage_Example/Api-Function',
];

// Advanced section: category display order (most visually rewarding first).
// Builtin categories not listed here fall through to the plugin/other section.
const ADVANCED_CATEGORY_ORDER: string[] = [
  'LLM',
  'Diffusion',
  'Classical',
  'Transformer',
  'RNN',
  'RL',
  'Usage_Example',
];

// Optional fine-grained ordering inside one Advanced category (lower renders
// first). Unlisted paths sort after listed ones, keeping the backend's
// alphabetical order among themselves.
const ADVANCED_PATH_PRIORITY: Record<string, number> = {
  'Diffusion/Forward-Process': 0,
  'Diffusion/Toy-Sampling': 1,
  'Diffusion/Mini-UNet-Compact': 2,
};

interface GallerySection {
  key: string;
  titleKey: TranslationKey;
  items: ExampleSummary[];
}

function compareAdvanced(a: ExampleSummary, b: ExampleSummary): number {
  const catDiff =
    ADVANCED_CATEGORY_ORDER.indexOf(a.category) - ADVANCED_CATEGORY_ORDER.indexOf(b.category);
  if (catDiff !== 0) return catDiff;
  return (
    (ADVANCED_PATH_PRIORITY[a.path] ?? Number.MAX_SAFE_INTEGER) -
    (ADVANCED_PATH_PRIORITY[b.path] ?? Number.MAX_SAFE_INTEGER)
  );
}

/** Split the flat example list into the ordered gallery sections.
 *
 * 1. Quick Start  — the three pinned starter examples.
 * 2. Advanced     — remaining runnable builtin examples, curated order.
 * 3. Plugin/other — plugin-shipped examples and unknown categories.
 * 4. Architectures — Model_Architecture, always last.
 * Empty sections are dropped.
 */
function groupExamples(examples: ExampleSummary[]): GallerySection[] {
  const byPath = new Map(examples.map((e) => [e.path, e]));
  const quickStart = QUICK_START_PATHS.map((p) => byPath.get(p)).filter(
    (e): e is ExampleSummary => e !== undefined,
  );
  const pinned = new Set(QUICK_START_PATHS);

  const advanced: ExampleSummary[] = [];
  const architectures: ExampleSummary[] = [];
  const other: ExampleSummary[] = [];
  for (const e of examples) {
    if (pinned.has(e.path)) continue;
    if (e.path.startsWith('plugin:')) other.push(e);
    else if (e.category === 'Model_Architecture') architectures.push(e);
    else if (ADVANCED_CATEGORY_ORDER.includes(e.category)) advanced.push(e);
    else other.push(e);
  }
  advanced.sort(compareAdvanced);

  const sections: GallerySection[] = [
    { key: 'quickstart', titleKey: 'empty.section.quickstart', items: quickStart },
    { key: 'advanced', titleKey: 'empty.section.advanced', items: advanced },
    { key: 'plugin', titleKey: 'empty.section.plugin', items: other },
    { key: 'architecture', titleKey: 'empty.section.architecture', items: architectures },
  ];
  return sections.filter((s) => s.items.length > 0);
}

function renderCard(
  example: ExampleSummary,
  onClick: (e: ExampleSummary) => void,
  t: (k: TranslationKey, vars?: Record<string, string | number>) => string,
) {
  const catColor = EXAMPLE_CATEGORY_COLORS[example.category] ?? EXAMPLE_CATEGORY_FALLBACK;
  const catLabel = example.category.replace(/_/g, ' ');
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
      <div className={styles.presetCardDesc}>
        {example.description.length > 80
          ? example.description.slice(0, 80) + '...'
          : example.description}
      </div>
      <div className={styles.presetCardFooter}>
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
          {catLabel}
        </span>
        <span className={styles.nodeCount}>{t('empty.nodeCount', { count: example.node_count })}</span>
      </div>
    </button>
  );
}

export function EmptyCanvasOverlay() {
  const { t } = useI18n();

  const [examples, setExamples] = useState<ExampleSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listExamples()
      .then((all) => setExamples(all))
      .catch(() => setExamples([]))
      .finally(() => setLoading(false));
  }, []);

  const sections = groupExamples(examples);

  // Loading an example lives in `utils/openExample` so the sidebar's
  // Templates tab (#126) opens one identically.
  const handleClick = useCallback(
    (example: ExampleSummary) => void openExample(example.path),
    [],
  );

  return (
    <div className={styles.overlay}>
      <div className={styles.inner}>
        <div className={styles.title}>{t('empty.title')}</div>
        <div className={styles.subtitle}>{t('empty.subtitle')}</div>

        {/* The curated sections below stay the fast path; this is the way to
            the full, searchable list — the same modal the toolbar and the
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

        {!loading && sections.map((section) => (
          <div key={section.key} className={styles.section}>
            <div className={styles.sectionTitle}>{t(section.titleKey)}</div>
            <div className={styles.quickStartGrid}>
              {section.items.map((example) => renderCard(example, handleClick, t))}
            </div>
          </div>
        ))}

        <div className={styles.hint}>{t('empty.hint')}</div>
      </div>
    </div>
  );
}
