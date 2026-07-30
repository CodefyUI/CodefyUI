import { useState, useEffect, useCallback } from 'react';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useTabStore } from '../../store/tabStore';
import { useI18n, type TranslationKey } from '../../i18n';
import { resolveSerializedNodes, resolveSerializedEdges } from '../../utils';
import { listExamples, loadExample } from '../../api/rest';
import type { ExampleSummary } from '../../api/rest';
import { useToastStore } from '../../store/toastStore';
import styles from './EmptyCanvasOverlay.module.css';

// Restrained material-style hue per builtin category; unknown categories
// (e.g. plugin-defined folders) fall back to orange.
const EXAMPLE_CATEGORY_COLORS: Record<string, string> = {
  Usage_Example: '#4CAF50',
  Model_Architecture: '#2196F3',
  Classical: '#26A69A',
  LLM: '#AB47BC',
  Diffusion: '#EC407A',
  Transformer: '#26C6DA',
  RNN: '#5C6BC0',
  RL: '#EF5350',
};

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
  const catColor = EXAMPLE_CATEGORY_COLORS[example.category] ?? '#FF9800';
  const catLabel = example.category.replace(/_/g, ' ');
  return (
    <button type="button"
      key={example.path}
      onClick={() => onClick(example)}
      className={styles.presetCard}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = '#D4A017';
        e.currentTarget.style.boxShadow = '0 4px 16px rgba(212,160,23,0.15)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = '#3a3a3a';
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
          style={{ background: `${catColor}22`, color: catColor }}
        >
          {catLabel}
        </span>
        <span className={styles.nodeCount}>{t('empty.nodeCount', { count: example.node_count })}</span>
      </div>
    </button>
  );
}

export function EmptyCanvasOverlay() {
  const setNodes = useTabStore((s) => s.setNodes);
  const setEdges = useTabStore((s) => s.setEdges);
  const renameTab = useTabStore((s) => s.renameTab);
  const { t } = useI18n();
  const addToast = useToastStore((s) => s.addToast);

  const [examples, setExamples] = useState<ExampleSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listExamples()
      .then((all) => setExamples(all))
      .catch(() => setExamples([]))
      .finally(() => setLoading(false));
  }, []);

  const sections = groupExamples(examples);

  const handleClick = useCallback(
    async (example: ExampleSummary) => {
      try {
        const data = await loadExample(example.path);
        const rawNodes = data.nodes ?? [];
        const edges = data.edges ?? [];

        const store = useNodeDefStore.getState();
        const importedPresets = Array.isArray(data.presets) ? data.presets : [];
        const mergedPresets = [...store.presets];
        for (const p of importedPresets) {
          if (!mergedPresets.some((ep) => ep.preset_name === p.preset_name)) {
            mergedPresets.push(p);
          }
        }

        const resolvedNodes = resolveSerializedNodes(rawNodes, store.definitions, mergedPresets);
        const resolvedEdges = resolveSerializedEdges(edges, resolvedNodes);
        setNodes(resolvedNodes);
        setEdges(resolvedEdges);

        // Mirror the example name onto the active tab so saves, exports,
        // and the script header all use a meaningful name out of the box.
        const exampleName = typeof data.name === 'string' && data.name.trim() ? data.name.trim() : null;
        if (exampleName) {
          const { activeTabId } = useTabStore.getState();
          renameTab(activeTabId, exampleName);
        }

        if (importedPresets.length > 0) {
          useNodeDefStore.setState({ presets: mergedPresets });
        }
      } catch {
        addToast(t('empty.loadError'), 'error');
      }
    },
    [setNodes, setEdges, renameTab, t, addToast],
  );

  return (
    <div className={styles.overlay}>
      <div className={styles.inner}>
        <div className={styles.title}>{t('empty.title')}</div>
        <div className={styles.subtitle}>{t('empty.subtitle')}</div>

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
