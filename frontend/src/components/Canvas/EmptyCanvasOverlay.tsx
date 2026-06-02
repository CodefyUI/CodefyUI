import { useState, useEffect, useCallback } from 'react';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { resolveSerializedNodes, resolveSerializedEdges } from '../../utils';
import { listExamples, loadExample } from '../../api/rest';
import type { ExampleSummary } from '../../api/rest';
import { useToastStore } from '../../store/toastStore';
import styles from './EmptyCanvasOverlay.module.css';

const EXAMPLE_CATEGORY_COLORS: Record<string, string> = {
  Usage_Example: '#4CAF50',
  Model_Architecture: '#2196F3',
};

const SECTION_ORDER: { category: string; titleKey: 'empty.section.trainable' | 'empty.section.architecture' }[] = [
  { category: 'Usage_Example', titleKey: 'empty.section.trainable' },
  { category: 'Model_Architecture', titleKey: 'empty.section.architecture' },
];

function renderCard(
  example: ExampleSummary,
  onClick: (e: ExampleSummary) => void,
  t: (k: any, vars?: Record<string, string | number>) => string,
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

  const grouped = SECTION_ORDER.map((s) => ({
    ...s,
    items: examples.filter((e) => e.category === s.category),
  })).filter((s) => s.items.length > 0);

  const uncategorized = examples.filter(
    (e) => !SECTION_ORDER.some((s) => s.category === e.category),
  );

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
        const resolvedEdges = resolveSerializedEdges(edges);
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

        {!loading && grouped.map((section) => (
          <div key={section.category} className={styles.section}>
            <div className={styles.sectionTitle}>{t(section.titleKey)}</div>
            <div className={styles.quickStartGrid}>
              {section.items.map((example) => renderCard(example, handleClick, t))}
            </div>
          </div>
        ))}

        {!loading && uncategorized.length > 0 && (
          <div className={styles.section}>
            <div className={styles.quickStartGrid}>
              {uncategorized.map((example) => renderCard(example, handleClick, t))}
            </div>
          </div>
        )}

        <div className={styles.hint}>{t('empty.hint')}</div>
      </div>
    </div>
  );
}
