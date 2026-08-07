import { useState, useMemo } from 'react';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import type { PresetDefinition } from '../../types';
import { DIFFICULTY_COLORS } from '../../styles/theme';
import { orderCategories } from './categories';
import { CategoryList, type CategoryGroup } from './CategoryList';
import styles from './NodePalette.module.css';

// ── Preset Item ──

interface PresetItemProps {
  preset: PresetDefinition;
}

export function PresetItem({ preset }: PresetItemProps) {
  const [hovered, setHovered] = useState(false);
  const difficulty = preset.tags.find((t) => t in DIFFICULTY_COLORS) ?? 'beginner';
  const difficultyColor = DIFFICULTY_COLORS[difficulty];
  const { t } = useI18n();

  const handleDragStart = (event: React.DragEvent) => {
    event.dataTransfer.setData('application/codefyui-preset', preset.preset_name);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div
      draggable
      onDragStart={handleDragStart}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      title={preset.description}
      className={styles.presetItem}
      // Gold preset-hover tint: a semantic per-item accent (same family as
      // CATEGORY_COLORS/DIFFICULTY_COLORS below), not chrome, so it is left
      // outside the grey/accent token sweep. Close to --status-preset
      // (#e0a92b) but not identical, and there is no wash/alpha variant of
      // it to reach for; PresetsTab.test.tsx also pins this exact rgba
      // string. See migration report for the token gap.
      style={{
        background: hovered ? 'rgba(212,160,23,0.08)' : 'transparent',
        borderColor: hovered ? 'rgba(212,160,23,0.3)' : 'transparent',
      }}
    >
      <div className={styles.presetHeader}>
        <div className={styles.presetName}>
          {preset.preset_name}
        </div>
        <span
          className={styles.presetDifficultyBadge}
          style={{
            background: `${difficultyColor}22`,
            color: difficultyColor,
          }}
        >
          {difficulty}
        </span>
      </div>
      <div className={styles.presetDesc}>
        {preset.description}
      </div>
      <div className={styles.presetNodeCount}>
        {t('empty.nodeCount', { count: preset.nodes.length })}
      </div>
    </div>
  );
}

// ── Presets tab ──

/**
 * Composite presets, on their own rail tab since #126.
 *
 * They used to be folded into the node list under a "Composite" sub-header
 * inside each category, which meant a category's count mixed two different
 * kinds of thing and neither could be browsed on its own. Category ordering
 * and beginner-mode filtering are shared with the Nodes tab so the two lists
 * stay in step.
 */
export function PresetsTab() {
  const presetCategorized = useNodeDefStore((s) => s.presetCategorized);
  const loading = useNodeDefStore((s) => s.loading);
  const beginnerMode = useUIStore((s) => s.beginnerMode);
  const [searchQuery, setSearchQuery] = useState('');
  const { t } = useI18n();

  const groups = useMemo<CategoryGroup<PresetDefinition>[]>(() => {
    const q = searchQuery.trim().toLowerCase();
    const out: CategoryGroup<PresetDefinition>[] = [];
    // orderCategories only ever returns keys it was given, so the lookup below
    // is always a hit.
    for (const category of orderCategories(Object.keys(presetCategorized), beginnerMode)) {
      let items = presetCategorized[category];
      if (q) {
        items = items.filter(
          (p) =>
            p.preset_name.toLowerCase().includes(q) ||
            p.description.toLowerCase().includes(q) ||
            p.tags.some((tag) => tag.toLowerCase().includes(q)),
        );
      }
      if (items.length > 0) out.push({ category, items });
    }
    return out;
  }, [presetCategorized, beginnerMode, searchQuery]);

  return (
    <>
      <div className={styles.header}>
        <div className={styles.headerTitle}>
          {t('sidebar.tab.presets')}
        </div>
        <input
          type="text"
          placeholder={t('palette.searchPresets')}
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className={styles.searchInput}
        />
      </div>

      <div className={styles.panelBody}>
        {loading && (
          <div className={styles.stateMessage}>
            {t('palette.loading')}
          </div>
        )}

        {!loading && (
          groups.length === 0 ? (
            <div className={styles.stateMessageMuted}>
              {searchQuery ? t('palette.presets.noMatch') : t('palette.presets.empty')}
            </div>
          ) : (
            <CategoryList
              groups={groups}
              itemKey={(preset) => preset.preset_name}
              renderItem={(preset) => <PresetItem preset={preset} />}
            />
          )
        )}
      </div>

      <div className={styles.footer}>
        {t('palette.presets.hint')}
      </div>
    </>
  );
}
