import { useState, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { useNodeDefinitions } from '../../hooks/useNodeDefinitions';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import type { NodeDefinition, PresetDefinition } from '../../types';
import { CATEGORY_COLORS, DIFFICULTY_COLORS } from '../../styles/theme';
import styles from './NodePalette.module.css';

const CATEGORY_ORDER = ['Control', 'Data', 'IO', 'CNN', 'Normalization', 'RNN', 'Transformer', 'LLM', 'RL', 'Training', 'Tensor Operations', 'Utility'];
const BEGINNER_CATEGORIES = new Set(['Data', 'CNN', 'Training', 'IO']);

// ── Operation Node Item ──

interface NodeItemProps {
  definition: NodeDefinition;
}

function NodeItem({ definition }: NodeItemProps) {
  const [hovered, setHovered] = useState(false);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null);
  const itemRef = useRef<HTMLDivElement>(null);
  const tooltipsEnabled = useUIStore((s) => s.tooltipsEnabled);
  const { tn } = useI18n();

  const desc = tn(definition.node_name, 'description', definition.description);

  const handleMouseEnter = useCallback(() => {
    setHovered(true);
    if (itemRef.current) {
      const rect = itemRef.current.getBoundingClientRect();
      setTooltipPos({ x: rect.right + 8, y: rect.top });
    }
  }, []);

  const handleMouseLeave = useCallback(() => {
    setHovered(false);
    setTooltipPos(null);
  }, []);

  const handleDragStart = (event: React.DragEvent) => {
    event.dataTransfer.setData('application/codefyui-node', definition.node_name);
    event.dataTransfer.effectAllowed = 'move';
  };

  const showTooltip = tooltipsEnabled && desc && hovered && tooltipPos;

  return (
    <div
      ref={itemRef}
      draggable
      onDragStart={handleDragStart}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      className={styles.nodeItem}
      style={{
        background: hovered ? '#2a2a2a' : 'transparent',
        borderColor: hovered ? '#444' : 'transparent',
      }}
    >
      <div className={styles.nodeItemName}>
        {definition.node_name}
      </div>
      {desc && (
        <div className={styles.nodeItemDesc}>
          {desc}
        </div>
      )}
      {showTooltip && createPortal(
        <div
          className={styles.nodeTooltip}
          style={{ left: tooltipPos.x, top: tooltipPos.y }}
        >
          <div className={styles.nodeTooltipTitle}>{definition.node_name}</div>
          <div className={styles.nodeTooltipDesc}>{desc}</div>
        </div>,
        document.body,
      )}
    </div>
  );
}

// ── Preset Item ──

interface PresetItemProps {
  preset: PresetDefinition;
}

function PresetItem({ preset }: PresetItemProps) {
  const [hovered, setHovered] = useState(false);
  const difficulty = preset.tags.find((t) => t in DIFFICULTY_COLORS) ?? 'beginner';
  const difficultyColor = DIFFICULTY_COLORS[difficulty];

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
        {preset.nodes.length} nodes
      </div>
    </div>
  );
}

// ── Sub-header separator ──

function SubHeader({ label }: { label: string }) {
  return (
    <div className={styles.subHeader}>
      <span className={styles.subHeaderLine} />
      <span className={styles.subHeaderText}>{label}</span>
      <span className={styles.subHeaderLine} />
    </div>
  );
}

// ── Unified Category Section ──

interface UnifiedCategorySectionProps {
  category: string;
  presets: PresetDefinition[];
  nodes: NodeDefinition[];
}

function UnifiedCategorySection({ category, presets, nodes }: UnifiedCategorySectionProps) {
  const [expanded, setExpanded] = useState(true);
  const color = CATEGORY_COLORS[category] ?? '#607D8B';
  const { t } = useI18n();
  const hasBoth = presets.length > 0 && nodes.length > 0;

  return (
    <div className={styles.categorySection}>
      <button
        onClick={() => setExpanded((prev) => !prev)}
        className={styles.categoryButton}
        style={{ borderBottom: `2px solid ${color}` }}
      >
        <span
          className={styles.categoryDot}
          style={{ background: color }}
        />
        <span className={styles.categoryName} style={{ color }}>
          {category}
        </span>
        <span className={styles.categoryCount}>
          {presets.length + nodes.length}
        </span>
        <span className={styles.categoryChevron}>
          {expanded ? '▾' : '▸'}
        </span>
      </button>

      {expanded && (
        <div className={styles.categoryNodes}>
          {hasBoth && <SubHeader label={t('palette.composite')} />}
          {presets.map((p) => (
            <PresetItem key={p.preset_name} preset={p} />
          ))}
          {hasBoth && <SubHeader label={t('palette.basic')} />}
          {nodes.map((def) => (
            <NodeItem key={def.node_name} definition={def} />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Main Palette ──

export function NodePalette() {
  const { categorized, loading, error, refetch } = useNodeDefinitions();
  const presetCategorized = useNodeDefStore((s) => s.presetCategorized);
  const [searchQuery, setSearchQuery] = useState('');
  const { t } = useI18n();

  // Merge categories from both sources
  const allCategoryKeys = new Set([
    ...Object.keys(categorized),
    ...Object.keys(presetCategorized),
  ]);
  const beginnerMode = useUIStore((s) => s.beginnerMode);
  const orderedCategories = [
    ...CATEGORY_ORDER.filter((c) => allCategoryKeys.has(c)),
    ...[...allCategoryKeys].filter((c) => !CATEGORY_ORDER.includes(c)).sort(),
  ].filter((c) => !beginnerMode || BEGINNER_CATEGORIES.has(c));

  // Unified filtering
  const q = searchQuery.trim().toLowerCase();
  const mergedCategories: { category: string; presets: PresetDefinition[]; nodes: NodeDefinition[] }[] = [];

  for (const cat of orderedCategories) {
    let filteredNodes = categorized[cat] ?? [];
    let filteredPresets = presetCategorized[cat] ?? [];

    if (q) {
      filteredNodes = filteredNodes.filter(
        (n) =>
          n.node_name.toLowerCase().includes(q) ||
          n.description.toLowerCase().includes(q)
      );
      filteredPresets = filteredPresets.filter(
        (p) =>
          p.preset_name.toLowerCase().includes(q) ||
          p.description.toLowerCase().includes(q) ||
          p.tags.some((tag) => tag.toLowerCase().includes(q))
      );
    }

    if (filteredNodes.length > 0 || filteredPresets.length > 0) {
      mergedCategories.push({ category: cat, presets: filteredPresets, nodes: filteredNodes });
    }
  }

  return (
    <div className={styles.sidebar}>
      {/* Sidebar header */}
      <div className={styles.header}>
        <div className={styles.headerTitle}>
          {t('palette.title')}
        </div>
        <input
          type="text"
          placeholder={t('palette.search')}
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className={styles.searchInput}
        />
      </div>

      {/* Content */}
      <div className={styles.content}>
        {loading && (
          <div className={styles.stateMessage}>
            {t('palette.loading')}
          </div>
        )}

        {error && (
          <div className={styles.errorWrapper}>
            <div className={styles.errorText}>
              {t('palette.loadFail', { error })}
            </div>
            <button onClick={refetch} className={styles.retryButton}>
              {t('palette.retry')}
            </button>
          </div>
        )}

        {!loading && !error && (
          <>
            {mergedCategories.length === 0 && (
              <div className={styles.stateMessageMuted}>
                {searchQuery ? t('palette.noMatch') : t('palette.empty')}
              </div>
            )}
            {mergedCategories.map(({ category, presets, nodes }) => (
              <UnifiedCategorySection
                key={category}
                category={category}
                presets={presets}
                nodes={nodes}
              />
            ))}
          </>
        )}
      </div>

      {/* Footer hint */}
      <div className={styles.footer}>
        {t('palette.hint')}
      </div>
    </div>
  );
}
