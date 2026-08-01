import { useState, useRef, useCallback, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { useNodeDefinitions } from '../../hooks/useNodeDefinitions';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import type { NodeDefinition } from '../../types';
import { orderCategories } from './categories';
import { CategoryList, type CategoryGroup } from './CategoryList';
import styles from './NodePalette.module.css';

// ── Operation Node Item ──

interface NodeItemProps {
  definition: NodeDefinition;
}

export function NodeItem({ definition }: NodeItemProps) {
  const [hovered, setHovered] = useState(false);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null);
  const itemRef = useRef<HTMLDivElement>(null);
  const tooltipsEnabled = useUIStore((s) => s.tooltipsEnabled);
  const { tn } = useI18n();

  const desc = tn(definition.node_name, 'description', definition.description);

  const handleMouseEnter = useCallback(() => {
    setHovered(true);
    // mouseEnter fires on the element whose ref is itemRef, so it is always set
    /* v8 ignore start */
    if (itemRef.current) {
      const rect = itemRef.current.getBoundingClientRect();
      setTooltipPos({ x: rect.right + 8, y: rect.top });
    }
    /* v8 ignore stop */
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

// ── Nodes tab ──

/**
 * The node library: search, category accordions, drag-to-canvas.
 *
 * Lifted out of the old single-column `NodePalette` in #126 with its behaviour
 * intact; what changed is that presets moved to their own rail tab, so a
 * category here counts nodes only and the Composite/Basic sub-headers that
 * separated the two kinds are gone.
 */
export function NodesTab() {
  const { categorized, loading, error, refetch } = useNodeDefinitions();
  const beginnerMode = useUIStore((s) => s.beginnerMode);
  const [searchQuery, setSearchQuery] = useState('');
  const { t } = useI18n();

  const groups = useMemo<CategoryGroup<NodeDefinition>[]>(() => {
    const q = searchQuery.trim().toLowerCase();
    const out: CategoryGroup<NodeDefinition>[] = [];
    // orderCategories only ever returns keys it was given, so the lookup below
    // is always a hit.
    for (const category of orderCategories(Object.keys(categorized), beginnerMode)) {
      let items = categorized[category];
      if (q) {
        items = items.filter(
          (n) =>
            n.node_name.toLowerCase().includes(q) ||
            n.description.toLowerCase().includes(q),
        );
      }
      if (items.length > 0) out.push({ category, items });
    }
    return out;
  }, [categorized, beginnerMode, searchQuery]);

  return (
    <>
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

      <div className={styles.panelBody}>
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
            <button type="button" onClick={refetch} className={styles.retryButton}>
              {t('palette.retry')}
            </button>
          </div>
        )}

        {!loading && !error && (
          groups.length === 0 ? (
            <div className={styles.stateMessageMuted}>
              {searchQuery ? t('palette.noMatch') : t('palette.empty')}
            </div>
          ) : (
            <CategoryList
              groups={groups}
              itemKey={(def) => def.node_name}
              renderItem={(def) => <NodeItem definition={def} />}
            />
          )
        )}
      </div>

      <div className={styles.footer}>
        {t('palette.hint')}
      </div>
    </>
  );
}
