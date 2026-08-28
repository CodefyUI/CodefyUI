import { useState, useRef, useCallback, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import { nodeMissingPack, packTitle, usePackAvailability } from '../../utils/packAvailability';
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
  const { t, tn } = useI18n();
  // Three narrow slices that change only when a catalog refresh lands, so a
  // library of a hundred rows costs a selector compare each and nothing per
  // frame. The badge is deliberately NOT a gate: the item stays draggable,
  // and the pre-run check (plus the backend error) is the real safety net.
  const { byId, loaded, unsupported } = usePackAvailability();
  const missingPack = nodeMissingPack(definition, byId, loaded, unsupported);
  const packSentence =
    missingPack === null
      ? null
      : t('palette.needsPack.title', { pack: packTitle(byId, missingPack.packId) });

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

  // A pack-backed node with no description still earns a tooltip: the pack
  // sentence is the thing worth reading before the drag.
  const showTooltip = tooltipsEnabled && (desc || packSentence) && hovered && tooltipPos;

  return (
    <div
      ref={itemRef}
      draggable
      onDragStart={handleDragStart}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      className={styles.nodeItem}
      style={{
        background: hovered ? 'var(--surface-hover)' : 'transparent',
        borderColor: hovered ? 'var(--border-base)' : 'transparent',
      }}
    >
      <div className={styles.nodeItemName}>
        {definition.node_name}
      </div>
      {/* A sibling of the name rather than a child of it, so the name keeps
          its own ellipsis; the item's grid puts the two on one row. */}
      {packSentence !== null && (
        <span className={styles.nodeItemBadge} title={packSentence}>
          {t('palette.needsPack')}
        </span>
      )}
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
          {desc && <div className={styles.nodeTooltipDesc}>{desc}</div>}
          {packSentence !== null && (
            <div className={styles.nodeTooltipPack}>{packSentence}</div>
          )}
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
 *
 * A pure consumer of the catalog: this tab mounts only while it is the open
 * one, so it must not be what STARTS the catalog load — that belongs to the
 * always-mounted shell (see `useNodeDefinitionsBootstrap`).
 */
export function NodesTab() {
  const categorized = useNodeDefStore((s) => s.categorized);
  const loading = useNodeDefStore((s) => s.loading);
  const error = useNodeDefStore((s) => s.error);
  const refetch = useNodeDefStore((s) => s.fetchDefinitions);
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
