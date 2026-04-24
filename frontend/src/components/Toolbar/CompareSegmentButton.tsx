import { useTabStore } from '../../store/tabStore';
import { useToastStore } from '../../store/toastStore';
import { useI18n } from '../../i18n';
import { computeSegmentNodes } from '../../utils/segmentPath';
import { generateId } from '../../utils';
import { SURFACE, TEXT } from '../../styles/theme';
import styles from './Toolbar.module.css';

const ACCENT = '#ff9500';

/**
 * Toolbar entry-point for the Teaching Inspector's Segment feature.
 *
 * Click semantics — multiple segments can coexist on the canvas:
 *   1. Exactly two nodes selected  → create a new segment, append it to
 *      `segmentGroups`, mark it as the active one. Existing segments are
 *      left untouched; their orange bubbles stay visible.
 *   2. No two-node selection, but an active segment exists  → remove ONLY
 *      the active segment (the × on the bubble does the same thing, this
 *      button is just another affordance). Other segments survive.
 *   3. Neither condition met  → button is disabled.
 */
export function CompareSegmentButton() {
  const activeTab = useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId)!);
  const setActiveSegment = useTabStore((s) => s.setActiveSegment);
  const addSegmentGroup = useTabStore((s) => s.addSegmentGroup);
  const removeSegmentGroup = useTabStore((s) => s.removeSegmentGroup);
  const addToast = useToastStore((s) => s.addToast);
  const { t } = useI18n();

  const selected = activeTab.nodes.filter((n) => n.selected);
  const activeSegment = activeTab.activeSegment;
  const canCreate = selected.length === 2;
  const canClearActive = activeSegment !== null;

  const handleClick = () => {
    if (canCreate) {
      const [left, right] =
        selected[0].position.x <= selected[1].position.x
          ? [selected[0], selected[1]]
          : [selected[1], selected[0]];
      const seg = computeSegmentNodes(left.id, right.id, activeTab.nodes, activeTab.edges);
      if (seg.size === 0) {
        addToast(t('segment.noPath'), 'error');
        return;
      }
      const group = { id: generateId(), headNodeId: left.id, tailNodeId: right.id };
      addSegmentGroup(group);
      setActiveSegment(group);
      return;
    }
    if (canClearActive && activeSegment) {
      // Remove only the currently-active segment; the tab store clears
      // activeSegment when it matches the removed id.
      removeSegmentGroup(activeSegment.id);
      return;
    }
    // Nothing to do — prompt user.
    addToast(t('toolbar.compareSegment.needTwo'), 'warning');
  };

  const disabled = !canCreate && !canClearActive;
  const accent = canCreate ? ACCENT : canClearActive ? ACCENT : TEXT.muted;
  // When a segment is active AND no new selection, show "Clear active".
  // When two nodes are selected, always offer "Compare" — even if an active
  // segment exists, since the new click creates an additional segment.
  const label = canCreate
    ? t('toolbar.compareSegment')
    : canClearActive
      ? t('toolbar.clearActiveSegment')
      : t('toolbar.compareSegment');

  return (
    <button
      onClick={handleClick}
      disabled={disabled}
      className={styles.tooltipToggle}
      title={t('toolbar.compareSegment.title')}
      style={{
        color: accent,
        borderColor: canCreate || canClearActive ? ACCENT : SURFACE.borderMedium,
        background: canClearActive && !canCreate ? 'rgba(255, 149, 0, 0.1)' : 'transparent',
        opacity: disabled ? 0.5 : 1,
      }}
    >
      {label}
    </button>
  );
}
