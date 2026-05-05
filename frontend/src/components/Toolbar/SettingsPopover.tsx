import { useEffect, useRef } from 'react';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import { useToastStore } from '../../store/toastStore';
import { useI18n } from '../../i18n';
import { resetWeights } from '../../api/rest';
import { computeSegmentNodes } from '../../utils/segmentPath';
import { generateId } from '../../utils';
import styles from './SettingsPopover.module.css';

interface Props {
  open: boolean;
  onClose: () => void;
  triggerRef: React.RefObject<HTMLButtonElement | null>;
}

export function SettingsPopover({ open, onClose, triggerRef }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const { t } = useI18n();

  // Per-tab settings (Recording / Training)
  const activeTab = useTabStore((s) => s.tabs.find((tab) => tab.id === s.activeTabId)!);
  const recording = activeTab.recordOutputs ?? true;
  const verbose = activeTab.verboseMode ?? false;
  const persistent = activeTab.weightsPersistent ?? true;
  const backward = activeTab.backwardMode ?? false;
  const autoBackward = activeTab.autoBackward ?? false;
  const graphId = activeTab.graphId ?? '';
  const activeSegment = activeTab.activeSegment;

  const toggleRecord = useTabStore((s) => s.toggleRecord);
  const toggleVerbose = useTabStore((s) => s.toggleVerbose);
  const togglePersistWeights = useTabStore((s) => s.togglePersistWeights);
  const toggleBackward = useTabStore((s) => s.toggleBackward);
  const toggleAutoBackward = useTabStore((s) => s.toggleAutoBackward);
  const setActiveSegment = useTabStore((s) => s.setActiveSegment);
  const addSegmentGroup = useTabStore((s) => s.addSegmentGroup);
  const removeSegmentGroup = useTabStore((s) => s.removeSegmentGroup);

  // Global UI settings (Editor section)
  const gridSnapEnabled = useUIStore((s) => s.gridSnapEnabled);
  const tooltipsEnabled = useUIStore((s) => s.tooltipsEnabled);
  const beginnerMode = useUIStore((s) => s.beginnerMode);
  const toggleGridSnap = useUIStore((s) => s.toggleGridSnap);
  const toggleTooltips = useUIStore((s) => s.toggleTooltips);
  const toggleBeginnerMode = useUIStore((s) => s.toggleBeginnerMode);

  const addToast = useToastStore((s) => s.addToast);

  // Compare segment selection state
  const selected = activeTab.nodes.filter((n) => n.selected);
  const canCreateSegment = selected.length === 2;
  const canClearSegment = activeSegment !== null;

  // Close on outside click / Esc
  useEffect(() => {
    if (!open) return;
    const handleMouseDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (ref.current?.contains(target)) return;
      if (triggerRef.current?.contains(target)) return;
      onClose();
    };
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('mousedown', handleMouseDown);
    document.addEventListener('keydown', handleKey);
    return () => {
      document.removeEventListener('mousedown', handleMouseDown);
      document.removeEventListener('keydown', handleKey);
    };
  }, [open, onClose, triggerRef]);

  if (!open) return null;

  const handleResetWeights = async () => {
    if (!graphId) return;
    if (!window.confirm(t('toolbar.weights.resetAllConfirm'))) return;
    try {
      const r = await resetWeights(graphId);
      addToast(`${t('toolbar.weights.resetAllOk')} (${r.evicted})`, 'success');
    } catch (e) {
      addToast(`${t('toolbar.weights.resetAll')}: ${(e as Error).message}`, 'error');
    }
  };

  const handleCompare = () => {
    if (canCreateSegment) {
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
      onClose();
      return;
    }
    if (canClearSegment && activeSegment) {
      removeSegmentGroup(activeSegment.id);
      return;
    }
    addToast(t('toolbar.compareSegment.needTwo'), 'warning');
  };

  const compareLabel = canCreateSegment
    ? t('settings.compare.actionCreate')
    : canClearSegment
      ? t('settings.compare.actionClear')
      : t('settings.compare.actionDisabled');
  const compareDisabled = !canCreateSegment && !canClearSegment;

  return (
    <div ref={ref} className={styles.panel} role="dialog" aria-label={t('toolbar.settings')}>
      <div className={styles.head}>
        <h4>{t('toolbar.settings')}</h4>
        {/* Search input deferred until per-toggle filtering lands — a
            non-functional input at the top of the popover is more confusing
            than helpful. The i18n key `toolbar.settings.search` is kept for
            when it returns. */}
      </div>

      <div className={styles.body}>
        {/* ── Recording & Inspection ─────────────────────────────── */}
        <section className={styles.section}>
          <div className={styles.sectionTitle}>
            {t('toolbar.settings.section.recording')}
          </div>

          <Row
            name={t('settings.record.name')}
            desc={t('settings.record.desc')}
            onClick={toggleRecord}
            ctrl={
              <button
                type="button"
                aria-label={t('settings.record.name')}
                aria-pressed={recording}
                className={`${styles.toggle} ${recording ? styles.on : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  toggleRecord();
                }}
              />
            }
          />

          <Row
            name={t('settings.verbose.name')}
            desc={t('settings.verbose.desc')}
            onClick={toggleVerbose}
            ctrl={
              <button
                type="button"
                aria-label={t('settings.verbose.name')}
                aria-pressed={verbose}
                className={`${styles.toggle} ${verbose ? styles.on : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  toggleVerbose();
                }}
              />
            }
          />

          <Row
            name={t('settings.compare.name')}
            desc={t('settings.compare.desc')}
            ctrl={
              <button
                type="button"
                disabled={compareDisabled}
                onClick={(e) => {
                  e.stopPropagation();
                  handleCompare();
                }}
                className={`${styles.action} ${canClearSegment && !canCreateSegment ? styles.accent : ''}`}
              >
                {compareLabel}
              </button>
            }
          />
        </section>

        {/* ── Training Behavior ──────────────────────────────────── */}
        <section className={styles.section}>
          <div className={styles.sectionTitle}>
            {t('toolbar.settings.section.training')}
          </div>

          <Row
            name={t('settings.persist.name')}
            desc={t('settings.persist.desc')}
            onClick={togglePersistWeights}
            ctrl={
              <button
                type="button"
                aria-label={t('settings.persist.name')}
                aria-pressed={persistent}
                className={`${styles.toggle} ${persistent ? styles.on : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  togglePersistWeights();
                }}
              />
            }
          />

          <Row
            name={t('settings.resetWeights.name')}
            desc={t('settings.resetWeights.desc')}
            ctrl={
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  handleResetWeights();
                }}
                className={`${styles.action} ${styles.danger}`}
                disabled={!graphId}
              >
                {t('settings.resetWeights.action')}
              </button>
            }
          />

          <Row
            name={t('settings.gradients.name')}
            desc={t('settings.gradients.desc')}
            onClick={toggleBackward}
            ctrl={
              <button
                type="button"
                aria-label={t('settings.gradients.name')}
                aria-pressed={backward}
                className={`${styles.toggle} ${backward ? styles.on : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  toggleBackward();
                }}
              />
            }
          />

          <Row
            name={t('settings.autoLoss.name')}
            desc={t('settings.autoLoss.desc')}
            disabled={!backward}
            onClick={backward ? toggleAutoBackward : undefined}
            ctrl={
              <button
                type="button"
                aria-label={t('settings.autoLoss.name')}
                aria-pressed={autoBackward}
                disabled={!backward}
                className={`${styles.toggle} ${autoBackward && backward ? styles.on : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  if (backward) toggleAutoBackward();
                }}
              />
            }
          />
        </section>

        {/* ── Editor ─────────────────────────────────────────────── */}
        <section className={styles.section}>
          <div className={styles.sectionTitle}>
            {t('toolbar.settings.section.editor')}
          </div>

          <Row
            name={t('settings.gridSnap.name')}
            desc={t('settings.gridSnap.desc')}
            onClick={toggleGridSnap}
            ctrl={
              <button
                type="button"
                aria-label={t('settings.gridSnap.name')}
                aria-pressed={gridSnapEnabled}
                className={`${styles.toggle} ${gridSnapEnabled ? styles.on : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  toggleGridSnap();
                }}
              />
            }
          />

          <Row
            name={t('settings.tooltips.name')}
            desc={t('settings.tooltips.desc')}
            onClick={toggleTooltips}
            ctrl={
              <button
                type="button"
                aria-label={t('settings.tooltips.name')}
                aria-pressed={tooltipsEnabled}
                className={`${styles.toggle} ${tooltipsEnabled ? styles.on : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  toggleTooltips();
                }}
              />
            }
          />

          <Row
            name={t('settings.nodeMode.name')}
            desc={t('settings.nodeMode.desc')}
            ctrl={
              <div className={styles.seg} role="group" aria-label={t('settings.nodeMode.name')}>
                <button
                  type="button"
                  className={`${styles.segItem} ${beginnerMode ? styles.active : ''}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (!beginnerMode) toggleBeginnerMode();
                  }}
                >
                  {t('settings.nodeMode.basic')}
                </button>
                <button
                  type="button"
                  className={`${styles.segItem} ${!beginnerMode ? styles.active : ''}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (beginnerMode) toggleBeginnerMode();
                  }}
                >
                  {t('settings.nodeMode.all')}
                </button>
              </div>
            }
          />
        </section>
      </div>
    </div>
  );
}

interface RowProps {
  name: string;
  desc: string;
  ctrl: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
}

function Row({ name, desc, ctrl, onClick, disabled }: RowProps) {
  const interactive = onClick !== undefined;
  return (
    <div
      className={`${styles.row} ${interactive ? styles.interactive : ''} ${disabled ? styles.disabled : ''}`}
      onClick={onClick}
      role={interactive ? 'button' : undefined}
      tabIndex={interactive ? 0 : undefined}
      onKeyDown={(e) => {
        if (interactive && (e.key === 'Enter' || e.key === ' ')) {
          e.preventDefault();
          onClick?.();
        }
      }}
    >
      <div>
        <div className={styles.name}>{name}</div>
        <div className={styles.desc}>{desc}</div>
      </div>
      <div className={styles.ctrl}>{ctrl}</div>
    </div>
  );
}
