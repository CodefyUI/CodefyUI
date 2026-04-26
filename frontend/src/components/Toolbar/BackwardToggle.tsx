import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { SURFACE, TEXT } from '../../styles/theme';
import styles from './Toolbar.module.css';

/**
 * Two-button group: a backward-mode toggle plus an "auto" sub-toggle.
 * When backward mode is ON the next Run captures gradients via
 * loss.backward(); auto mode synthesises a loss when no Loss/BackwardOnce
 * node is present.
 */
export function BackwardToggle() {
  const backward = useTabStore((s) => {
    const tab = s.tabs.find((t) => t.id === s.activeTabId);
    return tab?.backwardMode ?? false;
  });
  const auto = useTabStore((s) => {
    const tab = s.tabs.find((t) => t.id === s.activeTabId);
    return tab?.autoBackward ?? false;
  });
  const toggleBackward = useTabStore((s) => s.toggleBackward);
  const toggleAuto = useTabStore((s) => s.toggleAutoBackward);
  const { t } = useI18n();

  const onColor = '#bb6bd9';

  return (
    <span style={{ display: 'inline-flex', gap: 2 }}>
      <button
        onClick={toggleBackward}
        className={styles.tooltipToggle}
        title={t('toolbar.backward.title')}
        style={{
          color: backward ? onColor : TEXT.muted,
          borderColor: backward ? onColor : SURFACE.borderMedium,
          background: backward ? 'rgba(187, 107, 217, 0.1)' : 'transparent',
          display: 'inline-flex',
          alignItems: 'center',
          gap: 6,
        }}
      >
        {/* The ∂ glyph lives inside the i18n label so we don't double up. */}
        {t(backward ? 'toolbar.backward.on' : 'toolbar.backward.off')}
      </button>
      <button
        onClick={toggleAuto}
        className={styles.tooltipToggle}
        title={t('toolbar.autoBackward.title')}
        disabled={!backward}
        style={{
          color: auto && backward ? onColor : TEXT.muted,
          borderColor: auto && backward ? onColor : SURFACE.borderMedium,
          background: auto && backward ? 'rgba(187, 107, 217, 0.06)' : 'transparent',
          opacity: backward ? 1 : 0.45,
          fontSize: 11,
          padding: '0 6px',
        }}
      >
        {t(auto ? 'toolbar.autoBackward.on' : 'toolbar.autoBackward.off')}
      </button>
    </span>
  );
}
