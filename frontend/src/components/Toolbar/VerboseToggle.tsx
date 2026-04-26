import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { SURFACE, TEXT } from '../../styles/theme';
import styles from './Toolbar.module.css';

/**
 * Toggle the per-tab "verbose / step-trace" mode. When ON the next Run
 * sends `verbose_mode: true` to the backend; instrumented nodes record
 * intermediate algorithm tensors via StepRecorder so the Inspector
 * Steps tab can display them.
 */
export function VerboseToggle() {
  const verbose = useTabStore((s) => {
    const tab = s.tabs.find((t) => t.id === s.activeTabId);
    return tab?.verboseMode ?? false;
  });
  const toggle = useTabStore((s) => s.toggleVerbose);
  const { t } = useI18n();

  // Match the Inspector accent (orange) — Verbose mode feeds the Inspector's
  // Steps tab, so the visual coupling tells students "turn this on, look there".
  const onColor = '#ff9500';

  return (
    <button
      onClick={toggle}
      className={styles.tooltipToggle}
      title={t('toolbar.verbose.title')}
      style={{
        color: verbose ? onColor : TEXT.muted,
        borderColor: verbose ? onColor : SURFACE.borderMedium,
        background: verbose ? 'rgba(255, 149, 0, 0.1)' : 'transparent',
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
      }}
    >
      <span aria-hidden="true" style={{ fontSize: 11, lineHeight: 1 }}>
        {'{ƒ}'}
      </span>
      {t(verbose ? 'toolbar.verbose.on' : 'toolbar.verbose.off')}
    </button>
  );
}
