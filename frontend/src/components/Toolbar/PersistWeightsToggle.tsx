import { useTabStore } from '../../store/tabStore';
import { useToastStore } from '../../store/toastStore';
import { useI18n } from '../../i18n';
import { resetWeights } from '../../api/rest';
import { SURFACE, TEXT } from '../../styles/theme';
import styles from './Toolbar.module.css';

/**
 * Two-button group: a persistence toggle (default ON) plus a small "↻"
 * action that asks the backend to drop all NodeStateStore entries for the
 * current graph. Mirrors the style of RecordToggle / VerboseToggle.
 */
export function PersistWeightsToggle() {
  const persistent = useTabStore((s) => {
    const tab = s.tabs.find((t) => t.id === s.activeTabId);
    return tab?.weightsPersistent ?? true;
  });
  const graphId = useTabStore((s) => {
    const tab = s.tabs.find((t) => t.id === s.activeTabId);
    return tab?.graphId ?? '';
  });
  const toggle = useTabStore((s) => s.togglePersistWeights);
  const addToast = useToastStore((s) => s.addToast);
  const { t } = useI18n();

  const onColor = '#a5e063';

  const onResetAll = async () => {
    if (!graphId) return;
    if (!window.confirm(t('toolbar.weights.resetAllConfirm'))) return;
    try {
      const r = await resetWeights(graphId);
      addToast(`${t('toolbar.weights.resetAllOk')} (${r.evicted})`, 'success');
    } catch (e) {
      addToast(`Reset failed: ${(e as Error).message}`, 'error');
    }
  };

  return (
    <span style={{ display: 'inline-flex', gap: 2 }}>
      <button
        onClick={toggle}
        className={styles.tooltipToggle}
        title={t('toolbar.weights.title')}
        style={{
          color: persistent ? onColor : TEXT.muted,
          borderColor: persistent ? onColor : SURFACE.borderMedium,
          background: persistent ? 'rgba(165, 224, 99, 0.1)' : 'transparent',
          display: 'inline-flex',
          alignItems: 'center',
          gap: 6,
        }}
      >
        <span aria-hidden="true" style={{ fontSize: 11, lineHeight: 1 }}>
          {'⚙'}
        </span>
        {t(persistent ? 'toolbar.weights.on' : 'toolbar.weights.off')}
      </button>
      <button
        onClick={onResetAll}
        className={styles.tooltipToggle}
        title={t('toolbar.weights.resetAll')}
        style={{
          color: TEXT.muted,
          borderColor: SURFACE.borderMedium,
          minWidth: 26,
          padding: '0 6px',
        }}
      >
        ↻
      </button>
    </span>
  );
}
