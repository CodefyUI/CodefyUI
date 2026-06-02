import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import styles from './ShortcutsModal.module.css';

const isMac = navigator.platform.toUpperCase().includes('MAC');
const mod = isMac ? 'Cmd' : 'Ctrl';

export function ShortcutsModal() {
  const open = useUIStore((s) => s.shortcutsModalOpen);
  const toggle = useUIStore((s) => s.toggleShortcutsModal);
  const { t } = useI18n();

  if (!open) return null;

  const shortcuts = [
    { keys: `${mod}+Z`, action: t('shortcuts.undo') },
    { keys: `${mod}+Shift+Z`, action: t('shortcuts.redo') },
    { keys: `${mod}+Y`, action: t('shortcuts.redoAlt') },
    { keys: `${mod}+C`, action: t('shortcuts.copy') },
    { keys: `${mod}+V`, action: t('shortcuts.paste') },
    { keys: 'Delete', action: t('shortcuts.delete') },
    { keys: t('shortcuts.doubleClickKey'), action: t('shortcuts.quickSearch') },
    { keys: '?', action: t('shortcuts.help') },
  ];

  return (
    <div className={styles.overlay} onClick={toggle}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <h3 className={styles.title}>{t('shortcuts.title')}</h3>
          <button type="button" className={styles.close} onClick={toggle}>&times;</button>
        </div>
        <div className={styles.list}>
          {shortcuts.map((s, i) => (
            <div key={i} className={styles.row}>
              <kbd className={styles.keys}>{s.keys}</kbd>
              <span className={styles.action}>{s.action}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
