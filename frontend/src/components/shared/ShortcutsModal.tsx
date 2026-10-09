import { useEffect, useRef } from 'react';
import { useDialogStore } from '../../store/dialogStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import { MOD_LABEL as mod } from '../../utils/platform';
import styles from './ShortcutsModal.module.css';

/*
 * Stack policy for the shortcuts sheet (#490).
 *
 * The sheet may open over any panel: the Package and Plugin Centers, the
 * template gallery, the diff window, the node, preset, layers and viz
 * windows, the Custom Nodes manager and the New Sweep dialog. It is a help
 * layer, so it renders on its own rung above all of them
 * (`ShortcutsModal.module.css`); only the confirm dialog, the toast stack,
 * the workspace lock and the restart overlay sit higher.
 *
 * While it is open it owns Escape. Its listener runs in the document's
 * capture phase, ahead of every panel's window-level handler and every React
 * handler inside a panel, and stops the press there: one press closes the
 * sheet and nothing under it, and focus goes back to whatever held it when
 * the sheet opened. The panels therefore carry no shortcuts-sheet exception
 * in their own Escape handlers. The three layers above the sheet keep the
 * key: the restart overlay and the workspace lock stop it in the window's
 * capture phase before it gets here, and the confirm dialog, which listens
 * on the window, is let through by the check below.
 *
 * `?` (`useKeyboardShortcuts`) and the toolbar's ? button are the only
 * openers. The toolbar button is behind every panel's backdrop, so over a
 * panel the sheet is reached by `?` alone.
 */
export function ShortcutsModal() {
  const open = useUIStore((s) => s.shortcutsModalOpen);
  if (!open) return null;
  return <ShortcutsSheet />;
}

function ShortcutsSheet() {
  const toggle = useUIStore((s) => s.toggleShortcutsModal);
  const { t } = useI18n();
  const surfaceRef = useRef<HTMLDivElement>(null);

  // Focus moves onto the sheet and goes back where it came from on close, as
  // it does for the panels.
  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    surfaceRef.current?.focus();
    return () => {
      if (previouslyFocused && previouslyFocused.isConnected) {
        previouslyFocused.focus();
      }
    };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      // A confirm dialog renders above the sheet and cancels on Escape itself.
      if (useDialogStore.getState().active !== null) return;
      e.preventDefault();
      e.stopPropagation();
      useUIStore.setState({ shortcutsModalOpen: false });
    };
    document.addEventListener('keydown', onKey, true);
    return () => document.removeEventListener('keydown', onKey, true);
  }, []);

  const shortcuts = [
    { keys: `${mod}+Z`, action: t('shortcuts.undo') },
    { keys: `${mod}+Shift+Z`, action: t('shortcuts.redo') },
    { keys: `${mod}+Y`, action: t('shortcuts.redoAlt') },
    { keys: `${mod}+C`, action: t('shortcuts.copy') },
    { keys: `${mod}+V`, action: t('shortcuts.paste') },
    { keys: `${mod}+S`, action: t('shortcuts.save') },
    { keys: 'Delete', action: t('shortcuts.delete') },
    // Belongs to the Source Control message box rather than the canvas, and
    // says so: the global handler skips every textarea, so this chord only
    // exists while that box has focus.
    { keys: `${mod}+Enter`, action: t('shortcuts.commit') },
    // Ctrl+B is context-sensitive (core#128), so both halves are listed —
    // and the unconditional sidebar chord right after them, because a user
    // who only ever wanted the sidebar needs to know it still exists.
    { keys: `${mod}+B`, action: t('shortcuts.bypass') },
    { keys: `${mod}+B`, action: t('shortcuts.toggleSidebar') },
    { keys: `${mod}+Shift+B`, action: t('shortcuts.toggleSidebarAlways') },
    { keys: t('shortcuts.doubleClickKey'), action: t('shortcuts.quickSearch') },
    { keys: '?', action: t('shortcuts.help') },
  ];

  return (
    <div className={styles.overlay} onClick={toggle}>
      <div
        ref={surfaceRef}
        className={styles.modal}
        role="dialog"
        aria-modal="true"
        aria-label={t('shortcuts.title')}
        tabIndex={-1}
        onClick={(e) => e.stopPropagation()}
      >
        <div className={styles.header}>
          <h3 className={styles.title}>{t('shortcuts.title')}</h3>
          <button
            type="button"
            className={styles.close}
            onClick={toggle}
            aria-label={t('shortcuts.close')}
          >
            &times;
          </button>
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
