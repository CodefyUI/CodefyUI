import { useCallback } from 'react';
import { useI18n } from '../../i18n';
import { useToastStore } from '../../store/toastStore';
import { CopyIcon } from '../shared/Icons';
import styles from './PackCenterModal.module.css';

/**
 * A terminal command the user has to run, with a copy button.
 *
 * Shown in two places that mean the same thing — the GPU card, which cannot
 * swap the torch wheel from inside the running interpreter, and the
 * `needs_restart` banner, which reports an install that got as far as it can
 * without one — so it is one component rather than two that drift.
 *
 * The text stays selectable whatever the clipboard does: `navigator.clipboard`
 * is absent outside a secure context (`cdui start --host <LAN-IP>` over plain
 * http, for one) and can be refused by permission, and the fallback for both
 * is the same sentence telling the user to select it by hand.
 */
export function CommandBlock({ command }: { command: string }) {
  const { t } = useI18n();

  const copy = useCallback(() => {
    const { addToast } = useToastStore.getState();
    // Wrapped in a promise chain rather than called directly, so a missing
    // `navigator.clipboard` throws INTO the rejection path instead of past it.
    void Promise.resolve()
      .then(() => navigator.clipboard.writeText(command))
      .then(
        () => addToast(t('packs.copied'), 'success'),
        () => addToast(t('packs.copyFailed'), 'error'),
      );
  }, [command, t]);

  return (
    <div className={styles.commandRow}>
      <pre className={styles.commandBlock}>
        <code>{command}</code>
      </pre>
      <button
        type="button"
        className={styles.iconBtn}
        onClick={copy}
        title={t('packs.copy')}
        aria-label={t('packs.copy')}
      >
        <CopyIcon />
      </button>
    </div>
  );
}
