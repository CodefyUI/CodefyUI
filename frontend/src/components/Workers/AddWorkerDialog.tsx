import { useEffect, useId, useRef, type KeyboardEvent } from 'react';
import { createPortal } from 'react-dom';
import { useI18n } from '../../i18n';
import { useToastStore } from '../../store/toastStore';
import { trapTab } from './focusTrap';
import styles from './WorkersPanel.module.css';

/**
 * How the dialog tells the user to add a worker. `same-host` (this wave):
 * `cdui agent start` beside the server. #780 adds the remote mode, a
 * one-line enrollment command generated for another host.
 */
export type AddWorkerMode = 'same-host';

export const AGENT_START_COMMAND = 'cdui agent start';
export const AGENT_STOP_COMMAND = 'cdui agent stop';

interface AddWorkerDialogProps {
  mode: AddWorkerMode;
  onClose: () => void;
}

function CommandLine({ command }: { command: string }) {
  const { t } = useI18n();
  const copy = async () => {
    const { addToast } = useToastStore.getState();
    try {
      await navigator.clipboard.writeText(command);
      addToast(t('workers.add.copied'), 'success');
    } catch {
      addToast(t('workers.add.copyFailed'), 'error');
    }
  };
  return (
    <div className={styles.commandLine}>
      <code className={styles.command}>{command}</code>
      <button type="button" className={styles.btn} onClick={() => void copy()}>
        {t('workers.add.copy')}
      </button>
    </div>
  );
}

function SameHostInstructions() {
  const { t } = useI18n();
  return (
    <>
      <p>{t('workers.add.sameHost.intro')}</p>
      <p>{t('workers.add.sameHost.start')}</p>
      <CommandLine command={AGENT_START_COMMAND} />
      <p>{t('workers.add.sameHost.what')}</p>
      <h3 className={styles.subTitle}>{t('workers.add.sameHost.options')}</h3>
      <ul className={styles.options}>
        <li><code>--name &lt;name&gt;</code> {t('workers.add.sameHost.optName')}</li>
        <li><code>--devices cuda:0,cuda:1</code> {t('workers.add.sameHost.optDevices')}</li>
        <li><code>--foreground</code> {t('workers.add.sameHost.optForeground')}</li>
      </ul>
      <p>{t('workers.add.sameHost.stop')}</p>
      <CommandLine command={AGENT_STOP_COMMAND} />
    </>
  );
}

/** The Workers panel's "Add worker" dialog, over the panel. */
export function AddWorkerDialog({ mode, onClose }: AddWorkerDialogProps) {
  const { t } = useI18n();
  const titleId = useId();
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    return () => {
      if (previous?.isConnected) previous.focus();
    };
  }, []);

  const handleKeys = (event: KeyboardEvent<HTMLDivElement>) => {
    // React events bubble through the portal to the Workers panel, whose own
    // Escape would close it too.
    event.stopPropagation();
    if (event.key === 'Escape') {
      event.preventDefault();
      onClose();
      return;
    }
    trapTab(event, dialogRef.current);
  };

  return createPortal(
    <div
      className={`${styles.backdrop} ${styles.backdropTop}`}
      onMouseDown={(event) => {
        event.stopPropagation();
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        ref={dialogRef}
        className={`${styles.dialog} ${styles.dialogNarrow}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        onKeyDown={handleKeys}
        data-mode={mode}
      >
        <header className={styles.header}>
          <h2 id={titleId}>{t('workers.add.title')}</h2>
          <button
            ref={closeRef}
            type="button"
            className={styles.close}
            onClick={onClose}
            aria-label={t('workers.add.close')}
          >
            &#215;
          </button>
        </header>
        <div className={`${styles.body} ${styles.prose}`}>
          <SameHostInstructions />
        </div>
      </div>
    </div>,
    document.body,
  );
}
