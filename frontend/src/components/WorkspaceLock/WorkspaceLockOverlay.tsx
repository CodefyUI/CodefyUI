import { useEffect, useId, useRef, useSyncExternalStore } from 'react';
import { useI18n } from '../../i18n';
import { usePackStore } from '../../store/packStore';
import {
  getWorkspaceLock,
  type WorkspaceLock,
  type WorkspaceRole,
} from '../../utils/workspaceLock';
import styles from './WorkspaceLockOverlay.module.css';

// All three, so a shortcut bound to keyup or keypress is stopped as well.
const KEY_EVENTS = ['keydown', 'keyup', 'keypress'] as const;

type BlockedRole = Exclude<WorkspaceRole, 'pending' | 'editor'>;

/**
 * What a page that does not edit the workspace shows over everything (#554).
 *
 * One overlay over the whole app rather than a read-only flag on each control:
 * a flag has to be remembered by every component that can edit, and the one
 * that forgets loses work. This covers them all at once, keyboard shortcuts
 * included. The autosave gate in `tabPersistence` is the net under it, for
 * the edits no click makes.
 *
 * Mounted once at the app root. It reads no tab, so it is safe over an empty
 * workspace.
 */
export function WorkspaceLockOverlay({
  lock = getWorkspaceLock(),
}: {
  lock?: WorkspaceLock | null;
}) {
  if (lock === null) return null;
  return <LockGate lock={lock} />;
}

function LockGate({ lock }: { lock: WorkspaceLock }) {
  const role = useSyncExternalStore(lock.subscribe, lock.getRole);
  if (role === 'pending' || role === 'editor') return null;
  return <LockCard role={role} onTakeOver={lock.takeOver} />;
}

function LockCard({ role, onTakeOver }: { role: BlockedRole; onTakeOver: () => void }) {
  const { t } = useI18n();
  const titleId = useId();
  const bodyId = useId();
  const hintId = useId();
  const cardRef = useRef<HTMLDivElement | null>(null);
  const editRef = useRef<HTMLButtonElement | null>(null);

  // The restart overlay is the one thing drawn above this card, and while it
  // is up it owns focus: "Edit here" underneath it can be neither seen nor
  // safely pressed. Read from the store that drives that overlay.
  const restartShown = usePackStore((state) => state.restart.phase !== 'idle');
  const canTakeOver = role === 'readonly' || role === 'displaced';
  const moved = role === 'releasing' || role === 'displaced';
  const busy = role === 'releasing' || role === 'claiming';
  const body =
    role === 'releasing'
      ? t('workspaceLock.saving')
      : role === 'claiming'
        ? t('workspaceLock.waiting')
        : role === 'displaced'
          ? t('workspaceLock.moved.body')
          : t('workspaceLock.readOnly.body');

  // On the button when there is one, so Enter takes over; on the card while
  // there is nothing to press. Not while the restart overlay is up.
  useEffect(() => {
    if (restartShown) return;
    if (canTakeOver) editRef.current?.focus();
    else cardRef.current?.focus();
  }, [canTakeOver, restartShown]);

  // Capture phase on the window runs before every handler on the page below,
  // and stopping the event there keeps all of them from seeing it: the
  // canvas's Delete, undo, paste and the rest. Default actions are left
  // alone, so Enter and Space still press the button and the browser's own
  // keys (reload, close) still work. Tab is the one default stopped, because
  // it would walk focus into the page behind -- except under the restart
  // overlay, whose own capture listener handles Tab.
  useEffect(() => {
    const swallow = (event: KeyboardEvent) => {
      event.stopPropagation();
      if (restartShown) return;
      if (event.type === 'keydown' && event.key === 'Tab') {
        event.preventDefault();
        (editRef.current ?? cardRef.current)?.focus();
      }
    };
    for (const type of KEY_EVENTS) window.addEventListener(type, swallow, true);
    return () => {
      for (const type of KEY_EVENTS) window.removeEventListener(type, swallow, true);
    };
  }, [restartShown]);

  return (
    <div className={styles.backdrop}>
      <div
        ref={cardRef}
        className={styles.card}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={bodyId}
        aria-busy={busy || undefined}
        tabIndex={-1}
      >
        <h2 id={titleId} className={styles.title}>
          {moved ? t('workspaceLock.moved.title') : t('workspaceLock.readOnly.title')}
        </h2>
        <p id={bodyId} className={styles.body} aria-live="polite">
          {body}
        </p>
        {canTakeOver && (
          <div className={styles.actions}>
            <p id={hintId} className={styles.hint}>
              {t('workspaceLock.editHere.hint')}
            </p>
            <button
              ref={editRef}
              type="button"
              className={styles.editBtn}
              aria-describedby={hintId}
              onClick={onTakeOver}
            >
              {t('workspaceLock.editHere')}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
