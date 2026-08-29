import { useEffect, useId, useRef, useState } from 'react';
import { useI18n } from '../../i18n';
import { usePackStore, type RestartState } from '../../store/packStore';
import { ProgressBar } from '../shared/ProgressBar';
import styles from './RestartOverlay.module.css';

const selectRestart = (state: ReturnType<typeof usePackStore.getState>): RestartState =>
  state.restart;

/**
 * The blocking overlay shown while the server is being restarted under the
 * page (the GPU PyTorch pack swaps the torch wheel, which no process can do to
 * its own interpreter).
 *
 * Blocking is the point. Between the restart request and the reload, every
 * other surface in the app is a lie: the canvas cannot run, the catalog cannot
 * refresh, and a click on either would fail against a server that is mid-exit.
 * So this sits above the modals and the toasts, takes focus, and swallows the
 * two keys that would take the user back to the page underneath.
 *
 * Mounted once at the app root and driven entirely by `packStore.restart` —
 * the handshake itself (health polling, the boot-id comparison, the reload)
 * lives in the store, so closing or reopening anything cannot interrupt it.
 */
export function RestartOverlay() {
  const restart = usePackStore(selectRestart);
  if (restart.phase === 'idle') return null;
  return <RestartOverlayBody restart={restart} />;
}

function RestartOverlayBody({ restart }: { restart: RestartState }) {
  const { t } = useI18n();
  const { phase, startedAt, command } = restart;
  const waiting = phase === 'waiting';
  const titleId = useId();
  const descId = useId();
  const cardRef = useRef<HTMLDivElement | null>(null);
  const reloadRef = useRef<HTMLButtonElement | null>(null);

  // Wall clock rather than a tick count: a laptop that slept through the
  // restart fires no timers, and a counter of turns would claim four seconds
  // had passed after an hour.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!waiting) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [waiting]);
  const seconds = startedAt === null ? 0 : Math.max(0, Math.floor((now - startedAt) / 1000));

  // Focus starts on the card, and MOVES to the reload button the moment a
  // terminal phase appears: that button is the only thing left to do, and the
  // user has been sitting with nothing to press for up to ten minutes.
  useEffect(() => {
    if (waiting) cardRef.current?.focus();
    else reloadRef.current?.focus();
  }, [waiting]);

  // Capture phase, so this runs before any handler on the page underneath.
  // Only while waiting: once a reload button exists, Tab has to reach it.
  useEffect(() => {
    if (!waiting) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Tab' && e.key !== 'Escape') return;
      e.preventDefault();
      e.stopPropagation();
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [waiting]);

  const heading = waiting
    ? t('packs.restart.title')
    : phase === 'timeout'
      ? t('packs.restart.timeout')
      : t('packs.restart.notStarted');

  return (
    <div className={styles.backdrop}>
      <div
        ref={cardRef}
        className={styles.card}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={waiting ? descId : undefined}
        aria-busy={waiting || undefined}
        tabIndex={-1}
      >
        <h2 id={titleId} className={styles.title}>
          {heading}
        </h2>

        {waiting ? (
          <>
            <p id={descId} className={styles.body}>
              {t('packs.restart.body')}
            </p>
            <ProgressBar value={null} label={t('packs.restart.title')} />
            {/* Paired with the bar on purpose — see `.elapsed` in the CSS. */}
            <div className={styles.elapsed}>
              {t('packs.restart.elapsed', { seconds })}
            </div>
          </>
        ) : (
          <>
            {/* No copy button here, unlike the pack cards: the "Copied" toast
                sits at z-index 10000, i.e. BEHIND this overlay, so the button
                would look broken. The command is selectable text instead. */}
            {command !== null && (
              <pre className={styles.commandBlock}>
                <code>{command}</code>
              </pre>
            )}
            <div className={styles.actions}>
              <button
                ref={reloadRef}
                type="button"
                className={styles.reloadBtn}
                onClick={() => window.location.reload()}
              >
                {t('packs.restart.reload')}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
