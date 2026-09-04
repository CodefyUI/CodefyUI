import { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { PackInstallMode } from '../../api/rest';
import { useDialogStore } from '../../store/dialogStore';
import { usePackStore } from '../../store/packStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import { RefreshIcon } from '../shared/Icons';
import { PackActivityPane } from './PackActivityPane';
import { PackCard } from './PackCard';
import styles from './PackCenterModal.module.css';

/** How long a jumped-to card wears the accent ring before it settles down. */
export const HIGHLIGHT_MS = 2000;

/**
 * The Package Center (core#…): every optional pack, what of it is on this
 * machine, and one place to change that.
 *
 * Mounted once at the app root and driven by `uiStore.packCenterOpen`, like the
 * template gallery — and, like it, a pure VIEW: the catalog, the install job,
 * the long-poll follower and the restart handshake all live in `packStore`, so
 * closing this window cannot interrupt a two-gigabyte download.
 */
export function PackCenterModal() {
  const open = useUIStore((s) => s.packCenterOpen);
  if (!open) return null;
  return <PackCenterBody />;
}

function PackCenterBody() {
  const { t } = useI18n();
  const close = useUIStore((s) => s.closePackCenter);
  const focusPackId = useUIStore((s) => s.packCenterFocusPackId);
  const setFocusPack = useUIStore((s) => s.setPackCenterFocus);

  const packs = usePackStore((s) => s.packs);
  const byId = usePackStore((s) => s.byId);
  const loading = usePackStore((s) => s.loading);
  const error = usePackStore((s) => s.error);
  const unsupported = usePackStore((s) => s.unsupported);
  const remoteInstallAllowed = usePackStore((s) => s.remoteInstallAllowed);
  const launchMode = usePackStore((s) => s.launchMode);
  const restartAvailable = usePackStore((s) => s.restartAvailable);
  const gpu = usePackStore((s) => s.gpu);
  const job = usePackStore((s) => s.job);
  const busy = usePackStore((s) => s.busy);
  const cancelling = usePackStore((s) => s.cancelling);
  const refresh = usePackStore((s) => s.refresh);
  const install = usePackStore((s) => s.install);
  const cancel = usePackStore((s) => s.cancel);
  const removeItem = usePackStore((s) => s.removeItem);
  const dismissJob = usePackStore((s) => s.dismissJob);

  const surfaceRef = useRef<HTMLDivElement | null>(null);
  const cardRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const [highlighted, setHighlighted] = useState<string | null>(null);

  // One read on open. The catalog is cheap, the panel is the one surface where
  // "is it installed yet" has to be current, and `refresh` adopts a job that
  // was started in another tab while it is at it.
  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Focus starts inside the panel and goes back where it came from. Not a
  // focus trap — Tab still walks out into the page underneath, exactly as it
  // does in the template gallery and the node detail modal. Trapping is worth
  // doing, but as one change to all three rather than a divergence here.
  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    surfaceRef.current?.focus();
    return () => {
      if (previouslyFocused && previouslyFocused.isConnected) {
        previouslyFocused.focus();
      }
    };
  }, []);

  // A restart is not a modal that can be dismissed: the server this panel
  // describes is on its way out, so Escape and the backdrop are both dead
  // until it comes back. Read from the store rather than subscribed, so a
  // restart does not re-render the whole panel to answer one keypress.
  const closable = useCallback(() => usePackStore.getState().restart.phase === 'idle', []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      // Something can sit ON TOP of this panel: a confirm dialog (removing an
      // item asks first), or the restart overlay, which `closable` answers for.
      // The shortcuts modal is here for a different reason — it renders at
      // `z-index: 9000`, so it opens BEHIND this panel rather than over it, and
      // it has no Escape handler of its own; swallowing the key while it is
      // open is what keeps one press from closing the panel and leaving a
      // shortcuts window nobody can see.
      if (useDialogStore.getState().active !== null) return;
      if (useUIStore.getState().shortcutsModalOpen) return;
      // The Plugin Center is a second window with a second Escape handler on
      // the same key, and it defers to this one for exactly the same reason.
      // Both can be open at once — a toast from either store offers to open
      // the other — and without this, one press would close the window the
      // user is NOT looking at and leave the one they are.
      if (useUIStore.getState().pluginCenterOpen) return;
      if (!closable()) return;
      e.preventDefault();
      close();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [close, closable]);

  // Jump to a pack somebody asked for — from a node badge, a greyed-out select
  // option, or a "install X first" link on another card. Deliberately waits
  // for the card to EXIST: the request usually arrives with the panel, before
  // the catalog has answered, and clearing it early would lose the jump.
  useEffect(() => {
    if (focusPackId === null) return;
    // Own keys only. The map is a bare object keyed by pack id, so a node
    // asking for `constructor` or `toString` would otherwise get a FUNCTION
    // back — truthy, and `scrollIntoView` is not one of its methods, so the
    // panel would die of a TypeError instead of quietly not finding the card.
    const card = Object.prototype.hasOwnProperty.call(cardRefs.current, focusPackId)
      ? cardRefs.current[focusPackId]
      : null;
    if (!card) return;
    card.scrollIntoView({ block: 'nearest' });
    setHighlighted(focusPackId);
    setFocusPack(null);
  }, [focusPackId, packs, setFocusPack]);

  // Its own effect, keyed on the highlight rather than on the request: the
  // request is cleared immediately above, and a timer hung off THAT effect
  // would be torn down by the very change that cleared it.
  useEffect(() => {
    if (highlighted === null) return;
    const timer = setTimeout(() => setHighlighted(null), HIGHLIGHT_MS);
    return () => clearTimeout(timer);
  }, [highlighted]);

  const onInstall = useCallback(
    (packId: string, items: string[] | undefined, mode: PackInstallMode, variant?: string) => {
      void install(packId, mode === 'restart' ? { mode, variant } : { items, mode });
    },
    [install],
  );

  // The pack comes from the banner that asked, not from the store: a confirm
  // dialog stays open across catalog polls, and a `refresh` that adopts
  // another tab's job mid-dialog must not redirect this install to it. No
  // variant — the conflict this retries is a live install of a pack that has
  // no builds to choose between, and the server picks for the GPU pack.
  const onRestartInstall = useCallback(
    (packId: string) => {
      void install(packId, { mode: 'restart' });
    },
    [install],
  );

  // `byId` is built from parsed JSON, so a bare index answers with an
  // inherited member for a pack id like `constructor`.
  const jobPack =
    job !== null && Object.prototype.hasOwnProperty.call(byId, job.packId)
      ? byId[job.packId]
      : undefined;

  return createPortal(
    <div
      className={styles.backdrop}
      onMouseDown={(e) => {
        if (e.target === e.currentTarget && closable()) close();
      }}
    >
      <div
        ref={surfaceRef}
        className={styles.modal}
        role="dialog"
        aria-modal="true"
        aria-label={t('packs.title')}
        tabIndex={-1}
      >
        <div className={styles.header}>
          <div className={styles.titleBlock}>
            <div className={styles.title}>{t('packs.title')}</div>
            <div className={styles.subtitle}>{t('packs.subtitle')}</div>
          </div>
          <button
            type="button"
            className={styles.iconBtn}
            onClick={() => void refresh()}
            disabled={loading}
            title={t('packs.refresh')}
            aria-label={t('packs.refresh')}
          >
            <RefreshIcon />
          </button>
          <button
            type="button"
            className={styles.closeBtn}
            onClick={close}
            title={t('packs.close')}
            aria-label={t('packs.close')}
          >
            &#215;
          </button>
        </div>

        <div className={styles.body}>
          <section className={styles.list} aria-label={t('packs.list')}>
            {/* A refresh over rows that are already on screen must not blank
                them: only a FIRST read with nothing to show says "loading". */}
            {loading && packs.length === 0 && (
              <div className={styles.stateMessage}>{t('packs.loading')}</div>
            )}

            {/* Shown ABOVE the rows rather than instead of them: the store
                keeps the last good catalog through a failed refresh, and
                blanking it would turn one dropped packet into an empty
                Package Center. */}
            {!loading && error !== null && (
              <div className={styles.stateMessage}>
                <div className={styles.errorText}>{t('packs.loadFail', { error })}</div>
                <button type="button" className={styles.retryBtn} onClick={() => void refresh()}>
                  {t('palette.retry')}
                </button>
              </div>
            )}

            {unsupported && (
              <div className={styles.stateMessage}>{t('packs.unsupported')}</div>
            )}

            {!loading && !unsupported && error === null && packs.length === 0 && (
              <div className={styles.stateMessage}>{t('packs.empty')}</div>
            )}

            {packs.map((pack) => (
              <div
                key={pack.id}
                ref={(node) => {
                  cardRefs.current[pack.id] = node;
                }}
              >
                <PackCard
                  pack={pack}
                  byId={byId}
                  job={job}
                  busy={busy[pack.id] === true}
                  highlighted={highlighted === pack.id}
                  canInstall={remoteInstallAllowed}
                  launchMode={launchMode}
                  restartAvailable={restartAvailable}
                  gpu={gpu}
                  onInstall={(items, mode, variant) =>
                    onInstall(pack.id, items, mode, variant)
                  }
                  onRemoveItem={(itemId) => void removeItem(pack.id, itemId)}
                  onFocusPack={setFocusPack}
                />
              </div>
            ))}
          </section>

          <aside className={styles.activity} aria-label={t('packs.activity')}>
            <PackActivityPane
              job={job}
              pack={jobPack}
              restartAvailable={restartAvailable}
              canInstall={remoteInstallAllowed}
              busy={job !== null && busy[job.packId] === true}
              cancelling={cancelling}
              onCancel={() => void cancel()}
              onDismiss={dismissJob}
              onRestartInstall={onRestartInstall}
            />
          </aside>
        </div>

        {!remoteInstallAllowed && (
          <div className={styles.footer}>{t('packs.remoteDisabled')}</div>
        )}
      </div>
    </div>,
    document.body,
  );
}
