import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { PluginCatalogEntry } from '../../api/rest';
import { useDialogStore } from '../../store/dialogStore';
import { usePluginStore } from '../../store/pluginStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import { RefreshIcon } from '../shared/Icons';
// The pack panel's, not a copy: one duration for "this is the row you asked
// for" across both windows, and one place to change it.
import { HIGHLIGHT_MS } from '../PackCenter/PackCenterModal';
import { PluginCard } from './PluginCard';
import { PluginFilterBar } from './PluginFilterBar';
import { PluginReviewCard } from './PluginReviewCard';
import { PluginSourceForm } from './PluginSourceForm';
import { matchesFilter, type PluginFilter } from './pluginStatus';
import styles from '../PackCenter/PackCenterModal.module.css';

/**
 * The Plugin Center (core#…): every plugin this server knows about, what of it
 * is installed, and one place to change that.
 *
 * Mounted once at the app root and driven by `uiStore.pluginCenterOpen`, like
 * the Package Center — and, like it, a pure VIEW: the catalog, the install
 * job, the long-poll follower and the confirm dialogs all live in
 * `pluginStore`, so closing this window cannot interrupt an install.
 *
 * The chrome is the Package Center's, imported rather than reimplemented: the
 * backdrop, the modal, the header, the list, the pane and the 860 px stack are
 * one stylesheet, so the two surfaces cannot drift apart.
 */
export function PluginCenterModal() {
  const open = useUIStore((s) => s.pluginCenterOpen);
  if (!open) return null;
  return <PluginCenterBody />;
}

/**
 * The node types *pluginId* registers, as the catalog last said.
 *
 * Own keys only: `byId` is a bare object built from parsed JSON, so a plugin
 * called `constructor` would otherwise hand back a function and `.nodes`
 * would be undefined on it.
 */
function ownNodes(byId: Record<string, PluginCatalogEntry>, pluginId: string): string[] {
  return Object.prototype.hasOwnProperty.call(byId, pluginId)
    ? byId[pluginId].nodes
    : [];
}

function PluginCenterBody() {
  const { t } = useI18n();
  const close = useUIStore((s) => s.closePluginCenter);
  const focusPluginId = useUIStore((s) => s.pluginCenterFocusPluginId);
  const setFocus = useUIStore((s) => s.setPluginCenterFocus);

  const plugins = usePluginStore((s) => s.plugins);
  const byId = usePluginStore((s) => s.byId);
  const loading = usePluginStore((s) => s.loading);
  const error = usePluginStore((s) => s.error);
  const unsupported = usePluginStore((s) => s.unsupported);
  const remoteInstallAllowed = usePluginStore((s) => s.remoteInstallAllowed);
  const job = usePluginStore((s) => s.job);
  const busy = usePluginStore((s) => s.busy);
  const inspection = usePluginStore((s) => s.inspection);
  const refresh = usePluginStore((s) => s.refresh);
  const install = usePluginStore((s) => s.install);
  const inspect = usePluginStore((s) => s.inspect);
  const installInspected = usePluginStore((s) => s.installInspected);
  const clearInspection = usePluginStore((s) => s.clearInspection);
  const update = usePluginStore((s) => s.update);
  const uninstall = usePluginStore((s) => s.uninstall);
  const setEnabled = usePluginStore((s) => s.setEnabled);

  const surfaceRef = useRef<HTMLDivElement | null>(null);
  const cardRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const [highlighted, setHighlighted] = useState<string | null>(null);
  const [filter, setFilter] = useState<PluginFilter>('all');

  // One read on open. The catalog is cheap, this panel is the one surface
  // where "is it installed yet" has to be current, and `refresh` adopts a job
  // started in another tab while it is at it.
  useEffect(() => {
    void refresh();
  }, [refresh]);

  // A refusal belongs to the box it was typed into, and does not outlive the
  // window that box was in: reopening the panel over "Could not fetch
  // owner/demo: ..." would explain a request nobody here made. A REVIEW
  // survives a close on purpose — it is a decision still waiting for an
  // answer, and the store keeps the install job running behind it either way.
  useEffect(() => () => {
    if (usePluginStore.getState().inspection.phase === 'error') clearInspection();
  }, [clearInspection]);

  // Focus starts inside the panel and goes back where it came from. Not a
  // focus trap — Tab still walks out into the page underneath, exactly as it
  // does in the Package Center and the template gallery. Trapping is worth
  // doing, but as one change to all of them rather than a divergence here.
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
      // Something can sit ON TOP of this panel: a confirm dialog (uninstall
      // asks first), or the shortcuts modal, which renders at `z-index: 9000`
      // — behind this panel rather than over it — and has no Escape handler of
      // its own, so swallowing the key is what keeps one press from closing
      // this panel and leaving a shortcuts window nobody can see.
      if (useDialogStore.getState().active !== null) return;
      if (useUIStore.getState().shortcutsModalOpen) return;
      // The Package Center is a second window with a second Escape handler on
      // the same key. Both can be open at once — a node badge opens one, the
      // sidebar the other — and without this, one press would close both.
      if (useUIStore.getState().packCenterOpen) return;
      e.preventDefault();
      close();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [close]);

  // Jump to the plugin somebody asked for — from a toast, the sidebar, or the
  // settings popover. Deliberately waits for the card to EXIST: the request
  // usually arrives with the panel, before the catalog has answered, and
  // clearing it early would lose the jump.
  useEffect(() => {
    if (focusPluginId === null) return;
    // Own keys only. The map is a bare object keyed by plugin id, so a plugin
    // called `constructor` or `toString` would otherwise hand back a FUNCTION
    // — truthy, and `scrollIntoView` is not one of its methods, so the panel
    // would die of a TypeError instead of quietly not finding the card.
    const card = Object.prototype.hasOwnProperty.call(cardRefs.current, focusPluginId)
      ? cardRefs.current[focusPluginId]
      : null;
    if (!card) {
      // The catalog HAS this plugin and the filter is what is hiding it. A
      // toast that says "open the Plugin Center on demo" must not land on a
      // list without demo in it, so the filter gives way to the request.
      if (
        filter !== 'all'
        && Object.prototype.hasOwnProperty.call(byId, focusPluginId)
      ) {
        setFilter('all');
      }
      return;
    }
    card.scrollIntoView({ block: 'nearest' });
    setHighlighted(focusPluginId);
    setFocus(null);
  }, [focusPluginId, plugins, filter, byId, setFocus]);

  // Its own effect, keyed on the highlight rather than on the request: the
  // request is cleared immediately above, and a timer hung off THAT effect
  // would be torn down by the very change that cleared it.
  useEffect(() => {
    if (highlighted === null) return;
    const timer = setTimeout(() => setHighlighted(null), HIGHLIGHT_MS);
    return () => clearTimeout(timer);
  }, [highlighted]);

  const visible = plugins.filter((entry) => matchesFilter(filter, entry.status));

  return createPortal(
    <div
      className={styles.backdrop}
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) close();
      }}
    >
      <div
        ref={surfaceRef}
        className={styles.modal}
        role="dialog"
        aria-modal="true"
        aria-label={t('pluginCenter.title')}
        tabIndex={-1}
      >
        <div className={styles.header}>
          <div className={styles.titleBlock}>
            <div className={styles.title}>{t('pluginCenter.title')}</div>
            <div className={styles.subtitle}>{t('pluginCenter.subtitle')}</div>
          </div>
          <button
            type="button"
            className={styles.iconBtn}
            onClick={() => void refresh()}
            disabled={loading}
            title={t('pluginCenter.refresh')}
            aria-label={t('pluginCenter.refresh')}
          >
            <RefreshIcon />
          </button>
          <button
            type="button"
            className={styles.closeBtn}
            onClick={close}
            title={t('pluginCenter.close')}
            aria-label={t('pluginCenter.close')}
          >
            &#215;
          </button>
        </div>

        <div className={styles.body}>
          <section className={styles.list} aria-label={t('pluginCenter.list')}>
            {/* Above the filter, because both are about a plugin that is not
                in the list yet. A server with no Plugin Center is offered
                neither. */}
            {!unsupported && (
              <PluginSourceForm
                inspection={inspection}
                canInstall={remoteInstallAllowed}
                onReview={(source) => void inspect(source)}
              />
            )}

            {/* Not on the phase alone. `install()` inspects before it
                installs, and `runInspect` leaves the review READY between
                those two round trips — so a built-in pack, which asks for
                nothing and is installed straight from that inspection, would
                flash a consent card at the top of the list and scroll to it
                for as long as the install request takes.

                The card is rendered for the question it answers instead: a
                source somebody typed (`forPluginId === null`), a plugin that
                asks for something, or a refusal a control on this card is the
                fix for. The window the store passes through on its way to an
                automatic install is the only thing this drops.

                Keyed on the inspection: a second Review answers a different
                manifest, and the boxes ticked for the first one must not
                carry over to it. */}
            {!unsupported && inspection.phase === 'ready'
              && (inspection.forPluginId === null
                || inspection.data.consent_required
                || inspection.error !== null) && (
              <PluginReviewCard
                key={inspection.data.inspection_id}
                inspection={inspection}
                nodes={ownNodes(byId, inspection.data.plugin_id)}
                busy={busy[inspection.data.plugin_id] === true}
                canInstall={remoteInstallAllowed}
                onInstall={(opts) => void installInspected(opts)}
                onCancel={clearInspection}
              />
            )}

            {/* No filter over a list nobody can act on: a server with no
                Plugin Center has no rows, and neither has one whose catalog
                has not answered yet. */}
            {!unsupported && plugins.length > 0 && (
              <PluginFilterBar value={filter} onChange={setFilter} />
            )}

            {/* A refresh over rows that are already on screen must not blank
                them: only a FIRST read with nothing to show says "loading". */}
            {loading && plugins.length === 0 && (
              <div className={styles.stateMessage}>{t('pluginCenter.loading')}</div>
            )}

            {/* Shown ABOVE the rows rather than instead of them: the store
                keeps the last good catalog through a failed refresh, and
                blanking it would turn one dropped packet into an empty Plugin
                Center. */}
            {!loading && error !== null && (
              <div className={styles.stateMessage}>
                <div className={styles.errorText}>{t('pluginCenter.loadFail', { error })}</div>
                <button type="button" className={styles.retryBtn} onClick={() => void refresh()}>
                  {t('palette.retry')}
                </button>
              </div>
            )}

            {unsupported && (
              <div className={styles.stateMessage}>{t('pluginCenter.unsupported')}</div>
            )}

            {/* About the CATALOG, not about the filter: "no plugins" over a
                list the user has just narrowed to Available would be the panel
                answering its own question wrongly. A filter that matches
                nothing shows the pressed button and an empty list, which says
                the same thing without claiming the server has nothing. */}
            {!loading && !unsupported && error === null && plugins.length === 0 && (
              <div className={styles.stateMessage}>{t('pluginCenter.empty')}</div>
            )}

            {visible.map((entry) => (
              <div
                key={entry.id}
                ref={(node) => {
                  cardRefs.current[entry.id] = node;
                }}
              >
                <PluginCard
                  entry={entry}
                  job={job}
                  busy={busy[entry.id] === true}
                  highlighted={highlighted === entry.id}
                  canInstall={remoteInstallAllowed}
                  onInstall={() => void install(entry.id)}
                  onUpdate={() => void update(entry.id)}
                  onUninstall={() => void uninstall(entry.id)}
                  onSetEnabled={(enabled) => void setEnabled(entry.id, enabled)}
                />
              </div>
            ))}
          </section>

          {/* The activity pane lands here: what is installing, how far it has
              got, and how it ended. Its own element already, so that arrival
              is a swap rather than a second aside. */}
          <aside className={styles.activity} aria-label={t('packs.activity')} />
        </div>

        {!remoteInstallAllowed && (
          <div className={styles.footer}>{t('packs.remoteDisabled')}</div>
        )}
      </div>
    </div>,
    document.body,
  );
}
