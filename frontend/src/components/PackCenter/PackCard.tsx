import { useState } from 'react';
import type {
  LaunchMode,
  PackGpuInfo,
  PackInstallMode,
  PackStatus,
  PackSummary,
} from '../../api/rest';
import type { PackItemProgress, PackJob } from '../../store/packStore';
import { useI18n } from '../../i18n';
import { localizedPackTitle, type PackIndex } from '../../utils/packAvailability';
import { GpuPackDetails } from './GpuPackDetails';
import { PackItemRow } from './PackItemRow';
import {
  catalogKey,
  formatBytes,
  missingItems,
  statusKey,
  statusTone,
} from './packStatus';
import styles from './PackCenterModal.module.css';

export function StatusPill({ status }: { status: PackStatus }) {
  const { t } = useI18n();
  const tone = statusTone(status);
  return (
    <span className={styles.pill} data-tone={tone}>
      {status === 'installing' && (
        // A pulsing dot rather than a spinner: it says "still going" without
        // claiming to know how far, and it stops dead under reduced motion.
        <span className={styles.pillDot} data-role="pulse" aria-hidden="true" />
      )}
      {t(statusKey(status))}
    </span>
  );
}

export interface PackCardProps {
  pack: PackSummary;
  /** The whole catalog, for naming and grading this pack's dependencies. */
  byId: PackIndex;
  /** The install job in flight anywhere, or null. May be another pack's. */
  job: PackJob | null;
  /** This pack has an install request in flight. */
  busy: boolean;
  highlighted: boolean;
  /** False when the server refuses installs from this browser (remote). */
  canInstall: boolean;
  launchMode: LaunchMode;
  /** The server's own answer to "can I install this and come back?". */
  restartAvailable: boolean;
  gpu: PackGpuInfo | null;
  onInstall: (items: string[] | undefined, mode: PackInstallMode, variant?: string) => void;
  onRemoveItem: (itemId: string) => void;
  onFocusPack: (packId: string) => void;
}

/**
 * One pack: what it is, what it costs, what of it is already here, and the one
 * button that changes that.
 *
 * The selection lives HERE rather than in the modal because it is per-pack
 * scratch state with a lifetime shorter than the panel's: it is seeded from
 * the catalog, and the catalog is the thing that invalidates it.
 */
export function PackCard({
  pack,
  byId,
  job,
  busy,
  highlighted,
  canInstall,
  launchMode,
  restartAvailable,
  gpu,
  onInstall,
  onRemoveItem,
  onFocusPack,
}: PackCardProps) {
  const { t } = useI18n();

  // The seed changes when — and only when — an item's state changes on disk.
  // Comparing the packs themselves would reseed on every catalog poll and
  // throw away a selection the user was in the middle of making.
  const signature = pack.items.map((item) => `${item.id}:${item.status}`).join('|');
  const [selected, setSelected] = useState<Set<string>>(() => defaultSelection(pack));
  const [seenSignature, setSeenSignature] = useState(signature);
  if (seenSignature !== signature) {
    // React's documented "adjust state while rendering" pattern: cheaper than
    // an effect, and it avoids the extra committed frame where the boxes are
    // still ticked for a model that has already landed.
    setSeenSignature(signature);
    setSelected(defaultSelection(pack));
  }

  const jobHere = job !== null && job.packId === pack.id;
  const running = jobHere && job.status === 'running';
  const locked = busy || running;

  const title = localizedPackTitle(t, byId, pack.id);
  const descKey = catalogKey(pack.id, 'desc');
  const description = descKey !== null ? t(descKey) : pack.description;

  const chosen = pack.items.filter((item) => selected.has(item.id));
  const chosenBytes = chosen.reduce((sum, item) => sum + item.size_bytes, 0);
  const blockedBy = pack.blocked_by;

  // The Python half can be missing while every FILE is already on disk — an
  // install that fetched the models and then failed at pip, or a pack whose
  // wheels were removed since. There is still something to install, and the
  // backend runs exactly the pip step for an empty `items`, so the button
  // must stay alive with nothing ticked.
  const pipMissing = pack.pip.length > 0 && !pack.pip_ready;
  const nothingToDo = chosen.length === 0 && !pipMissing;

  const disabled = !canInstall || locked || blockedBy.length > 0 || nothingToDo;
  const reason = !canInstall
    ? t('packs.remoteDisabled')
    : blockedBy.length > 0
      ? t('packs.dependsOnMissing', { pack: localizedPackTitle(t, byId, blockedBy[0]) })
      : nothingToDo
        // Its own sentence, not another control's label: a tooltip explaining
        // why a button is dead has to say what to DO next.
        ? t('packs.selectSomething')
        : undefined;

  const toggle = (itemId: string) =>
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(itemId)) next.delete(itemId);
      else next.add(itemId);
      return next;
    });

  return (
    <div
      className={`${styles.card} ${highlighted ? styles.cardHighlighted : ''}`}
      data-status={pack.status}
      data-pack-id={pack.id}
    >
      <div className={styles.cardHeader}>
        <span className={styles.cardTitle}>{title}</span>
        <StatusPill status={pack.status} />
        {pack.size_bytes_total > 0 && (
          <span className={styles.cardSize}>
            {t('packs.size', { size: formatBytes(pack.size_bytes_total) })}
          </span>
        )}
      </div>

      <p className={styles.cardDesc}>{description}</p>

      {pack.pip.length > 0 && (
        <div className={styles.cardMeta}>
          {t('packs.pip', { specs: pack.pip.map((entry) => entry.spec).join(', ') })}
          {/* Said out loud in exactly one case: every model is already on disk
              and the libraries are not, which is why the Install button is
              alive with no box ticked. In every other state the pack's status
              pill already says 未安裝 / 已安裝 and this would repeat it. Each
              part keeps its own element so the specs and the state are each
              one phrase to a reader — and to a query — rather than a run-on.
              The dash between them is punctuation, not information. */}
          {pipMissing && missingItems(pack).length === 0 && (
            <>
              {' '}
              <span aria-hidden="true">—</span>{' '}
              <span>{t('packs.pipMissing')}</span>
            </>
          )}
        </div>
      )}

      {pack.depends_on.length > 0 && (
        <div className={styles.deps}>
          <span>{t('packs.dependsOnLabel')}</span>
          {pack.depends_on.map((depId) => {
            // `byId` is built from parsed JSON, so a bare index answers with
            // an inherited member for an id like `constructor`.
            const dep = Object.prototype.hasOwnProperty.call(byId, depId)
              ? byId[depId]
              : undefined;
            const name = localizedPackTitle(t, byId, depId);
            return (
              <span key={depId} className={styles.dep}>
                {/* The NAME is the link, not a second sentence beside it: the
                    line used to read "需要先安裝：句向量模型 未安裝 請先安裝
                    句向量模型". `dependsOnMissing` survives as the control's
                    accessible name, where it says what the click does. */}
                {blockedBy.includes(depId) ? (
                  <button
                    type="button"
                    className={styles.linkBtn}
                    title={t('packs.dependsOnMissing', { pack: name })}
                    aria-label={t('packs.dependsOnMissing', { pack: name })}
                    onClick={() => onFocusPack(depId)}
                  >
                    {name}
                  </button>
                ) : (
                  <span>{name}</span>
                )}
                {dep && <StatusPill status={dep.status} />}
              </span>
            );
          })}
        </div>
      )}

      {pack.install_mode === 'restart' ? (
        <GpuPackDetails
          pack={pack}
          gpu={gpu}
          launchMode={launchMode}
          canInstall={canInstall}
          busy={locked}
          restartAvailable={restartAvailable}
          onInstall={(variant) => onInstall(undefined, 'restart', variant)}
        />
      ) : (
        <>
          {pack.items.length > 0 && (
            <div className={styles.items}>
              {pack.items.map((item) => (
                <PackItemRow
                  key={item.id}
                  item={item}
                  checked={selected.has(item.id)}
                  onToggle={toggle}
                  // Only this pack's job describes this pack's items; a job on
                  // another pack must not paint bars on these rows — and
                  // neither must one that has STOPPED. Every requested item is
                  // seeded with a zero-byte entry and a settled job keeps its
                  // items, so a cancel before the first byte otherwise left an
                  // empty bar reading "0 B" on rows nothing ever downloaded.
                  progress={
                    jobHere && job.status === 'running'
                      ? itemProgress(job, item.id)
                      : null
                  }
                  disabled={locked}
                  onRemove={onRemoveItem}
                />
              ))}
            </div>
          )}

          {/* One primary action and what it costs. There was a "Select all
              missing" button here; the card already OPENS with exactly that
              selection ticked, so it was a control whose first click could
              only ever be a no-op. */}
          <div className={styles.cardActions}>
            <button
              type="button"
              className={styles.primaryBtn}
              disabled={disabled}
              title={reason}
              onClick={() => onInstall(chosen.map((item) => item.id), 'live')}
            >
              {t('packs.installSelected')}
            </button>
            <span className={styles.selectedSize}>
              {t('packs.sizeSelected', { size: formatBytes(chosenBytes) })}
            </span>
          </div>
        </>
      )}
    </div>
  );
}

/**
 * One item's bar out of the running job, treating an inherited member as absent.
 *
 * `job.items` is a bare object keyed by whatever ids the catalog ships, so an
 * item called `constructor` or `toString` would come back as a FUNCTION from a
 * plain index — truthy, and `PackItemRow` would read byte counts off it and
 * paint a NaN bar instead of no bar at all.
 */
function itemProgress(job: PackJob, itemId: string): PackItemProgress | null {
  // `?? null` because the index is typed as total and is not: a job that names
  // an item this pack does not list — or one whose key was explicitly set to
  // undefined — would otherwise hand back `undefined` under a `| null` type
  // and put the row one property access away from a crash.
  return Object.prototype.hasOwnProperty.call(job.items, itemId)
    ? job.items[itemId] ?? null
    : null;
}

/** Everything with bytes still to fetch — what a fresh card starts ticked. */
function defaultSelection(pack: PackSummary): Set<string> {
  return new Set(missingItems(pack).map((item) => item.id));
}
