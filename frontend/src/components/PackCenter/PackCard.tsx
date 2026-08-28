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
import { packTitle, type PackIndex } from '../../utils/packAvailability';
import { GpuPackDetails } from './GpuPackDetails';
import { PackItemRow } from './PackItemRow';
import {
  catalogKey,
  formatBytes,
  missingItems,
  statusKey,
  statusTone,
  type Translate,
} from './packStatus';
import styles from './PackCenterModal.module.css';

/**
 * Whether the server can install a restart-mode pack and relaunch itself.
 *
 * FALSE for the whole of this PR: the backend answers a restart-mode install
 * with 409 and the command to run by hand, so offering an "Install and
 * restart" button would offer a button that cannot work. PR 5 lands the
 * supervisor handshake and flips this one constant.
 */
export const RESTART_INSTALL_AVAILABLE = false;

/** The label for a pack, preferring this build's copy over the server's. */
function displayTitle(
  t: Translate,
  byId: PackIndex,
  packId: string,
  serverTitle?: string,
): string {
  const key = catalogKey(packId, 'title');
  if (key !== null) return t(key);
  return serverTitle ?? packTitle(byId, packId);
}

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

  const title = displayTitle(t, byId, pack.id, pack.title);
  const descKey = catalogKey(pack.id, 'desc');
  const description = descKey !== null ? t(descKey) : pack.description;

  const chosen = pack.items.filter((item) => selected.has(item.id));
  const chosenBytes = chosen.reduce((sum, item) => sum + item.size_bytes, 0);
  const blockedBy = pack.blocked_by;

  const disabled = !canInstall || locked || blockedBy.length > 0 || chosen.length === 0;
  const reason = !canInstall
    ? t('packs.remoteDisabled')
    : blockedBy.length > 0
      ? t('packs.dependsOnMissing', { pack: displayTitle(t, byId, blockedBy[0]) })
      : chosen.length === 0
        // Its own sentence, not the "Select all missing" button's label: a
        // tooltip explaining why a button is dead has to say what to DO, and
        // naming another button is not that.
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
        </div>
      )}

      {pack.depends_on.length > 0 && (
        <div className={styles.deps}>
          <span>
            {t('packs.dependsOn', {
              packs: pack.depends_on
                .map((depId) => displayTitle(t, byId, depId))
                .join(', '),
            })}
          </span>
          {pack.depends_on.map((depId) => {
            // `byId` is built from parsed JSON, so a bare index answers with
            // an inherited member for an id like `constructor`.
            const dep = Object.prototype.hasOwnProperty.call(byId, depId)
              ? byId[depId]
              : undefined;
            return (
              <span key={depId} className={styles.dep}>
                {dep && <StatusPill status={dep.status} />}
                {blockedBy.includes(depId) && (
                  <button
                    type="button"
                    className={styles.linkBtn}
                    onClick={() => onFocusPack(depId)}
                  >
                    {t('packs.dependsOnMissing', { pack: displayTitle(t, byId, depId) })}
                  </button>
                )}
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
          restartAvailable={RESTART_INSTALL_AVAILABLE}
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
                  // another pack must not paint bars on these rows.
                  progress={jobHere ? itemProgress(job, item.id) : null}
                  disabled={locked}
                  onRemove={onRemoveItem}
                />
              ))}
            </div>
          )}

          <div className={styles.cardActions}>
            <button
              type="button"
              className={styles.secondaryBtn}
              disabled={locked || missingItems(pack).length === 0}
              onClick={() => setSelected(defaultSelection(pack))}
            >
              {t('packs.selectAll')}
            </button>
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
  return Object.prototype.hasOwnProperty.call(job.items, itemId)
    ? job.items[itemId]
    : null;
}

/** Everything with bytes still to fetch — what a fresh card starts ticked. */
function defaultSelection(pack: PackSummary): Set<string> {
  return new Set(missingItems(pack).map((item) => item.id));
}
