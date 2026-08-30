import type { PackItem, PackItemStatus } from '../../api/rest';
import type { PackItemProgress } from '../../store/packStore';
import { useI18n } from '../../i18n';
import type { TranslationKey } from '../../i18n/locales/en';
import { ProgressBar } from '../shared/ProgressBar';
import { formatBytes } from './packStatus';
import styles from './PackCenterModal.module.css';

const ITEM_STATUS_KEY: Record<PackItemStatus, TranslationKey> = {
  missing: 'packs.item.missing',
  present: 'packs.item.present',
  downloading: 'packs.item.downloading',
};

/**
 * What to call one downloadable item.
 *
 * The repo id is the name the user recognises (`sentence-transformers/LaBSE`);
 * a plain asset gets its file name, with the query string dropped so a signed
 * URL does not become the label. The item id is the last resort — always
 * present, and already human-readable in the shipped catalog.
 */
export function itemDisplayName(item: PackItem): string {
  if (item.repo_id) return item.repo_id;
  if (item.url) {
    const path = item.url.split(/[?#]/)[0];
    const base = path.substring(path.lastIndexOf('/') + 1);
    if (base) return base;
  }
  return item.id;
}

export interface PackItemRowProps {
  item: PackItem;
  checked: boolean;
  onToggle: (itemId: string) => void;
  /** Live bytes for this item, or null when nothing is downloading it. */
  progress: PackItemProgress | null;
  disabled: boolean;
  /** Absent when the row must not offer to delete (nothing is downloaded). */
  onRemove?: (itemId: string) => void;
}

/**
 * One model or file inside a pack, with its own tick box and its own bar.
 *
 * Per-item rather than per-pack because that is the unit the user actually
 * wants: the embeddings pack is four models totalling 1.4 GB, and somebody on
 * a slow connection who only needs the English one should not have to fetch
 * the other three to find out it works.
 */
export function PackItemRow({
  item,
  checked,
  onToggle,
  progress,
  disabled,
  onRemove,
}: PackItemRowProps) {
  const { t } = useI18n();
  const name = itemDisplayName(item);
  const present = item.status === 'present';
  const licenseTitle = item.license
    ? t('packs.item.license', { license: item.license })
    : undefined;

  return (
    <div className={styles.itemRow} data-item-id={item.id}>
      {present ? (
        // No tick box on something already downloaded: there is nothing to
        // fetch, and an unselectable checkbox is a worse answer than none.
        <span className={styles.itemCheckSpacer} aria-hidden="true" />
      ) : (
        <input
          type="checkbox"
          className={styles.itemCheck}
          aria-label={name}
          checked={checked}
          disabled={disabled}
          onChange={() => onToggle(item.id)}
        />
      )}

      {/* Name and licence share one line. Each tooltip recovers its OWN text:
          the name is the part that gets ellipsized, so `title` gives it back
          in full, and the bare licence code is what needs the word "License"
          spelling out. They used to be swapped, which put the licence on the
          screen twice and the name nowhere. */}
      <span className={styles.itemNameCell}>
        <span className={styles.itemName} title={name}>
          {name}
        </span>
        {item.license && (
          <span className={styles.itemLicense} title={licenseTitle}>
            {item.license}
          </span>
        )}
      </span>

      <span className={styles.itemSize}>{formatBytes(item.size_bytes)}</span>

      {present && onRemove ? (
        <button
          type="button"
          className={styles.itemRemove}
          // Named, because a pack card carries one of these per downloaded
          // model: "Remove" three times over is one control repeated to
          // anyone navigating by name, and the visible label has to stay the
          // short word for the row to fit.
          aria-label={t('packs.item.removeNamed', { item: name })}
          // Same lock as the tick boxes: deleting a file out from under the
          // job that is writing to this pack is the one destructive thing
          // this row can do, and it must not be reachable mid-install.
          disabled={disabled}
          onClick={() => onRemove(item.id)}
        >
          {t('packs.item.remove')}
        </button>
      ) : (
        <span className={styles.itemStatus} data-status={item.status}>
          {t(ITEM_STATUS_KEY[item.status] ?? ITEM_STATUS_KEY.missing)}
        </span>
      )}

      {progress && (
        <div className={styles.itemProgress}>
          <ProgressBar size="sm" value={progress.percent} label={name} />
          <span className={styles.itemBytes}>
            {progress.bytesTotal === null
              ? formatBytes(progress.bytesDone)
              : `${formatBytes(progress.bytesDone)} / ${formatBytes(progress.bytesTotal)}`}
          </span>
        </div>
      )}
    </div>
  );
}
