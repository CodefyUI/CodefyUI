import type { PluginStatus } from '../../api/rest';
import { useI18n, type TranslationKey } from '../../i18n';
import styles from './PluginCenterModal.module.css';

/** Which half of the catalog the list is showing. */
export type PluginFilter = 'all' | 'installed' | 'available';

/**
 * Whether a row belongs to the "Available" half.
 *
 * Written as the narrow half and negated for the other, so the two halves are
 * exhaustive by construction: a status this build has never heard of lands
 * under "Installed" rather than under neither, and no filter can make a row
 * disappear from both tabs. `removed` is available — a tombstone is a plugin
 * that is not here and can be put back — while `missing_files` and
 * `installing` are not: the lockfile has them.
 */
export function isAvailableStatus(status: PluginStatus): boolean {
  return status === 'available' || status === 'removed';
}

/** Whether *status* passes *filter*. */
export function matchesFilter(filter: PluginFilter, status: PluginStatus): boolean {
  if (filter === 'all') return true;
  return isAvailableStatus(status) === (filter === 'available');
}

const OPTIONS: { value: PluginFilter; key: TranslationKey }[] = [
  { value: 'all', key: 'pluginCenter.filter.all' },
  { value: 'installed', key: 'pluginCenter.filter.installed' },
  { value: 'available', key: 'pluginCenter.filter.available' },
];

export interface PluginFilterBarProps {
  value: PluginFilter;
  onChange: (value: PluginFilter) => void;
}

/**
 * All | Installed | Available.
 *
 * Buttons rather than tabs, and `aria-pressed` rather than a class: this
 * filters a list that is already on screen, so nothing here is a navigation.
 *
 * Deliberately three and not five. Official and GitHub are facts the card
 * states about itself; a filter for each would be two more controls over a
 * catalog that is a dozen rows long.
 */
export function PluginFilterBar({ value, onChange }: PluginFilterBarProps) {
  const { t } = useI18n();

  return (
    <div className={styles.filterBar}>
      {OPTIONS.map((option) => (
        <button
          key={option.value}
          type="button"
          aria-pressed={value === option.value}
          onClick={() => onChange(option.value)}
        >
          {t(option.key)}
        </button>
      ))}
    </div>
  );
}
