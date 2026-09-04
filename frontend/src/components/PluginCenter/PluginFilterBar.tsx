import { useI18n, type TranslationKey } from '../../i18n';
// The partition itself is a rule, not a control: it lives with the other pure
// rules, where a test can pin which half `installing` falls in without
// mounting a bar to press.
import type { PluginFilter } from './pluginStatus';
import styles from './PluginCenterModal.module.css';

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
