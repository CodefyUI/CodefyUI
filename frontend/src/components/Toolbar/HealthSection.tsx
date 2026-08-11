import { useEffect, useState } from 'react';
import { useI18n } from '../../i18n';
import type { TranslationKey } from '../../i18n/locales/en';
import { fetchHealth, type CacheUsage, type HealthInfo } from '../../api/rest';
import { formatBytes } from '../../utils/formatBytes';
import { SettingsRow } from './SettingsRow';
import styles from './SettingsPopover.module.css';

/**
 * Human names for the stores `/api/health` reports.
 *
 * A map rather than a `settings.health.cache.${name}` template because `t()`
 * takes a typed key, and because an unknown store must still be listed: the
 * payload can grow a fourth store (the unbounded lm_blocks cache of #306 is
 * one candidate), and a store this table has never heard of is more useful
 * shown under its raw name than silently dropped from the list.
 */
const CACHE_LABEL_KEYS: Record<string, TranslationKey> = {
  execution_cache: 'settings.health.cache.execution_cache',
  run_output_store: 'settings.health.cache.run_output_store',
  node_state_store: 'settings.health.cache.node_state_store',
};

/**
 * First numeric value among `keys`, or undefined.
 *
 * The three stores do not agree on a budget key -- `max_bytes` for the
 * run-output and node-state stores, `max_bytes_each` for the execution cache
 * (one instance per WebSocket, so the total has no single ceiling). Checked at
 * runtime rather than typed, because `CacheUsage` is deliberately an open map
 * of whatever that store chose to report.
 */
function pick(usage: CacheUsage, ...keys: string[]): number | undefined {
  for (const key of keys) {
    const value = usage[key];
    if (typeof value === 'number' && Number.isFinite(value)) return value;
  }
  return undefined;
}

/**
 * The "This Server" section of the settings popover (#193 item 2).
 *
 * `/api/health` has reported the version, the registry counts and per-store
 * cache bytes since #135, and the frontend read exactly one field of it
 * (`project`, once, at bootstrap). Users on a bounded memory budget had no way
 * to see how close they were.
 *
 * Fetched when this mounts, which is when the popover opens -- SettingsPopover
 * returns null while closed, so "on open" needs no extra plumbing. No polling:
 * the numbers do move during a run, but a panel that refetches on a timer
 * while sitting open costs a request per interval for information nobody is
 * necessarily reading. The Refresh button is the explicit ask instead.
 */
export function HealthSection() {
  const { t } = useI18n();
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [failed, setFailed] = useState(false);
  const [busy, setBusy] = useState(false);
  // Bumped by Refresh. Re-running the same effect keeps one code path for the
  // first read and every later one, so a refresh cannot drift from the mount
  // fetch (different error handling, a missed cancel, etc.).
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setBusy(true);
    fetchHealth()
      .then((info) => {
        if (cancelled) return;
        setHealth(info);
        setFailed(false);
      })
      .catch(() => {
        // Inline, not a toast: the user is looking at this panel, and a toast
        // for a number they opened a popover to read would outlive the popover.
        if (!cancelled) setFailed(true);
      })
      .finally(() => {
        if (!cancelled) setBusy(false);
      });
    return () => {
      cancelled = true;
    };
  }, [attempt]);

  const caches = Object.entries(health?.caches ?? {});

  return (
    <section className={styles.section}>
      <div className={styles.sectionTitle}>{t('toolbar.settings.section.system')}</div>

      <SettingsRow
        name={t('settings.health.name')}
        desc={t('settings.health.desc')}
        ctrl={
          <button
            type="button"
            // The LLM section's button reads "Refresh" too; the sections tell
            // them apart visually, an accessible name has to do it in words.
            aria-label={t('settings.health.refreshAria')}
            className={styles.action}
            disabled={busy}
            onClick={(e) => {
              e.stopPropagation();
              setAttempt((n) => n + 1);
            }}
          >
            {t('settings.health.refresh')}
          </button>
        }
      />

      <div className={styles.health}>
        {/* A failed refresh keeps the numbers it already had on screen: stale
            counts plus "could not read" is more informative than a blank
            panel, and says which half of the panel is untrustworthy. */}
        {failed && (
          <div className={styles.healthError} role="status">
            {t('settings.health.failed')}
          </div>
        )}

        {health === null ? (
          !failed && <div className={styles.healthNote}>{t('settings.health.loading')}</div>
        ) : (
          <>
            <div className={styles.healthStats}>
              <Stat label={t('settings.health.version')} value={health.version ?? t('settings.health.unknown')} />
              <Stat label={t('settings.health.nodes')} value={String(health.nodes_loaded)} />
              <Stat label={t('settings.health.presets')} value={String(health.presets_loaded)} />
            </div>

            <div className={styles.healthLabel}>{t('settings.health.caches')}</div>
            {caches.length === 0 ? (
              // Reachable, not an error: the backend omits a store that is not
              // running rather than reporting it as empty.
              <div className={styles.healthNote}>{t('settings.health.cachesEmpty')}</div>
            ) : (
              <ul className={styles.cacheList}>
                {caches.map(([name, usage]) => {
                  const labelKey = CACHE_LABEL_KEYS[name];
                  const used = formatBytes(pick(usage, 'bytes') ?? 0);
                  const budget = pick(usage, 'max_bytes', 'max_bytes_each');
                  return (
                    <li key={name} className={styles.cacheRow}>
                      <span className={styles.cacheName}>{labelKey ? t(labelKey) : name}</span>
                      <span className={styles.cacheValue}>
                        {budget === undefined
                          ? used
                          : t('settings.health.cacheOf', { used, budget: formatBytes(budget) })}
                      </span>
                    </li>
                  );
                })}
              </ul>
            )}
            <div className={styles.healthHint}>{t('settings.health.cachesHint')}</div>
          </>
        )}
      </div>
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className={styles.healthStat}>
      <span className={styles.healthStatLabel}>{label}</span>
      <span className={styles.healthStatValue}>{value}</span>
    </div>
  );
}
