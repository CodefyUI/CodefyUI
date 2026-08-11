/** Byte counts, for humans.
 *
 * Binary steps (1024) with the short KB/MB/GB names rather than KiB/MiB/GiB,
 * because the numbers this formats come from `/api/health`, and the budgets
 * there are configured in the same units: `EXECUTION_CACHE_MAX_MB * 1024 *
 * 1024` (backend/app/core/cache.py). Dividing by 1000 instead would render a
 * configured 512 MB ceiling as "536.9 MB", which reads as the app
 * misreporting the setting rather than as a unit convention.
 *
 * There is no house formatter to reuse -- this is the first place in the
 * frontend that shows a size (#193 item 2).
 */
const UNITS = ['B', 'KB', 'MB', 'GB', 'TB'] as const;

export function formatBytes(bytes: number): string {
  // A store that is not running is omitted from the health payload entirely,
  // so a caller reading `caches.x?.bytes` can land here with undefined ->
  // NaN. "NaN B" in a settings panel reads as a broken server; zero is the
  // honest rendering of "nothing to report".
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';

  let value = bytes;
  let unit = 0;
  while (unit < UNITS.length - 1 && value >= 1024) {
    value /= 1024;
    unit += 1;
  }
  // 1048575 B settles at 1023.999… KB, and one decimal place would print that
  // as "1024.0 KB". Promote on the ROUNDED value, not the raw one, so the
  // printed number never reaches the next unit's threshold.
  if (unit < UNITS.length - 1 && Number(value.toFixed(1)) >= 1024) {
    value /= 1024;
    unit += 1;
  }
  // Whole bytes below 1 KB: a tenth of a byte is not a quantity, and every
  // byte count the backend sends is an integer.
  return unit === 0 ? `${Math.round(value)} B` : `${value.toFixed(1)} ${UNITS[unit]}`;
}
