import type { PackItem, PackStatus, PackSummary } from '../../api/rest';
import type { PackJob } from '../../store/packStore';
import en, { type TranslationKey } from '../../i18n/locales/en';

/**
 * The pure half of the Package Center: everything the panel needs to decide
 * that does not need a DOM, a store or a clock.
 *
 * Kept out of the components on purpose. Each of these is a rule with an edge
 * case worth pinning down in a test — a download that overshoots its stated
 * size, a step id this build has never heard of, a pack the backend ships and
 * this frontend has no copy for — and none of them are worth mounting a modal
 * to exercise.
 */

/** Signature of `useI18n`'s `t`, so the helpers stay callable outside React. */
export type Translate = (
  key: TranslationKey,
  vars?: Record<string, string | number>,
) => string;

/** Which wash a status pill wears. Names a ROLE, not a colour. */
export type StatusTone = 'success' | 'warning' | 'info' | 'neutral';

/**
 * A pack status as a tone.
 *
 * `partial` and `needs_restart` share the warning wash because they are the
 * same sentence to the user: something is here, and it is not finished.
 */
export function statusTone(status: PackStatus): StatusTone {
  switch (status) {
    case 'installed':
      return 'success';
    case 'installing':
      return 'info';
    case 'partial':
    case 'needs_restart':
      return 'warning';
    default:
      return 'neutral';
  }
}

/**
 * The label for one pack status.
 *
 * Deliberately NOT extended with `packs.status.failed`: that string labels a
 * finished JOB in the activity pane, and a `PackStatus` can never be `failed`
 * — a pack whose install failed is back to `not_installed` or `partial`.
 */
const STATUS_KEY: Record<PackStatus, TranslationKey> = {
  not_installed: 'packs.status.not_installed',
  partial: 'packs.status.partial',
  installed: 'packs.status.installed',
  installing: 'packs.status.installing',
  needs_restart: 'packs.status.needs_restart',
};

export function statusKey(status: PackStatus): TranslationKey {
  return STATUS_KEY[status] ?? STATUS_KEY.not_installed;
}

/**
 * The i18n key for a pack's shipped copy, or null when this build has none.
 *
 * Catalog copy is keyed by PACK ID, and the backend is free to ship a pack
 * this frontend predates. `hasOwnProperty` rather than `in`, because the
 * message table is a plain object: an id like `constructor` would otherwise
 * "exist" and translate to the prototype's own member.
 */
export function catalogKey(
  packId: string,
  field: 'title' | 'desc',
): TranslationKey | null {
  const key = `packs.catalog.${packId}.${field}`;
  return Object.prototype.hasOwnProperty.call(en, key) ? (key as TranslationKey) : null;
}

/** Everything in *pack* that still has bytes to fetch. */
export function missingItems(pack: PackSummary): PackItem[] {
  // `!== 'present'` rather than `=== 'missing'`: a half-written download has
  // not arrived either, and this is the same set the store sends when an
  // install omits `items`.
  return pack.items.filter((item) => item.status !== 'present');
}

const UNITS = ['B', 'KB', 'MB', 'GB', 'TB'];

/**
 * A byte count as a phrase.
 *
 * Anything that is not a real, positive number reads as `0 B`: the size is a
 * caption on a download button, and "NaN GB" is worse than saying nothing.
 */
export function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return '0 B';
  let value = n;
  let unit = 0;
  while (value >= 1024 && unit < UNITS.length - 1) {
    value /= 1024;
    unit += 1;
  }
  // Whole bytes, and one decimal above them only while it says something:
  // "1.5 GB" earns its digit, "352.0 MB" does not.
  const decimals = unit === 0 || value >= 10 ? 0 : 1;
  return `${value.toFixed(decimals)} ${UNITS[unit]}`;
}

function clamp(value: number): number {
  return Math.min(100, Math.max(0, value));
}

/**
 * How far the whole job has got, 0..100, or null when nobody can say yet.
 *
 * Weighted by BYTES, not by item count: a pack whose 900 MB model is half
 * done and whose 100 MB model is finished is 55 % done, and an item-counting
 * bar would claim 75 % and then appear to stall for minutes.
 *
 * The step count is the fallback for the part of a job that has no bytes at
 * all — a pip-only pack, and the window before the first `progress` event.
 * Null (indeterminate) is the answer when there is neither, because 0 % about
 * a job that is clearly working is a wrong number rather than no number.
 */
export function jobOverallPercent(
  job: PackJob | null,
  pack: PackSummary | undefined,
): number | null {
  if (!job) return null;

  const entries = Object.entries(job.items);
  if (entries.length > 0) {
    const sizes = new Map((pack?.items ?? []).map((item) => [item.id, item.size_bytes]));
    let total = 0;
    let done = 0;
    for (const [id, progress] of entries) {
      // The catalog's size is the honest weight; the download's own total is
      // the fallback for an item the catalog does not list (a newer backend).
      const weight = sizes.get(id) ?? progress.bytesTotal ?? 0;
      if (!Number.isFinite(weight) || weight <= 0) continue;
      total += weight;
      // A server that reports more bytes than it promised must not push the
      // bar past full, or past the other items' share of it.
      done += Math.min(weight, Math.max(0, progress.bytesDone));
    }
    if (total > 0) return clamp((done / total) * 100);
  }

  if (job.steps.length === 0) return null;
  const finished = job.steps.filter((step) => step.state === 'done').length;
  return clamp((finished / job.steps.length) * 100);
}

/**
 * A job step as a sentence, keyed off the step ID rather than its text.
 *
 * The server's `label` is English and written for a log; the step id
 * (`pip`, `download:<item>`, `convert:<item>`, `verify`) is a stable
 * vocabulary this can translate. An id from a newer backend falls back to the
 * server's own label, which is at least true, and to the raw id when even
 * that is empty.
 */
export function stepLabel(t: Translate, step: string, label: string): string {
  const separator = step.indexOf(':');
  const kind = separator === -1 ? step : step.slice(0, separator);
  const item = separator === -1 ? '' : step.slice(separator + 1);
  const fallback = label || step;

  switch (kind) {
    case 'pip':
      return t('packs.activity.step.pip');
    case 'verify':
      return t('packs.activity.step.verify');
    case 'download':
      return item ? t('packs.activity.step.download', { item }) : fallback;
    case 'convert':
      return item ? t('packs.activity.step.convert', { item }) : fallback;
    default:
      return fallback;
  }
}

/**
 * The step the pane should be showing: the one still running, or the last one
 * to have finished while the next has not been announced yet.
 *
 * Returns the 1-based index too, because that is what `packs.activity.step`
 * puts in front of the label.
 */
export function currentStep(
  job: PackJob,
): { index: number; step: string; label: string } | null {
  if (job.steps.length === 0) return null;
  const running = job.steps.findIndex((step) => step.state === 'running');
  const index = running === -1 ? job.steps.length - 1 : running;
  const step = job.steps[index];
  return { index: index + 1, step: step.step, label: step.label };
}
