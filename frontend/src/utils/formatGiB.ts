/** Memory budgets and card figures in GiB (#788), one decimal: the unit
 * `target.memory_gib` is declared in and the backend's messages use
 * (`memory_admission.format_gib`), so "needs 6.0 GiB" in a refusal and the
 * 6.0 GiB on a card's bar are the same number. A positive figure below
 * 0.1 GiB prints as 0.1, never as 0.0. */
const GIB = 1024 ** 3;

export function formatGiB(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0.0 GiB';
  const value = Math.max(bytes / GIB, 0.1);
  return `${value.toFixed(1)} GiB`;
}
