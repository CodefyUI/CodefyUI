/** The name a physical device has inside a run given a device set (#774).
 *
 * A run whose target lists several CUDA cards gets them as
 * `CUDA_VISIBLE_DEVICES` in the order listed, so inside the run they are
 * `cuda:0..N-1`: `["cuda:2", "cuda:0"]` makes `cuda:2` the run's `cuda:0` and
 * `cuda:0` its `cuda:1`. A node's `device` param holds the in-run name; the
 * server never rewrites it. Mirrors `device_aliases` in
 * `backend/app/agent/runner.py`; #775's dialog shows it as
 * `cuda:2 -> cuda:1 in this run`.
 *
 * Returns `null` when `physical` is not a CUDA card of the set. Non-CUDA
 * devices (`cpu`, `mps`) have no alias: they are not renumbered.
 */
const CUDA_CARD = /^cuda:(\d+)$/;

function cardIndex(device: string): number | null {
  const match = CUDA_CARD.exec(device.trim().toLowerCase());
  return match ? Number(match[1]) : null;
}

export function aliasForDevice(devices: readonly string[], physical: string): string | null {
  const wanted = cardIndex(physical);
  if (wanted === null) return null;
  const cards = devices.map(cardIndex).filter((index): index is number => index !== null);
  const position = cards.indexOf(wanted);
  return position === -1 ? null : `cuda:${position}`;
}
