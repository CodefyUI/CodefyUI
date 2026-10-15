/**
 * Where a sweep variant runs (#777), from the fields `GET /api/sweeps/{id}`
 * copies off the child run row. Absent when the run is gone (or the server
 * predates #777); the cells then read `-`.
 */
import type { SweepVariant } from '../../api/rest';

/** `local` (this server), a worker's name, or `any` while an `any` variant
 * waits to be placed. */
export function variantWorker(variant: SweepVariant): string {
  return variant.worker_id ?? '-';
}

/**
 * The variant's device set when it has one, else its device. A worker
 * variant's queue key is `<worker>/<device>` and the worker has its own
 * column, so only the device is shown, as in the Runs list.
 */
export function variantDevices(variant: SweepVariant): string {
  if (variant.devices && variant.devices.length > 0) return variant.devices.join(', ');
  const key = variant.queue_key;
  return key ? key.slice(key.lastIndexOf('/') + 1) : '-';
}
