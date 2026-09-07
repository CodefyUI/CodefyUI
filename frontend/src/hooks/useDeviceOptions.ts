/**
 * The compute devices this server offers, shared by every device dropdown.
 *
 * One fetch per page load: the Settings popover and the graph toolbar both
 * render the same list, and the answer does not change while the page is
 * open. The promise is cached at module level, so a second consumer mounts
 * onto the result the first one already asked for. A failed fetch leaves the
 * CPU-only fallback in place; CPU always exists, so every dropdown still has
 * a valid choice.
 */
import { useEffect, useState } from 'react';
import { fetchDevices, type DeviceInfo } from '../api/rest';

export interface DeviceOptions {
  devices: DeviceInfo[];
  /** The best device the server reports (`cuda` > `mps` > `cpu`); `cpu` when unknown. */
  serverDefault: string;
}

const CPU_ONLY: DeviceInfo[] = [{ value: 'cpu', label: 'CPU', detail: '', available: true }];
const FALLBACK: DeviceOptions = { devices: CPU_ONLY, serverDefault: 'cpu' };

let _pending: Promise<DeviceOptions> | null = null;
let _resolved: DeviceOptions | null = null;

function loadDeviceOptions(): Promise<DeviceOptions> {
  if (!_pending) {
    _pending = fetchDevices()
      .then((r) => ({
        devices: r.devices.length > 0 ? r.devices : CPU_ONLY,
        serverDefault: r.default || 'cpu',
      }))
      .catch(() => FALLBACK)
      .then((options) => {
        _resolved = options;
        return options;
      });
  }
  return _pending;
}

/** The option label a dropdown shows for one device. */
/**
 * Option text for a device. A single-device backend shows its label only
 * ("Apple MPS"); a per-card entry (cuda:1) adds the id and the card name so
 * two cards of one vendor stay distinguishable.
 */
export function deviceLabel(d: DeviceInfo): string {
  if (!d.value.includes(':')) return d.label;
  return d.detail ? `${d.label} · ${d.value} · ${d.detail}` : `${d.label} · ${d.value}`;
}

export function useDeviceOptions(): DeviceOptions {
  // A consumer mounting after the fetch settled starts on the answer, so a
  // popover opened later never flashes the CPU-only list.
  const [options, setOptions] = useState<DeviceOptions>(() => _resolved ?? FALLBACK);
  useEffect(() => {
    void loadDeviceOptions().then(setOptions);
  }, []);
  return options;
}

/** Test seam: forget the cached answer so the next consumer fetches again. */
export function _resetDeviceOptionsForTesting(): void {
  _pending = null;
  _resolved = null;
}
