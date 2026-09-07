/**
 * Readers for the `settings` block of a graph file.
 *
 * The same pattern the backend applies to `settings.device` (`DEVICE_PATTERN`
 * in `device_utils.py`): the four validators agree, so a file the server
 * accepts is a file the canvas reads, and a value the canvas rejects is one
 * the server would refuse at save time.
 */
export const DEVICE_PATTERN = /^(cpu|auto|cuda(:\d+)?|mps(:\d+)?)$/;

/**
 * The device a graph file assigns, or null when it assigns none.
 *
 * `settings` is untrusted input off a file, so every shape short of a valid
 * string reads as "no assignment": a missing block, a block that is not an
 * object, a `device` that is not a string, or a value outside the pattern.
 * The value is trimmed and lower-cased before the match, so `' CUDA:1 '`
 * reads as `cuda:1`.
 */
export function readGraphDevice(settings: unknown): string | null {
  const raw =
    settings && typeof settings === 'object'
      ? (settings as { device?: unknown }).device
      : undefined;
  if (typeof raw !== 'string') return null;
  const v = raw.trim().toLowerCase();
  return DEVICE_PATTERN.test(v) ? v : null;
}
