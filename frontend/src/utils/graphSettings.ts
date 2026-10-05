/**
 * Readers for the `settings` block of a graph file: `settings.device` and,
 * since 2.8.9, `settings.seed`.
 *
 * Each follows the rule the backend validates the same field with
 * (`DEVICE_PATTERN` in `device_utils.py`, `MAX_SEED` in `seeding.py`), so a
 * file the server accepts is a file the canvas reads, and a value the canvas
 * rejects is one the server would refuse at save time.
 */
export const DEVICE_PATTERN = /^(cpu|auto|cuda(:\d+)?|mps(:\d+)?)$/;

/**
 * The largest seed a run accepts: the backend's `MAX_SEED`, 2**32 - 1.
 * numpy's legacy `RandomState` refuses anything wider, so every RNG a run
 * seeds is held to its range.
 */
export const MAX_RUN_SEED = 4294967295;

/** A whole number in 0..MAX_RUN_SEED: a seed a run, a save and an export accept. */
export function isRunSeed(v: unknown): v is number {
  return typeof v === 'number' && Number.isInteger(v) && v >= 0 && v <= MAX_RUN_SEED;
}

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

/**
 * The run seed a graph file stores, or null when it stores none.
 *
 * Untrusted input, read like `readGraphDevice`: a missing block, a block that
 * is not an object, a `seed` that is not a number (a numeric string
 * included), and a number that is not a whole one in 0..MAX_RUN_SEED all
 * read as "no seed". Nothing is coerced: a server-written file always holds
 * a plain number, so anything else was edited by hand.
 */
export function readGraphSeed(settings: unknown): number | null {
  const raw =
    settings && typeof settings === 'object'
      ? (settings as { seed?: unknown }).seed
      : undefined;
  return isRunSeed(raw) ? raw : null;
}
