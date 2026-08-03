/**
 * The host's one frame-coalescing primitive.
 *
 * Extracted from `store/nodeUpdateQueue.ts` (#125) when the plugin API grew a
 * second consumer (#132): a stream that fans execution events out to plugin
 * subscribers. Two hand-rolled rAF loops would have meant two sets of
 * fallback rules, two answers to "what happens in a background tab", and two
 * places to get `cancelAnimationFrame` wrong — so both callers share this.
 *
 * Semantics, unchanged from the original:
 *
 *   - `schedule()` is idempotent while a flush is pending; the flush runs once
 *     per frame however many times it was asked for.
 *   - `flush` runs with the slot already cleared, so re-entrant `schedule()`
 *     calls from inside it queue the NEXT frame instead of being swallowed.
 *   - With no `requestAnimationFrame` at all (jsdom without `pretendToBeVisual`,
 *     a worker) it falls back to a `setTimeout(..., 0)` macrotask, which still
 *     coalesces a burst arriving in the same tick.
 *
 * The case the fallback is NOT for, and the one worth knowing about: a hidden,
 * backgrounded or occluded document still HAS `requestAnimationFrame`, it
 * simply never calls back. The rAF branch is taken and the flush waits,
 * indefinitely, until the document is painted again. That is deliberate —
 * there is no reason to rebuild anything for pixels nobody is looking at — but
 * it does mean every consumer must keep whatever it buffers BOUNDED, because
 * the wait has no upper limit.
 */

export interface FrameFlusher {
  /** Ask for a flush on the next frame. No-op while one is already pending. */
  schedule(): void;
  /** Drop a pending flush without running it. */
  cancel(): void;
  /** Whether a flush is currently pending. Exposed for tests. */
  isScheduled(): boolean;
}

export function createFrameFlusher(flush: () => void): FrameFlusher {
  // rAF and setTimeout hand back ids from different spaces and must be
  // cancelled with their own canceller, so remember which one produced this.
  let handle: number | null = null;
  let handleIsRaf = false;

  const cancel = (): void => {
    if (handle === null) return;
    if (handleIsRaf) {
      // A test can strip rAF between scheduling and cancelling; dropping the
      // handle is still correct because every flush re-checks its own buffer.
      if (typeof cancelAnimationFrame === 'function') cancelAnimationFrame(handle);
    } else {
      clearTimeout(handle);
    }
    handle = null;
  };

  const schedule = (): void => {
    if (handle !== null) return;
    if (typeof requestAnimationFrame === 'function') {
      handleIsRaf = true;
      handle = requestAnimationFrame(() => {
        handle = null;
        flush();
      });
    } else {
      handleIsRaf = false;
      handle = setTimeout(() => {
        handle = null;
        flush();
      }, 0) as unknown as number;
    }
  };

  return { schedule, cancel, isScheduled: () => handle !== null };
}
