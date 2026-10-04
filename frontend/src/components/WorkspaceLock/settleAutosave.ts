import { useTabStore } from '../../store/tabStore';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { snapshotWritesInFlight, whenSnapshotWritesSettle } from '../../store/tabPersistence';

/**
 * Longer than the tab store's autosave debounce (`SAVE_DEBOUNCE_MS`, 250 ms),
 * with room for a late timer. The two-page test in `workspaceLock.test.ts`
 * fails if the debounce ever outgrows it.
 */
const QUIET_MS = 400;
/** How long to wait for a store that keeps changing to go quiet. */
const MAX_WAIT_MS = 5_000;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Wait until whatever the tab store's autosave still holds is on disk. A page
 * awaits this before it hands editing to another page (#554), so the page
 * that takes over loads the last change rather than the one before it.
 *
 * The store saves on a trailing debounce that any change to the tab store or
 * to the node list re-arms, and nothing outside the store can fire it early.
 * So "on disk" is read off from outside: nothing has changed for longer than
 * the debounce, and no write is still in flight.
 *
 * A store that never goes quiet never saves on its own either: a visible page
 * drawing a running graph's progress commits every frame. Past `maxWaitMs`
 * this stops waiting for quiet and settles for the writes already started,
 * rather than holding the other page until the run ends.
 */
export async function settleAutosave({
  maxWaitMs = MAX_WAIT_MS,
}: { maxWaitMs?: number } = {}): Promise<void> {
  // Assume a change just now: one may have landed a moment before this was
  // called, its save still pending, with nothing here to have seen it.
  let lastChange = Date.now();
  const unsubscribeTabs = useTabStore.subscribe(() => {
    lastChange = Date.now();
  });
  const unsubscribeDefs = useNodeDefStore.subscribe((state, previous) => {
    // The two fields whose arrival re-arms the autosave (see tabStore).
    if (state.definitions !== previous.definitions || state.presets !== previous.presets) {
      lastChange = Date.now();
    }
  });
  const deadline = Date.now() + maxWaitMs;
  try {
    for (;;) {
      const wait = Math.min(lastChange + QUIET_MS, deadline) - Date.now();
      if (wait > 0) {
        await sleep(wait);
        continue;
      }
      await whenSnapshotWritesSettle();
      // A save queued behind the write that just finished starts a few
      // microtasks later; one macrotask lets it show up as in flight.
      await sleep(0);
      const quiet = Date.now() - lastChange >= QUIET_MS;
      if ((quiet && snapshotWritesInFlight() === 0) || Date.now() >= deadline) {
        await whenSnapshotWritesSettle();
        return;
      }
    }
  } finally {
    unsubscribeTabs();
    unsubscribeDefs();
  }
}
