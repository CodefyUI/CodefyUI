/**
 * "The list of saved graphs on the server just changed."
 *
 * The same shape as `worktreeWrite.ts` next door, and for the same reason: a
 * Save is `saveActiveGraph`'s business and the Graphs panel is the sidebar's,
 * and neither should have to import the other to stay in step. The panel
 * registers itself while it is mounted; the resting state is "nobody is
 * listening", which costs one null check per save.
 *
 * It exists because the panel and the toolbar are on screen together. A Save
 * from the toolbar writes a file the panel is showing a list of, and without
 * this the new graph did not appear in that list until something else
 * re-read it -- the refresh button, or closing and reopening the tab.
 *
 * Deliberately ONE listener rather than a list: this is a nudge to the panel
 * that owns the list, not an event bus.
 */

type GraphsWriteListener = () => void;

let listener: GraphsWriteListener | null = null;

/**
 * Watch for writes to the saved-graph list, or pass null to stop watching.
 *
 * A single slot, so registering twice replaces rather than adds. Unregister
 * by passing null only while your own listener is the one installed -- an
 * effect cleanup that fires after a remount would otherwise clear the
 * listener the remount has already put back.
 */
export function setGraphsWriteListener(next: GraphsWriteListener | null): void {
  listener = next;
}

/** The listener currently installed, or null. Lets a caller check its own. */
export function getGraphsWriteListener(): GraphsWriteListener | null {
  return listener;
}

/** Say that a saved graph was written. Nothing may be listening. */
export function announceGraphsWrite(): void {
  listener?.();
}
