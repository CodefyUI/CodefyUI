/**
 * "A file in the project directory was just written from inside the app."
 *
 * A signal with no imports, which is the whole point. The code that WRITES --
 * `saveActiveGraph` today, whatever else grows a write tomorrow -- should not
 * have to know that a Source Control tab exists, and should not pull that
 * tab's store (and through it the entire `/api/git` client) into its own
 * module graph just to say that something happened. The Toolbar's tests, and
 * every other test that stubs `api/rest` wholesale, would then be loading the
 * git client for a save.
 *
 * The Source Control store registers itself while its tab is attached and
 * clears the registration on the way out, so "nobody is listening" is the
 * resting state and costs one null check per save. Deliberately ONE listener
 * rather than a list: this is a nudge to the panel that polls the worktree,
 * not an event bus, and a second subscriber would be a second poll.
 */

type WorktreeWriteListener = () => void;

let listener: WorktreeWriteListener | null = null;

/**
 * Watch for local writes, or pass null to stop watching.
 *
 * A single slot, so registering twice replaces rather than adds. The Source
 * Control store's `attach()` / `detach()` are the only caller, and their
 * reference counting is what keeps the slot honest when the tab is mounted
 * twice.
 */
export function setWorktreeWriteListener(next: WorktreeWriteListener | null): void {
  listener = next;
}

/** Say that the project directory was written to. Nothing may be listening. */
export function announceWorktreeWrite(): void {
  listener?.();
}
