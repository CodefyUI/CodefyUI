import { useDialogStore } from './dialogStore';
import { useTabStore } from './tabStore';
import { useUIStore } from './uiStore';

/**
 * "Is anything modal on screen?", asked in one place (#475).
 *
 * Two very different consumers need the same answer, which is why this is not
 * a field on any one store:
 *
 *  - `useKeyboardShortcuts` asks imperatively, inside a `document` keydown
 *    handler that has no React context to read from -> `isAnyModalOpen()`.
 *  - the two `<ReactFlow>` call sites ask during render, and have to
 *    re-render when the answer changes so `deleteKeyCode` is re-armed the
 *    moment the panel closes -> `useAnyModalOpen()`.
 *
 * Both read the same three stores through the same predicate below. Rebuilt
 * separately in each place, they would drift the first time a modal is added
 * -- which is exactly how the Package Center, the Plugin Center, the Template
 * Gallery and the Git diff came to be invisible to the shortcut hook while
 * the four older modals were not.
 *
 * ADDING A MODAL: give it a name in `ModalName` and a line in the matching
 * `*ModalOpen` reader. Nothing else has to change.
 */
export type ModalName =
  /** The in-app confirm / prompt (`utils/dialog`). */
  | 'dialog'
  | 'shortcuts'
  | 'templateGallery'
  | 'packCenter'
  | 'pluginCenter'
  | 'gitDiff'
  | 'nodeDetail'
  | 'presetModal'
  | 'layersModal'
  | 'vizModal';

type UIStoreState = ReturnType<typeof useUIStore.getState>;
type TabStoreState = ReturnType<typeof useTabStore.getState>;
type DialogStoreState = ReturnType<typeof useDialogStore.getState>;

/**
 * Shared empty list, so the no-argument call passes the SAME reference every
 * time. A fresh `[]` per call would be harmless here (every selector returns
 * a boolean, which zustand compares by value) but the constant says outright
 * that nothing is being allocated per keystroke.
 */
const IGNORE_NOTHING: readonly ModalName[] = [];

/**
 * Every reader below tests the field for TRUTH rather than for `!== null`.
 * Each of these is either a boolean or an id, an id is never the empty
 * string, and a record that predates a field -- a tab restored from an older
 * workspace, a fixture built by hand -- carries `undefined` where the store
 * writes `null`. Neither of those is an open modal, and `!== null` said both
 * were. The `if` chain (rather than one `||` expression) is what keeps the
 * return a boolean instead of whichever id happened to be set.
 */
function dialogModalOpen(s: DialogStoreState, ignore: readonly ModalName[]): boolean {
  return !ignore.includes('dialog') && Boolean(s.active);
}

function uiModalOpen(s: UIStoreState, ignore: readonly ModalName[]): boolean {
  if (!ignore.includes('shortcuts') && s.shortcutsModalOpen) return true;
  if (!ignore.includes('templateGallery') && s.templateGalleryOpen) return true;
  if (!ignore.includes('packCenter') && s.packCenterOpen) return true;
  if (!ignore.includes('pluginCenter') && s.pluginCenterOpen) return true;
  if (!ignore.includes('gitDiff') && s.gitDiff) return true;
  return false;
}

/**
 * The four per-tab modals. They hang off the ACTIVE tab, and since #472 there
 * may be no active tab at all (the welcome screen) -- which is not an error
 * and not a modal either: no tab, nothing open.
 */
function tabModalOpen(s: TabStoreState, ignore: readonly ModalName[]): boolean {
  const tab = s.tabs.find((t) => t.id === s.activeTabId);
  if (!tab) return false;
  if (!ignore.includes('nodeDetail') && tab.nodeDetailNodeId) return true;
  if (!ignore.includes('presetModal') && tab.presetModalNodeId) return true;
  if (!ignore.includes('layersModal') && tab.layersModalNodeId) return true;
  if (!ignore.includes('vizModal') && tab.vizModalNodeId) return true;
  return false;
}

/**
 * Is any modal open right now? Reads the stores directly, for callers outside
 * React's render pass (the global keydown handler).
 *
 * `ignore` names the modals the CALLER is itself inside, or is itself about
 * to close -- a caller that counted its own modal would gate away the very
 * key that dismisses it.
 */
export function isAnyModalOpen(ignore: readonly ModalName[] = IGNORE_NOTHING): boolean {
  return (
    dialogModalOpen(useDialogStore.getState(), ignore) ||
    uiModalOpen(useUIStore.getState(), ignore) ||
    tabModalOpen(useTabStore.getState(), ignore)
  );
}

/**
 * The same answer, subscribed: the component re-renders when it flips.
 *
 * Three selectors rather than three whole-store subscriptions, because each
 * one returns a boolean -- so a component only re-renders when a modal
 * actually opens or closes, not on every node drag or toggled preference.
 *
 * `ignore` may be a fresh array each render without causing extra work, for
 * the same reason: what zustand compares is the boolean that comes out.
 */
export function useAnyModalOpen(ignore: readonly ModalName[] = IGNORE_NOTHING): boolean {
  const dialog = useDialogStore((s) => dialogModalOpen(s, ignore));
  const ui = useUIStore((s) => uiModalOpen(s, ignore));
  const tab = useTabStore((s) => tabModalOpen(s, ignore));
  return dialog || ui || tab;
}
