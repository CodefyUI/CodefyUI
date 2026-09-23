import { useDialogStore } from './dialogStore';
import { useTabStore } from './tabStore';
import { useUIStore } from './uiStore';

/**
 * "Is anything modal on screen?", asked in one place (#475).
 *
 * Several places need the same answer, which is why this is not a field on
 * any one store:
 *
 *  - `useKeyboardShortcuts` asks inside a `document` keydown handler that has
 *    no React context to read from, and the tab strip asks in its own.
 *  - the two canvases ask from `onBeforeDelete`, at the moment their Delete key
 *    (`useDeleteKey`) is about to delete the selection. React Flow's own Delete
 *    binding is off on both, `deleteKeyCode={null}` at all times (#501). They
 *    used to switch it off only while a modal was up, and the binding then
 *    missed the release of the key that opened the modal: that key stayed
 *    "held" and swallowed the first Delete after the modal closed (#491).
 *
 * They all ask `isAnyModalOpen()`, which reads the three stores through the
 * one predicate below. Rebuilt separately in each place, the question would
 * drift the first time a modal is added -- which is exactly how the Package
 * Center, the Plugin Center, the Template Gallery and the Git diff came to be
 * invisible to the shortcut hook while the four older modals were not.
 *
 * ADDING A MODAL: give it a name in `ModalName` and a line in the matching
 * `*ModalOpen` reader. Nothing else has to change -- provided its open flag
 * lives in one of these three stores. A flag held in a component's own
 * `useState` is invisible here, which is how the Custom Nodes manager went on
 * letting Delete and Shift+L through after this file landed.
 */
export type ModalName =
  /** The in-app confirm / prompt (`utils/dialog`). */
  | 'dialog'
  | 'shortcuts'
  | 'templateGallery'
  | 'packCenter'
  | 'pluginCenter'
  | 'customNodeManager'
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
 * time. A fresh `[]` per call would be harmless, but the constant says
 * outright that nothing is being allocated per keystroke.
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
  if (!ignore.includes('customNodeManager') && s.customNodeManagerOpen) return true;
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
 * Is any modal open right now? Reads the stores directly, at the moment of
 * asking, which is when every caller needs the answer: the global keydown
 * handler, the tab strip's keys, the canvases' `onBeforeDelete`.
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
