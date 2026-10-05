import { useEffect } from 'react';
import { useTabStore } from '../store/tabStore';
import { useUIStore } from '../store/uiStore';
import { isAnyModalOpen, type ModalName } from '../store/modalState';
import { saveActiveGraph } from '../utils/saveActiveGraph';

/** Node kinds with no detail modal to open (mirrors NodeDetailModal). */
const NO_DETAIL_NODE_TYPES = new Set(['noteNode']);

/**
 * Elements that answer Enter themselves. Buttons and links must keep firing
 * their own activation, and a `<select>` uses Enter to commit its choice —
 * hijacking any of them to open a modal would be a bug, not a shortcut.
 */
const ENTER_OWNING_TAGS = new Set(['BUTTON', 'A', 'SELECT', 'SUMMARY']);

/**
 * Is this key going to something the user types into? An input, a textarea,
 * an editable element, or a `<select>`, which uses printable keys for
 * type-ahead (the toolbar's device select is one). Every shortcut but Save
 * leaves such a key to it, and so does the canvases' Delete (`useDeleteKey`).
 */
export function isTypingTarget(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  const tag = el?.tagName;
  return (
    tag === 'INPUT' ||
    tag === 'TEXTAREA' ||
    tag === 'SELECT' ||
    Boolean(el?.isContentEditable)
  );
}

/**
 * Is the user holding a selection of ordinary page text?
 *
 * A collapsed selection (a caret, or nothing at all) is not one, and
 * `getSelection` is absent in enough embeddings to be worth guarding.
 */
function hasTextSelection(): boolean {
  return (window.getSelection?.()?.toString() ?? '') !== '';
}

/**
 * The chord that shows (or hides) the shortcuts sheet. A named test because
 * the modal gate below has to let exactly this one key through -- see there.
 */
function isHelpKey(e: KeyboardEvent): boolean {
  return e.key === '?' || (e.shiftKey && e.key === '/');
}

/** The shortcuts sheet, as the one modal `?` is allowed to act on. */
const HELP_KEY_IGNORES: readonly ModalName[] = ['shortcuts'];

/**
 * Ctrl+S / Cmd+S exactly: Shift and Alt make other chords. Lower-cased for
 * Caps Lock, as the letters in the handler below are. A text note being
 * edited asks too, to let this one key out to the handler (NoteNode).
 */
export function isSaveChord(e: KeyboardEvent): boolean {
  return (
    (e.metaKey || e.ctrlKey) &&
    !e.shiftKey &&
    !e.altKey &&
    (e.key ?? '').toLowerCase() === 's'
  );
}

/**
 * The browser's "Save page as" is never what Ctrl+S means here, whatever has
 * focus. Refused in the window's capture phase, the first stop of every key
 * event, because a handler further in can stop a key before it bubbles up to
 * the shortcut handler: the workspace lock overlay stops every key on the way
 * down. Saving is the handler's job.
 */
function refuseBrowserSave(e: KeyboardEvent) {
  if (isSaveChord(e)) e.preventDefault();
}

export function useKeyboardShortcuts() {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      // The letter in lower case for the chords below: Caps Lock alone
      // reports 'Z' with shiftKey false, and Shift reports 'Z' too, so a
      // compare with 'z' dropped Ctrl+Z under Caps Lock and every
      // Ctrl+Shift+Z. Not `e.code`, which names the key's position: a
      // different letter on AZERTY and QWERTZ keyboards. `?? ''` because a
      // keydown from an autofill pick can arrive without a `key` at all.
      const key = (e.key ?? '').toLowerCase();
      const tag = (e.target as HTMLElement)?.tagName;

      // Ctrl+S / Cmd+S — Save, in every mode (`refuseBrowserSave` has already
      // stopped the browser's "Save page as"). ID9 kept it to project mode
      // while a save outside a project always asked for a name; since
      // `saveActiveGraph` has one rule in both modes, this key and the
      // toolbar's Save do the same thing everywhere. Ahead of both gates
      // below: the one shortcut that also works from a field, since the
      // Inspector's fields commit on every keystroke and the store already
      // holds what is on screen. Never under a modal (the Layers editor and
      // a preset's Configure hold edits not applied yet, and the name prompt
      // is a modal itself), nor on key repeat (a save and a toast per repeat).
      if (isSaveChord(e)) {
        if (e.repeat || isAnyModalOpen()) return;
        void saveActiveGraph();
        return;
      }

      // Skip while the user types in a field (see `isTypingTarget`).
      if (isTypingTarget(e.target)) return;

      // Skip while ANY modal is open (#475). Every shortcut below acts on the
      // canvas, and this handler is bound to `document` — which a modal panel
      // is a portal into, not a separate surface. Nothing in a panel has to
      // take focus for the key to arrive here, so without this gate Shift+L
      // re-laid-out the graph the user could not see, and Ctrl+Z undid work
      // behind a confirm dialog.
      //
      // `?` is the single exception, and only because it is what OPENED the
      // shortcuts sheet: its own branch re-asks, counting every modal but
      // that one, so it still closes the sheet and still refuses to stack a
      // second one on the Package Center.
      if (isAnyModalOpen() && !isHelpKey(e)) return;

      // Ctrl+Z / Cmd+Z — Undo
      if (mod && !e.shiftKey && key === 'z') {
        e.preventDefault();
        useTabStore.getState().undo();
        return;
      }

      // Ctrl+Shift+Z / Cmd+Shift+Z — Redo
      if (mod && e.shiftKey && key === 'z') {
        e.preventDefault();
        useTabStore.getState().redo();
        return;
      }

      // Ctrl+Y / Cmd+Y — Redo (alternative). Not with Shift: Ctrl+Shift+Y is
      // the browser's (Edge opens Collections), as it was while the compare
      // was case-sensitive.
      if (mod && !e.shiftKey && key === 'y') {
        e.preventDefault();
        useTabStore.getState().redo();
        return;
      }

      // Ctrl+C / Cmd+C — Copy
      //
      // Yields to a real text selection, on top of the modal gate above. The
      // tag guard only skips inputs and textareas, so selecting ordinary page
      // text — a node label, a log line, a result — and pressing Ctrl+C copied
      // the SELECTED NODES instead and put nothing on the clipboard. A
      // non-empty selection means the user is copying text, which is the
      // browser's job, not ours.
      if (mod && !e.shiftKey && key === 'c') {
        if (hasTextSelection()) return;
        e.preventDefault();
        useTabStore.getState().copySelectedNodes();
        return;
      }

      // Ctrl+V / Cmd+V — Paste. Same yield: a selection is the user working
      // with text, and pasting nodes over it is not what they asked for.
      if (mod && !e.shiftKey && key === 'v') {
        if (hasTextSelection()) return;
        e.preventDefault();
        useTabStore.getState().pasteNodes();
        return;
      }

      // Ctrl+Shift+B / Cmd+Shift+B — Collapse/expand the left sidebar,
      // unconditionally. See the note on Ctrl+B below for why this exists.
      if (mod && e.shiftKey && e.key.toLowerCase() === 'b') {
        e.preventDefault();
        useUIStore.getState().toggleSidebarCollapsed();
        return;
      }

      // Ctrl+B / Cmd+B — CONTEXT-SENSITIVE (core#128).
      //
      // Two features want this chord. #126 gave it to the sidebar (VS Code's
      // binding); ComfyUI — which is where anyone reaching for "mute this
      // node" learned the gesture — gives it to bypass, and every graph
      // editor this one is measured against agrees. Neither could simply be
      // moved without breaking the muscle memory it was chosen for.
      //
      // Resolution: the selection decides. With a bypassable node selected
      // the canvas has the keyboard's attention and Ctrl+B means bypass;
      // with nothing selected there is no bypass to perform and it means the
      // sidebar, exactly as before. `toggleBypassForSelection` returning
      // false is what makes that fall-through total — a selection of only
      // notes / Start / preset nodes lands on the sidebar rather than doing
      // nothing at all.
      //
      // Ctrl+Shift+B (handled above) is the unconditional sidebar toggle, so
      // there is always a chord that does the sidebar regardless of what is
      // selected. Both are listed in the shortcuts modal.
      // `toLowerCase` because Caps Lock alone reports 'B' with shiftKey
      // false — without it the chord silently stops working.
      if (mod && !e.shiftKey && e.key.toLowerCase() === 'b') {
        e.preventDefault();
        if (useTabStore.getState().toggleBypassForSelection()) return;
        useUIStore.getState().toggleSidebarCollapsed();
        return;
      }

      // ? — Toggle shortcuts help.
      //
      // The one key the modal gate above lets past, so it can close the sheet
      // it opened. It still has to refuse every OTHER modal: pressing ? over
      // the Package Center used to put a second modal on top of it — and,
      // because the sheet sits below the panels in the stacking order, an
      // invisible one whose open state then swallowed Escape (#380).
      if (isHelpKey(e)) {
        if (isAnyModalOpen(HELP_KEY_IGNORES)) return;
        e.preventDefault();
        useUIStore.getState().toggleShortcutsModal();
        return;
      }

      // Shift+L — Auto Layout (last-used mode)
      if (!mod && e.shiftKey && e.key.toLowerCase() === 'l') {
        e.preventDefault();
        const mode = useUIStore.getState().lastLayoutMode;
        useTabStore.getState().applyLayout(mode);
        return;
      }

      // Enter — open the selected node's detail modal (#127). Enter is the
      // most overloaded key on the page: it must not steal activation from a
      // focused control, must not fire behind a confirm dialog whose primary
      // button is focused, and must not re-open a modal that is already up.
      //
      // Only the first of those is checked here. The other two used to be an
      // enumeration on this branch — the dialog, the shortcuts sheet and the
      // four per-tab modals — and the modal gate above now covers all six,
      // and the four panels the enumeration never knew about (#475). One
      // list, and this branch keeps only the guard that is its own.
      if (!mod && !e.shiftKey && !e.altKey && e.key === 'Enter') {
        if (ENTER_OWNING_TAGS.has(tag)) return;
        const { tabs, activeTabId } = useTabStore.getState();
        const activeTab = tabs.find((t) => t.id === activeTabId);
        if (!activeTab) return;
        const selectedId = activeTab.selectedNodeId;
        if (!selectedId) return;
        const node = activeTab.nodes.find((n) => n.id === selectedId);
        if (!node || NO_DETAIL_NODE_TYPES.has(node.type ?? '')) return;
        e.preventDefault();
        useTabStore.getState().openNodeDetail(selectedId);
        return;
      }
    };

    window.addEventListener('keydown', refuseBrowserSave, true);
    document.addEventListener('keydown', handler);
    return () => {
      window.removeEventListener('keydown', refuseBrowserSave, true);
      document.removeEventListener('keydown', handler);
    };
  }, []);
}
