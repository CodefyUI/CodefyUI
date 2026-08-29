import { useEffect } from 'react';
import { useTabStore } from '../store/tabStore';
import { useUIStore } from '../store/uiStore';
import { useDialogStore } from '../store/dialogStore';
import { useProjectStore } from '../store/projectStore';
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
 * Is the user holding a selection of ordinary page text?
 *
 * A collapsed selection (a caret, or nothing at all) is not one, and
 * `getSelection` is absent in enough embeddings to be worth guarding.
 */
function hasTextSelection(): boolean {
  return (window.getSelection?.()?.toString() ?? '') !== '';
}

export function useKeyboardShortcuts() {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      const tag = (e.target as HTMLElement)?.tagName;
      // Skip if user is typing in an input/textarea
      if (tag === 'INPUT' || tag === 'TEXTAREA' || (e.target as HTMLElement)?.isContentEditable) {
        return;
      }

      // Ctrl+Z / Cmd+Z — Undo
      if (mod && !e.shiftKey && e.key === 'z') {
        e.preventDefault();
        useTabStore.getState().undo();
        return;
      }

      // Ctrl+Shift+Z / Cmd+Shift+Z — Redo
      if (mod && e.shiftKey && e.key === 'z') {
        e.preventDefault();
        useTabStore.getState().redo();
        return;
      }

      // Ctrl+Y / Cmd+Y — Redo (alternative)
      if (mod && e.key === 'y') {
        e.preventDefault();
        useTabStore.getState().redo();
        return;
      }

      // Ctrl+C / Cmd+C — Copy
      //
      // Yields to a real text selection. The guard above only skips inputs and
      // textareas, so selecting ordinary page text — the install command in
      // the Package Center's <pre>, which the "could not copy" toast tells the
      // user to copy by hand — and pressing Ctrl+C copied the SELECTED NODES
      // instead and put nothing on the clipboard. A non-empty selection means
      // the user is copying text, which is the browser's job, not ours.
      if (mod && !e.shiftKey && e.key === 'c') {
        if (hasTextSelection()) return;
        e.preventDefault();
        useTabStore.getState().copySelectedNodes();
        return;
      }

      // Ctrl+V / Cmd+V — Paste. Same yield: a selection is the user working
      // with text, and pasting nodes over it is not what they asked for.
      if (mod && !e.shiftKey && e.key === 'v') {
        if (hasTextSelection()) return;
        e.preventDefault();
        useTabStore.getState().pasteNodes();
        return;
      }

      // Ctrl+S / Cmd+S — Save. Project mode only, so non-project keeps the
      // browser's native behavior and this never hijacks it (ID9).
      if (mod && !e.shiftKey && e.key === 's') {
        if (useProjectStore.getState().projectDir !== null) {
          e.preventDefault();
          void saveActiveGraph();
        }
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

      // ? — Toggle shortcuts help
      if (e.key === '?' || (e.shiftKey && e.key === '/')) {
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

      // Enter — open the selected node's detail modal (#127). Every guard
      // below exists because Enter is the most overloaded key on the page:
      // it must not steal activation from a focused control, must not fire
      // behind a confirm dialog whose primary button is focused, and must not
      // re-open a modal that is already up.
      if (!mod && !e.shiftKey && !e.altKey && e.key === 'Enter') {
        if (ENTER_OWNING_TAGS.has(tag)) return;
        if (useDialogStore.getState().active !== null) return;
        if (useUIStore.getState().shortcutsModalOpen) return;
        const { tabs, activeTabId } = useTabStore.getState();
        const activeTab = tabs.find((t) => t.id === activeTabId);
        if (!activeTab) return;
        if (
          activeTab.nodeDetailNodeId ||
          activeTab.presetModalNodeId ||
          activeTab.layersModalNodeId
        ) {
          return;
        }
        const selectedId = activeTab.selectedNodeId;
        if (!selectedId) return;
        const node = activeTab.nodes.find((n) => n.id === selectedId);
        if (!node || NO_DETAIL_NODE_TYPES.has(node.type ?? '')) return;
        e.preventDefault();
        useTabStore.getState().openNodeDetail(selectedId);
        return;
      }
    };

    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, []);
}
