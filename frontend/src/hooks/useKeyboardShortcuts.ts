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
      if (mod && !e.shiftKey && e.key === 'c') {
        e.preventDefault();
        useTabStore.getState().copySelectedNodes();
        return;
      }

      // Ctrl+V / Cmd+V — Paste
      if (mod && !e.shiftKey && e.key === 'v') {
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

      // Ctrl+B / Cmd+B — Collapse/expand the left sidebar to its icon rail
      // (#126). Shift excluded so future Ctrl+Shift+B combinations stay free.
      if (mod && !e.shiftKey && e.key === 'b') {
        e.preventDefault();
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
          activeTab.subgraphModalNodeId
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
