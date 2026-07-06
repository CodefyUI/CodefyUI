import { useEffect } from 'react';
import { useTabStore } from '../store/tabStore';
import { useUIStore } from '../store/uiStore';
import { useProjectStore } from '../store/projectStore';
import { saveActiveGraph } from '../utils/saveActiveGraph';

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
    };

    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, []);
}
