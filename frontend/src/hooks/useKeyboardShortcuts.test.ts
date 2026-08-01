import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useKeyboardShortcuts } from './useKeyboardShortcuts';
import { useTabStore } from '../store/tabStore';
import { useUIStore } from '../store/uiStore';
import { useProjectStore } from '../store/projectStore';
import { saveActiveGraph } from '../utils/saveActiveGraph';

vi.mock('../utils/saveActiveGraph', () => ({
  saveActiveGraph: vi.fn(),
}));

// Spy holders for the store actions the handler dispatches to.
let undo: ReturnType<typeof vi.fn>;
let redo: ReturnType<typeof vi.fn>;
let copySelectedNodes: ReturnType<typeof vi.fn>;
let pasteNodes: ReturnType<typeof vi.fn>;
let applyLayout: ReturnType<typeof vi.fn>;
let toggleShortcutsModal: ReturnType<typeof vi.fn>;
let toggleSidebarCollapsed: ReturnType<typeof vi.fn>;
let toggleBypassForSelection: ReturnType<typeof vi.fn>;

beforeEach(() => {
  undo = vi.fn();
  redo = vi.fn();
  copySelectedNodes = vi.fn();
  pasteNodes = vi.fn();
  applyLayout = vi.fn();
  toggleShortcutsModal = vi.fn();
  toggleSidebarCollapsed = vi.fn();
  // Defaults to "nothing bypassable was selected", so mod+B falls through to
  // the sidebar unless a test says otherwise (core#128).
  toggleBypassForSelection = vi.fn().mockReturnValue(false);

  // Override only the actions exercised here; leave the rest of the store intact.
  useTabStore.setState({
    undo, redo, copySelectedNodes, pasteNodes, applyLayout, toggleBypassForSelection,
  } as any);
  useUIStore.setState({
    toggleShortcutsModal,
    toggleSidebarCollapsed,
    lastLayoutMode: 'all',
  } as any);
  useProjectStore.setState({ projectDir: null, projectName: null, loaded: false });
  vi.mocked(saveActiveGraph).mockClear();
});

afterEach(() => {
  vi.restoreAllMocks();
});

/**
 * Dispatch a keydown on `document` with an explicit target element so the
 * input/textarea/contentEditable guard can be exercised. jsdom sets
 * event.target to the dispatch target, so we dispatch on the element itself.
 */
function dispatchKey(
  init: KeyboardEventInit,
  target: EventTarget = document.body,
) {
  const event = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init });
  target.dispatchEvent(event);
  return event;
}

describe('useKeyboardShortcuts', () => {
  it('Ctrl+Z triggers undo and prevents default', () => {
    renderHook(() => useKeyboardShortcuts());
    const e = dispatchKey({ key: 'z', ctrlKey: true });
    expect(undo).toHaveBeenCalledTimes(1);
    expect(e.defaultPrevented).toBe(true);
  });

  it('Cmd+Z (metaKey) also triggers undo', () => {
    renderHook(() => useKeyboardShortcuts());
    dispatchKey({ key: 'z', metaKey: true });
    expect(undo).toHaveBeenCalledTimes(1);
  });

  it('Ctrl+Shift+Z triggers redo', () => {
    renderHook(() => useKeyboardShortcuts());
    const e = dispatchKey({ key: 'z', ctrlKey: true, shiftKey: true });
    expect(redo).toHaveBeenCalledTimes(1);
    expect(undo).not.toHaveBeenCalled();
    expect(e.defaultPrevented).toBe(true);
  });

  it('Ctrl+Y triggers redo (alternative)', () => {
    renderHook(() => useKeyboardShortcuts());
    const e = dispatchKey({ key: 'y', ctrlKey: true });
    expect(redo).toHaveBeenCalledTimes(1);
    expect(e.defaultPrevented).toBe(true);
  });

  it('Ctrl+C triggers copySelectedNodes', () => {
    renderHook(() => useKeyboardShortcuts());
    const e = dispatchKey({ key: 'c', ctrlKey: true });
    expect(copySelectedNodes).toHaveBeenCalledTimes(1);
    expect(e.defaultPrevented).toBe(true);
  });

  it('Ctrl+V triggers pasteNodes', () => {
    renderHook(() => useKeyboardShortcuts());
    const e = dispatchKey({ key: 'v', ctrlKey: true });
    expect(pasteNodes).toHaveBeenCalledTimes(1);
    expect(e.defaultPrevented).toBe(true);
  });

  it('Ctrl+S saves in project mode and prevents the browser default', () => {
    useProjectStore.setState({ projectDir: '/proj', projectName: 'proj', loaded: true });
    renderHook(() => useKeyboardShortcuts());
    const e = dispatchKey({ key: 's', ctrlKey: true });
    expect(saveActiveGraph).toHaveBeenCalledTimes(1);
    expect(e.defaultPrevented).toBe(true);
  });

  it('Cmd+S (metaKey) also saves in project mode', () => {
    useProjectStore.setState({ projectDir: '/proj', projectName: 'proj', loaded: true });
    renderHook(() => useKeyboardShortcuts());
    dispatchKey({ key: 's', metaKey: true });
    expect(saveActiveGraph).toHaveBeenCalledTimes(1);
  });

  it('Ctrl+S is a no-op in non-project mode, leaving the browser Save dialog untouched', () => {
    renderHook(() => useKeyboardShortcuts());
    const e = dispatchKey({ key: 's', ctrlKey: true });
    expect(saveActiveGraph).not.toHaveBeenCalled();
    expect(e.defaultPrevented).toBe(false);
  });

  it('Ctrl+Shift+S does not trigger save (reserved combination, shiftKey excluded)', () => {
    useProjectStore.setState({ projectDir: '/proj', projectName: 'proj', loaded: true });
    renderHook(() => useKeyboardShortcuts());
    dispatchKey({ key: 's', ctrlKey: true, shiftKey: true });
    expect(saveActiveGraph).not.toHaveBeenCalled();
  });

  // ── mod+B, the contested chord (core#128) ──
  //
  // #126 gave Ctrl+B to the sidebar; ComfyUI users reach for it to bypass a
  // node. It is now context-sensitive: bypass when the selection has one,
  // sidebar when it does not, with Ctrl+Shift+B as the unconditional sidebar
  // toggle. `toggleBypassForSelection`'s return value is the switch.

  it('Ctrl+B toggles the sidebar when nothing bypassable is selected', () => {
    renderHook(() => useKeyboardShortcuts());
    const e = dispatchKey({ key: 'b', ctrlKey: true });
    expect(toggleBypassForSelection).toHaveBeenCalledTimes(1);
    expect(toggleSidebarCollapsed).toHaveBeenCalledTimes(1);
    expect(e.defaultPrevented).toBe(true);
  });

  it('Cmd+B (metaKey) behaves the same way', () => {
    renderHook(() => useKeyboardShortcuts());
    dispatchKey({ key: 'b', metaKey: true });
    expect(toggleSidebarCollapsed).toHaveBeenCalledTimes(1);
  });

  it('Ctrl+B bypasses instead of touching the sidebar when a node is selected', () => {
    toggleBypassForSelection.mockReturnValue(true);
    renderHook(() => useKeyboardShortcuts());
    const e = dispatchKey({ key: 'b', ctrlKey: true });
    expect(toggleBypassForSelection).toHaveBeenCalledTimes(1);
    expect(toggleSidebarCollapsed).not.toHaveBeenCalled();
    expect(e.defaultPrevented).toBe(true);
  });

  it('Ctrl+Shift+B always toggles the sidebar, selection or not', () => {
    toggleBypassForSelection.mockReturnValue(true);
    renderHook(() => useKeyboardShortcuts());
    const e = dispatchKey({ key: 'b', ctrlKey: true, shiftKey: true });
    expect(toggleSidebarCollapsed).toHaveBeenCalledTimes(1);
    expect(toggleBypassForSelection).not.toHaveBeenCalled();
    expect(e.defaultPrevented).toBe(true);
  });

  it('Ctrl+Shift+B matches an uppercase key value too', () => {
    renderHook(() => useKeyboardShortcuts());
    dispatchKey({ key: 'B', ctrlKey: true, shiftKey: true });
    expect(toggleSidebarCollapsed).toHaveBeenCalledTimes(1);
  });

  it('plain Ctrl+B survives Caps Lock (uppercase key, shiftKey false)', () => {
    toggleBypassForSelection.mockReturnValue(true);
    renderHook(() => useKeyboardShortcuts());
    dispatchKey({ key: 'B', ctrlKey: true });
    expect(toggleBypassForSelection).toHaveBeenCalledTimes(1);
    expect(toggleSidebarCollapsed).not.toHaveBeenCalled();
  });

  it('Ctrl+B is ignored while typing in an input', () => {
    renderHook(() => useKeyboardShortcuts());
    const input = document.createElement('input');
    document.body.appendChild(input);
    dispatchKey({ key: 'b', ctrlKey: true }, input);
    expect(toggleSidebarCollapsed).not.toHaveBeenCalled();
    input.remove();
  });

  it('? toggles the shortcuts modal', () => {
    renderHook(() => useKeyboardShortcuts());
    const e = dispatchKey({ key: '?' });
    expect(toggleShortcutsModal).toHaveBeenCalledTimes(1);
    expect(e.defaultPrevented).toBe(true);
  });

  it('Shift+/ also toggles the shortcuts modal (alias branch)', () => {
    renderHook(() => useKeyboardShortcuts());
    const e = dispatchKey({ key: '/', shiftKey: true });
    expect(toggleShortcutsModal).toHaveBeenCalledTimes(1);
    expect(e.defaultPrevented).toBe(true);
  });

  it('Shift+L applies the last-used layout mode', () => {
    renderHook(() => useKeyboardShortcuts());
    const e = dispatchKey({ key: 'L', shiftKey: true });
    expect(applyLayout).toHaveBeenCalledTimes(1);
    expect(applyLayout).toHaveBeenCalledWith('all');
    expect(e.defaultPrevented).toBe(true);
  });

  it('Shift+l (lowercase) also applies layout via key.toLowerCase()', () => {
    renderHook(() => useKeyboardShortcuts());
    dispatchKey({ key: 'l', shiftKey: true });
    expect(applyLayout).toHaveBeenCalledWith('all');
  });

  it('skips all shortcuts when the target is an INPUT', () => {
    renderHook(() => useKeyboardShortcuts());
    const input = document.createElement('input');
    document.body.appendChild(input);
    dispatchKey({ key: 'z', ctrlKey: true }, input);
    expect(undo).not.toHaveBeenCalled();
    input.remove();
  });

  it('skips all shortcuts when the target is a TEXTAREA', () => {
    renderHook(() => useKeyboardShortcuts());
    const ta = document.createElement('textarea');
    document.body.appendChild(ta);
    dispatchKey({ key: 'z', ctrlKey: true }, ta);
    expect(undo).not.toHaveBeenCalled();
    ta.remove();
  });

  it('skips all shortcuts when the target is contentEditable', () => {
    renderHook(() => useKeyboardShortcuts());
    const div = document.createElement('div');
    // jsdom does not compute isContentEditable from the attribute; force it.
    Object.defineProperty(div, 'isContentEditable', { value: true, configurable: true });
    document.body.appendChild(div);
    dispatchKey({ key: 'z', ctrlKey: true }, div);
    expect(undo).not.toHaveBeenCalled();
    div.remove();
  });

  it('does nothing for an unhandled key combination', () => {
    renderHook(() => useKeyboardShortcuts());
    const e = dispatchKey({ key: 'a', ctrlKey: true });
    expect(undo).not.toHaveBeenCalled();
    expect(redo).not.toHaveBeenCalled();
    expect(copySelectedNodes).not.toHaveBeenCalled();
    expect(pasteNodes).not.toHaveBeenCalled();
    expect(applyLayout).not.toHaveBeenCalled();
    expect(toggleShortcutsModal).not.toHaveBeenCalled();
    expect(toggleSidebarCollapsed).not.toHaveBeenCalled();
    expect(e.defaultPrevented).toBe(false);
  });

  it('removes the listener on unmount (cleanup)', () => {
    const removeSpy = vi.spyOn(document, 'removeEventListener');
    const { unmount } = renderHook(() => useKeyboardShortcuts());
    unmount();
    expect(removeSpy).toHaveBeenCalledWith('keydown', expect.any(Function));

    // After unmount the handler no longer fires.
    dispatchKey({ key: 'z', ctrlKey: true });
    expect(undo).not.toHaveBeenCalled();
  });
});
