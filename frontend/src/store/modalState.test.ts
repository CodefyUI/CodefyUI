import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

import { isAnyModalOpen, useAnyModalOpen, type ModalName } from './modalState';
import { useDialogStore } from './dialogStore';
import { useTabStore } from './tabStore';
import { useUIStore } from './uiStore';

const ORIGINAL_TABS = useTabStore.getState().tabs;
const ORIGINAL_ACTIVE = useTabStore.getState().activeTabId;
const TAB_ID = 'tab-modalstate-test';

/** Patch the active tab, where the four per-tab modal ids live. */
function setActiveTab(patch: Record<string, unknown>) {
  useTabStore.setState((s: any) => ({
    tabs: s.tabs.map((t: any) => (t.id === TAB_ID ? { ...t, ...patch } : t)),
  }) as any);
}

/** Each modal, by name, as the store write that opens it. */
const MODALS: Array<[ModalName, () => void]> = [
  ['dialog', () => useDialogStore.setState({ active: { kind: 'confirm', title: 'sure?' }, resolve: null })],
  ['shortcuts', () => useUIStore.setState({ shortcutsModalOpen: true })],
  ['templateGallery', () => useUIStore.setState({ templateGalleryOpen: true })],
  ['packCenter', () => useUIStore.setState({ packCenterOpen: true })],
  ['pluginCenter', () => useUIStore.setState({ pluginCenterOpen: true })],
  ['gitDiff', () => useUIStore.setState({ gitDiff: { path: 'a.py', scope: 'worktree' } })],
  ['nodeDetail', () => setActiveTab({ nodeDetailNodeId: 'n1' })],
  ['presetModal', () => setActiveTab({ presetModalNodeId: 'n1' })],
  ['layersModal', () => setActiveTab({ layersModalNodeId: 'n1' })],
  ['vizModal', () => setActiveTab({ vizModalNodeId: 'n1' })],
];

beforeEach(() => {
  useTabStore.setState({
    tabs: [
      {
        ...ORIGINAL_TABS[0],
        id: TAB_ID,
        presetModalNodeId: null,
        layersModalNodeId: null,
        nodeDetailNodeId: null,
        vizModalNodeId: null,
      },
    ],
    activeTabId: TAB_ID,
  } as any);
  useUIStore.setState({
    shortcutsModalOpen: false,
    templateGalleryOpen: false,
    packCenterOpen: false,
    pluginCenterOpen: false,
    gitDiff: null,
  });
  useDialogStore.setState({ active: null, resolve: null });
});

afterEach(() => {
  useTabStore.setState({ tabs: ORIGINAL_TABS, activeTabId: ORIGINAL_ACTIVE } as any);
});

describe('isAnyModalOpen', () => {
  it('is false with nothing open', () => {
    expect(isAnyModalOpen()).toBe(false);
  });

  it.each(MODALS)('is true while %s is open', (_name, open) => {
    open();
    expect(isAnyModalOpen()).toBe(true);
  });

  it.each(MODALS)('ignores %s when the caller names it', (name, open) => {
    open();
    expect(isAnyModalOpen([name])).toBe(false);
  });

  it('still counts the others when one is ignored', () => {
    // The Package Center on top of the layers editor: ignoring its own modal
    // must not blind the layers editor to the panel above it (#475).
    setActiveTab({ layersModalNodeId: 'n1' });
    useUIStore.setState({ packCenterOpen: true });
    expect(isAnyModalOpen(['layersModal'])).toBe(true);
  });

  it('is false with no active tab at all', () => {
    // Since #472 the last tab is closable and `tabs: []` persists. No tab is
    // a real state, not a crash and not an open modal.
    useTabStore.setState({ tabs: [], activeTabId: null } as any);
    expect(isAnyModalOpen()).toBe(false);
  });

  it('reads the ACTIVE tab only — a modal on a background tab is not on screen', () => {
    useTabStore.setState((s: any) => ({
      tabs: [...s.tabs, { ...s.tabs[0], id: 'other', nodeDetailNodeId: 'n9' }],
    }) as any);
    expect(isAnyModalOpen()).toBe(false);
  });

  it('treats an absent modal id as closed, not open', () => {
    // A tab restored from an older workspace, or built by a fixture, carries
    // `undefined` where the store writes `null`. `!== null` called that open.
    setActiveTab({ nodeDetailNodeId: undefined, vizModalNodeId: undefined });
    expect(isAnyModalOpen()).toBe(false);
  });
});

describe('useAnyModalOpen', () => {
  it('starts false and follows the store when a modal opens and closes', () => {
    const { result } = renderHook(() => useAnyModalOpen());
    expect(result.current).toBe(false);

    act(() => useUIStore.setState({ packCenterOpen: true }));
    expect(result.current).toBe(true);

    act(() => useUIStore.setState({ packCenterOpen: false }));
    expect(result.current).toBe(false);
  });

  it('follows the per-tab modals too', () => {
    const { result } = renderHook(() => useAnyModalOpen());
    act(() => setActiveTab({ vizModalNodeId: 'n1' }));
    expect(result.current).toBe(true);
  });

  it('follows the dialog store', () => {
    const { result } = renderHook(() => useAnyModalOpen());
    act(() =>
      useDialogStore.setState({ active: { kind: 'prompt', title: 'name?' }, resolve: null }),
    );
    expect(result.current).toBe(true);
  });

  it('honours the ignore list, and still sees what is stacked above', () => {
    setActiveTab({ layersModalNodeId: 'n1' });
    const { result } = renderHook(() => useAnyModalOpen(['layersModal']));
    expect(result.current).toBe(false);

    act(() => useUIStore.setState({ pluginCenterOpen: true }));
    expect(result.current).toBe(true);
  });
});
