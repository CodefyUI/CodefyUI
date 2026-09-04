import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, render } from '@testing-library/react';
import { useDialogStore } from '../../store/dialogStore';
import { usePackStore, _resetPackStoreForTesting } from '../../store/packStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import { PackCenterModal } from '../PackCenter/PackCenterModal';

/**
 * One Escape key, two centers.
 *
 * The Package Center and the Plugin Center are separate windows with separate
 * Escape handlers, and both can be on screen at once -- a toast from either
 * store offers to open the other, and toasts sit above a modal. Each one
 * therefore stands down while the other is open, the same way both already
 * stand down for a confirm dialog and for the shortcuts modal.
 *
 * This case is the PACK side of that pair, and it lives here rather than in
 * `PackCenter/PackCenterModal.test.tsx` because that file is a frozen gate
 * for this branch: the Plugin Center reuses the pack panel's stylesheet, its
 * pill and its job follower, and the pack tests staying byte-identical is
 * what proves none of that reuse changed the panel it came from. The
 * PLUGIN side of the same rule is in `PluginCenterModal.test.tsx`, under
 * "leaves Escape alone while something is stacked above it".
 */

/** Fresh mocks for every action the pack panel calls on mount. */
function makeActions() {
  return {
    refresh: vi.fn(async () => {}),
    install: vi.fn(async () => {}),
    cancel: vi.fn(async () => {}),
    removeItem: vi.fn(async () => {}),
    dismissJob: vi.fn(() => {}),
    stopFollowing: vi.fn(() => {}),
  };
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useDialogStore.setState({ active: null });
  useUIStore.setState({
    packCenterOpen: false,
    packCenterFocusPackId: null,
    pluginCenterOpen: false,
    pluginCenterFocusPluginId: null,
    shortcutsModalOpen: false,
  });
  _resetPackStoreForTesting();
  // Actions installed through setState, never `vi.spyOn` on a getState()
  // snapshot: the panel reads `refresh` on mount, and a spy taken off a
  // snapshot outlives the store it was taken from.
  usePackStore.setState({ loaded: true, ...makeActions() });
});

afterEach(() => {
  // Inside act(): this runs BEFORE Testing Library's cleanup, so the panel is
  // still mounted and subscribed when `packCenterOpen` goes false.
  act(() => {
    useUIStore.setState({
      packCenterOpen: false,
      packCenterFocusPackId: null,
      pluginCenterOpen: false,
      pluginCenterFocusPluginId: null,
    });
  });
  vi.restoreAllMocks();
});

describe('the Package Center and the Plugin Center on one Escape key', () => {
  it('leaves Escape to the Plugin Center while both are open', () => {
    act(() => {
      useUIStore.getState().openPackCenter();
      useUIStore.getState().openPluginCenter();
    });
    render(<PackCenterModal />);

    act(() => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    });
    // The window underneath must not be the one that closes: the Plugin
    // Center is the later portal and the one the user is looking at.
    expect(useUIStore.getState().packCenterOpen).toBe(true);

    act(() => {
      useUIStore.getState().closePluginCenter();
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    });
    expect(useUIStore.getState().packCenterOpen).toBe(false);
  });
});
