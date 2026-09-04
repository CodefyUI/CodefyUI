import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, render } from '@testing-library/react';
import { useDialogStore } from '../../store/dialogStore';
import { usePackStore, _resetPackStoreForTesting } from '../../store/packStore';
import { usePluginStore, _resetPluginStoreForTesting } from '../../store/pluginStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import { PackCenterModal } from '../PackCenter/PackCenterModal';
import { PluginCenterModal } from './PluginCenterModal';

/**
 * One Escape key, two centers.
 *
 * The Package Center and the Plugin Center are separate windows with separate
 * Escape handlers, and both can be on screen at once: each is reachable from
 * Settings and from the sidebar's Custom & Plugins tab, so one can be opened
 * over the other in either order.
 *
 * The rule is that the top-most window closes and the one underneath stands
 * down -- and "top-most" is a fact rather than an assumption about who opened
 * first: the Plugin Center's backdrop renders one z-index rung above the pack
 * one. Both handlers are on `window`, registered in OPEN order (each panel
 * renders null while closed, so its listener does not exist until it opens),
 * which is why the pack guard alone is not enough: with the Package Center
 * opened first its handler runs first, reads a `pluginCenterOpen` the plugin
 * handler has not cleared yet, and stands down; with the Plugin Center opened
 * first, the plugin handler runs first, closes, and `stopImmediatePropagation`
 * is what keeps the pack handler from acting on a store the same press has
 * already changed.
 *
 * Both orders live here rather than in `PackCenter/PackCenterModal.test.tsx`
 * because that file is a frozen gate for this branch: the Plugin Center reuses
 * the pack panel's stylesheet, its pill and its job follower, and the pack
 * tests staying byte-identical is what proves none of that reuse changed the
 * panel it came from.
 */

/** Fresh mocks for every action the pack panel calls on mount. */
function makePackActions() {
  return {
    refresh: vi.fn(async () => {}),
    install: vi.fn(async () => {}),
    cancel: vi.fn(async () => {}),
    removeItem: vi.fn(async () => {}),
    dismissJob: vi.fn(() => {}),
    stopFollowing: vi.fn(() => {}),
  };
}

/** The same for the plugin panel, whose body reads the catalog on mount. */
function makePluginActions() {
  return {
    refresh: vi.fn(async () => {}),
    cancel: vi.fn(async () => {}),
    dismissJob: vi.fn(() => {}),
  };
}

/** Both panels mounted and closed, so opening one is what registers its key. */
function renderBoth() {
  return render(
    <>
      <PackCenterModal />
      <PluginCenterModal />
    </>,
  );
}

const escape = () => act(() => {
  window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
});

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
  _resetPluginStoreForTesting();
  // Actions installed through setState, never `vi.spyOn` on a getState()
  // snapshot: both panels read `refresh` on mount, and a spy taken off a
  // snapshot outlives the store it was taken from.
  usePackStore.setState({ loaded: true, ...makePackActions() });
  usePluginStore.setState({ loaded: true, ...makePluginActions() });
});

afterEach(() => {
  // Inside act(): this runs BEFORE Testing Library's cleanup, so both panels
  // are still mounted and subscribed when their flags go false.
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
  it('closes the Plugin Center first when the Package Center opened first', () => {
    renderBoth();
    act(() => {
      useUIStore.getState().openPackCenter();
    });
    act(() => {
      useUIStore.getState().openPluginCenter();
    });

    escape();

    // The window underneath must not be the one that closes.
    expect(useUIStore.getState().pluginCenterOpen).toBe(false);
    expect(useUIStore.getState().packCenterOpen).toBe(true);

    escape();
    expect(useUIStore.getState().packCenterOpen).toBe(false);
  });

  it('closes the Plugin Center first when it opened first', () => {
    // The order the pack guard alone gets wrong: the pack handler is
    // registered first, so it runs after the plugin handler has already
    // written `pluginCenterOpen: false` -- and would close a second window on
    // one press.
    renderBoth();
    act(() => {
      useUIStore.getState().openPluginCenter();
    });
    act(() => {
      useUIStore.getState().openPackCenter();
    });

    escape();

    expect(useUIStore.getState().pluginCenterOpen).toBe(false);
    expect(useUIStore.getState().packCenterOpen).toBe(true);

    escape();
    expect(useUIStore.getState().packCenterOpen).toBe(false);
  });
});
