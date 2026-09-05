import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, render } from '@testing-library/react';
import { useDialogStore } from '../../store/dialogStore';
import { usePackStore, _resetPackStoreForTesting } from '../../store/packStore';
import { usePluginStore, _resetPluginStoreForTesting } from '../../store/pluginStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import { PackCenterModal } from '../PackCenter/PackCenterModal';
import { GitDiffModal } from '../SourceControl/GitDiffModal';
import { PluginCenterModal } from './PluginCenterModal';

// The diff window reads a patch as it opens. Stubbed at the module: these
// cases are about one keypress, and a real read would leave a fetch in
// flight for each of them.
vi.mock('../../api/git', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/git')>();
  return {
    ...actual,
    getGitDiff: vi.fn(async () => ({
      patch: '',
      binary: false,
      truncated: false,
      oldRef: 'index',
      newRef: 'worktree',
      oldMissing: false,
      newMissing: false,
    })),
    getGitFile: vi.fn(),
  };
});

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

/** The same, with the diff window -- a third handler on the same key. */
function renderAll() {
  return render(
    <>
      <PackCenterModal />
      <PluginCenterModal />
      <GitDiffModal />
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
    gitDiff: null,
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
      gitDiff: null,
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

describe('the diff window, which is the rung above both of them', () => {
  /*
   * A third window on the same key, and the only one that can be opened from
   * a panel BEHIND a Center is not the question -- it is opened from the
   * sidebar, which either Center covers. What matters is the other direction:
   * a Center left open underneath must not be closed by the press that closes
   * the diff, whichever handler happens to be registered first.
   */
  it('closes on top of the Package Center and leaves it open', async () => {
    renderAll();
    act(() => {
      useUIStore.getState().openPackCenter();
    });
    await act(async () => {
      useUIStore.getState().openGitDiff({ path: 'src/model.py', scope: 'worktree' });
    });

    escape();

    expect(useUIStore.getState().gitDiff).toBeNull();
    expect(useUIStore.getState().packCenterOpen).toBe(true);
  });

  it('stops the press it acted on, so nothing opened after it also closes', async () => {
    // The other registration order: this window's handler runs FIRST, and by
    // the time a Center's runs, `close()` has already emptied the `gitDiff`
    // that Center stands down on. `stopImmediatePropagation` is what keeps
    // one press from closing two windows here.
    renderAll();
    await act(async () => {
      useUIStore.getState().openGitDiff({ path: 'src/model.py', scope: 'worktree' });
    });
    act(() => {
      useUIStore.getState().openPackCenter();
    });

    escape();

    expect(useUIStore.getState().gitDiff).toBeNull();
    expect(useUIStore.getState().packCenterOpen).toBe(true);
  });

  it('closes on top of the Plugin Center and leaves it open', async () => {
    // The order the diff window's own `stopImmediatePropagation` does not
    // cover on its own: the plugin handler is registered FIRST here, so it
    // runs first and has to stand down on a `gitDiff` it can see.
    renderAll();
    act(() => {
      useUIStore.getState().openPluginCenter();
    });
    await act(async () => {
      useUIStore.getState().openGitDiff({ path: 'src/model.py', scope: 'worktree' });
    });

    escape();

    expect(useUIStore.getState().gitDiff).toBeNull();
    expect(useUIStore.getState().pluginCenterOpen).toBe(true);
  });
});
