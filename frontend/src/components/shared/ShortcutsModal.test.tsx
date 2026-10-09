import { readFileSync } from 'node:fs';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, render, screen, fireEvent } from '@testing-library/react';
import { ShortcutsModal } from './ShortcutsModal';
import { useDialogStore } from '../../store/dialogStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useUIStore.setState({ shortcutsModalOpen: false });
  useDialogStore.setState({ active: null });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('ShortcutsModal', () => {
  it('renders nothing when the modal is closed', () => {
    const { container } = render(<ShortcutsModal />);
    expect(container.firstChild).toBeNull();
  });

  it('renders the title and every shortcut row when open', () => {
    useUIStore.setState({ shortcutsModalOpen: true });
    render(<ShortcutsModal />);
    expect(screen.getByText(useI18n.getState().t('shortcuts.title'))).toBeTruthy();
    // 13 shortcut rows → 13 <kbd> elements. (core#128 added the bypass row
    // and the unconditional sidebar chord alongside the context-sensitive one;
    // the Source Control commit chord is the twelfth, Save the thirteenth.)
    expect(document.querySelectorAll('kbd').length).toBe(13);
    // A platform-prefixed combo is present (Cmd+Z or Ctrl+Z).
    expect(screen.getByText(/(Cmd|Ctrl)\+Z$/)).toBeTruthy();
    expect(screen.getByText('Delete')).toBeTruthy();
    expect(screen.getByText('?')).toBeTruthy();
  });

  it('spells out both halves of the context-sensitive mod+B (core#128)', () => {
    useUIStore.setState({ shortcutsModalOpen: true });
    render(<ShortcutsModal />);
    const { t } = useI18n.getState();
    // Two rows share the chord — bypass when a node is selected, sidebar
    // otherwise — so the modal has to name both rather than pick one.
    expect(screen.getAllByText(/(Cmd|Ctrl)\+B$/)).toHaveLength(2);
    expect(screen.getByText(t('shortcuts.bypass'))).toBeTruthy();
    expect(screen.getByText(t('shortcuts.toggleSidebar'))).toBeTruthy();
    // ...plus the unconditional sidebar chord.
    expect(screen.getByText(/(Cmd|Ctrl)\+Shift\+B$/)).toBeTruthy();
    expect(screen.getByText(t('shortcuts.toggleSidebarAlways'))).toBeTruthy();
  });

  // The one chord in the list that belongs to a text box rather than the
  // canvas: the global handler skips every textarea, so it is only reachable
  // from the Source Control message box and the row has to say which box.
  it('lists the Source Control commit chord', () => {
    useUIStore.setState({ shortcutsModalOpen: true });
    render(<ShortcutsModal />);
    expect(screen.getByText(/(Cmd|Ctrl)\+Enter$/)).toBeTruthy();
    expect(
      screen.getByText(useI18n.getState().t('shortcuts.commit')),
    ).toBeTruthy();
  });

  // Ctrl+S saves in every mode, so the sheet lists it. The literal text, not
  // `t(...)`: a missing key would render as the key itself and still match.
  it('lists Ctrl+S as Save graph', () => {
    useUIStore.setState({ shortcutsModalOpen: true });
    render(<ShortcutsModal />);
    const keys = screen.getByText(/^(Cmd|Ctrl)\+S$/);
    expect(keys.parentElement?.textContent).toMatch(/^(Cmd|Ctrl)\+SSave graph$/);
  });

  it('clicking the overlay toggles (closes) the modal', () => {
    useUIStore.setState({ shortcutsModalOpen: true });
    const { container } = render(<ShortcutsModal />);
    const overlay = container.firstElementChild as HTMLElement;
    fireEvent.click(overlay);
    expect(useUIStore.getState().shortcutsModalOpen).toBe(false);
  });

  it('clicking inside the modal does not toggle (stopPropagation)', () => {
    useUIStore.setState({ shortcutsModalOpen: true });
    render(<ShortcutsModal />);
    // The header/title sits inside the modal; clicking it must not bubble.
    fireEvent.click(screen.getByText(useI18n.getState().t('shortcuts.title')));
    expect(useUIStore.getState().shortcutsModalOpen).toBe(true);
  });

  it('clicking the close (×) button toggles the modal', () => {
    useUIStore.setState({ shortcutsModalOpen: true });
    render(<ShortcutsModal />);
    fireEvent.click(screen.getByRole('button'));
    expect(useUIStore.getState().shortcutsModalOpen).toBe(false);
  });

  it('names itself and its close button for a screen reader', () => {
    useUIStore.setState({ shortcutsModalOpen: true });
    render(<ShortcutsModal />);
    expect(screen.getByRole('dialog', { name: 'Keyboard Shortcuts' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Close keyboard shortcuts' })).toBeTruthy();
  });

  it('uses the Mac modifier label when navigator.platform is a Mac', async () => {
    vi.resetModules();
    vi.stubGlobal('navigator', { platform: 'MacIntel', language: 'en-US' } as Navigator);
    const mod = await import('./ShortcutsModal');
    const ui = await import('../../store/uiStore');
    ui.useUIStore.setState({ shortcutsModalOpen: true });
    render(<mod.ShortcutsModal />);
    expect(screen.getByText('Cmd+Z')).toBeTruthy();
    vi.unstubAllGlobals();
  });

  it('uses the Ctrl modifier label on non-Mac platforms', async () => {
    vi.resetModules();
    vi.stubGlobal('navigator', { platform: 'Win32', language: 'en-US' } as Navigator);
    const mod = await import('./ShortcutsModal');
    const ui = await import('../../store/uiStore');
    ui.useUIStore.setState({ shortcutsModalOpen: true });
    render(<mod.ShortcutsModal />);
    expect(screen.getByText('Ctrl+Z')).toBeTruthy();
    vi.unstubAllGlobals();
  });
});

// Stack policy (#490): the sheet may open over any panel, so it sits above
// them all and owns Escape while it is open.
describe('ShortcutsModal — Escape and focus', () => {
  /** A press as a keyboard makes it: on the focused element, bubbling. */
  function pressAtFocus(key: string) {
    const e = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true });
    act(() => {
      (document.activeElement ?? document.body).dispatchEvent(e);
    });
    return e;
  }

  function renderWithOpener() {
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();
    render(<ShortcutsModal />);
    act(() => {
      useUIStore.setState({ shortcutsModalOpen: true });
    });
    return opener;
  }

  afterEach(() => {
    document.body.querySelectorAll(':scope > button').forEach((b) => b.remove());
  });

  it('takes focus as it opens', () => {
    renderWithOpener();
    expect(document.activeElement).toBe(screen.getByRole('dialog'));
  });

  it('closes on Escape, hands focus back, and keeps the press from what is underneath', () => {
    const opener = renderWithOpener();
    const underneath = vi.fn();
    window.addEventListener('keydown', underneath);
    try {
      const e = pressAtFocus('Escape');
      expect(useUIStore.getState().shortcutsModalOpen).toBe(false);
      expect(e.defaultPrevented).toBe(true);
      expect(underneath).not.toHaveBeenCalled();
      expect(document.activeElement).toBe(opener);
    } finally {
      window.removeEventListener('keydown', underneath);
    }
  });

  it('lets every other key through', () => {
    renderWithOpener();
    const underneath = vi.fn();
    window.addEventListener('keydown', underneath);
    try {
      pressAtFocus('a');
      expect(useUIStore.getState().shortcutsModalOpen).toBe(true);
      expect(underneath).toHaveBeenCalledTimes(1);
    } finally {
      window.removeEventListener('keydown', underneath);
    }
  });

  it('stands down for a confirm dialog, which renders above it', () => {
    renderWithOpener();
    useDialogStore.setState({ active: { kind: 'confirm', title: 'x' } as never });
    const e = pressAtFocus('Escape');
    expect(useUIStore.getState().shortcutsModalOpen).toBe(true);
    expect(e.defaultPrevented).toBe(false);
  });

  it('does not move focus back to an element that is gone', () => {
    const opener = renderWithOpener();
    opener.remove();
    pressAtFocus('Escape');
    expect(useUIStore.getState().shortcutsModalOpen).toBe(false);
    expect(document.activeElement).not.toBe(opener);
  });
});

// The z-index ladder the policy depends on, read from the stylesheets
// themselves: every literal there carries its rung, and these are the rungs
// the sheet has to sit between.
describe('ShortcutsModal — stacking order', () => {
  /** The `z-index` of one rule, by selector, in one stylesheet under src/. */
  function zIndexOf(path: string, selector: string): number {
    const css = readFileSync(`src/${path}`, 'utf8');
    const escaped = selector.replace('.', '\\.');
    const rule = new RegExp(`(?:^|\\n)${escaped}\\s*\\{([^}]*)\\}`).exec(css);
    if (!rule) throw new Error(`${selector} not found in ${path}`);
    const z = /\bz-index:\s*(\d+)/.exec(rule[1]);
    if (!z) throw new Error(`${selector} in ${path} has no z-index`);
    return Number(z[1]);
  }

  const sheet = () => zIndexOf('components/shared/ShortcutsModal.module.css', '.overlay');

  it.each([
    ['the Package Center', 'components/PackCenter/PackCenterModal.module.css', '.backdrop'],
    ['the Plugin Center', 'components/PluginCenter/PluginCenterModal.module.css', '.topBackdrop'],
    ['the diff window', 'components/SourceControl/GitDiffModal.module.css', '.backdrop'],
    ['the template gallery', 'components/TemplateGallery/TemplateGalleryModal.module.css', '.backdrop'],
    ['node details', 'components/NodeDetailModal/NodeDetailModal.module.css', '.backdrop'],
    ['the preset editor', 'components/PresetModal/PresetConfigModal.module.css', '.overlay'],
    ['the Custom Nodes manager', 'components/CustomNodeManager/CustomNodeManager.module.css', '.overlay'],
    ['the scatter viewer', 'components/shared/ScatterModal.module.css', '.backdrop'],
    ['the heatmap viewer', 'components/shared/HeatmapModal.module.css', '.backdrop'],
    ['the New Sweep dialog', 'components/ResultsPanel/NewSweepDialog.module.css', '.backdrop'],
  ])('renders above %s', (_name, path, selector) => {
    expect(sheet()).toBeGreaterThan(zIndexOf(path, selector));
  });

  it.each([
    ['the confirm dialog', 'components/shared/DialogContainer.module.css', '.backdrop'],
    ['the toast stack', 'components/shared/Toast.module.css', '.container'],
    ['the workspace lock', 'components/WorkspaceLock/WorkspaceLockOverlay.module.css', '.backdrop'],
    ['the restart overlay', 'components/PackCenter/RestartOverlay.module.css', '.backdrop'],
  ])('renders below %s', (_name, path, selector) => {
    expect(sheet()).toBeLessThan(zIndexOf(path, selector));
  });
});
