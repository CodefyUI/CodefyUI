import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ShortcutsModal } from './ShortcutsModal';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useUIStore.setState({ shortcutsModalOpen: false });
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
    // 8 shortcut rows → 8 <kbd> elements.
    expect(document.querySelectorAll('kbd').length).toBe(8);
    // A platform-prefixed combo is present (Cmd+Z or Ctrl+Z).
    expect(screen.getByText(/(Cmd|Ctrl)\+Z$/)).toBeTruthy();
    expect(screen.getByText('Delete')).toBeTruthy();
    expect(screen.getByText('?')).toBeTruthy();
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
