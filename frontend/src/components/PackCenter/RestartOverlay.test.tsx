import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, render, screen, fireEvent } from '@testing-library/react';
import { useI18n } from '../../i18n';
import { _resetPackStoreForTesting, usePackStore, type RestartPhase } from '../../store/packStore';
import { RestartOverlay } from './RestartOverlay';

let originalLocation: Location;
let reload: ReturnType<typeof vi.fn>;

function seed(phase: RestartPhase, over: { command?: string | null; agoMs?: number } = {}) {
  usePackStore.setState({
    restart: {
      phase,
      packId: 'gpu-torch',
      startedAt: phase === 'idle' ? null : Date.now() - (over.agoMs ?? 0),
      command: over.command ?? null,
    },
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  useI18n.setState({ locale: 'en' });
  _resetPackStoreForTesting();
  originalLocation = window.location;
  reload = vi.fn();
  Object.defineProperty(window, 'location', {
    value: { ...originalLocation, reload },
    configurable: true,
  });
});

afterEach(() => {
  Object.defineProperty(window, 'location', {
    value: originalLocation,
    configurable: true,
  });
  // Inside act(): this hook runs BEFORE Testing Library's cleanup, so the
  // overlay is still mounted and subscribed when the reset puts `restart`
  // back to idle. Unwrapped, that printed an "update was not wrapped in
  // act(...)" line for every case that rendered one.
  act(() => {
    _resetPackStoreForTesting();
  });
  vi.useRealTimers();
});

describe('RestartOverlay — idle', () => {
  it('renders nothing at all', () => {
    render(<RestartOverlay />);
    expect(screen.queryByRole('alertdialog')).toBeNull();
  });
});

describe('RestartOverlay — waiting', () => {
  it('blocks the page, says what is happening, and counts', () => {
    seed('waiting', { agoMs: 5000 });
    render(<RestartOverlay />);

    const dialog = screen.getByRole('alertdialog');
    expect(dialog).toHaveAttribute('aria-modal', 'true');
    expect(dialog).toHaveAttribute('aria-busy', 'true');
    expect(screen.getByText('Server restarting')).toBeInTheDocument();
    expect(
      screen.getByText('Waiting for the server to come back. This page reloads by itself.'),
    ).toBeInTheDocument();
    expect(screen.getByText('Waiting for 5 s')).toBeInTheDocument();
    // Focus starts inside the overlay rather than on the page behind it.
    expect(dialog).toHaveFocus();
  });

  it('shows an indeterminate bar, paired with a status line that keeps moving', () => {
    seed('waiting', { agoMs: 0 });
    render(<RestartOverlay />);

    const bar = screen.getByRole('progressbar', { name: 'Server restarting' });
    // No `aria-valuenow` is how ARIA spells "progress unknown".
    expect(bar).not.toHaveAttribute('aria-valuenow');

    // Under reduced motion the bar is a static sliver, so the elapsed counter
    // is the only thing that proves the page has not frozen.
    expect(screen.getByText('Waiting for 0 s')).toBeInTheDocument();
    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(screen.getByText('Waiting for 2 s')).toBeInTheDocument();
  });

  it('swallows Tab and Escape so the page underneath cannot be reached', () => {
    seed('waiting');
    render(<RestartOverlay />);

    for (const key of ['Tab', 'Escape']) {
      const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true });
      document.body.dispatchEvent(event);
      expect(event.defaultPrevented).toBe(true);
    }

    // Everything else is left alone: this is not a keyboard trap for its own
    // sake, only for the two keys that would leave or dismiss the overlay.
    const other = new KeyboardEvent('keydown', { key: 'a', bubbles: true, cancelable: true });
    document.body.dispatchEvent(other);
    expect(other.defaultPrevented).toBe(false);
  });

  it('offers no way out while the server is still expected back', () => {
    seed('waiting');
    render(<RestartOverlay />);
    expect(screen.queryByRole('button', { name: 'Reload now' })).toBeNull();
  });
});

describe('RestartOverlay — the server did not come back', () => {
  it('hands over the command and a reload button when nothing picked the restart up', () => {
    seed('notStarted', { command: 'cdui install --gpu cu128' });
    render(<RestartOverlay />);

    expect(
      screen.getByText('The server did not restart. Run this command, then reload:'),
    ).toBeInTheDocument();
    expect(screen.getByText('cdui install --gpu cu128')).toBeInTheDocument();
    expect(screen.getByRole('alertdialog')).not.toHaveAttribute('aria-busy');

    fireEvent.click(screen.getByRole('button', { name: 'Reload now' }));
    expect(reload).toHaveBeenCalledTimes(1);
  });

  it('reloads from the timeout state too, and says how long it waited', () => {
    seed('timeout');
    render(<RestartOverlay />);
    expect(
      screen.getByText('The server has not come back after 10 minutes.'),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Reload now' }));
    expect(reload).toHaveBeenCalledTimes(1);
  });

  it('stops swallowing keys, and puts focus on the button that gets out', () => {
    seed('waiting');
    render(<RestartOverlay />);

    act(() => {
      seed('timeout');
    });

    const button = screen.getByRole('button', { name: 'Reload now' });
    expect(button).toHaveFocus();

    const event = new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true });
    document.body.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
  });

  it('skips the command block when the server never sent one', () => {
    seed('notStarted', { command: null });
    render(<RestartOverlay />);
    expect(screen.getByRole('button', { name: 'Reload now' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Copy command' })).toBeNull();
  });
});
