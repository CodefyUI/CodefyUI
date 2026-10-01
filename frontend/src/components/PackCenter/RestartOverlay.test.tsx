import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, render, screen, fireEvent } from '@testing-library/react';
import { useI18n } from '../../i18n';
import { _resetPackStoreForTesting, usePackStore, type RestartPhase } from '../../store/packStore';
import { useTabStore } from '../../store/tabStore';
import type { NodeDefinition } from '../../types';
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
  useTabStore.setState({
    tabs: [], activeTabId: null as unknown as string, clipboard: null,
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
    // "Waiting for the server to come back" was the heading and the live
    // counter said a third time, over an indeterminate progress bar. What is
    // left is the only thing the overlay does not otherwise show: that the
    // reader does not have to reload by hand.
    expect(screen.getByText('This page reloads by itself.')).toBeInTheDocument();
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
    expect(screen.queryByRole('button', { name: 'Return to CodefyUI' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Reload now' })).toBeNull();
  });
});

describe('RestartOverlay — the server did not come back', () => {
  it.each(['notStarted', 'timeout'] as const)(
    'offers Return first and Reload second in the %s state',
    (phase) => {
      seed(phase, { command: phase === 'notStarted' ? 'cdui install --gpu cu128' : null });
      render(<RestartOverlay />);

      const buttons = screen.getAllByRole('button');
      expect(buttons.map((button) => button.textContent)).toEqual([
        'Return to CodefyUI',
        'Reload now',
      ]);
      expect(buttons[0]).toHaveFocus();
    },
  );

  it('returns to the page without reloading', () => {
    seed('notStarted', { command: 'cdui install --gpu cu128' });
    render(<RestartOverlay />);

    fireEvent.click(screen.getByRole('button', { name: 'Return to CodefyUI' }));

    expect(screen.queryByRole('alertdialog')).toBeNull();
    expect(usePackStore.getState().restart.phase).toBe('idle');
    expect(reload).not.toHaveBeenCalled();
  });

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

  it('stops swallowing keys, and puts focus on Return before Reload', () => {
    seed('waiting');
    render(<RestartOverlay />);

    act(() => {
      seed('timeout');
    });

    const returnButton = screen.getByRole('button', { name: 'Return to CodefyUI' });
    const reloadButton = screen.getByRole('button', { name: 'Reload now' });
    expect(returnButton).toHaveFocus();

    const event = new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true });
    document.body.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
    expect(reloadButton).toBeEnabled();
  });

  it('keeps canvas state, undo history, logs, and session-only secrets when returning', () => {
    const definition: NodeDefinition = {
      node_name: 'SecretNode',
      category: 'test',
      description: '',
      inputs: [],
      outputs: [],
      params: [
        {
          name: 'label', param_type: 'string', default: 'before',
          description: '', options: [], min_value: null, max_value: null,
        },
        {
          name: 'api_key', param_type: 'secret', default: '',
          description: '', options: [], min_value: null, max_value: null,
        },
      ],
    };
    useTabStore.setState({
      tabs: [], activeTabId: null as unknown as string, clipboard: null,
    });
    useTabStore.getState().addTab('kept');
    useTabStore.getState().addNode(definition, { x: 24, y: 48 });
    const nodeId = useTabStore.getState().getActiveTab().nodes[0].id;
    useTabStore.getState().pushUndoSnapshot();
    useTabStore.getState().updateNodeParams(nodeId, {
      label: 'after', api_key: 'sk-session',
    });
    useTabStore.getState().addLog({ message: 'kept log', type: 'info' });

    const before = useTabStore.getState().getActiveTab();
    const activeTabId = before.id;
    const nodesBefore = before.nodes;
    const undoDepthBefore = before.undoStack.length;
    const logsBefore = before.logs.map((entry) => entry.message);

    seed('timeout');
    render(<RestartOverlay />);
    fireEvent.click(screen.getByRole('button', { name: 'Return to CodefyUI' }));

    const after = useTabStore.getState().getActiveTab();
    expect(after.id).toBe(activeTabId);
    expect(after.nodes).toEqual(nodesBefore);
    expect(after.undoStack).toHaveLength(undoDepthBefore);
    expect(after.logs.map((entry) => entry.message)).toEqual(logsBefore);
    expect(after.nodes[0].data.params.api_key).toBe('sk-session');
    expect(reload).not.toHaveBeenCalled();
  });

  it('skips the command block when the server never sent one', () => {
    seed('notStarted', { command: null });
    render(<RestartOverlay />);
    expect(screen.getByRole('button', { name: 'Reload now' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Copy command' })).toBeNull();
  });
});
