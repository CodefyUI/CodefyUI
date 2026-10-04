import { StrictMode } from 'react';
import { describe, it, expect, beforeEach, afterEach, vi, type Mock } from 'vitest';
import { act, render, screen, fireEvent } from '@testing-library/react';
import { useI18n } from '../../i18n';
import { _resetPackStoreForTesting, usePackStore } from '../../store/packStore';
import { useTabStore } from '../../store/tabStore';
import type { WorkspaceLock, WorkspaceRole } from '../../utils/workspaceLock';
import { RestartOverlay } from '../PackCenter/RestartOverlay';
import { WorkspaceLockOverlay } from './WorkspaceLockOverlay';

/** A claim whose role the test sets; "Edit here" moves it to claiming. */
function stubLock(initial: WorkspaceRole): {
  lock: WorkspaceLock & { takeOver: Mock<() => void> };
  setRole: (next: WorkspaceRole) => void;
} {
  let role = initial;
  const listeners = new Set<() => void>();
  const apply = (next: WorkspaceRole) => {
    role = next;
    for (const listener of [...listeners]) listener();
  };
  const lock = {
    backend: 'locks' as const,
    getRole: () => role,
    getEpoch: () => null,
    canWrite: () => role === 'editor',
    supersede: () => {},
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    decided: () => Promise.resolve(role),
    takeOver: vi.fn<() => void>(() => apply('claiming')),
    dispose: () => {},
  };
  return { lock, setRole: (next) => act(() => apply(next)) };
}

/** A key press on whatever has focus, the way the browser delivers one. */
function press(key: string, init: KeyboardEventInit = {}): KeyboardEvent {
  const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init });
  act(() => {
    (document.activeElement ?? document.body).dispatchEvent(event);
  });
  return event;
}

const READ_ONLY_TITLE = 'Read-only in this browser tab';
const MOVED_TITLE = 'Editing moved to another browser tab';

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
});

describe('WorkspaceLockOverlay - nothing to block', () => {
  it('renders nothing while this page edits, or before it knows', () => {
    for (const role of ['pending', 'editor'] as const) {
      const { unmount } = render(<WorkspaceLockOverlay lock={stubLock(role).lock} />);
      expect(screen.queryByRole('alertdialog')).toBeNull();
      unmount();
    }
  });

  it('renders nothing when the page started no claim', () => {
    render(<WorkspaceLockOverlay lock={null} />);
    expect(screen.queryByRole('alertdialog')).toBeNull();
  });
});

describe('WorkspaceLockOverlay - read-only', () => {
  it('says why, offers Edit here, and takes focus', () => {
    render(<WorkspaceLockOverlay lock={stubLock('readonly').lock} />);

    const dialog = screen.getByRole('alertdialog', { name: READ_ONLY_TITLE });
    expect(dialog).toHaveAccessibleDescription(
      'Only one browser tab can edit this workspace at a time. Changes made here are not saved.',
    );
    const button = screen.getByRole('button', { name: 'Edit here' });
    expect(button).toHaveFocus();
    expect(button).toHaveAccessibleDescription('Reloads this page with the latest saved work.');
  });

  it('hands over on Edit here, then waits with nothing left to press', () => {
    const { lock } = stubLock('readonly');
    render(<WorkspaceLockOverlay lock={lock} />);

    fireEvent.click(screen.getByRole('button', { name: 'Edit here' }));
    expect(lock.takeOver).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('alertdialog')).toHaveAccessibleDescription(
      'Waiting for the other browser tab to save…',
    );
    expect(screen.queryByRole('button')).toBeNull();
  });

  it('covers an empty workspace too', () => {
    // No tab at all is a legal saved state; the overlay reads no tab.
    useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
    render(<WorkspaceLockOverlay lock={stubLock('readonly').lock} />);
    expect(screen.getByRole('alertdialog', { name: READ_ONLY_TITLE })).toBeInTheDocument();
  });
});

describe('WorkspaceLockOverlay - editing moved away', () => {
  it('blocks the page as soon as the handover starts, and says it is saving', () => {
    const { lock, setRole } = stubLock('editor');
    render(<WorkspaceLockOverlay lock={lock} />);
    expect(screen.queryByRole('alertdialog')).toBeNull();

    setRole('releasing');
    const dialog = screen.getByRole('alertdialog', { name: MOVED_TITLE });
    expect(dialog).toHaveAccessibleDescription('Saving your last changes…');
    expect(dialog).toHaveAttribute('aria-busy', 'true');
    expect(dialog).toHaveFocus();
    expect(screen.queryByRole('button')).toBeNull();
  });

  it('says so once it has let go, and offers to take editing back', () => {
    const { lock, setRole } = stubLock('releasing');
    render(<WorkspaceLockOverlay lock={lock} />);

    setRole('displaced');
    const dialog = screen.getByRole('alertdialog', { name: MOVED_TITLE });
    expect(dialog).toHaveAccessibleDescription('Changes made here are no longer saved.');
    expect(dialog).not.toHaveAttribute('aria-busy');
    const button = screen.getByRole('button', { name: 'Edit here' });
    expect(button).toHaveFocus();
    fireEvent.click(button);
    expect(lock.takeOver).toHaveBeenCalledTimes(1);
  });
});

describe('WorkspaceLockOverlay - keyboard', () => {
  const onPageKey = vi.fn();

  beforeEach(() => {
    onPageKey.mockClear();
    document.addEventListener('keydown', onPageKey);
  });

  afterEach(() => {
    document.removeEventListener('keydown', onPageKey);
  });

  it('keeps every key from the page underneath', () => {
    render(<WorkspaceLockOverlay lock={stubLock('readonly').lock} />);

    // The shortcuts that edit: undo, delete, paste.
    press('z', { ctrlKey: true });
    press('Delete');
    press('v', { ctrlKey: true });
    expect(onPageKey).not.toHaveBeenCalled();
  });

  it('keeps focus on Edit here when Tab is pressed', () => {
    render(<WorkspaceLockOverlay lock={stubLock('readonly').lock} />);
    const button = screen.getByRole('button', { name: 'Edit here' });

    const event = press('Tab');
    expect(event.defaultPrevented).toBe(true);
    expect(button).toHaveFocus();
    press('Tab', { shiftKey: true });
    expect(button).toHaveFocus();
  });

  it('gives the keys back once it is gone', () => {
    const { unmount } = render(<WorkspaceLockOverlay lock={stubLock('readonly').lock} />);
    unmount();
    press('z', { ctrlKey: true });
    expect(onPageKey).toHaveBeenCalledTimes(1);
  });

  it('leaves focus to the restart overlay while a server restart is shown above it', () => {
    // The restart card sits on top, and while it waits it has no button; Tab
    // must not walk focus to "Edit here" underneath it, where Enter would
    // press a button nobody can see.
    usePackStore.setState({
      restart: { phase: 'waiting', packId: 'gpu-torch', startedAt: Date.now(), command: null },
    });
    try {
      render(
        <>
          <WorkspaceLockOverlay lock={stubLock('readonly').lock} />
          <RestartOverlay />
        </>,
      );
      const edit = screen.getByRole('button', { name: 'Edit here' });
      expect(edit).not.toHaveFocus();
      press('Tab');
      expect(edit).not.toHaveFocus();
      press('Tab', { shiftKey: true });
      expect(edit).not.toHaveFocus();

      // The restart is over: this card is the top one again.
      act(() => {
        _resetPackStoreForTesting();
      });
      expect(edit).toHaveFocus();
    } finally {
      act(() => {
        _resetPackStoreForTesting();
      });
    }
  });

  it("survives StrictMode's mount, unmount and mount again", () => {
    // Development mounts every effect twice; the second mount has to leave
    // exactly one listener in place, with focus where it belongs.
    const { lock } = stubLock('readonly');
    const { unmount } = render(
      <StrictMode>
        <WorkspaceLockOverlay lock={lock} />
      </StrictMode>,
    );
    expect(screen.getByRole('button', { name: 'Edit here' })).toHaveFocus();
    press('Delete');
    expect(onPageKey).not.toHaveBeenCalled();

    unmount();
    press('Delete');
    expect(onPageKey).toHaveBeenCalledTimes(1);
  });
});
