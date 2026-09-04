import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { useState } from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { ActionMenu, type ActionMenuItem } from './ActionMenu';

/*
 * The menu primitive the Source Control header and commit box open. What is
 * asserted here is the keyboard and focus contract those two rely on — the
 * ARIA wiring, arrow-key wrap, and where focus lands when the menu goes away.
 */

const onA = vi.fn();
const onB = vi.fn();
const onC = vi.fn();

function items(overrides: Partial<ActionMenuItem>[] = []): ActionMenuItem[] {
  const base: ActionMenuItem[] = [
    { id: 'a', label: 'Alpha', onSelect: onA },
    { id: 'b', label: 'Bravo', onSelect: onB },
    { id: 'c', label: 'Charlie', onSelect: onC },
  ];
  return base.map((item, i) => ({ ...item, ...(overrides[i] ?? {}) }));
}

function renderMenu(props: Partial<Parameters<typeof ActionMenu>[0]> = {}) {
  return render(
    <ActionMenu label="More actions" items={items()} {...props}>
      dots
    </ActionMenu>,
  );
}

function trigger() {
  return screen.getByRole('button', { name: 'More actions' });
}

function item(name: string) {
  return screen.getByRole('menuitem', { name });
}

beforeEach(() => {
  onA.mockReset();
  onB.mockReset();
  onC.mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('ActionMenu', () => {
  it('starts closed, with a trigger that says what it opens', () => {
    renderMenu();
    expect(trigger().getAttribute('aria-haspopup')).toBe('menu');
    expect(trigger().getAttribute('aria-expanded')).toBe('false');
    expect(screen.queryByRole('menu')).toBeNull();
  });

  it('opens on click, names the menu, and focuses the first row', () => {
    renderMenu();
    fireEvent.click(trigger());
    expect(screen.getByRole('menu', { name: 'More actions' })).toBeTruthy();
    expect(trigger().getAttribute('aria-expanded')).toBe('true');
    expect(document.activeElement).toBe(item('Alpha'));
  });

  it('keeps exactly one row in the tab order', () => {
    renderMenu();
    fireEvent.click(trigger());
    expect(item('Alpha').getAttribute('tabindex')).toBe('0');
    expect(item('Bravo').getAttribute('tabindex')).toBe('-1');
    expect(item('Charlie').getAttribute('tabindex')).toBe('-1');
  });

  it('clicking the trigger again closes it and gives focus back', () => {
    renderMenu();
    fireEvent.click(trigger());
    fireEvent.click(trigger());
    expect(screen.queryByRole('menu')).toBeNull();
    expect(document.activeElement).toBe(trigger());
  });

  // ── Keyboard ───────────────────────────────────────────────────────────────

  it('ArrowDown and ArrowUp walk the rows and wrap at both ends', () => {
    renderMenu();
    fireEvent.click(trigger());

    fireEvent.keyDown(item('Alpha'), { key: 'ArrowDown' });
    expect(document.activeElement).toBe(item('Bravo'));
    fireEvent.keyDown(item('Bravo'), { key: 'ArrowDown' });
    expect(document.activeElement).toBe(item('Charlie'));
    // ...off the end and back to the top.
    fireEvent.keyDown(item('Charlie'), { key: 'ArrowDown' });
    expect(document.activeElement).toBe(item('Alpha'));
    // ...and off the top back to the end.
    fireEvent.keyDown(item('Alpha'), { key: 'ArrowUp' });
    expect(document.activeElement).toBe(item('Charlie'));
  });

  it('Home and End jump to the first and last row', () => {
    renderMenu();
    fireEvent.click(trigger());
    fireEvent.keyDown(item('Alpha'), { key: 'End' });
    expect(document.activeElement).toBe(item('Charlie'));
    fireEvent.keyDown(item('Charlie'), { key: 'Home' });
    expect(document.activeElement).toBe(item('Alpha'));
  });

  it('prevents the default for the keys it handles, so the page does not scroll', () => {
    renderMenu();
    fireEvent.click(trigger());
    expect(fireEvent.keyDown(item('Alpha'), { key: 'ArrowDown' })).toBe(false);
    expect(fireEvent.keyDown(item('Bravo'), { key: 'End' })).toBe(false);
    // ...and leaves everything else alone.
    expect(fireEvent.keyDown(item('Charlie'), { key: 'x' })).toBe(true);
  });

  it('Escape closes the menu and returns focus to the trigger', () => {
    renderMenu();
    fireEvent.click(trigger());
    fireEvent.keyDown(item('Alpha'), { key: 'Escape' });
    expect(screen.queryByRole('menu')).toBeNull();
    expect(document.activeElement).toBe(trigger());
  });

  it('Escape does not also reach a handler outside the menu', () => {
    const outside = vi.fn();
    document.addEventListener('keydown', outside);
    try {
      renderMenu();
      fireEvent.click(trigger());
      fireEvent.keyDown(item('Alpha'), { key: 'Escape' });
      expect(outside).not.toHaveBeenCalled();
    } finally {
      document.removeEventListener('keydown', outside);
    }
  });

  it('Tab closes the menu rather than walking its rows', () => {
    renderMenu();
    fireEvent.click(trigger());
    expect(fireEvent.keyDown(item('Alpha'), { key: 'Tab' })).toBe(false);
    expect(screen.queryByRole('menu')).toBeNull();
    expect(document.activeElement).toBe(trigger());
  });

  it('ArrowDown on the trigger opens onto the first row, ArrowUp onto the last', () => {
    renderMenu();
    fireEvent.keyDown(trigger(), { key: 'ArrowDown' });
    expect(document.activeElement).toBe(item('Alpha'));

    fireEvent.keyDown(item('Alpha'), { key: 'Escape' });
    fireEvent.keyDown(trigger(), { key: 'ArrowUp' });
    expect(document.activeElement).toBe(item('Charlie'));
  });

  // ── Choosing a row ─────────────────────────────────────────────────────────

  it('choosing a command runs it, closes the menu and returns focus', () => {
    renderMenu();
    fireEvent.click(trigger());
    fireEvent.click(item('Bravo'));
    expect(onB).toHaveBeenCalledTimes(1);
    expect(onA).not.toHaveBeenCalled();
    expect(screen.queryByRole('menu')).toBeNull();
    expect(document.activeElement).toBe(trigger());
  });

  it('a checkbox row toggles aria-checked in place and leaves the menu open', () => {
    function Host() {
      const [hidden, setHidden] = useState(false);
      return (
        <ActionMenu
          label="More actions"
          items={[
            {
              id: 'hide',
              label: 'Hide layout files',
              checked: hidden,
              onSelect: () => setHidden((v) => !v),
            },
            { id: 'docs', label: 'Setup guide', onSelect: onA },
          ]}
        >
          dots
        </ActionMenu>
      );
    }
    render(<Host />);
    fireEvent.click(trigger());

    const box = () => screen.getByRole('menuitemcheckbox', { name: 'Hide layout files' });
    expect(box().getAttribute('aria-checked')).toBe('false');
    fireEvent.click(box());
    expect(box().getAttribute('aria-checked')).toBe('true');
    // A setting is not a command: the menu is still there to change again.
    expect(screen.getByRole('menu')).toBeTruthy();
    fireEvent.click(box());
    expect(box().getAttribute('aria-checked')).toBe('false');
  });

  it('gives a plain command no aria-checked at all', () => {
    renderMenu();
    fireEvent.click(trigger());
    expect(item('Alpha').hasAttribute('aria-checked')).toBe(false);
    expect(screen.queryAllByRole('menuitemcheckbox')).toHaveLength(0);
  });

  // ── Disabled ───────────────────────────────────────────────────────────────

  it('skips a disabled row with the arrow keys and refuses to run it', () => {
    render(
      <ActionMenu label="More actions" items={items([{}, { disabled: true }])}>
        dots
      </ActionMenu>,
    );
    fireEvent.click(trigger());
    expect(document.activeElement).toBe(item('Alpha'));

    fireEvent.keyDown(item('Alpha'), { key: 'ArrowDown' });
    expect(document.activeElement).toBe(item('Charlie'));
    fireEvent.keyDown(item('Charlie'), { key: 'ArrowDown' });
    expect(document.activeElement).toBe(item('Alpha'));

    fireEvent.click(item('Bravo'));
    expect(onB).not.toHaveBeenCalled();
    expect(screen.getByRole('menu')).toBeTruthy();
  });

  it('opens onto the last ENABLED row, not the last row', () => {
    render(
      <ActionMenu label="More actions" items={items([{}, {}, { disabled: true }])}>
        dots
      </ActionMenu>,
    );
    fireEvent.keyDown(trigger(), { key: 'ArrowUp' });
    expect(document.activeElement).toBe(item('Bravo'));
    fireEvent.keyDown(item('Bravo'), { key: 'End' });
    expect(document.activeElement).toBe(item('Bravo'));
  });

  it('focuses the panel itself when every row is disabled, so Escape still works', () => {
    render(
      <ActionMenu
        label="More actions"
        items={items([{ disabled: true }, { disabled: true }, { disabled: true }])}
      >
        dots
      </ActionMenu>,
    );
    fireEvent.click(trigger());
    const menu = screen.getByRole('menu');
    expect(document.activeElement).toBe(menu);
    fireEvent.keyDown(menu, { key: 'ArrowDown' });
    expect(document.activeElement).toBe(menu);
    fireEvent.keyDown(menu, { key: 'Escape' });
    expect(screen.queryByRole('menu')).toBeNull();
    expect(document.activeElement).toBe(trigger());
  });

  it('a disabled trigger opens nothing', () => {
    renderMenu({ disabled: true });
    expect(trigger().hasAttribute('disabled')).toBe(true);
    fireEvent.keyDown(trigger(), { key: 'ArrowDown' });
    expect(screen.queryByRole('menu')).toBeNull();
  });

  // ── Dismissal by pointer ───────────────────────────────────────────────────

  it('a press anywhere else closes the menu, and leaves focus where it went', () => {
    renderMenu();
    fireEvent.click(trigger());
    fireEvent.mouseDown(document.body);
    expect(screen.queryByRole('menu')).toBeNull();
    // Deliberately NOT the trigger: the pointer has already chosen a target.
    expect(document.activeElement).not.toBe(trigger());
  });

  it('a press inside the menu does not close it', () => {
    renderMenu();
    fireEvent.click(trigger());
    fireEvent.mouseDown(screen.getByRole('menu'));
    expect(screen.getByRole('menu')).toBeTruthy();
  });

  it('stops listening for outside presses once it is closed', () => {
    const add = vi.spyOn(document, 'addEventListener');
    const remove = vi.spyOn(document, 'removeEventListener');
    renderMenu();
    fireEvent.click(trigger());
    expect(add.mock.calls.some(([type]) => type === 'mousedown')).toBe(true);
    fireEvent.keyDown(item('Alpha'), { key: 'Escape' });
    expect(remove.mock.calls.some(([type]) => type === 'mousedown')).toBe(true);
  });

  // ── Placement ──────────────────────────────────────────────────────────────

  // A DOMRect's fields are prototype getters, so spreading one yields {} and
  // every coordinate arrives as NaN. Spelled out instead.
  function fakeTriggerRect() {
    const rect = {
      x: 40, y: 100, width: 28, height: 24, top: 100, left: 40, bottom: 124, right: 68,
    };
    vi.spyOn(HTMLButtonElement.prototype, 'getBoundingClientRect').mockReturnValue({
      ...rect,
      toJSON: () => rect,
    } as DOMRect);
  }

  it('hangs the panel below the trigger, from its left edge by default', () => {
    fakeTriggerRect();
    renderMenu();
    fireEvent.click(trigger());
    const menu = screen.getByRole('menu');
    expect(menu.style.top).toBe('124px');
    expect(menu.style.left).toBe('40px');
    expect(menu.style.right).toBe('');
  });

  it("lines the panel up with the trigger's right edge when asked", () => {
    fakeTriggerRect();
    renderMenu({ align: 'end' });
    fireEvent.click(trigger());
    const menu = screen.getByRole('menu');
    expect(menu.style.top).toBe('124px');
    expect(menu.style.right).toBe(`${window.innerWidth - 68}px`);
    expect(menu.style.left).toBe('');
  });
});
