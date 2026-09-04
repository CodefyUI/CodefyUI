import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { RefSection, refSectionIds } from './RefSection';

/*
 * The shell the three reference lists share: a titled, collapsible region with
 * a count, an actions slot and a list. What is pinned here is the disclosure
 * contract -- who owns the open state, what the button reports, and what the
 * list is called -- because the header's branch button opens one of these from
 * outside the component that draws it.
 */

/** The list this section owns, hidden or not. `getByRole` skips a hidden one. */
function list(kind: 'branches' | 'remotes' | 'stashes'): HTMLElement {
  const el = document.getElementById(refSectionIds(kind).listId);
  if (el === null) throw new Error(`no list for ${kind}`);
  return el;
}

function renderSection(over: Partial<Parameters<typeof RefSection>[0]> = {}) {
  const onOpenChange = vi.fn();
  const view = render(
    <RefSection
      kind="branches"
      title="Branches"
      count={2}
      open={false}
      onOpenChange={onOpenChange}
      {...over}
    >
      <li>main</li>
      <li>work</li>
    </RefSection>,
  );
  return { ...view, onOpenChange };
}

const toggle = () => screen.getByRole('button', { name: 'Branches' });

describe('RefSection', () => {
  it('is a named region whose button reports what it controls', () => {
    renderSection();
    expect(screen.getByRole('region', { name: 'Branches' })).toBeTruthy();
    expect(toggle().getAttribute('aria-expanded')).toBe('false');
    expect(toggle().getAttribute('aria-controls')).toBe(list('branches').id);
    // The heading is the first thing to lose room at a 180px panel, so the
    // name in full stays in a `title`.
    expect(toggle().getAttribute('title')).toBe('Branches');
  });

  it('asks the caller to open it rather than opening itself', () => {
    // The open state is persisted in the store, so the section that draws it
    // cannot also be the section that owns it.
    const { onOpenChange, rerender } = renderSection();
    fireEvent.click(toggle());
    expect(onOpenChange).toHaveBeenCalledWith(true);
    expect(toggle().getAttribute('aria-expanded')).toBe('false');

    rerender(
      <RefSection
        kind="branches"
        title="Branches"
        count={2}
        open
        onOpenChange={onOpenChange}
      >
        <li>main</li>
      </RefSection>,
    );
    expect(toggle().getAttribute('aria-expanded')).toBe('true');
    fireEvent.click(toggle());
    expect(onOpenChange).toHaveBeenLastCalledWith(false);
  });

  it('hides its rows while it is closed and shows them when it is open', () => {
    const { rerender, onOpenChange } = renderSection();
    expect(list('branches').hidden).toBe(true);

    rerender(
      <RefSection
        kind="branches"
        title="Branches"
        count={2}
        open
        onOpenChange={onOpenChange}
      >
        <li>main</li>
        <li>work</li>
      </RefSection>,
    );
    expect(list('branches').hidden).toBe(false);
    expect(screen.getAllByRole('listitem')).toHaveLength(2);
  });

  it('says how many rows it holds, whether or not it is open', () => {
    renderSection({ count: 7 });
    expect(screen.getByText('7')).toBeTruthy();
  });

  it('counts nothing at all until the list has been read', () => {
    // `null` is "not read yet", and the sections are closed on a fresh
    // profile -- so a `0` here is a claim about a repository nobody has asked
    // about, printed beside five branches. No number is the honest answer, and
    // the first read fills it in.
    renderSection({ count: null });
    // The ELEMENT, not the text: a span rendering `null` draws nothing and
    // would pass a `queryByText`, while still holding the column the header's
    // title is fighting for at 180px.
    expect(document.querySelector('[class*="groupCount"]')).toBeNull();
    expect(screen.getByRole('region', { name: 'Branches' })).toBeTruthy();
  });

  it('carries the header actions the section offers', () => {
    renderSection({
      actions: (
        <button type="button" aria-label="New Branch...">
          +
        </button>
      ),
    });
    expect(screen.getByRole('button', { name: 'New Branch...' })).toBeTruthy();
  });

  it('gives each kind stable ids, so a control outside it can point at its list', () => {
    // The header's branch button is not inside the section it expands, so the
    // ids cannot come from a `useId` only the section can see.
    const branches = refSectionIds('branches');
    const remotes = refSectionIds('remotes');
    expect(branches.listId).not.toBe(remotes.listId);
    expect(branches.headingId).not.toBe(branches.listId);

    renderSection({ kind: 'remotes', title: 'Remotes' });
    const heading = screen.getByRole('button', { name: 'Remotes' });
    expect(heading.id).toBe(remotes.headingId);
    expect(heading.getAttribute('aria-controls')).toBe(remotes.listId);
    expect(
      screen.getByRole('region', { name: 'Remotes' }).getAttribute('aria-labelledby'),
    ).toBe(remotes.headingId);
  });

  it('keeps the heading id with the kind when the section id is overridden', () => {
    // The header scrolls to `refSectionIds('branches').headingId`, which it
    // reads from the kind alone. A heading id derived from an overridden
    // `sectionId` would take that scroll target with it and say nothing.
    renderSection({ sectionId: 'somewhere-else' });
    const heading = screen.getByRole('button', { name: 'Branches' });
    expect(heading.id).toBe(refSectionIds('branches').headingId);
    expect(screen.getByRole('region', { name: 'Branches' }).id).toBe('somewhere-else');
  });
});
