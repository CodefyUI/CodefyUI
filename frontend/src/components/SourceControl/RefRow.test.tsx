import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { RefRow } from './RefRow';
import { useI18n } from '../../i18n';
import { _resetGitStoreForTesting } from '../../store/gitStore';
import styles from './SourceControl.module.css';

/*
 * The box every reference row is drawn in, asked the two questions no
 * component test above it can ask: which half of the row keeps its width, and
 * how a meta that is a SHORTHAND is announced.
 *
 * jsdom applies no stylesheet, so a width cannot be measured here -- what can
 * be pinned is the class each half is given, which is what the stylesheet
 * hangs the two proportions off. The widths themselves are a browser pass at
 * 180px and 250px.
 */

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  _resetGitStoreForTesting();
});

afterEach(() => {
  _resetGitStoreForTesting();
  vi.restoreAllMocks();
});

/** The two halves of the row showing *name*. */
function halves(name: string): { rowName: HTMLElement; meta: HTMLElement | null } {
  const rowName = screen.getByText(name);
  const li = rowName.closest('li');
  if (li === null) throw new Error(`no row named ${name}`);
  return {
    rowName,
    meta: li.querySelector<HTMLElement>(`.${styles.rowDir}`),
  };
}

describe('RefRow: which half gives way', () => {
  it('lets the name take the room by default, with the meta firm beside it', () => {
    // A branch row: a name, and a count that is two glyphs wide. Shrinking is
    // weighted by the base size, so the long half takes pixels off the short
    // one whatever the shrink factors say -- and here the short half is the
    // one that means something at a glance.
    render(<RefRow name="feature/long-branch-name" meta="↑2 ↓3" actions={[]} />);
    const { rowName, meta } = halves('feature/long-branch-name');

    expect(rowName.className.split(' ')).toContain(styles.nameElastic);
    expect(rowName.className.split(' ')).not.toContain(styles.nameFirm);
    expect(meta?.className.split(' ')).toContain(styles.metaFirm);
  });

  it('keeps the NAME whole where the meta is the long half', () => {
    // A remote row. At 250px `origin` was drawn as "ori..." beside a URL with
    // 200px to give, which is a row nobody can tell from `origin-backup`.
    render(
      <RefRow
        name="origin"
        meta="https://github.com/owner/repository.git"
        firm="name"
        actions={[]}
      />,
    );
    const { rowName, meta } = halves('origin');

    expect(rowName.className.split(' ')).toContain(styles.nameFirm);
    expect(rowName.className.split(' ')).not.toContain(styles.nameElastic);
    // ...and the URL is the flexible one, which is `.rowDir` on its own.
    expect(meta?.className.split(' ')).not.toContain(styles.metaFirm);
  });
});

describe('RefRow: a meta that is a shorthand', () => {
  it('draws the short form and announces the sentence', () => {
    render(
      <RefRow
        name="work"
        action={{ label: 'Switch to work', onSelect: vi.fn() }}
        meta="↑2 ↓3"
        metaLabel="2 to push, 3 to pull"
        actions={[]}
      />,
    );

    const button = screen.getByRole('button', { name: 'Switch to work' });
    expect(button).toHaveAccessibleDescription('2 to push, 3 to pull');
    // The glyphs are on screen and out of the accessibility tree: a reader
    // getting both would hear the count twice, once unpronounceably.
    const glyphs = screen.getByText('↑2 ↓3');
    expect(glyphs.getAttribute('aria-hidden')).toBe('true');
    // The tooltip is the sentence too, for a pointer.
    expect(glyphs.closest('span[title]')?.getAttribute('title')).toBe(
      '2 to push, 3 to pull',
    );
  });

  it('still announces it when there is nothing to draw', () => {
    // A branch level with its upstream: two zeros say only that they are
    // zero, so nothing is drawn -- and a reader, who has no glance to spend,
    // still gets the answer.
    render(
      <RefRow
        name="level"
        action={{ label: 'Switch to level', onSelect: vi.fn() }}
        meta={null}
        metaLabel="0 to push, 0 to pull"
        actions={[]}
      />,
    );

    expect(
      screen.getByRole('button', { name: 'Switch to level' }),
    ).toHaveAccessibleDescription('0 to push, 0 to pull');
  });

  it('says a plain meta once, with no hidden copy of it', () => {
    render(
      <RefRow
        name="before the refactor"
        action={null}
        meta="main, 2 hours ago"
        actions={[]}
      />,
    );

    const meta = screen.getByText('main, 2 hours ago');
    expect(meta.getAttribute('aria-hidden')).toBeNull();
    expect(meta.getAttribute('title')).toBe('main, 2 hours ago');
  });
});
