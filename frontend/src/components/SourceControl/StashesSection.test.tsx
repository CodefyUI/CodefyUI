import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { StashesSection } from './StashesSection';
import { useI18n } from '../../i18n';
import { _resetGitStoreForTesting, useGitStore } from '../../store/gitStore';
import { confirm } from '../../utils/dialog';
import type { GitStatus, StashInfo } from '../../api/git';
import styles from './SourceControl.module.css';

/*
 * The stash list. Two things here are not taste: the index is git's own and
 * never the array position (dropping stash@{0} renumbers every stash below
 * it), and the message is shown VERBATIM -- it is the one string in this panel
 * a user wrote themselves, and a list that tidied it up would be a list they
 * could not search.
 */
vi.mock('../../utils/dialog', () => ({
  confirm: vi.fn(async () => true),
  prompt: vi.fn(async () => null),
}));

const askedConfirm = vi.mocked(confirm);

/** Fixed so a relative date is a fixed string; `created_at` is epoch seconds. */
const NOW = 1_700_000_000_000;
const secondsAgo = (n: number) => Math.floor(NOW / 1000) - n;

function stash(over: Partial<StashInfo> = {}): StashInfo {
  return {
    index: 0,
    message: 'before the refactor',
    branch: 'main',
    created_at: secondsAgo(2 * 60 * 60),
    ...over,
  };
}

function status(over: Partial<GitStatus> = {}): GitStatus {
  return {
    branch: 'main',
    detached: false,
    head: 'abc1234',
    unborn: false,
    upstream: null,
    ahead: null,
    behind: null,
    upstream_gone: false,
    staged: [],
    unstaged: [],
    untracked: [],
    conflicted: [],
    stash_count: 0,
    merge_in_progress: false,
    rebase_in_progress: false,
    ...over,
  };
}

/** The store's own action types, so a fake cannot drift from the real one. */
type GitActions = ReturnType<typeof useGitStore.getState>;

let stashPop: ReturnType<typeof vi.fn<GitActions['stashPop']>>;
let stashApply: ReturnType<typeof vi.fn<GitActions['stashApply']>>;
let stashDrop: ReturnType<typeof vi.fn<GitActions['stashDrop']>>;
let setSectionOpen: ReturnType<typeof vi.fn<GitActions['setSectionOpen']>>;
let refreshRefs: ReturnType<typeof vi.fn<GitActions['refreshRefs']>>;

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  vi.setSystemTime(NOW);
  useI18n.setState({ locale: 'en' });
  _resetGitStoreForTesting();
  askedConfirm.mockReset();
  askedConfirm.mockResolvedValue(true);
  stashPop = vi.fn(async () => true);
  stashApply = vi.fn(async () => true);
  stashDrop = vi.fn(async () => true);
  setSectionOpen = vi.fn();
  refreshRefs = vi.fn(async () => {});
  useGitStore.setState({
    repoState: 'ready',
    // The two agree, which is the ordinary state: the poll and the list read
    // are answers about the same stack, and a section that disagrees with the
    // status re-reads itself (see the case below).
    status: status({ stash_count: 1 }),
    stashes: [stash()],
    sections: { branches: false, remotes: false, stashes: true },
    stashPop,
    stashApply,
    stashDrop,
    setSectionOpen,
    refreshRefs,
  });
});

afterEach(() => {
  _resetGitStoreForTesting();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

const section = () => screen.getByRole('region', { name: 'Stashes' });

/** The row a stash whose message is *message* is drawn in. */
function rowFor(message: string): HTMLElement {
  const row = within(section()).getByText(message).closest('li');
  if (row === null) throw new Error(`no stash row for ${message}`);
  return row;
}

describe('StashesSection: the rows', () => {
  it('shows git\'s own selector, the message as written, the branch and when', () => {
    render(<StashesSection />);
    expect(within(section()).getByText('stash@{0}')).toBeTruthy();
    expect(within(section()).getByText('before the refactor')).toBeTruthy();
    expect(within(section()).getByText('main, 2 hours ago')).toBeTruthy();
  });

  it('keeps a message exactly as it was typed', () => {
    // The one string in this panel the user wrote. Trimming, capitalising or
    // truncating it would make the list unsearchable by the words in it.
    useGitStore.setState({ stashes: [stash({ message: '  WIP: fix  the  thing  ' })] });
    render(<StashesSection />);
    // An identity normalizer, because the default one collapses the very
    // whitespace this case is about.
    expect(
      within(section()).getByText('  WIP: fix  the  thing  ', {
        normalizer: (value) => value,
      }),
    ).toBeTruthy();
  });

  it('numbers rows by git\'s index, not by where they sit in the array', () => {
    // `stash@{N}` is what the next command line will say, and dropping one
    // renumbers every stash below it.
    useGitStore.setState({
      stashes: [stash({ index: 3, message: 'newest' }), stash({ index: 7, message: 'older' })],
    });
    render(<StashesSection />);
    expect(within(section()).getByText('stash@{3}')).toBeTruthy();
    expect(within(section()).getByText('stash@{7}')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Drop stash@{7}' })).toBeTruthy();
  });

  it('says nothing about a branch a stash does not name', () => {
    useGitStore.setState({ stashes: [stash({ branch: null })] });
    render(<StashesSection />);
    expect(within(section()).getByText('2 hours ago')).toBeTruthy();
  });

  it('counts from the status until the list has been read', () => {
    useGitStore.setState({ stashes: null, status: status({ stash_count: 4 }) });
    render(<StashesSection />);
    expect(within(section()).getByText('4')).toBeTruthy();
    // Not read yet is not "none": the sentence waits for an answer.
    expect(screen.queryByText('No stashes')).toBeNull();
    // Nothing to disagree with, so nothing is re-read.
    expect(refreshRefs).not.toHaveBeenCalled();
  });

  it('counts from the status once the list has been read too', () => {
    // The status is what the fifteen-second poll refreshes; the list is read
    // when the section opens and after a stash write. A `git stash push` at
    // the command line moves the first and not the second, and the count
    // stayed 0 beside a status that said 1 for as long as the tab was open.
    useGitStore.setState({ stashes: [], status: status({ stash_count: 2 }) });
    render(<StashesSection />);
    expect(within(section()).getByText('2')).toBeTruthy();
  });

  it('re-reads the list when the status disagrees with it, and only once', async () => {
    useGitStore.setState({ stashes: [], status: status({ stash_count: 2 }) });
    render(<StashesSection />);
    await waitFor(() => expect(refreshRefs).toHaveBeenCalledWith('stashes'));
    // The read is out; a re-render while it is in flight must not send it
    // again, or a server that keeps answering the old list would be asked
    // once per render, forever.
    useGitStore.setState({ status: status({ stash_count: 2 }) });
    expect(refreshRefs).toHaveBeenCalledTimes(1);
  });

  it('sends one read while one is already out, and asks again after it lands', async () => {
    // A stash write moves the two halves at different moments: the status the
    // write answers with says 1 straight away, and the list is a second read.
    // Every render in that window disagrees, and `askedFor` only covers the
    // ones where the COUNT is unchanged -- a second stash pushed at the
    // command line moves it again and opened a second concurrent GET.
    let land: () => void = () => {};
    const landed = new Promise<void>((resolve) => {
      land = resolve;
    });
    refreshRefs.mockImplementation(async () => {
      await landed;
      // The answer to the FIRST question, which the second has already
      // outrun -- the same shape as a slow read the store completes.
      useGitStore.setState({ stashes: [stash()] });
    });

    useGitStore.setState({ stashes: [], status: status({ stash_count: 1 }) });
    render(<StashesSection />);
    await waitFor(() => expect(refreshRefs).toHaveBeenCalledTimes(1));

    useGitStore.setState({ status: status({ stash_count: 2 }) });
    // The count on the heading is the proof the render and its effect ran.
    await waitFor(() => expect(within(section()).getByText('2')).toBeTruthy());
    expect(refreshRefs).toHaveBeenCalledTimes(1);

    // ...and the guard does not swallow the question. A skipped ask leaves
    // `askedFor` alone, so the next answer that still disagrees asks it again
    // -- the answer AFTER the one being waited on, because the store's own
    // `set` reaches React before the promise's `finally` does. For an open
    // section that is at most one poll away.
    land();
    await waitFor(() => expect(within(section()).getAllByRole('listitem')).toHaveLength(1));
    useGitStore.setState({ stashes: [stash()] });
    await waitFor(() => expect(refreshRefs).toHaveBeenCalledTimes(2));
  });

  it('reads nothing while the section is collapsed', async () => {
    // A collapsed section draws no rows, and the count beside its heading is
    // the STATUS's own -- so a list that disagrees with it is not on screen
    // to be wrong. Opening reads the list anyway (`setSectionOpen`), which is
    // the read that settles it.
    useGitStore.setState({
      sections: { branches: false, remotes: false, stashes: false },
      stashes: [],
      status: status({ stash_count: 2 }),
    });
    render(<StashesSection />);

    await waitFor(() => expect(within(section()).getByText('2')).toBeTruthy());
    expect(refreshRefs).not.toHaveBeenCalled();
  });

  it('stops re-reading once the two agree again', async () => {
    useGitStore.setState({ stashes: [], status: status({ stash_count: 1 }) });
    render(<StashesSection />);
    await waitFor(() => expect(refreshRefs).toHaveBeenCalledTimes(1));

    useGitStore.setState({ stashes: [stash()] });
    await waitFor(() => expect(within(section()).getByText('1')).toBeTruthy());
    expect(refreshRefs).toHaveBeenCalledTimes(1);
  });

  it('says a repository has no stashes once the list has been read', () => {
    useGitStore.setState({ stashes: [], status: status({ stash_count: 0 }) });
    render(<StashesSection />);
    expect(within(section()).getByText('No stashes')).toBeTruthy();
  });

  it('reports a list that could not be read, inside the section', () => {
    useGitStore.setState({
      stashes: null,
      refsError: { branches: null, remotes: null, stashes: 'Failed to fetch' },
    });
    render(<StashesSection />);
    expect(
      within(section()).getByText('Could not read Stashes: Failed to fetch'),
    ).toBeTruthy();
  });
});

describe('StashesSection: what a narrow row drops', () => {
  /*
   * Three things want one row: a message somebody wrote, the branch and date
   * it was written on, and git's own selector. At 180px there is room for two
   * -- the meta takes its 60% and the chip cannot shrink, so the message was
   * drawn as "ex..." beside "authte..." and the row could not be told from the
   * one under it. Below the same 380px the row's verbs collapse at, the meta
   * comes out of the row and the message takes the width.
   *
   * jsdom applies no stylesheet, so what can be pinned here is the class the
   * container query flips and what stays reachable while it is flipped. The
   * widths themselves are a browser pass at 180px and 520px.
   */
  it('hangs the meta off the class the narrow layout takes out of the row', () => {
    render(<StashesSection />);
    const meta = within(section()).getByText('main, 2 hours ago');
    expect(meta.className.split(' ')).toContain(styles.metaOptional);
    // Still the firm half where it IS drawn: at 520px this is a stash row's
    // proportion and nothing about it changes.
    expect(meta.className.split(' ')).toContain(styles.metaFirm);
  });

  it('keeps the branch and the date in the row, off screen rather than gone', () => {
    // Clipped, not `display: none`: a reader hears the row's text and never
    // had the glance the meta was dropped to save, so dropping it from the
    // accessibility tree would cost them the one thing it says.
    render(<StashesSection />);
    const meta = within(section()).getByText('main, 2 hours ago');
    expect(meta.getAttribute('aria-hidden')).toBeNull();
    expect(rowFor('before the refactor').textContent).toContain('main, 2 hours ago');
  });

  it('answers a pointer with both halves, since the clipped one cannot be hovered', () => {
    // The row's own tooltip, because below the threshold the meta is a 1px box
    // nothing can be over. The message is first and whole: it is what the row
    // ellipsised, and what the reader is hovering to finish reading.
    render(<StashesSection />);
    expect(rowFor('before the refactor').getAttribute('title')).toBe(
      'before the refactor\nmain, 2 hours ago',
    );
  });

  it('says only what a stash naming no branch has to say', () => {
    useGitStore.setState({ stashes: [stash({ branch: null })] });
    render(<StashesSection />);
    expect(rowFor('before the refactor').getAttribute('title')).toBe(
      'before the refactor\n2 hours ago',
    );
  });
});

describe('StashesSection: the three actions', () => {
  it('pops and applies by git\'s index, with nothing to answer first', async () => {
    // Neither throws work away: both put the stash back into the tree, and
    // the store raises the reload offer for any graph they changed.
    render(<StashesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'Pop stash@{0}' }));
    await waitFor(() => expect(stashPop).toHaveBeenCalledWith(0));

    fireEvent.click(screen.getByRole('button', { name: 'Apply stash@{0}' }));
    await waitFor(() => expect(stashApply).toHaveBeenCalledWith(0));
    expect(askedConfirm).not.toHaveBeenCalled();
  });

  it('asks before dropping, which is the one that throws work away', async () => {
    render(<StashesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'Drop stash@{0}' }));
    await waitFor(() => expect(stashDrop).toHaveBeenCalledWith(0));
    expect(askedConfirm).toHaveBeenCalledWith({
      title: 'Drop stash stash@{0}?',
      confirmText: 'Drop',
      variant: 'danger',
    });
  });

  it('drops nothing when the question is answered no', async () => {
    askedConfirm.mockResolvedValue(false);
    render(<StashesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'Drop stash@{0}' }));
    await waitFor(() => expect(askedConfirm).toHaveBeenCalled());
    expect(stashDrop).not.toHaveBeenCalled();
  });

  it('puts focus back on the section heading when the row is gone', async () => {
    render(<StashesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'Pop stash@{0}' }));
    await waitFor(() =>
      expect(document.activeElement).toBe(
        screen.getByRole('button', { name: 'Stashes' }),
      ),
    );
  });

  it('repeats all three inside one compact menu, for a 180px panel', async () => {
    // Three verbs beside a chip that cannot shrink is more row than a 180px
    // panel HAS, so each row carries both shapes and a container query picks
    // one. Both are in the DOM here, because jsdom applies no CSS at all.
    render(<StashesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'More actions stash@{0}' }));
    const menu = screen.getByRole('menu', { name: 'More actions stash@{0}' });
    for (const verb of ['Pop', 'Apply', 'Drop']) {
      expect(within(menu).getByRole('menuitem', { name: verb })).toBeTruthy();
    }
    fireEvent.click(within(menu).getByRole('menuitem', { name: 'Apply' }));
    await waitFor(() => expect(stashApply).toHaveBeenCalledWith(0));
  });

  it('leaves focus where it was after an apply, which keeps the row', async () => {
    render(<StashesSection />);
    const apply = screen.getByRole('button', { name: 'Apply stash@{0}' });
    apply.focus();
    fireEvent.click(apply);
    await waitFor(() => expect(stashApply).toHaveBeenCalled());
    expect(document.activeElement).toBe(apply);
  });
});
