import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { StashesSection } from './StashesSection';
import { useI18n } from '../../i18n';
import { _resetGitStoreForTesting, useGitStore } from '../../store/gitStore';
import { confirm } from '../../utils/dialog';
import type { GitStatus, StashInfo } from '../../api/git';

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
  useGitStore.setState({
    repoState: 'ready',
    status: status(),
    stashes: [stash()],
    sections: { branches: false, remotes: false, stashes: true },
    stashPop,
    stashApply,
    stashDrop,
    setSectionOpen,
  });
});

afterEach(() => {
  _resetGitStoreForTesting();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

const section = () => screen.getByRole('region', { name: 'Stashes' });

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
      within(section()).getByText('Could not read repository status: Failed to fetch'),
    ).toBeTruthy();
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

  it('leaves focus where it was after an apply, which keeps the row', async () => {
    render(<StashesSection />);
    const apply = screen.getByRole('button', { name: 'Apply stash@{0}' });
    apply.focus();
    fireEvent.click(apply);
    await waitFor(() => expect(stashApply).toHaveBeenCalled());
    expect(document.activeElement).toBe(apply);
  });
});
