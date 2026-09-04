import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { MergeGroup } from './MergeGroup';
import { SCM_FOCUS } from './ScmHeader';
import { useI18n } from '../../i18n';
import { _resetGitStoreForTesting, useGitStore } from '../../store/gitStore';
import { confirm } from '../../utils/dialog';
import type { GitFile, GitStatus } from '../../api/git';

/*
 * The conflict list, which is the one group in the tab whose rows are a
 * DECISION rather than a bookkeeping move. Three ways out per file, and each
 * of them overwrites the file -- so what is pinned here is that every button
 * names the file it acts on, that the abort is asked for first, and that
 * Discard, which the server refuses on a conflicted path anyway, is nowhere.
 */
vi.mock('../../utils/dialog', () => ({
  confirm: vi.fn(async () => true),
  prompt: vi.fn(async () => null),
}));

const askedConfirm = vi.mocked(confirm);

function conflict(path: string): GitFile {
  return { path, orig_path: null, kind: 'conflict', xy: 'UU', score: null };
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
    merge_in_progress: true,
    rebase_in_progress: false,
    ...over,
  };
}

/** The store's own action types, so a fake cannot drift from the real one. */
type GitActions = ReturnType<typeof useGitStore.getState>;

let resolve: ReturnType<typeof vi.fn<GitActions['resolve']>>;
let abortMerge: ReturnType<typeof vi.fn<GitActions['abortMerge']>>;

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  _resetGitStoreForTesting();
  askedConfirm.mockReset();
  askedConfirm.mockResolvedValue(true);
  resolve = vi.fn(async () => true);
  abortMerge = vi.fn(async () => true);
  useGitStore.setState({ repoState: 'ready', status: status(), resolve, abortMerge });
});

afterEach(() => {
  _resetGitStoreForTesting();
  vi.restoreAllMocks();
});

/** The compact form's rows, which live behind one trigger per file. */
function openRowMenu(name: string) {
  fireEvent.click(screen.getByRole('button', { name: `More actions ${name}` }));
  return screen.getByRole('menu', { name: `More actions ${name}` });
}

describe('MergeGroup: the group', () => {
  it('is a named region holding one row per conflict', () => {
    render(<MergeGroup files={[conflict('src/train.py'), conflict('src/model.py')]} />);
    const group = screen.getByRole('region', { name: 'Merge Changes' });
    expect(within(group).getAllByRole('listitem')).toHaveLength(2);
    expect(within(group).getByText('2')).toBeTruthy();
  });

  it('carries the merge banner itself, so the sentence is said once', () => {
    // The tab used to draw this line above the groups, which put it on screen
    // beside the group it describes.
    render(<MergeGroup files={[conflict('a.py')]} />);
    const group = screen.getByRole('region', { name: 'Merge Changes' });
    expect(
      within(group).getByText('Merge in progress: resolve each file, then commit.'),
    ).toBeTruthy();
  });

  it('keeps the banner while the rows are collapsed away', () => {
    // The banner is the state of the repository, not one of the rows.
    render(<MergeGroup files={[conflict('a.py')]} />);
    fireEvent.click(screen.getByRole('button', { name: 'Merge Changes' }));
    expect(screen.queryAllByRole('listitem')).toHaveLength(0);
    expect(
      screen.getByText('Merge in progress: resolve each file, then commit.'),
    ).toBeTruthy();
  });

  it('is drawn for a merge that has nothing left to resolve', () => {
    // Resolving every file as "mine" empties the list and leaves MERGE_HEAD:
    // the only two ways out of that are the commit box and this Abort.
    render(<MergeGroup files={[]} />);
    expect(screen.getByRole('region', { name: 'Merge Changes' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Abort Merge' })).toBeTruthy();
  });
});

describe('MergeGroup: abort', () => {
  it('asks with the banner sentence before throwing the merge away', async () => {
    render(<MergeGroup files={[conflict('a.py')]} />);
    fireEvent.click(screen.getByRole('button', { name: 'Abort Merge' }));
    await waitFor(() => expect(abortMerge).toHaveBeenCalled());
    expect(askedConfirm).toHaveBeenCalledWith({
      title: 'Merge in progress: resolve each file, then commit.',
      confirmText: 'Abort Merge',
      variant: 'danger',
    });
  });

  it('aborts nothing when the question is answered no', async () => {
    askedConfirm.mockResolvedValue(false);
    render(<MergeGroup files={[conflict('a.py')]} />);
    fireEvent.click(screen.getByRole('button', { name: 'Abort Merge' }));
    await waitFor(() => expect(askedConfirm).toHaveBeenCalled());
    expect(abortMerge).not.toHaveBeenCalled();
  });

  it('offers no abort where there is no merge to abort', () => {
    // A stash pop can leave conflicts with no MERGE_HEAD behind them, and
    // `git merge --abort` has nothing to undo there.
    useGitStore.setState({ status: status({ merge_in_progress: false }) });
    render(<MergeGroup files={[conflict('a.py')]} />);
    expect(screen.queryByRole('button', { name: 'Abort Merge' })).toBeNull();
  });
});

describe('MergeGroup: one conflicted file', () => {
  it('offers the three resolutions, each naming the file it overwrites', () => {
    render(<MergeGroup files={[conflict('src/train.py')]} />);
    for (const verb of ['Keep mine', 'Take incoming', 'Mark resolved']) {
      expect(screen.getByRole('button', { name: `${verb} train.py` })).toBeTruthy();
    }
  });

  it('sends each button to the side it names', async () => {
    render(<MergeGroup files={[conflict('src/train.py')]} />);
    fireEvent.click(screen.getByRole('button', { name: 'Keep mine train.py' }));
    await waitFor(() => expect(resolve).toHaveBeenCalledWith('src/train.py', 'ours'));

    fireEvent.click(screen.getByRole('button', { name: 'Take incoming train.py' }));
    await waitFor(() => expect(resolve).toHaveBeenCalledWith('src/train.py', 'theirs'));

    fireEvent.click(screen.getByRole('button', { name: 'Mark resolved train.py' }));
    await waitFor(() => expect(resolve).toHaveBeenCalledWith('src/train.py', 'mark'));
  });

  it('offers no Discard, which the server refuses on a conflict anyway', () => {
    render(<MergeGroup files={[conflict('src/train.py')]} />);
    expect(screen.queryByRole('button', { name: /Discard/ })).toBeNull();
    expect(screen.queryByRole('button', { name: /Unstage/ })).toBeNull();
  });

  it('names a renamed file by both halves, like every other row', () => {
    render(
      <MergeGroup
        files={[{ ...conflict('src/new.py'), orig_path: 'src/old.py' }]}
      />,
    );
    expect(
      screen.getByRole('button', { name: 'Keep mine old.py -> new.py' }),
    ).toBeTruthy();
  });

  it('repeats the three inside one compact menu, for a 180px panel', () => {
    // The three verbs do not fit in a narrow row in any language, so the row
    // carries both forms and the container query picks one. Both are in the
    // DOM here, because jsdom applies no CSS at all.
    render(<MergeGroup files={[conflict('src/train.py')]} />);
    const menu = openRowMenu('train.py');
    for (const verb of ['Keep mine', 'Take incoming', 'Mark resolved']) {
      expect(within(menu).getByRole('menuitem', { name: verb })).toBeTruthy();
    }
  });

  it('resolves from the compact menu too', async () => {
    render(<MergeGroup files={[conflict('src/train.py')]} />);
    const menu = openRowMenu('train.py');
    fireEvent.click(within(menu).getByRole('menuitem', { name: 'Take incoming' }));
    await waitFor(() => expect(resolve).toHaveBeenCalledWith('src/train.py', 'theirs'));
  });
});

describe('MergeGroup: focus after a row leaves', () => {
  it('moves focus to the heading when a resolution lands', async () => {
    render(<MergeGroup files={[conflict('src/train.py')]} />);
    fireEvent.click(screen.getByRole('button', { name: 'Keep mine train.py' }));
    await waitFor(() =>
      expect(document.activeElement).toBe(
        screen.getByRole('button', { name: 'Merge Changes' }),
      ),
    );
  });

  it('falls back to the message box when the group itself is gone', async () => {
    // The last conflict resolved takes the whole group with it, so there is
    // no heading left to catch the focus the row was holding.
    useGitStore.setState({
      resolve: vi.fn(async () => {
        useGitStore.setState({ status: status({ merge_in_progress: false }) });
        return true;
      }),
    });
    render(<Panel />);
    fireEvent.click(screen.getByRole('button', { name: 'Mark resolved train.py' }));
    await waitFor(() =>
      expect(document.activeElement).toBe(screen.getByLabelText('message')),
    );
  });
});

/** The panel in miniature: the group is drawn only while a merge is running. */
function Panel() {
  const current = useGitStore((s) => s.status);
  return (
    <>
      <textarea data-scm-focus={SCM_FOCUS.commit} aria-label="message" />
      {current?.merge_in_progress === true && (
        <MergeGroup files={[conflict('src/train.py')]} />
      )}
    </>
  );
}
