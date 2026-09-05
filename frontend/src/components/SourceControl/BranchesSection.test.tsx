import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { BranchesSection } from './BranchesSection';
import { useI18n } from '../../i18n';
import { _resetGitStoreForTesting, useGitStore } from '../../store/gitStore';
import { confirm, prompt } from '../../utils/dialog';
import type { BranchInfo, BranchesResponse, RemoteBranchInfo } from '../../api/git';

/*
 * The branch list: what each row says about a branch, and the four things that
 * can be done to one. Two of those four are destructive in a way the panel
 * cannot undo -- a delete, and the force-delete git asks about a second time --
 * so what is pinned here is the question each one asks and what it does with
 * the answer, not just that the store was called.
 */
vi.mock('../../utils/dialog', () => ({
  confirm: vi.fn(async () => true),
  prompt: vi.fn(async () => null),
}));

const askedConfirm = vi.mocked(confirm);
const askedPrompt = vi.mocked(prompt);

function branch(name: string, over: Partial<BranchInfo> = {}): BranchInfo {
  return {
    name,
    sha: 'abc1234',
    current: false,
    upstream: null,
    ahead: null,
    behind: null,
    gone: false,
    subject: 'a commit',
    committed_at: 0,
    ...over,
  };
}

/**
 * One remote-tracking branch, in the shape the SERVER sends.
 *
 * The two halves are separate fields and neither repeats the other:
 * `refs.list_branches` splits `origin/feat/deep` into `remote: 'origin'` and
 * `name: 'feat/deep'` (pinned in `backend/tests/test_git_refs.py`). A fixture
 * that put the whole ref in `name` is a shape no server produces, and it made
 * the row's checkout look right while it sent half a ref.
 */
function remoteBranch(ref: string): RemoteBranchInfo {
  const slash = ref.indexOf('/');
  return {
    name: ref.slice(slash + 1),
    remote: ref.slice(0, slash),
    sha: 'abc1234',
    subject: 'a commit',
    committed_at: 0,
  };
}

function branches(over: Partial<BranchesResponse> = {}): BranchesResponse {
  return {
    current: 'main',
    detached: false,
    local: [branch('main', { current: true }), branch('work')],
    remote: [],
    ...over,
  };
}

/** The store's own action types, so a fake cannot drift from the real one. */
type GitActions = ReturnType<typeof useGitStore.getState>;

let createBranch: ReturnType<typeof vi.fn<GitActions['createBranch']>>;
let checkout: ReturnType<typeof vi.fn<GitActions['checkout']>>;
let renameBranch: ReturnType<typeof vi.fn<GitActions['renameBranch']>>;
let deleteBranch: ReturnType<typeof vi.fn<GitActions['deleteBranch']>>;
let setSectionOpen: ReturnType<typeof vi.fn<GitActions['setSectionOpen']>>;

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  _resetGitStoreForTesting();
  askedConfirm.mockReset();
  askedConfirm.mockResolvedValue(true);
  askedPrompt.mockReset();
  askedPrompt.mockResolvedValue(null);
  createBranch = vi.fn(async () => true);
  checkout = vi.fn(async () => true);
  renameBranch = vi.fn(async () => true);
  deleteBranch = vi.fn(async () => true);
  setSectionOpen = vi.fn();
  useGitStore.setState({
    repoState: 'ready',
    branches: branches(),
    sections: { branches: true, remotes: false, stashes: false, history: false },
    createBranch,
    checkout,
    renameBranch,
    deleteBranch,
    setSectionOpen,
  });
});

afterEach(() => {
  _resetGitStoreForTesting();
  vi.restoreAllMocks();
});

const section = () => screen.getByRole('region', { name: 'Branches' });

describe('BranchesSection: the rows', () => {
  it('is the Branches section, counting the local branches', () => {
    render(<BranchesSection />);
    expect(within(section()).getByText('2')).toBeTruthy();
    expect(within(section()).getAllByRole('listitem')).toHaveLength(2);
  });

  it('counts nothing before the list has been read', () => {
    // The tab opens with all three sections closed and nothing reads the
    // branches until one is expanded, so `0` beside a repository with five
    // branches is what a fresh profile showed -- and it stayed wrong until
    // somebody opened the section it was lying about.
    useGitStore.setState({ branches: null });
    render(<BranchesSection />);
    expect(within(section()).queryByText('0')).toBeNull();
  });

  it('marks the branch that is checked out rather than offering to switch', () => {
    render(<BranchesSection />);
    expect(within(section()).getByText('Current')).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Switch to main' })).toBeNull();
    expect(screen.getByRole('button', { name: 'Switch to work' })).toBeTruthy();
  });

  it('switches to a branch from its own name', async () => {
    render(<BranchesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'Switch to work' }));
    await waitFor(() => expect(checkout).toHaveBeenCalledWith('work', 'local'));
  });

  it('says how far each branch is from its upstream', () => {
    useGitStore.setState({
      branches: branches({
        local: [
          branch('main', { current: true, upstream: 'origin/main', ahead: 2, behind: 3 }),
          branch('gone-one', { upstream: 'origin/gone-one', gone: true }),
          branch('unpublished'),
        ],
      }),
    });
    render(<BranchesSection />);
    // The count is DRAWN as the two numbers -- a row at 250px was spending
    // the branch name's room on the clause -- and said in full off screen.
    expect(within(section()).getByText('↑2 ↓3')).toBeTruthy();
    expect(within(section()).getByText('2 to push, 3 to pull').className)
      .toContain('srOnly');
    // A state rather than a count, so it stays as words on screen.
    const gone = within(section()).getByText('Upstream deleted');
    expect(gone.closest('[aria-hidden="true"]')).toBeNull();
    // A branch with no upstream says nothing: "Not published" beside every
    // local branch of a repository with no remote is noise, not information.
    expect(within(section()).queryByText('Not published')).toBeNull();
  });

  it('draws nothing for a branch level with its upstream, and still says so', () => {
    useGitStore.setState({
      branches: branches({
        local: [branch('level', { upstream: 'origin/level', ahead: 0, behind: 0 })],
      }),
    });
    render(<BranchesSection />);
    expect(within(section()).queryByText(/↑/)).toBeNull();
    expect(
      screen.getByRole('button', { name: 'Switch to level' }),
    ).toHaveAccessibleDescription('0 to push, 0 to pull');
  });

  it('says it to a reader too, as the switch button\'s description', () => {
    // The row's name IS the switch button, and an `aria-label` wins over the
    // text inside it -- so a tracking half folded into the name is announced
    // to nobody. It is the button's DESCRIPTION instead: "Switch to work",
    // then "2 to push, 3 to pull". This is the only thing a branch row says
    // beyond the name, so losing it loses the list's whole second column.
    useGitStore.setState({
      branches: branches({
        local: [
          branch('work', { upstream: 'origin/work', ahead: 2, behind: 3 }),
          branch('gone-one', { upstream: 'origin/gone-one', gone: true }),
        ],
      }),
    });
    render(<BranchesSection />);
    expect(
      screen.getByRole('button', { name: 'Switch to work' }),
    ).toHaveAccessibleDescription('2 to push, 3 to pull');
    expect(
      screen.getByRole('button', { name: 'Switch to gone-one' }),
    ).toHaveAccessibleDescription('Upstream deleted');
  });

  it('reports a list that could not be read, naming the list', () => {
    // Not on the error line: nobody pressed a button for the fifteen-second
    // poll that failed. And not the status poll's sentence either -- the
    // status is fine here and one `git branch` read is not.
    useGitStore.setState({
      refsError: { branches: 'Failed to fetch', remotes: null, stashes: null },
    });
    render(<BranchesSection />);
    expect(
      within(section()).getByText('Could not read Branches: Failed to fetch'),
    ).toBeTruthy();
  });

  it('leaves that sentence to the header when the server itself has stopped', () => {
    // One outage, one sentence. A backend that stops answering fails the
    // status poll and every open section's refs read with the same words, and
    // the header is already drawing it -- four copies of one fact on a 180px
    // panel is not four facts.
    useGitStore.setState({
      loadError: 'Failed to fetch',
      refsError: { branches: 'Failed to fetch', remotes: null, stashes: null },
    });
    render(<BranchesSection />);
    expect(
      within(section()).queryByText('Could not read Branches: Failed to fetch'),
    ).toBeNull();
  });

  it('says it here when the header is drawing a refusal instead', () => {
    // The header shows ONE sentence and the operation's refusal wins it, so
    // a `lastError` means the status failure is not on screen at all -- and
    // suppressing this one leaves an open, empty section with no reason
    // given, which is what the line exists to prevent.
    useGitStore.setState({
      loadError: 'Failed to fetch',
      lastError: {
        code: 'dirty_tree',
        message: 'uncommitted changes',
        hint: null,
        stderr: null,
        op: 'checkout',
      },
      refsError: { branches: 'Failed to fetch', remotes: null, stashes: null },
    });
    render(<BranchesSection />);
    expect(
      within(section()).getByText('Could not read Branches: Failed to fetch'),
    ).toBeTruthy();
  });
});

describe('BranchesSection: creating and renaming', () => {
  it('creates a branch from the section header, and switches to it', async () => {
    askedPrompt.mockResolvedValue('  feature/login  ');
    render(<BranchesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'New Branch...' }));
    await waitFor(() => expect(createBranch).toHaveBeenCalledWith('feature/login'));
    expect(askedPrompt.mock.calls[0][0].title).toBe('Branch name');
  });

  it('refuses a name git would refuse, while the box is still open', async () => {
    askedPrompt.mockResolvedValue(null);
    render(<BranchesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'New Branch...' }));
    await waitFor(() => expect(askedPrompt).toHaveBeenCalled());
    const validate = askedPrompt.mock.calls[0][0].validate;
    expect(validate?.('feature/login')).toBeNull();
    expect(validate?.('  feature/login  ')).toBeNull();
    expect(validate?.('has space')).toBe('Not a valid branch name');
    expect(validate?.('')).toBe('Not a valid branch name');
    expect(createBranch).not.toHaveBeenCalled();
  });

  it('renames a branch, asking with the name it is renaming', async () => {
    askedPrompt.mockResolvedValue('shipped');
    render(<BranchesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'Rename work' }));
    await waitFor(() => expect(renameBranch).toHaveBeenCalledWith('work', 'shipped'));
    expect(askedPrompt.mock.calls[0][0].title).toBe('New name for work');
    // No default value in the box: the old name is in the question, and a
    // prefilled box is one keystroke from renaming a branch to itself.
    expect(askedPrompt.mock.calls[0][0].defaultValue).toBeUndefined();
  });

  it('renames nothing when the name comes back unchanged', async () => {
    askedPrompt.mockResolvedValue('work');
    render(<BranchesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'Rename work' }));
    await waitFor(() => expect(askedPrompt).toHaveBeenCalled());
    expect(renameBranch).not.toHaveBeenCalled();
  });
});

describe('BranchesSection: deleting', () => {
  it('asks before deleting, and deletes without force', async () => {
    render(<BranchesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'Delete work' }));
    await waitFor(() => expect(deleteBranch).toHaveBeenCalledWith('work', false));
    expect(askedConfirm).toHaveBeenCalledWith({
      title: 'Delete branch work?',
      confirmText: 'Delete',
      variant: 'danger',
    });
  });

  it('deletes nothing when the question is answered no', async () => {
    askedConfirm.mockResolvedValue(false);
    render(<BranchesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'Delete work' }));
    await waitFor(() => expect(askedConfirm).toHaveBeenCalled());
    expect(deleteBranch).not.toHaveBeenCalled();
  });

  it('asks a second time when git says the branch is not merged', async () => {
    const dismissError = vi.fn();
    useGitStore.setState({
      dismissError,
      deleteBranch: vi.fn(async (_name: string, force: boolean) => {
        if (force) return true;
        useGitStore.setState({
          lastError: {
            code: 'branch_not_merged',
            message: 'not fully merged',
            hint: null,
            stderr: null,
            op: 'delete_branch',
          },
        });
        return false;
      }),
    });
    render(<BranchesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'Delete work' }));

    await waitFor(() => expect(askedConfirm).toHaveBeenCalledTimes(2));
    expect(askedConfirm.mock.calls[1][0]).toEqual({
      title: 'work has unmerged commits. Delete anyway?',
      confirmText: 'Delete',
      variant: 'danger',
    });
    await waitFor(() =>
      expect(useGitStore.getState().deleteBranch).toHaveBeenCalledWith('work', true),
    );
    // The question IS the answer to that refusal, so the red line git's own
    // words would have left behind is taken down before it is asked.
    expect(dismissError).toHaveBeenCalled();
  });

  it('forces nothing when the second question is answered no', async () => {
    const forced = vi.fn(async (_name: string, force: boolean) => {
      if (!force) {
        useGitStore.setState({
          lastError: {
            code: 'branch_not_merged',
            message: 'not fully merged',
            hint: null,
            stderr: null,
            op: 'delete_branch',
          },
        });
      }
      return false;
    });
    useGitStore.setState({ deleteBranch: forced });
    askedConfirm.mockResolvedValueOnce(true).mockResolvedValueOnce(false);
    render(<BranchesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'Delete work' }));

    await waitFor(() => expect(askedConfirm).toHaveBeenCalledTimes(2));
    expect(forced).toHaveBeenCalledTimes(1);
    expect(forced).toHaveBeenCalledWith('work', false);
  });

  it('never asks twice for a refusal that is not about merging', async () => {
    useGitStore.setState({
      deleteBranch: vi.fn(async () => {
        useGitStore.setState({
          lastError: {
            code: 'busy',
            message: 'busy',
            hint: null,
            stderr: null,
            op: 'delete_branch',
          },
        });
        return false;
      }),
    });
    render(<BranchesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'Delete work' }));
    await waitFor(() => expect(askedConfirm).toHaveBeenCalledTimes(1));
    expect(askedConfirm).toHaveBeenCalledTimes(1);
  });

  it('offers no delete for the branch that is checked out', () => {
    render(<BranchesSection />);
    expect(screen.queryByRole('button', { name: 'Delete main' })).toBeNull();
    expect(screen.getByRole('button', { name: 'Rename main' })).toBeTruthy();
  });

  it('offers the same two inside one compact menu, for a 180px panel', async () => {
    // "Rename / Delete" is more row than a 180px panel has, so each row
    // carries both shapes and a container query picks one. jsdom applies no
    // CSS, so both are in the DOM here.
    render(<BranchesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'More actions work' }));
    const menu = screen.getByRole('menu', { name: 'More actions work' });
    expect(within(menu).getByRole('menuitem', { name: 'Rename' })).toBeTruthy();
    // The destructive one is still marked as such in here, which is where a
    // 180px panel does all of its choosing.
    expect(within(menu).getByRole('menuitem', { name: 'Delete' }).className)
      .toMatch(/danger/);
    expect(within(menu).getByRole('menuitem', { name: 'Rename' }).className)
      .not.toMatch(/danger/);
    fireEvent.click(within(menu).getByRole('menuitem', { name: 'Delete' }));
    await waitFor(() => expect(deleteBranch).toHaveBeenCalledWith('work', false));
  });

  it('gives a row with nothing to offer no menu at all', () => {
    // A remote-tracking branch has one action and it IS the row: pressing the
    // name checks it out. An empty menu trigger beside it would be a control
    // that opens nothing.
    useGitStore.setState({
      branches: branches({ remote: [remoteBranch('origin/wip')] }),
    });
    render(<BranchesSection />);
    expect(screen.queryByRole('button', { name: 'More actions origin/wip' })).toBeNull();
  });

  it('puts focus back on the section heading when the row is gone', async () => {
    render(<BranchesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'Delete work' }));
    await waitFor(() =>
      expect(document.activeElement).toBe(
        screen.getByRole('button', { name: 'Branches' }),
      ),
    );
  });
});

describe('BranchesSection: the remote branches', () => {
  beforeEach(() => {
    useGitStore.setState({
      branches: branches({
        remote: [remoteBranch('origin/main'), remoteBranch('origin/wip')],
      }),
    });
  });

  it('lists them under their own heading, apart from the local ones', () => {
    render(<BranchesSection />);
    const sub = screen.getByRole('list', { name: 'Remote branches' });
    expect(within(sub).getAllByRole('listitem')).toHaveLength(2);
    // The count on the section is the LOCAL branches: those are the ones the
    // rows above act on, and a remote-tracking ref is not a branch you have.
    expect(within(section()).getByText('2')).toBeTruthy();
  });

  it('checks one out by creating a branch that tracks it', async () => {
    render(<BranchesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'Switch to origin/wip' }));
    await waitFor(() => expect(checkout).toHaveBeenCalledWith('origin/wip', 'remote'));
  });

  it('says nothing at all when there are none', () => {
    useGitStore.setState({ branches: branches() });
    render(<BranchesSection />);
    expect(screen.queryByText('Remote branches')).toBeNull();
  });

  it('tells two remotes carrying the same branch name apart', async () => {
    // `main` on two remotes is two rows, and the branch name alone names
    // neither of them: the checkout the server takes is `<remote>/<branch>`,
    // and two rows that read `main` are two rows a reader cannot choose
    // between -- nor React tell apart, when the name is also the key.
    useGitStore.setState({
      branches: branches({
        remote: [remoteBranch('origin/main'), remoteBranch('upstream/main')],
      }),
    });
    render(<BranchesSection />);
    const sub = screen.getByRole('list', { name: 'Remote branches' });
    expect(within(sub).getAllByRole('listitem')).toHaveLength(2);
    fireEvent.click(screen.getByRole('button', { name: 'Switch to upstream/main' }));
    await waitFor(() => expect(checkout).toHaveBeenCalledWith('upstream/main', 'remote'));
  });

  it('switches to a remote branch whose own name has a slash in it', async () => {
    // `origin/feat/x` arrives as `{remote: 'origin', name: 'feat/x'}`, and the
    // server re-splits whatever it is sent on the FIRST slash -- so sending
    // the name half alone asks for a branch `x` on a remote `feat`.
    useGitStore.setState({
      branches: branches({ remote: [remoteBranch('origin/feat/deep')] }),
    });
    render(<BranchesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'Switch to origin/feat/deep' }));
    await waitFor(() =>
      expect(checkout).toHaveBeenCalledWith('origin/feat/deep', 'remote'));
  });
});
