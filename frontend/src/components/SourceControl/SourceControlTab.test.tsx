import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { SourceControlTab } from './SourceControlTab';
import { useI18n } from '../../i18n';
import { _resetGitStoreForTesting, useGitStore } from '../../store/gitStore';
import type { BranchInfo, FileKind, GitFile, GitStatus, RepoInfo } from '../../api/git';

/*
 * The panel itself: which screen a repository state draws, and what opening
 * the tab costs. The sidebar mounts only the tab that is open, so this
 * component's own effect is where the poll, the focus/visibility listeners and
 * the save hook are started and stopped -- which is what the last describe
 * block measures with real spies rather than with a comment.
 *
 * `api/git` is stubbed at the module: the store is the real one here, and a
 * real `getGitStatus` would issue a fetch on mount.
 */
vi.mock('../../api/git', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/git')>();
  return {
    ...actual,
    getGitStatus: vi.fn(),
    // The header asks for the remotes as soon as a repository answers: "Publish
    // or Sync" cannot be decided from a list nobody has fetched.
    getGitBranches: vi.fn(async () => ({
      current: 'main',
      detached: false,
      local: [],
      remote: [],
    })),
    getGitRemotes: vi.fn(async () => []),
    getGitStashes: vi.fn(async () => []),
  };
});

const { getGitStatus } = await import('../../api/git');
const readStatus = vi.mocked(getGitStatus);

// Captured before any case swaps an action out: `_resetGitStoreForTesting`
// restores the store's DATA, not its actions.
const realAttach = useGitStore.getState().attach;
const realDetach = useGitStore.getState().detach;

function file(path: string, kind: FileKind = 'modified'): GitFile {
  return { path, orig_path: null, kind, xy: 'M.', score: null };
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

function branch(name: string): BranchInfo {
  return {
    name,
    sha: 'abc1234',
    current: name === 'main',
    upstream: null,
    ahead: null,
    behind: null,
    gone: false,
    subject: 'a commit',
    committed_at: 0,
  };
}

function repo(over: Partial<RepoInfo> = {}): RepoInfo {
  return {
    state: 'ready',
    project_dir: 'D:/work/demo',
    git_version: '2.45.0',
    nested_toplevel: null,
    ...over,
  };
}

/** The store's own action types, so a fake cannot drift from the real one. */
type GitActions = ReturnType<typeof useGitStore.getState>;

let attach: ReturnType<typeof vi.fn<GitActions['attach']>>;
let detach: ReturnType<typeof vi.fn<GitActions['detach']>>;
let init: ReturnType<typeof vi.fn<GitActions['init']>>;

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  _resetGitStoreForTesting();
  readStatus.mockReset();
  readStatus.mockResolvedValue({ repo: repo({ state: 'no_project' }), status: null });
  attach = vi.fn();
  detach = vi.fn();
  init = vi.fn(async () => true);
  useGitStore.setState({ attach, detach, init });
});

afterEach(() => {
  _resetGitStoreForTesting();
  useGitStore.setState({ attach: realAttach, detach: realDetach });
  vi.restoreAllMocks();
});

describe('SourceControlTab: one screen per repository state', () => {
  it('waits, saying what it is waiting for', () => {
    render(<SourceControlTab />);
    expect(screen.getByText('Running status...')).toBeTruthy();
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('reports a status that could not be read instead of waiting forever', () => {
    useGitStore.setState({ loadError: 'Failed to fetch' });
    render(<SourceControlTab />);
    expect(screen.getByRole('alert').textContent).toContain(
      'Could not read repository status: Failed to fetch',
    );
    expect(screen.queryByText('Running status...')).toBeNull();
  });

  it('keeps waiting when a ready repository has answered no status yet', () => {
    // The route sends the repository and its status together, so this is
    // either a first read still in flight or a server that broke its own
    // contract. Both are a wait; neither is a header above an empty body.
    useGitStore.setState({ repoState: 'ready', repo: repo(), status: null });
    render(<SourceControlTab />);
    expect(screen.getByText('Running status...')).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Commit' })).toBeNull();
  });

  it('asks for a project directory with the two commands that make one', () => {
    useGitStore.setState({ repoState: 'no_project', repo: repo({ state: 'no_project' }) });
    render(<SourceControlTab />);
    expect(screen.getByText('Source control needs a project directory.')).toBeTruthy();
    expect(screen.getByText('cdui project init my-project')).toBeTruthy();
    expect(screen.getByText('cdui start --project my-project')).toBeTruthy();
  });

  it('offers to initialize a project that is not a repository', async () => {
    useGitStore.setState({ repoState: 'not_repo', repo: repo({ state: 'not_repo' }) });
    render(<SourceControlTab />);
    expect(screen.getByText('This project is not a git repository yet.')).toBeTruthy();
    screen.getByRole('button', { name: 'Initialize Repository' }).click();
    await waitFor(() => expect(init).toHaveBeenCalledTimes(1));
  });

  it('warns first when the project sits inside another repository', () => {
    useGitStore.setState({
      repoState: 'not_repo',
      repo: repo({ state: 'not_repo', nested_toplevel: 'D:/work/monorepo' }),
    });
    render(<SourceControlTab />);
    expect(
      screen.getByText(
        'It sits inside another repository (D:/work/monorepo); initializing creates a separate one here.',
      ),
    ).toBeTruthy();
  });

  it('reports a server with no git, and one whose git is too old', () => {
    useGitStore.setState({
      repoState: 'git_missing',
      repo: repo({ state: 'git_missing', git_version: null }),
    });
    const view = render(<SourceControlTab />);
    expect(screen.getByText('git is not installed on the server computer.')).toBeTruthy();
    view.unmount();

    useGitStore.setState({
      repoState: 'git_too_old',
      repo: repo({ state: 'git_too_old', git_version: '2.19.1' }),
    });
    render(<SourceControlTab />);
    expect(
      screen.getByText('git 2.19.1 is too old; 2.23 or newer is required.'),
    ).toBeTruthy();
  });
});

describe('SourceControlTab: the ready panel', () => {
  beforeEach(() => {
    useGitStore.setState({
      repoState: 'ready',
      repo: repo(),
      status: status({
        staged: [file('graphs/a.graph.json')],
        unstaged: [file('src/train.py')],
        untracked: [file('notes.txt', 'untracked')],
      }),
    });
  });

  it('draws the commit box and the two standing groups', () => {
    render(<SourceControlTab />);
    expect(screen.getByRole('button', { name: 'Commit' })).toBeTruthy();
    expect(screen.getByRole('region', { name: 'Staged Changes' })).toBeTruthy();
    expect(screen.getByRole('region', { name: 'Changes' })).toBeTruthy();
    expect(screen.queryByRole('region', { name: 'Merge Changes' })).toBeNull();
  });

  it('puts unstaged and untracked files in one Changes group', () => {
    render(<SourceControlTab />);
    const changes = screen.getByRole('region', { name: 'Changes' });
    expect(changes.textContent).toContain('train.py');
    expect(changes.textContent).toContain('notes.txt');
  });

  it('adds the Merge group while a conflict exists, and says so once', () => {
    useGitStore.setState({
      status: status({
        conflicted: [file('src/train.py', 'conflict')],
        merge_in_progress: true,
      }),
    });
    render(<SourceControlTab />);
    const group = screen.getByRole('region', { name: 'Merge Changes' });
    // The group carries the banner; the tab drew a second copy of that
    // sentence a few pixels above it until this one had a heading of its own.
    expect(
      screen.getAllByText('Merge in progress: resolve each file, then commit.'),
    ).toHaveLength(1);
    expect(
      within(group).getByRole('button', { name: 'Keep mine train.py' }),
    ).toBeTruthy();
    // Never Discard on a conflict: the server refuses the path outright.
    expect(within(group).queryByRole('button', { name: /Discard/ })).toBeNull();
  });

  it('keeps the Merge group on a merge with nothing left to resolve', () => {
    // Settling every file as "mine" changes no file, so the tree is clean and
    // the conflict list is empty -- and Abort Merge is one of the only two
    // ways out of MERGE_HEAD from there.
    useGitStore.setState({ status: status({ merge_in_progress: true }) });
    render(<SourceControlTab />);
    expect(screen.getByText('No changes')).toBeTruthy();
    expect(screen.getByRole('region', { name: 'Merge Changes' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Abort Merge' })).toBeTruthy();
  });

  it('hides layout files from Changes only while the filter is on', () => {
    useGitStore.setState({
      hideLayout: true,
      status: status({
        unstaged: [file('graphs/a.graph.json'), file('layout/a.layout.json')],
      }),
    });
    const view = render(<SourceControlTab />);
    expect(screen.getByText('a.graph.json')).toBeTruthy();
    expect(screen.queryByText('a.layout.json')).toBeNull();

    view.unmount();
    useGitStore.setState({ hideLayout: false });
    render(<SourceControlTab />);
    expect(screen.getByText('a.layout.json')).toBeTruthy();
  });

  it('says so plainly when there is nothing to commit', () => {
    useGitStore.setState({ status: status() });
    render(<SourceControlTab />);
    expect(screen.getByText('No changes')).toBeTruthy();
    expect(screen.queryByRole('region', { name: 'Changes' })).toBeNull();
  });

  it('shows the commit identity form only when the store opens it', () => {
    const view = render(<SourceControlTab />);
    expect(screen.queryByText('Commit identity')).toBeNull();
    view.unmount();

    useGitStore.setState({ identityFormOpen: true });
    render(<SourceControlTab />);
    expect(screen.getByText('Commit identity')).toBeTruthy();
  });
});

describe('SourceControlTab: the three reference sections', () => {
  beforeEach(() => {
    useGitStore.setState({ repoState: 'ready', repo: repo(), status: status() });
  });

  const section = (name: string) => screen.getByRole('region', { name });

  it('draws all three, collapsed, on a repository with nothing to commit', () => {
    // A clean tree is not an empty panel: branches, remotes and stashes are
    // there whether or not anything has been edited.
    render(<SourceControlTab />);
    expect(screen.getByText('No changes')).toBeTruthy();
    for (const name of ['Branches', 'Remotes', 'Stashes']) {
      expect(section(name)).toBeTruthy();
      expect(
        screen.getByRole('button', { name }).getAttribute('aria-expanded'),
      ).toBe('false');
    }
  });

  it('counts what each list holds', () => {
    useGitStore.setState({
      branches: {
        current: 'main',
        detached: false,
        local: [branch('main'), branch('work')],
        remote: [],
      },
      remotes: [{ name: 'origin', fetch_url: 'u', push_url: 'u' }],
      stashes: [],
      status: status({ stash_count: 4 }),
    });
    render(<SourceControlTab />);
    expect(within(section('Branches')).getByText('2')).toBeTruthy();
    expect(within(section('Remotes')).getByText('1')).toBeTruthy();
    // The STATUS wins for the stashes: it is the half the poll keeps fresh,
    // and a `git stash push` at the command line moves it while the list --
    // read when the section opens -- stays where it was. The section re-reads
    // itself on the disagreement; the count is right in the meantime.
    expect(within(section('Stashes')).getByText('4')).toBeTruthy();
  });

  it('counts stashes from the status until the list has been read', () => {
    useGitStore.setState({ stashes: null, status: status({ stash_count: 4 }) });
    render(<SourceControlTab />);
    expect(within(section('Stashes')).getByText('4')).toBeTruthy();
  });

  it('opens one through the store, which is what remembers it', () => {
    const setSectionOpen = vi.fn();
    useGitStore.setState({ setSectionOpen });
    render(<SourceControlTab />);
    fireEvent.click(screen.getByRole('button', { name: 'Remotes' }));
    expect(setSectionOpen).toHaveBeenCalledWith('remotes', true);
  });
});

describe('SourceControlTab: the announcement region', () => {
  const regions = () => screen.getAllByRole('status');
  const said = () => regions().map((region) => region.textContent);

  /** One write that landed, in the order the store writes it. */
  const land = (message: string) => {
    act(() => {
      useGitStore.setState({ busyOp: 'stage', lastError: null });
      useGitStore.setState({ liveMessage: message });
      useGitStore.setState({ busyOp: null });
    });
  };

  /** The same, for the other lane: a network operation holds `netOp`. */
  const landNet = (message: string) => {
    act(() => {
      useGitStore.setState({ netOp: 'fetch', lastError: null });
      useGitStore.setState({ liveMessage: message });
      useGitStore.setState({ netOp: null });
    });
  };

  it('is two hidden polite regions, one of them carrying the store\'s sentence', () => {
    useGitStore.setState({ liveMessage: 'Staged Changes 1, Changes 0' });
    render(<SourceControlTab />);
    expect(regions()).toHaveLength(2);
    for (const region of regions()) {
      expect(region.getAttribute('aria-live')).toBe('polite');
    }
    expect(said()).toEqual(['Staged Changes 1, Changes 0', '']);
  });

  it('is replaced on each finished write, so the same sentence is said twice', () => {
    render(<SourceControlTab />);
    const [first, second] = regions();

    land('Staged Changes 1, Changes 0');
    // The sentence moved to the OTHER region and the first was emptied --
    // and both of them are the nodes that were on the page before the write.
    // A live region inserted with its text already inside it is one assistive
    // tech may never announce.
    expect(said()).toEqual(['', 'Staged Changes 1, Changes 0']);
    expect(regions()[0]).toBe(first);
    expect(regions()[1]).toBe(second);

    // The same words again: an unchanged text node is announced zero times, so
    // they change sides instead.
    land('Staged Changes 1, Changes 0');
    expect(said()).toEqual(['Staged Changes 1, Changes 0', '']);
  });

  it('changes sides for a network operation too, so a second fetch is still said', () => {
    // Fetch, pull, push, sync and publish hold `netOp`, not `busyOp`: a guard
    // that watched the local lane alone announced the first fetch (the text
    // changed) and every one after it not at all.
    render(<SourceControlTab />);
    landNet('Fetched');
    expect(said()).toEqual(['', 'Fetched']);
    landNet('Fetched');
    expect(said()).toEqual(['Fetched', '']);
  });

  it('says nothing for the identity write, which moves nothing in the panel', () => {
    render(<SourceControlTab />);
    act(() => {
      useGitStore.setState({ liveMessage: 'Staged Changes 1, Changes 0' });
    });
    expect(said()).toEqual(['Staged Changes 1, Changes 0', '']);
    act(() => {
      useGitStore.setState({ busyOp: 'identity', lastError: null });
      useGitStore.setState({ busyOp: null });
    });
    // A swap here would re-read whatever the last real operation said.
    expect(said()).toEqual(['Staged Changes 1, Changes 0', '']);
  });

  it('says nothing again after a write that failed', () => {
    render(<SourceControlTab />);
    act(() => {
      useGitStore.setState({ liveMessage: 'Staged Changes 1, Changes 0' });
    });
    expect(said()).toEqual(['Staged Changes 1, Changes 0', '']);
    act(() => {
      useGitStore.setState({ busyOp: 'commit', lastError: null });
      useGitStore.setState({
        lastError: { code: 'nothing_to_commit', message: 'no', hint: null, stderr: null },
      });
      useGitStore.setState({ busyOp: null });
    });
    // Nothing moved, so nothing was read out: the refusal is the header's job.
    expect(said()).toEqual(['Staged Changes 1, Changes 0', '']);
  });
});

describe('SourceControlTab: what opening the tab costs', () => {
  it('attaches on mount and detaches on unmount, exactly once each', () => {
    const { unmount } = render(<SourceControlTab />);
    expect(attach).toHaveBeenCalledTimes(1);
    expect(detach).not.toHaveBeenCalled();
    unmount();
    expect(detach).toHaveBeenCalledTimes(1);
  });

  it('starts the poll and the listeners, and takes all three back down', async () => {
    useGitStore.setState({ attach: realAttach, detach: realDetach });
    const setInterval = vi.spyOn(globalThis, 'setInterval');
    const clearInterval = vi.spyOn(globalThis, 'clearInterval');
    const listen = vi.spyOn(document, 'addEventListener');
    const unlisten = vi.spyOn(document, 'removeEventListener');

    let view!: ReturnType<typeof render>;
    await act(async () => {
      view = render(<SourceControlTab />);
    });

    expect(setInterval).toHaveBeenCalledTimes(1);
    expect(listen).toHaveBeenCalledWith('visibilitychange', expect.any(Function));
    expect(readStatus).toHaveBeenCalledTimes(1);

    await act(async () => {
      view.unmount();
    });
    expect(clearInterval).toHaveBeenCalledTimes(1);
    expect(unlisten).toHaveBeenCalledWith('visibilitychange', expect.any(Function));
  });
});
