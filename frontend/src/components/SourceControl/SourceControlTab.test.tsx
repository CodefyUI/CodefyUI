import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, render, screen, waitFor } from '@testing-library/react';
import { SourceControlTab } from './SourceControlTab';
import { useI18n } from '../../i18n';
import { _resetGitStoreForTesting, useGitStore } from '../../store/gitStore';
import type { FileKind, GitFile, GitStatus, RepoInfo } from '../../api/git';

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
  return { ...actual, getGitStatus: vi.fn() };
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

  it('adds the Merge group only while a conflict exists, and its banner', () => {
    useGitStore.setState({
      status: status({
        conflicted: [file('src/train.py', 'conflict')],
        merge_in_progress: true,
      }),
    });
    render(<SourceControlTab />);
    expect(screen.getByRole('region', { name: 'Merge Changes' })).toBeTruthy();
    expect(
      screen.getByText('Merge in progress: resolve each file, then commit.'),
    ).toBeTruthy();
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

describe('SourceControlTab: the announcement region', () => {
  it('is one hidden polite region carrying the store\'s sentence', () => {
    useGitStore.setState({ liveMessage: 'Staged Changes 1, Changes 0' });
    render(<SourceControlTab />);
    const live = screen.getByRole('status');
    expect(live.getAttribute('aria-live')).toBe('polite');
    expect(live.textContent).toBe('Staged Changes 1, Changes 0');
  });

  it('is replaced on each finished write, so the same sentence is said twice', () => {
    render(<SourceControlTab />);
    const before = screen.getByRole('status');

    const land = (message: string) => {
      act(() => {
        useGitStore.setState({ busyOp: 'stage', lastError: null });
        useGitStore.setState({ liveMessage: message });
        useGitStore.setState({ busyOp: null });
      });
    };

    land('Staged Changes 1, Changes 0');
    const after = screen.getByRole('status');
    expect(after).not.toBe(before);

    // The same words again: an unchanged text node is announced zero times, so
    // the node itself has to be a new one.
    land('Staged Changes 1, Changes 0');
    expect(screen.getByRole('status')).not.toBe(after);
  });

  it('says nothing again after a write that failed', () => {
    render(<SourceControlTab />);
    act(() => {
      useGitStore.setState({ liveMessage: 'Staged Changes 1, Changes 0' });
    });
    const before = screen.getByRole('status');
    act(() => {
      useGitStore.setState({ busyOp: 'commit', lastError: null });
      useGitStore.setState({
        lastError: { code: 'nothing_to_commit', message: 'no', hint: null, stderr: null },
      });
      useGitStore.setState({ busyOp: null });
    });
    expect(screen.getByRole('status')).toBe(before);
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
