import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { CommitBox } from './CommitBox';
import { useI18n } from '../../i18n';
import { MOD_LABEL } from '../../utils/platform';
import { _resetGitStoreForTesting, useGitStore } from '../../store/gitStore';
import type { GitFile, GitStatus } from '../../api/git';

/*
 * The message box and the split button beside it. What is pinned here is the
 * two rules that decide whether Commit can be pressed at all, the chord that
 * bypasses the button, and the two entries behind the chevron -- including the
 * one git would refuse before it ran.
 */

function file(path: string): GitFile {
  return { path, orig_path: null, kind: 'modified', xy: 'M.', score: null };
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

let commit: ReturnType<typeof vi.fn<GitActions['commit']>>;
let setAmend: ReturnType<typeof vi.fn<GitActions['setAmend']>>;
let setCommitMessage: ReturnType<typeof vi.fn<GitActions['setCommitMessage']>>;
let announce: ReturnType<typeof vi.fn<GitActions['announce']>>;

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  _resetGitStoreForTesting();
  commit = vi.fn(async () => true);
  setAmend = vi.fn();
  setCommitMessage = vi.fn();
  announce = vi.fn();
  useGitStore.setState({
    repoState: 'ready',
    status: status({ staged: [file('a.py')] }),
    commitMessage: 'a message',
    commit,
    setAmend,
    setCommitMessage,
    announce,
  });
});

afterEach(() => {
  _resetGitStoreForTesting();
  vi.restoreAllMocks();
});

const commitButton = () => screen.getByRole('button', { name: 'Commit' });
const box = () =>
  screen.getByPlaceholderText(`Message (${MOD_LABEL}+Enter to commit)`);

function openOptions() {
  fireEvent.click(screen.getByRole('button', { name: 'Commit options' }));
  return screen.getByRole('menu', { name: 'Commit options' });
}

describe('CommitBox: when Commit can be pressed', () => {
  it('commits a message against a filled index', () => {
    render(<CommitBox />);
    expect(commitButton()).not.toBeDisabled();
    fireEvent.click(commitButton());
    expect(commit).toHaveBeenCalledWith({ all: false });
  });

  it('refuses an empty message and says which half is missing', () => {
    useGitStore.setState({ commitMessage: '   ' });
    render(<CommitBox />);
    // `aria-disabled`, not `disabled`: a disabled button opens no tooltip in
    // Chrome, so the button with a reason to give could not give it.
    expect(commitButton()).toHaveAttribute('aria-disabled', 'true');
    expect(commitButton()).not.toBeDisabled();
    expect(commitButton().getAttribute('title')).toBe('Enter a message');
  });

  it('refuses an empty index and says so', () => {
    useGitStore.setState({ status: status({ unstaged: [file('a.py')] }) });
    render(<CommitBox />);
    expect(commitButton()).toHaveAttribute('aria-disabled', 'true');
    expect(commitButton().getAttribute('title')).toBe('Nothing staged');
  });

  it('does nothing when the refused button is pressed anyway', () => {
    useGitStore.setState({ commitMessage: '' });
    render(<CommitBox />);
    fireEvent.click(commitButton());
    expect(commit).not.toHaveBeenCalled();
  });

  it('lets an amend through with nothing staged: it is rewriting a message', () => {
    useGitStore.setState({ amend: true, status: status() });
    render(<CommitBox />);
    expect(commitButton()).toHaveAttribute('aria-disabled', 'false');
    expect(commitButton().getAttribute('title')).toBeNull();
    fireEvent.click(commitButton());
    expect(commit).toHaveBeenCalledWith({ all: false });
  });

  it('carries no reason once there is nothing wrong', () => {
    render(<CommitBox />);
    expect(commitButton()).toHaveAttribute('aria-disabled', 'false');
    expect(commitButton().getAttribute('title')).toBeNull();
  });
});

describe('CommitBox: the message', () => {
  it('writes every keystroke to the store', () => {
    render(<CommitBox />);
    fireEvent.change(box(), { target: { value: 'fix the thing' } });
    expect(setCommitMessage).toHaveBeenCalledWith('fix the thing');
  });

  it('commits on the modifier chord, either modifier', () => {
    render(<CommitBox />);
    fireEvent.keyDown(box(), { key: 'Enter', ctrlKey: true });
    fireEvent.keyDown(box(), { key: 'Enter', metaKey: true });
    expect(commit).toHaveBeenCalledTimes(2);
    expect(commit).toHaveBeenLastCalledWith({ all: false });
  });

  it('leaves a plain Enter to the textarea', () => {
    render(<CommitBox />);
    fireEvent.keyDown(box(), { key: 'Enter' });
    expect(commit).not.toHaveBeenCalled();
  });

  it('will not commit through the chord what the button refuses', () => {
    useGitStore.setState({ commitMessage: '' });
    render(<CommitBox />);
    fireEvent.keyDown(box(), { key: 'Enter', ctrlKey: true });
    expect(commit).not.toHaveBeenCalled();
  });

  it('says why the chord did nothing, in the words the button uses', () => {
    // The tooltip is on a button the keyboard never went near, so without
    // this the refusal is a keystroke that does nothing and says nothing.
    useGitStore.setState({ commitMessage: '' });
    render(<CommitBox />);
    fireEvent.keyDown(box(), { key: 'Enter', ctrlKey: true });
    expect(announce).toHaveBeenCalledWith('Enter a message');
  });

  it('names the other reason when the index is what is empty', () => {
    useGitStore.setState({ status: status({ unstaged: [file('a.py')] }) });
    render(<CommitBox />);
    fireEvent.keyDown(box(), { key: 'Enter', ctrlKey: true });
    expect(announce).toHaveBeenCalledWith('Nothing staged');
  });

  it('says nothing when the chord actually commits', () => {
    render(<CommitBox />);
    fireEvent.keyDown(box(), { key: 'Enter', ctrlKey: true });
    expect(commit).toHaveBeenCalledTimes(1);
    expect(announce).not.toHaveBeenCalled();
  });
});

describe('CommitBox: the options menu', () => {
  it('commits everything, staged or not', () => {
    useGitStore.setState({ status: status({ unstaged: [file('a.py')] }) });
    render(<CommitBox />);
    // The button is off (nothing staged) and Commit All is still on: staging
    // is what it does.
    expect(commitButton()).toHaveAttribute('aria-disabled', 'true');
    openOptions();
    fireEvent.click(
      screen.getByRole('menuitem', {
        name: 'Commit All (stages every change, including new files)',
      }),
    );
    expect(commit).toHaveBeenCalledWith({ all: true });
  });

  it('refuses Commit All without a message', () => {
    useGitStore.setState({ commitMessage: '' });
    render(<CommitBox />);
    openOptions();
    expect(
      screen.getByRole('menuitem', {
        name: 'Commit All (stages every change, including new files)',
      }),
    ).toBeDisabled();
  });

  it('turns amend on through the store, as a checkbox', () => {
    render(<CommitBox />);
    openOptions();
    const row = screen.getByRole('menuitemcheckbox', { name: 'Amend Last Commit' });
    expect(row.getAttribute('aria-checked')).toBe('false');
    fireEvent.click(row);
    expect(setAmend).toHaveBeenCalledWith(true);
  });

  it('says the last commit is already pushed instead of just greying out', () => {
    useGitStore.setState({
      status: status({ staged: [file('a.py')], upstream: 'origin/main', ahead: 0, behind: 0 }),
    });
    render(<CommitBox />);
    openOptions();
    const row = screen.getByRole('menuitemcheckbox', {
      name: 'Cannot amend: the last commit is already pushed',
    });
    expect(row).toBeDisabled();
    expect(screen.queryByRole('menuitemcheckbox', { name: 'Amend Last Commit' })).toBeNull();
  });

  it('cannot amend a branch that has no commit yet', () => {
    useGitStore.setState({
      status: status({ staged: [file('a.py')], unborn: true, head: null }),
    });
    render(<CommitBox />);
    openOptions();
    expect(
      screen.getByRole('menuitemcheckbox', { name: 'Amend Last Commit' }),
    ).toBeDisabled();
  });

  it('amends freely while the branch is ahead of its upstream', () => {
    useGitStore.setState({
      status: status({ staged: [file('a.py')], upstream: 'origin/main', ahead: 1, behind: 0 }),
    });
    render(<CommitBox />);
    openOptions();
    expect(
      screen.getByRole('menuitemcheckbox', { name: 'Amend Last Commit' }),
    ).not.toBeDisabled();
  });
});

describe('CommitBox: the amending chip', () => {
  it('is absent while a normal commit is being written', () => {
    render(<CommitBox />);
    expect(screen.queryByText('Amending')).toBeNull();
  });

  it('appears the moment amend is on', () => {
    useGitStore.setState({ amend: true });
    render(<CommitBox />);
    expect(screen.getByText('Amending')).toBeTruthy();
  });
});
