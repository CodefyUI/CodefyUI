import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { cleanup, render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ChangeGroup } from './ChangeGroup';
import { useI18n } from '../../i18n';
import { _resetGitStoreForTesting, useGitStore } from '../../store/gitStore';
import { useUIStore } from '../../store/uiStore';
import { confirm } from '../../utils/dialog';
import * as gitApi from '../../api/git';
import type { FileKind, GitFile, GitStatus } from '../../api/git';

vi.mock('../../utils/dialog', () => ({
  confirm: vi.fn(async () => true),
  prompt: vi.fn(async () => null),
}));

function file(path: string, kind: FileKind = 'modified'): GitFile {
  return { path, orig_path: null, kind, xy: '.M', score: null };
}

function status(over: Partial<GitStatus> = {}): GitStatus {
  return {
    branch: 'main', detached: false, head: 'abc1234', unborn: false, upstream: null,
    ahead: null, behind: null, upstream_gone: false, staged: [], unstaged: [],
    untracked: [], conflicted: [], stash_count: 0, merge_in_progress: false,
    rebase_in_progress: false, ...over,
  };
}

const logic = (value: string, extra = false) => ({
  nodes: [
    { id: 'start', type: 'Start', data: { params: {} } },
    { id: 'ds', type: 'Dataset', data: { params: { name: value } } },
    ...(extra ? [{ id: 'new', type: 'Print', data: { params: {} } }] : []),
  ],
  edges: [],
});

type GitActions = ReturnType<typeof useGitStore.getState>;
let stage: ReturnType<typeof vi.fn<GitActions['stage']>>;
let discard: ReturnType<typeof vi.fn<GitActions['discard']>>;

const files = [file('graphs/train/nodes/ds.json'), file('graphs/train/nodes/new.json', 'untracked')];

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  _resetGitStoreForTesting();
  stage = vi.fn(async () => true);
  discard = vi.fn(async () => true);
  useGitStore.setState({
    repoState: 'ready',
    status: status({
      unstaged: [files[0], file('layout/train.layout.json')],
      untracked: [files[1]],
    }),
    stage,
    discard,
  });
  useUIStore.setState({ gitDiff: null, openGitDiff: vi.fn() });
  vi.spyOn(gitApi, 'getGitGraph').mockImplementation(async (_name, ref) => ({
    name: 'train', ref, storage: 'dir', repairs: [],
    logic: ref === 'index' ? logic('MNIST') : logic('CIFAR', true),
  }));
});

afterEach(() => {
  cleanup();
  _resetGitStoreForTesting();
  vi.restoreAllMocks();
});

describe('GraphFolderRow', () => {
  it('shows a folder graph as one row with what changed in the graph', async () => {
    render(<ChangeGroup kind="changes" files={[...files, file('README.md')]} />);
    expect(screen.getByText('train/')).toBeTruthy();
    expect(screen.getByText('2 file(s)')).toBeTruthy();
    // The two files are inside the closed row, not rows of their own.
    expect(screen.queryByRole('button', { name: /Stage ds\.json/ })).toBeNull();
    await waitFor(() => expect(gitApi.getGitGraph).toHaveBeenCalledWith('train', 'worktree'));
    expect(gitApi.getGitGraph).toHaveBeenCalledWith('train', 'index');
    expect(await screen.findByText(/1 node.* added/i)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Show the files of train' }));
    expect(screen.getByRole('button', { name: 'Stage ds.json' })).toBeTruthy();
  });

  it('stages and discards the whole folder with its layout file', async () => {
    render(<ChangeGroup kind="changes" files={files} />);
    fireEvent.click(screen.getByRole('button', { name: 'Stage train/' }));
    await waitFor(() => expect(stage).toHaveBeenCalled());
    expect(stage.mock.calls[0][0]).toEqual([
      'graphs/train/nodes/ds.json', 'graphs/train/nodes/new.json', 'layout/train.layout.json',
    ]);
    fireEvent.click(screen.getByRole('button', { name: 'Discard Changes train/' }));
    await waitFor(() => expect(discard).toHaveBeenCalled());
    expect(vi.mocked(confirm).mock.lastCall?.[0].title)
      .toBe('Discard every change to the graph train (3 files)?');
    expect(discard.mock.calls[0][0]).toHaveLength(3);
  });
});
