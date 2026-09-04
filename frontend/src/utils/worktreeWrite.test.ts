import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

/**
 * The write signal, and the one line in the save path that raises it.
 *
 * Both halves in one file on purpose: the slot on its own is four lines that
 * cannot be wrong, and the thing that CAN be wrong is a save that stops
 * announcing itself -- after which the Source Control tab quietly goes back
 * to showing a graph up to fifteen seconds after it was saved, with every
 * other test still green.
 *
 * `api/rest` is mocked around the real module rather than replaced, so
 * `saveGraph` is a spy and everything else in it stays itself.
 */
vi.mock('../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/rest')>();
  return {
    ...actual,
    saveGraph: vi.fn().mockResolvedValue({}),
    listGraphs: vi.fn().mockResolvedValue([]),
  };
});

vi.mock('./dialog', () => ({
  prompt: vi.fn(),
  confirm: vi.fn().mockResolvedValue(true),
}));

import { announceWorktreeWrite, setWorktreeWriteListener } from './worktreeWrite';
import { saveActiveGraph } from './saveActiveGraph';
import { saveGraph } from '../api/rest';
import { prompt } from './dialog';
import { useProjectStore } from '../store/projectStore';
import { useTabStore } from '../store/tabStore';
import { useToastStore } from '../store/toastStore';

const saveGraphMock = vi.mocked(saveGraph);
const promptMock = vi.mocked(prompt);

let heard = vi.fn(() => undefined);

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  saveGraphMock.mockResolvedValue({});
  useToastStore.setState({ toasts: [] });
  useTabStore.setState({ tabs: [], activeTabId: '' });
  useTabStore.getState().addTab('a tab');
  useProjectStore.setState({ projectDir: '/proj', projectName: 'proj', loaded: true });
  heard = vi.fn(() => undefined);
  setWorktreeWriteListener(heard);
});

afterEach(() => {
  setWorktreeWriteListener(null);
  useProjectStore.setState({ projectDir: null, projectName: null, loaded: false });
});

describe('the write signal', () => {
  it('reaches the registered listener, and nobody once it is cleared', () => {
    announceWorktreeWrite();
    expect(heard).toHaveBeenCalledTimes(1);

    setWorktreeWriteListener(null);
    announceWorktreeWrite();
    expect(heard).toHaveBeenCalledTimes(1);
  });

  it('holds one listener, so registering again replaces', () => {
    const second = vi.fn(() => undefined);
    setWorktreeWriteListener(second);
    announceWorktreeWrite();

    expect(heard).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledTimes(1);
  });
});

describe('saveActiveGraph announces a project write', () => {
  it('says so after a save into the project', async () => {
    useTabStore.getState().setCurrentGraphFile('demo');
    await saveActiveGraph();

    expect(saveGraphMock).toHaveBeenCalledTimes(1);
    expect(heard).toHaveBeenCalledTimes(1);
  });

  it('says nothing when the save failed', async () => {
    useTabStore.getState().setCurrentGraphFile('demo');
    saveGraphMock.mockRejectedValue(new Error('disk full'));
    await saveActiveGraph();

    expect(heard).not.toHaveBeenCalled();
  });

  it('says nothing outside project mode, where there is no repository', async () => {
    useProjectStore.setState({ projectDir: null, projectName: null, loaded: true });
    promptMock.mockResolvedValue('legacy');
    await saveActiveGraph();

    expect(saveGraphMock).toHaveBeenCalledTimes(1);
    expect(heard).not.toHaveBeenCalled();
  });
});
