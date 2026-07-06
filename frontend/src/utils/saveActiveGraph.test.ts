import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('../api/rest', () => ({
  saveGraph: vi.fn().mockResolvedValue({}),
  listGraphs: vi.fn().mockResolvedValue([]),
}));
vi.mock('./dialog', () => ({
  prompt: vi.fn(),
  confirm: vi.fn().mockResolvedValue(true),
}));

import { saveActiveGraph } from './saveActiveGraph';
import { saveGraph } from '../api/rest';
import { prompt } from './dialog';
import { useTabStore } from '../store/tabStore';
import { useProjectStore } from '../store/projectStore';

function freshTab() {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('test');
}

beforeEach(() => {
  vi.clearAllMocks();
  freshTab();
  useProjectStore.setState({ projectDir: null, projectName: null, loaded: true });
});

describe('saveActiveGraph', () => {
  it('non-project mode always prompts (legacy behavior)', async () => {
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue('legacy');
    useTabStore.getState().setCurrentGraphFile('bound');
    await saveActiveGraph();
    expect(prompt).toHaveBeenCalledTimes(1);
    expect(saveGraph).toHaveBeenCalledWith(expect.objectContaining({ name: 'legacy' }));
  });

  it('project mode + bound overwrites IN PLACE with no prompt', async () => {
    useProjectStore.setState({ projectDir: '/proj', projectName: 'proj', loaded: true });
    useTabStore.getState().setCurrentGraphFile('classifier');
    await saveActiveGraph();
    expect(prompt).not.toHaveBeenCalled();
    expect(saveGraph).toHaveBeenCalledWith(expect.objectContaining({ name: 'classifier' }));
  });

  it('project mode Save As prompts even when bound', async () => {
    useProjectStore.setState({ projectDir: '/proj', projectName: 'proj', loaded: true });
    useTabStore.getState().setCurrentGraphFile('classifier');
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue('copy');
    await saveActiveGraph({ saveAs: true });
    expect(prompt).toHaveBeenCalledTimes(1);
    expect(saveGraph).toHaveBeenCalledWith(expect.objectContaining({ name: 'copy' }));
  });

  it('project mode + unbound (gallery/import) prompts', async () => {
    useProjectStore.setState({ projectDir: '/proj', projectName: 'proj', loaded: true });
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue('fresh');
    await saveActiveGraph();
    expect(prompt).toHaveBeenCalledTimes(1);
    expect(saveGraph).toHaveBeenCalledWith(expect.objectContaining({ name: 'fresh' }));
  });

  it('empty prompt aborts the save', async () => {
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue('');
    await saveActiveGraph();
    expect(saveGraph).not.toHaveBeenCalled();
  });
});
