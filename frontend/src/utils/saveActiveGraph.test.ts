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
  // Per-project storage keys (e.g. last-saved-graph memory) leak across the
  // '/proj'/'/a'/'/b' describe blocks below without this (issue #88).
  localStorage.clear();
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

describe('saveActiveGraph cross-project guard (ID10)', () => {
  it('refuses to save a tab whose origin differs from the open project', async () => {
    useProjectStore.setState({ projectDir: '/b', projectName: 'b', loaded: true });
    useTabStore.getState().setCurrentGraphFile('classifier');
    useTabStore.getState().stampActiveTabProject('/a'); // belongs to project A
    await saveActiveGraph();
    expect(saveGraph).not.toHaveBeenCalled();
  });

  it('stamps the origin after a successful project save', async () => {
    useProjectStore.setState({ projectDir: '/b', projectName: 'b', loaded: true });
    useTabStore.getState().setCurrentGraphFile('classifier');
    await saveActiveGraph();
    const st = useTabStore.getState();
    const tab = st.tabs.find((t) => t.id === st.activeTabId)!;
    expect(tab.projectOrigin).toBe('/b');
  });
});

describe('saveActiveGraph read-only guard (ID8)', () => {
  it('refuses to save a read-only (newer-format) graph', async () => {
    useTabStore.getState().setTabReadOnly(true);
    await saveActiveGraph();
    expect(saveGraph).not.toHaveBeenCalled();
  });

  // ID8 fast-follow (task 16 review Adjudication B / Important finding 1):
  // clear() must reset readOnly, otherwise a cleared (fresh, empty) graph is
  // stuck refusing Save forever even though it is trivially current-format.
  it('clear() resets readOnly so a subsequent save is no longer refused', async () => {
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue('fresh-after-clear');
    useTabStore.getState().setTabReadOnly(true);

    useTabStore.getState().clear();

    await saveActiveGraph();
    expect(saveGraph).toHaveBeenCalledWith(expect.objectContaining({ name: 'fresh-after-clear' }));
  });
});
