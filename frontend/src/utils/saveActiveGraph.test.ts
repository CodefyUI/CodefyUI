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
import { buildInstanceNode } from './subgraph';

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

  // #200 item 9. The binding is the whole of what decides "overwrite in
  // place, no prompt", so a document installed with no binding has to take
  // the prompt path even when the tab was bound a moment earlier -- which is
  // what opening an example into a tab bound to a file now does.
  it('project mode: a document opened with no file binding prompts instead of overwriting the file that was open', async () => {
    useProjectStore.setState({ projectDir: '/proj', projectName: 'proj', loaded: true });
    useTabStore.getState().setCurrentGraphFile('classifier');
    // Exactly what `openExample` hands the store now.
    useTabStore.getState().loadGraphDocument({ nodes: [], edges: [], boundFile: null });
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue('from-template');

    await saveActiveGraph();

    expect(prompt).toHaveBeenCalledTimes(1);
    expect(saveGraph).toHaveBeenCalledWith(expect.objectContaining({ name: 'from-template' }));
    expect(saveGraph).not.toHaveBeenCalledWith(expect.objectContaining({ name: 'classifier' }));
  });

  it('empty prompt aborts the save', async () => {
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue('');
    await saveActiveGraph();
    expect(saveGraph).not.toHaveBeenCalled();
  });

  // core#137 review, CRITICAL 1. The payload used to omit `subgraphs`
  // entirely, so saving a graph with a collapsed block wrote out the
  // instance node and none of its contents. This is the cheap half of the
  // guard; `saveActiveGraph.wire.test.ts` is the half that matters, because
  // it checks the real request body rather than a mocked call argument.
  //
  // The instance node is NOT decoration in this fixture: serialization prunes
  // definitions nothing on the canvas can reach (review finding 9), so a
  // definition with no instance would be dropped on purpose and this test
  // would be asserting against a state the app cannot produce.
  it('forwards subgraph definitions to saveGraph', async () => {
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue('with-blocks');
    const definition = {
      id: 'blk',
      name: 'Block',
      description: '',
      nodes: [],
      edges: [],
      interface: { inputs: [], outputs: [], triggerTargets: [] },
    };
    useTabStore.getState().setSubgraphs([definition]);
    useTabStore.getState().setNodes([
      buildInstanceNode(definition, { x: 0, y: 0 }, 'blk-1'),
    ]);
    await saveActiveGraph();
    expect(saveGraph).toHaveBeenCalledWith(
      expect.objectContaining({ subgraphs: [definition] }),
    );
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
