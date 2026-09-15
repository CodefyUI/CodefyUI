import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import type { Node } from '@xyflow/react';
import { GraphsTab, graphMatches, sortGraphs } from './GraphsTab';
import { useI18n } from '../../i18n';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useProjectStore } from '../../store/projectStore';
import { useTabStore } from '../../store/tabStore';
import { useToastStore } from '../../store/toastStore';
import { setWorktreeWriteListener } from '../../utils/worktreeWrite';
import { confirm, prompt } from '../../utils/dialog';
import { importGraphFile } from '../../utils/importGraphFile';
import { saveActiveGraph } from '../../utils/saveActiveGraph';
import * as rest from '../../api/rest';
import type { SavedGraphSummary } from '../../api/rest';
import type { NodeData } from '../../types';

// The five graph routes are stubbed; everything between the click and the
// canvas -- `openSavedGraph`, `resolveSavedGraph`, `loadGraphDocument` -- runs
// for real, because "a row click opens AND BINDS" is a fact about the tab
// store, not about which function was called.
vi.mock('../../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/rest')>();
  return {
    ...actual,
    listGraphs: vi.fn(),
    loadGraph: vi.fn(),
    deleteGraph: vi.fn(),
    renameGraph: vi.fn(),
  };
});
vi.mock('../../utils/dialog', () => ({ confirm: vi.fn(), prompt: vi.fn() }));
vi.mock('../../utils/importGraphFile', () => ({ importGraphFile: vi.fn() }));
vi.mock('../../utils/saveActiveGraph', () => ({ saveActiveGraph: vi.fn() }));

const mockedRest = vi.mocked(rest);
const mockedConfirm = vi.mocked(confirm);
const mockedPrompt = vi.mocked(prompt);
const mockedImport = vi.mocked(importGraphFile);
const mockedSaveAs = vi.mocked(saveActiveGraph);

const NOW_SECONDS = Math.floor(Date.now() / 1000);

function graph(overrides: Partial<SavedGraphSummary> = {}): SavedGraphSummary {
  return { name: 'alpha', file: 'alpha', modified: NOW_SECONDS, ...overrides };
}

/** One node on the canvas, which is all "there is work here to lose" means. */
function someNode(): Node<NodeData> {
  return {
    id: 'n1',
    type: 'baseNode',
    position: { x: 0, y: 0 },
    data: { label: 'Linear', type: 'Linear', params: {} },
  } as Node<NodeData>;
}

function freshTab() {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
  useTabStore.getState().addTab('Tab 1');
}

const activeTab = () => useTabStore.getState().getActiveTab();
const tabById = (id: string) => useTabStore.getState().tabs.find((tb) => tb.id === id)!;

/**
 * A list read this test decides the answer to, and when.
 *
 * Every mutation refetches, so two reads are in flight together as a matter
 * of course; the ones that matter here are the pairs that answer out of
 * order.
 */
function deferredList() {
  let settle: (rows: SavedGraphSummary[]) => void = () => {};
  const promise = new Promise<SavedGraphSummary[]>((resolve) => { settle = resolve; });
  return { promise, settle: (rows: SavedGraphSummary[]) => { settle(rows); } };
}

/** Let a settled read reach the component, or be discarded by it. */
async function flush() {
  await act(async () => {
    await new Promise((resolve) => { setTimeout(resolve, 0); });
  });
}

/** The rows on screen, top to bottom, by the name each one shows. */
function rowNames(): string[] {
  return screen
    .getAllByRole('button', { name: /^Open / })
    .map((el) => (el.getAttribute('aria-label') ?? '').replace(/^Open /, ''));
}

/** Open one row's overflow menu and hand back the menu element. */
function openRowMenu(name: string): HTMLElement {
  fireEvent.click(screen.getByRole('button', { name: `More actions ${name}` }));
  return screen.getByRole('menu', { name: `More actions ${name}` });
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useNodeDefStore.setState({ definitions: [], presets: [] });
  useToastStore.setState({ toasts: [] });
  useProjectStore.setState({ projectDir: null, projectName: null, loaded: true });
  localStorage.clear();
  freshTab();
  vi.clearAllMocks();
  mockedRest.listGraphs.mockResolvedValue([]);
  mockedRest.loadGraph.mockResolvedValue({ nodes: [], edges: [] });
  mockedRest.deleteGraph.mockResolvedValue({});
  mockedRest.renameGraph.mockResolvedValue({});
  mockedConfirm.mockResolvedValue(true);
  mockedPrompt.mockResolvedValue(null);
  mockedImport.mockResolvedValue(true);
  mockedSaveAs.mockResolvedValue(undefined);
});

afterEach(() => {
  setWorktreeWriteListener(null);
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('sortGraphs', () => {
  it('puts the most recently modified first', () => {
    const sorted = sortGraphs([
      graph({ name: 'old', file: 'old', modified: 100 }),
      graph({ name: 'new', file: 'new', modified: 300 }),
      graph({ name: 'mid', file: 'mid', modified: 200 }),
    ]);
    expect(sorted.map((g) => g.name)).toEqual(['new', 'mid', 'old']);
  });

  it('falls back to the name when nothing is dated', () => {
    const sorted = sortGraphs([
      graph({ name: 'zeta', file: 'zeta', modified: undefined }),
      graph({ name: 'alpha', file: 'alpha', modified: undefined }),
    ]);
    expect(sorted.map((g) => g.name)).toEqual(['alpha', 'zeta']);
  });

  it('leads with the dated rows when only some carry a timestamp', () => {
    const sorted = sortGraphs([
      graph({ name: 'undated', file: 'undated', modified: undefined }),
      graph({ name: 'dated', file: 'dated', modified: 100 }),
    ]);
    expect(sorted.map((g) => g.name)).toEqual(['dated', 'undated']);
  });
});

describe('graphMatches', () => {
  it('matches the display name and the sanitized file stem', () => {
    const g = graph({ name: 'My Graph', file: 'My_Graph' });
    expect(graphMatches(g, 'my graph')).toBe(true);
    expect(graphMatches(g, 'my_graph')).toBe(true);
    expect(graphMatches(g, 'other')).toBe(false);
  });
});

describe('GraphsTab list', () => {
  it('renders the graphs most recently modified first', async () => {
    mockedRest.listGraphs.mockResolvedValue([
      graph({ name: 'oldest', file: 'oldest', modified: NOW_SECONDS - 7200 }),
      graph({ name: 'newest', file: 'newest', modified: NOW_SECONDS - 60 }),
      graph({ name: 'middle', file: 'middle', modified: NOW_SECONDS - 3600 }),
    ]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open newest' });
    expect(rowNames()).toEqual(['newest', 'middle', 'oldest']);
  });

  it('shows when each graph was last written', async () => {
    mockedRest.listGraphs.mockResolvedValue([
      graph({ modified: NOW_SECONDS - 3600 }),
    ]);
    render(<GraphsTab />);
    expect(await screen.findByText('Modified 1 hour ago')).toBeTruthy();
  });

  it('marks the row the active tab is bound to', async () => {
    useTabStore.getState().setCurrentGraphFile('beta');
    mockedRest.listGraphs.mockResolvedValue([
      graph({ name: 'alpha', file: 'alpha' }),
      graph({ name: 'beta', file: 'beta' }),
    ]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open beta' });
    // Exactly one row carries it, and it is the bound one.
    const marks = screen.getAllByText('Current');
    expect(marks).toHaveLength(1);
    expect(marks[0].closest('li')?.textContent).toContain('beta');
  });

  it('says which row is bound to a reader who cannot see the chip', async () => {
    useTabStore.getState().setCurrentGraphFile('beta');
    mockedRest.listGraphs.mockResolvedValue([
      graph({ name: 'alpha', file: 'alpha' }),
      graph({ name: 'beta', file: 'beta' }),
    ]);
    render(<GraphsTab />);
    const bound = await screen.findByRole('button', { name: 'Open beta' });
    const chip = screen.getByText('Current');
    // Inside the row button, the chip was content that button's `aria-label`
    // replaced -- on screen and announced to nobody.
    expect(chip.closest('button')).toBeNull();
    expect(chip.closest('li')).toBe(bound.closest('li'));
    expect(bound).toHaveAccessibleDescription(/Current/);
    expect(screen.getByRole('button', { name: 'Open alpha' }))
      .not.toHaveAccessibleDescription(/Current/);
  });

  it('spends two characters on the chip in zh-TW, as the branch list does', async () => {
    useI18n.setState({ locale: 'zh-TW' });
    useTabStore.getState().setCurrentGraphFile('alpha');
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    // Twice this wide left about 59px for the name at the 180px sidebar
    // floor. `git.branch.current` is the same two characters.
    expect(await screen.findByText('目前')).toBeTruthy();
  });

  it('filters on the name and on the file stem', async () => {
    mockedRest.listGraphs.mockResolvedValue([
      graph({ name: 'My Graph', file: 'My_Graph', modified: NOW_SECONDS }),
      graph({ name: 'classifier', file: 'classifier', modified: NOW_SECONDS - 10 }),
    ]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open My Graph' });
    const search = screen.getByPlaceholderText('Search saved graphs...');

    fireEvent.change(search, { target: { value: 'classi' } });
    expect(rowNames()).toEqual(['classifier']);

    // The sanitized stem is what the file is actually called, and someone who
    // typed it meant the graph whose display name still has the space in it.
    fireEvent.change(search, { target: { value: 'my_graph' } });
    expect(rowNames()).toEqual(['My Graph']);

    fireEvent.change(search, { target: { value: 'nothing' } });
    expect(screen.getByText('No matching graphs')).toBeTruthy();
  });

  it('does not echo the query back into the no-match line', async () => {
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    // One unbroken token, which is what a paste is: repeating it into a
    // centred message inside a panel that clips leaves it nowhere to wrap.
    const pasted = 'z'.repeat(200);
    fireEvent.change(screen.getByPlaceholderText('Search saved graphs...'), {
      target: { value: pasted },
    });
    expect(screen.getByText('No matching graphs')).toBeTruthy();
    expect(screen.queryByText(pasted, { exact: false })).toBeNull();
  });

  it('refreshing re-reads the list', async () => {
    mockedRest.listGraphs
      .mockResolvedValueOnce([graph({ name: 'before', file: 'before' })])
      .mockResolvedValueOnce([graph({ name: 'after', file: 'after' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open before' });
    fireEvent.click(screen.getByRole('button', { name: 'Refresh the graph list' }));
    expect(await screen.findByRole('button', { name: 'Open after' })).toBeTruthy();
  });

  it('shows one line and the import action when there is nothing saved', async () => {
    render(<GraphsTab />);
    expect(await screen.findByText('No saved graphs')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Import JSON...' })).toBeTruthy();
    // Nothing else: no list, no explanation of what a saved graph is.
    expect(screen.queryByRole('list')).toBeNull();
  });

  it('reports a failed list read and retries it', async () => {
    mockedRest.listGraphs.mockRejectedValueOnce(new Error('server down'));
    mockedRest.listGraphs.mockResolvedValueOnce([graph({ name: 'back', file: 'back' })]);
    render(<GraphsTab />);
    expect(await screen.findByText('Failed to load graphs: server down')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(await screen.findByRole('button', { name: 'Open back' })).toBeTruthy();
  });
});

describe('GraphsTab opening', () => {
  it('a row click opens the graph and binds the tab to it', async () => {
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    fireEvent.click(await screen.findByRole('button', { name: 'Open alpha' }));
    await waitFor(() => {
      expect(mockedRest.loadGraph).toHaveBeenCalledWith('alpha');
    });
    await waitFor(() => {
      expect(activeTab().currentGraphFile).toBe('alpha');
    });
    // An empty canvas has nothing to lose, so nothing was asked.
    expect(mockedConfirm).not.toHaveBeenCalled();
  });

  it('asks first when the canvas has work on it', async () => {
    useTabStore.getState().setNodes([someNode()]);
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    fireEvent.click(await screen.findByRole('button', { name: 'Open alpha' }));
    await waitFor(() => {
      expect(mockedConfirm).toHaveBeenCalledWith(
        expect.objectContaining({ title: 'Replace this canvas with "alpha"?' }),
      );
    });
    await waitFor(() => {
      expect(activeTab().currentGraphFile).toBe('alpha');
    });
  });

  it('cancelling the question leaves the canvas and the binding alone', async () => {
    useTabStore.getState().setNodes([someNode()]);
    useTabStore.getState().setCurrentGraphFile('other');
    mockedConfirm.mockResolvedValue(false);
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    fireEvent.click(await screen.findByRole('button', { name: 'Open alpha' }));
    await waitFor(() => {
      expect(mockedConfirm).toHaveBeenCalled();
    });
    expect(mockedRest.loadGraph).not.toHaveBeenCalled();
    expect(activeTab().nodes).toHaveLength(1);
    expect(activeTab().currentGraphFile).toBe('other');
  });

  it('"onto canvas" replaces the graph but binds the tab to nothing', async () => {
    useTabStore.getState().setCurrentGraphFile('other');
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    const menu = openRowMenu('alpha');
    fireEvent.click(
      within(menu).getByRole('menuitem', { name: 'Load onto canvas without binding' }),
    );
    await waitFor(() => {
      expect(activeTab().currentGraphFile).toBeNull();
    });
  });

  it('opens into a new tab without disturbing the one in front of the user', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ nodes: [], edges: [] }),
      }),
    );
    useTabStore.getState().setNodes([someNode()]);
    const firstTabId = useTabStore.getState().activeTabId;
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    const menu = openRowMenu('alpha');
    fireEvent.click(within(menu).getByRole('menuitem', { name: 'Open in new tab' }));
    await waitFor(() => {
      expect(useTabStore.getState().tabs).toHaveLength(2);
    });
    const opened = activeTab();
    expect(opened.id).not.toBe(firstTabId);
    expect(opened.name).toBe('alpha');
    expect(opened.currentGraphFile).toBe('alpha');
    // The tab that was open still has its work, and its binding.
    const first = useTabStore.getState().tabs.find((tb) => tb.id === firstTabId)!;
    expect(first.nodes).toHaveLength(1);
  });
});

describe('GraphsTab rename', () => {
  it('renames the file, re-reads the list and says so', async () => {
    mockedPrompt.mockResolvedValue('Renamed Graph');
    mockedRest.renameGraph.mockResolvedValue({
      message: 'Graph renamed', name: 'Renamed Graph', file: 'Renamed_Graph',
    });
    mockedRest.listGraphs
      .mockResolvedValueOnce([graph({ name: 'alpha', file: 'alpha' })])
      .mockResolvedValueOnce([graph({ name: 'Renamed Graph', file: 'Renamed_Graph' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Rename...' }));
    await waitFor(() => {
      expect(mockedRest.renameGraph).toHaveBeenCalledWith('alpha', 'Renamed Graph');
    });
    expect(await screen.findByRole('button', { name: 'Open Renamed Graph' })).toBeTruthy();
    expect(useToastStore.getState().toasts[0]).toMatchObject({
      message: 'Renamed to "Renamed Graph".', type: 'success',
    });
  });

  it('moves the active tab\'s binding onto the new file', async () => {
    useTabStore.getState().setCurrentGraphFile('alpha');
    mockedPrompt.mockResolvedValue('Renamed Graph');
    mockedRest.renameGraph.mockResolvedValue({
      message: 'Graph renamed', name: 'Renamed Graph', file: 'Renamed_Graph',
    });
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Rename...' }));
    await waitFor(() => {
      expect(activeTab().currentGraphFile).toBe('Renamed_Graph');
    });
  });

  it('moves the binding of a tab that is not the one in front of the user', async () => {
    // Tab 1 opened alpha; the user is now in tab 2, which never saw it.
    useTabStore.getState().setCurrentGraphFile('alpha');
    const background = useTabStore.getState().activeTabId;
    useTabStore.getState().addTab('Tab 2');
    mockedPrompt.mockResolvedValue('Renamed Graph');
    mockedRest.renameGraph.mockResolvedValue({ file: 'Renamed_Graph' });
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Rename...' }));
    // Left on the old name, tab 1's next Save would write the graph to disk
    // a second time, under a file the user renamed away.
    await waitFor(() => {
      expect(tabById(background).currentGraphFile).toBe('Renamed_Graph');
    });
    expect(activeTab().currentGraphFile).toBeNull();
  });

  it('leaves a binding on a different file alone', async () => {
    useTabStore.getState().setCurrentGraphFile('beta');
    mockedPrompt.mockResolvedValue('Renamed Graph');
    mockedRest.renameGraph.mockResolvedValue({ file: 'Renamed_Graph' });
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Rename...' }));
    await waitFor(() => {
      expect(mockedRest.renameGraph).toHaveBeenCalled();
    });
    expect(activeTab().currentGraphFile).toBe('beta');
  });

  it('cancelling the prompt writes nothing', async () => {
    mockedPrompt.mockResolvedValue(null);
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Rename...' }));
    await waitFor(() => {
      expect(mockedPrompt).toHaveBeenCalled();
    });
    expect(mockedRest.renameGraph).not.toHaveBeenCalled();
  });

  it('reports a refused rename and keeps the list as it was', async () => {
    mockedPrompt.mockResolvedValue('beta');
    mockedRest.renameGraph.mockRejectedValue(new Error("Graph 'beta' already exists"));
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Rename...' }));
    await waitFor(() => {
      expect(useToastStore.getState().toasts[0]).toMatchObject({
        message: "Rename failed: Graph 'beta' already exists", type: 'error',
      });
    });
    expect(screen.getByRole('button', { name: 'Open alpha' })).toBeTruthy();
  });

  it('project mode tells the Source Control tab about the write', async () => {
    useProjectStore.setState({ projectDir: '/proj', projectName: 'proj', loaded: true });
    const wrote = vi.fn();
    setWorktreeWriteListener(wrote);
    mockedPrompt.mockResolvedValue('Renamed Graph');
    mockedRest.renameGraph.mockResolvedValue({ file: 'Renamed_Graph' });
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Rename...' }));
    await waitFor(() => {
      expect(wrote).toHaveBeenCalled();
    });
  });
});

describe('GraphsTab delete', () => {
  it('asks, deletes, re-reads the list and says so', async () => {
    mockedRest.listGraphs
      .mockResolvedValueOnce([graph({ name: 'alpha', file: 'alpha' })])
      .mockResolvedValueOnce([]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Delete' }));
    await waitFor(() => {
      expect(mockedConfirm).toHaveBeenCalledWith(
        expect.objectContaining({
          title: 'Delete "alpha"? The file is removed from disk.',
          variant: 'danger',
        }),
      );
    });
    await waitFor(() => {
      expect(mockedRest.deleteGraph).toHaveBeenCalledWith('alpha');
    });
    expect(await screen.findByText('No saved graphs')).toBeTruthy();
    expect(useToastStore.getState().toasts[0]).toMatchObject({
      message: 'Deleted "alpha".', type: 'success',
    });
  });

  it('cancelling deletes nothing', async () => {
    mockedConfirm.mockResolvedValue(false);
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Delete' }));
    await waitFor(() => {
      expect(mockedConfirm).toHaveBeenCalled();
    });
    expect(mockedRest.deleteGraph).not.toHaveBeenCalled();
  });

  it('clears the active tab\'s binding when its own file is the one deleted', async () => {
    useTabStore.getState().setCurrentGraphFile('alpha');
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Delete' }));
    await waitFor(() => {
      expect(activeTab().currentGraphFile).toBeNull();
    });
  });

  it('clears the binding of a tab that is not the one in front of the user', async () => {
    useTabStore.getState().setCurrentGraphFile('alpha');
    const background = useTabStore.getState().activeTabId;
    useTabStore.getState().addTab('Tab 2');
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Delete' }));
    // Otherwise a Ctrl+S in tab 1 recreates the file that was just deleted,
    // with no prompt and no collision check.
    await waitFor(() => {
      expect(tabById(background).currentGraphFile).toBeNull();
    });
  });

  it('reports a failed delete', async () => {
    mockedRest.deleteGraph.mockRejectedValue(new Error("Graph 'alpha' not found"));
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Delete' }));
    await waitFor(() => {
      expect(useToastStore.getState().toasts[0]).toMatchObject({
        message: "Delete failed: Graph 'alpha' not found", type: 'error',
      });
    });
  });
});

describe('GraphsTab refetching', () => {
  it('discards a list read that lands after a newer one', async () => {
    const stale = deferredList();
    mockedRest.listGraphs
      // Mount.
      .mockResolvedValueOnce([
        graph({ name: 'alpha', file: 'alpha' }),
        graph({ name: 'beta', file: 'beta', modified: NOW_SECONDS - 60 }),
      ])
      // The Save As refetch, still in flight when the delete happens.
      .mockImplementationOnce(() => stale.promise)
      // The delete's own refetch, which is the one that is true.
      .mockResolvedValueOnce([graph({ name: 'beta', file: 'beta' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });

    fireEvent.click(screen.getByRole('button', { name: 'Save as...' }));
    await waitFor(() => {
      expect(mockedRest.listGraphs).toHaveBeenCalledTimes(2);
    });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Delete' }));
    await waitFor(() => {
      expect(rowNames()).toEqual(['beta']);
    });

    // The read that started BEFORE the delete answers last, carrying a list
    // the deleted row is still in.
    stale.settle([
      graph({ name: 'alpha', file: 'alpha' }),
      graph({ name: 'beta', file: 'beta', modified: NOW_SECONDS - 60 }),
    ]);
    await flush();
    expect(rowNames()).toEqual(['beta']);
  });

  it('keeps the list on screen while a delete refetches, and blanks it for the refresh button', async () => {
    const afterDelete = deferredList();
    const afterRefresh = deferredList();
    mockedRest.listGraphs
      .mockResolvedValueOnce([
        graph({ name: 'alpha', file: 'alpha' }),
        graph({ name: 'beta', file: 'beta', modified: NOW_SECONDS - 60 }),
      ])
      .mockImplementationOnce(() => afterDelete.promise)
      .mockImplementationOnce(() => afterRefresh.promise);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });

    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Delete' }));
    await waitFor(() => {
      expect(mockedRest.listGraphs).toHaveBeenCalledTimes(2);
    });
    // One row went; the rest of the list -- and the reader's place in it --
    // stays where it was.
    expect(screen.queryByText('Loading graphs...')).toBeNull();
    expect(screen.getByRole('button', { name: 'Open beta' })).toBeTruthy();
    afterDelete.settle([graph({ name: 'beta', file: 'beta' })]);
    await flush();

    // The refresh button is a read the user asked for, and says so.
    fireEvent.click(screen.getByRole('button', { name: 'Refresh the graph list' }));
    expect(screen.getByText('Loading graphs...')).toBeTruthy();
    afterRefresh.settle([graph({ name: 'beta', file: 'beta' })]);
    await flush();
    expect(rowNames()).toEqual(['beta']);
  });
});

describe('GraphsTab import and save', () => {
  it('the footer button proxies the click to the hidden file input', async () => {
    const { container } = render(<GraphsTab />);
    await screen.findByText('No saved graphs');
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    expect(fileInput.accept).toBe('.json');
    const clickSpy = vi.spyOn(fileInput, 'click').mockImplementation(() => {});
    fireEvent.click(screen.getByRole('button', { name: 'Import JSON...' }));
    expect(clickSpy).toHaveBeenCalled();
  });

  it('a picked file goes to importGraphFile and the input is cleared', async () => {
    const { container } = render(<GraphsTab />);
    await screen.findByText('No saved graphs');
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(['{"nodes":[]}'], 'graph.json', { type: 'application/json' });
    fireEvent.change(fileInput, { target: { files: [file] } });
    expect(mockedImport).toHaveBeenCalledWith(file);
    // Cleared now, not after the read, so picking the SAME file again fires.
    expect(fileInput.value).toBe('');
  });

  it('picking nothing imports nothing', async () => {
    const { container } = render(<GraphsTab />);
    await screen.findByText('No saved graphs');
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [] } });
    expect(mockedImport).not.toHaveBeenCalled();
  });

  it('save-as runs the toolbar\'s Save As and re-reads the list', async () => {
    render(<GraphsTab />);
    await screen.findByText('No saved graphs');
    fireEvent.click(screen.getByRole('button', { name: 'Save as...' }));
    await waitFor(() => {
      expect(mockedSaveAs).toHaveBeenCalledWith({ saveAs: true });
    });
    await waitFor(() => {
      expect(mockedRest.listGraphs).toHaveBeenCalledTimes(2);
    });
  });
});
