import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act } from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { Toolbar } from './Toolbar';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useToastStore } from '../../store/toastStore';
import { useDialogStore } from '../../store/dialogStore';
import { useProjectStore } from '../../store/projectStore';
import { useI18n } from '../../i18n';
import * as rest from '../../api/rest';
import * as exportDiagram from '../../utils/exportDiagram';

// ── Mocks ─────────────────────────────────────────────────────────────

const execute = vi.fn();
const stop = vi.fn();
vi.mock('../../hooks/useGraphExecution', () => ({
  useGraphExecution: () => ({ execute, stop }),
}));

vi.mock('../../api/rest', () => ({
  // Used directly by Toolbar
  saveGraph: vi.fn(),
  loadGraph: vi.fn(),
  listGraphs: vi.fn(),
  createPreset: vi.fn(),
  exportGraph: vi.fn(),
  // Used by the child CustomNodeManager
  listCustomNodes: vi.fn(),
  toggleCustomNode: vi.fn(),
  uploadCustomNode: vi.fn(),
  deleteCustomNode: vi.fn(),
  // Used by the child SettingsPopover
  resetWeights: vi.fn(),
  fetchDevices: vi.fn(() =>
    Promise.resolve({ default: 'cpu', devices: [{ value: 'cpu', label: 'CPU', detail: '', available: true }] }),
  ),
  fetchCodexStatus: vi.fn(() => Promise.resolve({ status: 'logged_out' })),
  startCodexLogin: vi.fn(() => Promise.resolve({ auth_url: 'https://auth.example' })),
  logoutCodex: vi.fn(() => Promise.resolve({ status: 'logged_out' })),
}));

const mockedRest = vi.mocked(rest);

// Keep the real graphToSvg (pure), but stub PNG rasterization — it relies on
// Image/<canvas>, which jsdom does not implement.
vi.mock('../../utils/exportDiagram', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../utils/exportDiagram')>();
  return { ...actual, svgToPngBlob: vi.fn() };
});
const mockedExportDiagram = vi.mocked(exportDiagram);

// ── Helpers ───────────────────────────────────────────────────────────

function setActiveTab(overrides: Record<string, unknown> = {}) {
  // Build a fully-valid TabState by reusing the store's existing real tab
  // (so undoStack / redoStack / dirtyNodeIds / ws etc. are present) and
  // layering the test overrides on top.
  const real = useTabStore.getState().tabs[0];
  const tab = {
    ...real,
    id: 'tab-1',
    name: 'My Graph',
    nodes: [] as any[],
    edges: [] as any[],
    status: 'idle',
    recordOutputs: true,
    verboseMode: false,
    weightsPersistent: true,
    backwardMode: false,
    autoBackward: false,
    graphId: 'graph-xyz',
    activeSegment: null as any,
    segmentGroups: [] as any[],
    undoStack: [],
    redoStack: [],
    ...overrides,
  };
  useTabStore.setState({ tabs: [tab as never], activeTabId: 'tab-1' });
}

/** Resolve a pending dialog (confirm/prompt) from the dialog store. */
async function resolveDialog(value: boolean | string | null) {
  await waitFor(() => expect(useDialogStore.getState().active).not.toBeNull());
  await act(async () => {
    useDialogStore.getState().close(value);
  });
}

describe('Toolbar', () => {
  beforeEach(() => {
    useI18n.setState({ locale: 'en' });
    useToastStore.setState({ toasts: [] });
    useDialogStore.setState({ active: null, resolve: null });
    useUIStore.setState({
      lastLayoutMode: 'experiments',
      gridSnapEnabled: false,
      tooltipsEnabled: true,
      beginnerMode: false,
      shortcutsModalOpen: false,
      fontSize: 'default',
    });
    useNodeDefStore.setState({ definitions: [], presets: [], categorized: {}, presetCategorized: {} });
    useProjectStore.setState({ projectDir: null, projectName: null, loaded: false });
    setActiveTab();

    // Stub blob-download plumbing (jsdom lacks createObjectURL).
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:mock');
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    // Default async resolutions
    mockedRest.listGraphs.mockResolvedValue([]);
    mockedRest.listCustomNodes.mockResolvedValue([]);
    // saveGraph is a shared vi.fn() from the module mock — restoreAllMocks
    // does not reset factory mocks, so clear its call history each test to
    // keep per-test "was/was not called" assertions order-independent.
    mockedRest.saveGraph.mockClear();

    mockedExportDiagram.svgToPngBlob.mockReset();
    mockedExportDiagram.svgToPngBlob.mockResolvedValue(new Blob(['png'], { type: 'image/png' }));

    execute.mockReset();
    stop.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // ── Basic render ────────────────────────────────────────────────────

  it('renders the brand, run/stop, menus and right cluster', () => {
    render(<Toolbar />);
    expect(screen.getByText('Codefy')).toBeInTheDocument();
    expect(screen.getByText('UI')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Run' })).toBeInTheDocument();
    expect(screen.getByText('Stop')).toBeInTheDocument();
    expect(screen.getByText('File')).toBeInTheDocument();
    expect(screen.getByText('Load')).toBeInTheDocument();
    expect(screen.getByText('Export')).toBeInTheDocument();
    expect(screen.getByText('Reload Nodes')).toBeInTheDocument();
    expect(screen.getByText('Custom Nodes')).toBeInTheDocument();
    expect(screen.getByText('Auto Layout')).toBeInTheDocument();
  });

  // ── Run / Stop ──────────────────────────────────────────────────────

  it('idle: Run enabled, Stop disabled; clicking Run executes', () => {
    render(<Toolbar />);
    const run = screen.getByRole('button', { name: 'Run' });
    const stopBtn = screen.getByText('Stop');
    expect(run).not.toBeDisabled();
    expect(stopBtn).toBeDisabled();
    fireEvent.click(run);
    expect(execute).toHaveBeenCalledTimes(1);
  });

  it('running: Run disabled & shows "Running...", Stop enabled; clicking Stop stops', () => {
    setActiveTab({ status: 'running' });
    render(<Toolbar />);
    const run = screen.getByRole('button', { name: 'Running...' });
    const stopBtn = screen.getByText('Stop');
    expect(run).toBeDisabled();
    expect(stopBtn).not.toBeDisabled();
    fireEvent.click(stopBtn);
    expect(stop).toHaveBeenCalledTimes(1);
  });

  // ── Status visuals (statusDotColor map + glow + running text color) ─

  it.each([
    ['idle', 'Idle'],
    ['running', 'Running'],
    ['completed', 'Completed'],
    ['error', 'Error'],
    ['cached', 'Cached'],
    ['skipped', 'Skipped'],
  ] as const)('renders status label for %s', (status, label) => {
    setActiveTab({ status });
    render(<Toolbar />);
    expect(screen.getByText(label)).toBeInTheDocument();
  });

  it('uses the fallback status color for an unknown status', () => {
    setActiveTab({ status: 'weird-unknown' as never });
    render(<Toolbar />);
    // The status key is unknown so t() echoes the key.
    expect(screen.getByText('status.weird-unknown')).toBeInTheDocument();
  });

  // ── File menu (MenuDropdown) ────────────────────────────────────────

  it('opens and closes the File menu via toggle', () => {
    render(<Toolbar />);
    const fileBtn = screen.getByText('File');
    fireEvent.click(fileBtn);
    expect(screen.getByText('Save')).toBeInTheDocument();
    expect(screen.getByText('Clear Canvas')).toBeInTheDocument();
    // toggle again closes
    fireEvent.click(fileBtn);
    expect(screen.queryByText('Save')).toBeNull();
  });

  it('File menu closes on outside mousedown', () => {
    render(<Toolbar />);
    fireEvent.click(screen.getByText('File'));
    expect(screen.getByText('Save')).toBeInTheDocument();
    fireEvent.mouseDown(document.body);
    expect(screen.queryByText('Save')).toBeNull();
  });

  it('File menu does NOT close when mousedown is inside it', () => {
    render(<Toolbar />);
    fireEvent.click(screen.getByText('File'));
    fireEvent.mouseDown(screen.getByText('Save'));
    expect(screen.getByText('Save')).toBeInTheDocument();
  });

  it('opening a second menu closes the first (toggleMenu prev===name false branch)', () => {
    render(<Toolbar />);
    fireEvent.click(screen.getByText('File'));
    expect(screen.getByText('Save')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Export'));
    expect(screen.queryByText('Save')).toBeNull();
    expect(screen.getByText('Export as JSON')).toBeInTheDocument();
  });

  // ── Save action ─────────────────────────────────────────────────────

  it('Save: empty/blank name aborts without calling saveGraph', async () => {
    render(<Toolbar />);
    fireEvent.click(screen.getByText('File'));
    fireEvent.click(screen.getByText('Save'));
    await resolveDialog('   '); // whitespace -> trimmed empty
    expect(mockedRest.saveGraph).not.toHaveBeenCalled();
  });

  it('Save: cancel (null) aborts', async () => {
    render(<Toolbar />);
    fireEvent.click(screen.getByText('File'));
    fireEvent.click(screen.getByText('Save'));
    await resolveDialog(null);
    expect(mockedRest.saveGraph).not.toHaveBeenCalled();
  });

  it('Save: success path calls saveGraph and toasts success', async () => {
    mockedRest.saveGraph.mockResolvedValueOnce({} as never);
    setActiveTab({
      nodes: [
        { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { type: 'Add', params: {} } },
      ],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('File'));
    fireEvent.click(screen.getByText('Save'));
    await resolveDialog('my-graph');
    await waitFor(() =>
      expect(mockedRest.saveGraph).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'my-graph', description: '' }),
      ),
    );
    await waitFor(() =>
      expect(useToastStore.getState().toasts.some((t) => t.type === 'success')).toBe(true),
    );
  });

  it('Save: failure path toasts error', async () => {
    mockedRest.saveGraph.mockRejectedValueOnce(new Error('disk full'));
    render(<Toolbar />);
    fireEvent.click(screen.getByText('File'));
    fireEvent.click(screen.getByText('Save'));
    await resolveDialog('g');
    await waitFor(() =>
      expect(
        useToastStore.getState().toasts.some((t) => t.type === 'error' && t.message.includes('disk full')),
      ).toBe(true),
    );
  });

  it('Save: carries the tab description through to saveGraph (round-trip half)', async () => {
    mockedRest.saveGraph.mockResolvedValueOnce({} as never);
    setActiveTab({ description: 'my important description' });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('File'));
    fireEvent.click(screen.getByText('Save'));
    await resolveDialog('my-graph');
    await waitFor(() =>
      expect(mockedRest.saveGraph).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'my-graph', description: 'my important description' }),
      ),
    );
  });

  it('Save: forwards segmentGroups from the serialized graph', async () => {
    mockedRest.saveGraph.mockResolvedValueOnce({} as never);
    setActiveTab({ segmentGroups: [{ id: 'g1', headNodeId: 'a', tailNodeId: 'b' }] as never });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('File'));
    fireEvent.click(screen.getByText('Save'));
    await resolveDialog('seg-graph');
    await waitFor(() =>
      expect(mockedRest.saveGraph).toHaveBeenCalledWith(
        expect.objectContaining({ segmentGroups: [{ id: 'g1', headNodeId: 'a', tailNodeId: 'b' }] }),
      ),
    );
  });

  it('Save: warns before overwriting a DIFFERENT existing graph and aborts on cancel', async () => {
    // A saved graph "Existing" (file "existing") is present; the tab is not
    // bound to it (currentGraphFile null), so saving as "existing" collides.
    mockedRest.listGraphs.mockResolvedValue([{ name: 'Existing', file: 'existing' }] as never);
    setActiveTab({ currentGraphFile: null });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('File'));
    fireEvent.click(screen.getByText('Save'));
    await resolveDialog('existing');   // prompt: name sanitizes to 'existing' -> collides
    await resolveDialog(false);         // decline the overwrite confirm
    expect(mockedRest.saveGraph).not.toHaveBeenCalled();
  });

  it('Save: overwrite confirmed proceeds to saveGraph', async () => {
    mockedRest.saveGraph.mockResolvedValueOnce({} as never);
    mockedRest.listGraphs.mockResolvedValue([{ name: 'Existing', file: 'existing' }] as never);
    setActiveTab({ currentGraphFile: null });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('File'));
    fireEvent.click(screen.getByText('Save'));
    await resolveDialog('existing');   // prompt
    await resolveDialog(true);          // confirm overwrite
    await waitFor(() =>
      expect(mockedRest.saveGraph).toHaveBeenCalledWith(expect.objectContaining({ name: 'existing' })),
    );
  });

  it('Save: re-saving the currently-open graph does NOT warn', async () => {
    mockedRest.saveGraph.mockResolvedValueOnce({} as never);
    mockedRest.listGraphs.mockResolvedValue([{ name: 'Existing', file: 'existing' }] as never);
    // Tab is already bound to "existing" -> re-saving it is silent.
    setActiveTab({ currentGraphFile: 'existing' });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('File'));
    fireEvent.click(screen.getByText('Save'));
    await resolveDialog('existing');   // only the prompt; no overwrite confirm
    await waitFor(() =>
      expect(mockedRest.saveGraph).toHaveBeenCalledWith(expect.objectContaining({ name: 'existing' })),
    );
  });

  // ── Project-mode Save / Save As (delegated through saveActiveGraph -- ID9) ──

  it('Save (project mode, bound): overwrites the bound file in place, no prompt', async () => {
    useProjectStore.setState({ projectDir: '/proj', projectName: 'proj', loaded: true });
    mockedRest.saveGraph.mockResolvedValueOnce({} as never);
    setActiveTab({ currentGraphFile: 'bound-graph' });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('File'));
    fireEvent.click(screen.getByText('Save'));
    await waitFor(() =>
      expect(mockedRest.saveGraph).toHaveBeenCalledWith(expect.objectContaining({ name: 'bound-graph' })),
    );
    // No dialog was ever opened for the in-place overwrite.
    expect(useDialogStore.getState().active).toBeNull();
  });

  it('Save As (project mode, bound): still prompts, saving under the entered name', async () => {
    useProjectStore.setState({ projectDir: '/proj', projectName: 'proj', loaded: true });
    mockedRest.saveGraph.mockResolvedValueOnce({} as never);
    setActiveTab({ currentGraphFile: 'bound-graph' });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('File'));
    fireEvent.click(screen.getByText('Save As...'));
    await resolveDialog('bound-graph-copy');
    await waitFor(() =>
      expect(mockedRest.saveGraph).toHaveBeenCalledWith(expect.objectContaining({ name: 'bound-graph-copy' })),
    );
  });

  // ── Clear action ────────────────────────────────────────────────────

  it('Clear: confirmed clears the canvas', async () => {
    setActiveTab({
      nodes: [{ id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { type: 'Add', params: {} } }],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('File'));
    fireEvent.click(screen.getByText('Clear Canvas'));
    await resolveDialog(true);
    await waitFor(() => expect(useTabStore.getState().tabs[0].nodes).toHaveLength(0));
  });

  it('Clear: cancelled leaves the canvas intact', async () => {
    setActiveTab({
      nodes: [{ id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { type: 'Add', params: {} } }],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('File'));
    fireEvent.click(screen.getByText('Clear Canvas'));
    await resolveDialog(false);
    expect(useTabStore.getState().tabs[0].nodes).toHaveLength(1);
  });

  // ── Load submenu ────────────────────────────────────────────────────

  it('Load: shows loading then empty when no saved graphs', async () => {
    let resolveList: (v: unknown) => void = () => {};
    mockedRest.listGraphs.mockReturnValueOnce(
      new Promise((res) => {
        resolveList = res;
      }) as never,
    );
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Load'));
    expect(screen.getByText('Loading...')).toBeInTheDocument();
    await act(async () => {
      resolveList([]);
    });
    await waitFor(() => expect(screen.getByText('No saved graphs')).toBeInTheDocument());
  });

  it('Load: lists graphs and loading one resolves nodes/edges', async () => {
    mockedRest.listGraphs.mockResolvedValueOnce([
      { name: 'Alpha', file: 'alpha.json' },
      { name: 'Beta', file: 'beta.json' },
    ] as never);
    mockedRest.loadGraph.mockResolvedValueOnce({
      nodes: [{ id: 'n1', type: 'Add', position: { x: 0, y: 0 }, data: { params: {} } }],
      edges: [],
      presets: [{ preset_name: 'P1' }],
    } as never);
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Load'));
    await waitFor(() => expect(screen.getByText('Alpha')).toBeInTheDocument());
    expect(screen.getByText('Beta')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Alpha'));
    await waitFor(() => expect(mockedRest.loadGraph).toHaveBeenCalledWith('alpha.json'));
    // savedPresets length > 0 -> presets merged into nodeDefStore
    await waitFor(() =>
      expect(useNodeDefStore.getState().presets.some((p) => p.preset_name === 'P1')).toBe(true),
    );
  });

  it('Load: a graph with no presets / null nodes uses fallbacks and does not write presets', async () => {
    mockedRest.listGraphs.mockResolvedValueOnce([{ name: 'Gamma', file: 'gamma.json' }] as never);
    // nodes/edges/presets all absent -> ?? [] fallbacks; presets not array
    mockedRest.loadGraph.mockResolvedValueOnce({} as never);
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Load'));
    await waitFor(() => expect(screen.getByText('Gamma')).toBeInTheDocument());
    fireEvent.click(screen.getByText('Gamma'));
    await waitFor(() => expect(mockedRest.loadGraph).toHaveBeenCalled());
    expect(useNodeDefStore.getState().presets).toHaveLength(0);
  });

  it('Load: merges presets, skipping ones that already exist (some() predicate)', async () => {
    // Pre-seed an existing preset so the merge predicate's callback runs and
    // the "already exists" (true) branch is exercised, plus a new one (false).
    useNodeDefStore.setState({ presets: [{ preset_name: 'Existing' } as never] });
    mockedRest.listGraphs.mockResolvedValueOnce([{ name: 'G', file: 'g.json' }] as never);
    mockedRest.loadGraph.mockResolvedValueOnce({
      nodes: [],
      edges: [],
      presets: [{ preset_name: 'Existing' }, { preset_name: 'Fresh' }],
    } as never);
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Load'));
    await waitFor(() => expect(screen.getByText('G')).toBeInTheDocument());
    fireEvent.click(screen.getByText('G'));
    await waitFor(() => {
      const names = useNodeDefStore.getState().presets.map((p) => p.preset_name);
      expect(names).toContain('Fresh');
      // Existing appears exactly once (not duplicated)
      expect(names.filter((n) => n === 'Existing')).toHaveLength(1);
    });
  });

  it('Load: a cancelled fetch (menu closed before resolve) does not set state', async () => {
    let resolveList: (v: unknown) => void = () => {};
    mockedRest.listGraphs.mockReturnValueOnce(
      new Promise((res) => {
        resolveList = res;
      }) as never,
    );
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Load'));
    expect(screen.getByText('Loading...')).toBeInTheDocument();
    // Close the menu (unmounts the panel -> cleanup sets cancelled=true)
    fireEvent.mouseDown(document.body);
    await waitFor(() => expect(screen.queryByText('Loading...')).toBeNull());
    // Resolve AFTER unmount: the `if (!cancelled)` guards take their false branch.
    await act(async () => {
      resolveList([{ name: 'Late', file: 'late.json' }]);
    });
    // Re-opening fetches afresh (default empty) and the late data never surfaced.
    expect(screen.queryByText('Late')).toBeNull();
  });

  it('Load: a cancelled fetch that rejects after unmount is swallowed', async () => {
    let rejectList: (e: unknown) => void = () => {};
    mockedRest.listGraphs.mockReturnValueOnce(
      new Promise((_res, rej) => {
        rejectList = rej;
      }) as never,
    );
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Load'));
    expect(screen.getByText('Loading...')).toBeInTheDocument();
    fireEvent.mouseDown(document.body);
    await waitFor(() => expect(screen.queryByText('Loading...')).toBeNull());
    await act(async () => {
      rejectList(new Error('late error'));
    });
    // No crash, nothing rendered.
    expect(screen.queryByText('No saved graphs')).toBeNull();
  });

  it('Load: restores description + segmentGroups and binds the graph file', async () => {
    mockedRest.listGraphs.mockResolvedValueOnce([{ name: 'Alpha', file: 'alpha' }] as never);
    mockedRest.loadGraph.mockResolvedValueOnce({
      nodes: [],
      edges: [],
      description: 'desc-abc',
      segmentGroups: [{ id: 'g1', headNodeId: 'a', tailNodeId: 'b' }],
    } as never);
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Load'));
    await waitFor(() => expect(screen.getByText('Alpha')).toBeInTheDocument());
    fireEvent.click(screen.getByText('Alpha'));
    await waitFor(() => expect(mockedRest.loadGraph).toHaveBeenCalledWith('alpha'));
    await waitFor(() => {
      const tab = useTabStore.getState().tabs[0];
      expect(tab.description).toBe('desc-abc');
      expect(tab.segmentGroups).toEqual([{ id: 'g1', headNodeId: 'a', tailNodeId: 'b' }]);
      // Bound to the loaded file so re-saving under the same name is silent.
      expect(tab.currentGraphFile).toBe('alpha');
    });
  });

  // -- Load: project-mode origin stamping (Task 13 review gap, ID10) --
  // handleLoadGraph stamps the active tab's projectOrigin with the open
  // project's dir right after a successful load (Toolbar.tsx:317-318), but
  // only while a project is open. Scoped in its own describe so the extra
  // localStorage reset doesn't touch the rest of this file's tests.
  describe('Load: project origin stamping (ID10)', () => {
    beforeEach(() => {
      localStorage.clear();
      // setActiveTab() (outer beforeEach) copies forward whatever
      // `projectOrigin` the previous test's tab was left with via `...real` --
      // pin a clean baseline here so these two tests are independent of run
      // order and of each other.
      setActiveTab({ projectOrigin: null });
    });

    afterEach(() => {
      localStorage.clear();
    });

    it('project mode: stamps the active tab projectOrigin with the open project dir', async () => {
      useProjectStore.setState({ projectDir: '/proj', projectName: 'proj', loaded: true });
      mockedRest.listGraphs.mockResolvedValueOnce([{ name: 'Alpha', file: 'alpha' }] as never);
      mockedRest.loadGraph.mockResolvedValueOnce({ nodes: [], edges: [] } as never);
      render(<Toolbar />);
      fireEvent.click(screen.getByText('Load'));
      await waitFor(() => expect(screen.getByText('Alpha')).toBeInTheDocument());
      fireEvent.click(screen.getByText('Alpha'));
      await waitFor(() => expect(mockedRest.loadGraph).toHaveBeenCalledWith('alpha'));
      await waitFor(() => {
        const tab = useTabStore.getState().tabs[0];
        expect(tab.projectOrigin).toBe('/proj');
        expect(tab.currentGraphFile).toBe('alpha');
      });
    });

    it('non-project mode: projectOrigin stays null after the same load', async () => {
      // projectDir stays null via the outer beforeEach default -- no project open.
      mockedRest.listGraphs.mockResolvedValueOnce([{ name: 'Alpha', file: 'alpha' }] as never);
      mockedRest.loadGraph.mockResolvedValueOnce({ nodes: [], edges: [] } as never);
      render(<Toolbar />);
      fireEvent.click(screen.getByText('Load'));
      await waitFor(() => expect(screen.getByText('Alpha')).toBeInTheDocument());
      fireEvent.click(screen.getByText('Alpha'));
      await waitFor(() => expect(mockedRest.loadGraph).toHaveBeenCalledWith('alpha'));
      await waitFor(() => {
        const tab = useTabStore.getState().tabs[0];
        // Load still ran to completion (proves the guard's false branch, not
        // just an untouched default) -- only projectOrigin stays unset.
        expect(tab.currentGraphFile).toBe('alpha');
        expect(tab.projectOrigin).toBeNull();
      });
    });
  });

  // ── Load: layout_missing branch (Task 11 gate; Task 12 controller item 2) ──
  //
  // Pins handleLoadGraph's project-mode fallback: a `layout_missing: true`
  // response runs the nodes through autoLayout + stackUnboundNotes (instead
  // of using the file's stored positions) with NO toast and NO undo-stack
  // push (setNodes is called directly, bypassing applyLayout). Black-box
  // against handleLoadGraph's public behavior only -- no internal spies.

  it('Load: layout_missing true auto-lays-out nodes and stacks unbound notes, with no toast/undo push', async () => {
    mockedRest.listGraphs.mockResolvedValueOnce([{ name: 'Proj', file: 'proj' }] as never);
    mockedRest.loadGraph.mockResolvedValueOnce({
      nodes: [
        // 9999,9999 is not a placement dagre's LR ranked layout would ever
        // produce for a 2-node graph -- any change proves a real layout ran,
        // independent of dagre's own coordinate convention (which may start
        // a rank at x=0, indistinguishable from an untouched (0,0) input).
        { id: 'n1', type: 'Add', position: { x: 9999, y: 9999 }, data: { params: {} } },
        { id: 'n2', type: 'Add', position: { x: 9999, y: 9999 }, data: { params: {} } },
        { id: 'note1', type: 'note', position: { x: 999, y: 999 }, data: {} },
      ],
      edges: [{ id: 'e1', source: 'n1', target: 'n2', sourceHandle: '', targetHandle: '' }],
      layout_missing: true,
    } as never);
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Load'));
    await waitFor(() => expect(screen.getByText('Proj')).toBeInTheDocument());
    fireEvent.click(screen.getByText('Proj'));
    await waitFor(() => expect(mockedRest.loadGraph).toHaveBeenCalledWith('proj'));

    await waitFor(() => {
      const tab = useTabStore.getState().tabs[0];
      const n1 = tab.nodes.find((n) => n.id === 'n1')!;
      const n2 = tab.nodes.find((n) => n.id === 'n2')!;
      const note = tab.nodes.find((n) => n.id === 'note1')!;
      // dagre actually laid the connected pair out (positions diverge from
      // the identical (9999,9999)/(9999,9999) the "file" supplied, and from
      // each other -- LR rank order puts n1/n2 at different x).
      expect(n1.position).not.toEqual({ x: 9999, y: 9999 });
      expect(n2.position).not.toEqual({ x: 9999, y: 9999 });
      expect(n1.position.x).not.toBe(n2.position.x);
      // stackUnboundNotes deterministically places the lone unbound note.
      expect(note.position).toEqual({ x: -320, y: 0 });
    });
    // No success toast and no undo snapshot from the layout_missing path.
    expect(useToastStore.getState().toasts).toHaveLength(0);
    expect(useTabStore.getState().tabs[0].undoStack).toHaveLength(0);
  });

  it('Load: layout_missing absent (or false) keeps the file positions unchanged', async () => {
    mockedRest.listGraphs.mockResolvedValueOnce([{ name: 'Proj', file: 'proj' }] as never);
    mockedRest.loadGraph.mockResolvedValueOnce({
      nodes: [
        { id: 'n1', type: 'Add', position: { x: 123, y: 456 }, data: { params: {} } },
        { id: 'note1', type: 'note', position: { x: 50, y: 50 }, data: {} },
      ],
      edges: [],
      // layout_missing intentionally omitted
    } as never);
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Load'));
    await waitFor(() => expect(screen.getByText('Proj')).toBeInTheDocument());
    fireEvent.click(screen.getByText('Proj'));
    await waitFor(() => expect(mockedRest.loadGraph).toHaveBeenCalledWith('proj'));

    await waitFor(() => {
      const tab = useTabStore.getState().tabs[0];
      expect(tab.nodes.find((n) => n.id === 'n1')!.position).toEqual({ x: 123, y: 456 });
      // Unbound note is untouched too -- stackUnboundNotes never runs on this branch.
      expect(tab.nodes.find((n) => n.id === 'note1')!.position).toEqual({ x: 50, y: 50 });
    });
    expect(useToastStore.getState().toasts).toHaveLength(0);
    expect(useTabStore.getState().tabs[0].undoStack).toHaveLength(0);
  });

  it('Load: loadGraph rejection toasts an error', async () => {
    mockedRest.listGraphs.mockResolvedValueOnce([{ name: 'Bad', file: 'bad.json' }] as never);
    mockedRest.loadGraph.mockRejectedValueOnce(new Error('404'));
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Load'));
    await waitFor(() => expect(screen.getByText('Bad')).toBeInTheDocument());
    fireEvent.click(screen.getByText('Bad'));
    await waitFor(() =>
      expect(useToastStore.getState().toasts.some((t) => t.type === 'error' && t.message.includes('404'))).toBe(true),
    );
  });

  it('Load: listGraphs rejecting yields the empty state', async () => {
    mockedRest.listGraphs.mockRejectedValueOnce(new Error('nope'));
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Load'));
    await waitFor(() => expect(screen.getByText('No saved graphs')).toBeInTheDocument());
  });

  it('Load: listGraphs returning a non-array falls back to []', async () => {
    mockedRest.listGraphs.mockResolvedValueOnce({ not: 'an array' } as never);
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Load'));
    await waitFor(() => expect(screen.getByText('No saved graphs')).toBeInTheDocument());
  });

  it('Load: Import button triggers the hidden file input click', async () => {
    mockedRest.listGraphs.mockResolvedValueOnce([] as never);
    const inputClick = vi.spyOn(HTMLInputElement.prototype, 'click').mockImplementation(() => {});
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Load'));
    await waitFor(() => expect(screen.getByText('Import JSON...')).toBeInTheDocument());
    fireEvent.click(screen.getByText('Import JSON...'));
    expect(inputClick).toHaveBeenCalled();
  });

  it('Load: closes on outside mousedown', async () => {
    mockedRest.listGraphs.mockResolvedValueOnce([] as never);
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Load'));
    await waitFor(() => expect(screen.getByText('Import JSON...')).toBeInTheDocument());
    fireEvent.mouseDown(document.body);
    await waitFor(() => expect(screen.queryByText('Import JSON...')).toBeNull());
  });

  it('Load: mousedown inside the submenu keeps it open', async () => {
    mockedRest.listGraphs.mockResolvedValueOnce([] as never);
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Load'));
    await waitFor(() => expect(screen.getByText('Import JSON...')).toBeInTheDocument());
    fireEvent.mouseDown(screen.getByText('Import JSON...'));
    expect(screen.getByText('Import JSON...')).toBeInTheDocument();
  });

  // ── Import file (handleImportFile) ──────────────────────────────────

  function fileInput(): HTMLInputElement {
    return document.querySelector('input[type="file"][accept=".json"]') as HTMLInputElement;
  }

  it('Import: no file selected is a no-op', () => {
    render(<Toolbar />);
    fireEvent.change(fileInput(), { target: { files: [] } });
    // nothing thrown, nodes unchanged
    expect(useTabStore.getState().tabs[0].nodes).toHaveLength(0);
  });

  it('Import: valid JSON sets nodes/edges and merges presets', async () => {
    render(<Toolbar />);
    const payload = JSON.stringify({
      nodes: [{ id: 'n1', type: 'Add', position: { x: 0, y: 0 }, data: { params: {} } }],
      edges: [],
      presets: [{ preset_name: 'ImpPreset' }],
    });
    const file = new File([payload], 'graph.json', { type: 'application/json' });
    fireEvent.change(fileInput(), { target: { files: [file] } });
    await waitFor(() =>
      expect(useNodeDefStore.getState().presets.some((p) => p.preset_name === 'ImpPreset')).toBe(true),
    );
    expect(fileInput().value).toBe('');
  });

  it('Import: JSON without presets / missing arrays uses fallbacks, no preset write', async () => {
    render(<Toolbar />);
    const payload = JSON.stringify({}); // nodes/edges absent -> ?? [] ; presets not array
    const file = new File([payload], 'g.json', { type: 'application/json' });
    fireEvent.change(fileInput(), { target: { files: [file] } });
    // Wait a tick for FileReader.onload
    await waitFor(() => expect(useTabStore.getState().tabs[0].nodes).toHaveLength(0));
    expect(useNodeDefStore.getState().presets).toHaveLength(0);
  });

  it('Import: merges presets, skipping ones that already exist', async () => {
    useNodeDefStore.setState({ presets: [{ preset_name: 'Existing' } as never] });
    render(<Toolbar />);
    const payload = JSON.stringify({
      nodes: [],
      edges: [],
      presets: [{ preset_name: 'Existing' }, { preset_name: 'NewOne' }],
    });
    const file = new File([payload], 'g.json', { type: 'application/json' });
    fireEvent.change(fileInput(), { target: { files: [file] } });
    await waitFor(() => {
      const names = useNodeDefStore.getState().presets.map((p) => p.preset_name);
      expect(names).toContain('NewOne');
      expect(names.filter((n) => n === 'Existing')).toHaveLength(1);
    });
  });

  it('Import: nodes not an array throws "Invalid graph format" -> error toast', async () => {
    render(<Toolbar />);
    const payload = JSON.stringify({ nodes: 'oops', edges: [] });
    const file = new File([payload], 'g.json', { type: 'application/json' });
    fireEvent.change(fileInput(), { target: { files: [file] } });
    await waitFor(() =>
      expect(useToastStore.getState().toasts.some((t) => t.type === 'error')).toBe(true),
    );
  });

  it('Import: malformed JSON triggers the catch -> error toast', async () => {
    render(<Toolbar />);
    const file = new File(['{not valid json'], 'g.json', { type: 'application/json' });
    fireEvent.change(fileInput(), { target: { files: [file] } });
    await waitFor(() =>
      expect(
        useToastStore.getState().toasts.some((t) => t.type === 'error' && t.message.includes('Import failed')),
      ).toBe(true),
    );
  });

  // -- Import: readOnly handling (ID8 fast-follow, task 16 review Adjudication B) --
  // handleImportFile previously never touched `readOnly`: importing an
  // ordinary file into a tab that had loaded a too-new graph left it stuck
  // read-only forever (over-blocking), while importing a NEWER-format file
  // into a normal tab left it editable (the same lossy-copy hazard Ruling A
  // documented for loads, but unguarded on the import path). Scoped in its
  // own describe, like the sibling "Load: project origin stamping" block
  // above, because `setActiveTab()` (outer beforeEach) copies `readOnly`
  // forward from whatever the previous test's tab was left with via
  // `...real` -- pin a clean baseline here and reset after so later tests
  // in this file can't inherit a stuck read-only flag from run order.
  describe('Import: readOnly handling (ID8 fast-follow)', () => {
    beforeEach(() => {
      setActiveTab({ readOnly: false, projectOrigin: null });
    });

    afterEach(() => {
      useTabStore.getState().setTabReadOnly(false);
    });

    it('importing a current-format file into a read-only tab clears readOnly and Save is no longer refused', async () => {
      mockedRest.saveGraph.mockResolvedValueOnce({} as never);
      setActiveTab({ readOnly: true, projectOrigin: null });
      render(<Toolbar />);

      const payload = JSON.stringify({ nodes: [], edges: [], format_version: 1 });
      const file = new File([payload], 'current.json', { type: 'application/json' });
      fireEvent.change(fileInput(), { target: { files: [file] } });

      await waitFor(() => expect(useTabStore.getState().tabs[0].readOnly).toBe(false));

      // Save is no longer refused: saveGraph is actually reached (the
      // read-only guard in saveActiveGraph would otherwise short-circuit
      // before ever calling it).
      fireEvent.click(screen.getByText('File'));
      fireEvent.click(screen.getByText('Save'));
      await resolveDialog('now-editable');
      await waitFor(() =>
        expect(mockedRest.saveGraph).toHaveBeenCalledWith(expect.objectContaining({ name: 'now-editable' })),
      );
    });

    it('importing a current-format file into an already-editable tab leaves readOnly false', async () => {
      render(<Toolbar />);
      const payload = JSON.stringify({ nodes: [], edges: [], format_version: 1 });
      const file = new File([payload], 'current.json', { type: 'application/json' });
      fireEvent.change(fileInput(), { target: { files: [file] } });
      await waitFor(() => expect(fileInput().value).toBe(''));
      expect(useTabStore.getState().tabs[0].readOnly).toBe(false);
    });

    it('importing a newer-format file into a normal tab sets readOnly and toasts a warning', async () => {
      render(<Toolbar />);
      const payload = JSON.stringify({ nodes: [], edges: [], format_version: 999 });
      const file = new File([payload], 'newer.json', { type: 'application/json' });
      fireEvent.change(fileInput(), { target: { files: [file] } });

      await waitFor(() => expect(useTabStore.getState().tabs[0].readOnly).toBe(true));
      expect(
        useToastStore.getState().toasts.some((t) => t.type === 'warning' && t.message.includes('v999')),
      ).toBe(true);
    });
  });

  // ── Export menu actions ─────────────────────────────────────────────

  it('Export JSON: empty canvas warns', () => {
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export as JSON'));
    expect(useToastStore.getState().toasts.some((t) => t.type === 'warning')).toBe(true);
    expect(URL.createObjectURL).not.toHaveBeenCalled();
  });

  it('Export JSON: with nodes downloads a blob', () => {
    setActiveTab({
      name: 'My Graph!!',
      nodes: [{ id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { type: 'Add', params: {} } }],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export as JSON'));
    expect(URL.createObjectURL).toHaveBeenCalled();
    expect(URL.revokeObjectURL).toHaveBeenCalled();
  });

  it('Export JSON: uses "graph" fallback when the tab name is empty', () => {
    setActiveTab({
      name: '',
      nodes: [{ id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { type: 'Add', params: {} } }],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export as JSON'));
    expect(URL.createObjectURL).toHaveBeenCalled();
  });

  it('Export JSON: strips SECRET param values from the downloaded document', async () => {
    const definition = {
      node_name: 'LLMChat', category: 'LLM', description: '', inputs: [], outputs: [],
      params: [
        { name: 'openai_api_key', param_type: 'secret', default: '', description: '', options: [], min_value: null, max_value: null },
      ],
    };
    setActiveTab({
      description: 'exported',
      nodes: [
        { id: 'n1', type: 'baseNode', position: { x: 1.6, y: 2.4 }, data: { label: 'LLM', type: 'LLMChat', params: { openai_api_key: 'sk-secret' }, definition } },
      ],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export as JSON'));
    expect(URL.createObjectURL).toHaveBeenCalled();
    const blob = (URL.createObjectURL as unknown as ReturnType<typeof vi.fn>).mock.calls[0][0] as Blob;
    // jsdom's Blob has no .text(); read it via FileReader (same path the
    // import flow uses).
    const text = await new Promise<string>((resolve, reject) => {
      const fr = new FileReader();
      fr.onload = () => resolve(String(fr.result));
      fr.onerror = () => reject(fr.error);
      fr.readAsText(blob);
    });
    const doc = JSON.parse(text);
    // Secret blanked; description carried; position rounded.
    expect(doc.nodes[0].data.params.openai_api_key).toBe('');
    expect(doc.description).toBe('exported');
    expect(doc.nodes[0].position).toEqual({ x: 2, y: 2 });
  });

  it('Export Subgraph: empty canvas warns', () => {
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export as Subgraph'));
    expect(useToastStore.getState().toasts.some((t) => t.type === 'warning')).toBe(true);
  });

  it('Export Subgraph: blank prompt name aborts', async () => {
    setActiveTab({
      nodes: [{ id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { type: 'Add', params: {} } }],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export as Subgraph'));
    await resolveDialog('  ');
    expect(mockedRest.createPreset).not.toHaveBeenCalled();
  });

  it('Export Subgraph: success calls createPreset + fetchDefinitions + toast', async () => {
    mockedRest.createPreset.mockResolvedValueOnce({} as never);
    const fetchDefinitions = vi.fn().mockResolvedValue(undefined);
    useNodeDefStore.setState({ fetchDefinitions });
    setActiveTab({
      nodes: [{ id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { type: 'Add', params: {} } }],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export as Subgraph'));
    await resolveDialog('my-preset');
    await waitFor(() =>
      expect(mockedRest.createPreset).toHaveBeenCalledWith(expect.objectContaining({ name: 'my-preset' })),
    );
    await waitFor(() => expect(fetchDefinitions).toHaveBeenCalled());
    await waitFor(() =>
      expect(useToastStore.getState().toasts.some((t) => t.type === 'success')).toBe(true),
    );
  });

  it('Export Subgraph: createPreset rejection toasts error', async () => {
    mockedRest.createPreset.mockRejectedValueOnce(new Error('dup name'));
    setActiveTab({
      nodes: [{ id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { type: 'Add', params: {} } }],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export as Subgraph'));
    await resolveDialog('p');
    await waitFor(() =>
      expect(useToastStore.getState().toasts.some((t) => t.type === 'error' && t.message.includes('dup name'))).toBe(true),
    );
  });

  it('Export Python: empty canvas warns', () => {
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export as Python'));
    expect(useToastStore.getState().toasts.some((t) => t.type === 'warning')).toBe(true);
  });

  it('Export Python: success downloads the script', async () => {
    mockedRest.exportGraph.mockResolvedValueOnce({ script: 'print(1)' });
    setActiveTab({
      nodes: [{ id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { type: 'Add', params: {} } }],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export as Python'));
    await waitFor(() => expect(mockedRest.exportGraph).toHaveBeenCalled());
    await waitFor(() => expect(URL.createObjectURL).toHaveBeenCalled());
  });

  it('Export Python: uses "graph" fallback when tab name empty', async () => {
    mockedRest.exportGraph.mockResolvedValueOnce({ script: 'x' });
    setActiveTab({
      name: '',
      nodes: [{ id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { type: 'Add', params: {} } }],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export as Python'));
    await waitFor(() => expect(mockedRest.exportGraph).toHaveBeenCalledWith(expect.anything(), expect.anything(), 'graph'));
  });

  it('Export Python: exportGraph rejection toasts error', async () => {
    mockedRest.exportGraph.mockRejectedValueOnce(new Error('compile error'));
    setActiveTab({
      nodes: [{ id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { type: 'Add', params: {} } }],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export as Python'));
    await waitFor(() =>
      expect(useToastStore.getState().toasts.some((t) => t.type === 'error' && t.message.includes('compile error'))).toBe(true),
    );
  });

  // ── Export Diagram (SVG / PNG architecture) ─────────────────────────

  it('Export Diagram: empty canvas warns and does not download', () => {
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export Diagram (SVG)'));
    expect(useToastStore.getState().toasts.some((t) => t.type === 'warning')).toBe(true);
    expect(URL.createObjectURL).not.toHaveBeenCalled();
  });

  it('Export Diagram: a canvas with only notes warns (notes are not architecture)', () => {
    setActiveTab({
      nodes: [
        { id: 'note1', type: 'noteNode', position: { x: 0, y: 0 }, data: { type: 'note', params: {} } },
      ],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export Diagram (PNG)'));
    expect(useToastStore.getState().toasts.some((t) => t.type === 'warning')).toBe(true);
    expect(URL.createObjectURL).not.toHaveBeenCalled();
  });

  it('Export Diagram (SVG): with nodes downloads an SVG blob', () => {
    setActiveTab({
      name: 'My Graph!!',
      nodes: [
        { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'Add', type: 'Add', params: {} } },
        { id: 'n2', type: 'baseNode', position: { x: 300, y: 0 }, data: { label: 'ReLU', type: 'ReLU', params: {} } },
      ],
      edges: [{ id: 'e1', source: 'n1', target: 'n2', style: { stroke: '#4CAF50' } }],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export Diagram (SVG)'));
    expect(URL.createObjectURL).toHaveBeenCalled();
    expect(URL.revokeObjectURL).toHaveBeenCalled();
    expect(mockedExportDiagram.svgToPngBlob).not.toHaveBeenCalled();
  });

  it('Export Diagram (SVG): uses the "graph" filename fallback when the tab name is empty', () => {
    setActiveTab({
      name: '',
      nodes: [
        { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'Add', type: 'Add', params: {} } },
      ],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export Diagram (SVG)'));
    expect(URL.createObjectURL).toHaveBeenCalled();
  });

  it('Export Diagram (PNG): rasterizes the SVG and downloads a PNG blob', async () => {
    setActiveTab({
      nodes: [
        { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'Add', type: 'Add', params: {} } },
      ],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export Diagram (PNG)'));
    await waitFor(() => expect(mockedExportDiagram.svgToPngBlob).toHaveBeenCalled());
    await waitFor(() => expect(URL.createObjectURL).toHaveBeenCalled());
  });

  it('Export Diagram (PNG): a rasterization failure toasts an error', async () => {
    mockedExportDiagram.svgToPngBlob.mockRejectedValueOnce(new Error('canvas boom'));
    setActiveTab({
      nodes: [
        { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'Add', type: 'Add', params: {} } },
      ],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export Diagram (PNG)'));
    await waitFor(() =>
      expect(
        useToastStore.getState().toasts.some((t) => t.type === 'error' && t.message.includes('canvas boom')),
      ).toBe(true),
    );
  });

  // ── Reload nodes ────────────────────────────────────────────────────

  it('Reload Nodes: success calls store.reload', async () => {
    const reload = vi.fn().mockResolvedValue(undefined);
    useNodeDefStore.setState({ reload });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Reload Nodes'));
    await waitFor(() => expect(reload).toHaveBeenCalled());
  });

  it('Reload Nodes: failure toasts error', async () => {
    const reload = vi.fn().mockRejectedValue(new Error('reload boom'));
    useNodeDefStore.setState({ reload });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Reload Nodes'));
    await waitFor(() =>
      expect(useToastStore.getState().toasts.some((t) => t.type === 'error' && t.message.includes('reload boom'))).toBe(true),
    );
  });

  // ── Custom Node Manager open/close ──────────────────────────────────

  it('Custom Nodes: opens the manager and closes it', async () => {
    mockedRest.listCustomNodes.mockResolvedValue([]);
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Custom Nodes'));
    // The manager renders a dialog-ish modal with a title from i18n.
    await waitFor(() => expect(screen.getByText('x')).toBeInTheDocument());
    fireEvent.click(screen.getByText('x'));
    await waitFor(() => expect(screen.queryByText('x')).toBeNull());
  });

  // ── Auto Layout split button + dropdown ─────────────────────────────

  it('Auto Layout main button runs layout with the last mode and persists it', () => {
    render(<Toolbar />);
    const applySpy = vi.spyOn(useTabStore.getState(), 'applyLayout');
    fireEvent.click(screen.getByText('Auto Layout'));
    expect(useUIStore.getState().lastLayoutMode).toBe('experiments');
    applySpy.mockRestore();
  });

  it('Auto Layout caret toggles the dropdown and selecting a mode applies it', () => {
    render(<Toolbar />);
    const caret = screen.getByRole('button', { name: 'Layout mode' });
    fireEvent.click(caret);
    expect(screen.getByText('Layout Experiments')).toBeInTheDocument();
    expect(screen.getByText('Layout All')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Layout All'));
    expect(useUIStore.getState().lastLayoutMode).toBe('all');
    // dropdown closes after selection
    expect(screen.queryByText('Layout Experiments')).toBeNull();
  });

  it('Auto Layout: selecting "Layout Experiments" from the dropdown applies it', () => {
    useUIStore.setState({ lastLayoutMode: 'all' });
    render(<Toolbar />);
    fireEvent.click(screen.getByRole('button', { name: 'Layout mode' }));
    fireEvent.click(screen.getByText('Layout Experiments'));
    expect(useUIStore.getState().lastLayoutMode).toBe('experiments');
  });

  it('Auto Layout: dropdown marks "Layout Selected" active when that is the last mode', () => {
    useUIStore.setState({ lastLayoutMode: 'selected' });
    setActiveTab({
      nodes: [{ id: 'n1', type: 'baseNode', selected: true, position: { x: 0, y: 0 }, data: { type: 'Add', params: {} } }],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByRole('button', { name: 'Layout mode' }));
    expect(screen.getByText('Layout Selected (1)')).toBeInTheDocument();
  });

  it('Auto Layout caret toggles closed when clicked twice', () => {
    render(<Toolbar />);
    const caret = screen.getByRole('button', { name: 'Layout mode' });
    fireEvent.click(caret);
    expect(screen.getByText('Layout All')).toBeInTheDocument();
    fireEvent.click(caret);
    expect(screen.queryByText('Layout All')).toBeNull();
  });

  it('Auto Layout: "Layout Selected" is disabled with 0 selected and clicking is a no-op', () => {
    render(<Toolbar />);
    fireEvent.click(screen.getByRole('button', { name: 'Layout mode' }));
    const selected = screen.getByText(/Layout Selected/);
    fireEvent.click(selected);
    // selectedCount 0 -> runLayout('selected') NOT called -> mode unchanged
    expect(useUIStore.getState().lastLayoutMode).toBe('experiments');
    // dropdown stays open (runLayout not invoked, so it didn't close)
    expect(screen.getByText('Layout Experiments')).toBeInTheDocument();
  });

  it('Auto Layout: "Layout Selected" applies when nodes are selected', () => {
    setActiveTab({
      nodes: [
        { id: 'n1', type: 'baseNode', selected: true, position: { x: 0, y: 0 }, data: { type: 'Add', params: {} } },
        { id: 'n2', type: 'baseNode', selected: true, position: { x: 10, y: 0 }, data: { type: 'Add', params: {} } },
      ],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByRole('button', { name: 'Layout mode' }));
    fireEvent.click(screen.getByText(/Layout Selected/));
    expect(useUIStore.getState().lastLayoutMode).toBe('selected');
  });

  it('Auto Layout: dropdown highlights the active mode and reflects selected count', () => {
    useUIStore.setState({ lastLayoutMode: 'all' });
    setActiveTab({
      nodes: [{ id: 'n1', type: 'baseNode', selected: true, position: { x: 0, y: 0 }, data: { type: 'Add', params: {} } }],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByRole('button', { name: 'Layout mode' }));
    expect(screen.getByText('Layout Selected (1)')).toBeInTheDocument();
  });

  it('Auto Layout: dropdown closes on outside mousedown', () => {
    render(<Toolbar />);
    fireEvent.click(screen.getByRole('button', { name: 'Layout mode' }));
    expect(screen.getByText('Layout Experiments')).toBeInTheDocument();
    fireEvent.mouseDown(document.body);
    expect(screen.queryByText('Layout Experiments')).toBeNull();
  });

  it('Auto Layout: mousedown inside the dropdown keeps it open', () => {
    render(<Toolbar />);
    fireEvent.click(screen.getByRole('button', { name: 'Layout mode' }));
    fireEvent.mouseDown(screen.getByText('Layout Experiments'));
    expect(screen.getByText('Layout Experiments')).toBeInTheDocument();
  });

  // ── Settings popover toggle ─────────────────────────────────────────

  it('Settings: gear button toggles the popover open and closed', () => {
    render(<Toolbar />);
    const gear = screen.getByRole('button', { name: 'Settings' });
    expect(gear).toHaveAttribute('aria-expanded', 'false');
    fireEvent.click(gear);
    expect(gear).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    fireEvent.click(gear);
    expect(gear).toHaveAttribute('aria-expanded', 'false');
  });

  it('Settings: the popover closing itself (Escape) drives the parent onClose', () => {
    render(<Toolbar />);
    const gear = screen.getByRole('button', { name: 'Settings' });
    fireEvent.click(gear);
    expect(gear).toHaveAttribute('aria-expanded', 'true');
    // SettingsPopover's own Escape handler invokes the onClose prop (line 586).
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(gear).toHaveAttribute('aria-expanded', 'false');
  });

  // ── Help button ─────────────────────────────────────────────────────

  it('Help button toggles the shortcuts modal in the UI store', () => {
    render(<Toolbar />);
    expect(useUIStore.getState().shortcutsModalOpen).toBe(false);
    fireEvent.click(screen.getByRole('button', { name: 'Keyboard Shortcuts' }));
    expect(useUIStore.getState().shortcutsModalOpen).toBe(true);
  });

  // ── Font size menu ──────────────────────────────────────────────────

  it('Font size: Aa button toggles the menu and a selection updates the store', () => {
    render(<Toolbar />);
    const aa = screen.getByRole('button', { name: 'Font size' });
    expect(aa).toHaveAttribute('aria-expanded', 'false');
    fireEvent.click(aa);
    expect(aa).toHaveAttribute('aria-expanded', 'true');
    fireEvent.click(screen.getByText('Large'));
    expect(useUIStore.getState().fontSize).toBe('large');
    // menu closed after selection
    expect(aa).toHaveAttribute('aria-expanded', 'false');
  });

  // ── Language menu ───────────────────────────────────────────────────

  it('Language: shows current locale label and lists options', () => {
    render(<Toolbar />);
    const langBtn = screen.getByRole('button', { name: 'Language' });
    expect(langBtn).toHaveTextContent('EN');
    fireEvent.click(langBtn);
    expect(screen.getByText('English')).toBeInTheDocument();
    expect(screen.getByText('繁體中文')).toBeInTheDocument();
    // active option shows a check mark
    expect(screen.getByText('✓')).toBeInTheDocument();
  });

  it('Language: selecting a different locale switches and closes the menu', () => {
    render(<Toolbar />);
    fireEvent.click(screen.getByRole('button', { name: 'Language' }));
    fireEvent.click(screen.getByText('繁體中文'));
    expect(useI18n.getState().locale).toBe('zh-TW');
    // menu closed
    expect(screen.queryByText('English')).toBeNull();
  });

  it('Language: clicking the overlay closes the menu', () => {
    render(<Toolbar />);
    fireEvent.click(screen.getByRole('button', { name: 'Language' }));
    expect(screen.getByText('English')).toBeInTheDocument();
    // The overlay is the sibling div with an onClick; it is the element right
    // before the menu panel. Grab it by class-free traversal: it has no text.
    const panel = screen.getByText('English').closest('div')!;
    const overlay = panel.parentElement!.querySelector(':scope > div') as HTMLElement;
    fireEvent.click(overlay);
    expect(screen.queryByText('English')).toBeNull();
  });

  it('Language: falls back to the raw locale code when it is unsupported', () => {
    useI18n.setState({ locale: 'fr' as never });
    render(<Toolbar />);
    expect(screen.getByRole('button', { name: 'Language' })).toHaveTextContent('fr');
  });
});
