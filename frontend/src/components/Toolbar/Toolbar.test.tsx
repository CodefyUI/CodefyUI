import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act } from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { Toolbar } from './Toolbar';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useToastStore } from '../../store/toastStore';
import { useDialogStore } from '../../store/dialogStore';
import { useProjectStore } from '../../store/projectStore';
import { usePackStore } from '../../store/packStore';
import { usePluginStore } from '../../store/pluginStore';
import { useI18n } from '../../i18n';
import * as rest from '../../api/rest';
import * as exportDiagram from '../../utils/exportDiagram';
import { _resetDeviceOptionsForTesting } from '../../hooks/useDeviceOptions';
// The layout test below asserts on where the separators sit in the tree, so
// it needs the same generated class names the component renders with.
import styles from './Toolbar.module.css';

// ── Mocks ─────────────────────────────────────────────────────────────

const execute = vi.fn();
const stop = vi.fn();
vi.mock('../../hooks/useGraphExecution', () => ({
  useGraphExecution: () => ({ execute, stop }),
}));

vi.mock('../../api/rest', async (importOriginal) => ({
  // The REAL error class, not a stub: `packStore.refresh()` narrows a failed
  // catalog read with `err instanceof PackApiError`, and an undefined export
  // makes that line throw a TypeError instead of reporting the 404 an older
  // server answers with.
  PackApiError: (await importOriginal<typeof import('../../api/rest')>()).PackApiError,
  // `pluginStore.refresh()` narrows the same way, on the shared class.
  ApiError: (await importOriginal<typeof import('../../api/rest')>()).ApiError,
  // And again, for the same reason: `saveActiveGraph` tells the taken-name
  // 409 from an ordinary failure with `err instanceof GraphExistsError`, so a
  // stub here would make every failed save throw a TypeError instead (#455).
  GraphExistsError: (await importOriginal<typeof import('../../api/rest')>()).GraphExistsError,
  // Used by the toolbar's Save, through saveActiveGraph. The overwrite
  // question comes back from `saveGraph` itself now; `listGraphs` is left
  // mocked because the module is, not because the save path calls it.
  saveGraph: vi.fn(),
  listGraphs: vi.fn(),
  // Used directly by Toolbar
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
  // Used by the popover's "This Server" section (#193 item 2)
  fetchHealth: vi.fn(() =>
    Promise.resolve({
      status: 'ok', version: '2.2.0', nodes_loaded: 0, presets_loaded: 0,
      caches: {}, project: null,
    }),
  ),
  // Used by the popover's "Optional packs" row, which asks `packStore` to
  // read the catalog when it opens onto one nobody has read yet. Every case
  // here starts from a catalog that has already arrived (see beforeEach), so
  // this is the belt to that braces.
  listPacks: vi.fn(() =>
    Promise.resolve({
      packs: [],
      active_job: null,
      last_restart_job: null,
      remote_install_allowed: true,
      launch_mode: 'start',
      restart_available: false,
      gpu: null,
    }),
  ),
  // ...and the Plugins row beside it, which reads its own catalog the same
  // way and under the same guard.
  listPluginCatalog: vi.fn(() =>
    Promise.resolve({
      entries: [],
      active_job: null,
      remote_install_allowed: true,
      generation: 0,
    }),
  ),
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
    // Pin the fresh-tab defaults (tabStore.ts) BEFORE the overrides spread:
    // `...real` would otherwise copy forward whatever projectOrigin/readOnly
    // the previous test's tab was left with -- a latent cross-describe leak
    // (issue #88). Tests that need other values override them explicitly.
    projectOrigin: null,
    readOnly: false,
    // Same reason: a device one test assigned must not become the next
    // test's `settings.device` in a save or export body.
    graphDevice: null,
    // Same reason, and the sharpest of them: a successful Save stamps
    // `currentGraphFile` on the tab, and saveActiveGraph overwrites a BOUND
    // file in place with no prompt at all. Carried forward, that binding
    // quietly turns every later Save case into an in-place overwrite -- the
    // name prompt those cases wait on never opens, and they fail on a dialog
    // that was never going to appear rather than on what they assert.
    currentGraphFile: null,
    // The binding is TWO fields, and pinning one of them is pinning none: an
    // in-place save is "bound to a file AND knowing its name", so a name left
    // behind by a previous test's successful Save combines with the next
    // test's own `currentGraphFile` override into a binding neither test
    // wrote. That save then goes out under the LEAKED name, and the test
    // fails on a graph title it never mentions.
    currentGraphName: null,
    // Same reason (core#137): a test that seeds a collapsed block would
    // otherwise hand its definitions to every test that runs after it, and
    // `subgraphs` is now a positional argument of `exportGraph` — the leak
    // shows up as a neighbouring test asserting on the wrong export payload.
    subgraphs: [],
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
    useNodeDefStore.setState({ definitions: [], presets: [], categorized: {} });
    useProjectStore.setState({ projectDir: null, projectName: null, loaded: false });
    // Empty catalogs that have already ARRIVED: the settings popover only
    // asks for one nobody has read yet, so opening it here stays offline.
    usePackStore.setState({ packs: [], byId: {}, loaded: true, loading: false, job: null });
    usePluginStore.setState({ plugins: [], byId: {}, loaded: true, loading: false, job: null });
    setActiveTab();
    // The device list is cached at module level; each test fetches afresh so
    // a per-test `fetchDevices` override is what the toolbar renders.
    _resetDeviceOptionsForTesting();
    useUIStore.setState({ globalDevice: 'cpu' });

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
    // listGraphs is cleared for the same reason: saveActiveGraph reads it on
    // every Save, and the overwrite tests below assert on what came back.
    mockedRest.saveGraph.mockClear();
    mockedRest.listGraphs.mockClear();
    mockedRest.exportGraph.mockReset();

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
    expect(screen.getByText('Export')).toBeInTheDocument();
    expect(screen.getByText('Reload Nodes')).toBeInTheDocument();
    expect(screen.getByText('Custom Nodes')).toBeInTheDocument();
    expect(screen.getByText('Auto Layout')).toBeInTheDocument();
  });

  // ── Graph device select (A8) ────────────────────────────────────────

  describe('graph device select', () => {
    const MPS = { value: 'mps', label: 'Apple MPS', detail: 'Metal Performance Shaders', available: true };
    const CPU = { value: 'cpu', label: 'CPU', detail: '', available: true };
    const select = () =>
      screen.getByRole('combobox', { name: 'Device for this graph' }) as HTMLSelectElement;

    it('lists the server devices behind a "follow Settings" option that names the Settings device', async () => {
      mockedRest.fetchDevices.mockResolvedValueOnce({ default: 'mps', devices: [CPU, MPS] });
      render(<Toolbar />);
      await waitFor(() =>
        expect(within(select()).getByRole('option', { name: /Apple MPS/ })).toBeInTheDocument(),
      );
      const options = within(select()).getAllByRole('option');
      expect(options.map((o) => o.textContent)).toEqual([
        'Follow Settings (CPU)', 'CPU', 'Apple MPS',
      ]);
      expect(select().value).toBe('');
    });

    it('marks the Settings device as unserved when the server does not list it', async () => {
      useUIStore.setState({ globalDevice: 'cuda' });
      render(<Toolbar />);
      await waitFor(() => expect(mockedRest.fetchDevices).toHaveBeenCalled());
      // Still named as it is stored -- and followed by where the run really
      // lands, because this server downgrades such a run to CPU and the bare
      // "Follow Settings (cuda)" promised the opposite.
      expect(
        within(select()).getByRole('option', {
          name: 'Follow Settings (cuda → CPU)',
        }),
      ).toBeInTheDocument();
      expect(
        within(select()).queryByRole('option', { name: 'Follow Settings (cuda)' }),
      ).toBeNull();
    });

    it('choosing a device writes graphDevice; the empty option clears it', async () => {
      mockedRest.fetchDevices.mockResolvedValueOnce({ default: 'mps', devices: [CPU, MPS] });
      render(<Toolbar />);
      await waitFor(() =>
        expect(within(select()).getByRole('option', { name: /Apple MPS/ })).toBeInTheDocument(),
      );
      fireEvent.change(select(), { target: { value: 'mps' } });
      expect(useTabStore.getState().tabs[0].graphDevice).toBe('mps');
      expect(select().value).toBe('mps');
      fireEvent.change(select(), { target: { value: '' } });
      expect(useTabStore.getState().tabs[0].graphDevice).toBeNull();
    });

    it('is disabled on a read-only tab', () => {
      setActiveTab({ readOnly: true });
      render(<Toolbar />);
      expect(select()).toBeDisabled();
    });

    it('stays enabled while a run is in flight', () => {
      setActiveTab({ status: 'running' });
      render(<Toolbar />);
      expect(select()).not.toBeDisabled();
    });

    it.each(['cuda:1', 'auto'])(
      'keeps a stored %s the server does not list, as a disabled option',
      async (stored) => {
        setActiveTab({ graphDevice: stored });
        render(<Toolbar />);
        await waitFor(() => expect(mockedRest.fetchDevices).toHaveBeenCalled());
        const synthetic = within(select()).getByRole('option', { name: stored }) as HTMLOptionElement;
        expect(synthetic).toBeDisabled();
        expect(synthetic.value).toBe(stored);
        // The select keeps the file's value, so a Save keeps the assignment.
        expect(select().value).toBe(stored);
      },
    );

    it('adds no synthetic option for a stored device the server lists', async () => {
      setActiveTab({ graphDevice: 'cpu' });
      render(<Toolbar />);
      await waitFor(() => expect(mockedRest.fetchDevices).toHaveBeenCalled());
      expect(within(select()).getAllByRole('option')).toHaveLength(2);
      expect(select().value).toBe('cpu');
    });
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

  // ── Wrapped-row layout ──────────────────────────────────────────────

  it('every separator trails a cluster from inside it, so none can lead a row', () => {
    const { container } = render(<Toolbar />);
    const root = container.querySelector<HTMLElement>(`.${styles.root}`)!;

    // `.root` is the one wrapping flex container in the toolbar, and a
    // separator that is a flex item of it is free to be carried onto the
    // next line on its own: the second row then opens with a 1px rule that
    // separates nothing, and the cluster after it is indented past the row
    // above. Adding the Save icon moved the wrap point far enough to show
    // exactly that in zh-TW at around 860px. Keeping every rule inside the
    // cluster it trails makes the case unreachable at any width, and this
    // assertion is what stops the next separator from being added back as a
    // root sibling.
    const strays = Array.from(root.children).filter((el) =>
      el.classList.contains(styles.divider),
    );
    expect(strays).toHaveLength(0);

    // The line above is also satisfied by a toolbar with no separators at
    // all, which is not the invariant we mean -- so name the real one: each
    // rule that exists closes out a cluster.
    const rules = root.querySelectorAll<HTMLElement>(`.${styles.divider}`);
    expect(rules.length).toBeGreaterThan(0);
    rules.forEach((rule) => {
      const parent = rule.parentElement!;
      expect(parent.classList.contains(styles.cluster)).toBe(true);
      expect(parent.lastElementChild).toBe(rule);
    });
  });

  // ── Save icon (the one-click twin of File -> Save) ───────────────────

  describe('the Save icon', () => {
    // The button carries no text of its own, so every case here goes through
    // its accessible name. That name is unique while the menus are shut: the
    // File menu's Save is not in the DOM until the menu is opened.
    const saveIcon = () => screen.getByRole('button', { name: 'Save' });

    it('renders with an accessible name of Save', () => {
      render(<Toolbar />);
      expect(saveIcon()).toBeInTheDocument();
      // An icon-only button has nothing on screen to read, so the hover text
      // is the only label there is -- losing it leaves a blank square.
      expect(saveIcon()).toHaveAttribute('title', 'Save');
    });

    it('clicking it runs the same save as File -> Save', async () => {
      mockedRest.saveGraph.mockResolvedValueOnce({} as never);
      render(<Toolbar />);
      fireEvent.click(saveIcon());
      // Same prompt, same payload as the menu item's save above: the icon is
      // wired to handleSave itself, not to a second copy of the logic.
      await resolveDialog('my-graph');
      await waitFor(() =>
        expect(mockedRest.saveGraph).toHaveBeenCalledWith(
          expect.objectContaining({ name: 'my-graph' }),
        ),
      );
    });

    it('sits outside the File menu, so it takes one click and not two', () => {
      render(<Toolbar />);
      const icon = saveIcon();
      // Nothing has been clicked yet, so the File menu has never rendered and
      // its Save has no text node on the page -- while the icon is already a
      // button. A Save that lived only in the menu would fail this pair.
      expect(icon).toBeInTheDocument();
      expect(screen.queryByText('Save')).toBeNull();
      // And once the menu does open, the two are separate controls rather
      // than the same element found twice.
      fireEvent.click(screen.getByText('File'));
      expect(screen.getByText('Save')).not.toBe(icon);
      expect(icon).toBeInTheDocument();
    });
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
    // No assignment, no `settings` key: the saved file stays byte-identical.
    expect('settings' in mockedRest.saveGraph.mock.calls[0][0]).toBe(false);
    await waitFor(() =>
      expect(useToastStore.getState().toasts.some((t) => t.type === 'success')).toBe(true),
    );
  });

  it('Save: carries settings.device when the graph assigns one', async () => {
    mockedRest.saveGraph.mockResolvedValueOnce({} as never);
    setActiveTab({
      graphDevice: 'cuda:1',
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
        expect.objectContaining({ settings: { device: 'cuda:1' } }),
      ),
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
    // bound to it (currentGraphFile null), so the server refuses the save as
    // "existing" with the taken-name 409 rather than writing over it (#455).
    mockedRest.saveGraph.mockRejectedValueOnce(new rest.GraphExistsError('existing', 'Existing'));
    setActiveTab({ currentGraphFile: null });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('File'));
    fireEvent.click(screen.getByText('Save'));
    await resolveDialog('existing');   // prompt: the server resolves it to 'existing'
    await resolveDialog(false);         // decline the overwrite confirm
    // The refused attempt wrote nothing, and no retry follows a no.
    expect(mockedRest.saveGraph).toHaveBeenCalledTimes(1);
  });

  it('Save: overwrite confirmed proceeds to saveGraph', async () => {
    mockedRest.saveGraph.mockRejectedValueOnce(new rest.GraphExistsError('existing', 'Existing'));
    mockedRest.saveGraph.mockResolvedValueOnce({} as never);
    setActiveTab({ currentGraphFile: null });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('File'));
    fireEvent.click(screen.getByText('Save'));
    await resolveDialog('existing');   // prompt
    await resolveDialog(true);          // confirm overwrite
    await waitFor(() =>
      expect(mockedRest.saveGraph).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'existing', file: 'existing', overwrite: true }),
      ),
    );
  });

  it('Save: re-saving the currently-open graph does NOT warn', async () => {
    mockedRest.saveGraph.mockResolvedValueOnce({} as never);
    // Tab is already bound to "existing" -> re-saving it is silent. Since a
    // bound tab overwrites its own file in place, "silent" is now the whole
    // truth: no name prompt, and so no overwrite confirm either -- the only
    // graph it could collide with is the one it came from.
    setActiveTab({ currentGraphFile: 'existing', currentGraphName: 'existing' });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('File'));
    fireEvent.click(screen.getByText('Save'));
    await waitFor(() =>
      expect(mockedRest.saveGraph).toHaveBeenCalledWith(expect.objectContaining({ name: 'existing' })),
    );
    expect(useDialogStore.getState().active).toBeNull();
  });

  // ── Project-mode Save / Save As (delegated through saveActiveGraph -- ID9) ──

  it('Save (project mode, bound): overwrites the bound file in place, no prompt', async () => {
    useProjectStore.setState({ projectDir: '/proj', projectName: 'proj', loaded: true });
    mockedRest.saveGraph.mockResolvedValueOnce({} as never);
    setActiveTab({ currentGraphFile: 'bound-graph', currentGraphName: 'bound-graph' });
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
    setActiveTab({ currentGraphFile: 'bound-graph', currentGraphName: 'bound-graph' });
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

  it('Export JSON: writes settings.device when assigned and no settings key otherwise', async () => {
    const readBlob = () => {
      const calls = (URL.createObjectURL as unknown as ReturnType<typeof vi.fn>).mock.calls;
      const blob = calls[calls.length - 1][0] as Blob;
      return new Promise<string>((resolve, reject) => {
        const fr = new FileReader();
        fr.onload = () => resolve(String(fr.result));
        fr.onerror = () => reject(fr.error);
        fr.readAsText(blob);
      });
    };
    const nodes = [{ id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { type: 'Add', params: {} } }];
    setActiveTab({ nodes, graphDevice: 'mps' });
    const view = render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export as JSON'));
    expect(JSON.parse(await readBlob()).settings).toEqual({ device: 'mps' });

    view.unmount();
    setActiveTab({ nodes });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export as JSON'));
    expect('settings' in JSON.parse(await readBlob())).toBe(false);
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

  it('Export Workspace: is the last Export item, set apart by a divider', () => {
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    const item = screen.getByRole('button', { name: 'Workspace (.cduiworkspace)' });
    expect(item.title).toBe('One file with every open tab');

    // Every menu item sits in a wrapper div of its own inside the panel.
    const wrapper = item.parentElement as HTMLElement;
    const panel = wrapper.parentElement as HTMLElement;
    expect(panel.lastElementChild).toBe(wrapper);
    // `dividerAfter` is on the item ABOVE, so the rule closes that wrapper.
    expect(
      wrapper.previousElementSibling?.querySelector(`.${styles.menuDivider}`),
    ).not.toBeNull();
  });

  it('Export Workspace: downloads a file whose name ends in .cduiworkspace', () => {
    setActiveTab({
      nodes: [{ id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { type: 'Add', params: {} } }],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Workspace (.cduiworkspace)'));

    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
    const click = HTMLAnchorElement.prototype.click as unknown as ReturnType<typeof vi.fn>;
    const anchor = click.mock.contexts[0] as HTMLAnchorElement;
    expect(anchor.download).toMatch(/^workspace-\d{4}-\d{2}-\d{2}\.cduiworkspace$/);
  });

  it('Export Workspace: nothing exportable warns and downloads nothing', () => {
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Workspace (.cduiworkspace)'));
    expect(useToastStore.getState().toasts.some((t) => t.type === 'warning')).toBe(true);
    expect(URL.createObjectURL).not.toHaveBeenCalled();
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

  // core#137 review, MAJOR 2 (sibling). A preset is stored as {nodes, edges}
  // with no slot for a subgraph definition, so exporting a canvas containing
  // an instance node would have registered a preset holding a bare
  // `subgraph:<id>` node that no definition can ever accompany -- permanently
  // broken, and outliving the graph it came from. Refuse and name the block
  // rather than silently strip it out of the user's preset.
  it('Export Subgraph: refuses a canvas containing a collapsed block and names it', async () => {
    // createPreset is a shared factory mock; restoreAllMocks leaves its call
    // history alone, so the "was not called" assertion below would otherwise
    // read the earlier Export Subgraph tests' calls.
    mockedRest.createPreset.mockClear();
    setActiveTab({
      nodes: [
        { id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { type: 'Add', params: {} } },
        { id: 'inst', type: 'subgraphNode', position: { x: 0, y: 0 }, data: { type: 'subgraph:blk', params: {} } },
      ],
      subgraphs: [
        {
          id: 'blk', name: 'Encoder', description: '', nodes: [], edges: [],
          interface: { inputs: [], outputs: [], triggerTargets: [] },
        },
      ],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export as Subgraph'));
    const errors = useToastStore.getState().toasts.filter((t) => t.type === 'error');
    expect(errors).toHaveLength(1);
    // The message has to say WHICH block, or the user cannot act on it.
    expect(errors[0].message).toContain('Encoder');
    // And it must refuse before the name prompt, not after -- asking for a
    // name and then failing would be the worse UX of the two.
    expect(useDialogStore.getState().active).toBeNull();
    expect(mockedRest.createPreset).not.toHaveBeenCalled();
  });

  it('Export Subgraph: falls back to the definition id when a block is unnamed', () => {
    setActiveTab({
      nodes: [
        { id: 'inst', type: 'subgraphNode', position: { x: 0, y: 0 }, data: { type: 'subgraph:blk', params: {} } },
      ],
      subgraphs: [
        {
          id: 'blk', name: '', description: '', nodes: [], edges: [],
          interface: { inputs: [], outputs: [], triggerTargets: [] },
        },
      ],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export as Subgraph'));
    const errors = useToastStore.getState().toasts.filter((t) => t.type === 'error');
    expect(errors[0].message).toContain('blk');
  });

  it('Export Python: empty canvas warns', () => {
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export as Python'));
    expect(useToastStore.getState().toasts.some((t) => t.type === 'warning')).toBe(true);
  });

  it('Export Python: a canvas with only notes warns and does not call the API', () => {
    setActiveTab({
      nodes: [
        { id: 'note1', type: 'noteNode', position: { x: 0, y: 0 }, data: { type: 'note', params: {} } },
      ],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export as Python'));
    expect(useToastStore.getState().toasts.some((t) => t.type === 'warning')).toBe(true);
    expect(mockedRest.exportGraph).not.toHaveBeenCalled();
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

  it('Export Python: filters notes and sends embedded presets', async () => {
    mockedRest.exportGraph.mockResolvedValueOnce({ script: 'print(1)' });
    const preset = {
      preset_name: 'Portable',
      category: 'Test',
      description: '',
      tags: [],
      nodes: [],
      edges: [],
      exposed_inputs: [],
      exposed_outputs: [],
      exposed_params: [],
    };
    setActiveTab({
      nodes: [
        {
          id: 'preset1',
          type: 'baseNode',
          position: { x: 0, y: 0 },
          data: {
            type: 'preset:Portable',
            params: {},
            isPreset: true,
            presetDefinition: preset,
            internalParams: {},
          },
        },
        { id: 'note1', type: 'noteNode', position: { x: 0, y: 0 }, data: { type: 'note', params: {} } },
      ],
      edges: [
        { id: 'note-edge', source: 'preset1', target: 'note1', sourceHandle: 'x', targetHandle: 'y' },
      ],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export as Python'));
    await waitFor(() => expect(mockedRest.exportGraph).toHaveBeenCalled());
    const [nodes, edges, name, presets] = mockedRest.exportGraph.mock.calls[0];
    expect(nodes.map((node: any) => node.id)).toEqual(['preset1']);
    expect(edges).toEqual([]);
    expect(name).toBe('My Graph');
    expect(presets).toEqual([preset]);
  });

  // core#137 review, MAJOR 2. `exportGraph` was never given the definitions,
  // so the backend saw `subgraph:<id>` instance nodes it could not resolve
  // and answered 400 `Unknown subgraph: <id>` -- the whole codegen half of
  // the feature, and the documented example, unreachable from the UI.
  it('Export Python: sends subgraph definitions alongside the instance nodes', async () => {
    mockedRest.exportGraph.mockResolvedValueOnce({ script: 'print(1)' });
    const definition = {
      id: 'blk',
      name: 'Encoder',
      description: '',
      nodes: [{ id: 'inner', type: 'Add', position: { x: 0, y: 0 }, data: { params: {} } }],
      edges: [],
      interface: { inputs: [], outputs: [], triggerTargets: [] },
    };
    setActiveTab({
      nodes: [
        { id: 'inst', type: 'subgraphNode', position: { x: 0, y: 0 }, data: { type: 'subgraph:blk', params: {} } },
      ],
      subgraphs: [definition],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export as Python'));
    await waitFor(() => expect(mockedRest.exportGraph).toHaveBeenCalled());
    const call = mockedRest.exportGraph.mock.calls[0];
    expect(call[0].map((node: any) => node.type)).toEqual(['subgraph:blk']);
    expect(call[5]).toEqual([definition]);
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
    // The [] is `subgraphs` (core#137) -- a graph with no collapsed blocks
    // still sends the argument, it is just empty. The trailing `undefined`
    // is `settings`: a graph with no assigned device sends none, and the
    // request body then carries no `settings` key.
    await waitFor(() => expect(mockedRest.exportGraph).toHaveBeenCalledWith(
      expect.anything(), expect.anything(), 'graph', [], expect.anything(), [], undefined,
    ));
  });

  it('Export Python: forwards the graph settings so the script bakes in the device', async () => {
    mockedRest.exportGraph.mockResolvedValueOnce({ script: 'x' });
    setActiveTab({
      graphDevice: 'cuda:1',
      nodes: [{ id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { type: 'Add', params: {} } }],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export as Python'));
    await waitFor(() => expect(mockedRest.exportGraph).toHaveBeenCalledWith(
      expect.anything(), expect.anything(), 'My Graph', [], expect.anything(), [],
      { device: 'cuda:1' },
    ));
  });

  // core#136 review, M-6. The exported script carried no seed at all, so an
  // exported augmenting graph drew fresh entropy every invocation while the
  // docs promised the same crops every time. The tab's run settings now go
  // with the export and become the script's --seed / --deterministic
  // defaults.
  it('Export Python: sends the tab seed and determinism toggle', async () => {
    mockedRest.exportGraph.mockResolvedValueOnce({ script: 'x' });
    setActiveTab({
      seed: 4321,
      deterministic: true,
      nodes: [{ id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { type: 'Add', params: {} } }],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export as Python'));
    await waitFor(() => expect(mockedRest.exportGraph).toHaveBeenCalled());
    const run = mockedRest.exportGraph.mock.calls[0][4];
    expect(run).toEqual({ seed: 4321, deterministic: true });
  });

  it('Export Python: an unseeded tab exports without a seed', async () => {
    mockedRest.exportGraph.mockResolvedValueOnce({ script: 'x' });
    setActiveTab({
      seed: null,
      deterministic: false,
      nodes: [{ id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { type: 'Add', params: {} } }],
    });
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Export'));
    fireEvent.click(screen.getByText('Export as Python'));
    await waitFor(() => expect(mockedRest.exportGraph).toHaveBeenCalled());
    expect(mockedRest.exportGraph.mock.calls[0][4]).toEqual({
      seed: null,
      deterministic: false,
    });
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
    // The manager renders a dialog-ish modal with a title from i18n. Found
    // by the close button's accessible name: the glyph it draws is a
    // multiplication sign, which no query should be spelling out.
    const close = () => screen.queryByRole('button', { name: 'Close custom node manager' });
    await waitFor(() => expect(close()).toBeInTheDocument());
    fireEvent.click(close()!);
    await waitFor(() => expect(close()).toBeNull());
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
