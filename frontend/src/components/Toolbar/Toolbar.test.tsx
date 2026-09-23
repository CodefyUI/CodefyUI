import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
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
import { isAnyModalOpen } from '../../store/modalState';
import { useI18n } from '../../i18n';
import * as rest from '../../api/rest';
import * as exportDiagram from '../../utils/exportDiagram';
import { _resetDeviceOptionsForTesting } from '../../hooks/useDeviceOptions';
import { CustomNodeManagerModal } from '../CustomNodeManager/CustomNodeManager';
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
  // The real unwrapper, for the same reason: the toolbar reads a coded
  // preset-name refusal through it (#476), so a stub would make every failed
  // Export as Subgraph throw a TypeError instead of naming the broken rule.
  errorDetail: (await importOriginal<typeof import('../../api/rest')>()).errorDetail,
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
  // Used by the Custom Node Manager, which the tests below render beside the
  // toolbar the way App mounts it
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

/**
 * An element that stops `mousedown` from bubbling, as React Flow's pane does
 * (d3-zoom stops the event there), so a listener on `document` in the bubble
 * phase never hears a press on the canvas.
 */
function canvasPane(): HTMLElement {
  const pane = document.createElement('div');
  pane.addEventListener('mousedown', (e) => e.stopPropagation());
  document.body.appendChild(pane);
  return pane;
}

describe('Toolbar', () => {
  // Every test gets a fresh vi.fn() as the store's applyLayout, installed
  // with setState, and the real action back afterwards. Not a vi.spyOn on
  // getState(): zustand clones the state object on every set, so a spy on it
  // outlives its restore and carries its calls into the next test.
  const realApplyLayout = useTabStore.getState().applyLayout;
  let applyLayout: ReturnType<typeof vi.fn<typeof realApplyLayout>>;

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
      customNodeManagerOpen: false,
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

    applyLayout = vi.fn<typeof realApplyLayout>();
    useTabStore.setState({ applyLayout });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    useTabStore.setState({ applyLayout: realApplyLayout });
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

  it('File menu closes on a mousedown on the canvas, which stops it from bubbling', () => {
    render(<Toolbar />);
    fireEvent.click(screen.getByText('File'));
    expect(screen.getByText('Save')).toBeInTheDocument();
    const pane = canvasPane();
    fireEvent.mouseDown(pane);
    expect(screen.queryByText('Save')).toBeNull();
    pane.remove();
  });

  // File and Export share one `openMenu`, so a listener the File menu failed
  // to remove would close Export on a press inside it. Removing a capture
  // listener takes the capture flag again.
  it('a closed File menu stops listening: a press inside Export keeps Export open', () => {
    render(<Toolbar />);
    fireEvent.click(screen.getByText('File'));
    fireEvent.click(screen.getByText('Export'));
    fireEvent.mouseDown(screen.getByText('Export as JSON'));
    expect(screen.getByText('Export as JSON')).toBeInTheDocument();
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

  /**
   * #476. `POST /api/presets/create` now refuses an unstorable name with a
   * CODED 400 -- `{detail: {code, ...fields}}`, no `message` -- so the
   * sentence the user reads is written here, in the user's language. Before
   * this the toast was `Export failed: [object Object]`: the reason was on
   * the wire and thrown away one line from the screen.
   */
  describe('Export Subgraph: a name the server will not store', () => {
    /** A coded refusal exactly as `createPreset` now throws one. */
    function refusal(status: number, detail: Record<string, unknown>) {
      return new rest.ApiError(status, String(detail.code), { detail });
    }

    /**
     * Run the export with *err* waiting, and answer with the error toast.
     *
     * `menu`/`item` are the labels to click, because the one case that runs
     * in Traditional Chinese has a Traditional Chinese toolbar.
     */
    async function exportFailureToast(
      err: unknown,
      menu = 'Export',
      item = 'Export as Subgraph',
    ): Promise<string> {
      mockedRest.createPreset.mockRejectedValueOnce(err);
      setActiveTab({
        nodes: [{ id: 'n1', type: 'baseNode', position: { x: 0, y: 0 }, data: { type: 'Add', params: {} } }],
      });
      render(<Toolbar />);
      fireEvent.click(screen.getByText(menu));
      fireEvent.click(screen.getByText(item));
      await resolveDialog('whatever the user typed');
      await waitFor(() =>
        expect(useToastStore.getState().toasts.some((tt) => tt.type === 'error')).toBe(true),
      );
      return useToastStore.getState().toasts.find((tt) => tt.type === 'error')!.message;
    }

    // Every code `routes_presets` can answer with, and the part of the
    // sentence that has to survive: the rule it broke, and -- where the
    // refusal carries one -- the character, the name or the file it is
    // about. A code that reached the toast as itself would read
    // "Export failed: name_separator".
    it.each<[Record<string, unknown>, string]>([
      [{ code: 'name_empty' }, 'cannot be blank'],
      [{ code: 'name_separator', character: '/' }, '"/"'],
      [{ code: 'name_separator', character: '\\' }, '"\\"'],
      [{ code: 'name_separator', character: ':' }, '":"'],
      [{ code: 'name_control_character', codepoint: 9 }, 'U+0009'],
      [{ code: 'name_dot_segment' }, 'dots'],
      [{ code: 'name_reserved_device', reserved: 'com1' }, 'com1'],
      [{ code: 'name_escapes_presets_dir' }, 'presets folder'],
    ])('says what is wrong with %j', async (detail, expected) => {
      const message = await exportFailureToast(refusal(400, detail));
      expect(message).toContain(expected);
      expect(message).not.toContain('[object Object]');
      expect(message).not.toContain(String(detail.code));
    });

    it('names the file a 409 collided with', async () => {
      const message = await exportFailureToast(
        refusal(409, { code: 'preset_file_exists', filename: 'llm_preset.json' }),
      );
      expect(message).toContain('llm_preset.json');
      expect(message).not.toContain('preset_file_exists');
    });

    // A code this build has never heard of -- a rule added server-side after
    // it shipped. It still has to read as a sentence, and it still has to say
    // the code, because that is the only part a bug report can carry.
    it('falls back to a sentence that names an unknown future code', async () => {
      const message = await exportFailureToast(refusal(400, { code: 'name_too_long' }));
      expect(message).toContain('name_too_long');
      expect(message).not.toContain('[object Object]');
      expect(message).toMatch(/letters, numbers/);
    });

    // A coded refusal whose field is missing (an older or partial server)
    // must not render the placeholder: `{character}` on screen is worse than
    // the generic sentence.
    it('falls back rather than printing an unfilled placeholder', async () => {
      const message = await exportFailureToast(refusal(400, { code: 'name_separator' }));
      expect(message).not.toContain('{character}');
      expect(message).toContain('name_separator');
    });

    // The refusals that were already here answer with PROSE (`{detail:
    // "Preset 'x' already exists"}`), and that prose is still what the editor
    // shows: this fix translates the coded ones and leaves the rest alone.
    it('still shows a prose detail unchanged', async () => {
      const message = await exportFailureToast(
        new rest.ApiError(409, "Preset 'Vision' already exists", {
          detail: "Preset 'Vision' already exists",
        }),
      );
      expect(message).toContain("Preset 'Vision' already exists");
    });

    it('reads in Traditional Chinese when the editor does', async () => {
      useI18n.setState({ locale: 'zh-TW' });
      const message = await exportFailureToast(
        refusal(400, { code: 'name_separator', character: '/' }),
        '匯出',
        '匯出為子圖',
      );
      expect(message).toContain('「/」');
      expect(message).toContain('子圖名稱');
    });
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
  //
  // The Custom Nodes manager issue. This button used to open a manager of the
  // toolbar's own, from a `useState` flag no store held, so the keyboard gate
  // in `modalState` could not see it: Delete and Shift+L went through to the
  // canvas behind the scrim. It raises the store flag now, and the manager is
  // the one App mounts at the root, rendered beside the toolbar here.

  it('Custom Nodes: opens the one manager, as a modal, and closing it lowers the flag', async () => {
    mockedRest.listCustomNodes.mockResolvedValue([]);
    render(
      <>
        <Toolbar />
        <CustomNodeManagerModal />
      </>,
    );
    fireEvent.click(screen.getByText('Custom Nodes'));
    expect(useUIStore.getState().customNodeManagerOpen).toBe(true);
    expect(isAnyModalOpen()).toBe(true);
    // Found by the close button's accessible name: the glyph it draws is a
    // multiplication sign, which no query should be spelling out. Counted,
    // because a toolbar that still drew a manager of its own would show two.
    const closeButtons = () =>
      screen.queryAllByRole('button', { name: 'Close custom node manager' });
    expect(closeButtons()).toHaveLength(1);
    // The manager reads its list on mount; let that land before closing it.
    await screen.findByText('No custom nodes yet. Upload a .py file to add one.');

    fireEvent.click(closeButtons()[0]);
    expect(closeButtons()).toHaveLength(0);
    expect(useUIStore.getState().customNodeManagerOpen).toBe(false);
    expect(isAnyModalOpen()).toBe(false);
  });

  it('Custom Nodes: focus moves into the manager, and Escape hands it back to the button', async () => {
    // The manager is mounted after the whole editor, so it has to take focus
    // for a keyboard user to reach it at all (the Custom Nodes manager issue).
    mockedRest.listCustomNodes.mockResolvedValue([]);
    render(
      <>
        <Toolbar />
        <CustomNodeManagerModal />
      </>,
    );
    const button = screen.getByRole('button', { name: 'Custom Nodes' });
    // A keyboard press: the button already holds focus when it is activated.
    button.focus();
    fireEvent.click(button);
    expect(document.activeElement).toBe(
      screen.getByRole('dialog', { name: 'Custom Node Manager' }),
    );
    await screen.findByText('No custom nodes yet. Upload a .py file to add one.');

    fireEvent.keyDown(document.activeElement!, { key: 'Escape' });
    expect(useUIStore.getState().customNodeManagerOpen).toBe(false);
    expect(document.activeElement).toBe(button);
  });

  // ── Auto Layout split button + dropdown ─────────────────────────────

  it('Auto Layout main button runs layout with the last mode and persists it', () => {
    render(<Toolbar />);
    fireEvent.click(screen.getByText('Auto Layout'));
    expect(applyLayout).toHaveBeenCalledWith('experiments');
    expect(useUIStore.getState().lastLayoutMode).toBe('experiments');
  });

  it('Auto Layout caret toggles the dropdown and selecting a mode applies it', () => {
    render(<Toolbar />);
    const caret = screen.getByRole('button', { name: 'Layout mode' });
    fireEvent.click(caret);
    expect(screen.getByRole('menuitem', { name: 'Layout Experiments' })).toBeInTheDocument();
    expect(screen.getByRole('menuitem', { name: 'Layout All' })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('menuitem', { name: 'Layout All' }));
    expect(applyLayout).toHaveBeenCalledTimes(1);
    expect(applyLayout).toHaveBeenCalledWith('all');
    expect(useUIStore.getState().lastLayoutMode).toBe('all');
    // dropdown closes after selection
    expect(screen.queryByRole('menu')).toBeNull();
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
    expect(caret).toHaveAttribute('aria-expanded', 'false');
    fireEvent.click(caret);
    expect(screen.getByRole('menu', { name: 'Layout mode' })).toBeInTheDocument();
    expect(caret).toHaveAttribute('aria-expanded', 'true');
    fireEvent.click(caret);
    expect(screen.queryByRole('menu')).toBeNull();
    expect(caret).toHaveAttribute('aria-expanded', 'false');
  });

  it('Auto Layout: "Layout Selected" is disabled with 0 selected and clicking is a no-op', () => {
    render(<Toolbar />);
    fireEvent.click(screen.getByRole('button', { name: 'Layout mode' }));
    const selected = screen.getByRole('menuitem', { name: 'Layout Selected (0)' });
    expect(selected).toBeDisabled();
    fireEvent.click(selected);
    // selectedCount 0 -> runLayout('selected') NOT called -> mode unchanged
    expect(applyLayout).not.toHaveBeenCalled();
    expect(useUIStore.getState().lastLayoutMode).toBe('experiments');
    // dropdown stays open (runLayout not invoked, so it didn't close)
    expect(screen.getByRole('menu', { name: 'Layout mode' })).toBeInTheDocument();
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
    const selected = screen.getByRole('menuitem', { name: 'Layout Selected (2)' });
    expect(selected).toBeEnabled();
    fireEvent.click(selected);
    expect(applyLayout).toHaveBeenCalledWith('selected');
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

  it('Auto Layout: dropdown closes on a mousedown on the canvas, which stops it from bubbling', () => {
    render(<Toolbar />);
    fireEvent.click(screen.getByRole('button', { name: 'Layout mode' }));
    expect(screen.getByRole('menu', { name: 'Layout mode' })).toBeInTheDocument();
    const pane = canvasPane();
    fireEvent.mouseDown(pane);
    expect(screen.queryByRole('menu')).toBeNull();
    pane.remove();
  });

  it('Auto Layout: mousedown inside the dropdown keeps it open', () => {
    render(<Toolbar />);
    fireEvent.click(screen.getByRole('button', { name: 'Layout mode' }));
    fireEvent.mouseDown(screen.getByText('Layout Experiments'));
    expect(screen.getByText('Layout Experiments')).toBeInTheDocument();
  });

  // #507. The menu opens below the split button and was rendered inside it,
  // and the split button's `overflow: hidden` -- there to clip its two halves
  // to the rounded border -- clipped the whole menu with them: none of the
  // three items could be seen or clicked. The items were bare divs as well,
  // which Tab never reaches.
  describe('the layout menu (#507)', () => {
    const caret = () => screen.getByRole('button', { name: 'Layout mode' });
    const menu = () => screen.getByRole('menu', { name: 'Layout mode' });
    const item = (name: string) => screen.getByRole('menuitem', { name });

    /**
     * Toolbar.module.css as a list of rules, read as text.
     *
     * Nothing else here has the stylesheet: vitest hands a CSS module over as
     * class names and applies none of its rules, so no style in jsdom is ever
     * computed from it. What the rules do on screen is the browser's half of
     * the check.
     */
    function cssRules(): { selector: string; classes: string[]; body: string }[] {
      const css = readFileSync(
        join(dirname(fileURLToPath(import.meta.url)), 'Toolbar.module.css'),
        'utf8',
      ).replace(/\/\*[\s\S]*?\*\//g, '');
      return css.split('}').map((chunk) => {
        // The last two pieces, so a rule nested in @media reads like any other.
        const [selector = '', body = ''] = chunk.split('{').slice(-2);
        return {
          selector: selector.trim(),
          classes: Array.from(selector.matchAll(/\.([A-Za-z_][\w-]*)/g), (m) => m[1]),
          body,
        };
      });
    }

    /** A selector from the stylesheet, spelled with the class names the component renders. */
    const rendered = (selector: string) =>
      selector.replace(/\.([A-Za-z_][\w-]*)/g, (_, name: string) => `.${styles[name]}`);

    it('opens as a menu of three buttons', () => {
      render(<Toolbar />);
      fireEvent.click(caret());
      const items = within(menu()).getAllByRole('menuitem');
      expect(items.map((one) => one.textContent)).toEqual([
        'Layout Experiments',
        'Layout All',
        'Layout Selected (0)',
      ]);
      // Buttons, so Tab reaches them and Enter or Space runs them.
      items.forEach((one) => expect(one.tagName).toBe('BUTTON'));
    });

    // The key that already closes the plugin overflow menu and the font size
    // menu. Found by text rather than by role, so this case is about the key
    // and nothing else.
    it('closes on Escape', () => {
      render(<Toolbar />);
      fireEvent.click(caret());
      expect(screen.getByText('Layout All')).toBeInTheDocument();
      fireEvent.keyDown(document, { key: 'Escape' });
      expect(screen.queryByText('Layout All')).toBeNull();
      expect(applyLayout).not.toHaveBeenCalled();
    });

    it('opens inside nothing that clips it', () => {
      render(<Toolbar />);
      fireEvent.click(caret());
      // The panel is the items' parent. Found by text rather than by role, so
      // this case is about the clip and nothing else.
      const panel = screen.getByText('Layout All').parentElement!;
      const rules = cssRules();
      // A parse that lost the split button's own rule would find no clip on
      // it and pass for the wrong reason.
      expect(rules.some((rule) => rule.classes.includes('splitButton'))).toBe(true);
      const clippedBy = rules
        .filter((rule) => /\boverflow(?:-[xy])?\s*:\s*(?:hidden|clip|auto|scroll)\b/.test(rule.body))
        .flatMap((rule) => rule.classes)
        .filter((name) => panel.parentElement!.closest(`.${styles[name]}`) !== null);
      expect(clippedBy).toEqual([]);
    });

    // jsdom has no pointer and no cascade, so each hover rule on the items is
    // asked whether it would match the disabled one with the pointer on it:
    // the rule's selector without `:hover`, matched against the element.
    it('dims the disabled item and never lights it up on hover', () => {
      render(<Toolbar />);
      fireEvent.click(caret());
      const disabled = item('Layout Selected (0)');
      const rules = cssRules().filter((rule) => /\.layoutDropdownItem(?![\w-])/.test(rule.selector));
      const lit = rules.filter(
        (rule) => rule.selector.includes(':hover')
          && disabled.matches(rendered(rule.selector.replace(/:hover/g, ''))),
      );
      expect(lit.map((rule) => rule.selector)).toEqual([]);
      const dims = rules.filter(
        (rule) => /\bopacity\s*:/.test(rule.body) && disabled.matches(rendered(rule.selector)),
      );
      expect(dims).toHaveLength(1);
    });

    // The menu is wider than the split button and the toolbar wraps whole
    // clusters, so no one edge suits every width: at some widths Auto Layout
    // ends a full row, and a menu hung from the split button's left edge would
    // pass the right edge of the window. jsdom lays nothing out, so these
    // cases give the split button and the menu their rects by hand.
    describe('placement', () => {
      const jsdomWidth = window.innerWidth;
      const setWidth = (px: number) =>
        Object.defineProperty(window, 'innerWidth', { value: px, configurable: true, writable: true });

      afterEach(() => {
        setWidth(jsdomWidth);
      });

      /** A window `width` wide, the split button's left edge at `left`, a 200px menu. */
      function layOut(left: number, width: number) {
        setWidth(width);
        vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
          const [x, w] = this.getAttribute('role') === 'menu' ? [0, 200]
            : this.classList.contains(styles.splitButton) ? [left, 122]
            : [0, 0];
          return { x, y: 0, left: x, top: 0, right: x + w, bottom: 30, width: w, height: 30, toJSON: () => ({}) };
        });
      }

      const fromRightEdge = () => menu().classList.contains(styles.layoutDropdownRight);

      it('opens from the split button\'s left edge when the menu fits there', () => {
        layOut(16, 1100);
        render(<Toolbar />);
        fireEvent.click(caret());
        expect(fromRightEdge()).toBe(false);
      });

      it('opens from the right edge when the left edge would carry it past the window', () => {
        layOut(966, 1100); // 966 + 200 = 1166 > 1100
        render(<Toolbar />);
        fireEvent.click(caret());
        expect(fromRightEdge()).toBe(true);
      });

      it('moves to the edge that fits when the window is resized while it is open', () => {
        layOut(966, 1300);
        render(<Toolbar />);
        fireEvent.click(caret());
        expect(fromRightEdge()).toBe(false);
        act(() => {
          setWidth(1100);
          window.dispatchEvent(new Event('resize'));
        });
        expect(fromRightEdge()).toBe(true);
        act(() => {
          setWidth(1300);
          window.dispatchEvent(new Event('resize'));
        });
        expect(fromRightEdge()).toBe(false);
      });
    });
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
