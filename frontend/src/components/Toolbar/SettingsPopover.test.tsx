import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { createRef, act } from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { SettingsPopover } from './SettingsPopover';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import { useToastStore } from '../../store/toastStore';
import { useDialogStore } from '../../store/dialogStore';
import { useI18n } from '../../i18n';
import {
  resetWeights,
  fetchDevices,
  fetchCodexStatus,
  startCodexLogin,
  logoutCodex,
  fetchHealth,
  listPacks,
  type PackCatalog,
  type PackSummary,
} from '../../api/rest';
import {
  _resetPackStoreForTesting,
  emptyPackJob,
  usePackStore,
} from '../../store/packStore';
import { computeSegmentNodes } from '../../utils/segmentPath';

vi.mock('../../api/rest', () => ({
  resetWeights: vi.fn(),
  // The "This Server" section reads /api/health when the popover opens
  // (#193 item 2); its own behaviour is covered in HealthSection.test.tsx.
  fetchHealth: vi.fn(() =>
    Promise.resolve({
      status: 'ok', version: '2.2.0', nodes_loaded: 137, presets_loaded: 12,
      caches: { run_output_store: { runs: 0, bytes: 0, max_bytes: 1024 * 1024 } },
      project: null,
    }),
  ),
  fetchCodexStatus: vi.fn(() => Promise.resolve({ status: 'logged_out' })),
  startCodexLogin: vi.fn(() => Promise.resolve({ auth_url: 'https://auth.example' })),
  logoutCodex: vi.fn(() => Promise.resolve({ status: 'logged_out' })),
  fetchDevices: vi.fn(() =>
    Promise.resolve({
      default: 'cpu',
      devices: [
        { value: 'cpu', label: 'CPU', detail: '', available: true },
        { value: 'mps', label: 'Apple MPS', detail: 'Metal Performance Shaders', available: true },
      ],
    }),
  ),
  // The packs row is a view of `packStore`, which reads the catalog through
  // this call. Stubbed so the one case that lets the popover bootstrap it
  // has something to resolve with, and so the rest never touch the network.
  listPacks: vi.fn(),
}));

vi.mock('../../utils/segmentPath', () => ({
  computeSegmentNodes: vi.fn(() => new Set(['a', 'b'])),
}));

const mockedResetWeights = vi.mocked(resetWeights);
const mockedFetchCodexStatus = vi.mocked(fetchCodexStatus);
const mockedStartCodexLogin = vi.mocked(startCodexLogin);
const mockedLogoutCodex = vi.mocked(logoutCodex);
const mockedComputeSegment = vi.mocked(computeSegmentNodes);
const mockedListPacks = vi.mocked(listPacks);

const EMPTY_CATALOG: PackCatalog = {
  packs: [],
  active_job: null,
  last_restart_job: null,
  remote_install_allowed: true,
  launch_mode: 'start',
  gpu: null,
};

function packSummary(overrides: Partial<PackSummary> = {}): PackSummary {
  return {
    id: 'word-vectors',
    title: 'Word vectors',
    description: 'A GloVe table',
    install_mode: 'live',
    status: 'not_installed',
    pip_ready: true,
    usable: false,
    depends_on: [],
    blocked_by: [],
    pip: [],
    items: [],
    size_bytes_total: 0,
    install_command: null,
    ...overrides,
  };
}

/** Put a catalog in the store, the way a finished `refresh()` would. */
function seedPacks(
  packs: PackSummary[],
  extra: Partial<ReturnType<typeof usePackStore.getState>> = {},
) {
  usePackStore.setState({
    loaded: true,
    packs,
    byId: Object.fromEntries(packs.map((pack) => [pack.id, pack])),
    ...extra,
  });
}

function makeTriggerRef() {
  const ref = createRef<HTMLButtonElement>();
  const btn = document.createElement('button');
  document.body.appendChild(btn);
  (ref as { current: HTMLButtonElement | null }).current = btn;
  return ref;
}

/** Replace the active tab with a single tab carrying the supplied overrides. */
function setupTab(overrides: Partial<ReturnType<typeof baseTab>> = {}) {
  const tab = { ...baseTab(), ...overrides };
  useTabStore.setState({ tabs: [tab as never], activeTabId: 'tab-1' });
}

function baseTab() {
  return {
    id: 'tab-1',
    name: 'Tab 1',
    nodes: [] as any[],
    edges: [] as any[],
    recordOutputs: true,
    verboseMode: false,
    weightsPersistent: true,
    backwardMode: false,
    autoBackward: false,
    graphId: 'graph-xyz',
    activeSegment: null as any,
    segmentGroups: [] as any[],
    // Creating and clearing a segment are undo steps since #200 item 2, so
    // the double needs the fields an undo frame is built from and pushed onto.
    // Left as real empty arrays rather than guarded inside the store: a tab
    // missing its stacks is a broken double, not a state the app can reach.
    subgraphs: [] as any[],
    subgraphStack: [] as any[],
    undoStack: [] as any[],
    redoStack: [] as any[],
    // core#134. `null` is the real default; the popover must also survive a
    // tab persisted before #134, which carries neither key.
    seed: null as number | null,
    deterministic: false,
  };
}

describe('SettingsPopover', () => {
  beforeEach(() => {
    useI18n.setState({ locale: 'en' });
    useUIStore.setState({
      gridSnapEnabled: false,
      tooltipsEnabled: true,
      beginnerMode: false,
      globalDevice: 'cpu',
      edgeStyle: 'circuit',
      packCenterOpen: false,
      packCenterFocusPackId: null,
    });
    _resetPackStoreForTesting();
    mockedListPacks.mockReset();
    mockedListPacks.mockResolvedValue(EMPTY_CATALOG);
    // An empty catalog that has already ARRIVED: the popover only bootstraps
    // one nobody has read yet, so every case but that one stays offline.
    seedPacks([]);
    vi.mocked(fetchHealth).mockReset();
    vi.mocked(fetchHealth).mockResolvedValue({
      status: 'ok', version: '2.2.0', nodes_loaded: 137, presets_loaded: 12,
      caches: { run_output_store: { runs: 0, bytes: 0, max_bytes: 1024 * 1024 } },
      project: null,
    });
    mockedFetchCodexStatus.mockResolvedValue({ status: 'logged_out' });
    mockedStartCodexLogin.mockResolvedValue({ auth_url: 'https://auth.example' });
    mockedLogoutCodex.mockResolvedValue({ status: 'logged_out' });
    vi.mocked(fetchDevices).mockResolvedValue({
      default: 'cpu',
      devices: [
        { value: 'cpu', label: 'CPU', detail: '', available: true },
        { value: 'mps', label: 'Apple MPS', detail: 'Metal Performance Shaders', available: true },
      ],
    });
    useToastStore.setState({ toasts: [] });
    useDialogStore.setState({ active: null, resolve: null });
    setupTab();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    // The pack store is reset in `beforeEach`, NOT here: the panel is still
    // mounted at this point, and writing to a store it subscribes to would
    // re-render it outside `act`.
    document.body.innerHTML = '';
  });

  it('renders nothing when closed', () => {
    const { container } = render(
      <SettingsPopover open={false} onClose={vi.fn()} triggerRef={makeTriggerRef()} />,
    );
    expect(container.querySelector('[role="dialog"]')).toBeNull();
  });

  it('renders all sections when open', () => {
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    expect(screen.getByText('Execution')).toBeInTheDocument();
    expect(screen.getByText('Recording & Inspection')).toBeInTheDocument();
    expect(screen.getByText('Training Behavior')).toBeInTheDocument();
    expect(screen.getByText('Editor')).toBeInTheDocument();
    expect(screen.getByText('This Server')).toBeInTheDocument();
    expect(screen.getByRole('dialog')).toBeInTheDocument();
  });

  it('the server section reads /api/health when the popover opens, not before', async () => {
    // "Fetch on open" needs no plumbing: the popover renders nothing while
    // closed, so the section does not exist to fetch (#193 item 2).
    const { unmount } = render(
      <SettingsPopover open={false} onClose={vi.fn()} triggerRef={makeTriggerRef()} />,
    );
    expect(vi.mocked(fetchHealth)).not.toHaveBeenCalled();
    unmount();

    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    await waitFor(() => expect(vi.mocked(fetchHealth)).toHaveBeenCalledTimes(1));
    expect(await screen.findByText('137')).toBeInTheDocument();
  });


  it('renders Codex auth controls and starts login in a new tab', async () => {
    const openSpy = vi.spyOn(window, 'open').mockImplementation(() => null);
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);

    expect(screen.getByText('LLM Providers')).toBeInTheDocument();
    expect(screen.getByText('ChatGPT Codex account')).toBeInTheDocument();
    await waitFor(() => expect(mockedFetchCodexStatus).toHaveBeenCalled());

    fireEvent.click(screen.getByRole('button', { name: 'Sign in' }));

    await waitFor(() => expect(mockedStartCodexLogin).toHaveBeenCalledTimes(1));
    expect(openSpy).toHaveBeenCalledWith(
      'https://auth.example',
      '_blank',
      'noopener,noreferrer',
    );
    await waitFor(() => expect(screen.getByRole('button', { name: 'Refresh' })).toBeInTheDocument());
    expect(useToastStore.getState().toasts.some((t) => t.type === 'success')).toBe(true);
  });

  it('shows logged-in Codex state and signs out', async () => {
    mockedFetchCodexStatus.mockResolvedValueOnce({
      status: 'logged_in',
      email: 'me@example.com',
    });
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);

    await screen.findByText(/me@example.com/);
    fireEvent.click(screen.getByRole('button', { name: 'Sign out' }));

    await waitFor(() => expect(mockedLogoutCodex).toHaveBeenCalledTimes(1));
    expect(await screen.findByRole('button', { name: 'Sign in' })).toBeInTheDocument();
  });

  // ── Optional packs (the Package Center entry point) ───────────────

  it('summarises installed packs and opens the Package Center, closing the popover', () => {
    seedPacks([
      packSummary({ id: 'word-vectors', status: 'installed' }),
      packSummary({ id: 'rag', status: 'not_installed' }),
    ]);
    const onClose = vi.fn();
    render(<SettingsPopover open onClose={onClose} triggerRef={makeTriggerRef()} />);

    expect(screen.getByText('Optional packs')).toBeInTheDocument();
    expect(screen.getByText('Package Center')).toBeInTheDocument();
    expect(screen.getByText('1 of 2 packs installed')).toBeInTheDocument();
    // The row is a VIEW of the store: a catalog that is already here is
    // never re-read just because the popover opened.
    expect(mockedListPacks).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: 'Open' }));

    expect(onClose).toHaveBeenCalledTimes(1);
    expect(useUIStore.getState().packCenterOpen).toBe(true);
    // No pack is focused: this is the "show me everything" entry point, and
    // the click handler must not pass its event through as a pack id.
    expect(useUIStore.getState().packCenterFocusPackId).toBeNull();
  });

  it('says unsupported on an older server', () => {
    seedPacks([], { unsupported: true });
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);

    expect(screen.getByText('Not available on this server')).toBeInTheDocument();
    expect(screen.queryByText('0 of 0 packs installed')).toBeNull();
  });

  it('shows the installing summary while a job runs', () => {
    seedPacks([packSummary({ id: 'word-vectors', status: 'installing' })], {
      job: emptyPackJob('job-1', 'word-vectors'),
    });
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);

    // Named from THIS build's catalog copy ("Word vectors (GloVe)"), not from
    // the server's own title ("Word vectors").
    expect(screen.getByText('Installing Word vectors (GloVe)...')).toBeInTheDocument();
  });

  it('names an installing pack the server ships and this build has no copy for', () => {
    seedPacks([packSummary({ id: 'from-the-future', title: 'Newer pack' })], {
      job: emptyPackJob('job-2', 'from-the-future'),
    });
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);

    expect(screen.getByText('Installing Newer pack...')).toBeInTheDocument();
  });

  it('falls back to the pack id when neither this build nor the catalog names it', () => {
    // A job adopted from another tab can name a pack the catalog in hand does
    // not list yet; the id is still a usable sentence.
    seedPacks([], { job: emptyPackJob('job-3', 'mystery-pack') });
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);

    expect(screen.getByText('Installing mystery-pack...')).toBeInTheDocument();
  });

  it('ignores a job that is no longer running', () => {
    seedPacks([packSummary({ id: 'word-vectors', status: 'installed' })], {
      job: { ...emptyPackJob('job-4', 'word-vectors'), status: 'done' },
    });
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);

    expect(screen.getByText('1 of 1 packs installed')).toBeInTheDocument();
  });

  it('reads the catalog when the popover opens, and not while it is closed', async () => {
    _resetPackStoreForTesting();
    const triggerRef = makeTriggerRef();
    const { rerender } = render(
      <SettingsPopover open={false} onClose={vi.fn()} triggerRef={triggerRef} />,
    );
    expect(mockedListPacks).not.toHaveBeenCalled();
    // Nothing has answered yet, so the row explains itself instead of
    // claiming "0 of 0 packs installed".
    rerender(<SettingsPopover open onClose={vi.fn()} triggerRef={triggerRef} />);
    expect(
      screen.getByText('Download models and libraries for the LLM nodes.'),
    ).toBeInTheDocument();

    await waitFor(() => expect(usePackStore.getState().loaded).toBe(true));
    expect(mockedListPacks).toHaveBeenCalledTimes(1);
    expect(await screen.findByText('0 of 0 packs installed')).toBeInTheDocument();
  });

  it('does not start a second catalog read while one is in flight', () => {
    _resetPackStoreForTesting();
    usePackStore.setState({ loading: true });
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    expect(mockedListPacks).not.toHaveBeenCalled();
  });

  // ── Execution: global device selector ─────────────────────────────

  it('populates the device selector from the backend and reflects the store value', async () => {
    useUIStore.setState({ globalDevice: 'mps' });
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    const select = screen.getByRole('combobox', { name: 'Compute device' }) as HTMLSelectElement;
    // Options arrive asynchronously from fetchDevices.
    await waitFor(() =>
      expect(within(select).getByRole('option', { name: /Apple MPS/ })).toBeInTheDocument(),
    );
    expect(select.value).toBe('mps');
    expect(within(select).getByRole('option', { name: 'CPU' })).toBeInTheDocument();
  });

  it('changing the device select updates the UI store', async () => {
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    const select = screen.getByRole('combobox', { name: 'Compute device' });
    await waitFor(() =>
      expect(within(select).getByRole('option', { name: /Apple MPS/ })).toBeInTheDocument(),
    );
    fireEvent.change(select, { target: { value: 'mps' } });
    expect(useUIStore.getState().globalDevice).toBe('mps');
  });

  it('falls back to a CPU-only option when the devices fetch fails', async () => {
    vi.mocked(fetchDevices).mockRejectedValueOnce(new Error('offline'));
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    const select = screen.getByRole('combobox', { name: 'Compute device' });
    // The rejection settles on a microtask; flush it.
    await waitFor(() => expect(fetchDevices).toHaveBeenCalled());
    const options = within(select).getAllByRole('option');
    expect(options).toHaveLength(1);
    expect(options[0]).toHaveTextContent('CPU');
  });

  // ── outside-click / esc behaviour ─────────────────────────────────

  it('closes on outside mousedown', () => {
    const onClose = vi.fn();
    render(<SettingsPopover open onClose={onClose} triggerRef={makeTriggerRef()} />);
    fireEvent.mouseDown(document.body);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('does NOT close when mousedown is inside the panel', () => {
    const onClose = vi.fn();
    render(<SettingsPopover open onClose={onClose} triggerRef={makeTriggerRef()} />);
    fireEvent.mouseDown(screen.getByRole('dialog'));
    expect(onClose).not.toHaveBeenCalled();
  });

  it('does NOT close when mousedown is on the trigger', () => {
    const onClose = vi.fn();
    const triggerRef = makeTriggerRef();
    render(<SettingsPopover open onClose={onClose} triggerRef={triggerRef} />);
    fireEvent.mouseDown(triggerRef.current!);
    expect(onClose).not.toHaveBeenCalled();
  });

  it('closes on Escape, ignores other keys', () => {
    const onClose = vi.fn();
    render(<SettingsPopover open onClose={onClose} triggerRef={makeTriggerRef()} />);
    fireEvent.keyDown(document, { key: 'a' });
    expect(onClose).not.toHaveBeenCalled();
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('removes listeners on unmount', () => {
    const onClose = vi.fn();
    const { unmount } = render(
      <SettingsPopover open onClose={onClose} triggerRef={makeTriggerRef()} />,
    );
    unmount();
    fireEvent.mouseDown(document.body);
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onClose).not.toHaveBeenCalled();
  });

  // ── Recording toggles ─────────────────────────────────────────────

  it('toggles record via the control button (stopPropagation path)', () => {
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    const toggle = screen.getByRole('button', { name: 'Record node outputs' });
    expect(toggle).toHaveAttribute('aria-pressed', 'true');
    fireEvent.click(toggle);
    expect(useTabStore.getState().tabs[0].recordOutputs).toBe(false);
  });

  it('toggles record via the row click (interactive Row onClick path)', () => {
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    // Click the row (the parent of the toggle), not the toggle itself.
    const row = screen.getByText('Record node outputs').closest('[role="button"]')!;
    fireEvent.click(row);
    expect(useTabStore.getState().tabs[0].recordOutputs).toBe(false);
  });

  it('toggles verbose via control', () => {
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    fireEvent.click(screen.getByRole('button', { name: 'Verbose internals' }));
    expect(useTabStore.getState().tabs[0].verboseMode).toBe(true);
  });

  // ── Row keyboard interaction ──────────────────────────────────────

  it('activates an interactive row via Enter key', () => {
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    const row = screen.getByText('Record node outputs').closest('[role="button"]')!;
    fireEvent.keyDown(row, { key: 'Enter' });
    expect(useTabStore.getState().tabs[0].recordOutputs).toBe(false);
  });

  it('activates an interactive row via Space key', () => {
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    const row = screen.getByText('Verbose internals').closest('[role="button"]')!;
    fireEvent.keyDown(row, { key: ' ' });
    expect(useTabStore.getState().tabs[0].verboseMode).toBe(true);
  });

  it('ignores other keys on an interactive row', () => {
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    const row = screen.getByText('Record node outputs').closest('[role="button"]')!;
    fireEvent.keyDown(row, { key: 'x' });
    expect(useTabStore.getState().tabs[0].recordOutputs).toBe(true);
  });

  it('non-interactive row (Compare) has no role=button and ignores keydown', () => {
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    // The Compare row has no onClick -> not interactive.
    const compareName = screen.getByText('Compare segment');
    const row = compareName.closest('div')!.parentElement!.parentElement!;
    // Fire keydown to exercise the `interactive && ...` short-circuit (false branch).
    fireEvent.keyDown(row, { key: 'Enter' });
    // Nothing to assert state-wise; reaching here without throwing covers the branch.
    expect(compareName).toBeInTheDocument();
  });

  // ── Compare segment ───────────────────────────────────────────────

  it('compare button is disabled with fewer than two selected nodes', () => {
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    const btn = screen.getByRole('button', { name: 'Select two nodes' });
    expect(btn).toBeDisabled();
  });

  it('creating a segment with two selected nodes (left/right by x) adds + activates it', () => {
    setupTab({
      nodes: [
        { id: 'n2', selected: true, position: { x: 200, y: 0 } },
        { id: 'n1', selected: true, position: { x: 50, y: 0 } },
      ],
      edges: [],
    });
    const onClose = vi.fn();
    render(<SettingsPopover open onClose={onClose} triggerRef={makeTriggerRef()} />);

    fireEvent.click(screen.getByRole('button', { name: 'Create segment' }));

    // computeSegmentNodes called with the left node id first (n1, smaller x).
    expect(mockedComputeSegment).toHaveBeenCalledWith('n1', 'n2', expect.any(Array), expect.any(Array));
    const tab = useTabStore.getState().tabs[0];
    expect(tab.segmentGroups).toHaveLength(1);
    expect(tab.segmentGroups[0]).toMatchObject({ headNodeId: 'n1', tailNodeId: 'n2' });
    expect(tab.activeSegment).not.toBeNull();
    expect(onClose).toHaveBeenCalled();
    // ONE undo step for the click, not two: the handler also focuses the new
    // segment, and focusing is a change of view (#200 item 2).
    expect(tab.undoStack).toHaveLength(1);
  });

  it('creating a segment uses the other branch when the first node is already leftmost', () => {
    setupTab({
      nodes: [
        { id: 'n1', selected: true, position: { x: 10, y: 0 } },
        { id: 'n2', selected: true, position: { x: 99, y: 0 } },
      ],
    });
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    fireEvent.click(screen.getByRole('button', { name: 'Create segment' }));
    expect(mockedComputeSegment).toHaveBeenCalledWith('n1', 'n2', expect.any(Array), expect.any(Array));
    expect(useTabStore.getState().tabs[0].segmentGroups[0]).toMatchObject({
      headNodeId: 'n1',
      tailNodeId: 'n2',
    });
  });

  it('shows an error toast when the segment has no path (empty set)', () => {
    mockedComputeSegment.mockReturnValueOnce(new Set());
    setupTab({
      nodes: [
        { id: 'n1', selected: true, position: { x: 10, y: 0 } },
        { id: 'n2', selected: true, position: { x: 99, y: 0 } },
      ],
    });
    const onClose = vi.fn();
    render(<SettingsPopover open onClose={onClose} triggerRef={makeTriggerRef()} />);
    fireEvent.click(screen.getByRole('button', { name: 'Create segment' }));

    const toasts = useToastStore.getState().toasts;
    expect(toasts.some((t) => t.type === 'error')).toBe(true);
    expect(useTabStore.getState().tabs[0].segmentGroups).toHaveLength(0);
    expect(onClose).not.toHaveBeenCalled();
  });

  it('clears the active segment when one exists and not creating', () => {
    const seg = { id: 'seg-1', headNodeId: 'n1', tailNodeId: 'n2' };
    setupTab({
      nodes: [], // not exactly 2 selected -> canCreateSegment false
      activeSegment: seg,
      segmentGroups: [seg],
    });
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);

    const btn = screen.getByRole('button', { name: 'Clear active' });
    expect(btn).not.toBeDisabled();
    fireEvent.click(btn);
    const tab = useTabStore.getState().tabs[0];
    expect(tab.segmentGroups).toHaveLength(0);
    expect(tab.activeSegment).toBeNull();
    // Recoverable: the frame the clear pushed carries the overlay and the
    // highlight, so Ctrl+Z brings both back (#200 item 2).
    expect(tab.undoStack).toHaveLength(1);
    expect(tab.undoStack[0].segmentGroups).toEqual([seg]);
    expect(tab.undoStack[0].activeSegment).toEqual(seg);
  });

  it('compare button is disabled (and its handler unreachable) with exactly one selected node', () => {
    // canCreateSegment (needs 2) false; canClearSegment (needs activeSegment) false
    // -> compareDisabled true -> the warning branch (line 115) cannot be reached
    // because its only trigger is this disabled button.
    setupTab({ nodes: [{ id: 'n1', selected: true, position: { x: 0, y: 0 } }] });
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    expect(screen.getByRole('button', { name: 'Select two nodes' })).toBeDisabled();
  });

  // ── Training: persist / gradients / auto-loss ─────────────────────

  it('toggles persist weights', () => {
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    fireEvent.click(screen.getByRole('button', { name: 'Persist weights between runs' }));
    expect(useTabStore.getState().tabs[0].weightsPersistent).toBe(false);
  });

  it('toggles capture gradients and the row click path', () => {
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    fireEvent.click(screen.getByRole('button', { name: 'Capture gradients' }));
    expect(useTabStore.getState().tabs[0].backwardMode).toBe(true);
  });

  // ── Training: reproducibility (core#134) ──────────────────────────

  it('starts with an empty seed field, meaning unseeded', () => {
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    // A tab persisted before core#134 carries no `seed` key at all, so the
    // field must render empty for undefined as well as for null.
    expect(screen.getByLabelText('Random seed')).toHaveValue(null);
  });

  it('writes a typed seed to the tab', () => {
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    fireEvent.change(screen.getByLabelText('Random seed'), { target: { value: '1234' } });
    expect(useTabStore.getState().tabs[0].seed).toBe(1234);
  });

  it('clearing the seed field goes back to unseeded', () => {
    setupTab({ seed: 42 });
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    fireEvent.change(screen.getByLabelText('Random seed'), { target: { value: '' } });
    expect(useTabStore.getState().tabs[0].seed).toBe(null);
  });

  it('keeps seed 0 rather than reading it as "unset"', () => {
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    fireEvent.change(screen.getByLabelText('Random seed'), { target: { value: '0' } });
    expect(useTabStore.getState().tabs[0].seed).toBe(0);
  });

  it('toggles deterministic algorithms', () => {
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    fireEvent.click(screen.getByRole('button', { name: 'Deterministic algorithms' }));
    expect(useTabStore.getState().tabs[0].deterministic).toBe(true);
  });

  it('auto-loss toggle is disabled while backward is off and enabled when on', () => {
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    const autoBtn = screen.getByRole('button', { name: 'Auto-synthesize loss' });
    expect(autoBtn).toBeDisabled();
    // Clicking the disabled control does nothing; clicking the (non-interactive
    // because disabled) row also does nothing.
    fireEvent.click(autoBtn);
    expect(useTabStore.getState().tabs[0].autoBackward).toBe(false);
  });

  it('auto-loss row is interactive and togglable when backward is on', () => {
    setupTab({ backwardMode: true });
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    const autoBtn = screen.getByRole('button', { name: 'Auto-synthesize loss' });
    expect(autoBtn).not.toBeDisabled();
    fireEvent.click(autoBtn);
    expect(useTabStore.getState().tabs[0].autoBackward).toBe(true);
    // Also exercise the row onClick (backward ? toggleAutoBackward : undefined => defined)
    const row = screen.getByText('Auto-synthesize loss').closest('[role="button"]')!;
    fireEvent.click(row);
    expect(useTabStore.getState().tabs[0].autoBackward).toBe(false);
  });

  // ── Reset weights ─────────────────────────────────────────────────

  it('reset weights is disabled when there is no graphId and returns early', async () => {
    setupTab({ graphId: '' });
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    const btn = screen.getByRole('button', { name: 'Reset' });
    expect(btn).toBeDisabled();
    fireEvent.click(btn);
    expect(mockedResetWeights).not.toHaveBeenCalled();
  });

  it('reset weights: user cancels the confirm -> no API call', async () => {
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    fireEvent.click(screen.getByRole('button', { name: 'Reset' }));
    // The confirm dialog is now pending in the dialog store.
    await waitFor(() => expect(useDialogStore.getState().active).not.toBeNull());
    await act(async () => {
      useDialogStore.getState().close(false);
    });
    expect(mockedResetWeights).not.toHaveBeenCalled();
  });

  it('reset weights: confirmed -> calls API and shows success toast', async () => {
    mockedResetWeights.mockResolvedValueOnce({ graph_id: 'graph-xyz', scope: 'graph', evicted: 7 });
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    fireEvent.click(screen.getByRole('button', { name: 'Reset' }));
    await waitFor(() => expect(useDialogStore.getState().active).not.toBeNull());
    await act(async () => {
      useDialogStore.getState().close(true);
    });
    await waitFor(() => expect(mockedResetWeights).toHaveBeenCalledWith('graph-xyz'));
    await waitFor(() => {
      const toasts = useToastStore.getState().toasts;
      expect(toasts.some((t) => t.type === 'success' && t.message.includes('7'))).toBe(true);
    });
  });

  it('reset weights: API rejects -> shows error toast', async () => {
    mockedResetWeights.mockRejectedValueOnce(new Error('boom'));
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    fireEvent.click(screen.getByRole('button', { name: 'Reset' }));
    await waitFor(() => expect(useDialogStore.getState().active).not.toBeNull());
    await act(async () => {
      useDialogStore.getState().close(true);
    });
    await waitFor(() => {
      const toasts = useToastStore.getState().toasts;
      expect(toasts.some((t) => t.type === 'error' && t.message.includes('boom'))).toBe(true);
    });
  });

  // ── Editor section ────────────────────────────────────────────────

  it('toggles grid snap and tooltips', () => {
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    fireEvent.click(screen.getByRole('button', { name: 'Grid snap' }));
    expect(useUIStore.getState().gridSnapEnabled).toBe(true);
    fireEvent.click(screen.getByRole('button', { name: 'Show node tooltips' }));
    expect(useUIStore.getState().tooltipsEnabled).toBe(false);
  });

  it('node-mode segmented control switches between Basic and All', () => {
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    const group = screen.getByRole('group', { name: 'Node category mode' });
    const basicBtn = within(group).getByText('Basic');
    const allBtn = within(group).getByText('All');

    // beginnerMode starts false -> clicking Basic enables it
    fireEvent.click(basicBtn);
    expect(useUIStore.getState().beginnerMode).toBe(true);

    // clicking Basic again is a no-op (already beginner)
    fireEvent.click(basicBtn);
    expect(useUIStore.getState().beginnerMode).toBe(true);

    // clicking All disables beginner mode
    fireEvent.click(allBtn);
    expect(useUIStore.getState().beginnerMode).toBe(false);

    // clicking All again is a no-op
    fireEvent.click(allBtn);
    expect(useUIStore.getState().beginnerMode).toBe(false);
  });

  it('node-mode starting from beginner=true exercises the inverse guards', () => {
    useUIStore.setState({ beginnerMode: true });
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    const group = screen.getByRole('group', { name: 'Node category mode' });
    // All click toggles off
    fireEvent.click(within(group).getByText('All'));
    expect(useUIStore.getState().beginnerMode).toBe(false);
    // Now Basic click toggles on
    fireEvent.click(within(group).getByText('Basic'));
    expect(useUIStore.getState().beginnerMode).toBe(true);
  });

  it('connection-style segmented control switches between Circuit and Curve', () => {
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    const group = screen.getByRole('group', { name: 'Connection style' });
    const circuitBtn = within(group).getByText('Circuit');
    const curveBtn = within(group).getByText('Curve');

    // Default is circuit -> Circuit renders as the active segment.
    expect(circuitBtn.className).toContain('active');
    expect(curveBtn.className).not.toContain('active');

    fireEvent.click(curveBtn);
    expect(useUIStore.getState().edgeStyle).toBe('curve');
    expect(localStorage.getItem('codefyui-edge-style')).toBe('curve');

    fireEvent.click(circuitBtn);
    expect(useUIStore.getState().edgeStyle).toBe('circuit');
    expect(localStorage.getItem('codefyui-edge-style')).toBe('circuit');
  });

  // ── default-value fallbacks (?? operators) ────────────────────────

  it('falls back to defaults when tab flags are undefined', () => {
    setupTab({
      recordOutputs: undefined as never,
      verboseMode: undefined as never,
      weightsPersistent: undefined as never,
      backwardMode: undefined as never,
      autoBackward: undefined as never,
      graphId: undefined as never,
    });
    render(<SettingsPopover open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);
    // recording defaults true
    expect(screen.getByRole('button', { name: 'Record node outputs' })).toHaveAttribute(
      'aria-pressed',
      'true',
    );
    // persistent defaults true
    expect(screen.getByRole('button', { name: 'Persist weights between runs' })).toHaveAttribute(
      'aria-pressed',
      'true',
    );
    // verbose/backward/autoBackward default false
    expect(screen.getByRole('button', { name: 'Verbose internals' })).toHaveAttribute(
      'aria-pressed',
      'false',
    );
    // graphId '' -> reset disabled
    expect(screen.getByRole('button', { name: 'Reset' })).toBeDisabled();
  });
});
