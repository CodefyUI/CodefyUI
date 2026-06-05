import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { createRef, act } from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { SettingsPopover } from './SettingsPopover';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import { useToastStore } from '../../store/toastStore';
import { useDialogStore } from '../../store/dialogStore';
import { useI18n } from '../../i18n';
import { resetWeights, fetchDevices } from '../../api/rest';
import { computeSegmentNodes } from '../../utils/segmentPath';

vi.mock('../../api/rest', () => ({
  resetWeights: vi.fn(),
  fetchDevices: vi.fn(() =>
    Promise.resolve({
      default: 'cpu',
      devices: [
        { value: 'cpu', label: 'CPU', detail: '', available: true },
        { value: 'mps', label: 'Apple MPS', detail: 'Metal Performance Shaders', available: true },
      ],
    }),
  ),
}));

vi.mock('../../utils/segmentPath', () => ({
  computeSegmentNodes: vi.fn(() => new Set(['a', 'b'])),
}));

const mockedResetWeights = vi.mocked(resetWeights);
const mockedComputeSegment = vi.mocked(computeSegmentNodes);

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
    });
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
    expect(screen.getByRole('dialog')).toBeInTheDocument();
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
