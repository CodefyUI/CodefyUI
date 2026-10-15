import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, screen, waitFor, within } from '@testing-library/react';
import * as rest from '../../api/rest';
import type { DeviceInfo, WorkerInfo } from '../../api/rest';
import { QueueRunDialog } from './QueueRunDialog';
import { _resetDeviceOptionsForTesting } from '../../hooks/useDeviceOptions';
import { useI18n } from '../../i18n';
import { isAnyModalOpen } from '../../store/modalState';
import { _resetRunStoreForTesting, useRunStore } from '../../store/runStore';
import { _resetSweepStoreForTesting, useSweepStore } from '../../store/sweepStore';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import { renderSettled, settle } from '../../test/utils';

vi.mock('../../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof rest>();
  return { ...actual, listWorkers: vi.fn(), submitRun: vi.fn(), fetchDevices: vi.fn() };
});

const api = vi.mocked(rest);

const CPU: DeviceInfo = { value: 'cpu', label: 'CPU', detail: '', available: true };
const CUDA: DeviceInfo = { value: 'cuda', label: 'NVIDIA CUDA', detail: 'RTX 4090', available: true };
const GRAPH = { nodes: [{ id: 'n1' }], edges: [], presets: [], settings: { device: 'cpu' } };

function worker(partial: Partial<WorkerInfo> & { name: string }): WorkerInfo {
  return {
    id: partial.name,
    status: 'online',
    draining: false,
    version: '3.0.2',
    fingerprint_digest: null,
    platform: {},
    mounts: {},
    devices: [
      { value: 'cpu', kind: 'cpu', index: null, name: '', memory: null },
      { value: 'cuda:0', kind: 'cuda', index: 0, name: 'A100', memory: null },
      { value: 'cuda:1', kind: 'cuda', index: 1, name: 'A100', memory: null },
      { value: 'cuda:2', kind: 'cuda', index: 2, name: 'A100', memory: null },
    ],
    running: [],
    connected_at: 0,
    last_heartbeat_at: 0,
    ...partial,
  };
}

const onClose = vi.fn();
const serialize = vi.fn((_options?: { keepSecrets?: boolean }) => GRAPH);
const refresh = vi.fn(() => Promise.resolve());
const select = vi.fn(() => Promise.resolve());
const closeSweep = vi.fn();
const realSerialize = useTabStore.getState().getSerializedGraph;

function patchTab(patch: Record<string, unknown>) {
  useTabStore.setState((state) => ({
    tabs: state.tabs.map((tab) => (tab.id === state.activeTabId ? { ...tab, ...patch } : tab)),
  }));
}

async function open() {
  const result = await renderSettled(<QueueRunDialog onClose={onClose} />);
  return result;
}

const dialog = () => screen.getByRole('dialog', { name: 'Queue run' });
const targetSelect = () => screen.getByRole('combobox', { name: 'Target' }) as HTMLSelectElement;
const submitButton = () => screen.getByRole('button', { name: 'Queue' });

async function submit() {
  await act(async () => {
    fireEvent.click(submitButton());
  });
}

function sentRequest() {
  return api.submitRun.mock.calls[0][0];
}

beforeEach(() => {
  vi.clearAllMocks();
  useI18n.setState({ locale: 'en' });
  _resetDeviceOptionsForTesting();
  _resetRunStoreForTesting();
  _resetSweepStoreForTesting();
  useTabStore.setState({ tabs: [], activeTabId: '' });
  useTabStore.getState().addTab('My graph');
  useTabStore.setState({ getSerializedGraph: serialize as never });
  patchTab({ graphDevice: null, seed: null, deterministic: false, recordOutputs: true });
  useUIStore.setState({ globalDevice: 'cpu', queueRunOpen: true, runsTabRequest: 0 });
  useRunStore.setState({ refresh, select });
  useSweepStore.setState({ closeSweep });
  api.fetchDevices.mockResolvedValue({ default: 'cpu', devices: [CPU] });
  api.listWorkers.mockResolvedValue([]);
  api.submitRun.mockResolvedValue({ run_id: 'run-1', status: 'queued' });
  onClose.mockReset();
  serialize.mockClear();
  refresh.mockClear();
  select.mockClear();
  closeSweep.mockClear();
});

afterEach(() => {
  useTabStore.setState({ getSerializedGraph: realSerialize });
  useUIStore.setState({ queueRunOpen: false });
});

describe('QueueRunDialog: this server', () => {
  it('offers This server alone when no worker is connected, and submits like cdui run', async () => {
    patchTab({ seed: 7, deterministic: true });
    await open();
    expect(within(targetSelect()).getAllByRole('option').map((o) => o.textContent)).toEqual(['This server']);
    expect(targetSelect()).toHaveFocus();
    expect(screen.getByRole('radio', { name: /cpu/ })).toBeChecked();
    expect(screen.getByText('A run on this server uses one device.')).toBeInTheDocument();
    expect(screen.getByRole('spinbutton', { name: 'Random seed' })).toHaveValue(7);
    expect(screen.getByRole('checkbox', { name: 'Deterministic algorithms' })).toBeChecked();
    expect(screen.getByRole('checkbox', { name: 'Record node outputs' })).toBeChecked();
    expect(screen.getByRole('textbox', { name: 'Name' })).toHaveAttribute('placeholder', 'My graph');

    await submit();

    expect(serialize).toHaveBeenCalledWith({ keepSecrets: true });
    const tab = useTabStore.getState().getActiveTab();
    expect(sentRequest()).toEqual({
      graph: GRAPH,
      options: {
        target: { worker: 'local', devices: ['cpu'] },
        device: 'cpu',
        record_outputs: true,
        verbose: tab.verboseMode,
        graph_id: tab.graphId,
        weights_persistent: tab.weightsPersistent,
        backward_mode: tab.backwardMode,
        auto_backward: tab.autoBackward,
        seed: 7,
        deterministic: true,
      },
      name: 'My graph',
    });
    // Shown where queued runs live, with the new run open.
    expect(closeSweep).toHaveBeenCalled();
    expect(useUIStore.getState().runsTabRequest).toBe(1);
    expect(refresh).toHaveBeenCalled();
    expect(select).toHaveBeenCalledWith('run-1');
    expect(onClose).toHaveBeenCalled();
  });

  it('sends no seed when the field is blank, the typed name, and the edited checkboxes', async () => {
    await open();
    fireEvent.change(screen.getByRole('textbox', { name: 'Name' }), { target: { value: '  nightly  ' } });
    fireEvent.click(screen.getByRole('checkbox', { name: 'Deterministic algorithms' }));
    fireEvent.click(screen.getByRole('checkbox', { name: 'Record node outputs' }));
    await submit();
    const { options, name } = sentRequest();
    expect(name).toBe('nightly');
    expect(options).not.toHaveProperty('seed');
    expect(options).toMatchObject({ deterministic: true, record_outputs: false });
  });

  it('defaults to the graph device, a lone CUDA card named as cuda:0', async () => {
    api.fetchDevices.mockResolvedValue({ default: 'cuda', devices: [CPU, CUDA] });
    patchTab({ graphDevice: 'cuda' });
    await open();
    expect(screen.getByRole('radio', { name: /^cuda:0 / })).toBeChecked();
    expect(screen.getByText('NVIDIA CUDA · RTX 4090')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('radio', { name: /cpu/ }));
    await submit();
    expect(sentRequest().options).toMatchObject({ target: { worker: 'local', devices: ['cpu'] }, device: 'cpu' });
  });

  it('refuses a seed that is not a whole number without asking the server', async () => {
    await open();
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Random seed' }), { target: { value: '1.5' } });
    await submit();
    expect(screen.getByRole('alert')).toHaveTextContent('The seed must be a whole number, 0 or more.');
    expect(api.submitRun).not.toHaveBeenCalled();
  });

  it('says so when this server lists no device a set can name', async () => {
    api.fetchDevices.mockResolvedValue({
      default: 'cpu', devices: [{ value: 'auto', label: 'Auto', detail: '', available: true }],
    });
    await open();
    expect(screen.getByText('no devices')).toBeInTheDocument();
    await submit();
    expect(screen.getByRole('alert')).toHaveTextContent('Choose at least one device.');
    expect(api.submitRun).not.toHaveBeenCalled();
  });
});

describe('QueueRunDialog: workers', () => {
  it('lists connected workers with status and devices, leaving out the disconnected and disabling the draining', async () => {
    api.listWorkers.mockResolvedValue([
      worker({ name: 'gpu-box' }),
      worker({ name: 'slow-box', status: 'degraded', devices: [] }),
      worker({ name: 'busy-box', status: 'draining', draining: true }),
      worker({ name: 'gone-box', status: 'disconnected' }),
    ]);
    await open();
    const options = within(targetSelect()).getAllByRole('option') as HTMLOptionElement[];
    expect(options.map((o) => o.textContent)).toEqual([
      'This server',
      'gpu-box (online): cpu, cuda:0, cuda:1, cuda:2',
      'slow-box (heartbeat late): no devices',
      'busy-box (draining): cpu, cuda:0, cuda:1, cuda:2',
      'Any available',
    ]);
    expect(options[3]).toBeDisabled();
  });

  it('builds a device set of one kind and shows each card under its in-run name', async () => {
    api.listWorkers.mockResolvedValue([worker({ name: 'gpu-box' })]);
    patchTab({ graphDevice: 'cuda:1' });
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'gpu-box' } });
    // The graph's device, as this worker has it.
    expect(screen.getByRole('checkbox', { name: /^cuda:1 / })).toBeChecked();
    expect(screen.getByText('cuda:1 -> cuda:0 in this run')).toBeInTheDocument();

    // A card of the same kind joins the set; the order is the in-run order.
    fireEvent.click(screen.getByRole('checkbox', { name: /^cuda:2 / }));
    fireEvent.click(screen.getByRole('checkbox', { name: /^cuda:0 / }));
    expect(screen.getByText('cuda:2 -> cuda:1 in this run')).toBeInTheDocument();
    expect(screen.getByText('cuda:0 -> cuda:2 in this run')).toBeInTheDocument();

    // Outputs cannot be recorded on a worker yet (#789).
    const record = screen.getByRole('checkbox', { name: 'Record node outputs' });
    expect(record).toBeDisabled();
    expect(record).not.toBeChecked();
    expect(screen.getByText(/cannot record outputs yet/)).toBeInTheDocument();

    await submit();
    expect(sentRequest().options).toMatchObject({
      target: { worker: 'gpu-box', devices: ['cuda:1', 'cuda:2', 'cuda:0'] },
      device: 'cuda:1',
      record_outputs: false,
    });
  });

  it('replaces the set with a device of another kind, and unchecks one already in it', async () => {
    api.listWorkers.mockResolvedValue([worker({ name: 'gpu-box' })]);
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'gpu-box' } });
    // The graph's device is cpu, which the worker has.
    expect(screen.getByRole('checkbox', { name: /^cpu/ })).toBeChecked();
    fireEvent.click(screen.getByRole('checkbox', { name: /^cuda:0 / }));
    expect(screen.getByRole('checkbox', { name: /^cpu/ })).not.toBeChecked();
    // A lone cuda:0 keeps its name in the run: no alias line.
    expect(screen.queryByText(/in this run/)).toBeNull();
    fireEvent.click(screen.getByRole('checkbox', { name: /^cuda:0 / }));
    await submit();
    expect(screen.getByRole('alert')).toHaveTextContent('Choose at least one device.');
    // Another target starts over, without the old message.
    fireEvent.change(targetSelect(), { target: { value: 'local' } });
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('shows a refused placement in the dialog and stays open', async () => {
    api.listWorkers.mockResolvedValue([worker({ name: 'gpu-box' })]);
    api.submitRun.mockRejectedValueOnce(new rest.ApiError(400, "worker 'gpu-box' does not offer cuda:7"));
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'gpu-box' } });
    await submit();
    expect(screen.getByRole('alert')).toHaveTextContent("worker 'gpu-box' does not offer cuda:7");
    expect(onClose).not.toHaveBeenCalled();
    expect(submitButton()).toBeEnabled();
    expect(select).not.toHaveBeenCalled();
  });

  it('shows a refusal that is not an Error as text', async () => {
    api.submitRun.mockRejectedValueOnce('run service not initialised');
    await open();
    await submit();
    expect(screen.getByRole('alert')).toHaveTextContent('run service not initialised');
  });

  it('labels the button while the submit is in flight', async () => {
    let finish: (value: { run_id: string; status: 'queued' }) => void = () => {};
    api.submitRun.mockReturnValueOnce(new Promise((resolve) => { finish = resolve; }));
    await open();
    await submit();
    expect(screen.getByRole('button', { name: 'Queueing...' })).toBeDisabled();
    await act(async () => finish({ run_id: 'run-2', status: 'queued' }));
    expect(select).toHaveBeenCalledWith('run-2');
  });
});

describe('QueueRunDialog: any available', () => {
  it('asks for a kind, defaulting to the graph device kind, and sends no device set', async () => {
    api.listWorkers.mockResolvedValue([worker({ name: 'gpu-box' })]);
    patchTab({ graphDevice: 'cuda' });
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'any' } });
    const group = within(screen.getByRole('group', { name: 'Devices' }));
    expect(group.getAllByRole('radio').map((r) => r.closest('label')!.textContent)).toEqual(['CPU', 'CUDA GPU']);
    expect(group.getByRole('radio', { name: 'CUDA GPU' })).toBeChecked();
    expect(screen.getByText(/first connected worker with a free device/)).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'Record node outputs' })).toBeDisabled();
    fireEvent.click(group.getByRole('radio', { name: 'CPU' }));
    await submit();
    expect(sentRequest().options).toMatchObject({ target: { worker: 'any' }, device: 'cpu', record_outputs: false });
  });
});

describe('QueueRunDialog: the worker list', () => {
  it('says why only This server is offered when the list cannot be read', async () => {
    api.listWorkers.mockRejectedValueOnce(new Error('worker registry is not available'));
    await open();
    expect(screen.getByText('Could not list workers: worker registry is not available')).toBeInTheDocument();
    expect(within(targetSelect()).getAllByRole('option')).toHaveLength(1);
  });

  it('reads a refusal that is not an Error as text', async () => {
    api.listWorkers.mockRejectedValueOnce('offline');
    await open();
    expect(screen.getByText('Could not list workers: offline')).toBeInTheDocument();
  });

  it('drops an answer that arrives after the dialog closed', async () => {
    let answer: (workers: WorkerInfo[]) => void = () => {};
    let refuse: (reason: unknown) => void = () => {};
    api.listWorkers
      .mockReturnValueOnce(new Promise((resolve) => { answer = resolve; }))
      .mockReturnValueOnce(new Promise((_resolve, reject) => { refuse = reject; }));
    const first = await open();
    first.unmount();
    const second = await open();
    second.unmount();
    await act(async () => {
      answer([worker({ name: 'gpu-box' })]);
      refuse(new Error('late'));
    });
    await settle();
    expect(screen.queryByRole('dialog')).toBeNull();
  });
});

describe('QueueRunDialog: keyboard and closing', () => {
  it('is a modal while open and clears its flag when it goes', async () => {
    const { unmount } = await open();
    expect(isAnyModalOpen()).toBe(true);
    unmount();
    expect(useUIStore.getState().queueRunOpen).toBe(false);
    expect(isAnyModalOpen()).toBe(false);
  });

  it('closes on Escape, the close button and a press on the backdrop, not one inside', async () => {
    await open();
    fireEvent.keyDown(dialog(), { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    expect(onClose).toHaveBeenCalledTimes(2);
    fireEvent.mouseDown(dialog());
    expect(onClose).toHaveBeenCalledTimes(2);
    fireEvent.mouseDown(dialog().parentElement!);
    expect(onClose).toHaveBeenCalledTimes(3);
  });

  it('keeps Tab inside the dialog, both ways', async () => {
    await open();
    const queue = submitButton();
    queue.focus();
    fireEvent.keyDown(queue, { key: 'Tab' });
    expect(screen.getByRole('button', { name: 'Close' })).toHaveFocus();
    fireEvent.keyDown(screen.getByRole('button', { name: 'Close' }), { key: 'Tab', shiftKey: true });
    expect(queue).toHaveFocus();
    // From the dialog itself (a click between controls), Shift+Tab wraps too.
    dialog().focus();
    fireEvent.keyDown(dialog(), { key: 'Tab', shiftKey: true });
    expect(queue).toHaveFocus();
    // Anywhere in the middle, Tab is the browser's.
    targetSelect().focus();
    fireEvent.keyDown(targetSelect(), { key: 'Tab' });
    expect(targetSelect()).toHaveFocus();
    fireEvent.keyDown(targetSelect(), { key: 'a' });
    expect(onClose).not.toHaveBeenCalled();
  });

  it('hands the focus back to what had it', async () => {
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();
    const { unmount } = await open();
    expect(targetSelect()).toHaveFocus();
    unmount();
    expect(opener).toHaveFocus();
    opener.remove();
  });

  it('leaves the focus alone when what had it is gone', async () => {
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();
    const { unmount } = await open();
    opener.remove();
    unmount();
    expect(opener).not.toHaveFocus();
  });

  it('sends no name when neither the field nor the tab has one', async () => {
    patchTab({ name: '' });
    await open();
    await submit();
    expect(sentRequest().name).toBeNull();
  });

  it('renders the zh-TW strings', async () => {
    useI18n.setState({ locale: 'zh-TW' });
    await open();
    expect(screen.getByRole('dialog', { name: '排入佇列執行' })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole('option', { name: '本伺服器' })).toBeInTheDocument());
  });
});
