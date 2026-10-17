import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, screen, waitFor, within } from '@testing-library/react';
import * as rest from '../../api/rest';
import type { DeviceInfo } from '../../api/rest';
import * as workersApi from '../../api/workers';
import type { WorkerInfo } from '../../api/workers';
import { QueueRunDialog } from './QueueRunDialog';
import { _resetDeviceOptionsForTesting } from '../../hooks/useDeviceOptions';
import { useI18n } from '../../i18n';
import { isAnyModalOpen } from '../../store/modalState';
import { _resetRunStoreForTesting, useRunStore } from '../../store/runStore';
import { _resetSweepStoreForTesting, useSweepStore } from '../../store/sweepStore';
import { useTabStore } from '../../store/tabStore';
import { useToastStore } from '../../store/toastStore';
import { useUIStore } from '../../store/uiStore';
import { renderSettled, settle } from '../../test/utils';

vi.mock('../../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof rest>();
  return { ...actual, submitRun: vi.fn(), fetchDevices: vi.fn(), estimateMemory: vi.fn() };
});

vi.mock('../../api/workers', async (importOriginal) => {
  const actual = await importOriginal<typeof workersApi>();
  return { ...actual, fetchWorkers: vi.fn() };
});

const workersMock = vi.mocked(workersApi);

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

const NO_ESTIMATE = { suggested_gib: null, source: null, peak_bytes: null, from_run: null };

/** A worker whose cuda:1 is shared by memory (#788). */
function sharedWorker(name = 'gpu-box'): WorkerInfo {
  const base = worker({ name, memory_budget: true });
  return { ...base, devices: base.devices.map((d) => (d.value === 'cuda:1' ? { ...d, policy: 'shared' } : d)) };
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
  useToastStore.setState({ toasts: [] });
  useSweepStore.setState({ closeSweep });
  api.fetchDevices.mockResolvedValue({ default: 'cpu', devices: [CPU] });
  workersMock.fetchWorkers.mockResolvedValue([]);
  api.submitRun.mockResolvedValue({ run_id: 'run-1', status: 'queued' });
  api.estimateMemory.mockResolvedValue(NO_ESTIMATE);
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
    expect(screen.getByRole('textbox', { name: 'Random seed' })).toHaveValue('7');
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
    // Confirmed whether or not the dock is in view, with a way to it.
    const toast = useToastStore.getState().toasts[0];
    expect(toast).toMatchObject({ message: 'Queued My graph', type: 'success' });
    select.mockClear();
    act(() => toast.action!.onClick());
    expect(useUIStore.getState().runsTabRequest).toBe(2);
    await waitFor(() => expect(select).toHaveBeenCalledWith('run-1'));
  });

  it('names the run by its id in the toast when it has no name', async () => {
    patchTab({ name: '' });
    api.submitRun.mockResolvedValueOnce({ run_id: 'abcdef123456', status: 'running' });
    await open();
    await submit();
    expect(useToastStore.getState().toasts[0].message).toBe('Queued abcdef12');
  });

  it('treats a failed list refresh after the submit as non-blocking', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    refresh.mockRejectedValueOnce(new Error('offline'));
    select.mockRejectedValueOnce(new Error('gone'));
    await open();
    await submit();
    await settle();
    // Queued once, closed, confirmed: no error, no second chance to submit.
    expect(api.submitRun).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole('alert')).toBeNull();
    expect(screen.getByRole('button', { name: 'Queueing...' })).toBeDisabled();
    expect(select).toHaveBeenCalledWith('run-1');
    expect(useToastStore.getState().toasts[0].type).toBe('success');
    expect(warn).toHaveBeenCalledTimes(2);
  });

  it('sends an explicit null seed when the field is blank, the typed name, and the edited checkboxes', async () => {
    await open();
    fireEvent.change(screen.getByRole('textbox', { name: 'Name' }), { target: { value: '  nightly  ' } });
    fireEvent.click(screen.getByRole('checkbox', { name: 'Deterministic algorithms' }));
    fireEvent.click(screen.getByRole('checkbox', { name: 'Record node outputs' }));
    await submit();
    const { options, name } = sentRequest();
    expect(name).toBe('nightly');
    expect(options).toHaveProperty('seed', null);
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

  it.each(['12a', '1.5', '-1', '1e3', '4294967296'])('refuses the seed %s without asking the server', async (raw) => {
    await open();
    fireEvent.change(screen.getByRole('textbox', { name: 'Random seed' }), { target: { value: raw } });
    await submit();
    expect(screen.getByRole('alert')).toHaveTextContent('The seed must be a whole number, 0 or more.');
    expect(api.submitRun).not.toHaveBeenCalled();
  });

  it('takes the largest run seed, spaces trimmed', async () => {
    await open();
    fireEvent.change(screen.getByRole('textbox', { name: 'Random seed' }), { target: { value: ' 4294967295 ' } });
    await submit();
    expect(sentRequest().options).toHaveProperty('seed', 4294967295);
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
  it('lists workers with status and devices, an offline one too, leaving out the disconnected, disabling the draining', async () => {
    workersMock.fetchWorkers.mockResolvedValue([
      worker({ name: 'gpu-box' }),
      worker({ name: 'slow-box', status: 'degraded', devices: [] }),
      worker({ name: 'busy-box', status: 'draining', draining: true }),
      worker({ name: 'gone-box', status: 'disconnected' }),
      worker({ name: 'cold-box', status: 'offline' }),
    ]);
    await open();
    const options = within(targetSelect()).getAllByRole('option') as HTMLOptionElement[];
    expect(options.map((o) => o.textContent)).toEqual([
      'This server',
      'gpu-box (Online): cpu, cuda:0, cuda:1, cuda:2',
      'slow-box (Degraded): no devices',
      'busy-box (Draining): cpu, cuda:0, cuda:1, cuda:2',
      'cold-box (Offline): cpu, cuda:0, cuda:1, cuda:2',
      'Any available',
      'A one-shot worker (cdui agent start --one-shot)',
    ]);
    expect(options[3]).toBeDisabled();
    expect(screen.queryByText(/is offline/)).toBeNull();
    // An offline worker can be chosen; the run waits for it.
    fireEvent.change(targetSelect(), { target: { value: 'cold-box' } });
    expect(screen.getByText('cold-box is offline: the run waits in the queue until it connects.')).toBeInTheDocument();
  });

  it('builds a device set of one kind and shows each card under its in-run name', async () => {
    workersMock.fetchWorkers.mockResolvedValue([worker({ name: 'gpu-box' })]);
    patchTab({ graphDevice: 'cuda:1' });
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'gpu-box' } });
    // The graph's device, as this worker has it.
    expect(screen.getByRole('checkbox', { name: /^cuda:1 / })).toBeChecked();
    expect(screen.getByText('cuda:1 -> cuda:0 in this run')).toBeInTheDocument();

    // A card of the same kind joins the set, which stays in the worker's
    // order: the in-run names follow the cards, not the clicks.
    fireEvent.click(screen.getByRole('checkbox', { name: /^cuda:2 / }));
    expect(screen.getByText('cuda:1 -> cuda:0 in this run')).toBeInTheDocument();
    expect(screen.getByText('cuda:2 -> cuda:1 in this run')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('checkbox', { name: /^cuda:0 / }));
    expect(screen.getByText('cuda:0 -> cuda:0 in this run')).toBeInTheDocument();
    expect(screen.getByText('cuda:1 -> cuda:1 in this run')).toBeInTheDocument();
    expect(screen.getByText('cuda:2 -> cuda:2 in this run')).toBeInTheDocument();
    // Unticked and ticked again, cuda:0 goes back to the front.
    fireEvent.click(screen.getByRole('checkbox', { name: /^cuda:0 / }));
    expect(screen.getByText('cuda:1 -> cuda:0 in this run')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('checkbox', { name: /^cuda:0 / }));
    expect(screen.getByText('cuda:0 -> cuda:0 in this run')).toBeInTheDocument();

    // Outputs cannot be recorded on a worker yet (#789).
    const record = screen.getByRole('checkbox', { name: 'Record node outputs' });
    expect(record).toBeDisabled();
    expect(record).not.toBeChecked();
    expect(screen.getByText(/cannot record outputs yet/)).toBeInTheDocument();

    await submit();
    expect(sentRequest().options).toMatchObject({
      target: { worker: 'gpu-box', devices: ['cuda:0', 'cuda:1', 'cuda:2'] },
      device: 'cuda:0',
      record_outputs: false,
    });
  });

  it('replaces the set with a device of another kind, and unchecks one already in it', async () => {
    workersMock.fetchWorkers.mockResolvedValue([worker({ name: 'gpu-box' })]);
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'gpu-box' } });
    // The graph's device is cpu, which the worker has.
    expect(screen.getByRole('checkbox', { name: /^cpu/ })).toBeChecked();
    fireEvent.click(screen.getByRole('checkbox', { name: /^cuda:0 / }));
    expect(screen.getByRole('checkbox', { name: /^cpu/ })).not.toBeChecked();
    // A lone cuda:0 keeps its name in the run, and says so.
    expect(screen.getByText('cuda:0 -> cuda:0 in this run')).toBeInTheDocument();
    // A CPU device has no in-run name.
    fireEvent.click(screen.getByRole('checkbox', { name: /^cpu/ }));
    expect(screen.queryByText(/in this run/)).toBeNull();
    fireEvent.click(screen.getByRole('checkbox', { name: /^cuda:0 / }));
    fireEvent.click(screen.getByRole('checkbox', { name: /^cuda:0 / }));
    await submit();
    expect(screen.getByRole('alert')).toHaveTextContent('Choose at least one device.');
    // Another target starts over, without the old message.
    fireEvent.change(targetSelect(), { target: { value: 'local' } });
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('shows a refused placement in the dialog and stays open', async () => {
    workersMock.fetchWorkers.mockResolvedValue([worker({ name: 'gpu-box' })]);
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
    await waitFor(() => expect(select).toHaveBeenCalledWith('run-2'));
  });
});

describe('QueueRunDialog: any available', () => {
  it('asks for a kind, defaulting to the graph device kind, and sends no device set', async () => {
    workersMock.fetchWorkers.mockResolvedValue([worker({ name: 'gpu-box' })]);
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

describe('QueueRunDialog: a one-shot worker (#790)', () => {
  it('sends the run to the one-shot pool with any device kind, and explains the wait', async () => {
    // An offline enrolled worker is enough: the pool waits for one to appear.
    workersMock.fetchWorkers.mockResolvedValue([worker({ name: 'cold-box', status: 'offline' })]);
    patchTab({ graphDevice: 'cuda:1' });
    await open();
    expect(within(targetSelect()).queryByRole('option', { name: 'Any available' })).toBeNull();
    fireEvent.change(targetSelect(), { target: { value: 'pool:one-shot' } });
    const group = within(screen.getByRole('group', { name: 'Devices' }));
    expect(group.getAllByRole('radio').map((r) => r.closest('label')!.textContent)).toEqual([
      'CPU', 'CUDA GPU', 'Apple MPS',
    ]);
    expect(group.getByRole('radio', { name: 'CUDA GPU' })).toBeChecked();
    expect(screen.getByText(/waits for a one-shot worker with a device of this kind/)).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'Record node outputs' })).toBeDisabled();
    await submit();
    expect(sentRequest().options).toMatchObject({
      target: { worker: 'any', pool: 'one-shot' }, device: 'cuda', record_outputs: false,
    });
  });

  it('is not offered on a server without workers', async () => {
    await open();
    expect(within(targetSelect()).queryByRole('option', { name: /one-shot/ })).toBeNull();
  });
});

describe('QueueRunDialog: the worker list', () => {
  it('says why only This server is offered when the list cannot be read', async () => {
    workersMock.fetchWorkers.mockRejectedValueOnce(new Error('worker registry is not available'));
    await open();
    expect(screen.getByText('Could not list workers: worker registry is not available')).toBeInTheDocument();
    expect(within(targetSelect()).getAllByRole('option')).toHaveLength(1);
  });

  it('reads a refusal that is not an Error as text', async () => {
    workersMock.fetchWorkers.mockRejectedValueOnce('offline');
    await open();
    expect(screen.getByText('Could not list workers: offline')).toBeInTheDocument();
  });

  it('drops an answer that arrives after the dialog closed', async () => {
    let answer: (workers: WorkerInfo[]) => void = () => {};
    let refuse: (reason: unknown) => void = () => {};
    workersMock.fetchWorkers
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

describe('QueueRunDialog: memory budget (#788)', () => {
  const memoryField = () => screen.getByRole('textbox', { name: 'Memory budget (GiB per device)' });

  it('asks for a budget only on a shared device, filled with the suggestion', async () => {
    workersMock.fetchWorkers.mockResolvedValue([sharedWorker()]);
    api.estimateMemory.mockResolvedValue({
      suggested_gib: 2.4, source: 'history', peak_bytes: 2 * 1024 ** 3, from_run: 'r1',
    });
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'gpu-box' } });
    // The CPU is picked first (the graph's device) and cuda:0 is exclusive:
    // no budget asked for.
    expect(screen.queryByRole('textbox', { name: /Memory budget/ })).toBeNull();
    fireEvent.click(screen.getByRole('checkbox', { name: /^cuda:0 / }));
    expect(screen.queryByRole('textbox', { name: /Memory budget/ })).toBeNull();
    fireEvent.click(screen.getByRole('checkbox', { name: /^cuda:0 / }));
    fireEvent.click(screen.getByRole('checkbox', { name: /^cuda:1 / }));
    await settle();
    expect(api.estimateMemory).toHaveBeenCalledWith(GRAPH);
    expect(serialize).toHaveBeenCalledWith({ keepSecrets: true });
    expect(memoryField()).toHaveValue('2.4');
    expect(screen.getByText('Suggested from the last run of this graph: its peak was 2.0 GiB, plus 20%.')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Wait in the queue when it does not fit now' }));
    await submit();
    expect(sentRequest().options.target).toEqual({
      worker: 'gpu-box', devices: ['cuda:1'], memory_gib: 2.4, wait: true,
    });
  });

  it('sends a typed budget, none for a blank field, and refuses one that is not a number', async () => {
    workersMock.fetchWorkers.mockResolvedValue([sharedWorker()]);
    patchTab({ graphDevice: 'cuda:1' });
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'gpu-box' } });
    await settle();
    expect(memoryField()).toHaveValue('');
    expect(screen.getByText(/No earlier run of this graph reported its peak memory/)).toBeInTheDocument();
    fireEvent.change(memoryField(), { target: { value: 'lots' } });
    await submit();
    expect(screen.getByRole('alert')).toHaveTextContent('The memory budget must be a number above 0 and at most 4096.');
    expect(api.submitRun).not.toHaveBeenCalled();
    fireEvent.change(memoryField(), { target: { value: '6' } });
    await submit();
    expect(sentRequest().options.target).toEqual({ worker: 'gpu-box', devices: ['cuda:1'], memory_gib: 6 });
  });

  it('leaves the budget to the server for a blank field', async () => {
    workersMock.fetchWorkers.mockResolvedValue([sharedWorker()]);
    patchTab({ graphDevice: 'cuda:1' });
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'gpu-box' } });
    await settle();
    await submit();
    expect(sentRequest().options.target).toEqual({ worker: 'gpu-box', devices: ['cuda:1'] });
  });

  it('says it is looking, then why there is no suggestion when the lookup fails', async () => {
    workersMock.fetchWorkers.mockResolvedValue([sharedWorker()]);
    let fail: (reason: unknown) => void = () => {};
    api.estimateMemory.mockReturnValueOnce(new Promise((_, reject) => { fail = reject; }));
    patchTab({ graphDevice: 'cuda:1' });
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'gpu-box' } });
    await settle();
    expect(screen.getByText('Looking up the last run of this graph...')).toBeInTheDocument();
    await act(async () => fail('offline'));
    expect(screen.getByText('Could not look up a suggestion: offline')).toBeInTheDocument();
    expect(memoryField()).toHaveValue('');
  });

  it('reads a failed lookup that is an Error', async () => {
    workersMock.fetchWorkers.mockResolvedValue([sharedWorker()]);
    api.estimateMemory.mockRejectedValueOnce(new Error('graph.nodes is empty'));
    patchTab({ graphDevice: 'cuda:1' });
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'gpu-box' } });
    await settle();
    expect(screen.getByText('Could not look up a suggestion: graph.nodes is empty')).toBeInTheDocument();
  });

  it('asks for a budget for any when a connected worker shares the kind, and never sends wait', async () => {
    workersMock.fetchWorkers.mockResolvedValue([sharedWorker()]);
    patchTab({ graphDevice: 'cuda' });
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'any' } });
    await settle();
    expect(screen.queryByRole('checkbox', { name: /Wait in the queue/ })).toBeNull();
    fireEvent.change(memoryField(), { target: { value: '3' } });
    await submit();
    expect(sentRequest().options).toMatchObject({ target: { worker: 'any', memory_gib: 3 }, device: 'cuda' });
    expect(sentRequest().options.target).not.toHaveProperty('wait');
  });

  it('renders the budget field in zh-TW', async () => {
    useI18n.setState({ locale: 'zh-TW' });
    workersMock.fetchWorkers.mockResolvedValue([sharedWorker()]);
    patchTab({ graphDevice: 'cuda:1' });
    await open();
    fireEvent.change(screen.getByRole('combobox', { name: '目標' }), { target: { value: 'gpu-box' } });
    await settle();
    expect(screen.getByRole('textbox', { name: '記憶體預算（每個裝置，GiB）' })).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: '目前放不下時在佇列中等待' })).toBeInTheDocument();
  });
});
