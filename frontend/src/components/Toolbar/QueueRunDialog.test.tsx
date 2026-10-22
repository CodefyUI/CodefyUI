import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, screen, waitFor, within } from '@testing-library/react';
import * as rest from '../../api/rest';
import type { DeviceInfo, MemoryEstimate, StaticMemoryEstimate } from '../../api/rest';
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
import { _resetWorkerStoreForTesting, useWorkerStore } from '../../store/workerStore';
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

const CARD_MEMORY = { total_bytes: 80 * 1024 ** 3, free_bytes: 70 * 1024 ** 3, source: 'nvml' };

/** A worker whose cuda:1 is shared by memory (#788), with a memory total
 * unless *total* is false (an MPS card reports none). */
function sharedWorker(name = 'gpu-box', total = true): WorkerInfo {
  const base = worker({ name, memory_budget: true });
  return {
    ...base,
    devices: base.devices.map((d) => (d.value === 'cuda:1'
      ? { ...d, policy: 'shared', memory: total ? CARD_MEMORY : null }
      : d)),
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
    // Says so, with a way to add one (#840).
    expect(screen.getByText(/No worker is connected to this server yet/)).toBeInTheDocument();
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

  it('sends node provenance only when it is asked for (#344)', async () => {
    await open();
    const choice = screen.getByRole('combobox', { name: 'Data and model provenance' });
    expect(choice).toHaveValue('off');
    expect(screen.queryByText(/reads each file once/)).toBeNull();
    fireEvent.change(choice, { target: { value: 'content' } });
    expect(screen.getByText(/reads each file once/)).toBeInTheDocument();
    await submit();
    expect(sentRequest().options).toMatchObject({ provenance: 'content' });
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
  it('lists workers with status and devices, offline and disconnected ones too, disabling the drained', async () => {
    workersMock.fetchWorkers.mockResolvedValue([
      worker({ name: 'gpu-box' }),
      worker({ name: 'slow-box', status: 'degraded', devices: [] }),
      worker({ name: 'busy-box', status: 'draining', draining: true }),
      worker({ name: 'gone-box', status: 'disconnected' }),
      worker({ name: 'off-box', status: 'disconnected', draining: true, ssh_enabled: false }),
      worker({ name: 'cold-box', status: 'offline' }),
    ]);
    await open();
    const options = within(targetSelect()).getAllByRole('option') as HTMLOptionElement[];
    expect(options.map((o) => o.textContent)).toEqual([
      'This server',
      'gpu-box (Online): cpu, cuda:0, cuda:1, cuda:2',
      'slow-box (Degraded): no devices',
      'busy-box (Draining): cpu, cuda:0, cuda:1, cuda:2',
      'gone-box (Disconnected): cpu, cuda:0, cuda:1, cuda:2',
      'off-box (Disabled): cpu, cuda:0, cuda:1, cuda:2',
      'cold-box (Offline): cpu, cuda:0, cuda:1, cuda:2',
      'Any available',
      'A one-shot worker (cdui agent start --one-shot)',
    ]);
    expect(options[3]).toBeDisabled();
    // A disabled SSH worker reads Disabled, as in the Workers panel, and is
    // drained: no run goes there, and a hint says why (#840 review).
    expect(options[5]).toBeDisabled();
    expect(screen.getByText('A disabled SSH worker takes no runs: enable it in the Workers panel first.')).toBeInTheDocument();
    expect(options[4]).toBeEnabled();
    expect(screen.queryByText(/is offline/)).toBeNull();
    expect(screen.queryByText(/No worker is connected/)).toBeNull();
    // An offline worker can be chosen; the run waits for it.
    fireEvent.change(targetSelect(), { target: { value: 'cold-box' } });
    expect(screen.getByText('cold-box is offline: the run waits in the queue until it connects.')).toBeInTheDocument();
    // A disconnected one too, as the server accepts it (#840).
    fireEvent.change(targetSelect(), { target: { value: 'gone-box' } });
    expect(screen.getByText('gone-box is disconnected: the run waits in the queue until it reconnects.')).toBeInTheDocument();
  });

  it('starts a worker on its accelerator when the device only follows Settings (#840)', async () => {
    workersMock.fetchWorkers.mockResolvedValue([worker({ name: 'gpu-box' })]);
    await open();
    // This server: the Settings device.
    expect(screen.getByRole('radio', { name: /cpu/ })).toBeChecked();
    fireEvent.change(targetSelect(), { target: { value: 'gpu-box' } });
    expect(screen.getByRole('checkbox', { name: /^cuda:0 / })).toBeChecked();
    expect(screen.getByRole('checkbox', { name: /^cpu/ })).not.toBeChecked();
  });

  it('keeps a device the tab names on a worker', async () => {
    workersMock.fetchWorkers.mockResolvedValue([worker({ name: 'gpu-box' })]);
    patchTab({ graphDevice: 'cpu' });
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'gpu-box' } });
    expect(screen.getByRole('checkbox', { name: /^cpu/ })).toBeChecked();
  });

  it('opens the Workers panel from the no-worker hint, closing itself', async () => {
    _resetWorkerStoreForTesting();
    await open();
    fireEvent.click(screen.getByRole('button', { name: 'Add a worker' }));
    expect(onClose).toHaveBeenCalled();
    expect(useWorkerStore.getState().panelOpen).toBe(true);
    act(() => useWorkerStore.getState().closePanel());
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

    // A worker records outputs too (#789): they stay on it, as the hint says.
    const record = screen.getByRole('checkbox', { name: 'Record node outputs' });
    expect(record).toBeEnabled();
    expect(record).toBeChecked();
    expect(screen.getByText(/Recorded outputs stay on the worker/)).toBeInTheDocument();
    fireEvent.click(record);
    expect(screen.queryByText(/Recorded outputs stay on the worker/)).toBeNull();
    fireEvent.click(record);

    await submit();
    expect(sentRequest().options).toMatchObject({
      target: { worker: 'gpu-box', devices: ['cuda:0', 'cuda:1', 'cuda:2'] },
      device: 'cuda:0',
      record_outputs: true,
    });
  });

  it('replaces the set with a device of another kind, and unchecks one already in it', async () => {
    workersMock.fetchWorkers.mockResolvedValue([worker({ name: 'gpu-box' })]);
    patchTab({ graphDevice: 'cpu' });
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'gpu-box' } });
    // The tab's device is cpu, which the worker has.
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
  it('starts on the accelerator a worker offers when the device only follows Settings (#840 review)', async () => {
    workersMock.fetchWorkers.mockResolvedValue([worker({ name: 'gpu-box' })]);
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'any' } });
    const group = within(screen.getByRole('group', { name: 'Devices' }));
    expect(group.getByRole('radio', { name: 'CUDA GPU' })).toBeChecked();
    await submit();
    expect(sentRequest().options).toMatchObject({ target: { worker: 'any' }, device: 'cuda' });
  });

  it('starts on a kind a worker that takes runs offers, not a drained one\'s (#840 review)', async () => {
    workersMock.fetchWorkers.mockResolvedValue([
      worker({ name: 'busy-box', status: 'draining', draining: true }),
      worker({ name: 'off-box', ssh_enabled: false }),
    ]);
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'any' } });
    const group = within(screen.getByRole('group', { name: 'Devices' }));
    // CUDA is still offered (a choice), but not where the run starts.
    expect(group.getByRole('radio', { name: 'CUDA GPU' })).not.toBeChecked();
    expect(group.getByRole('radio', { name: 'CPU' })).toBeChecked();
  });

  it('keeps the kind of a device the tab names', async () => {
    workersMock.fetchWorkers.mockResolvedValue([worker({ name: 'gpu-box' })]);
    patchTab({ graphDevice: 'cpu' });
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'any' } });
    expect(within(screen.getByRole('group', { name: 'Devices' })).getByRole('radio', { name: 'CPU' })).toBeChecked();
  });

  it('asks for a kind, defaulting to the graph device kind, and sends no device set', async () => {
    workersMock.fetchWorkers.mockResolvedValue([worker({ name: 'gpu-box' })]);
    patchTab({ graphDevice: 'cuda' });
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'any' } });
    const group = within(screen.getByRole('group', { name: 'Devices' }));
    expect(group.getAllByRole('radio').map((r) => r.closest('label')!.textContent)).toEqual(['CPU', 'CUDA GPU']);
    expect(group.getByRole('radio', { name: 'CUDA GPU' })).toBeChecked();
    expect(screen.getByText(/first connected worker with a free device/)).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'Record node outputs' })).toBeEnabled();
    fireEvent.click(group.getByRole('radio', { name: 'CPU' }));
    await submit();
    expect(sentRequest().options).toMatchObject({ target: { worker: 'any' }, device: 'cpu', record_outputs: true });
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
    // Its agent exits with the run (#789): the box is off and says why.
    const record = screen.getByRole('checkbox', { name: 'Record node outputs' });
    expect(record).toBeDisabled();
    expect(record).not.toBeChecked();
    expect(screen.getByText(/A one-shot worker exits when its run ends/)).toBeInTheDocument();
    expect(screen.queryByText(/Recorded outputs stay on the worker/)).toBeNull();
    await submit();
    expect(sentRequest().options).toMatchObject({
      target: { worker: 'any', pool: 'one-shot' }, device: 'cuda', record_outputs: false,
    });
  });

  it('does not record outputs on a named one-shot worker, and does again elsewhere', async () => {
    workersMock.fetchWorkers.mockResolvedValue([
      worker({ name: 'once', one_shot: { run_id: null, took: null } }),
      worker({ name: 'box' }),
    ]);
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'once' } });
    const record = screen.getByRole('checkbox', { name: 'Record node outputs' });
    expect(record).toBeDisabled();
    expect(record).not.toBeChecked();
    fireEvent.change(targetSelect(), { target: { value: 'box' } });
    expect(record).toBeEnabled();
    expect(record).toBeChecked();
    expect(screen.queryByText(/A one-shot worker exits/)).toBeNull();
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
    // Not "no worker": the list is unknown.
    expect(screen.queryByText(/No worker is connected/)).toBeNull();
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

// ResNet-18 at batch 128 under SGD with momentum, in float32: 11,173,962 x
// 12 bytes of weights, gradients and momentum.
const STATIC_WEIGHTS = 11173962 * 12;
const STATIC_ACTIVATIONS = 1.3 * 1024 ** 3;
const STATIC: StaticMemoryEstimate = {
  bytes: Math.ceil((STATIC_WEIGHTS + STATIC_ACTIVATIONS) * 1.2 + 128 * 1024 ** 2),
  confidence: 'medium', parameters: 11173962, trainable_parameters: 11173962,
  training: true, optimizer: 'SGD', multiplier: 3, batch_size: 128, eval_batch_size: 512, precision: 'fp32',
  weights_bytes: STATIC_WEIGHTS, activation_bytes: STATIC_ACTIVATIONS, workspace_bytes: 128 * 1024 ** 2,
  models: [{ type: 'SequentialModel', parameters: 11173962, activations_known: true, probe_input: [[3, 32, 32]] }],
  uncounted: [], notes: [], batch_assumed: false, activation_phase: 'training',
};

describe('QueueRunDialog: memory budget (#788)', () => {
  const memoryField = () => screen.getByRole('textbox', { name: 'Memory budget (GiB per device)' });

  it('asks for a budget only on a shared device, filled with the suggestion', async () => {
    workersMock.fetchWorkers.mockResolvedValue([sharedWorker()]);
    api.estimateMemory.mockResolvedValue({
      suggested_gib: 2.4, source: 'history', peak_bytes: 2 * 1024 ** 3, from_run: 'r1',
    });
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'gpu-box' } });
    // A worker starts on its first accelerator (#840), and cuda:0 is
    // exclusive: no budget asked for.
    expect(screen.getByRole('checkbox', { name: /^cuda:0 / })).toBeChecked();
    expect(screen.queryByRole('textbox', { name: /Memory budget/ })).toBeNull();
    fireEvent.click(screen.getByRole('checkbox', { name: /^cuda:0 / }));
    expect(screen.queryByRole('textbox', { name: /Memory budget/ })).toBeNull();
    fireEvent.click(screen.getByRole('checkbox', { name: /^cuda:1 / }));
    await settle();
    expect(api.estimateMemory).toHaveBeenCalledWith(GRAPH, { device: 'cuda:1', target: { devices: ['cuda:1'] } });
    // The estimate is asked without the session's secrets.
    expect(serialize).toHaveBeenCalledWith();
    expect(serialize).not.toHaveBeenCalledWith({ keepSecrets: true });
    expect(memoryField()).toHaveValue('2.4');
    expect(screen.getByText('Suggested from the last run of this graph: its peak was 2.0 GiB, plus 20%.')).toBeInTheDocument();
    // #794: a measured peak is tagged as one.
    expect(screen.getByText('Measured')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Wait in the queue when it does not fit now' }));
    await submit();
    // Left as suggested: the server derives the same default and keeps
    // the run it came from.
    expect(sentRequest().options.target).toEqual({
      worker: 'gpu-box', devices: ['cuda:1'], wait: true,
    });
  });

  it('asks again when the run would use another number of devices, and keeps only the last answer', async () => {
    workersMock.fetchWorkers.mockResolvedValue([sharedWorker()]);
    let late: (value: MemoryEstimate) => void = () => {};
    api.estimateMemory
      .mockReturnValueOnce(new Promise((resolve) => { late = resolve; }))
      .mockResolvedValueOnce({ suggested_gib: 5, source: 'history', peak_bytes: 4 * 1024 ** 3, from_run: 'r2' });
    patchTab({ graphDevice: 'cuda:1' });
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'gpu-box' } });
    await settle();
    fireEvent.click(screen.getByRole('checkbox', { name: /^cuda:0 / }));
    await settle();
    expect(api.estimateMemory).toHaveBeenLastCalledWith(GRAPH, { device: 'cuda:0', target: { devices: ['cuda:0', 'cuda:1'] } });
    expect(memoryField()).toHaveValue('5');
    await act(async () => late({ suggested_gib: 1, source: 'history', peak_bytes: 1024 ** 3, from_run: 'r1' }));
    expect(memoryField()).toHaveValue('5');
    // Typed, even to the same number: declared.
    fireEvent.change(memoryField(), { target: { value: '5.0' } });
    await submit();
    expect(sentRequest().options.target).toMatchObject({ memory_gib: 5 });
  });

  it('drops a failed lookup for a shape it no longer asks about', async () => {
    workersMock.fetchWorkers.mockResolvedValue([sharedWorker()]);
    let fail: (reason: unknown) => void = () => {};
    api.estimateMemory
      .mockReturnValueOnce(new Promise((_, reject) => { fail = reject; }))
      .mockResolvedValueOnce({ suggested_gib: 5, source: 'history', peak_bytes: 4 * 1024 ** 3, from_run: 'r2' });
    patchTab({ graphDevice: 'cuda:1' });
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'gpu-box' } });
    await settle();
    fireEvent.click(screen.getByRole('checkbox', { name: /^cuda:0 / }));
    await settle();
    await act(async () => fail('offline'));
    expect(screen.queryByText(/Could not look up a suggestion/)).toBeNull();
    expect(memoryField()).toHaveValue('5');
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

  it('offers a static estimate as one and sends it marked as one when accepted (#794)', async () => {
    workersMock.fetchWorkers.mockResolvedValue([sharedWorker()]);
    api.estimateMemory.mockResolvedValue({
      suggested_gib: null, source: null, peak_bytes: null, from_run: null, confidence: 'medium',
      history: [], static_estimate_bytes: STATIC.bytes, estimate_gib: 1.86, static: STATIC,
    });
    patchTab({ graphDevice: 'cuda:1' });
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'gpu-box' } });
    await settle();
    expect(memoryField()).toHaveValue('1.86');
    expect(screen.getByText('Estimate')).toBeInTheDocument();
    expect(screen.queryByText('Measured')).toBeNull();
    expect(screen.getByText(/^Not a measured peak/)).toHaveTextContent(
      'Not a measured peak: no earlier run of this graph on these devices reported one. Estimated from the model: '
      + '11,173,962 parameters x 4 bytes (weights) = 42.6 MB, 11,173,962 trainable parameters x 8 bytes (gradients, '
      + 'optimizer state) = 85.3 MB, activations of a training step at batch 128 about 1.3 GB, 128.0 MB of library '
      + 'workspace, and 291.8 MB of allocator reserve (20%). Queued as it is, the run is capped to this estimate '
      + '(marked est. in Runs); change it if you know better. The next run of this graph is measured.',
      { normalizeWhitespace: true },
    );
    await submit();
    // The server never applies an estimate on its own: accepted, it goes
    // with the run as a declared budget that says where it came from.
    expect(sentRequest().options.target).toEqual({
      worker: 'gpu-box', devices: ['cuda:1'], memory_gib: 1.86, memory_source: 'estimate',
    });
  });

  it('sends an edited estimate as a plain declared budget (#794)', async () => {
    workersMock.fetchWorkers.mockResolvedValue([sharedWorker()]);
    api.estimateMemory.mockResolvedValue({
      ...NO_ESTIMATE, confidence: 'medium', static_estimate_bytes: STATIC.bytes, estimate_gib: 1.86, static: STATIC,
    });
    patchTab({ graphDevice: 'cuda:1' });
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'gpu-box' } });
    await settle();
    fireEvent.change(memoryField(), { target: { value: '3' } });
    await submit();
    expect(sentRequest().options.target).toEqual({ worker: 'gpu-box', devices: ['cuda:1'], memory_gib: 3 });
  });

  it('sends nothing for an accepted estimate cleared to blank (#794)', async () => {
    workersMock.fetchWorkers.mockResolvedValue([sharedWorker()]);
    api.estimateMemory.mockResolvedValue({
      ...NO_ESTIMATE, confidence: 'medium', static_estimate_bytes: STATIC.bytes, estimate_gib: 1.86, static: STATIC,
    });
    patchTab({ graphDevice: 'cuda:1' });
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'gpu-box' } });
    await settle();
    fireEvent.change(memoryField(), { target: { value: '' } });
    await submit();
    expect(sentRequest().options.target).toEqual({ worker: 'gpu-box', devices: ['cuda:1'] });
  });

  it('explains a shared MPS card with no memory total instead of asking for a budget (#788)', async () => {
    const base = worker({ name: 'mac', memory_budget: true });
    const mac: WorkerInfo = {
      ...base,
      devices: [
        { value: 'cpu', kind: 'cpu', index: null, name: '', memory: null },
        { value: 'mps', kind: 'mps', index: 0, name: 'Apple M3', memory: null, policy: 'shared' },
      ],
    };
    workersMock.fetchWorkers.mockResolvedValue([mac]);
    patchTab({ graphDevice: 'mps' });
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'mac' } });
    await settle();
    expect(screen.queryByRole('textbox', { name: /Memory budget/ })).toBeNull();
    expect(api.estimateMemory).not.toHaveBeenCalled();
    expect(screen.getByRole('note')).toHaveTextContent(
      'mps is shared by memory but reports no memory total (an MPS card reports none), so it cannot admit a run '
      + 'by budget and refuses this one. Set it to exclusive in Workers to run here.',
    );
    await submit();
    expect(sentRequest().options.target).toEqual({ worker: 'mac', devices: ['mps'] });
  });

  it('keeps the field for a CUDA card that has not reported its memory yet (#794 review 2)', async () => {
    workersMock.fetchWorkers.mockResolvedValue([sharedWorker('gpu-box', false)]);
    patchTab({ graphDevice: 'cuda:1' });
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'gpu-box' } });
    await settle();
    expect(memoryField()).toBeInTheDocument();
    expect(screen.queryByRole('note')).toBeNull();
    expect(screen.getByText(
      "cuda:1 has not reported its memory yet (it comes with the worker's next heartbeat): the budget is judged "
      + 'against the card once it has.',
    )).toBeInTheDocument();
  });

  it('shows a rough estimate with why it is not offered, and leaves the field empty (#794)', async () => {
    workersMock.fetchWorkers.mockResolvedValue([sharedWorker()]);
    api.estimateMemory.mockResolvedValue({
      suggested_gib: null, source: null, peak_bytes: null, from_run: null, confidence: 'low',
      history: [], static_estimate_bytes: 3 * 1024 ** 3,
      static: {
        ...STATIC, bytes: 3 * 1024 ** 3, confidence: 'low', uncounted: ['BradleyTerryTrain'], batch_assumed: true,
        batch_size: 32,
        models: [
          { type: 'VLAModel', parameters: 10, activations_known: false, probe_input: null },
          { type: 'Empty', parameters: 0, activations_known: false, probe_input: null },
        ],
      },
    });
    patchTab({ graphDevice: 'cuda:1' });
    await open();
    fireEvent.change(targetSelect(), { target: { value: 'gpu-box' } });
    await settle();
    expect(memoryField()).toHaveValue('');
    expect(screen.getByText('Rough estimate')).toBeInTheDocument();
    expect(screen.getByText(/^About 3.0 GiB/)).toHaveTextContent(
      'About 3.0 GiB, too rough to offer: could not count BradleyTerryTrain, no input shape for the activations of '
      + 'VLAModel, and no batch size in the graph (assumed 32). Declare a budget.',
    );
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
    expect(screen.getByText('Looking up earlier runs of this graph and estimating its memory...')).toBeInTheDocument();
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
