import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { cleanup, render, screen, fireEvent, waitFor } from '@testing-library/react';
import type { ParamDefinition } from '../../types';
import type { WorkerInfo } from '../../api/workers';
import { useI18n } from '../../i18n';
import { useWorkerStore } from '../../store/workerStore';
import { DeviceParamSelect, isDeviceParam, workerDeviceChoices } from './DeviceParamSelect';
import { ParamField } from './ParamField';

const deviceParam = (over: Partial<ParamDefinition> = {}): ParamDefinition => ({
  name: 'device',
  param_type: 'select',
  default: 'auto',
  description: '',
  options: ['auto', 'cpu', 'mps'],
  min_value: null,
  max_value: null,
  ...over,
});

const worker = (name: string, devices: [string, string][]): WorkerInfo =>
  ({
    id: name,
    name,
    status: 'online',
    draining: false,
    version: '3.1.0',
    fingerprint_digest: null,
    platform: {},
    mounts: {},
    devices: devices.map(([value, kind]) => ({
      value,
      kind,
      index: null,
      name: value,
      memory: null,
    })),
    running: [],
    connected_at: 1,
    last_heartbeat_at: 1,
  }) as WorkerInfo;

const initial = useWorkerStore.getState();
let refresh: ReturnType<typeof vi.fn<() => Promise<void>>>;

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  refresh = vi.fn<() => Promise<void>>(async () => undefined);
  useWorkerStore.setState({ workers: [], loaded: true, refresh });
});

afterEach(() => {
  // Unmount first: resetting the store under a mounted select would
  // re-render it outside act().
  cleanup();
  useWorkerStore.setState(initial, true);
});

function select(): HTMLSelectElement {
  return screen.getByRole('combobox') as HTMLSelectElement;
}

function optionTexts(): string[] {
  return Array.from(select().querySelectorAll('option')).map((o) => o.textContent ?? '');
}

describe('isDeviceParam', () => {
  it('is a select named device with a device vocabulary', () => {
    expect(isDeviceParam(deviceParam())).toBe(true);
    expect(isDeviceParam(deviceParam({ name: 'mode' }))).toBe(false);
    expect(isDeviceParam(deviceParam({ param_type: 'string' }))).toBe(false);
    // A `device` select that is not about compute devices (a camera index).
    expect(isDeviceParam(deviceParam({ options: ['0', '1'] }))).toBe(false);
  });
});

describe('workerDeviceChoices', () => {
  it('lists accelerators the local list lacks, merging workers per value', () => {
    const choices = workerDeviceChoices(
      [
        worker('gpu-a', [['cpu', 'cpu'], ['cuda:0', 'cuda'], ['cuda:1', 'cuda']]),
        worker('gpu-b', [['cuda:0', 'cuda']]),
        worker('mac', [['mps', 'mps'], ['mps:0', 'mps']]),
        worker('dup', [['cuda:0', 'cuda'], ['cuda:0', 'cuda']]),
      ],
      ['auto', 'cpu', 'mps'],
    );
    expect(choices).toEqual([
      { value: 'cuda:0', workers: ['gpu-a', 'gpu-b', 'dup'] },
      { value: 'cuda:1', workers: ['gpu-a'] },
    ]);
  });

  it('treats mps and mps:0 as one device', () => {
    expect(
      workerDeviceChoices([worker('a', [['mps:0', 'mps']]), worker('b', [['mps', 'mps']])], ['cpu']),
    ).toEqual([{ value: 'mps:0', workers: ['a', 'b'] }]);
  });
});

describe('DeviceParamSelect', () => {
  it('B4: shows a stored cuda:0 the server does not list, never auto', () => {
    render(
      <DeviceParamSelect param={deviceParam()} value="cuda:0" onChange={vi.fn()} displayLabel="device" />,
    );
    expect(select().value).toBe('cuda:0');
    expect(optionTexts()).toEqual(['auto', 'cpu', 'mps', 'cuda:0 (not on this server)']);
    const hint = screen.getByText(/This server has no cuda:0/);
    expect(select().getAttribute('aria-describedby')).toBe(hint.id);
  });

  it('a value this server lists needs no extra option or hint', () => {
    render(<DeviceParamSelect param={deviceParam()} value="mps" onChange={vi.fn()} displayLabel="device" />);
    expect(select().value).toBe('mps');
    expect(optionTexts()).toEqual(['auto', 'cpu', 'mps']);
    expect(select().getAttribute('aria-describedby')).toBeNull();
  });

  it('mps:0 stored on a server listing mps is the same device', () => {
    render(<DeviceParamSelect param={deviceParam()} value="mps:0" onChange={vi.fn()} displayLabel="device" />);
    expect(select().value).toBe('mps');
    expect(optionTexts()).toEqual(['auto', 'cpu', 'mps']);
    expect(screen.queryByText(/This server has no/)).toBeNull();
  });

  it('falls back to the default, and to an empty value without one', () => {
    const { unmount } = render(
      <DeviceParamSelect param={deviceParam()} value={undefined} onChange={vi.fn()} displayLabel="device" />,
    );
    expect(select().value).toBe('auto');
    unmount();
    render(
      <DeviceParamSelect
        param={deviceParam({ default: null })}
        value={null}
        onChange={vi.fn()}
        displayLabel="device"
      />,
    );
    expect(optionTexts()).toEqual(['auto', 'cpu', 'mps']);
    expect(screen.queryByText(/This server has no/)).toBeNull();
  });

  it("U1: offers the workers' cards by their in-run names", () => {
    useWorkerStore.setState({
      workers: [worker('meow', [['cpu', 'cpu'], ['cuda:0', 'cuda'], ['cuda:1', 'cuda']])],
    });
    const onChange = vi.fn();
    render(<DeviceParamSelect param={deviceParam()} value="auto" onChange={onChange} displayLabel="device" />);
    const group = select().querySelector('optgroup');
    expect(group?.getAttribute('label')).toBe('Worker devices');
    expect(optionTexts()).toEqual(['auto', 'cpu', 'mps', 'cuda:0 · meow', 'cuda:1 · meow']);
    fireEvent.change(select(), { target: { value: 'cuda:1' } });
    expect(onChange).toHaveBeenCalledWith('device', 'cuda:1');
  });

  it('a stored mps on a server without MPS selects a worker mps:0', () => {
    useWorkerStore.setState({ workers: [worker('mac', [['mps:0', 'mps']])] });
    render(
      <DeviceParamSelect
        param={deviceParam({ options: ['auto', 'cpu'] })}
        value="mps"
        onChange={vi.fn()}
        displayLabel="device"
      />,
    );
    expect(select().value).toBe('mps:0');
    expect(optionTexts()).toEqual(['auto', 'cpu', 'mps:0 · mac']);
    expect(screen.getByText(/This server has no mps/)).toBeTruthy();
  });

  it('a stored worker card is selected in the worker group, with the hint', () => {
    useWorkerStore.setState({ workers: [worker('meow', [['cuda:0', 'cuda']])] });
    render(<DeviceParamSelect param={deviceParam()} value="cuda:0" onChange={vi.fn()} displayLabel="device" />);
    expect(select().value).toBe('cuda:0');
    expect(optionTexts()).toEqual(['auto', 'cpu', 'mps', 'cuda:0 · meow']);
    expect(screen.getByText(/This server has no cuda:0/)).toBeTruthy();
  });

  it('reads the worker list once when nothing has yet', async () => {
    useWorkerStore.setState({ loaded: false, workers: [worker('meow', [['cuda:0', 'cuda']])] });
    render(<DeviceParamSelect param={deviceParam()} value="auto" onChange={vi.fn()} displayLabel="device" />);
    await waitFor(() => expect(refresh).toHaveBeenCalledTimes(1));
    // Not listed until the read answers.
    expect(optionTexts()).toEqual(['auto', 'cpu', 'mps']);
  });

  it('does not read again when the list is loaded', () => {
    render(<DeviceParamSelect param={deviceParam()} value="auto" onChange={vi.fn()} displayLabel="device" />);
    expect(refresh).not.toHaveBeenCalled();
  });

  it('speaks zh-TW', () => {
    useI18n.setState({ locale: 'zh-TW' });
    useWorkerStore.setState({ workers: [worker('meow', [['cuda:1', 'cuda']])] });
    render(<DeviceParamSelect param={deviceParam()} value="cuda:0" onChange={vi.fn()} displayLabel="device" />);
    expect(optionTexts()).toContain('cuda:0（本伺服器沒有）');
    expect(select().querySelector('optgroup')?.getAttribute('label')).toBe('worker 的裝置');
    expect(screen.getByText(/本伺服器沒有 cuda:0/)).toBeTruthy();
  });
});

describe('ParamField routes a device select here', () => {
  it('renders the stored device instead of auto', () => {
    render(<ParamField param={deviceParam()} value="cuda:0" onChange={vi.fn()} />);
    expect(select().value).toBe('cuda:0');
    expect(screen.getByText('cuda:0 (not on this server)')).toBeTruthy();
  });
});
