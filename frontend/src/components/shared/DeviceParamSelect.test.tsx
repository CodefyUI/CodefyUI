import { Profiler } from 'react';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, cleanup, render, screen, fireEvent, waitFor } from '@testing-library/react';
import type { ParamDefinition } from '../../types';
import type { WorkerInfo } from '../../api/workers';
import { useI18n } from '../../i18n';
import { useWorkerStore } from '../../store/workerStore';
import {
  DeviceParamSelect,
  _resetWorkerListRequestForTesting,
  inRunChoicesKey,
  isDeviceParam,
} from './DeviceParamSelect';
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

const gpuBox = (cards: number) =>
  worker(
    `gpu-${cards}`,
    [['cpu', 'cpu'], ...Array.from({ length: cards }, (_, i): [string, string] => [`cuda:${i}`, 'cuda'])],
  );

const initial = useWorkerStore.getState();
let refresh: ReturnType<typeof vi.fn<() => Promise<void>>>;

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  refresh = vi.fn<() => Promise<void>>(async () => undefined);
  useWorkerStore.setState({ workers: [], loaded: true, refresh });
  _resetWorkerListRequestForTesting();
});

afterEach(() => {
  // Unmount first: resetting the store under a mounted select would
  // re-render it outside act().
  cleanup();
  useWorkerStore.setState(initial, true);
});

function select(index = 0): HTMLSelectElement {
  return screen.getAllByRole('combobox')[index] as HTMLSelectElement;
}

function optionTexts(): string[] {
  return Array.from(select().querySelectorAll('option')).map((o) => o.textContent ?? '');
}

const renderSelect = (value: unknown, over: Partial<ParamDefinition> = {}) =>
  render(<DeviceParamSelect param={deviceParam(over)} value={value} onChange={vi.fn()} displayLabel="device" />);

describe('isDeviceParam', () => {
  it('is a select named device with a device vocabulary', () => {
    expect(isDeviceParam(deviceParam())).toBe(true);
    expect(isDeviceParam(deviceParam({ name: 'mode' }))).toBe(false);
    expect(isDeviceParam(deviceParam({ param_type: 'string' }))).toBe(false);
    expect(isDeviceParam(deviceParam({ options: ['0', '1'] }))).toBe(false);
  });
});

describe('inRunChoicesKey', () => {
  it('names cuda:0..N-1 for the most cards any worker has, never a physical card', () => {
    expect(inRunChoicesKey([gpuBox(2), gpuBox(4)], ['auto', 'cpu', 'mps'])).toBe(
      'cuda:0 cuda:1 cuda:2 cuda:3',
    );
  });

  it('adds mps when a worker offers it and this server does not', () => {
    const mac = worker('mac', [['cpu', 'cpu'], ['mps', 'mps']]);
    expect(inRunChoicesKey([mac], ['auto', 'cpu'])).toBe('mps');
    expect(inRunChoicesKey([mac], ['auto', 'cpu', 'mps'])).toBe('');
  });

  it('leaves out names this server lists (cuda:0 is its bare cuda on one card)', () => {
    expect(inRunChoicesKey([gpuBox(2)], ['auto', 'cpu', 'cuda'])).toBe('cuda:1');
    expect(inRunChoicesKey([gpuBox(3)], ['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1'])).toBe('cuda:2');
  });

  it('is empty without workers', () => {
    expect(inRunChoicesKey([], ['auto', 'cpu'])).toBe('');
  });
});

describe('DeviceParamSelect', () => {
  it('B4: shows a stored cuda:0 the server does not list, never auto', () => {
    renderSelect('cuda:0');
    expect(select().value).toBe('cuda:0');
    expect(optionTexts()).toEqual(['auto', 'cpu', 'mps', 'cuda:0 (not on this server)']);
    const hint = screen.getByText('This server has no cuda:0: a run here falls back to another device and notes it in the run log.');
    expect(select().getAttribute('aria-describedby')).toBe(hint.id);
  });

  it('a value this server lists needs no extra option or hint', () => {
    renderSelect('mps');
    expect(select().value).toBe('mps');
    expect(optionTexts()).toEqual(['auto', 'cpu', 'mps']);
    expect(select().getAttribute('aria-describedby')).toBeNull();
  });

  it('mps:0 stored on a server listing mps is the same device', () => {
    renderSelect('mps:0');
    expect(select().value).toBe('mps');
    expect(optionTexts()).toEqual(['auto', 'cpu', 'mps']);
    expect(screen.queryByText(/This server has no/)).toBeNull();
  });

  it('review 3: cuda:0 stored on a one-card server listing cuda is that card', () => {
    renderSelect('cuda:0', { options: ['auto', 'cpu', 'cuda'] });
    expect(select().value).toBe('cuda');
    expect(optionTexts()).toEqual(['auto', 'cpu', 'cuda']);
    expect(screen.queryByText(/This server has no/)).toBeNull();
  });

  it('cuda:0 on a multi-card server is its own option, not the bare cuda', () => {
    renderSelect('cuda:0', { options: ['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1'] });
    expect(select().value).toBe('cuda:0');
  });

  it('falls back to the default, and to an empty value without one', () => {
    const { unmount } = renderSelect(undefined);
    expect(select().value).toBe('auto');
    unmount();
    renderSelect(null, { default: null });
    expect(optionTexts()).toEqual(['auto', 'cpu', 'mps']);
    expect(screen.queryByText(/This server has no/)).toBeNull();
  });

  it('review 2: offers in-run names with the numbering hint, not physical cards', () => {
    useWorkerStore.setState({ workers: [gpuBox(2)] });
    const onChange = vi.fn();
    render(<DeviceParamSelect param={deviceParam()} value="auto" onChange={onChange} displayLabel="device" />);
    expect(select().querySelector('optgroup')?.getAttribute('label')).toBe("Inside a run's device set");
    expect(optionTexts()).toEqual(['auto', 'cpu', 'mps', 'cuda:0', 'cuda:1']);
    expect(optionTexts().some((text) => text.includes('gpu-2'))).toBe(false);
    const numbering = screen.getByText(/sees the cards of its device set as cuda:0, cuda:1/);
    expect(select().getAttribute('aria-describedby')).toBe(numbering.id);
    fireEvent.change(select(), { target: { value: 'cuda:1' } });
    expect(onChange).toHaveBeenCalledWith('device', 'cuda:1');
  });

  it('a stored in-run name is selected in its group, with both hints', () => {
    useWorkerStore.setState({ workers: [gpuBox(1)] });
    renderSelect('cuda:0');
    expect(select().value).toBe('cuda:0');
    expect(optionTexts()).toEqual(['auto', 'cpu', 'mps', 'cuda:0']);
    expect(select().getAttribute('aria-describedby')?.split(' ')).toHaveLength(2);
  });

  it('a stored cuda:5 beyond every worker is listed as stored', () => {
    useWorkerStore.setState({ workers: [gpuBox(2)] });
    renderSelect('cuda:5');
    expect(optionTexts()).toEqual(['auto', 'cpu', 'mps', 'cuda:5 (not on this server)', 'cuda:0', 'cuda:1']);
  });

  it('review 8: several selects read the worker list once between them', async () => {
    useWorkerStore.setState({ loaded: false });
    render(
      <>
        <DeviceParamSelect param={deviceParam()} value="auto" onChange={vi.fn()} displayLabel="a" />
        <DeviceParamSelect param={deviceParam()} value="cpu" onChange={vi.fn()} displayLabel="b" />
        <DeviceParamSelect param={deviceParam()} value="mps" onChange={vi.fn()} displayLabel="c" />
      </>,
    );
    await waitFor(() => expect(refresh).toHaveBeenCalledTimes(1));
    // Not listed until the read answers.
    expect(optionTexts()).toEqual(['auto', 'cpu', 'mps']);
  });

  it('does not read when the list is loaded', () => {
    renderSelect('auto');
    expect(refresh).not.toHaveBeenCalled();
  });

  it('review 8: a poll that changes nothing does not re-render the select', () => {
    useWorkerStore.setState({ workers: [gpuBox(2)] });
    let commits = 0;
    render(
      <Profiler id="device" onRender={() => { commits += 1; }}>
        <DeviceParamSelect param={deviceParam()} value="auto" onChange={vi.fn()} displayLabel="device" />
      </Profiler>,
    );
    const afterMount = commits;
    // The Workers poll replaces the list with an equal one.
    act(() => useWorkerStore.setState({ workers: [gpuBox(2)] }));
    expect(commits).toBe(afterMount);
    // A worker with more cards changes the names, and the select follows.
    act(() => useWorkerStore.setState({ workers: [gpuBox(3)] }));
    expect(commits).toBe(afterMount + 1);
    expect(optionTexts()).toEqual(['auto', 'cpu', 'mps', 'cuda:0', 'cuda:1', 'cuda:2']);
  });

  it('speaks zh-TW', () => {
    useI18n.setState({ locale: 'zh-TW' });
    useWorkerStore.setState({ workers: [gpuBox(1)] });
    renderSelect('cuda:3');
    expect(optionTexts()).toContain('cuda:3（本伺服器沒有）');
    expect(select().querySelector('optgroup')?.getAttribute('label')).toBe('run 裝置集合內的名稱');
    expect(screen.getByText(/本伺服器沒有 cuda:3/)).toBeTruthy();
    expect(screen.getByText(/依裝置集合的順序/)).toBeTruthy();
  });
});

describe('ParamField routes a device select here', () => {
  it('renders the stored device instead of auto', () => {
    render(<ParamField param={deviceParam()} value="cuda:0" onChange={vi.fn()} />);
    expect(select().value).toBe('cuda:0');
    expect(screen.getByText('cuda:0 (not on this server)')).toBeTruthy();
  });
});
