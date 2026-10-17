import { describe, expect, it } from 'vitest';
import { useI18n } from '../i18n';
import { deviceAliasLines, runBudget, runDevice, runDevices, runWorker, runWorkerLabel } from './runPlacement';

const t = (...args: Parameters<ReturnType<typeof useI18n.getState>['t']>) => useI18n.getState().t(...args);

describe('runPlacement', () => {
  it('names the worker, this server for a row without one', () => {
    useI18n.setState({ locale: 'en' });
    expect(runWorker({})).toBe('local');
    expect(runWorker({ worker_id: null })).toBe('local');
    expect(runWorkerLabel({}, t)).toBe('This server');
    expect(runWorkerLabel({ worker_id: 'any' }, t)).toBe('Any available');
    expect(runWorkerLabel({ worker_id: 'gpu-box' }, t)).toBe('gpu-box');
    // The one-shot pool (#790) is named as such while its run waits.
    const pooled = { worker_id: 'any', options: { target: { worker: 'any', pool: 'one-shot' } } };
    expect(runWorkerLabel(pooled, t)).toBe('One-shot pool');
    expect(runWorkerLabel({ worker_id: 'any', options: { target: null } }, t)).toBe('Any available');
    useI18n.setState({ locale: 'zh-TW' });
    expect(runWorkerLabel({ worker_id: 'local' }, t)).toBe('本伺服器');
    expect(runWorkerLabel(pooled, t)).toBe('單次 worker 池');
    useI18n.setState({ locale: 'en' });
  });

  it('shows the device it got, else the one it asked for', () => {
    expect(runDevice({ queue_key: 'gpu-box/cuda:1' })).toBe('gpu-box/cuda:1');
    expect(runDevice({ options: { device: 'auto' } })).toBe('auto');
    expect(runDevice({ options: { device: 3 } })).toBe('-');
    expect(runDevice({})).toBe('-');
    expect(runDevices({ queue_key: 'gpu-box/cuda:1' })).toBe('cuda:1');
    expect(runDevices({ devices: ['cuda:2', 'cuda:0'] })).toBe('cuda:2, cuda:0');
    expect(runDevices({ devices: [], options: { device: 'cpu' } })).toBe('cpu');
  });

  it('lists the in-run names of a worker run only', () => {
    useI18n.setState({ locale: 'en' });
    expect(deviceAliasLines({ worker_id: 'box', devices: ['cuda:2', 'cuda:0'] }, t))
      .toBe('cuda:2 -> cuda:0 in this run\ncuda:0 -> cuda:1 in this run');
    expect(deviceAliasLines({ worker_id: 'box', devices: ['cuda:0'] }, t)).toBeNull();
    expect(deviceAliasLines({ worker_id: 'box' }, t)).toBeNull();
    expect(deviceAliasLines({ devices: ['cuda:1'] }, t)).toBeNull();
  });

  it('shows the memory budget it was admitted with, else the one it asked for (#788)', () => {
    const gib = 1024 ** 3;
    expect(runBudget({ execution_facts: { memory_budget: { bytes: 6 * gib, source: 'declared' } } }, t))
      .toEqual({ text: '6.0 GiB', title: 'Memory budget per device, declared at submit' });
    expect(runBudget({
      execution_facts: { memory_budget: { bytes: 3 * gib, source: 'history', from_run: 'abcdef0123' } },
    }, t)).toEqual({ text: '3.0 GiB', title: 'Memory budget per device, from the peak of run abcdef01' });
    expect(runBudget({ execution_facts: { memory_budget: { bytes: 'x' } }, options: { target: { memory_gib: 2 } } }, t))
      .toEqual({ text: '2.0 GiB', title: 'Memory budget asked for; used only on a device shared by memory' });
    expect(runBudget({ options: { target: { worker: 'box' } } }, t)).toBeNull();
    expect(runBudget({ execution_facts: null }, t)).toBeNull();
  });
});
