import { describe, expect, it } from 'vitest';
import type { DeviceInfo } from '../../api/rest';
import type { WorkerInfo } from '../../api/workers';
import {
  anyKinds,
  connectedWorkers,
  defaultDevices,
  targetableWorkers,
  defaultKind,
  deviceKind,
  localDeviceChoices,
  parseMemoryGib,
  toggleDevice,
  usesSharedDevice,
  workerDeviceChoices,
  type DeviceChoice,
} from './queueTargets';

const info = (value: string, label = value.toUpperCase(), detail = ''): DeviceInfo =>
  ({ value, label, detail, available: true });

function worker(name: string, status: WorkerInfo['status'], devices: Array<[string, string]>): WorkerInfo {
  return {
    id: name, name, status, draining: status === 'draining', version: '3.0.2',
    fingerprint_digest: null, platform: {}, mounts: {}, running: [],
    connected_at: 0, last_heartbeat_at: 0,
    devices: devices.map(([value, kind]) => ({ value, kind, index: null, name: '', memory: null })),
  };
}

const choice = (value: string): DeviceChoice => ({ value, kind: deviceKind(value), detail: '' });

describe('localDeviceChoices', () => {
  it('names a lone CUDA card cuda:0 and drops auto', () => {
    expect(localDeviceChoices([info('auto'), info('cpu', 'CPU'), info('cuda', 'NVIDIA CUDA', 'RTX 4090'), info('mps', 'Apple MPS')]))
      .toEqual([
        { value: 'cpu', kind: 'cpu', detail: 'CPU' },
        { value: 'cuda:0', kind: 'cuda', detail: 'NVIDIA CUDA · RTX 4090' },
        { value: 'mps', kind: 'mps', detail: 'Apple MPS' },
      ]);
  });

  it('keeps the per-card rows of a multi-GPU box and drops the bare kind', () => {
    expect(localDeviceChoices([info('cpu'), info('cuda'), info('cuda:0'), info('cuda:1')]).map((c) => c.value))
      .toEqual(['cpu', 'cuda:0', 'cuda:1']);
  });
});

describe('workerDeviceChoices / connectedWorkers', () => {
  it('reads the reported kind, else the value', () => {
    const box = worker('gpu-box', 'online', [['cuda:0', 'cuda'], ['mps', '']]);
    expect(workerDeviceChoices(box).map((c) => c.kind)).toEqual(['cuda', 'mps']);
  });

  it('leaves out disconnected and offline workers', () => {
    const list = [worker('a', 'online', []), worker('b', 'disconnected', []), worker('c', 'draining', []), worker('d', 'offline', [])];
    expect(connectedWorkers(list).map((w) => w.name)).toEqual(['a', 'c']);
  });

  it('lets a run name an offline (enrolled) worker, never a disconnected one', () => {
    const list = [worker('a', 'online', []), worker('b', 'disconnected', []), worker('d', 'offline', [])];
    expect(targetableWorkers(list).map((w) => w.name)).toEqual(['a', 'd']);
  });
});

describe('defaultDevices', () => {
  const choices = [choice('cpu'), choice('cuda:0'), choice('cuda:1'), choice('mps:0')];

  it('takes the graph device when the target has it', () => {
    expect(defaultDevices(choices, 'cuda:1')).toEqual(['cuda:1']);
    expect(defaultDevices(choices, ' CPU ')).toEqual(['cpu']);
    expect(defaultDevices(choices, 'mps')).toEqual(['mps:0']);
  });

  it('resolves a bare kind to the first device of it', () => {
    expect(defaultDevices(choices, 'cuda')).toEqual(['cuda:0']);
  });

  it('falls back to the first accelerator for auto or a device the target lacks', () => {
    expect(defaultDevices(choices, 'auto')).toEqual(['cuda:0']);
    expect(defaultDevices([choice('cpu'), choice('mps')], 'cuda:3')).toEqual(['mps']);
    // A card the target lacks still names its kind.
    expect(defaultDevices([choice('cpu'), choice('cuda:0'), choice('cuda:1')], 'cuda:3')).toEqual(['cuda:0']);
  });

  it('falls back to the first device, or nothing', () => {
    expect(defaultDevices([choice('cpu')], 'cuda')).toEqual(['cpu']);
    expect(defaultDevices([], 'cpu')).toEqual([]);
  });
});

describe('toggleDevice', () => {
  it('adds a device of the same kind, removes one already in, and replaces across kinds', () => {
    const order = ['cpu', 'cuda:0', 'cuda:1', 'cuda:2'];
    expect(toggleDevice(['cuda:0'], 'cuda:2', order)).toEqual(['cuda:0', 'cuda:2']);
    expect(toggleDevice(['cuda:0', 'cuda:2'], 'cuda:0', order)).toEqual(['cuda:2']);
    expect(toggleDevice(['cuda:0', 'cuda:2'], 'cpu', order)).toEqual(['cpu']);
    expect(toggleDevice([], 'cpu', order)).toEqual(['cpu']);
  });

  it('keeps the set in the reported order, whatever the click order', () => {
    const order = ['cpu', 'cuda:0', 'cuda:1', 'cuda:2'];
    expect(toggleDevice(['cuda:2'], 'cuda:0', order)).toEqual(['cuda:0', 'cuda:2']);
    expect(toggleDevice(['cuda:2', 'cuda:0'].sort(), 'cuda:1', order)).toEqual(['cuda:0', 'cuda:1', 'cuda:2']);
  });
});

describe('anyKinds / defaultKind', () => {
  it('offers CPU and each accelerator kind this server or a connected worker has', () => {
    const workers = [worker('a', 'online', [['cuda:0', 'cuda']]), worker('b', 'disconnected', [['mps', 'mps']])];
    expect(anyKinds([choice('cpu')], workers)).toEqual(['cpu', 'cuda']);
    expect(anyKinds([choice('cpu'), choice('mps')], [])).toEqual(['cpu', 'mps']);
  });

  it('starts from the graph device kind, else the first accelerator, else CPU', () => {
    expect(defaultKind(['cpu', 'cuda'], 'cuda:1')).toBe('cuda');
    expect(defaultKind(['cpu', 'cuda'], 'auto')).toBe('cuda');
    expect(defaultKind(['cpu'], 'mps')).toBe('cpu');
  });

  it('knows when a run may land on a device shared by memory (#788)', () => {
    const box = worker('box', 'online', [['cuda:0', 'cuda'], ['cuda:1', 'cuda']]);
    box.memory_budget = true;
    box.devices[1].policy = 'shared';
    const old = worker('old', 'online', [['cuda:0', 'cuda']]);
    old.devices[0].policy = 'shared';
    expect(usesSharedDevice('box', box, ['cuda:1'], [box], 'cuda')).toBe(true);
    expect(usesSharedDevice('box', box, ['cuda:0'], [box], 'cuda')).toBe(false);
    expect(usesSharedDevice('old', old, ['cuda:0'], [old], 'cuda')).toBe(false);
    expect(usesSharedDevice('local', null, ['cuda:0'], [box], 'cuda')).toBe(false);
    expect(usesSharedDevice('any', null, [], [old, box], 'cuda')).toBe(true);
    expect(usesSharedDevice('any', null, [], [box], 'cpu')).toBe(false);
  });

  it('reads a memory budget field (#788)', () => {
    expect(parseMemoryGib('  ')).toBeNull();
    expect(parseMemoryGib('6')).toBe(6);
    expect(parseMemoryGib('2.5')).toBe(2.5);
    expect(parseMemoryGib('.5')).toBe(0.5);
    for (const bad of ['0', '-1', '1e3', 'abc', '4097', '1.2.3']) {
      expect(parseMemoryGib(bad)).toBeUndefined();
    }
  });
});
