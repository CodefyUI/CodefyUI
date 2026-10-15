import { describe, expect, it } from 'vitest';
import type { SweepVariant } from '../../api/rest';
import { variantDevices, variantWorker } from './sweepPlacement';

function variant(patch: Partial<SweepVariant>): SweepVariant {
  return {
    index: 0, domain_index: 0, run_id: 'r0', status: 'running', params: [], seed: null, objective: null,
    ...patch,
  };
}

describe('sweepPlacement (#777)', () => {
  it('names the worker, or a dash when the run is gone', () => {
    expect(variantWorker(variant({ worker_id: 'local' }))).toBe('local');
    expect(variantWorker(variant({ worker_id: 'gpu-box' }))).toBe('gpu-box');
    expect(variantWorker(variant({}))).toBe('-');
  });

  it('lists the device set, else the device of the queue key, else a dash', () => {
    expect(variantDevices(variant({ devices: ['cuda:2', 'cuda:0'], queue_key: 'box/cuda:2' }))).toBe('cuda:2, cuda:0');
    expect(variantDevices(variant({ devices: [], queue_key: 'cpu' }))).toBe('cpu');
    expect(variantDevices(variant({ devices: null, queue_key: 'box/cuda:1' }))).toBe('cuda:1');
    expect(variantDevices(variant({}))).toBe('-');
  });
});
