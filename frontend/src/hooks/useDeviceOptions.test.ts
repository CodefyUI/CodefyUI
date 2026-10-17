import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import {
  useDeviceOptions,
  deviceLabel,
  deviceOptionLabel,
  isDeviceServed,
  _resetDeviceOptionsForTesting,
} from './useDeviceOptions';
import { fetchDevices } from '../api/rest';
import { useI18n } from '../i18n';

vi.mock('../api/rest', () => ({ fetchDevices: vi.fn() }));
const fetchDevicesMock = vi.mocked(fetchDevices);

const CPU = { value: 'cpu', label: 'CPU', detail: '', available: true };
const MPS = { value: 'mps', label: 'Apple MPS', detail: 'Metal Performance Shaders', available: true };

beforeEach(() => {
  _resetDeviceOptionsForTesting();
  fetchDevicesMock.mockReset();
});

describe('useDeviceOptions', () => {
  it('starts CPU-only and fills in from the server', async () => {
    fetchDevicesMock.mockResolvedValue({ default: 'mps', devices: [CPU, MPS] });
    const { result } = renderHook(() => useDeviceOptions());
    expect(result.current.devices).toEqual([CPU]);
    expect(result.current.serverDefault).toBe('cpu');
    await waitFor(() => expect(result.current.devices).toEqual([CPU, MPS]));
    expect(result.current.serverDefault).toBe('mps');
  });

  it('keeps the CPU-only fallback when the fetch fails', async () => {
    fetchDevicesMock.mockRejectedValue(new Error('offline'));
    const { result } = renderHook(() => useDeviceOptions());
    await waitFor(() => expect(fetchDevicesMock).toHaveBeenCalledTimes(1));
    expect(result.current.devices).toEqual([CPU]);
    expect(result.current.serverDefault).toBe('cpu');
  });

  it('keeps the CPU-only list for an empty answer, and reads a blank default as cpu', async () => {
    fetchDevicesMock.mockResolvedValue({ default: '', devices: [] });
    const { result } = renderHook(() => useDeviceOptions());
    await waitFor(() => expect(fetchDevicesMock).toHaveBeenCalledTimes(1));
    expect(result.current.devices).toEqual([CPU]);
    expect(result.current.serverDefault).toBe('cpu');
  });

  it('fetches once across consumers; a later consumer starts on the answer', async () => {
    fetchDevicesMock.mockResolvedValue({ default: 'mps', devices: [CPU, MPS] });
    const first = renderHook(() => useDeviceOptions());
    await waitFor(() => expect(first.result.current.devices).toEqual([CPU, MPS]));

    const second = renderHook(() => useDeviceOptions());
    // No CPU-only flash: the initial state is the cached answer.
    expect(second.result.current.devices).toEqual([CPU, MPS]);
    expect(fetchDevicesMock).toHaveBeenCalledTimes(1);
  });
});

describe('deviceLabel', () => {
  it('appends the detail when there is one', () => {
    expect(deviceLabel(CPU)).toBe('CPU');
    expect(deviceLabel(MPS)).toBe('Apple MPS');
    expect(deviceLabel({ value: 'cuda:1', label: 'NVIDIA CUDA', detail: 'RTX 4090', available: true }))
      .toBe('NVIDIA CUDA · cuda:1 · RTX 4090');
    expect(deviceLabel({ value: 'cuda:0', label: 'NVIDIA CUDA', detail: '', available: true }))
      .toBe('NVIDIA CUDA · cuda:0');
  });

  describe('deviceOptionLabel: memory (#767)', () => {
    const GIB = 1024 ** 3;
    const mem = (free: number | null, total: number | null, source = 'nvml') => ({
      free_bytes: free === null ? null : free * GIB,
      total_bytes: total === null ? null : total * GIB,
      source,
    });

    afterEach(() => useI18n.setState({ locale: 'en' }));

    it('shows free / total GiB for an accelerator', () => {
      useI18n.setState({ locale: 'en' });
      expect(deviceOptionLabel({ ...MPS, memory: mem(10.24, 16, 'torch.mps') }))
        .toBe('Apple MPS · 10.2 / 16.0 GiB free');
      expect(deviceOptionLabel({
        value: 'cuda:1', label: 'NVIDIA CUDA', detail: 'RTX 4090', available: true,
        memory: mem(20, 24),
      })).toBe('NVIDIA CUDA · cuda:1 · RTX 4090 · 20.0 / 24.0 GiB free');
    });

    it('stays out of deviceLabel, which names a device in sentences', () => {
      expect(deviceLabel({ ...MPS, memory: mem(8, 16) })).toBe('Apple MPS');
    });

    it('is translated', () => {
      useI18n.setState({ locale: 'zh-TW' });
      expect(deviceOptionLabel({ ...MPS, memory: mem(8, 16) })).toBe('Apple MPS · 可用 8.0 / 16.0 GiB');
    });

    it('leaves CPU and unknown memory out', () => {
      expect(deviceOptionLabel({ ...CPU, memory: mem(4, 32, 'psutil') })).toBe('CPU');
      expect(deviceOptionLabel({ ...MPS, memory: null })).toBe('Apple MPS');
      expect(deviceOptionLabel({ ...MPS, memory: mem(null, 16) })).toBe('Apple MPS');
      expect(deviceOptionLabel({ ...MPS, memory: mem(8, null) })).toBe('Apple MPS');
    });
  });
});

describe('isDeviceServed', () => {
  it('matches on the exact device value', () => {
    expect(isDeviceServed([CPU, MPS], 'mps')).toBe(true);
    expect(isDeviceServed([CPU, MPS], 'cuda')).toBe(false);
  });
});
