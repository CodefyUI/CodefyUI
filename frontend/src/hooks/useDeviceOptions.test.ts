import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import {
  useDeviceOptions,
  deviceLabel,
  _resetDeviceOptionsForTesting,
} from './useDeviceOptions';
import { fetchDevices } from '../api/rest';

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
});
