import { describe, it, expect } from 'vitest';
import { DEVICE_PATTERN, readGraphDevice } from './graphSettings';

describe('readGraphDevice', () => {
  it('reads null for a missing or non-object settings block', () => {
    expect(readGraphDevice(undefined)).toBeNull();
    expect(readGraphDevice(null)).toBeNull();
    expect(readGraphDevice('cpu')).toBeNull();
    expect(readGraphDevice(42)).toBeNull();
  });

  it('reads null when device is absent or not a string', () => {
    expect(readGraphDevice({})).toBeNull();
    expect(readGraphDevice({ device: 1 })).toBeNull();
    expect(readGraphDevice({ device: null })).toBeNull();
  });

  it('normalises case and surrounding whitespace', () => {
    expect(readGraphDevice({ device: ' CUDA:1 ' })).toBe('cuda:1');
    expect(readGraphDevice({ device: 'MPS' })).toBe('mps');
  });

  it('accepts every form the backend pattern accepts and nothing else', () => {
    expect(readGraphDevice({ device: 'cpu' })).toBe('cpu');
    expect(readGraphDevice({ device: 'auto' })).toBe('auto');
    expect(readGraphDevice({ device: 'cuda' })).toBe('cuda');
    expect(readGraphDevice({ device: 'mps:0' })).toBe('mps:0');
    expect(readGraphDevice({ device: 'gpu' })).toBeNull();
    expect(readGraphDevice({ device: '' })).toBeNull();
    expect(readGraphDevice({ device: 'cuda:' })).toBeNull();
  });

  it('exposes the pattern the backend validates with', () => {
    expect(DEVICE_PATTERN.test('cuda:12')).toBe(true);
    expect(DEVICE_PATTERN.test('rocm')).toBe(false);
  });
});
