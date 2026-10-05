import { describe, it, expect } from 'vitest';
import {
  DEVICE_PATTERN,
  MAX_RUN_SEED,
  isRunSeed,
  readGraphDevice,
  readGraphSeed,
} from './graphSettings';

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

describe('readGraphSeed', () => {
  it('reads every whole number the backend accepts, both ends included', () => {
    expect(readGraphSeed({ seed: 0 })).toBe(0);
    expect(readGraphSeed({ seed: 42 })).toBe(42);
    expect(readGraphSeed({ seed: 4294967295 })).toBe(4294967295);
  });

  it('reads null for a number the backend refuses', () => {
    expect(readGraphSeed({ seed: -1 })).toBeNull();
    expect(readGraphSeed({ seed: 4294967296 })).toBeNull();
    expect(readGraphSeed({ seed: 1.5 })).toBeNull();
    expect(readGraphSeed({ seed: NaN })).toBeNull();
    expect(readGraphSeed({ seed: Infinity })).toBeNull();
  });

  it('reads a number only, never a numeric string', () => {
    expect(readGraphSeed({ seed: '7' })).toBeNull();
    expect(readGraphSeed({ seed: true })).toBeNull();
  });

  it('reads null when seed is absent or null, or settings is not an object', () => {
    expect(readGraphSeed({})).toBeNull();
    expect(readGraphSeed({ seed: null })).toBeNull();
    expect(readGraphSeed({ device: 'cpu' })).toBeNull();
    expect(readGraphSeed(undefined)).toBeNull();
    expect(readGraphSeed(null)).toBeNull();
    expect(readGraphSeed(7)).toBeNull();
    expect(readGraphSeed('seed')).toBeNull();
  });
});

describe('isRunSeed', () => {
  it('bounds a seed the way the backend does (MAX_SEED = 2**32 - 1)', () => {
    expect(MAX_RUN_SEED).toBe(2 ** 32 - 1);
    expect(isRunSeed(0)).toBe(true);
    expect(isRunSeed(MAX_RUN_SEED)).toBe(true);
    expect(isRunSeed(MAX_RUN_SEED + 1)).toBe(false);
    expect(isRunSeed(-1)).toBe(false);
    expect(isRunSeed(2.5)).toBe(false);
    expect(isRunSeed(null)).toBe(false);
  });
});
