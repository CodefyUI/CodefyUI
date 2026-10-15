import { describe, it, expect } from 'vitest';
import { aliasForDevice } from './deviceAliases';

describe('aliasForDevice', () => {
  it('numbers the cards of a set by their position in it', () => {
    expect(aliasForDevice(['cuda:0', 'cuda:2'], 'cuda:0')).toBe('cuda:0');
    expect(aliasForDevice(['cuda:0', 'cuda:2'], 'cuda:2')).toBe('cuda:1');
  });

  it('follows the order the devices were listed in, not their indices', () => {
    expect(aliasForDevice(['cuda:2', 'cuda:0'], 'cuda:2')).toBe('cuda:0');
    expect(aliasForDevice(['cuda:2', 'cuda:0'], 'cuda:0')).toBe('cuda:1');
  });

  it('reads the spelling the way the backend does', () => {
    expect(aliasForDevice([' CUDA:3 '], 'cuda:3')).toBe('cuda:0');
    expect(aliasForDevice(['cuda:3'], 'Cuda:3')).toBe('cuda:0');
  });

  it('has no alias for a card outside the set', () => {
    expect(aliasForDevice(['cuda:0', 'cuda:2'], 'cuda:1')).toBeNull();
    expect(aliasForDevice([], 'cuda:0')).toBeNull();
  });

  it('has no alias for a device that is not a numbered CUDA card', () => {
    expect(aliasForDevice(['cpu'], 'cpu')).toBeNull();
    expect(aliasForDevice(['mps:0'], 'mps:0')).toBeNull();
    expect(aliasForDevice(['cuda:0'], 'cuda')).toBeNull();
  });

  it('skips non-CUDA entries when counting positions', () => {
    expect(aliasForDevice(['cpu', 'cuda:5'], 'cuda:5')).toBe('cuda:0');
  });
});
