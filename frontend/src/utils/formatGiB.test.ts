import { describe, expect, it } from 'vitest';
import { formatGiB } from './formatGiB';

const GIB = 1024 ** 3;

describe('formatGiB', () => {
  it('prints GiB with one decimal, as the backend does', () => {
    expect(formatGiB(6 * GIB)).toBe('6.0 GiB');
    expect(formatGiB(2.4 * GIB)).toBe('2.4 GiB');
    expect(formatGiB(24.3 * GIB)).toBe('24.3 GiB');
  });

  it('never prints a positive figure as zero', () => {
    expect(formatGiB(1024)).toBe('0.1 GiB');
  });

  it('prints nothing to report as zero', () => {
    expect(formatGiB(0)).toBe('0.0 GiB');
    expect(formatGiB(-1)).toBe('0.0 GiB');
    expect(formatGiB(Number.NaN)).toBe('0.0 GiB');
  });
});
