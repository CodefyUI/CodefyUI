import { describe, it, expect } from 'vitest';
import { formatBytes } from './formatBytes';

describe('formatBytes', () => {
  it('reads an empty store as zero bytes, not as an empty string', () => {
    // The health panel renders this straight into a value column, so "nothing
    // cached" has to look like a number the reader can compare against the
    // budget next to it.
    expect(formatBytes(0)).toBe('0 B');
  });

  it('keeps sub-kilobyte sizes in whole bytes', () => {
    expect(formatBytes(1)).toBe('1 B');
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(1023)).toBe('1023 B');
  });

  it('switches to KB at exactly 1024 bytes', () => {
    expect(formatBytes(1024)).toBe('1.0 KB');
    expect(formatBytes(1536)).toBe('1.5 KB');
  });

  it('switches to MB at exactly 1024 KB', () => {
    expect(formatBytes(1024 * 1024)).toBe('1.0 MB');
    expect(formatBytes(512 * 1024 * 1024)).toBe('512.0 MB');
  });

  it('switches to GB at exactly 1024 MB, and to TB above that', () => {
    expect(formatBytes(1024 ** 3)).toBe('1.0 GB');
    expect(formatBytes(3 * 1024 ** 3 + 512 * 1024 ** 2)).toBe('3.5 GB');
    expect(formatBytes(1024 ** 4)).toBe('1.0 TB');
  });

  it('promotes a value that would round up into the next unit', () => {
    // 1048575 B is 1023.999… KB. One decimal place prints that as
    // "1024.0 KB" -- a number nobody expects to see in a KB column, so the
    // unit is promoted once more instead.
    expect(formatBytes(1024 * 1024 - 1)).toBe('1.0 MB');
    expect(formatBytes(1024 - 1)).toBe('1023 B');
  });

  it('stays in TB rather than inventing a unit it has no name for', () => {
    expect(formatBytes(1024 ** 5)).toBe('1024.0 TB');
  });

  it('renders a missing or nonsensical number as zero rather than NaN', () => {
    // /api/health omits a store that is not running, so a caller can hand this
    // `undefined.bytes`; "NaN B" in the panel would read as a bug in the
    // server rather than a gap in the payload.
    expect(formatBytes(Number.NaN)).toBe('0 B');
    expect(formatBytes(Number.POSITIVE_INFINITY)).toBe('0 B');
    expect(formatBytes(-1)).toBe('0 B');
  });
});
