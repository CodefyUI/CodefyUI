import { describe, expect, it } from 'vitest';
import { MAX_RUN_NAME_LENGTH, storableRunName, tabDefaultName } from './runName';

describe('storableRunName', () => {
  it('trims and cuts to the server limit in code points', () => {
    expect(storableRunName('  nightly  ')).toBe('nightly');
    const long = '\u{1F600}'.repeat(MAX_RUN_NAME_LENGTH + 5);
    expect(Array.from(storableRunName(long))).toHaveLength(MAX_RUN_NAME_LENGTH);
  });

  it('sends half of a surrogate pair as "?"', () => {
    expect(storableRunName('a\uD800b')).toBe('a?b');
  });
});

describe('tabDefaultName', () => {
  it('is the tab name, storable, or empty for a tab without one', () => {
    expect(tabDefaultName({ name: ' Train ' })).toBe('Train');
    expect(tabDefaultName({ name: 3 })).toBe('');
    expect(tabDefaultName(null)).toBe('');
  });
});
