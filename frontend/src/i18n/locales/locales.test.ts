import { describe, it, expect } from 'vitest';
import en from './en';
import zhTW from './zh-TW';

/**
 * What TypeScript already guarantees, and what it does not.
 *
 * `zh-TW.ts` is typed `Record<TranslationKey, string>`, so a missing or
 * misspelled key is a compile error and no test is needed for it. What the
 * type cannot see is what is INSIDE the strings: a `{name}` that was
 * translated along with the sentence around it, or a value that picked up a
 * stray space, renders as literal `{name}` or as a lopsided label at runtime
 * and nowhere else. That is what this file pins.
 */

type Dict = Record<string, string>;

const enDict = en as unknown as Dict;
const zhDict = zhTW as Dict;

/** The `{var}` slots `t()` will substitute, sorted so order is not compared. */
function placeholders(text: string): string[] {
  return (text.match(/\{[A-Za-z0-9_]+\}/g) ?? []).sort();
}

describe('locale tables', () => {
  it('define exactly the same keys', () => {
    // The en -> zh-TW direction is also a compile error; asserting it here
    // says so in the failure output rather than only in `tsc`.
    expect(Object.keys(enDict).filter((key) => !(key in zhDict))).toEqual([]);
    expect(Object.keys(zhDict).filter((key) => !(key in enDict))).toEqual([]);
  });

  it('carry the same {placeholders} in both locales', () => {
    const mismatched = Object.keys(enDict).filter(
      (key) =>
        placeholders(enDict[key]).join(' ') !== placeholders(zhDict[key]).join(' '),
    );
    expect(mismatched).toEqual([]);
  });

  it('have no value that is blank or padded with whitespace', () => {
    for (const [locale, dict] of [
      ['en', enDict],
      ['zh-TW', zhDict],
    ] as const) {
      for (const [key, value] of Object.entries(dict)) {
        expect(value, `${locale} ${key} is padded`).toBe(value.trim());
        expect(value, `${locale} ${key} is blank`).not.toBe('');
      }
    }
  });
});
