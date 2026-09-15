import { useMemo } from 'react';
import { useI18n } from '../i18n';
import type { ExampleSummary } from '../api/rest';

/** An example with its description already resolved for the active locale.
 *
 * `description` is what the reader sees, so every card, tooltip and detail
 * pane can keep reading that one field. `descriptionEn` keeps the English the
 * backend read out of `graph.json`, which the search boxes still match: an
 * engineer reading the UI in Chinese types `attention` as readily as
 * 「注意力」, and the example names are English in both locales anyway.
 */
export interface LocalizedExample extends ExampleSummary {
  descriptionEn: string;
}

/** Resolve each example's description for the active locale.
 *
 * Untranslated examples -- one added since this build, one a third-party
 * plugin ships -- fall back to their English text rather than disappearing.
 */
export function useLocalizedExamples(examples: ExampleSummary[]): LocalizedExample[] {
  const { te, locale } = useI18n();
  return useMemo(
    () =>
      examples.map((example) => ({
        ...example,
        description: te(example.path, example.description),
        descriptionEn: example.description,
      })),
    // `locale`, not `te`: zustand hands back the same `te` function across a
    // locale change (it reads the locale off the store when called), so
    // depending on it would memoise the English text forever.
    [examples, locale, te],
  );
}

/** Roughly how many Latin characters wide this string renders.
 *
 * CJK ideographs, kana, Hangul and full-width punctuation occupy two Latin
 * columns each. A card that fits 80 Latin characters fits about 40 Chinese
 * ones, so a cut measured in code points clips an English card correctly and
 * lets a Chinese one run twice as far past the edge.
 */
export function displayWidth(text: string): number {
  let width = 0;
  for (const ch of text) {
    const cp = ch.codePointAt(0) ?? 0;
    width +=
      (cp >= 0x1100 && cp <= 0x115f) || // Hangul Jamo
      (cp >= 0x2e80 && cp <= 0x303e) || // CJK radicals, kangxi, CJK punctuation
      (cp >= 0x3041 && cp <= 0x33ff) || // kana, Hangul compat, CJK compat
      (cp >= 0x3400 && cp <= 0x4dbf) || // CJK ext A
      (cp >= 0x4e00 && cp <= 0x9fff) || // CJK unified
      (cp >= 0xa000 && cp <= 0xa4cf) || // Yi
      (cp >= 0xac00 && cp <= 0xd7a3) || // Hangul syllables
      (cp >= 0xf900 && cp <= 0xfaff) || // CJK compat ideographs
      (cp >= 0xfe30 && cp <= 0xfe6f) || // CJK compat forms
      (cp >= 0xff01 && cp <= 0xff60) || // full-width forms
      (cp >= 0xffe0 && cp <= 0xffe6) ||
      (cp >= 0x20000 && cp <= 0x3fffd) // CJK ext B and beyond
        ? 2
        : 1;
  }
  return width;
}

/** Cut `text` to at most `maxWidth` Latin columns, appending `…` when cut.
 *
 * Returns the string unchanged when it already fits, so a short description
 * never grows an ellipsis it did not earn.
 */
export function truncateToWidth(text: string, maxWidth: number): string {
  if (displayWidth(text) <= maxWidth) return text;
  let width = 0;
  let out = '';
  for (const ch of text) {
    const w = displayWidth(ch);
    if (width + w > maxWidth) break;
    width += w;
    out += ch;
  }
  return out.trimEnd() + '...';
}

/** Does this example match a lowercased search query?
 *
 * Name and category are English in every locale; the description is matched in
 * both the displayed language and the original English.
 */
export function exampleMatches(example: LocalizedExample, query: string): boolean {
  return (
    example.name.toLowerCase().includes(query) ||
    example.description.toLowerCase().includes(query) ||
    example.descriptionEn.toLowerCase().includes(query) ||
    example.category.toLowerCase().includes(query)
  );
}
