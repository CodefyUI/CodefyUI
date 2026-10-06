import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it, expect } from 'vitest';
import zhTW from './zh-TW';
import zhTWNodes from '../nodeLocales/zh-TW';
import zhTWExamples from '../exampleLocales/zh-TW';

/**
 * One name for a preset in zh-TW: the one the Nodes tab shows as the group's
 * heading. Refusals, validation lines, the configure dialog's label and the
 * docs said 預設模組 for the same thing, so a student was told about a
 * 預設模組 the palette nowhere listed.
 */
describe('the zh-TW name for a preset', () => {
  it('is 預設組合 everywhere in the UI, as in the palette', () => {
    expect(zhTW['palette.presets.category']).toBe('預設組合');
    const tables = {
      ui: JSON.stringify(zhTW),
      nodes: JSON.stringify(zhTWNodes),
      examples: JSON.stringify(zhTWExamples),
    };
    expect(
      Object.entries(tables)
        .filter(([, text]) => text.includes('預設模組'))
        .map(([table]) => table),
    ).toEqual([]);
  });

  it('is 預設組合 in the zh-TW docs too', () => {
    const docs = join(dirname(fileURLToPath(import.meta.url)), '../../../../docs/i18n/zh-TW');
    const files = (dir: string): string[] =>
      readdirSync(dir).flatMap((name) => {
        const path = join(dir, name);
        return statSync(path).isDirectory() ? files(path) : [path];
      });
    const pages = files(docs).filter((path) => /\.(mdx?|json)$/.test(path));
    // The walk found the pages, so an empty answer below means something.
    expect(pages.some((path) => path.endsWith('presets.md'))).toBe(true);
    expect(
      pages
        .filter((path) => readFileSync(path, 'utf8').includes('預設模組'))
        .map((path) => relative(docs, path)),
    ).toEqual([]);
  });
});
