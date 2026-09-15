import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import zhTW from './zh-TW';
import { displayWidth, truncateToWidth } from '../../utils/localizeExamples';

/**
 * The Chinese half of the example-gallery invariants.
 *
 * `backend/tests/test_builtin_examples.py` pins the English descriptions: they
 * are read off `graph.json`, and it asserts that a requirement lands inside the
 * card's 80-column cut. Nothing was checking the Chinese, which is a separate
 * table written by hand -- so a translation could drop a GPU requirement, grow
 * past the card, or translate an English name that is supposed to stay English,
 * and every test would still pass.
 */

const REPO = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..');
const EXAMPLES = join(REPO, 'examples');
const PLUGINS = join(REPO, 'plugins');

/** How much of a description the empty-canvas card shows. Mirrors
 * `CARD_DESC_COLUMNS` in EmptyCanvasOverlay.tsx. */
const CARD_COLUMNS = 80;

interface Example {
  key: string;
  name: string;
  description: string;
}

/** Every `graph.json` under `dir`, keyed the way `/api/examples/list` keys it. */
function scan(dir: string, prefix: string, out: Example[] = []): Example[] {
  if (!existsSync(dir)) return out;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      scan(full, prefix ? `${prefix}/${entry.name}` : entry.name, out);
    } else if (entry.name === 'graph.json') {
      const data = JSON.parse(readFileSync(full, 'utf8'));
      out.push({ key: prefix, name: data.name ?? '', description: data.description ?? '' });
    }
  }
  return out;
}

const builtin = scan(EXAMPLES, '');
const plugins = existsSync(PLUGINS)
  ? readdirSync(PLUGINS, { withFileTypes: true })
      .filter((d) => d.isDirectory())
      .flatMap((d) => scan(join(PLUGINS, d.name, 'examples'), '').map((e) => ({
        ...e,
        key: `plugin:${d.name}/${e.key}`,
      })))
  : [];
const shipped = [...builtin, ...plugins];

describe('zh-TW example descriptions', () => {
  it('reads the examples off disk at all', () => {
    // A broken scan would make every assertion below vacuous.
    expect(shipped.length).toBeGreaterThan(50);
    expect(shipped.every((e) => e.description.length > 0)).toBe(true);
  });

  it('covers every example that ships in this repo', () => {
    const untranslated = shipped.map((e) => e.key).filter((key) => !(key in zhTW));
    expect(untranslated).toEqual([]);
  });

  it('has no entry for an example that no longer exists', () => {
    // A renamed or deleted example leaves a key that can never be read again,
    // and nothing in the running app would say so.
    const live = new Set(shipped.map((e) => e.key));
    expect(Object.keys(zhTW).filter((key) => !live.has(key))).toEqual([]);
  });

  it('translates no name -- the English name is the handle in both locales', () => {
    for (const [key, translation] of Object.entries(zhTW)) {
      expect(Object.keys(translation), `${key} carries more than a description`).toEqual([
        'description',
      ]);
    }
  });

  it('is written in Traditional Chinese, with no Simplified forms', () => {
    // Pairs, not a bare character list: 程, 像, 算 and 出 are the SAME in both
    // scripts, and screening on those flags correct text.
    const simplified = '数据网络图内这体国学习说见对么门车东书长风马个为们无产两点线级别还应该运动过时间实现样处发变变换维输层训练权随机误认识计记忆资优组给结经类状态传递归';
    for (const [key, { description }] of Object.entries(zhTW)) {
      const found = [...new Set(description ?? '')].filter((ch) => simplified.includes(ch));
      expect(found, `${key} uses Simplified characters`).toEqual([]);
    }
  });

  it('carries no pictographic emoji', () => {
    // The CLI prints these through a cp950 console on Windows.
    const emoji = /[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}\u{FE0F}]/u;
    for (const [key, { description }] of Object.entries(zhTW)) {
      expect(emoji.test(description ?? ''), `${key} contains emoji`).toBe(false);
    }
  });

  it('never runs longer than the English it replaces', () => {
    // Chinese is denser than English; a translation that grew is padding.
    const bloated = shipped
      .filter((e) => (zhTW[e.key]?.description?.length ?? 0) > e.description.length)
      .map((e) => `${e.key} (en ${e.description.length}, zh ${zhTW[e.key].description!.length})`);
    expect(bloated).toEqual([]);
  });

  it('keeps every requirement inside the card cut, as the English does', () => {
    // The empty-canvas card is what a reader sees BEFORE pressing Run, and it
    // is the one surface that shows a description without being hovered. A
    // requirement past the cut is a footnote nobody reads -- so wherever the
    // English names one inside its own 80 columns, the Chinese has to name it
    // inside its 80 too. Pairs, because "GPU" is not a word the Chinese uses.
    const REQUIREMENTS: { en: RegExp; zh: RegExp; what: string }[] = [
      { en: /GPU/, zh: /GPU|顯卡/, what: 'a GPU' },
      { en: /download/i, zh: /下載/, what: 'a download' },
      { en: /\bpacks?\b/i, zh: /套件包/, what: 'a pack install' },
      { en: /Ollama|API key/i, zh: /Ollama|金鑰/, what: 'a model endpoint' },
      { en: /about an hour/i, zh: /一小時|小時/, what: 'the runtime' },
    ];
    const missing: string[] = [];
    for (const example of shipped) {
      const zh = zhTW[example.key]?.description;
      if (!zh) continue;
      const enCard = truncateToWidth(example.description, CARD_COLUMNS);
      const zhCard = truncateToWidth(zh, CARD_COLUMNS);
      for (const { en, zh: zhPattern, what } of REQUIREMENTS) {
        if (en.test(enCard) && !zhPattern.test(zhCard)) {
          missing.push(`${example.key}: the English card names ${what}, the Chinese card does not`);
        }
      }
    }
    expect(missing).toEqual([]);
  });

  it('says something on the card rather than opening with setup', () => {
    // 80 columns is about 40 ideographs. A description whose first 40 do not
    // reach a verb has spent the whole card on preamble.
    for (const [key, { description }] of Object.entries(zhTW)) {
      expect(displayWidth(description ?? ''), `${key} is too short to say anything`).toBeGreaterThan(
        40,
      );
    }
  });
});
