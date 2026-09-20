import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import zhTW from './zh-TW';
import { displayWidth } from '../../utils/localizeExamples';

/**
 * The Chinese half of the example-gallery invariants.
 *
 * `backend/tests/test_builtin_examples.py` pins the English descriptions: they
 * are read off `graph.json`, and it asserts that no description states a
 * requirement and that every requirement is stated in a note on the canvas
 * instead. Nothing was checking the Chinese, which is a separate table written
 * by hand -- so a translation could keep a GPU requirement the English had
 * dropped, grow past the card, or translate an English name that is supposed
 * to stay English, and every test would still pass.
 *
 * The English name has its own rule -- five words, 34 columns, unique across
 * the repo -- and it lives in `test_example_descriptions.py` alone, because
 * the name is not translated: this table carries descriptions only, which the
 * "translates no name" case below pins.
 */

const REPO = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..');
const EXAMPLES = join(REPO, 'examples');
const PLUGINS = join(REPO, 'plugins');

/** One line, this wide -- the Chinese half of the rule
 * `backend/tests/test_example_descriptions.py` holds over the English. Forty
 * columns is twenty Chinese characters, which is what the line under the
 * title has room for in a 13rem card whose type grows to 19px on a wide
 * screen. `CARD_DESC_COLUMNS` in EmptyCanvasOverlay.tsx cuts at the same 40,
 * so a description that obeys this cap is never cut on screen and the card
 * shows the line whole. The long explanation belongs in a note on the canvas,
 * beside the nodes it is about, not in the one field every card and every row
 * has to render. Every example obeys it, so there is no exception list on
 * either side. */
const MAX_DESCRIPTION_COLUMNS = 40;

/** Widths both implementations of `displayWidth` are pinned to. */
const WIDTH_VECTORS = join(
  dirname(fileURLToPath(import.meta.url)),
  '..',
  '..',
  'utils',
  'displayWidth.vectors.json',
);

interface Vector {
  text: string;
  width: number;
}

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

  it('states no requirement -- a card says what the graph shows', () => {
    // The inverse of the rule this case used to hold. Forty columns is twenty
    // Chinese characters, and a line that spends half of them on 「需 1.5 GB
    // 下載」 has stopped saying what the graph is for -- which is the one
    // thing only the card can say, because it is what a reader picks a card
    // by. A download, a GPU, a pack, an API key or a sibling example that has
    // to be run first is named in the note on the canvas, beside the nodes it
    // is about, where there is room for how big it is and where it comes from.
    //
    // `backend/tests/test_builtin_examples.py` holds both halves of that over
    // the English and over the notes, which are bilingual in one string. This
    // file only ever sees the translation table, so it holds the Chinese
    // descriptions and asserts nothing about the notes.
    const RESOURCES: { pattern: RegExp; what: string }[] = [
      { pattern: /GPU|顯卡/, what: 'a GPU' },
      { pattern: /下載/, what: 'a download' },
      { pattern: /套件包/, what: 'a pack install' },
      { pattern: /金鑰|Ollama/, what: 'a key or a model endpoint' },
    ];
    // 需 covers 需要 and the bare 需 a short line writes (「需 GPU」). The
    // lookbehind is the negation guard `_CARD_REQUIREMENT_NEGATIONS` carries
    // on the Python side, for the same reason: 「不需下載」 names a download
    // to say it is NOT needed, and flagging it would tell the author to move
    // a reassurance off the card -- the one sentence a hesitant reader wanted.
    const REQUIRES = /(?<![不無毋])需|必須/;
    const offenders: string[] = [];
    for (const [key, { description }] of Object.entries(zhTW)) {
      const text = description ?? '';
      if (!REQUIRES.test(text)) continue;
      for (const { pattern, what } of RESOURCES) {
        if (pattern.test(text)) {
          offenders.push(`${key} states ${what}: ${text}`);
        }
      }
    }
    expect(
      offenders,
      'a requirement belongs in the note on the example canvas, beside the nodes it is about, and in the exceptions list in docs/ -- not on the card',
    ).toEqual([]);
  });

  it('measures a width the Python port agrees with', () => {
    // `backend/tests/test_example_descriptions.py` checks the same cap over
    // the English descriptions, with its own copy of `displayWidth`. Two
    // implementations of one measurement drift silently, so both read these
    // vectors.
    const { vectors } = JSON.parse(readFileSync(WIDTH_VECTORS, 'utf8'));
    expect(vectors.length).toBeGreaterThan(5);
    expect(vectors.map((v: Vector) => displayWidth(v.text))).toEqual(
      vectors.map((v: Vector) => v.width),
    );
  });

  it('says it in one line, inside the cap', () => {
    // Forty columns -- twenty Chinese characters -- is the line the card has
    // room for. A description that needs more has stopped being a label and
    // become the explanation, which belongs in a note on the canvas, next to
    // the nodes it is about. Every example obeys this now, so the rule holds
    // with no exceptions.
    const over = Object.entries(zhTW)
      .filter(([, { description }]) => {
        const text = description ?? '';
        return displayWidth(text) > MAX_DESCRIPTION_COLUMNS || text.includes('\n');
      })
      .map(([key, { description }]) => `${key} (${displayWidth(description ?? '')} columns)`);
    expect(over).toEqual([]);
  });
});
