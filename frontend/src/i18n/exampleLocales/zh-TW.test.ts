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

/** One line, this wide -- the Chinese half of the rule
 * `backend/tests/test_example_descriptions.py` holds over the English. The
 * long explanation belongs in a note on the canvas, beside the nodes it is
 * about, not in the one field every card and every row has to render. */
const MAX_DESCRIPTION_COLUMNS = 56;

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

/** Translations whose description still runs past the card. Only ever
 * shorter: the test below fails on an entry that no longer belongs.
 *
 * The 36 a pack ships, in the order `zh-TW.ts` declares them -- all that is
 * left of the 71 this list started with, now that every built-in says its one
 * line. `NOT_YET_SHORTENED` in
 * `backend/tests/test_example_descriptions.py` is the English twin. */
const NOT_YET_SHORTENED = new Set<string>([
  'plugin:foundations/C1-3/Kernel-Effects',
  'plugin:foundations/C2-1/Supervised-Learning-101',
  'plugin:foundations/C2-2/Concentric-Circles-Failure',
  'plugin:foundations/C2-3/Decision-Tree-Iris',
  'plugin:foundations/C2-3/SVM-RBF-Beats-Circles',
  'plugin:foundations/C2-4/MLP-Solves-Circles',
  'plugin:foundations/C2-4/MLP-Without-Activation',
  'plugin:foundations/C2-5/MLP-Inline-Demo',
  'plugin:foundations/C2-5/MLP-MNIST-Training',
  'plugin:foundations/Classical/Column-Stats-101',
  'plugin:foundations/Classical/KNN-from-Scratch',
  'plugin:foundations/Classical/Linear-Logistic-Compare',
  'plugin:deep/C3-1/Conv2D-Kernel-Effects',
  'plugin:deep/C3-1/LeNet-MNIST-Training',
  'plugin:deep/C3-2/UNet-Forward-Shapes',
  'plugin:deep/C3-3/Diffusion-Denoise-Loop',
  'plugin:deep/C4-1/LSTM-Sequence-Forward',
  'plugin:deep/C4-2/Co-Reference-Attention',
  'plugin:deep/C4-3/Transformer-Block-Assembled',
  'plugin:deep/C4-4/LLM-Inference-Pipeline',
  'plugin:deep/C6-1/World-Model-Next-State',
  'plugin:deep/C6-2/ViT-Full-Forward',
  'plugin:deep/C6-3/VLM-Cross-Modal-Attention',
  'plugin:deep/C6-4/MoE-Routing',
  'plugin:deep/Diffusion/Cross-Attention-101',
  'plugin:deep/Diffusion/Mini-UNet-Expanded',
  'plugin:deep/LLM/Multi-Head-Causal',
  'plugin:deep/LLM/Self-Attention-101',
  'plugin:deep/Transformer/Patchify-101',
  'plugin:rl/C5-1/RL-Trajectory-Mockup',
  'plugin:rl/C5-3/RLHF-Reward-Model',
  'plugin:rl/C5-4/GRPO-Group-Advantage',
  'plugin:rl/RL/Policy-Gradient-101',
  'plugin:stats/Stats/Confusion-Matrix-Heatmap',
  'plugin:stats/Stats/Iris-Describe-Table',
  'plugin:stats/Stats/Iris-GroupBy-Chart',
]);

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
    // The card cuts at 80 columns, but a description that needs 80 has
    // stopped being a label and become the explanation -- which belongs in a
    // note on the canvas, next to the nodes it is about.
    const over = Object.entries(zhTW)
      .filter(([key]) => !NOT_YET_SHORTENED.has(key))
      .filter(([, { description }]) => {
        const text = description ?? '';
        return displayWidth(text) > MAX_DESCRIPTION_COLUMNS || text.includes('\n');
      })
      .map(([key, { description }]) => `${key} (${displayWidth(description ?? '')} columns)`);
    expect(over).toEqual([]);
  });

  it('keeps no entry for a description that has already been shortened', () => {
    // The ratchet: a stale entry fails, so the list can only get shorter.
    const stale = [...NOT_YET_SHORTENED].filter((key) => {
      const text = zhTW[key]?.description;
      return text !== undefined && displayWidth(text) <= MAX_DESCRIPTION_COLUMNS
        && !text.includes('\n');
    });
    expect(stale).toEqual([]);
    expect([...NOT_YET_SHORTENED].filter((key) => !(key in zhTW))).toEqual([]);
  });
});
