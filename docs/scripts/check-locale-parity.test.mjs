// Tests for check-locale-parity.mjs. Run from docs/: node --test scripts/
import assert from 'node:assert/strict';
import {mkdirSync, mkdtempSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {afterEach, describe, test} from 'node:test';

import {
  checkTrees,
  comparePair,
  extractHeadings,
  headingPlainText,
  main,
} from './check-locale-parity.mjs';

const ids = (src) => extractHeadings(src).map((h) => h.id);
const shape = (src) => extractHeadings(src).map((h) => `${h.level}:${h.id}`);

describe('extractHeadings', () => {
  test('explicit MDX comment id', () => {
    assert.deepEqual(ids('## 裝置選擇 {/* #device-selection */}\n'), ['device-selection']);
  });

  test('explicit id keeps only the first word of the comment', () => {
    assert.deepEqual(ids('## 標題 {/* #the-id keep in sync */}\n'), ['the-id']);
  });

  test('a comment without a leading # is not an id', () => {
    assert.notDeepEqual(ids('## Title {/* note */}\n'), ['note']);
  });

  test('explicit classic {#id}', () => {
    assert.deepEqual(ids('## 安裝 {#installation}\n'), ['installation']);
  });

  test('explicit HTML comment id', () => {
    assert.deepEqual(ids('## 安裝 <!-- #installation -->\n'), ['installation']);
  });

  test('slugs with github-slugger, keeping non-ASCII letters', () => {
    assert.deepEqual(
      ids('## GPU & Device Setup\n## The float64 + MPS constraint\n## 裝置 選擇\n'),
      ['gpu--device-setup', 'the-float64--mps-constraint', '裝置-選擇'],
    );
  });

  test('duplicate headings get -1, -2 like the build', () => {
    assert.deepEqual(ids('## Example\n### Example\n## Example\n'), [
      'example',
      'example-1',
      'example-2',
    ]);
  });

  test('an explicit id does not take a slug from later duplicates', () => {
    // mdx-loader never passes an explicit id through the slugger.
    assert.deepEqual(ids('## Example {/* #example */}\n## Example\n'), [
      'example',
      'example',
    ]);
  });

  test('headings inside fenced code are ignored', () => {
    const src = [
      '## Real',
      '```bash',
      '# a shell comment',
      '## not a heading',
      '```',
      '~~~~',
      '```',
      '# still code: a shorter fence does not close',
      '```',
      '~~~~',
      '  ```python',
      '  # indented fence inside a list',
      '  ```',
      '## After',
    ].join('\n');
    assert.deepEqual(ids(src), ['real', 'after']);
  });

  test('headings in front matter and comments are ignored', () => {
    const src = [
      '---',
      'title: x',
      '# not a heading',
      '---',
      '<!--',
      '## commented out',
      '-->',
      '{/*',
      '## commented out',
      '*/}',
      '## Kept',
    ].join('\n');
    assert.deepEqual(ids(src), ['kept']);
  });

  test('records level, text and 1-based line', () => {
    assert.deepEqual(extractHeadings('intro\n\n### Hello `world` ##\n'), [
      {level: 3, text: 'Hello world', id: 'hello-world', line: 3},
    ]);
  });

  test('a run of # without a space is not a heading', () => {
    assert.deepEqual(ids('#hashtag\n####### seven\n'), []);
  });
});

describe('headingPlainText', () => {
  test('strips inline markup the way the build reads heading text', () => {
    assert.equal(headingPlainText('`api.ui` — editor UI'), 'api.ui — editor UI');
    assert.equal(headingPlainText('Why `full_model` is restricted'), 'Why full_model is restricted');
    assert.equal(headingPlainText('**Bold** and _em_ and snake_case'), 'Bold and em and snake_case');
    assert.equal(headingPlainText('[Link](./x.md) and ![alt](a.png)'), 'Link and alt');
    assert.equal(headingPlainText('A <span>tag</span> \\* star'), 'A tag * star');
    assert.equal(headingPlainText('Install flags &amp; env'), 'Install flags & env');
  });
});

describe('comparePair', () => {
  const EN = '# Title\n\n## Install\n\n### Flags\n\n## Usage\n';
  const ZH_OK = '# 標題\n\n## 安裝 {/* #install */}\n\n### 旗標 {/* #flags */}\n\n## 用法 {/* #usage */}\n';

  test('a matching pair has no problems, translated page title included', () => {
    assert.deepEqual(comparePair('en.md', EN, 'zh.md', ZH_OK), []);
  });

  test('an added heading fails and names the heading', () => {
    const zh = ZH_OK + '\n## 多出來的段落\n';
    const problems = comparePair('en.md', EN, 'zh.md', zh).join('\n');
    assert.match(problems, /zh\.md: 5 headings, en\.md has 4/);
    assert.match(problems, /not in English: zh\.md:9 ## 多出來的段落/);
  });

  test('a removed heading names the English anchor zh-TW lacks', () => {
    const zh = '# 標題\n\n## 安裝 {/* #install */}\n\n## 用法 {/* #usage */}\n';
    const problems = comparePair('en.md', EN, 'zh.md', zh).join('\n');
    assert.match(problems, /missing in zh-TW: en\.md:5 ### Flags \(#flags\)/);
    // The translated page title is not reported as a missing anchor.
    assert.doesNotMatch(problems, /# Title|# 標題/);
  });

  test('a level change fails and names both headings', () => {
    const zh = ZH_OK.replace('### 旗標', '## 旗標');
    const problems = comparePair('en.md', EN, 'zh.md', zh).join('\n');
    assert.match(problems, /zh\.md:5: heading level 2, English has 3/);
    assert.match(problems, /en: en\.md:5 ### Flags/);
    assert.match(problems, /zh: zh\.md:5 ## 旗標/);
  });

  test('renaming an English heading without the zh-TW id fails', () => {
    const en = EN.replace('## Usage', '## Running it');
    const problems = comparePair('en.md', en, 'zh.md', ZH_OK).join('\n');
    assert.match(problems, /zh\.md:7: anchor #usage, English has #running-it/);
    assert.match(problems, /add \{\/\* #running-it \*\/\}/);
  });

  test('a translated heading without an explicit id fails', () => {
    const zh = ZH_OK.replace(' {/* #usage */}', '');
    const problems = comparePair('en.md', EN, 'zh.md', zh).join('\n');
    assert.match(problems, /anchor #用法, English has #usage/);
  });

  test('a duplicate heading must resolve to the same suffixed id', () => {
    const en = '## Example\n## Example\n';
    assert.deepEqual(
      comparePair('en.md', en, 'zh.md', '## 範例 {/* #example */}\n## 範例 {/* #example-1 */}\n'),
      [],
    );
    const problems = comparePair(
      'en.md',
      en,
      'zh.md',
      '## 範例 {/* #example */}\n## 範例 {/* #example */}\n',
    ).join('\n');
    assert.match(problems, /anchor #example, English has #example-1/);
  });

  test('heading syntax inside a code fence on one side only is not a heading', () => {
    const zh = ZH_OK + '\n```bash\n# 這是註解\n```\n';
    assert.deepEqual(comparePair('en.md', EN, 'zh.md', zh), []);
  });

  test('a repeated explicit id on one side is reported even though every anchor exists', () => {
    const problems = comparePair(
      'en.md',
      '## A\n## B\n',
      'zh.md',
      '## 甲 {/* #a */}\n## 乙 {/* #b */}\n## 乙 {/* #b */}\n',
    ).join('\n');
    assert.match(problems, /zh\.md: 3 headings, en\.md has 2/);
    assert.match(problems, /a heading is repeated or a page title added on one side/);
  });
});

describe('checkTrees / main', () => {
  let dir;
  afterEach(() => dir && rmSync(dir, {recursive: true, force: true}));

  function tree(files) {
    dir = mkdtempSync(path.join(tmpdir(), 'locale-parity-'));
    for (const [rel, body] of Object.entries(files)) {
      const full = path.join(dir, rel);
      mkdirSync(path.dirname(full), {recursive: true});
      writeFileSync(full, body);
    }
    return {en: path.join(dir, 'en'), zh: path.join(dir, 'zh')};
  }

  const silent = () => {
    const out = {log: [], error: []};
    return {out, log: {log: (s) => out.log.push(s), error: (s) => out.error.push(s)}};
  };

  test('pairs pages automatically, including nested and new ones', () => {
    const t = tree({
      'en/intro.md': '## A\n',
      'en/usage/new-page.mdx': '## B\n',
      'en/usage/_category_.json': '{}',
      'zh/intro.md': '## 甲 {/* #a */}\n',
      'zh/usage/new-page.mdx': '## 乙 {/* #b */}\n',
    });
    assert.deepEqual(checkTrees(t.en, t.zh), {pages: 2, problems: []});
    const {out, log} = silent();
    assert.equal(main(['--en', t.en, '--zh', t.zh], log), 0);
    assert.match(out.log[0], /2 page pairs match/);
  });

  test('a missing twin fails and names the file', () => {
    const t = tree({
      'en/intro.md': '## A\n',
      'en/usage/gone.md': '## B\n',
      'zh/intro.md': '## 甲 {/* #a */}\n',
    });
    const {problems} = checkTrees(t.en, t.zh);
    assert.equal(problems.length, 1);
    assert.match(problems[0], /zh[/\\]usage[/\\]gone\.md: missing \(zh-TW twin of .*en[/\\]usage[/\\]gone\.md\)/);
    const {out, log} = silent();
    assert.equal(main(['--en', t.en, '--zh', t.zh], log), 1);
    assert.match(out.error.join('\n'), /parity check failed \(2 English pages\)/);
  });

  test('a zh-TW page with no English page fails', () => {
    const t = tree({
      'en/intro.md': '## A\n',
      'zh/intro.md': '## 甲 {/* #a */}\n',
      'zh/stale.md': '## 舊\n',
    });
    const {problems} = checkTrees(t.en, t.zh);
    assert.equal(problems.length, 1);
    assert.match(problems[0], /stale\.md: no English page/);
  });

  test('rejects an unknown argument', () => {
    assert.throws(() => main(['--bogus']), /unknown argument: --bogus/);
  });
});
