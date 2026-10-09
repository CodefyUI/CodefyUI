#!/usr/bin/env node
// Checks that every English docs page has a zh-TW twin with the same heading
// structure and the same anchor ids (core#399).
//
// Pages are paired by path: docs/docs/<path> with
// docs/i18n/zh-TW/docusaurus-plugin-content-docs/current/<path>, for every
// .md / .mdx file found on either side, so a new page is paired without
// touching this script. For each pair the headings must match one for one:
// same count, same level at each position and the same resolved anchor id
// (the level-1 page title excepted from the id rule; see comparePair).
// A zh-TW heading keeps its English anchor with an explicit id, either
// `{/* #id */}` (the form these docs use) or the classic `{#id}`.
//
// Anchor ids are resolved the way @docusaurus/mdx-loader's headings plugin
// does: an explicit id wins and does not reach the slugger; any other heading
// is slugged from its plain text by the slugger from @docusaurus/utils (a
// github-slugger wrapper), so a repeated heading gets `-1`, `-2`, ... as it
// does in the built page. Headings inside fenced code, HTML comments, MDX
// comments and front matter are ignored.
//
// This checks navigation structure only, never translation wording.
//
// Usage: node scripts/check-locale-parity.mjs [--en <dir>] [--zh <dir>]
// (run from docs/, after `pnpm install`). Exits 1 and prints one line per
// problem, each naming the file and the heading.

import {readdirSync, readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const DOCS_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const DEFAULT_EN_ROOT = path.join(DOCS_ROOT, 'docs');
export const DEFAULT_ZH_ROOT = path.join(
  DOCS_ROOT,
  'i18n/zh-TW/docusaurus-plugin-content-docs/current',
);

// The slugger the build uses. @docusaurus/utils is a dependency of
// @docusaurus/core, not of this package, so it is resolved from core's
// location (pnpm does not hoist it into docs/node_modules).
function loadDocusaurusUtils() {
  const require = createRequire(import.meta.url);
  const coreRequire = createRequire(require.resolve('@docusaurus/core/package.json'));
  return coreRequire('@docusaurus/utils');
}
const {createSlugger} = loadDocusaurusUtils();

const PAGE_EXT = /\.mdx?$/;

/** Every .md/.mdx file under `root`, as sorted posix paths relative to it. */
export function listPages(root) {
  const out = [];
  const walk = (dir) => {
    for (const entry of readdirSync(dir, {withFileTypes: true})) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (PAGE_EXT.test(entry.name)) {
        out.push(path.relative(root, full).split(path.sep).join('/'));
      }
    }
  };
  walk(root);
  return out.sort();
}

// --- Heading extraction ---------------------------------------------------

const ATX_HEADING = /^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$/;
const FENCE_OPEN = /^\s*(`{3,}|~{3,})/;

/** The heading's id from a trailing `{/* #id *\/}` or `<!-- #id -->`. */
function trailingCommentId(raw) {
  const m =
    /\{\/\*([\s\S]*?)\*\/\}\s*$/.exec(raw) ?? /<!--([\s\S]*?)-->\s*$/.exec(raw);
  if (!m) return {text: raw, id: undefined};
  // As in mdx-loader: only the comment's first word, and only with a leading #.
  const first = m[1].trim().split(' ')[0];
  if (!first?.startsWith('#') || first.length === 1) return {text: raw, id: undefined};
  return {text: raw.slice(0, m.index).trimEnd(), id: first.slice(1)};
}

/**
 * The plain text Docusaurus slugs a heading from (mdast-util-to-string over
 * the inline nodes): code spans keep their content, links and images keep
 * their label, emphasis markers, HTML tags and backslash escapes go.
 */
export function headingPlainText(raw) {
  let text = raw.replace(/[ \t]+#+[ \t]*$/, '').replace(/^#+$/, ''); // closing #s
  const parts = text.split(/(`+)([\s\S]*?[^`])\1(?!`)/);
  let out = '';
  for (let i = 0; i < parts.length; i += 3) {
    // Escaped punctuation is held aside so the markup rules below skip it.
    const escaped = [];
    const s = parts[i]
      .replace(/\\([!-/:-@[-`{-~])/g, (_, c) => `${escaped.push(c) - 1}`)
      .replace(/<\/?[A-Za-z][^>]*>/g, '')
      .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
      .replace(/!?\[([^\]]*)\]\[[^\]]*\]/g, '$1')
      .replace(/(\*+|(?<![A-Za-z0-9])_+|_+(?![A-Za-z0-9]))/g, '')
      .replace(/(\d+)/g, (_, n) => escaped[n])
      .replace(/&amp;/g, '&')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&quot;/g, '"');
    out += s;
    if (i + 2 < parts.length) {
      // A code span's content, one space trimmed from each end if both have one.
      const code = parts[i + 2];
      out += /^ .* $/s.test(code) && code.trim() ? code.slice(1, -1) : code;
    }
  }
  return out.trim();
}

/**
 * Headings of one page in document order: {level, text, id, line}.
 * `line` is 1-based.
 */
export function extractHeadings(source) {
  const lines = source.split(/\r?\n/);
  const slugger = createSlugger();
  const headings = [];
  let i = 0;

  // Front matter.
  if (lines[0]?.trim() === '---') {
    const end = lines.findIndex((l, n) => n > 0 && l.trim() === '---');
    if (end > 0) i = end + 1;
  }

  let fence = null; // {char, len}
  let inHtmlComment = false;
  let inMdxComment = false;
  for (; i < lines.length; i++) {
    const line = lines[i];
    if (fence) {
      const m = /^\s*(`{3,}|~{3,})\s*$/.exec(line);
      if (m && m[1][0] === fence.char && m[1].length >= fence.len) fence = null;
      continue;
    }
    if (inHtmlComment) {
      if (line.includes('-->')) inHtmlComment = false;
      continue;
    }
    if (inMdxComment) {
      if (line.includes('*/}')) inMdxComment = false;
      continue;
    }
    const open = FENCE_OPEN.exec(line);
    if (open) {
      fence = {char: open[1][0], len: open[1].length};
      continue;
    }
    if (/^\s*<!--/.test(line) && !line.includes('-->')) {
      inHtmlComment = true;
      continue;
    }
    if (/^\s*\{\/\*/.test(line) && !line.includes('*/}')) {
      inMdxComment = true;
      continue;
    }
    const m = ATX_HEADING.exec(line);
    if (!m) continue;
    const level = m[1].length;
    const fromComment = trailingCommentId(m[2] ?? '');
    const plain = headingPlainText(fromComment.text);
    let id = fromComment.id;
    let text = plain;
    if (id === undefined) {
      const classic = /\s*\{#(?<id>(?:.(?!\{#|\}))*.)\}$/.exec(plain);
      if (classic) {
        id = classic.groups.id.trim();
        text = plain.replace(classic[0], '');
      } else {
        id = slugger.slug(plain);
      }
    }
    headings.push({level, text, id, line: i + 1});
  }
  return headings;
}

// --- Comparison -----------------------------------------------------------

const fmt = (file, h) => `${file}:${h.line} ${'#'.repeat(h.level)} ${h.text} (#${h.id})`;

/**
 * Problems for one page pair, each a one-line string naming the file and the
 * heading. `enFile` / `zhFile` are the paths printed in messages.
 */
export function comparePair(enFile, enSource, zhFile, zhSource) {
  const en = extractHeadings(enSource);
  const zh = extractHeadings(zhSource);
  const problems = [];

  if (en.length !== zh.length) {
    problems.push(
      `${zhFile}: ${zh.length} headings, ${enFile} has ${en.length}`,
    );
    // Positions no longer line up; name the anchors each side lacks instead.
    // Page titles (level 1) are left out, as in the id rule below.
    const zhIds = new Set(zh.map((h) => h.id));
    const enIds = new Set(en.map((h) => h.id));
    for (const h of en) {
      if (h.level > 1 && !zhIds.has(h.id)) {
        problems.push(`  missing in zh-TW: ${fmt(enFile, h)}`);
      }
    }
    for (const h of zh) {
      if (h.level > 1 && !enIds.has(h.id)) {
        problems.push(`  not in English: ${fmt(zhFile, h)}`);
      }
    }
    if (problems.length === 1) {
      problems.push('  (every anchor exists on both sides; a heading is repeated or a page title added on one side)');
    }
    return problems;
  }

  for (let k = 0; k < en.length; k++) {
    const a = en[k];
    const b = zh[k];
    if (a.level !== b.level) {
      problems.push(
        `${zhFile}:${b.line}: heading level ${b.level}, English has ${a.level}\n` +
          `  en: ${fmt(enFile, a)}\n  zh: ${fmt(zhFile, b)}`,
      );
    }
    // A level-1 heading is the page title. Links go to the page, never to the
    // title's anchor, and it is left out of the table of contents, so the
    // translated title keeps its own slug.
    if (a.id !== b.id && !(a.level === 1 && b.level === 1)) {
      problems.push(
        `${zhFile}:${b.line}: anchor #${b.id}, English has #${a.id} ` +
          `(add {/* #${a.id} */} to the zh-TW heading)\n` +
          `  en: ${fmt(enFile, a)}\n  zh: ${fmt(zhFile, b)}`,
      );
    }
  }
  return problems;
}

/** Problems across both trees, including pages with no twin. */
export function checkTrees(enRoot, zhRoot) {
  const enPages = listPages(enRoot);
  const zhPages = new Set(listPages(zhRoot));
  // Paths relative to the working directory when they are under it.
  const rel = (p) => {
    const r = path.relative(process.cwd(), p);
    return r && !r.startsWith('..') ? r : p;
  };
  const problems = [];
  for (const page of enPages) {
    const enPath = path.join(enRoot, page);
    const zhPath = path.join(zhRoot, page);
    if (!zhPages.has(page)) {
      problems.push(`${rel(zhPath)}: missing (zh-TW twin of ${rel(enPath)})`);
      continue;
    }
    zhPages.delete(page);
    problems.push(
      ...comparePair(
        rel(enPath),
        readFileSync(enPath, 'utf8'),
        rel(zhPath),
        readFileSync(zhPath, 'utf8'),
      ),
    );
  }
  for (const page of zhPages) {
    problems.push(
      `${rel(path.join(zhRoot, page))}: no English page at ${rel(path.join(enRoot, page))}`,
    );
  }
  return {pages: enPages.length, problems};
}

function parseArgs(argv) {
  const opts = {en: DEFAULT_EN_ROOT, zh: DEFAULT_ZH_ROOT};
  for (let k = 0; k < argv.length; k++) {
    if (argv[k] === '--en') opts.en = path.resolve(argv[++k]);
    else if (argv[k] === '--zh') opts.zh = path.resolve(argv[++k]);
    else throw new Error(`unknown argument: ${argv[k]}`);
  }
  return opts;
}

export function main(argv = process.argv.slice(2), log = console) {
  const opts = parseArgs(argv);
  const {pages, problems} = checkTrees(opts.en, opts.zh);
  if (problems.length) {
    log.error(problems.join('\n'));
    log.error(
      `\nzh-TW parity check failed (${pages} English pages). A zh-TW page needs ` +
        'the same headings, at the same levels, with the same anchor ids as its ' +
        'English twin; keep an English anchor on a translated heading with {/* #id */}.',
    );
    return 1;
  }
  log.log(`zh-TW parity: ${pages} page pairs match.`);
  return 0;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exitCode = main();
}
