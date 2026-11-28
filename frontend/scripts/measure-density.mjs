#!/usr/bin/env node
/**
 * Density metrics for the space-use pass (phase 5). Prints JSON.
 *
 *   node scripts/measure-density.mjs <url> [--locale en-US|zh-TW]
 *        [--font small|default|large] [--sizes 13in,1080p,small] [--out file.json]
 *
 * Opens <url> fresh (empty storage) at each viewport and reports, on the blank
 * tab at the chosen font size:
 *
 * - root/body font size, toolbar box and row count, with and without one
 *   side-panel toggle (a clone of the Help button put where the toggles go:
 *   most installs have none, and the one-row rule must hold with one);
 * - palette, canvas and Results panel boxes, the canvas area;
 * - palette rows fully visible, at full density and names only;
 * - example cards fully visible: on Quick start (the default view; all of its
 *   cards should fit) and on All (the capacity of the blank tab);
 * - the smallest computed font size of any visible text (glyph-only text such
 *   as arrows and bullets excluded), and whether the page scrolls sideways;
 * - "Train CNN on MNIST": the zoom it opens at (fitted, so every node is
 *   always visible; a larger fit zoom means the same graph at a larger size)
 *   and the nodes that would be complete at 100% centred on the fit.
 *
 * The acceptance reviewer measures with their own script; where the two
 * disagree, the reason belongs in the report.
 */
import { createRequire } from 'node:module';
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(join(here, '..', 'package.json'));
const { chromium } = require('@playwright/test');

const args = process.argv.slice(2);
const url = args[0]?.startsWith('--') ? undefined : args[0];
const opt = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};
if (!url) {
  console.error('usage: measure-density.mjs <url> [--locale en-US] [--font default] [--sizes 13in,1080p,small] [--out f.json]');
  process.exit(2);
}
const locale = opt('locale', 'en-US');
const font = opt('font', 'default');
const ALL_SIZES = { '13in': [1470, 840], '1080p': [1920, 970], small: [1280, 720] };
const sizes = opt('sizes', Object.keys(ALL_SIZES).join(',')).split(',');
const out = opt('out', null);

const EXAMPLE_KEY = 'codefyui.examples.mainTag';

/** Runs in the page: everything measurable on the blank tab. */
function measurePage() {
  const box = (el) => {
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) };
  };
  const inside = (el, b) => {
    const r = el.getBoundingClientRect();
    return r.height > 0 && r.top >= b.top - 0.5 && r.bottom <= b.bottom + 0.5
      && r.left >= b.left - 0.5 && r.right <= b.right + 0.5;
  };
  const hasClass = (el, name) => [...el.classList].some((c) => c === name || new RegExp(`^_?${name}_`).test(c));
  const toolbarOf = () => {
    let el = document.querySelector('[data-tour="run"]');
    while (el && el.getBoundingClientRect().width < window.innerWidth - 2) el = el.parentElement;
    return el;
  };
  const rowsOf = (toolbar) => {
    const centers = [...toolbar.querySelectorAll('button,select')]
      .filter((e) => e.getBoundingClientRect().width > 0)
      .map((e) => { const r = e.getBoundingClientRect(); return r.top + r.height / 2; })
      .sort((a, b) => a - b);
    let rows = centers.length ? 1 : 0;
    for (let i = 1; i < centers.length; i++) if (centers[i] - centers[i - 1] > 16) rows++;
    return rows;
  };
  const toolbar = toolbarOf();
  const palette = document.querySelector('[data-tour="palette"]');
  const canvas = document.querySelector('[data-tour="canvas"]');
  const results = document.querySelector('[data-tour="results"]');
  const pbox = palette?.getBoundingClientRect();
  const cbox = canvas?.getBoundingClientRect();
  const rows = palette ? [...palette.querySelectorAll('[class*="nodeItem"]')].filter((e) => hasClass(e, 'nodeItem')) : [];
  const cards = canvas ? [...canvas.querySelectorAll('button')].filter((e) => hasClass(e, 'presetCard')) : [];
  const controls = toolbar ? [...toolbar.querySelectorAll('button,select')]
    .filter((e) => e.getBoundingClientRect().width > 0)
    .map((e) => Math.round(e.getBoundingClientRect().height)) : [];

  let minFont = 999;
  let minFontSample = '';
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let t = walker.nextNode(); t; t = walker.nextNode()) {
    if (!t.nodeValue.trim() || !/[\p{L}\p{N}]/u.test(t.nodeValue)) continue;
    const el = t.parentElement;
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0 || r.bottom < 0 || r.top > window.innerHeight) continue;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.opacity === '0') continue;
    const fs = parseFloat(cs.fontSize);
    if (fs < minFont) { minFont = fs; minFontSample = t.nodeValue.trim().slice(0, 30); }
  }
  return {
    rootFontPx: parseFloat(getComputedStyle(document.documentElement).fontSize),
    bodyFontPx: parseFloat(getComputedStyle(document.body).fontSize),
    toolbar: box(toolbar),
    toolbarRows: toolbar ? rowsOf(toolbar) : 0,
    toolbarTallestControl: controls.length ? Math.max(...controls) : 0,
    palette: box(palette),
    canvas: box(canvas),
    results: box(results),
    canvasArea: cbox ? Math.round(cbox.width * cbox.height) : 0,
    paletteRowsVisible: pbox ? rows.filter((e) => inside(e, pbox)).length : 0,
    paletteRowHeights: rows.slice(0, 4).map((e) => Math.round(e.getBoundingClientRect().height)),
    cardsVisible: cbox ? cards.filter((e) => inside(e, cbox)).length : 0,
    cardsTotal: cards.length,
    cardSize: cards[0] ? box(cards[0]) : null,
    minFontPx: minFont,
    minFontSample,
    horizontalScroll: document.documentElement.scrollWidth > window.innerWidth + 1,
  };
}

/** Runs in the page: one more side-panel toggle where the real ones go. */
function injectSideToggle() {
  const help = document.querySelector('[data-tour="help"]');
  const right = help?.parentElement;
  if (!help || !right) return false;
  const wrap = document.createElement('div');
  // The toggles' own wrapper is a plain .cluster inside the right cluster.
  wrap.className = [...right.classList].find((c) => /^_?cluster_/.test(c)) ?? '';
  const clone = help.cloneNode(true);
  clone.removeAttribute('data-tour');
  clone.textContent = 'C';
  wrap.replaceChildren(clone);
  wrap.dataset.measureInjected = '1';
  right.insertBefore(wrap, right.firstChild);
  return true;
}

function toolbarRowsNow() {
  let el = document.querySelector('[data-tour="run"]');
  while (el && el.getBoundingClientRect().width < window.innerWidth - 2) el = el.parentElement;
  const centers = [...el.querySelectorAll('button,select')]
    .filter((e) => e.getBoundingClientRect().width > 0)
    .map((e) => { const r = e.getBoundingClientRect(); return r.top + r.height / 2; })
    .sort((a, b) => a - b);
  let rows = centers.length ? 1 : 0;
  for (let i = 1; i < centers.length; i++) if (centers[i] - centers[i - 1] > 16) rows++;
  return { rows, h: Math.round(el.getBoundingClientRect().height) };
}

/** Runs in the page: the MNIST example's fit zoom and nodes complete at 100%. */
function measureFlow() {
  const canvas = document.querySelector('[data-tour="canvas"]').getBoundingClientRect();
  const viewport = document.querySelector('.react-flow__viewport');
  const m = /translate\(([-\d.]+)px,\s*([-\d.]+)px\)\s*scale\(([-\d.]+)\)/.exec(viewport?.style.transform ?? '');
  if (!m) return null;
  const [tx, ty, zoom] = [Number(m[1]), Number(m[2]), Number(m[3])];
  const pane = document.querySelector('.react-flow').getBoundingClientRect();
  // The flow point at the centre of the visible pane.
  const cx = (pane.width / 2 - tx) / zoom;
  const cy = (pane.height / 2 - ty) / zoom;
  const nodes = [...document.querySelectorAll('.react-flow__node')].map((el) => {
    const t = /translate\(([-\d.]+)px,\s*([-\d.]+)px\)/.exec(el.style.transform);
    return { x: Number(t?.[1] ?? 0), y: Number(t?.[2] ?? 0), w: el.offsetWidth, h: el.offsetHeight };
  });
  const half = { w: pane.width / 2, h: pane.height / 2 };
  const at100 = nodes.filter((n) => n.x >= cx - half.w && n.x + n.w <= cx + half.w
    && n.y >= cy - half.h && n.y + n.h <= cy + half.h).length;
  const heights = nodes.map((n) => n.h).sort((a, b) => a - b);
  return {
    nodes: nodes.length,
    fitZoom: Math.round(zoom * 1000) / 1000,
    completeAt100: at100,
    medianNodeHeight: heights[Math.floor(heights.length / 2)] ?? 0,
    canvas: { w: Math.round(canvas.width), h: Math.round(canvas.height) },
  };
}

async function freshPage(browser, w, h, initStorage) {
  const ctx = await browser.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 1, locale });
  await ctx.addInitScript(({ entries }) => {
    try {
      if (sessionStorage.getItem('__measure_seeded')) return;
      sessionStorage.setItem('__measure_seeded', '1');
      for (const [k, v] of entries) localStorage.setItem(k, v);
    } catch { /* storage blocked: defaults */ }
  }, { entries: Object.entries(initStorage) });
  const page = await ctx.newPage();
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1500);
  return { ctx, page };
}

const browser = await chromium.launch();
const result = { url, locale, font };
const baseStorage = { 'codefyui-font-size': font };
for (const name of sizes) {
  const [w, h] = ALL_SIZES[name];
  const r = {};

  // Quick start (the default view), full palette density.
  let { ctx, page } = await freshPage(browser, w, h, { ...baseStorage, [EXAMPLE_KEY]: 'quickstart' });
  Object.assign(r, await page.evaluate(measurePage));
  r.quickStartCardsVisible = r.cardsVisible;
  r.quickStartCardsTotal = r.cardsTotal;
  delete r.cardsVisible;
  delete r.cardsTotal;
  r.withSideToggle = (await page.evaluate(injectSideToggle)) ? await page.evaluate(toolbarRowsNow) : null;
  await ctx.close();

  // All: the capacity of the blank tab.
  ({ ctx, page } = await freshPage(browser, w, h, { ...baseStorage, [EXAMPLE_KEY]: 'all' }));
  const all = await page.evaluate(measurePage);
  r.allCardsVisible = all.cardsVisible;
  r.allCardsTotal = all.cardsTotal;
  await ctx.close();

  // Names-only palette (phase 5 toggle; ignored by a build without it).
  ({ ctx, page } = await freshPage(browser, w, h, { ...baseStorage, 'codefyui-palette-density': 'names' }));
  r.paletteRowsVisibleNamesOnly = (await page.evaluate(measurePage)).paletteRowsVisible;
  await ctx.close();

  // Train CNN on MNIST, opened from its card.
  ({ ctx, page } = await freshPage(browser, w, h, { ...baseStorage, [EXAMPLE_KEY]: 'all' }));
  const card = page.locator('button', { hasText: 'Train CNN on MNIST' }).first();
  if (await card.count()) {
    await card.click();
    await page.waitForTimeout(2000);
    r.mnist = await page.evaluate(measureFlow);
  }
  await ctx.close();

  result[name] = r;
}
await browser.close();

const json = JSON.stringify(result, null, 1);
if (out) writeFileSync(out, `${json}\n`);
console.log(json);
