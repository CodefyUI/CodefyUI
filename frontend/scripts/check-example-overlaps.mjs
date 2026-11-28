#!/usr/bin/env node
/**
 * Overlapping node cards in the built-in examples, as the browser draws them.
 *
 *   node scripts/check-example-overlaps.mjs <url> [--out file.json] [--jobs 4]
 *        [--max N]
 *
 * Opens every example from the blank tab's All view, one fresh browser
 * context each, waits for the cards to settle, and reads each
 * `.react-flow__node` rectangle in flow coordinates (its translate and its
 * unscaled size). Two cards overlap when their rectangles intersect by more
 * than 1px both ways; a card wholly inside another (a frame or group around
 * its members) is not counted. Prints per-example pairs and the total.
 *
 * Node sizes come from CSS (padding, port rows, header), so this is the check
 * that a density change did not push cards into each other: run it before
 * and after, and the total may not rise.
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
  console.error('usage: check-example-overlaps.mjs <url> [--out f.json] [--jobs 4] [--max N]');
  process.exit(2);
}
const jobs = Number(opt('jobs', '4'));
const max = Number(opt('max', '0'));
const out = opt('out', null);

const browser = await chromium.launch();

async function openAll() {
  const ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1, locale: 'en-US' });
  await ctx.addInitScript(() => {
    try {
      if (sessionStorage.getItem('__seeded')) return;
      sessionStorage.setItem('__seeded', '1');
      localStorage.setItem('codefyui.examples.mainTag', 'all');
    } catch { /* defaults */ }
  });
  const page = await ctx.newPage();
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1000);
  return { ctx, page };
}

const cardSelector = 'button[class*="presetCard"]:not([class*="blankCard"])';

// The list, in the order the All view draws it.
const first = await openAll();
const names = await first.page.$$eval(cardSelector, (els) =>
  els.map((e) => e.querySelector('[class*="presetCardName"]')?.textContent?.trim() ?? ''));
await first.ctx.close();
const total = max > 0 ? Math.min(max, names.length) : names.length;

function readNodes() {
  return [...document.querySelectorAll('.react-flow__node')].map((el) => {
    const t = /translate\(([-\d.]+)px,\s*([-\d.]+)px\)/.exec(el.style.transform);
    return {
      id: el.dataset.id,
      label: el.textContent.trim().slice(0, 24),
      x: Number(t?.[1] ?? 0), y: Number(t?.[2] ?? 0), w: el.offsetWidth, h: el.offsetHeight,
    };
  });
}

function overlaps(nodes) {
  const pairs = [];
  const contains = (a, b) => a.x <= b.x && a.y <= b.y && a.x + a.w >= b.x + b.w && a.y + a.h >= b.y + b.h;
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i];
      const b = nodes[j];
      const ox = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
      const oy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
      if (ox > 1 && oy > 1 && !contains(a, b) && !contains(b, a)) {
        pairs.push({ a: a.label, b: b.label, ox: Math.round(ox), oy: Math.round(oy) });
      }
    }
  }
  return pairs;
}

async function check(index) {
  const { ctx, page } = await openAll();
  try {
    await page.locator(cardSelector).nth(index).click();
    await page.waitForFunction(() => document.querySelectorAll('.react-flow__node').length > 0, null, { timeout: 15000 });
    await page.waitForTimeout(1200);
    const nodes = await page.evaluate(readNodes);
    return { name: names[index], nodes: nodes.length, pairs: overlaps(nodes) };
  } catch (err) {
    return { name: names[index], error: String(err).split('\n')[0] };
  } finally {
    await ctx.close();
  }
}

const results = new Array(total);
let next = 0;
await Promise.all(Array.from({ length: jobs }, async () => {
  while (next < total) {
    const i = next++;
    results[i] = await check(i);
  }
}));
await browser.close();

const withPairs = results.filter((r) => r.pairs?.length);
const summary = {
  url,
  examples: total,
  errors: results.filter((r) => r.error).length,
  nodesMeasured: results.reduce((n, r) => n + (r.nodes ?? 0), 0),
  overlappingPairs: results.reduce((n, r) => n + (r.pairs?.length ?? 0), 0),
  examplesWithOverlaps: withPairs.length,
  details: withPairs.concat(results.filter((r) => r.error)),
};
const json = JSON.stringify(summary, null, 1);
if (out) writeFileSync(out, `${json}\n`);
console.log(JSON.stringify({ ...summary, details: undefined }));
for (const r of summary.details) {
  console.log(r.error ? `ERROR ${r.name}: ${r.error}` : `${r.name}: ${r.pairs.map((p) => `${p.a} x ${p.b} (${p.ox}x${p.oy})`).join('; ')}`);
}
