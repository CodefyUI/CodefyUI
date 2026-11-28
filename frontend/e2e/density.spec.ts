/**
 * Density pass (phase 5): the two behaviours a screenshot cannot show.
 *
 * - The Results dock starts as its tab row alone, opens by itself when the
 *   first run writes to the log, and a height the user drags is still there
 *   after a reload.
 * - The main-tag chips on the blank tab are one row that scrolls sideways:
 *   a vertical mouse wheel over it scrolls it, and the chip chosen from the
 *   keyboard is scrolled into view.
 */
import { canvasNodes, expect, fixture, importFile, openApp, runGraph, test } from './app';

const dock = (page: import('@playwright/test').Page) => page.locator('[data-tour="results"]');

test('the Results dock opens on the first output and keeps a dragged height', async ({ page }) => {
  await openApp(page);
  // Fresh: the tab row alone, every tab still there to click.
  const fresh = await dock(page).boundingBox();
  expect(fresh!.height).toBeLessThanOrEqual(40);
  await expect(page.getByRole('button', { name: /^Execution Log/i })).toBeVisible();
  await expect(page.getByRole('button', { name: /^Runs/ })).toBeVisible();

  await importFile(page, fixture('tensor-chain.json'));
  await expect(canvasNodes(page)).toHaveCount(5);
  await runGraph(page);
  await expect.poll(async () => (await dock(page).boundingBox())!.height).toBe(160);

  // Drag the top edge up by 100px.
  const box = (await dock(page).boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2, box.y - 98, { steps: 5 });
  await page.mouse.up();
  await expect.poll(async () => (await dock(page).boundingBox())!.height).toBe(260);

  await page.reload();
  await expect(page.getByRole('button', { name: 'Run', exact: false }).first()).toBeVisible();
  await expect.poll(async () => (await dock(page).boundingBox())!.height).toBe(260);
});

test.describe('blank tab at 1280x720', () => {
  test.use({ viewport: { width: 1280, height: 720 } });

  test('the chip row scrolls with the wheel and keeps the chosen chip in view', async ({ page }) => {
    await page.addInitScript(() => {
      if (!sessionStorage.getItem('density-seeded')) {
        sessionStorage.setItem('density-seeded', '1');
        localStorage.setItem('codefyui.examples.mainTag', 'all');
      }
    });
    await openApp(page);
    const row = page.getByRole('tablist', { name: 'Example categories' });
    await expect(row).toBeVisible();
    const overflow = await row.evaluate((el) => el.scrollWidth - el.clientWidth);
    expect(overflow).toBeGreaterThan(0);
    // One row: every chip on the same line.
    const tops = await row.getByRole('tab').evaluateAll((els) =>
      [...new Set(els.map((el) => Math.round(el.getBoundingClientRect().top)))]);
    expect(tops).toHaveLength(1);

    // A vertical wheel turn over the row scrolls it sideways.
    const rowBox = (await row.boundingBox())!;
    await page.mouse.move(rowBox.x + rowBox.width / 2, rowBox.y + rowBox.height / 2);
    await page.mouse.wheel(0, 200);
    await expect.poll(() => row.evaluate((el) => el.scrollLeft)).toBeGreaterThan(0);
    await page.mouse.wheel(0, -1000);
    await expect.poll(() => row.evaluate((el) => el.scrollLeft)).toBe(0);

    // End selects the last chip, which scrolls into view.
    await row.getByRole('tab', { selected: true }).focus();
    await page.keyboard.press('End');
    const last = row.getByRole('tab').last();
    await expect(last).toHaveAttribute('aria-selected', 'true');
    await expect.poll(async () => {
      const [chip, strip] = [await last.boundingBox(), await row.boundingBox()];
      return chip!.x + chip!.width <= strip!.x + strip!.width + 0.5 && chip!.x >= strip!.x - 0.5;
    }).toBe(true);
  });
});
