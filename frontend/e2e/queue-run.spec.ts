/**
 * Queue on... (#775): the toolbar's queued submission.
 *
 * A tiny CPU tensor graph is queued on This server through the Run split
 * button's menu. The dialog posts it to `POST /api/runs` (the queued lane,
 * as `cdui run` does), the Results dock switches to its Runs tab, and the
 * run's row says where it executed (This server) and on what (cpu), then
 * finishes. With no worker agent connected, This server is the only target.
 */
import { canvasNodes, expect, fixture, importFile, openApp, test } from './app';

test('queues a CPU graph on This server and follows it in the Runs panel', async ({ page }, testInfo) => {
  await openApp(page);
  await importFile(page, fixture('tensor-chain.json'));
  await expect(canvasNodes(page)).toHaveCount(5);

  await page.getByRole('button', { name: 'More ways to run' }).click();
  await page.getByRole('menuitem', { name: 'Queue on...' }).click();
  const dialog = page.getByRole('dialog', { name: 'Queue run' });
  await expect(dialog).toBeVisible();

  // No worker agent is connected: This server is the only target.
  const target = dialog.getByRole('combobox', { name: 'Target' });
  await expect(target.getByRole('option')).toHaveText(['This server']);
  await expect(dialog.getByRole('radio', { name: /^cpu/ })).toBeChecked();

  const name = `e2e queued ${testInfo.repeatEachIndex}.${testInfo.retry}`;
  await dialog.getByRole('textbox', { name: 'Name' }).fill(name);
  const submitted = page.waitForResponse(
    (response) => new URL(response.url()).pathname === '/api/runs' && response.request().method() === 'POST',
  );
  await dialog.getByRole('button', { name: 'Queue' }).click();
  const response = await submitted;
  expect(response.status()).toBe(200);
  const body = response.request().postDataJSON() as {
    options: { target: unknown; device: string; seed: unknown };
    name: string;
  };
  expect(body.options.target).toEqual({ worker: 'local', devices: ['cpu'] });
  expect(body.options.device).toBe('cpu');
  // A blank seed field is an unseeded run, sent as such.
  expect(body.options.seed).toBeNull();
  expect(body.name).toBe(name);
  const { run_id: runId } = (await response.json()) as { run_id: string };
  await expect(dialog).toBeHidden();
  await expect(page.getByText(`Queued ${name}`)).toBeVisible();

  // The Runs tab is in front, with the new run's row and its detail open.
  const row = page.locator(`[data-run-id="${runId}"]`);
  await expect(row).toBeVisible();
  await expect(row).toHaveAttribute('aria-expanded', 'true');
  await expect(row.getByText('This server')).toBeVisible();
  await expect(row.getByText('cpu', { exact: true })).toBeVisible();
  await expect(row.getByText('Succeeded')).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId('run-detail')).toBeVisible();

  // The server recorded it as a run on this server.
  const run = (await (await page.request.get(`/api/runs/${runId}`)).json()) as {
    worker_id: string;
    devices: string[];
    status: string;
  };
  expect(run).toMatchObject({ worker_id: 'local', devices: ['cpu'], status: 'succeeded' });
});
