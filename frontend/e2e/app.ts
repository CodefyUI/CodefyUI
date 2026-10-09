/**
 * Shared steps for the browser workflow checks.
 *
 * Locators are the ones a user sees: roles and visible English labels (the
 * config pins `locale: 'en-US'`), plus React Flow's own `rf__node-<id>` /
 * `rf__edge-<id>` test ids, which carry the graph's ids and so let a check
 * name the node it means.
 */
import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';
import {
  test as base,
  expect,
  type Download,
  type Locator,
  type Page,
  type TestInfo,
} from '@playwright/test';
import { DATA, FIXTURES_DIR } from './env';

/**
 * `test` with an automatic fixture that records the browser console and
 * every failed or 4xx/5xx request, and attaches both to a failing test's
 * report next to Playwright's own trace and screenshot.
 */
export const test = base.extend<{ browserLogs: void }>({
  browserLogs: [
    async ({ page }, use, testInfo) => {
      const consoleLines: string[] = [];
      const networkLines: string[] = [];
      page.on('console', (message) => {
        consoleLines.push(`[${message.type()}] ${message.text()}`);
      });
      page.on('pageerror', (error) => {
        consoleLines.push(`[pageerror] ${error.stack ?? error.message}`);
      });
      page.on('requestfailed', (request) => {
        networkLines.push(
          `FAILED ${request.method()} ${request.url()} ${request.failure()?.errorText ?? ''}`,
        );
      });
      page.on('response', (response) => {
        if (response.status() >= 400) {
          networkLines.push(`${response.status()} ${response.request().method()} ${response.url()}`);
        }
      });
      await use();
      if (testInfo.status !== testInfo.expectedStatus) {
        await testInfo.attach('browser-console.log', {
          body: consoleLines.join('\n') || '(empty)',
          contentType: 'text/plain',
        });
        await testInfo.attach('network-failures.log', {
          body: networkLines.join('\n') || '(none)',
          contentType: 'text/plain',
        });
      }
    },
    { auto: true },
  ],
});

export { expect };

export const fixture = (name: string): string => path.join(FIXTURES_DIR, name);

/**
 * A name for something a test stores on the shared server (a saved graph, a
 * preset), distinct per repeat and retry, so `--repeat-each` and a retry
 * never find their own earlier copy and get an overwrite question instead.
 */
export const uniqueName = (testInfo: TestInfo, base: string): string =>
  `${base} ${testInfo.repeatEachIndex}.${testInfo.retry}`;

/**
 * Wait until the autosave has put `tabCount` tabs in IndexedDB. The tab
 * store writes on a 250 ms trailing debounce, so a reload right after a
 * change could otherwise come back without it.
 */
export async function waitForAutosave(page: Page, tabCount: number): Promise<void> {
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          new Promise<number>((resolve) => {
            const open = indexedDB.open('codefyui');
            open.onerror = () => resolve(-1);
            open.onsuccess = () => {
              const db = open.result;
              try {
                const read = db.transaction('kv', 'readonly').objectStore('kv').get('codefyui-tabs|meta');
                read.onsuccess = () => {
                  db.close();
                  const meta = read.result as { tabIds?: unknown[] } | undefined;
                  resolve(Array.isArray(meta?.tabIds) ? meta.tabIds.length : 0);
                };
                read.onerror = () => {
                  db.close();
                  resolve(-1);
                };
              } catch {
                db.close();
                resolve(-1);
              }
            };
          }),
      ),
    )
    .toBe(tabCount);
}

export const readJson = <T = Record<string, unknown>>(file: string): T =>
  JSON.parse(readFileSync(file, 'utf-8')) as T;

export interface GraphNode {
  id: string;
  type: string;
  data?: { params?: Record<string, unknown> };
}
export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  sourceHandle?: string;
  targetHandle?: string;
}
export interface GraphDocument {
  name: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  settings?: { seed?: number; device?: string };
  subgraphs?: unknown[];
}

/** The saved-graph file the server wrote for the graph called `name`. */
export function savedGraphFile(name: string): string {
  for (const entry of readdirSync(DATA.graphs)) {
    if (!entry.endsWith('.json')) continue;
    const file = path.join(DATA.graphs, entry);
    if (readJson<GraphDocument>(file).name === name) return file;
  }
  throw new Error(`No saved graph named "${name}" in ${DATA.graphs}`);
}

/** What a check compares between two copies of one graph: ids, types,
 * params and wires, with positions and other layout left out. */
export function documentShape(graph: GraphDocument) {
  return {
    nodes: graph.nodes
      .map((node) => ({ id: node.id, type: node.type, params: node.data?.params ?? {} }))
      .sort((a, b) => a.id.localeCompare(b.id)),
    edges: graph.edges
      .map((edge) => `${edge.source}.${edge.sourceHandle ?? ''}->${edge.target}.${edge.targetHandle ?? ''}`)
      .sort(),
  };
}

export async function openApp(page: Page): Promise<void> {
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Run', exact: false }).first()).toBeVisible();
}

export const node = (page: Page, id: string): Locator => page.getByTestId(`rf__node-${id}`);
export const edge = (page: Page, id: string): Locator => page.getByTestId(`rf__edge-${id}`);
export const canvasNodes = (page: Page): Locator => page.locator('.react-flow__node');
export const canvasEdges = (page: Page): Locator => page.locator('.react-flow__edge');
export const workspaceTabs = (page: Page): Locator =>
  page.getByRole('tablist', { name: 'Workspace tabs' }).getByRole('tab');

/** Frame the canvas, so a node is not under the panels a selection opens. */
export async function fitView(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'Fit View' }).click();
}

/** Click a node's title bar: selects it without touching a field. */
export async function selectNode(
  page: Page,
  id: string,
  options: { add?: boolean; button?: 'left' | 'right' } = {},
): Promise<void> {
  await fitView(page);
  await node(page, id).click({
    position: { x: 24, y: 10 },
    button: options.button ?? 'left',
    modifiers: options.add ? ['Shift'] : [],
  });
}

/** Import a .json graph or a .cduiworkspace through the Graphs panel. */
export async function importFile(page: Page, file: string): Promise<void> {
  const importButton = page.getByRole('button', { name: 'Import...' });
  // A second click on the panel's own tab folds the sidebar away.
  if (!(await importButton.isVisible())) {
    await page.getByRole('tab', { name: 'Graphs' }).click();
  }
  const chooser = page.waitForEvent('filechooser');
  await importButton.click();
  await (await chooser).setFiles(file);
}

/** Answer the app's one-line prompt dialog. */
export async function answerPrompt(page: Page, title: string | RegExp, value: string): Promise<void> {
  const dialog = page.getByRole('dialog', { name: title });
  await expect(dialog).toBeVisible();
  await dialog.getByRole('textbox').fill(value);
  await dialog.getByRole('button', { name: 'OK' }).click();
  await expect(dialog).toBeHidden();
}

/**
 * Click an item of the toolbar's Export menu and wait for its download. A
 * tab that was never saved is asked for a file name first; the suggested
 * one is kept.
 */
export async function exportDownload(page: Page, item: string): Promise<Download> {
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export', exact: true }).click();
  await page.getByRole('button', { name: item, exact: true }).click();
  const nameDialog = page.getByRole('dialog', { name: /^File name/ });
  const first = await Promise.race([
    download.then(() => 'download'),
    nameDialog.waitFor({ timeout: 10_000 }).then(() => 'dialog', () => 'none'),
  ]);
  if (first === 'dialog') {
    await nameDialog.getByRole('button', { name: 'OK' }).click();
  }
  return download;
}

/** Export the active tab as JSON and read the file it downloads. */
export async function exportGraphJson(page: Page, dir: string): Promise<GraphDocument> {
  const download = await exportDownload(page, 'Export as JSON');
  const file = path.join(dir, `${Date.now()}-${download.suggestedFilename()}`);
  await download.saveAs(file);
  return readJson<GraphDocument>(file);
}

/** Press Run and wait for this run, not an earlier one, to succeed. */
export async function runGraph(page: Page): Promise<void> {
  const done = page.getByText('Execution completed successfully', { exact: true });
  // Empty the Execution Log first, so the line waited for is this run's.
  const clear = page.getByRole('button', { name: 'Clear', exact: true });
  if (await clear.isEnabled()) {
    await clear.click();
    await expect(done).toHaveCount(0);
  }
  await page.getByRole('button', { name: 'Run', exact: false }).first().click();
  await expect(done).toHaveCount(1, { timeout: 60_000 });
}

/** The last line a Print node with `label` wrote in the Execution Log. */
export async function printedLine(page: Page, label: string): Promise<string> {
  const lines = page.getByText(new RegExp(`^\\[${label}\\] `));
  await expect(lines.first()).toBeVisible();
  return (await lines.last().innerText()).trim();
}
