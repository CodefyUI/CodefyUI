/**
 * Projects (phase 4A): create a project, save a graph into it, commit it,
 * switch to another project and back.
 *
 * The shared server pins every data folder (flat mode), so this spec starts
 * a server of its own through e2e/server.mjs, with nothing pinned: its data
 * folders come from the project it opens. It starts on a project named
 * "first" that the registry (projects.json) names as current, so it never
 * looks at the checkout's backend/data, and a global git config of its own
 * gives commits an author. Everything is under the run's folder.
 */
import { execFileSync, spawn, type ChildProcess } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, realpathSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import type { Page } from '@playwright/test';
import { answerPrompt, canvasNodes, expect, fixture, importFile, test } from './app';
import { BACKEND_DIR, E2E_DIR, PORT, PYTHON } from './env';

const OWN_PORT = PORT + 1;
const OWN_URL = `http://127.0.0.1:${OWN_PORT}`;
const ROOT = path.join(E2E_DIR, 'projects-spec');
const PROJECTS = path.join(ROOT, 'projects');
const GIT_CONFIG = path.join(ROOT, 'gitconfig');

let server: ChildProcess | null = null;

async function waitForHealth(): Promise<void> {
  const deadline = Date.now() + 120_000;
  for (;;) {
    try {
      const res = await fetch(`${OWN_URL}/api/health`);
      if (res.ok) return;
    } catch {
      /* not listening yet */
    }
    if (Date.now() > deadline) throw new Error(`no server on ${OWN_URL}; see ${ROOT}/server.log`);
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
}

test.beforeAll(async () => {
  const first = path.join(PROJECTS, 'first');
  mkdirSync(path.join(first, 'graphs'), { recursive: true });
  writeFileSync(path.join(first, 'codefyui.project.toml'), '[project]\nname = "first"\n');
  const userData = path.join(ROOT, 'user-data');
  mkdirSync(userData, { recursive: true });
  const firstPath = realpathSync(first);
  writeFileSync(path.join(userData, 'projects.json'), JSON.stringify({
    version: 1,
    current: firstPath,
    recent: [{ path: firstPath, name: 'first', last_opened: '2026-01-01T00:00:00+00:00' }],
  }));
  writeFileSync(GIT_CONFIG,
    '[user]\n\tname = CodefyUI E2E\n\temail = e2e@example.invalid\n');

  server = spawn(process.execPath, [path.join('e2e', 'server.mjs')], {
    env: {
      ...process.env,
      CODEFYUI_E2E_PYTHON: PYTHON,
      CODEFYUI_E2E_BACKEND_DIR: BACKEND_DIR,
      CODEFYUI_E2E_DIR: ROOT,
      CODEFYUI_E2E_SERVER_ENV: JSON.stringify({
        CODEFYUI_PORT: String(OWN_PORT),
        CODEFYUI_USER_DATA_DIR: userData,
        CODEFYUI_DB_PATH: path.join(ROOT, 'db', 'codefyui.db'),
        CODEFYUI_USER_PRESETS_DIR: path.join(ROOT, 'presets'),
        CODEFYUI_DATA_FILES_DIR: path.join(ROOT, 'files'),
        CODEFYUI_LOG_DIR: path.join(ROOT, 'logs'),
        CODEFYUI_PROJECTS_ROOT: PROJECTS,
        CODEFYUI_DEFAULT_PROJECT: '0',
      }),
      GIT_CONFIG_GLOBAL: GIT_CONFIG,
      GIT_CONFIG_NOSYSTEM: '1',
    },
    stdio: 'ignore',
  });
  await waitForHealth();
});

test.afterAll(async () => {
  if (server === null) return;
  const exited = new Promise((resolve) => server!.once('exit', resolve));
  server.kill('SIGTERM');
  await exited;
});

const chip = (page: Page) => page.getByTestId('project-chip-name');

/** git's output in *repo*, trimmed, or '' when git fails. */
function git(repo: string, ...args: string[]): string {
  try {
    return execFileSync('git', ['-C', repo, ...args], {
      env: { ...process.env, GIT_CONFIG_GLOBAL: GIT_CONFIG, GIT_CONFIG_NOSYSTEM: '1' },
      stdio: ['ignore', 'pipe', 'ignore'],
    }).toString().trim();
  } catch {
    return '';
  }
}

/** The newest commit's subject, or '' while the branch has none. */
function lastCommit(repo: string): string {
  return git(repo, 'log', '-1', '--format=%s');
}

async function openChipMenu(page: Page) {
  await page.getByRole('button', { name: /^Project: / }).click();
  return page.getByRole('menu');
}

test('create a project, save and commit a graph, switch away and back', async ({ page }) => {
  test.setTimeout(180_000);
  await page.goto(OWN_URL);
  await expect(chip(page)).toHaveText('first');
  await expect(page.getByRole('tab', { name: 'Graphs' })).toBeVisible();

  // A folder that is not a project is refused in the dialog, nothing switches.
  let menu = await openChipMenu(page);
  await menu.getByRole('menuitem', { name: 'Open folder…' }).click();
  let dialog = page.getByTestId('project-dialog-open');
  await dialog.getByLabel(/Project folder/).fill(path.join(ROOT, 'logs'));
  await dialog.getByRole('button', { name: 'Open', exact: true }).click();
  await expect(dialog.getByRole('alert')).toContainText('is not a CodefyUI project');
  // The focus is back in the field, and the keyboard alone closes the dialog.
  await expect(dialog.getByLabel(/Project folder/)).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(dialog).toHaveCount(0);
  await expect(chip(page)).toHaveText('first');

  // Create "second": scaffolded, git-initialised, opened, the page reloads.
  menu = await openChipMenu(page);
  await menu.getByRole('menuitem', { name: 'New project…' }).click();
  dialog = page.getByTestId('project-dialog-create');
  await dialog.getByLabel('Project name').fill('second');
  await dialog.getByRole('button', { name: 'Create and open' }).click();
  await expect(chip(page)).toHaveText('second');
  const second = path.join(PROJECTS, 'second');
  expect(existsSync(path.join(second, 'codefyui.project.toml'))).toBe(true);
  // On main, with the starter files committed: Source Control starts clean.
  expect(lastCommit(second)).toBe('Initial project');
  expect(git(second, 'branch', '--show-current')).toBe('main');
  await expect(page.getByRole('img', { name: 'Uncommitted changes' })).toHaveCount(0);

  // Save a graph into it.
  await importFile(page, fixture('tensor-chain.json'));
  await expect(canvasNodes(page)).toHaveCount(5);
  await page.getByRole('button', { name: 'Save', exact: true }).click();
  await answerPrompt(page, 'Enter a name for this graph:', 'in-second');
  await expect(page.getByText('Graph "in-second" saved successfully.')).toBeVisible();
  expect(readdirSync(path.join(second, 'graphs'))).toContain('in-second.graph.json');
  await expect(page.getByText('Graphs · second')).toBeVisible();

  // Commit it from Source Control.
  await page.getByRole('tab', { name: 'Source Control' }).click();
  await page.getByPlaceholder(/Message/).fill('Add in-second');
  await page.getByRole('button', { name: 'Commit options' }).click();
  await page.getByRole('menuitem', { name: 'Commit All (includes new files)' }).click();
  await expect.poll(() => lastCommit(second), { timeout: 30_000 }).toBe('Add in-second');
  await expect(page.getByRole('img', { name: 'Uncommitted changes' })).toHaveCount(0);

  // Switch back to "first": its own (empty) graph list, its own tabs.
  menu = await openChipMenu(page);
  await menu.getByRole('menuitem', { name: 'Open first' }).click();
  await expect(chip(page)).toHaveText('first');
  await page.getByRole('tab', { name: 'Graphs' }).click();
  await expect(page.getByText('Graphs · first')).toBeVisible();
  await expect(page.getByText('in-second', { exact: true })).toHaveCount(0);

  // And to "second" again: the graph is there.
  menu = await openChipMenu(page);
  await menu.getByRole('menuitem', { name: 'Open second' }).click();
  await expect(chip(page)).toHaveText('second');
  await page.getByRole('tab', { name: 'Graphs' }).click();
  await expect(page.getByText('in-second').first()).toBeVisible();
});

test('a window left on the old project is told to reload', async ({ browser }) => {
  const returning = await browser.newPage();
  const writer = await browser.newPage();
  const mover = await browser.newPage();
  for (const page of [returning, writer, mover]) await page.goto(OWN_URL);
  const before = (await (await fetch(`${OWN_URL}/api/projects`)).json()).current.name as string;
  for (const page of [returning, writer, mover]) await expect(chip(page)).toHaveText(before);
  const target = before === 'second' ? 'first' : 'second';

  const menu = await openChipMenu(mover);
  await menu.getByRole('menuitem', { name: `Open ${target}` }).click();
  await expect(chip(mover)).toHaveText(target);

  // Coming back to a window re-reads /api/health and shows the banner.
  await expect(returning.getByTestId('project-stale-banner')).toHaveCount(0);
  await returning.bringToFront();
  await returning.evaluate(() => window.dispatchEvent(new Event('focus')));
  await expect(returning.getByTestId('project-stale-banner')).toBeVisible();
  await returning.getByTestId('project-stale-banner').getByRole('button', { name: 'Reload' }).click();
  await expect(chip(returning)).toHaveText(target);

  // The backstop: a write from a window that never regained focus is
  // refused with 409 project_changed, and the banner says why.
  await importFile(writer, fixture('tensor-chain.json'));
  await expect(canvasNodes(writer)).toHaveCount(5);
  await writer.getByRole('button', { name: 'Save', exact: true }).click();
  await answerPrompt(writer, 'Enter a name for this graph:', 'from-stale-window');
  await expect(writer.getByTestId('project-stale-banner')).toBeVisible();
  const stalePath = path.join(PROJECTS, before, 'graphs');
  const targetPath = path.join(PROJECTS, target, 'graphs');
  for (const dir of [stalePath, targetPath]) {
    expect(readdirSync(dir).some((f) => f.startsWith('from-stale-window'))).toBe(false);
  }

  for (const page of [returning, writer, mover]) await page.close();
});
