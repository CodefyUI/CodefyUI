/**
 * Project mods (phase 4C, acceptance 11): a project from elsewhere with
 * Python under mods/nodes/ asks before anything loads; declining still
 * opens the project and its graph shows the node as not loaded, kept as
 * saved; trusting loads it; a changed file needs consent again.
 *
 * "From elsewhere" is a folder this install did not create, opened with
 * Open folder: the same trust state as a clone (which needs a network).
 * Like projects.spec.ts this starts a server of its own, with nothing
 * pinned, everything under the run's folder.
 */
import { spawn, type ChildProcess } from 'node:child_process';
import { existsSync, mkdirSync, realpathSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import type { Page } from '@playwright/test';
import { expect, test } from './app';
import { BACKEND_DIR, E2E_DIR, PORT, PYTHON } from './env';

const OWN_PORT = PORT + 2;
const OWN_URL = `http://127.0.0.1:${OWN_PORT}`;
const ROOT = path.join(E2E_DIR, 'project-mods-spec');
const PROJECTS = path.join(ROOT, 'projects');
const ELSEWHERE = path.join(ROOT, 'elsewhere', 'shared-lab');
const MARKER = path.join(ROOT, 'mod-ran.txt');

const NODE_SOURCE = (marker: string, note = '') => [
  'from app.core.node_base import BaseNode',
  '',
  '# Leaves a trace when it runs, so the test can tell that it did.',
  `open(${JSON.stringify(marker)}, 'a').write('ran\\n')`,
  note,
  '',
  'class FocalLoss(BaseNode):',
  "    NODE_NAME = 'FocalLoss'",
  "    CATEGORY = 'Loss'",
  "    DESCRIPTION = 'A loss from this project'",
  '    @classmethod',
  '    def define_inputs(cls):',
  '        return []',
  '    @classmethod',
  '    def define_outputs(cls):',
  '        return []',
  '    def execute(self, inputs, params):',
  '        return {}',
  '',
].join('\n');

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

function writeLf(file: string, text: string): void {
  mkdirSync(path.dirname(file), { recursive: true });
  writeFileSync(file, Buffer.from(text, 'utf-8'));
}

test.beforeAll(async () => {
  const home = path.join(PROJECTS, 'home');
  writeLf(path.join(home, 'codefyui.project.toml'), '[project]\nname = "home"\n');
  mkdirSync(path.join(home, 'graphs'), { recursive: true });

  // The project from elsewhere: a mod, and a graph that uses it.
  writeLf(path.join(ELSEWHERE, 'codefyui.project.toml'), '[project]\nname = "shared-lab"\n');
  writeLf(path.join(ELSEWHERE, 'mods', 'nodes', 'focal.py'), NODE_SOURCE(MARKER));
  writeLf(path.join(ELSEWHERE, 'graphs', 'uses-focal.json'), JSON.stringify({
    name: 'uses-focal',
    nodes: [
      { id: 'focal', type: 'project:FocalLoss', position: { x: 200, y: 120 },
        data: { label: 'My focal loss', params: { gamma: 3.5 } } },
      { id: 'print', type: 'Print', position: { x: 520, y: 120 },
        data: { params: {} } },
    ],
    edges: [],
  }, null, 2) + '\n');

  const userData = path.join(ROOT, 'user-data');
  mkdirSync(userData, { recursive: true });
  const homePath = realpathSync(home);
  writeLf(path.join(userData, 'projects.json'), JSON.stringify({
    version: 1,
    current: homePath,
    recent: [{ path: homePath, name: 'home', last_opened: '2026-01-01T00:00:00+00:00' }],
  }));

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
const consent = (page: Page) => page.getByTestId('project-mods-consent');

test('a project with mods asks first, opens without them, and asks again after a change', async ({ page }) => {
  test.setTimeout(180_000);
  await page.goto(OWN_URL);
  await expect(chip(page)).toHaveText('home');

  // Open the folder from elsewhere.
  await page.getByRole('button', { name: /^Project: / }).click();
  await page.getByRole('menu').getByRole('menuitem', { name: 'Open folder…' }).click();
  const dialog = page.getByTestId('project-dialog-open');
  await dialog.getByLabel(/Project folder/).fill(ELSEWHERE);
  await dialog.getByRole('button', { name: 'Open', exact: true }).click();
  await expect(chip(page)).toHaveText('shared-lab');

  // The prompt, before anything ran: the file, its node, the plain warning.
  await expect(consent(page)).toBeVisible();
  await expect(consent(page)).toContainText('mods/nodes/focal.py');
  await expect(consent(page)).toContainText('project:FocalLoss');
  await expect(consent(page)).toContainText(
    'These files are Python. Trusting them runs them in this CodefyUI server with your permissions.');
  expect(existsSync(MARKER)).toBe(false);

  // Not now: the project is open, the mod did not run.
  await consent(page).getByRole('button', { name: 'Not now' }).click();
  await expect(consent(page)).toHaveCount(0);
  expect(existsSync(MARKER)).toBe(false);

  // The graph opens; the node is there, marked not loaded, with a banner.
  await page.getByRole('tab', { name: 'Graphs' }).click();
  await page.getByRole('button', { name: 'Open uses-focal' }).click();
  const focal = page.getByTestId('rf__node-focal');
  await expect(focal).toContainText('My focal loss');
  await expect(focal.getByTestId('node-mods-not-loaded')).toHaveText('NOT LOADED');
  await expect(page.getByTestId('mods-not-loaded-banner')).toContainText(
    'uses 1 node(s) from this project’s mods');

  // Review mods from the banner, and trust: the mod runs, the badge goes.
  await page.getByTestId('mods-not-loaded-banner').getByRole('button', { name: 'Review mods' }).click();
  await expect(consent(page)).toBeVisible();
  await consent(page).getByTestId('mods-trust').click();
  await expect(consent(page)).toHaveCount(0);
  await expect(page.getByText('Loaded 1 node(s) from mods/.')).toBeVisible();
  await expect(focal.getByTestId('node-mods-not-loaded')).toHaveCount(0);
  await expect(page.getByTestId('mods-not-loaded-banner')).toHaveCount(0);
  expect(existsSync(MARKER)).toBe(true);

  // Change the file: the next load (Reload Nodes) does not run it and asks again.
  writeLf(path.join(ELSEWHERE, 'mods', 'nodes', 'focal.py'), NODE_SOURCE(MARKER, '# edited'));
  await page.getByTestId('toolbar-reload-nodes').click();
  await expect(consent(page)).toBeVisible();
  await expect(consent(page)).toContainText('changed since you last agreed');
  await expect(consent(page).getByText('changed', { exact: true })).toBeVisible();
  await expect(focal.getByTestId('node-mods-not-loaded')).toHaveText('NOT LOADED');
});
