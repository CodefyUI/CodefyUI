import { existsSync } from 'node:fs';
import path from 'node:path';
import { defineConfig, devices } from '@playwright/test';
import { BACKEND_DIR, BASE_URL, DATA, E2E_DIR, PORT, PYTHON, REPO_ROOT } from './e2e/env';

/**
 * Browser workflow checks (#642): the production build in frontend/dist,
 * served by the real backend, driven in Chromium. Run with `pnpm e2e`, which
 * builds first. See CONTRIBUTING.md, "Testing".
 *
 * One worker: the specs share one server, and each test gets a fresh
 * browser context (its own IndexedDB), so the order they run in does not
 * matter, but two of them saving at once would only add noise.
 */
if (!existsSync(path.join(REPO_ROOT, 'frontend', 'dist', 'index.html'))) {
  throw new Error('frontend/dist is missing: run `pnpm e2e` (it builds first), or `pnpm build`.');
}

export default defineConfig({
  testDir: './e2e',
  testMatch: '*.spec.ts',
  globalTeardown: './e2e/teardown.ts',
  outputDir: path.join('test-results', 'e2e'),
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: 0,
  timeout: 90_000,
  expect: { timeout: 15_000 },
  reporter: process.env.CI
    ? [['list'], ['html', { open: 'never', outputFolder: 'playwright-report' }], ['github']]
    : [['list']],
  use: {
    baseURL: BASE_URL,
    locale: 'en-US',
    timezoneId: 'UTC',
    viewport: { width: 1920, height: 1080 },
    acceptDownloads: true,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'off',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'], viewport: { width: 1920, height: 1080 } } }],
  webServer: {
    command: 'node e2e/server.mjs',
    url: `${BASE_URL}/api/health`,
    // Never attach to a server that is already running: it would be the
    // developer's own, with their data.
    reuseExistingServer: false,
    timeout: 120_000,
    // The server's log is also written to <CODEFYUI_E2E_DIR>/server.log.
    stdout: 'ignore',
    stderr: 'pipe',
    env: {
      CODEFYUI_E2E_PYTHON: PYTHON,
      CODEFYUI_E2E_BACKEND_DIR: BACKEND_DIR,
      CODEFYUI_E2E_DIR: E2E_DIR,
      CODEFYUI_E2E_SERVER_ENV: JSON.stringify({
        CODEFYUI_PORT: String(PORT),
        CODEFYUI_USER_DATA_DIR: DATA.userData,
        CODEFYUI_DB_PATH: DATA.db,
        CODEFYUI_GRAPHS_DIR: DATA.graphs,
        CODEFYUI_USER_PRESETS_DIR: DATA.presets,
        CODEFYUI_MODELS_DIR: DATA.models,
        CODEFYUI_IMAGES_DIR: DATA.images,
        CODEFYUI_DATA_FILES_DIR: DATA.files,
        CODEFYUI_MEDIA_DIR: DATA.media,
        CODEFYUI_LOG_DIR: DATA.logs,
      }),
    },
  },
});
