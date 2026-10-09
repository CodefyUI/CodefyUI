/**
 * Where the browser suite's server lives and keeps its data.
 *
 * Read by playwright.config.ts (to start the server through e2e/server.mjs)
 * and by the specs (to read what the server wrote). One temporary directory
 * per `playwright test` invocation: the config creates it and exports
 * CODEFYUI_E2E_DIR, and the worker processes inherit that variable, so every
 * process of one run agrees on the same folder. Set CODEFYUI_E2E_DIR
 * yourself to choose it (CI does, to upload the server log from it).
 */
import { mkdirSync, mkdtempSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));

export const REPO_ROOT = path.resolve(here, '..', '..');
export const BACKEND_DIR = path.join(REPO_ROOT, 'backend');
export const FIXTURES_DIR = path.join(here, 'fixtures');

/** The Python that runs the server and the exported scripts. */
export const PYTHON =
  process.env.CODEFYUI_E2E_PYTHON ||
  (process.platform === 'win32'
    ? path.join(BACKEND_DIR, '.venv', 'Scripts', 'python.exe')
    : path.join(BACKEND_DIR, '.venv', 'bin', 'python'));

export const PORT = Number(process.env.CODEFYUI_E2E_PORT || 18642);
export const BASE_URL = `http://127.0.0.1:${PORT}`;

if (process.env.CODEFYUI_E2E_DIR) {
  mkdirSync(process.env.CODEFYUI_E2E_DIR, { recursive: true });
} else {
  process.env.CODEFYUI_E2E_DIR = mkdtempSync(path.join(os.tmpdir(), 'codefyui-e2e-'));
  // Ours to delete when the run ends (e2e/teardown.ts). A folder chosen by
  // the caller is left for them.
  process.env.CODEFYUI_E2E_DIR_IS_TEMP = '1';
}

/** The run's temporary root. Everything the server writes is under it. */
export const E2E_DIR = process.env.CODEFYUI_E2E_DIR;

/**
 * The server's storage, every path under E2E_DIR. Passed as CODEFYUI_*
 * settings (backend/app/config.py), so nothing reaches the developer's
 * saved graphs, presets, run database or user data directory.
 */
export const DATA = {
  userData: path.join(E2E_DIR, 'user-data'),
  db: path.join(E2E_DIR, 'db', 'codefyui.db'),
  graphs: path.join(E2E_DIR, 'graphs'),
  presets: path.join(E2E_DIR, 'presets'),
  models: path.join(E2E_DIR, 'models'),
  images: path.join(E2E_DIR, 'images'),
  files: path.join(E2E_DIR, 'files'),
  media: path.join(E2E_DIR, 'media'),
  logs: path.join(E2E_DIR, 'logs'),
};
