// Starts the real backend for the browser suite (playwright.config.ts,
// `webServer`). It serves the production build in frontend/dist, exactly as
// `cdui start` does, with every storage path in a temporary folder.
//
// A wrapper rather than a bare `python -m uvicorn` command, because the
// environment has to be CLEAN: Playwright starts a webServer with the
// parent's environment plus the config's, so a developer's own
// CODEFYUI_PROJECT_DIR, CODEFYUI_GRAPHS_DIR or CODEFYUI_USER_DATA_DIR would
// otherwise reach the server and point it back at their real data. Every
// CODEFYUI_* variable is dropped here and only the suite's are set.
//
// The server's output also goes to <CODEFYUI_E2E_DIR>/server.log, which CI
// uploads when a run fails.
import { spawn } from 'node:child_process';
import { createWriteStream, existsSync, mkdirSync } from 'node:fs';
import path from 'node:path';

const settings = JSON.parse(process.env.CODEFYUI_E2E_SERVER_ENV ?? '{}');
const python = process.env.CODEFYUI_E2E_PYTHON;
const backendDir = process.env.CODEFYUI_E2E_BACKEND_DIR;
const e2eDir = process.env.CODEFYUI_E2E_DIR;
if (!python || !backendDir || !e2eDir || !settings.CODEFYUI_PORT) {
  console.error('e2e/server.mjs is started by playwright.config.ts; run `pnpm e2e`.');
  process.exit(2);
}
if (!existsSync(python)) {
  console.error(
    `No Python at ${python}. Create backend/.venv (see CONTRIBUTING.md) or set CODEFYUI_E2E_PYTHON.`,
  );
  process.exit(2);
}

const env = Object.fromEntries(
  Object.entries(process.env).filter(([name]) => !name.toUpperCase().startsWith('CODEFYUI_')),
);
Object.assign(env, settings, { PYTHONUNBUFFERED: '1', MPLBACKEND: 'Agg' });
for (const name of [
  'CODEFYUI_USER_DATA_DIR', 'CODEFYUI_GRAPHS_DIR', 'CODEFYUI_USER_PRESETS_DIR',
  'CODEFYUI_MODELS_DIR', 'CODEFYUI_IMAGES_DIR', 'CODEFYUI_DATA_FILES_DIR',
  'CODEFYUI_MEDIA_DIR', 'CODEFYUI_LOG_DIR',
]) {
  mkdirSync(settings[name], { recursive: true });
}
mkdirSync(path.dirname(settings.CODEFYUI_DB_PATH), { recursive: true });

const log = createWriteStream(path.join(e2eDir, 'server.log'), { flags: 'a' });
const child = spawn(
  python,
  ['-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', settings.CODEFYUI_PORT],
  { cwd: backendDir, env, stdio: ['ignore', 'pipe', 'pipe'] },
);
child.stdout.on('data', (chunk) => { process.stdout.write(chunk); log.write(chunk); });
child.stderr.on('data', (chunk) => { process.stderr.write(chunk); log.write(chunk); });
child.on('exit', (code, signal) => {
  log.end();
  process.exit(code ?? (signal ? 1 : 0));
});
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => child.kill(signal));
}
