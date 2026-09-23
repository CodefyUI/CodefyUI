/**
 * `pnpm test:act-baseline`: run the whole frontend suite and write each test
 * file's act() warning count to act-warnings.baseline.json (see
 * act-warnings.mjs, which does the counting and the writing).
 *
 * The baseline only goes down: if a file rose, the command fails and the
 * baseline is left as it was. `--allow-increase` writes the counts as they are.
 */

import { spawnSync } from 'node:child_process';
import { statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { BASELINE_PATH, MODE_ENV } from './act-warnings.mjs';

const args = process.argv.slice(2).filter((arg) => arg !== '--');
if (args.some((arg) => arg !== '--allow-increase')) {
  console.error('usage: pnpm test:act-baseline [--allow-increase]');
  process.exit(2);
}

const writtenAt = () => {
  try {
    return statSync(BASELINE_PATH).mtimeMs;
  } catch {
    return undefined;
  }
};

const before = writtenAt();
// The real CLI, not the programmatic API, so the run is the one `pnpm test`
// makes; the whole suite, because a partial run cannot say which files are clean.
// `--allowOnly=false`, as on CI: a stray local `it.only` fails the run instead
// of skipping the rest of its file and lowering that file's count.
const run = spawnSync(
  process.execPath,
  [fileURLToPath(new URL('../node_modules/vitest/vitest.mjs', import.meta.url)), 'run', '--allowOnly=false'],
  {
    cwd: fileURLToPath(new URL('..', import.meta.url)),
    stdio: 'inherit',
    env: { ...process.env, [MODE_ENV]: args.length > 0 ? 'accept' : 'lower' },
  },
);
if (run.error) throw run.error;
if (run.status === 0 && writtenAt() === before) {
  console.error('act-warnings.baseline.json was not written: is actWarningsGate() in vitest.config.ts?');
  process.exit(1);
}
process.exit(run.status ?? 1);
