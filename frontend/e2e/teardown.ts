import { rmSync } from 'node:fs';
import { E2E_DIR } from './env';

/** Delete the run's temporary data folder, unless the caller chose it. */
export default function teardown(): void {
  if (process.env.CODEFYUI_E2E_DIR_IS_TEMP !== '1' || process.env.CODEFYUI_E2E_KEEP === '1') {
    return;
  }
  try {
    rmSync(E2E_DIR, { recursive: true, force: true });
  } catch {
    // The server may still hold a file open (Windows); the OS cleans tmp.
  }
}
