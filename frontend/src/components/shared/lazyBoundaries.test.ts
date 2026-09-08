import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * The two lazy chunk boundaries (codemirrorSetup.ts, katexRenderer.tsx) are
 * undone silently by a single static import: the code keeps working, the
 * chunk just moves into the first paint. vite.config.ts catches the package
 * side of that at build time; this test catches the module side in
 * `pnpm test`, by reading the sources from disk the way
 * SourceControl.integration.test.tsx reads its stylesheets.
 */
const BOUNDARIES = ['codemirrorSetup', 'katexRenderer'];

const SRC = join(dirname(fileURLToPath(import.meta.url)), '..', '..');

function sourceFiles(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name !== 'test') sourceFiles(full, out);
    } else if (/\.tsx?$/.test(entry.name) && !/\.(test|d)\.tsx?$/.test(entry.name)) {
      out.push(full);
    }
  }
  return out;
}

// `import x from '...'`, `import '...'` and `export ... from '...'`; a
// `typeof import('...')` type or an `import()` expression is not matched.
const STATIC_IMPORT = /^\s*(?:import|export)\b[^;]*?\bfrom\s+['"]([^'"]+)['"]|^\s*import\s+['"]([^'"]+)['"]/gm;

describe('lazy chunk boundaries', () => {
  const files = sourceFiles(SRC);

  it('sees both boundary modules', () => {
    for (const name of BOUNDARIES) {
      expect(files.some((file) => new RegExp(`/${name}\\.tsx?$`).test(file))).toBe(true);
    }
  });

  it('are never imported statically', () => {
    const offenders: string[] = [];
    for (const file of files) {
      for (const match of readFileSync(file, 'utf8').matchAll(STATIC_IMPORT)) {
        if (/^\s*import\s+type\b/.test(match[0])) continue;
        const specifier = match[1] ?? match[2];
        if (BOUNDARIES.some((name) => specifier.endsWith(`/${name}`))) {
          offenders.push(`${relative(SRC, file)} imports ${specifier}`);
        }
      }
    }
    expect(offenders).toEqual([]);
  });
});
