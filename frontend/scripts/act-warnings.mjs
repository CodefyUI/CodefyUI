/**
 * act() warnings gate (#505).
 *
 * React reports a state update that runs outside act() through console.error:
 * "An update to X inside a test was not wrapped in act(...)". Nothing failed
 * on it, and whether a passing test's console output is printed at all
 * depends on the reporter (vitest picks its agent reporter, which hides it,
 * when an AI coding agent runs the tests), so the warnings grew unseen:
 * 2,422 in one run when #505 was filed, and one PR (#517) added about 270.
 *
 * `actWarningsGate()` is a vitest plugin, registered in vitest.config.ts. Its
 * `configureVitest` hook appends a reporter to the ones vitest has already
 * chosen, so what a run prints is unchanged. The reporter counts the warnings
 * each test file prints and, when the run ends, fails `vitest run` if a file
 * has more than act-warnings.baseline.json allows. A file the baseline does
 * not list is allowed none. A file that has fewer passes, with a note, except
 * in a run that picks tests inside its files (-t, file:line): that run sees
 * only some of their warnings, so it notes no fall and never writes.
 *
 * A test counts each distinct warning once, however often it prints it. How
 * many times a test repeats a warning depends on timing: over 20 full runs
 * under load, raw totals moved by one in two of them, while the distinct
 * warnings per test were the same in all 20.
 *
 * `pnpm test:act-baseline` (act-warnings-update.mjs) runs the whole suite and
 * writes the counts back. It only lowers numbers: if a file rose, the baseline
 * is left as it was and the command fails. `--allow-increase` writes the
 * counts as they are, for a rise a reviewer has agreed to. A write updates the
 * files that ran and keeps the entries of files that did not; only a file that
 * no longer exists loses its entry.
 *
 * The gate counts what reaches the console. A test that stubs console.error
 * hides its warnings from the gate, as it does from a reader.
 */

import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

/** Part of every act() warning React prints, on the warning's first line. */
const MARKER = 'not wrapped in act(';

export const BASELINE_PATH = fileURLToPath(
  new URL('./act-warnings.baseline.json', import.meta.url),
);

/** Set by act-warnings-update.mjs: "lower", or "accept" for --allow-increase. */
export const MODE_ENV = 'CODEFYUI_ACT_BASELINE';

const BASELINE_NAME = 'scripts/act-warnings.baseline.json';

/** Where a warning goes when vitest cannot tie it to a test file. */
const NO_FILE = '(no test file)';

/** @returns {Record<string, number>} test file -> warnings it may have */
function readBaseline() {
  let text;
  try {
    text = readFileSync(BASELINE_PATH, 'utf8');
  } catch (error) {
    // Missing, every file is allowed none: the run fails and names the file.
    if (error.code === 'ENOENT') return {};
    throw error;
  }
  const baseline = JSON.parse(text);
  for (const [file, allowed] of Object.entries(baseline)) {
    if (!Number.isInteger(allowed) || allowed < 0) {
      throw new Error(`${BASELINE_NAME}: "${file}" must map to a count, not ${JSON.stringify(allowed)}`);
    }
  }
  return baseline;
}

const sum = (numbers) => numbers.reduce((a, b) => a + b, 0);
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
const byName = ([a], [b]) => (a < b ? -1 : a > b ? 1 : 0);

class ActWarningsReporter {
  /** @param {string | undefined} mode unset to check; "lower" or "accept" to write */
  constructor(mode) {
    this.mode = mode;
    /** @type {Map<string, Set<string>>} test file -> "test id + warning line" seen */
    this.seen = new Map();
  }

  onInit(vitest) {
    this.vitest = vitest;
  }

  onTestRunStart(specifications) {
    // Watch mode: each rerun is judged on its own.
    this.seen.clear();
    // A run that picks tests inside its files (-t, file:line, an editor's
    // test ids) sees only some of each file's warnings: a rise is still a
    // rise, but a fall says nothing, and such a run must not write.
    // (A file named with no line carries `testLines: []`, hence the lengths.)
    this.partial =
      this.vitest.getGlobalTestNamePattern() !== undefined ||
      specifications.some((spec) =>
        spec.testNamePattern !== undefined ||
        spec.testLines?.length > 0 ||
        spec.testIds?.length > 0 ||
        spec.testTagsFilter?.length > 0);
  }

  onUserConsoleLog(log) {
    if (log.type !== 'stderr' || !log.content.includes(MARKER)) return;
    const entity = log.taskId ? this.vitest.state.getReportedEntityById(log.taskId) : undefined;
    const module = entity?.type === 'module' ? entity : entity?.module;
    const file = module ? module.relativeModuleId.replaceAll('\\', '/') : NO_FILE;
    let seen = this.seen.get(file);
    if (!seen) this.seen.set(file, (seen = new Set()));
    // One log can carry several console.error calls: vitest batches what a
    // test writes in one tick. The line names the component that updated.
    for (const line of log.content.split('\n')) {
      if (line.includes(MARKER)) seen.add(`${log.taskId}\n${line.trim()}`);
    }
  }

  count(file) {
    return this.seen.get(file)?.size ?? 0;
  }

  /** Each file that rose, then every test in it that warns, with the components it names. */
  riseReport(rises) {
    return rises
      .flatMap(([file, count, allowed]) => [
        `  ${file}: ${count} (baseline ${allowed})`,
        ...this.warningTests(file).map(([test, components]) => `    ${test}: ${components.join(', ')}`),
      ])
      .join('\n');
  }

  /** @returns {[string, string[]][]} test name -> components, for the tests in *file* that warned */
  warningTests(file) {
    const tests = new Map();
    for (const key of this.seen.get(file) ?? []) {
      const [taskId, line] = key.split('\n');
      const entity = this.vitest.state.getReportedEntityById(taskId);
      const test = entity && entity.type !== 'module' ? entity.fullName : '(outside a test)';
      // "An update to X inside a test was not wrapped in act(...)" names X.
      // Any other act() warning is shown as it reads.
      const component = /An update to (.+?) inside a test/.exec(line)?.[1] ?? line;
      if (!tests.has(test)) tests.set(test, new Set());
      tests.get(test).add(component);
    }
    return [...tests].map(([test, components]) => [test, [...components].sort()]).sort(byName);
  }

  onTestRunEnd(testModules, _unhandledErrors, reason) {
    if (reason === 'interrupted') return;
    const baseline = readBaseline();
    // Only a file that ran and passed is judged: a failing file fails the run
    // already, and its count is not the file's real count.
    const files = testModules
      .filter((testModule) => testModule.state() === 'passed')
      .map((testModule) => testModule.relativeModuleId.replaceAll('\\', '/'));
    const judged = this.seen.has(NO_FILE) ? [...files, NO_FILE] : files;

    const rises = [];
    const falls = [];
    for (const file of judged) {
      const count = this.count(file);
      const allowed = baseline[file] ?? 0;
      if (count > allowed) rises.push([file, count, allowed]);
      else if (count < allowed) falls.push([file, count, allowed]);
    }
    rises.sort(byName);
    const logger = this.vitest.logger;

    if (this.mode === 'lower' || this.mode === 'accept') {
      if (reason !== 'passed') {
        logger.error('\nact() baseline not written: the run failed.');
        return;
      }
      if (this.partial) {
        process.exitCode = 1;
        logger.error(
          '\nact() baseline not written: the run picked tests inside its files (-t, file:line), ' +
            'so it saw only some of their warnings.',
        );
        return;
      }
      if (rises.length > 0 && this.mode === 'lower') {
        process.exitCode = 1;
        logger.error(
          `\nact() baseline not written: ${plural(rises.length, 'file')} rose above it, and it only goes down.\n` +
            `${this.riseReport(rises)}\n` +
            'Fix the new warnings, or run `pnpm test:act-baseline --allow-increase` to accept them.',
        );
        return;
      }
      // A file this run judged gets its new count, and leaves the baseline at
      // zero. A file that did not run keeps its entry: a run of some files
      // says nothing about the others. Only a file that no longer exists is
      // dropped.
      const next = { ...baseline };
      for (const file of judged) {
        if (this.count(file) > 0) next[file] = this.count(file);
        else delete next[file];
      }
      const kept = Object.keys(next).filter((file) => !judged.includes(file));
      const gone = kept.filter((file) => file !== NO_FILE && !existsSync(resolve(this.vitest.config.root, file)));
      for (const file of gone) delete next[file];
      const entries = Object.entries(next).sort(byName);
      writeFileSync(BASELINE_PATH, `${JSON.stringify(Object.fromEntries(entries), null, 2)}\n`);
      const untouched = kept.length - gone.length;
      logger.log(
        `\nact() baseline written: ${sum(entries.map(([, count]) => count))} warnings in ` +
          `${plural(entries.length, 'file')} (was ${sum(Object.values(baseline))} in ${Object.keys(baseline).length})` +
          (untouched > 0 ? `; ${plural(untouched, 'file')} that did not run kept as they were.` : '.'),
      );
      return;
    }

    if (rises.length > 0) {
      process.exitCode = 1;
      logger.error(
        `\nact() warnings rose above ${BASELINE_NAME}:\n${this.riseReport(rises)}\n` +
          'Under each file: every test in it that warns (the baseline keeps counts, not names, so old ' +
          'warnings are listed too), with the components that updated outside act(). A test counts each ' +
          'distinct warning once. The full text: pnpm exec vitest run <file> --reporter=verbose\n' +
          'Wrap the update in act(), or await what the component does next (findBy..., waitFor). ' +
          'A renamed test file keeps its allowance only if its key in the baseline is renamed too.',
      );
      return;
    }
    if (files.length === 0) return;
    let summary =
      `\nact() warnings: ${sum(judged.map((file) => this.count(file)))} in ` +
      `${plural(files.length, 'test file')}, none above the baseline.`;
    if (falls.length > 0 && !this.partial) {
      const fell = sum(falls.map(([, count, allowed]) => allowed - count));
      summary += ` ${plural(falls.length, 'file')} fell by ${fell}: run \`pnpm test:act-baseline\` to lower it.`;
    }
    logger.log(summary);
  }
}

/** The gate, for `plugins` in vitest.config.ts. */
export function actWarningsGate() {
  return {
    name: 'codefyui:act-warnings-gate',
    configureVitest({ vitest }) {
      // Appended to the reporters vitest resolved (default, agent, verbose,
      // github-actions, ...), not put in their place.
      vitest.config.reporters.push(new ActWarningsReporter(process.env[MODE_ENV]));
    },
  };
}
