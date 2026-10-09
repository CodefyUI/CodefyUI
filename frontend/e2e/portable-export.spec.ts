/**
 * Golden flow 3 (#642): portable execution.
 *
 * Make a preset in the browser (Export as Subgraph), save a seeded graph
 * whose Map runs that preset, run it on the canvas, export it as Python, and
 * run the script the way a grader does (#639): `python -I` in an empty
 * folder, in a fresh process whose user-data and user-presets folders are
 * empty, so the preset is NOT installed there. The script has to carry its
 * own copy, and print the number the canvas printed.
 */
import { spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import {
  answerPrompt,
  canvasNodes,
  expect,
  exportDownload,
  fixture,
  importFile,
  openApp,
  printedLine,
  runGraph,
  savedGraphFile,
  test,
  uniqueName,
  workspaceTabs,
  type GraphDocument,
} from './app';
import { DATA, PYTHON } from './env';

test('an exported Map graph prints the canvas result where its preset is not installed', async ({
  page,
}, testInfo) => {
  const PRESET_NAME = uniqueName(testInfo, 'E2E row scale');
  const GRAPH_NAME = uniqueName(testInfo, 'E2E portable map');
  // The graph fixture, with its Map pointed at this test's preset.
  const graph = JSON.parse(readFileSync(fixture('portable-map.json'), 'utf-8')) as GraphDocument;
  graph.nodes.find((n) => n.id === 'map')!.data!.params!.subgraph = PRESET_NAME;
  const graphFile = testInfo.outputPath('portable-map.json');
  writeFileSync(graphFile, JSON.stringify(graph));

  // 1. The preset, made from a tab the way a user makes one.
  await openApp(page);
  await importFile(page, fixture('row-scale-body.json'));
  await expect(canvasNodes(page)).toHaveCount(2);
  await page.getByRole('button', { name: 'Export', exact: true }).click();
  await page.getByRole('button', { name: 'Export as Subgraph', exact: true }).click();
  await answerPrompt(page, 'Enter a name for this subgraph:', PRESET_NAME);
  await expect(page.getByText(`Subgraph "${PRESET_NAME}" is now in the Nodes tab.`)).toBeVisible();
  const presets = readdirSync(DATA.presets).map((name) =>
    JSON.parse(readFileSync(path.join(DATA.presets, name), 'utf-8')),
  );
  expect(presets.map((preset) => preset.preset_name)).toContain(PRESET_NAME);

  // 2. The graph that maps it, saved with its seed and device.
  await importFile(page, graphFile);
  await expect(workspaceTabs(page)).toHaveCount(2);
  await expect(canvasNodes(page)).toHaveCount(6);
  await page.getByRole('button', { name: 'Save', exact: true }).click();
  await answerPrompt(page, 'Enter a name for this graph:', GRAPH_NAME);
  await expect(page.getByText(`Graph "${GRAPH_NAME}" saved successfully.`)).toBeVisible();
  const saved = JSON.parse(readFileSync(savedGraphFile(GRAPH_NAME), 'utf-8')) as GraphDocument;
  expect(saved.settings?.seed).toBe(1234);
  expect(saved.nodes.find((n) => n.id === 'map')?.data?.params?.subgraph).toBe(PRESET_NAME);

  // 3. The canvas run: a seeded sum of three Map results.
  await runGraph(page);
  const canvasLine = await printedLine(page, 'e2e_portable');
  expect(canvasLine).toMatch(/^\[e2e_portable\] -?\d+\.\d+(e-?\d+)?$/);
  // A second run with the same seed prints the same number.
  await runGraph(page);
  expect(await printedLine(page, 'e2e_portable')).toBe(canvasLine);

  // 4. Export as Python.
  const download = await exportDownload(page, 'Export as Python');
  const judge = testInfo.outputPath('judge');
  mkdirSync(judge, { recursive: true });
  const script = path.join(judge, 'graph.py');
  await download.saveAs(script);

  // 5. A fresh process with nothing installed but CodefyUI itself.
  const emptyPresets = testInfo.outputPath('judge-user-presets');
  const emptyUserData = testInfo.outputPath('judge-user-data');
  mkdirSync(emptyPresets, { recursive: true });
  mkdirSync(emptyUserData, { recursive: true });
  const env = Object.fromEntries(
    Object.entries(process.env).filter(([name]) => !name.toUpperCase().startsWith('CODEFYUI_')),
  );
  const run = spawnSync(PYTHON, ['-I', 'graph.py'], {
    cwd: judge,
    env: {
      ...env,
      CODEFYUI_USER_PRESETS_DIR: emptyPresets,
      CODEFYUI_USER_DATA_DIR: emptyUserData,
      MPLBACKEND: 'Agg',
      PYTHONIOENCODING: 'utf-8',
    },
    encoding: 'utf-8',
    timeout: 120_000,
  });
  await testInfo.attach('exported-script-output.log', {
    body: `exit ${run.status}\n--- stdout\n${run.stdout}\n--- stderr\n${run.stderr}`,
    contentType: 'text/plain',
  });
  expect(run.status, run.stderr).toBe(0);
  const scriptLines = run.stdout.split(/\r?\n/).filter((line) => line.startsWith('[e2e_portable] '));
  expect(scriptLines).toEqual([canvasLine]);

  // It ran the copy of the preset the script carries.
  const source = readFileSync(script, 'utf-8');
  expect(source).toContain('_PRESET_COPIES');
  expect(source).toContain(PRESET_NAME);
});
