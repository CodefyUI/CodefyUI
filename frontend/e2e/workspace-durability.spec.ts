/**
 * Golden flow 1 (#642): workspace durability.
 *
 * Edit a graph, take one Delete back with ONE undo (#637), save it, export
 * it as JSON and as a workspace, import a graph and that workspace into the
 * occupied workspace (#550, #627), reload the page, and check that the
 * document on the server, the tab and every export still hold the same
 * graph.
 */
import { mkdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import {
  answerPrompt,
  canvasEdges,
  canvasNodes,
  documentShape,
  edge,
  expect,
  exportDownload,
  exportGraphJson,
  fixture,
  importFile,
  node,
  openApp,
  readJson,
  savedGraphFile,
  selectNode,
  test,
  uniqueName,
  waitForAutosave,
  workspaceTabs,
  type GraphDocument,
} from './app';
import { E2E_DIR } from './env';

test('a saved graph survives Delete + one Undo, imports into an occupied workspace and a reload', async ({
  page,
}, testInfo) => {
  const GRAPH_NAME = uniqueName(testInfo, 'E2E durable graph');
  const downloads = testInfo.outputPath('downloads');
  mkdirSync(downloads, { recursive: true });

  // The graph as the fixture has it, with the one edit made below.
  const expected = readJson<GraphDocument>(fixture('tensor-chain.json'));
  expected.nodes.find((n) => n.id === 'double')!.data!.params!.scalar = 2.5;
  const expectedShape = documentShape(expected);

  await openApp(page);
  await importFile(page, fixture('tensor-chain.json'));
  await expect(canvasNodes(page)).toHaveCount(5);
  await expect(canvasEdges(page)).toHaveCount(4);

  // Edit: ScalarMultiply's scalar 2 -> 2.5, in the node's settings panel.
  await selectNode(page, 'double');
  const scalar = page.getByRole('spinbutton');
  await expect(scalar).toHaveCount(1);
  await scalar.fill('2.5');
  await scalar.press('Enter');
  await expect(node(page, 'double')).toContainText('2.5');

  // Delete the node with the Delete key: the node and both its wires go.
  await selectNode(page, 'double');
  await page.keyboard.press('Delete');
  await expect(node(page, 'double')).toHaveCount(0);
  await expect(canvasEdges(page)).toHaveCount(2);

  // ONE undo brings back the node AND its wires, and keeps the edit.
  await page.keyboard.press('ControlOrMeta+z');
  await expect(node(page, 'double')).toHaveCount(1);
  await expect(edge(page, 'd1')).toHaveCount(1);
  await expect(edge(page, 'd2')).toHaveCount(1);
  await expect(canvasEdges(page)).toHaveCount(4);
  await expect(node(page, 'double')).toContainText('2.5');

  // Save: the server writes the graph into the suite's own folder.
  await page.getByRole('button', { name: 'Save', exact: true }).click();
  await answerPrompt(page, 'Enter a name for this graph:', GRAPH_NAME);
  await expect(page.getByText(`Graph "${GRAPH_NAME}" saved successfully.`)).toBeVisible();
  await expect(workspaceTabs(page).first()).toHaveAccessibleName(GRAPH_NAME);

  const savedFile = savedGraphFile(GRAPH_NAME);
  expect(path.relative(E2E_DIR, savedFile).startsWith('..')).toBe(false);
  const savedBytes = readFileSync(savedFile, 'utf-8');
  const saved = JSON.parse(savedBytes) as GraphDocument;
  expect(documentShape(saved)).toEqual(expectedShape);

  // Export as JSON: the same document.
  expect(documentShape(await exportGraphJson(page, downloads))).toEqual(expectedShape);

  // Export the workspace (every tab, one file).
  const workspaceDownload = await exportDownload(page, 'Workspace (.cduiworkspace)');
  const workspaceFile = path.join(downloads, workspaceDownload.suggestedFilename());
  await workspaceDownload.saveAs(workspaceFile);
  expect(workspaceFile.endsWith('.cduiworkspace')).toBe(true);

  // Import a graph into the occupied workspace: it gets a tab of its own.
  await importFile(page, fixture('two-blocks.json'));
  await expect(workspaceTabs(page)).toHaveCount(2);
  await expect(workspaceTabs(page).nth(1)).toHaveAttribute('aria-selected', 'true');
  await expect(node(page, 'blkA')).toHaveCount(1);

  // Import the workspace into it as well: its tab is added, nothing replaced.
  await importFile(page, workspaceFile);
  await expect(page.getByText('Imported 1 tab(s).')).toBeVisible();
  await expect(workspaceTabs(page)).toHaveCount(3);

  const checkTab = async (index: number) => {
    await workspaceTabs(page).nth(index).click();
    await expect(workspaceTabs(page).nth(index)).toHaveAttribute('aria-selected', 'true');
    await expect(canvasNodes(page)).toHaveCount(5);
    await expect(canvasEdges(page)).toHaveCount(4);
    expect(documentShape(await exportGraphJson(page, downloads))).toEqual(expectedShape);
  };
  await checkTab(0); // the original
  await checkTab(2); // the copy the workspace brought in

  // Close and reopen the page: every tab comes back, the original intact.
  await waitForAutosave(page, 3);
  await page.reload();
  await expect(workspaceTabs(page)).toHaveCount(3);
  await expect(workspaceTabs(page).first()).toHaveAccessibleName(GRAPH_NAME);
  await checkTab(0);
  await workspaceTabs(page).nth(1).click();
  await expect(node(page, 'blkA')).toHaveCount(1);
  await expect(node(page, 'blkB')).toHaveCount(1);

  // Nothing above wrote to the saved document again.
  expect(readFileSync(savedFile, 'utf-8')).toBe(savedBytes);
});
