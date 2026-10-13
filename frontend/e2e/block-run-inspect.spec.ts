/**
 * Golden flow 2 (#642): block execution and inspection.
 *
 * A CPU-only tensor graph: a 2x2 tensor of 1.5 -> x2 -> mean -> Print. The
 * middle two nodes are collapsed into a block, Run is pressed from inside
 * that block (#638), and an inner node is inspected (#633): the engine runs
 * it as `<block card>/<inner id>`, and the Inspector has to ask for that id.
 * Then the same for two instances of one block, each fed its own tensor, so
 * reading the wrong instance shows the wrong number.
 *
 * #559: while a block is open, each node inside it shows its own run status
 * (completed, error, bypassed), the card keeps the rolled-up one, and a plot
 * drawn inside a block reaches the Execution Log and the Inspector.
 */
import type { Page } from '@playwright/test';
import {
  answerPrompt,
  canvasNodes,
  expect,
  fixture,
  importFile,
  node,
  openApp,
  printedLine,
  runGraph,
  selectNode,
  test,
} from './app';

/** Wait for the Inspector's request for one port of one run-time node id. */
function capturedValue(page: Page, runNodeId: string, port: string) {
  return page.waitForResponse((response) => {
    const url = new URL(response.url());
    return (
      /\/api\/execution\/outputs\/[^/]+\/value$/.test(url.pathname) &&
      url.searchParams.get('node_id') === runNodeId &&
      url.searchParams.get('port') === port
    );
  }, { timeout: 15_000 });
}

async function enterBlock(page: Page, cardId: string): Promise<void> {
  await selectNode(page, cardId, { button: 'right' });
  await page.getByRole('button', { name: 'Enter subgraph' }).click();
  await expect(page.getByRole('button', { name: 'Back', exact: true })).toBeVisible();
}

async function leaveBlock(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'Back', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Back', exact: true })).toBeHidden();
}

/** The status line a node's card shows (the footer, or the BYPASS badge). */
function statusOf(page: Page, id: string, text: string | RegExp) {
  return node(page, id).getByText(text);
}

/** Press Run and wait for the run to fail. */
async function runGraphToError(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'Run', exact: false }).first().click();
  await expect(page.getByText(/^Execution error: /)).toHaveCount(1, { timeout: 60_000 });
}

/** Select an inner node and check what the Inspector read for it. */
async function inspectDouble(page: Page, runNodeId: string, value: number): Promise<void> {
  const response = capturedValue(page, runNodeId, 'tensor');
  await selectNode(page, 'double');
  expect((await response).status()).toBe(200);
  await expect(page.getByText(`min ${value} · max ${value} · mean ${value}`)).toBeVisible();
}

test('Run from inside a collapsed block runs the whole graph, and the Inspector reads the inner node', async ({
  page,
}) => {
  await openApp(page);
  await importFile(page, fixture('tensor-chain.json'));
  await expect(canvasNodes(page)).toHaveCount(5);

  // Collapse ScalarMultiply and Mean into a block called Blk.
  await selectNode(page, 'double');
  await selectNode(page, 'avg', { add: true });
  await selectNode(page, 'avg', { button: 'right' });
  await page.getByRole('button', { name: 'Collapse to subgraph' }).click();
  await answerPrompt(page, 'Name this subgraph', 'Blk');
  await expect(node(page, 'double')).toHaveCount(0);
  const card = canvasNodes(page).filter({ hasText: 'Blk' });
  await expect(card).toHaveCount(1);
  const cardId = (await card.getAttribute('data-id'))!;
  expect(cardId).toBeTruthy();

  // Run from inside the block: the whole graph runs, Print included.
  await enterBlock(page, cardId);
  await expect(canvasNodes(page)).toHaveCount(2);
  await runGraph(page);
  expect(await printedLine(page, 'e2e_result')).toBe('[e2e_result] 3.0');
  await expect(page.getByText('Node Print completed')).toBeVisible();

  // The nodes on screen are the ones inside the block, and each shows what
  // the run did to it (#559) -- not the card's roll-up, and not nothing.
  await expect(statusOf(page, 'double', 'Completed')).toBeVisible();
  await expect(statusOf(page, 'avg', 'Completed')).toBeVisible();

  // The inner ScalarMultiply ran as `<card>/double`: 1.5 * 2 = 3 everywhere.
  await inspectDouble(page, `${cardId}/double`, 3);

  // Back at the top level the card keeps its own, rolled-up status.
  await leaveBlock(page);
  await expect(statusOf(page, cardId, 'Completed')).toBeVisible();
});

test('two instances of one block each show their own inner values', async ({ page }) => {
  await openApp(page);
  await importFile(page, fixture('two-blocks.json'));
  await expect(canvasNodes(page)).toHaveCount(7);

  await runGraph(page);
  expect(await printedLine(page, 'e2e_a')).toBe('[e2e_a] 3.0');
  expect(await printedLine(page, 'e2e_b')).toBe('[e2e_b] 8.0');

  // B first, then A, so neither answer can be a leftover of the other.
  await enterBlock(page, 'blkB');
  await expect(statusOf(page, 'double', 'Completed')).toBeVisible();
  await inspectDouble(page, 'blkB/double', 8);
  await leaveBlock(page);

  await enterBlock(page, 'blkA');
  await inspectDouble(page, 'blkA/double', 3);
  await leaveBlock(page);
});

test('an open block shows the node that failed inside it, and the card the failure', async ({ page }) => {
  await openApp(page);
  await importFile(page, fixture('block-fails.json'));
  await expect(canvasNodes(page)).toHaveCount(4);

  // Run from inside the block: its Mean asks for a dimension a 1-D tensor
  // does not have, so the run stops there.
  await enterBlock(page, 'blkE');
  await runGraphToError(page);
  await expect(statusOf(page, 'double', 'Completed')).toBeVisible();
  await expect(statusOf(page, 'avg', /^Error: Dimension out of range/)).toBeVisible();

  // The Inspector says the node failed, and asks for nothing it could not have.
  await selectNode(page, 'avg');
  await expect(page.getByText('Failed in the last run').first()).toBeVisible();

  await leaveBlock(page);
  await expect(statusOf(page, 'blkE', /^Error: Dimension out of range/)).toBeVisible();
});

test('a plot drawn inside a block reaches the log and the Inspector, past a bypassed node', async ({
  page,
}) => {
  await openApp(page);
  await importFile(page, fixture('block-plot.json'));
  await expect(canvasNodes(page)).toHaveCount(3);

  await runGraph(page);
  // The picture the inner Visualize drew, filed under the card and naming
  // the node inside it.
  await expect(page.getByText('blkV › plot')).toBeVisible();
  await expect(page.getByAltText('Node output image')).toHaveCount(1);

  await enterBlock(page, 'blkV');
  await expect(statusOf(page, 'plot', 'Completed')).toBeVisible();
  await expect(statusOf(page, 'muted', 'BYPASS')).toBeVisible();

  // The bypassed node did not run: what passed through it is the tensor the
  // block was fed, read by the id the run gave it.
  const passed = capturedValue(page, 'blkV/muted', 'tensor');
  await selectNode(page, 'muted');
  expect((await passed).status()).toBe(200);
  await expect(page.getByText('min 1.5000 · max 1.5000 · mean 1.5000')).toBeVisible();

  // The plot's own row shows the picture, with no request for its base64.
  await selectNode(page, 'plot');
  await expect(page.getByAltText('image', { exact: true })).toBeVisible();
});
