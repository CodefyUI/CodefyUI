/**
 * Golden flow 2 (#642): block execution and inspection.
 *
 * A CPU-only tensor graph: a 2x2 tensor of 1.5 -> x2 -> mean -> Print. The
 * middle two nodes are collapsed into a block, Run is pressed from inside
 * that block (#638), and an inner node is inspected (#633): the engine runs
 * it as `<block card>/<inner id>`, and the Inspector has to ask for that id.
 * Then the same for two instances of one block, each fed its own tensor, so
 * reading the wrong instance shows the wrong number.
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

  // The inner ScalarMultiply ran as `<card>/double`: 1.5 * 2 = 3 everywhere.
  await inspectDouble(page, `${cardId}/double`, 3);
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
  await inspectDouble(page, 'blkB/double', 8);
  await leaveBlock(page);

  await enterBlock(page, 'blkA');
  await inspectDouble(page, 'blkA/double', 3);
  await leaveBlock(page);
});
