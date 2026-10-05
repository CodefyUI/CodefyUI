/**
 * An import into the empty current tab names the tab the way a new tab would
 * be named: after the graph, else after the file.
 *
 * It used to keep the label it had, so an exam starter imported into the
 * first tab sat under "Tab 1", and Export Python then wrote `Tab_1.py`.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { importFile, importGraphFile } from './importGraphFile';
import { useNodeDefStore } from '../store/nodeDefStore';
import { useTabStore } from '../store/tabStore';
import { useToastStore } from '../store/toastStore';
import { useI18n } from '../i18n';

const tabs = () => useTabStore.getState().tabs;

/** A picked graph file called `fileName`, holding one node and `extra`. */
function graphFile(fileName: string, extra: Record<string, unknown> = {}) {
  const body = JSON.stringify({
    nodes: [{ id: 'n1', type: 'Add', position: { x: 0, y: 0 }, data: { params: {} } }],
    edges: [],
    ...extra,
  });
  return new File([body], fileName, { type: 'application/json' });
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useNodeDefStore.setState({ definitions: [], presets: [] });
  useToastStore.setState({ toasts: [] });
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('Tab 1');
});

describe.each([
  ['importFile', importFile],
  ['importGraphFile', importGraphFile],
])('%s into the empty current tab', (_door, importer) => {
  it('names the tab after the graph', async () => {
    const [empty] = tabs();

    expect(await importer(graphFile('starter.json', { name: 'CF2D01' }))).toBe(true);

    expect(tabs().map((t) => t.id)).toEqual([empty.id]);
    expect(tabs()[0].name).toBe('CF2D01');
  });

  it('names the tab after the file when the graph has no name', async () => {
    await importer(graphFile('CF2D01.json'));

    expect(tabs()[0].name).toBe('CF2D01');
  });

  it('keeps the label when neither the graph nor the file gives a name', async () => {
    await importer(graphFile('.json', { name: '   ' }));

    expect(tabs()[0].name).toBe('Tab 1');
  });
});

it('gives a filled tab the same name a new tab gets for the same file', async () => {
  await importFile(graphFile('starter.json', { name: 'CF2D01' }));
  // The first tab now holds a graph, so the same file opens a tab of its own.
  await importFile(graphFile('starter.json', { name: 'CF2D01' }));

  expect(tabs().map((t) => t.name)).toEqual(['CF2D01', 'CF2D01']);
});
