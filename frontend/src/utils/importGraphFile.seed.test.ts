/**
 * Every reader of a graph document carries `settings.seed` (2.8.9), the way
 * each already carries `settings.device`: Import (into the empty tab it fills
 * or the new tab it opens), a saved graph opened from the Graphs panel, and
 * an example opened from the gallery. Merging an example INTO the canvas does
 * not, as it does not take the example's device either.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { importGraphFile } from './importGraphFile';
import { resolveSavedGraph } from './openSavedGraph';
import {
  insertExample,
  openExample,
  openExampleInNewTab,
  resolveExample,
  resolveUnboundDocument,
} from './openExample';
import { useNodeDefStore } from '../store/nodeDefStore';
import { useTabStore } from '../store/tabStore';
import { useToastStore } from '../store/toastStore';
import { useI18n } from '../i18n';
import * as rest from '../api/rest';

// Only the example fetch is stubbed; resolution and the store writes run for
// real, which is what these tests are about.
vi.mock('../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/rest')>();
  return { ...actual, loadExample: vi.fn() };
});

const mockedRest = vi.mocked(rest);
const store = () => useTabStore.getState();
const tabs = () => useTabStore.getState().tabs;
const activeTab = () => useTabStore.getState().getActiveTab();

/** A serialized node, in the shape `Export -> JSON` writes. */
function raw(id: string) {
  return { id, type: 'Add', position: { x: 0, y: 0 }, data: { params: {} } };
}

function jsonFile(body: unknown) {
  return new File([JSON.stringify(body)], 'g.json', { type: 'application/json' });
}

/** Give the active tab a graph, so an Import opens a tab of its own. */
function holdWork() {
  store().setNodes([raw('mine')] as never);
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useNodeDefStore.setState({ definitions: [], presets: [] });
  useToastStore.setState({ toasts: [] });
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  store().addTab('Tab 1');
  mockedRest.loadExample.mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('Import', () => {
  it('installs the file seed into the empty tab it fills', async () => {
    await importGraphFile(jsonFile({ nodes: [raw('n1')], edges: [], settings: { seed: 11 } }));
    expect(tabs()).toHaveLength(1);
    expect(tabs()[0].seed).toBe(11);
  });

  it('installs the file seed into the new tab it opens beside work', async () => {
    holdWork();
    store().setSeed(3);
    await importGraphFile(jsonFile({ nodes: [raw('n1')], edges: [], settings: { seed: 11 } }));
    expect(tabs()).toHaveLength(2);
    expect(tabs()[1].seed).toBe(11);
    // The tab that held the work keeps its own.
    expect(tabs()[0].seed).toBe(3);
  });

  it('opens a seedless file beside work with the seed that tab had', async () => {
    // The UAT case: set the seed, then Import a starter that predates 2.8.9.
    holdWork();
    store().setSeed(3);
    await importGraphFile(jsonFile({ nodes: [raw('n1')], edges: [] }));
    expect(tabs()).toHaveLength(2);
    expect(store().activeTabId).toBe(tabs()[1].id);
    expect(tabs()[1].seed).toBe(3);
  });

  it('keeps the tab seed for a file whose seed is not one', async () => {
    store().setSeed(3);
    await importGraphFile(jsonFile({ nodes: [], edges: [], settings: { seed: '7' } }));
    expect(tabs()[0].seed).toBe(3);
  });

  it('installs the seed and the device from one settings block', async () => {
    await importGraphFile(
      jsonFile({ nodes: [], edges: [], settings: { device: 'mps', seed: 0 } }),
    );
    expect(tabs()[0].seed).toBe(0);
    expect(tabs()[0].graphDevice).toBe('mps');
  });
});

describe('a saved graph', () => {
  it('resolveSavedGraph carries settings.seed', () => {
    expect(resolveSavedGraph({ nodes: [], edges: [], settings: { seed: 42 } }, 'g').seed).toBe(42);
    expect(resolveSavedGraph({ nodes: [], edges: [], settings: { seed: 0 } }, 'g').seed).toBe(0);
  });

  it('resolveSavedGraph answers null for a file with no usable seed', () => {
    expect(resolveSavedGraph({ nodes: [], edges: [] }, 'g').seed).toBeNull();
    expect(resolveSavedGraph({ nodes: [], edges: [], settings: { seed: -4 } }, 'g').seed)
      .toBeNull();
  });

  it('installs the saved seed when the document is opened', () => {
    store().setSeed(9);
    store().loadGraphDocument(
      resolveSavedGraph({ nodes: [raw('a')], edges: [], settings: { seed: 42 } }, 'g'),
    );
    expect(activeTab().seed).toBe(42);
  });
});

describe('an example', () => {
  it('resolveExample and resolveUnboundDocument carry settings.seed', () => {
    const data = { nodes: [], edges: [], settings: { seed: 5 } };
    expect(resolveExample(data).seed).toBe(5);
    expect(resolveUnboundDocument(data).seed).toBe(5);
    expect(resolveExample({ nodes: [], edges: [] }).seed).toBeNull();
  });

  it('openExample installs the example seed', async () => {
    store().setSeed(9);
    mockedRest.loadExample.mockResolvedValue({
      nodes: [raw('a')], edges: [], settings: { seed: 5 },
    });
    await expect(openExample('x')).resolves.toBe(true);
    expect(activeTab().seed).toBe(5);
  });

  it('openExample keeps the tab seed for an example that carries none', async () => {
    store().setSeed(9);
    mockedRest.loadExample.mockResolvedValue({ nodes: [raw('a')], edges: [] });
    await openExample('x');
    expect(activeTab().seed).toBe(9);
  });

  it('openExampleInNewTab gives the new tab the example seed, else the seed it started with', async () => {
    store().setSeed(9);
    mockedRest.loadExample.mockResolvedValue({
      nodes: [raw('a')], edges: [], settings: { seed: 5 },
    });
    await openExampleInNewTab('x');
    expect(activeTab().seed).toBe(5);

    mockedRest.loadExample.mockResolvedValue({ nodes: [raw('a')], edges: [] });
    await openExampleInNewTab('y');
    expect(tabs()).toHaveLength(3);
    // Started with the seed of the tab that was active (the seed-5 one).
    expect(activeTab().seed).toBe(5);
  });

  it('insertExample leaves the tab seed alone: the merged nodes join the graph that owns it', async () => {
    store().setSeed(9);
    mockedRest.loadExample.mockResolvedValue({
      nodes: [raw('a')], edges: [], settings: { seed: 5 },
    });
    await expect(insertExample('x')).resolves.toBe(true);
    expect(activeTab().seed).toBe(9);
  });
});
