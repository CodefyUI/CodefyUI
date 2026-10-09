import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { insertExample, openExample, openExampleInNewTab } from './openExample';
import { useNodeDefStore } from '../store/nodeDefStore';
import { useTabStore } from '../store/tabStore';
import { useToastStore } from '../store/toastStore';
import { useUIStore } from '../store/uiStore';
import { useI18n } from '../i18n';
import * as rest from '../api/rest';

/**
 * An example being opened is opened once (#622).
 *
 * A double-click is two clicks, and both reached the gallery while the
 * example was still loading: the empty-canvas gallery filled its tab and then
 * opened the same example again in a new one, and the welcome screen opened
 * two new tabs. Only the network call is stubbed, as in `openExample.test.ts`.
 */
vi.mock('../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/rest')>();
  return { ...actual, loadExample: vi.fn() };
});

const mockedRest = vi.mocked(rest);
const tabs = () => useTabStore.getState().tabs;

/** The example `/api/examples/load` hands back, once `arrive` is called. */
function holdTheLoad(): () => Promise<void> {
  let arrive!: (payload: unknown) => void;
  mockedRest.loadExample.mockReturnValue(
    new Promise((resolve) => {
      arrive = resolve;
    }),
  );
  return async () => {
    arrive({
      name: 'Twice',
      nodes: [{ id: 'a', type: 'Dropout', position: { x: 0, y: 0 }, data: { params: {} } }],
      edges: [],
    });
    // Let both callers' continuations run.
    await new Promise((resolve) => setTimeout(resolve, 0));
  };
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useNodeDefStore.setState({ definitions: [], presets: [] });
  useToastStore.setState({ toasts: [] });
  useUIStore.setState({ layoutFitRequests: {} });
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('Tab 1');
  mockedRest.loadExample.mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('an example already on its way (#622)', () => {
  it('fills the empty tab once, with no second tab, when picked twice', async () => {
    const arrive = holdTheLoad();
    const first = openExample('Usage_Example/Twice');
    const second = openExample('Usage_Example/Twice');
    await arrive();

    await expect(first).resolves.toBe(true);
    await expect(second).resolves.toBe(true);
    expect(mockedRest.loadExample).toHaveBeenCalledTimes(1);
    expect(tabs()).toHaveLength(1);
    expect(tabs()[0].nodes.map((n) => n.id)).toEqual(['a']);
  });

  it('opens one new tab, not two, from the welcome screen', async () => {
    const arrive = holdTheLoad();
    const first = openExampleInNewTab('Usage_Example/Twice');
    const second = openExampleInNewTab('Usage_Example/Twice');
    await arrive();

    await Promise.all([first, second]);
    expect(mockedRest.loadExample).toHaveBeenCalledTimes(1);
    expect(tabs()).toHaveLength(2);
  });

  it('opens it again once the first open has landed', async () => {
    let arrive = holdTheLoad();
    const first = openExampleInNewTab('Usage_Example/Twice');
    await arrive();
    await first;

    arrive = holdTheLoad();
    const again = openExampleInNewTab('Usage_Example/Twice');
    await arrive();
    await again;
    expect(mockedRest.loadExample).toHaveBeenCalledTimes(2);
    expect(tabs()).toHaveLength(3);
  });

  it('lets an example whose open failed be picked again', async () => {
    mockedRest.loadExample.mockRejectedValueOnce(new Error('offline'));
    await expect(openExample('Usage_Example/Twice')).resolves.toBe(false);

    const arrive = holdTheLoad();
    const retry = openExample('Usage_Example/Twice');
    await arrive();
    await expect(retry).resolves.toBe(true);
    expect(mockedRest.loadExample).toHaveBeenCalledTimes(2);
    expect(tabs()[0].nodes.map((n) => n.id)).toEqual(['a']);
  });

  it('opens two different examples both', async () => {
    const arrive = holdTheLoad();
    const one = openExampleInNewTab('Usage_Example/One');
    const other = openExampleInNewTab('Usage_Example/Other');
    await arrive();

    await Promise.all([one, other]);
    expect(mockedRest.loadExample).toHaveBeenCalledTimes(2);
    expect(tabs()).toHaveLength(3);
  });
});

describe('a template already on its way (#622)', () => {
  const nodesOnCanvas = () => tabs()[0].nodes;

  it('is inserted once when its item in the Templates tab is clicked twice', async () => {
    const arrive = holdTheLoad();
    const first = insertExample('Usage_Example/Twice');
    const second = insertExample('Usage_Example/Twice');
    await arrive();

    await expect(first).resolves.toBe(true);
    await expect(second).resolves.toBe(true);
    expect(mockedRest.loadExample).toHaveBeenCalledTimes(1);
    expect(nodesOnCanvas()).toHaveLength(1);
  });

  it('is inserted again once the first insert has landed', async () => {
    let arrive = holdTheLoad();
    const first = insertExample('Usage_Example/Twice');
    await arrive();
    await first;

    arrive = holdTheLoad();
    const again = insertExample('Usage_Example/Twice');
    await arrive();
    await again;
    expect(nodesOnCanvas()).toHaveLength(2);
  });

  it('is inserted at each of two drops: a drop is one gesture at one point', async () => {
    const arrive = holdTheLoad();
    const one = insertExample('Usage_Example/Twice', { x: 0, y: 0 });
    const other = insertExample('Usage_Example/Twice', { x: 600, y: 0 });
    await arrive();

    await Promise.all([one, other]);
    expect(mockedRest.loadExample).toHaveBeenCalledTimes(2);
    expect(nodesOnCanvas()).toHaveLength(2);
  });

  it('is inserted while the same example is being opened, not opened instead', async () => {
    const arrive = holdTheLoad();
    const open = openExampleInNewTab('Usage_Example/Twice');
    const insert = insertExample('Usage_Example/Twice');
    await arrive();

    await Promise.all([open, insert]);
    expect(mockedRest.loadExample).toHaveBeenCalledTimes(2);
    expect(tabs()).toHaveLength(2);
  });
});
