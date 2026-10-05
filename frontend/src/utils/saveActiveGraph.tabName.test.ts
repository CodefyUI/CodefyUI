/**
 * A save renames the tab to the name the graph was saved under.
 *
 * Save As used to write the file and bind the tab to it while the tab went on
 * saying what it said before: a starter imported as "CF2D01" and saved as
 * "CF2A01" still read "CF2D01" in the tab strip -- the name of the file the
 * student must NOT hand in.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('../api/rest', async (importOriginal) => ({
  saveGraph: vi.fn().mockResolvedValue({}),
  listGraphs: vi.fn().mockResolvedValue([]),
  // The real class: the save tells a taken name from a failure with
  // `instanceof`, so a stub would turn the overwrite question into an error.
  GraphExistsError: (await importOriginal<typeof import('../api/rest')>()).GraphExistsError,
}));
vi.mock('./dialog', () => ({
  prompt: vi.fn(),
  confirm: vi.fn().mockResolvedValue(true),
}));

import { saveActiveGraph } from './saveActiveGraph';
import { saveGraph, GraphExistsError } from '../api/rest';
import { confirm, prompt } from './dialog';
import { useTabStore } from '../store/tabStore';
import { useProjectStore } from '../store/projectStore';

const store = () => useTabStore.getState();
const tabOf = (id: string) => store().tabs.find((tb) => tb.id === id)!;

/** One open tab with this label, bound to no saved graph -- an imported starter. */
function openTab(label: string): string {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  store().addTab(label);
  return store().activeTabId;
}

/** The name the user types at the save's name prompt. */
function typeName(name: string | null) {
  vi.mocked(prompt).mockResolvedValue(name);
}

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  useProjectStore.setState({ projectDir: null, projectName: null, loaded: true });
});

describe('saveActiveGraph renames the tab', () => {
  it('Save As "CF2A01" of a tab labelled "CF2D01" relabels it "CF2A01"', async () => {
    const id = openTab('CF2D01');
    typeName('CF2A01');
    vi.mocked(saveGraph).mockResolvedValueOnce({ file: 'CF2A01' } as never);

    await saveActiveGraph({ saveAs: true });

    expect(tabOf(id).name).toBe('CF2A01');
    expect(tabOf(id).currentGraphFile).toBe('CF2A01');
  });

  it('Save As of a tab already bound to the starter moves the label with the binding', async () => {
    const id = openTab('CF2D01');
    store().setCurrentGraphFile('CF2D01', 'CF2D01');
    typeName('CF2A01');
    vi.mocked(saveGraph).mockResolvedValueOnce({ file: 'CF2A01' } as never);

    await saveActiveGraph({ saveAs: true });

    expect(tabOf(id).name).toBe('CF2A01');
    expect(tabOf(id).currentGraphName).toBe('CF2A01');
  });

  it('the first Save of a new canvas names its tab', async () => {
    const id = openTab('Tab 1');
    typeName('CF2A01');

    await saveActiveGraph();

    expect(tabOf(id).name).toBe('CF2A01');
  });

  // The label is the title as typed; only the file stem is sanitized.
  it('labels the tab with the title, not the stem it was saved as', async () => {
    const id = openTab('Tab 1');
    typeName('期中考 第一題');
    vi.mocked(saveGraph).mockResolvedValueOnce({ file: '期中考_第一題' } as never);

    await saveActiveGraph({ saveAs: true });

    expect(tabOf(id).name).toBe('期中考 第一題');
    expect(tabOf(id).currentGraphFile).toBe('期中考_第一題');
  });

  // A save that writes back to the graph the tab is bound to names nothing
  // new, so a label the user typed into the tab strip stays.
  it('a one-click Save of a bound tab keeps the label the user typed', async () => {
    const id = openTab('my label');
    store().setCurrentGraphFile('My_Graph', 'My Graph');

    await saveActiveGraph();

    expect(prompt).not.toHaveBeenCalled();
    expect(saveGraph).toHaveBeenCalledTimes(1);
    expect(tabOf(id).name).toBe('my label');
  });

  // A tab restored from a 2.8.0 record knows its file but not its title, and
  // is asked for the title once. That save still writes back in place.
  it('the one-time title question of a bound tab keeps the label too', async () => {
    const id = openTab('my label');
    store().setCurrentGraphFile('My_Graph', null);
    typeName('My Graph');

    await saveActiveGraph();

    expect(prompt).toHaveBeenCalledTimes(1);
    expect(tabOf(id).currentGraphName).toBe('My Graph');
    expect(tabOf(id).name).toBe('my label');
  });

  it('renames the tab the save started from when another tab came to the front meanwhile', async () => {
    const alpha = openTab('CF2D01');
    store().addTab('Other');
    const beta = store().activeTabId;
    store().setActiveTab(alpha);
    vi.mocked(prompt).mockImplementation(async () => {
      store().setActiveTab(beta);
      return 'CF2A01';
    });

    await saveActiveGraph({ saveAs: true });

    expect(tabOf(alpha).name).toBe('CF2A01');
    expect(tabOf(beta).name).toBe('Other');
  });
});

/**
 * The name question opens filled in, and selected, so Enter keeps the name
 * and typing replaces it. It used to open empty, on Save As too, where the
 * graph already has a name the user can see.
 */
describe('saveActiveGraph offers a name', () => {
  /** What the save's name question opened with. */
  const offered = () => vi.mocked(prompt).mock.calls[0][0].defaultValue;

  it("Save As of a saved graph offers the graph's title", async () => {
    openTab('my label');
    store().setCurrentGraphFile('My_Graph', 'My Graph');
    typeName('My Graph copy');

    await saveActiveGraph({ saveAs: true });

    expect(offered()).toBe('My Graph');
  });

  it("the first Save of a tab bound to nothing offers the tab's name", async () => {
    openTab('CF2D01');
    typeName('CF2A01');

    await saveActiveGraph();

    expect(offered()).toBe('CF2D01');
  });

  // A tab restored from a 2.8.0 record has a file but no title; its label
  // came from the graph when the graph was opened.
  it("the one-time title question of a 2.8.0 tab offers the tab's name", async () => {
    openTab('My Graph');
    store().setCurrentGraphFile('My_Graph', null);
    typeName('My Graph');

    await saveActiveGraph();

    expect(offered()).toBe('My Graph');
  });
});

describe('saveActiveGraph leaves the label alone when nothing was saved', () => {
  it('a save the server refused', async () => {
    const id = openTab('CF2D01');
    typeName('CF2A01');
    vi.mocked(saveGraph).mockRejectedValueOnce(new Error('disk full'));

    await saveActiveGraph({ saveAs: true });

    expect(tabOf(id).name).toBe('CF2D01');
  });

  it('a cancelled name prompt', async () => {
    const id = openTab('CF2D01');
    typeName(null);

    await saveActiveGraph({ saveAs: true });

    expect(saveGraph).not.toHaveBeenCalled();
    expect(tabOf(id).name).toBe('CF2D01');
  });

  it('a declined overwrite', async () => {
    const id = openTab('CF2D01');
    typeName('Taken');
    vi.mocked(saveGraph).mockRejectedValueOnce(new GraphExistsError('Taken', 'Taken'));
    vi.mocked(confirm).mockResolvedValueOnce(false);

    await saveActiveGraph({ saveAs: true });

    expect(tabOf(id).name).toBe('CF2D01');
  });
});
