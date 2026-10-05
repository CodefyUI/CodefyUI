/**
 * A rename in the Graphs panel relabels every tab holding the graph.
 *
 * `rebindGraphFile` moved the binding and the stored title of every holder and
 * left the tab's label alone, so "Exam" renamed to "Exam v2" listed, saved and
 * exported as Exam v2 while its tab still said "Exam". A rename is an explicit
 * new name for the graph, like Save As, which also replaces the label -- so it
 * replaces a label the user typed too. A delete, or any rebind to nothing,
 * keeps the label: the graph on screen is unchanged, it has only lost its file.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { useTabStore } from './tabStore';

const store = () => useTabStore.getState();

/** Open a tab labelled `label`, bound to `file` (or to nothing), and return its id. */
function openTab(label: string, file: string | null): string {
  store().addTab(label);
  store().setCurrentGraphFile(file, file);
  return store().activeTabId;
}

const tabOf = (id: string) => store().tabs.find((tab) => tab.id === id)!;

let holder: string;
let typedLabel: string;
let unbound: string;
let other: string;

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  holder = openTab('alpha', 'alpha');
  typedLabel = openTab('my label', 'alpha');
  unbound = openTab('Tab 3', null);
  other = openTab('beta', 'beta');
});

describe('rebindGraphFile relabels on a rename', () => {
  it('names every tab holding the renamed graph after its new title', () => {
    store().rebindGraphFile('alpha', { file: 'Renamed_Alpha', name: 'Renamed Alpha' });

    expect(tabOf(holder).name).toBe('Renamed Alpha');
    expect(tabOf(typedLabel).name).toBe('Renamed Alpha');
    expect(tabOf(holder).currentGraphFile).toBe('Renamed_Alpha');
    expect(tabOf(typedLabel).currentGraphName).toBe('Renamed Alpha');
  });

  it('leaves tabs that do not hold the graph alone', () => {
    store().rebindGraphFile('alpha', { file: 'Renamed_Alpha', name: 'Renamed Alpha' });

    expect(tabOf(unbound).name).toBe('Tab 3');
    expect(tabOf(other).name).toBe('beta');
    expect(tabOf(other).currentGraphFile).toBe('beta');
  });

  it('keeps the label when the graph is unbound, as a delete does', () => {
    store().rebindGraphFile('alpha', null);

    expect(tabOf(holder).name).toBe('alpha');
    expect(tabOf(typedLabel).name).toBe('my label');
    expect(tabOf(holder).currentGraphFile).toBeNull();
  });

  // The type lets a rebind carry a file with no name; a label is never
  // emptied by one.
  it.each([
    ['no name', null],
    ['an empty name', ''],
  ])('keeps the label when the new binding carries %s', (_case, name) => {
    store().rebindGraphFile('alpha', { file: 'Renamed_Alpha', name });

    expect(tabOf(holder).name).toBe('alpha');
    expect(tabOf(typedLabel).name).toBe('my label');
  });

  it('spares the excepted tab its label as well as its binding', () => {
    store().rebindGraphFile('alpha', { file: 'Renamed_Alpha', name: 'Renamed Alpha' }, typedLabel);

    expect(tabOf(typedLabel).name).toBe('my label');
    expect(tabOf(typedLabel).currentGraphFile).toBe('alpha');
    expect(tabOf(holder).name).toBe('Renamed Alpha');
  });
});
