import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { importGraphFile } from './importGraphFile';
import { useNodeDefStore } from '../store/nodeDefStore';
import { useTabStore } from '../store/tabStore';
import { useToastStore } from '../store/toastStore';
import { useI18n } from '../i18n';

const tabs = () => useTabStore.getState().tabs;
const toasts = () => useToastStore.getState().toasts;
const errorToasts = () => toasts().filter((t) => t.type === 'error');

/** A picked file holding exactly `body`. */
function jsonFile(body: string) {
  return new File([body], 'g.json', { type: 'application/json' });
}

/** A serialized node, in the shape `Export -> JSON` writes. */
function raw(id: string) {
  return { id, type: 'Add', position: { x: 0, y: 0 }, data: { params: {} } };
}

/**
 * A reader that fails the way an unreadable file makes the real one fail:
 * `error` fires and `load` never does. jsdom will happily read anything
 * handed to `new File()`, so the only way to reach that branch is to stand
 * in for the reader itself.
 */
function installFailingReader(readerError: { message: string } | null) {
  class FailingFileReader {
    error = readerError;
    onerror: (() => void) | null = null;
    onload: (() => void) | null = null;
    readAsText() {
      this.onerror?.();
    }
  }
  vi.stubGlobal('FileReader', FailingFileReader as unknown as typeof FileReader);
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useNodeDefStore.setState({ definitions: [], presets: [] });
  useToastStore.setState({ toasts: [] });
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('Tab 1');
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('importGraphFile', () => {
  it('installs the file onto the canvas', async () => {
    const body = JSON.stringify({
      nodes: [raw('n1')],
      edges: [],
      description: 'an imported graph',
      segmentGroups: [{ id: 's1' }],
    });
    expect(await importGraphFile(jsonFile(body))).toBe(true);
    expect(tabs()[0].nodes.map((n) => n.id)).toEqual(['n1']);
    expect(tabs()[0].description).toBe('an imported graph');
    expect(tabs()[0].segmentGroups).toHaveLength(1);
  });

  it('unbinds the tab, so the import cannot be saved over the file that was open', async () => {
    useTabStore.getState().setCurrentGraphFile('bound-graph', 'bound-graph');
    await importGraphFile(jsonFile(JSON.stringify({ nodes: [], edges: [] })));
    expect(tabs()[0].currentGraphFile).toBeNull();
  });

  it('merges presets the running server has never seen, keeping the ones it has', async () => {
    useNodeDefStore.setState({ presets: [{ preset_name: 'Existing' } as never] });
    const body = JSON.stringify({
      nodes: [],
      edges: [],
      presets: [{ preset_name: 'Existing' }, { preset_name: 'Fresh' }],
    });
    await importGraphFile(jsonFile(body));
    expect(useNodeDefStore.getState().presets.map((p) => p.preset_name)).toEqual([
      'Existing',
      'Fresh',
    ]);
  });

  it('writes no presets at all when the file carries none', async () => {
    const before = useNodeDefStore.getState().presets;
    await importGraphFile(jsonFile(JSON.stringify({ nodes: [raw('n1')], edges: [] })));
    expect(useNodeDefStore.getState().presets).toBe(before);
  });

  it.each([
    ['settings.device', { settings: { device: 'cuda:1' } }, 'cuda:1'],
    ['no settings block', {}, null],
    ['a device outside the pattern', { settings: { device: 'gpu' } }, null],
  ])('installs the graph device from %s', async (_label, extra, expected) => {
    useTabStore.getState().setGraphDevice('mps');
    await importGraphFile(jsonFile(JSON.stringify({ nodes: [], edges: [], ...extra })));
    expect(tabs()[0].graphDevice).toBe(expected);
  });

  it('opens a file written by a newer build read-only, and says so', async () => {
    await importGraphFile(jsonFile(JSON.stringify({ nodes: [], edges: [], format_version: 99 })));
    expect(tabs()[0].readOnly).toBe(true);
    expect(toasts().some((t) => t.type === 'warning' && t.message.includes('v99'))).toBe(true);
  });

  it('clears a stale read-only flag when an ordinary file is imported', async () => {
    useTabStore.getState().setTabReadOnly(true);
    await importGraphFile(jsonFile(JSON.stringify({ nodes: [], edges: [], format_version: 1 })));
    expect(tabs()[0].readOnly).toBe(false);
  });

  it('reports a file whose nodes are not a list', async () => {
    await importGraphFile(jsonFile(JSON.stringify({ nodes: 'oops', edges: [] })));
    expect(errorToasts()[0].message).toContain('Invalid graph format');
  });

  it('reports a file that is not JSON at all', async () => {
    expect(await importGraphFile(jsonFile('{not valid json'))).toBe(false);
    expect(errorToasts()).toHaveLength(1);
    expect(errorToasts()[0].message).toContain('Import failed');
  });

  // -- The two bugs the move fixed --

  describe('refusing JSON that is not a graph', () => {
    // `{}` passed every check the toolbar's import made: `data.nodes ?? []`
    // is an empty array. Picking a package.json therefore REPLACED the
    // canvas with nothing, through an install that pushes no undo frame --
    // so there was no way back to the graph that had been there.
    it.each([
      ['an empty object', '{}'],
      ['some other JSON file', JSON.stringify({ name: 'pkg', version: '1.0.0' })],
      ['a bare array', '[]'],
      ['a bare string', '"hello"'],
      ['literal null', 'null'],
      ['a number', '42'],
    ])('leaves the canvas alone for %s', async (_label, body) => {
      useTabStore.getState().setNodes([raw('onScreen')] as never);
      expect(await importGraphFile(jsonFile(body))).toBe(false);
      expect(tabs()[0].nodes.map((n) => n.id)).toEqual(['onScreen']);
      expect(errorToasts()[0].message).toContain('Not a graph file');
    });

    // The complement, and the reason the check is the KEY and not the
    // length: a graph the user really did empty is still a graph, and
    // importing it has to replace what is on screen.
    it('still imports a graph that genuinely holds no nodes', async () => {
      useTabStore.getState().setNodes([raw('onScreen')] as never);
      expect(await importGraphFile(jsonFile(JSON.stringify({ nodes: [], edges: [] })))).toBe(true);
      expect(tabs()[0].nodes).toEqual([]);
      expect(errorToasts()).toHaveLength(0);
    });
  });

  describe('an unreadable file', () => {
    // No `onerror` handler meant `readAsText` failing did nothing at all:
    // no canvas change, no message, nothing to tell the user their click
    // had been received.
    it('reports what the reader said', async () => {
      installFailingReader({ message: 'device is not readable' });
      expect(await importGraphFile(jsonFile('{}'))).toBe(false);
      expect(errorToasts()[0].message).toBe('Import failed: device is not readable');
    });

    it('still says something when the reader said nothing', async () => {
      installFailingReader(null);
      await importGraphFile(jsonFile('{}'));
      expect(errorToasts()[0].message).toBe('Import failed: Could not read the file');
    });
  });
});
