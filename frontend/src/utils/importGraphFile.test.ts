import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { importFile, importGraphFile } from './importGraphFile';
import { importWorkspaceFile } from './importWorkspaceFile';
import { buildWorkspaceFile } from './workspaceFile';
import { MAX_WORKSPACE_FILE_BYTES } from './workspaceLimits';
import { useNodeDefStore } from '../store/nodeDefStore';
import { useTabStore } from '../store/tabStore';
import { useToastStore } from '../store/toastStore';
import { useI18n } from '../i18n';

// The workspace importer runs for real everywhere below -- what the router
// hands it and what the tabs become is the point of these tests. The spy is
// here for the one test that needs it to REJECT, which no input of its own
// can make it do.
vi.mock('./importWorkspaceFile', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./importWorkspaceFile')>();
  return { ...actual, importWorkspaceFile: vi.fn(actual.importWorkspaceFile) };
});

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

describe('importFile', () => {
  /** A picked file with a name of the caller's choosing. */
  function namedFile(body: string, name: string) {
    return new File([body], name, { type: 'application/octet-stream' });
  }

  /** A valid one-tab workspace file, as parsed JSON the caller may then break. */
  function workspaceJson(): Record<string, unknown> {
    const file = buildWorkspaceFile({
      appVersion: '2.8.2',
      exportedAt: '2026-09-18T07:30:00.000Z',
      active: 0,
      tabs: [
        {
          title: 'Moved',
          graph: {
            name: 'Moved', description: '', nodes: [raw('n1')], edges: [],
            presets: [], segmentGroups: [], subgraphs: [], format_version: 1,
          },
          run: {
            seed: 7, deterministic: false, recordOutputs: true, verboseMode: false,
            weightsPersistent: true, backwardMode: false, autoBackward: false,
          },
        },
      ],
    });
    return JSON.parse(JSON.stringify(file)) as Record<string, unknown>;
  }

  /** Report a size over the cap without allocating 64 MiB. */
  function oversized(file: File): File {
    Object.defineProperty(file, 'size', { value: MAX_WORKSPACE_FILE_BYTES + 1 });
    return file;
  }

  /** A reader that counts what it is asked to read, and answers with `body`. */
  function installCountingReader(body: string): File[] {
    const reads: File[] = [];
    class CountingFileReader {
      error = null;
      onerror: (() => void) | null = null;
      onload: ((event: { target: { result: string } }) => void) | null = null;
      readAsText(file: File) {
        reads.push(file);
        this.onload?.({ target: { result: body } });
      }
    }
    vi.stubGlobal('FileReader', CountingFileReader as unknown as typeof FileReader);
    return reads;
  }

  it.each(['g.json', 'renamed.cduiworkspace'])(
    'still sends a plain graph in %s down the old path: it replaces the active tab and opens none',
    async (name) => {
      useTabStore.getState().setNodes([raw('onScreen')] as never);
      const body = JSON.stringify({ nodes: [raw('n1')], edges: [] });
      expect(await importFile(namedFile(body, name))).toBe(true);
      expect(tabs()).toHaveLength(1);
      expect(tabs()[0].nodes.map((n) => n.id)).toEqual(['n1']);
    },
  );

  it.each(['moved.cduiworkspace', 'renamed.json'])(
    'opens a workspace called %s as tabs: the content decides, not the name',
    async (name) => {
      useTabStore.getState().setNodes([raw('onScreen')] as never);
      expect(await importFile(namedFile(JSON.stringify(workspaceJson()), name))).toBe(true);
      expect(tabs().map((t) => t.name)).toEqual(['Tab 1', 'Moved']);
      // The graph that was on screen is still there.
      expect(tabs()[0].nodes.map((n) => n.id)).toEqual(['onScreen']);
      expect(tabs()[1].nodes.map((n) => n.id)).toEqual(['n1']);
      expect(tabs()[1].seed).toBe(7);
    },
  );

  it('keeps the tab default for a run value of the wrong type', async () => {
    const json = workspaceJson();
    const [first] = json.tabs as Record<string, unknown>[];
    json.tabs = [{ ...first, run: { seed: 'seven', verboseMode: true } }];
    await importFile(namedFile(JSON.stringify(json), 'w.cduiworkspace'));
    const moved = tabs().find((t) => t.name === 'Moved')!;
    expect(moved.seed).toBeNull();
    expect(moved.verboseMode).toBe(true);
  });

  it.each(['g.json', 'w.cduiworkspace'])('reports invalid JSON in %s', async (name) => {
    expect(await importFile(namedFile('{not valid json', name))).toBe(false);
    expect(errorToasts()).toHaveLength(1);
    expect(errorToasts()[0].message).toContain('Import failed');
  });

  it('reports a file that cannot be read', async () => {
    installFailingReader({ message: 'device is not readable' });
    expect(await importFile(namedFile('{}', 'w.cduiworkspace'))).toBe(false);
    expect(errorToasts()[0].message).toBe('Import failed: device is not readable');
  });

  it('refuses a workspace written by a newer CodefyUI, and creates nothing', async () => {
    const before = tabs();
    const body = JSON.stringify({ ...workspaceJson(), version: 2 });
    expect(await importFile(namedFile(body, 'w.cduiworkspace'))).toBe(false);
    expect(tabs()).toBe(before);
    expect(errorToasts().map((t) => t.message)).toEqual([
      'This workspace was written by a newer CodefyUI. Nothing was imported.',
    ]);
  });

  it('refuses a workspace with no tabs in it', async () => {
    const body = JSON.stringify({ ...workspaceJson(), tabs: [] });
    expect(await importFile(namedFile(body, 'w.cduiworkspace'))).toBe(false);
    expect(errorToasts().map((t) => t.message)).toEqual(['Not a valid workspace file.']);
  });

  // The importer raises its own toasts for everything it can see coming, but
  // its store calls sit outside its try: a rejection would otherwise escape
  // the `void importFile(file)` the panel makes, and the user would be told
  // nothing at all.
  it('reports an importer that rejects, rather than letting it escape', async () => {
    vi.mocked(importWorkspaceFile).mockClear();
    vi.mocked(importWorkspaceFile).mockRejectedValueOnce(new Error('the tab store is gone'));
    const body = JSON.stringify(workspaceJson());
    expect(await importFile(namedFile(body, 'w.cduiworkspace'))).toBe(false);
    expect(errorToasts().map((t) => t.message)).toEqual([
      'Import failed: the tab store is gone',
    ]);
    // Called once, so the queued one-shot rejection is spent here and cannot
    // leak into whichever test runs next.
    expect(vi.mocked(importWorkspaceFile)).toHaveBeenCalledTimes(1);
  });

  describe('the 64 MiB cap', () => {
    // The second row is the extension check's case: a name off a Windows
    // disk can arrive in any case, and the check is what decides whether the
    // file is read into memory at all.
    it.each(['big.cduiworkspace', 'BIG.CDUIWORKSPACE'])(
      'refuses an oversize %s without reading it',
      async (name) => {
        const reads = installCountingReader(JSON.stringify(workspaceJson()));
        const file = oversized(namedFile('ignored', name));
        expect(await importFile(file)).toBe(false);
        expect(reads).toHaveLength(0);
        expect(errorToasts().map((t) => t.message)).toEqual([
          'The workspace file is over 64 MiB.',
        ]);
      },
    );

    it('refuses an oversize workspace under another name as soon as it sees what it is', async () => {
      const before = tabs();
      const file = oversized(namedFile(JSON.stringify(workspaceJson()), 'renamed.json'));
      expect(await importFile(file)).toBe(false);
      expect(tabs()).toBe(before);
      expect(errorToasts().map((t) => t.message)).toEqual([
        'The workspace file is over 64 MiB.',
      ]);
    });

    it('does not apply to a plain graph, whose import is unchanged', async () => {
      const file = oversized(jsonFile(JSON.stringify({ nodes: [raw('n1')], edges: [] })));
      expect(await importFile(file)).toBe(true);
      expect(tabs()[0].nodes.map((n) => n.id)).toEqual(['n1']);
    });
  });
});
