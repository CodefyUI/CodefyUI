import { describe, it, expect } from 'vitest';
import {
  WORKSPACE_EXTENSION,
  WORKSPACE_FORMAT,
  WORKSPACE_VERSION,
  buildWorkspaceFile,
  isWorkspaceFile,
  parseWorkspaceFile,
  workspaceFileName,
  type WorkspaceBuildInput,
  type WorkspaceGraph,
  type WorkspaceRunSettings,
} from './workspaceFile';

const RUN: WorkspaceRunSettings = {
  seed: 7,
  deterministic: true,
  recordOutputs: false,
  verboseMode: true,
  weightsPersistent: false,
  backwardMode: true,
  autoBackward: true,
};

const DEFAULT_RUN: WorkspaceRunSettings = {
  seed: null,
  deterministic: false,
  recordOutputs: true,
  verboseMode: false,
  weightsPersistent: true,
  backwardMode: false,
  autoBackward: false,
};

function graph(name: string, extra: Partial<WorkspaceGraph> = {}): WorkspaceGraph {
  return {
    name,
    description: `${name} description`,
    nodes: [{ id: 'n1', type: 'Add', position: { x: 0, y: 0 }, data: { params: { k: 1 } } }],
    edges: [],
    presets: [],
    segmentGroups: [],
    subgraphs: [],
    format_version: 1,
    ...extra,
  };
}

function input(over: Partial<WorkspaceBuildInput> = {}): WorkspaceBuildInput {
  return {
    appVersion: '2.8.2',
    exportedAt: '2026-09-18T07:30:00.000Z',
    active: 1,
    tabs: [
      { title: 'MNIST CNN', graph: graph('MNIST CNN', { settings: { device: 'cuda:0' } }), run: RUN },
      { title: 'Second', graph: graph('Second'), run: DEFAULT_RUN },
    ],
    preferences: {
      locale: 'zh-TW',
      fontSize: 'large',
      edgeStyle: 'curve',
      gridSnap: true,
      tooltips: false,
      beginnerMode: true,
    },
    ...over,
  };
}

/** What a reader gets: the value after a trip through JSON text. */
function viaJson(value: unknown): unknown {
  return JSON.parse(JSON.stringify(value));
}

/** A valid two-tab file as parsed JSON, for cases that then break one field. */
function fileJson(): Record<string, unknown> {
  return viaJson(buildWorkspaceFile(input())) as Record<string, unknown>;
}

describe('buildWorkspaceFile', () => {
  it('writes the envelope in the documented order, stamped with the format and version', () => {
    const file = buildWorkspaceFile(input());
    expect(Object.keys(file)).toEqual([
      'format', 'version', 'app_version', 'exported_at', 'active', 'tabs', 'preferences',
    ]);
    expect(file.format).toBe('codefyui-workspace');
    expect(file.version).toBe(1);
    expect(WORKSPACE_FORMAT).toBe('codefyui-workspace');
    expect(WORKSPACE_VERSION).toBe(1);
    expect(WORKSPACE_EXTENSION).toBe('.cduiworkspace');
  });

  it('writes null for an unknown app version and for an active index that names no tab', () => {
    const file = buildWorkspaceFile(input({ appVersion: null, active: 5 }));
    expect(file.app_version).toBeNull();
    expect(file.active).toBeNull();
    expect(buildWorkspaceFile(input({ active: null })).active).toBeNull();
  });

  it('leaves the preferences block out when there is nothing in it', () => {
    expect('preferences' in buildWorkspaceFile(input({ preferences: {} }))).toBe(false);
    expect('preferences' in buildWorkspaceFile(input({ preferences: undefined }))).toBe(false);
  });
});

describe('build -> parse', () => {
  it('is lossless for everything the format carries', () => {
    const source = input();
    const parsed = parseWorkspaceFile(viaJson(buildWorkspaceFile(source)));
    if (!parsed.ok) throw new Error(`parse failed: ${parsed.reason}`);
    expect(parsed.workspace).toEqual({
      version: 1,
      appVersion: '2.8.2',
      exportedAt: '2026-09-18T07:30:00.000Z',
      active: 1,
      tabs: source.tabs,
      preferences: source.preferences,
    });
  });
});

describe('parseWorkspaceFile', () => {
  it.each([
    ['an empty object', {}],
    ['a plain graph', { nodes: [], edges: [] }],
    ['a bare array', []],
    ['literal null', null],
    ['a string', 'codefyui-workspace'],
    ['another format', { format: 'something-else', version: 1, tabs: [{}] }],
  ])('does not recognise %s', (_label, data) => {
    expect(isWorkspaceFile(data)).toBe(false);
    expect(parseWorkspaceFile(data)).toEqual({ ok: false, reason: 'not_workspace' });
  });

  it('recognises the file by its content alone', () => {
    expect(isWorkspaceFile({ format: 'codefyui-workspace' })).toBe(true);
  });

  it.each([
    ['missing', undefined],
    ['a string', '1'],
    ['fractional', 1.5],
    ['zero', 0],
    ['negative', -1],
  ])('refuses a version that is %s', (_label, version) => {
    expect(parseWorkspaceFile({ ...fileJson(), version })).toEqual({ ok: false, reason: 'invalid' });
  });

  it('refuses a newer envelope outright, whatever else the file holds', () => {
    // Checked before `tabs`: a v2 file may well shape its tabs differently,
    // and "written by a newer CodefyUI" is the true answer, not "invalid".
    expect(
      parseWorkspaceFile({
        format: 'codefyui-workspace',
        version: 2,
        tabs: 'a shape this build has never seen',
      }),
    ).toEqual({ ok: false, reason: 'too_new' });
  });

  it.each([
    ['missing', undefined],
    ['not a list', {}],
    ['empty', []],
  ])('refuses tabs that are %s', (_label, tabs) => {
    expect(parseWorkspaceFile({ ...fileJson(), tabs })).toEqual({ ok: false, reason: 'invalid' });
  });

  it('keeps every entry positional, including ones the importer will refuse', () => {
    const parsed = parseWorkspaceFile({
      ...fileJson(),
      tabs: [42, { title: 7, graph: 'nope' }, { title: 'ok', graph: { nodes: [], edges: [] } }],
    });
    if (!parsed.ok) throw new Error(`parse failed: ${parsed.reason}`);
    expect(parsed.workspace.tabs).toEqual([
      { title: undefined, graph: undefined, run: {} },
      { title: 7, graph: 'nope', run: {} },
      { title: 'ok', graph: { nodes: [], edges: [] }, run: {} },
    ]);
  });

  it('type-checks each run value: a wrong-typed or missing one is left out, so the tab keeps its default', () => {
    const parsed = parseWorkspaceFile({
      ...fileJson(),
      tabs: [
        {
          title: 'T',
          graph: {},
          run: {
            seed: '7',
            deterministic: 'yes',
            recordOutputs: false,
            verboseMode: 1,
            backwardMode: true,
            notARunSetting: true,
          },
        },
      ],
    });
    if (!parsed.ok) throw new Error(`parse failed: ${parsed.reason}`);
    expect(parsed.workspace.tabs[0].run).toEqual({ recordOutputs: false, backwardMode: true });
  });

  it('keeps an explicit null seed, which says "no seed" rather than nothing', () => {
    const parsed = parseWorkspaceFile({
      ...fileJson(),
      tabs: [{ title: 'T', graph: {}, run: { seed: null } }],
    });
    if (!parsed.ok) throw new Error(`parse failed: ${parsed.reason}`);
    expect(parsed.workspace.tabs[0].run).toEqual({ seed: null });
  });

  it.each([[-1], [2], [0.5], ['0'], [undefined]])(
    'reads an active index of %s as "the active tab was not exported"',
    (active) => {
      const parsed = parseWorkspaceFile({ ...fileJson(), active });
      if (!parsed.ok) throw new Error(`parse failed: ${parsed.reason}`);
      expect(parsed.workspace.active).toBeNull();
    },
  );

  it('ignores preferences it does not know or cannot use, silently', () => {
    const parsed = parseWorkspaceFile({
      ...fileJson(),
      preferences: {
        locale: 'fr',
        fontSize: 'huge',
        edgeStyle: 'zigzag',
        gridSnap: 'true',
        tooltips: false,
        beginnerMode: 0,
        globalDevice: 'cuda:0',
        sidebarWidth: 400,
      },
    });
    if (!parsed.ok) throw new Error(`parse failed: ${parsed.reason}`);
    expect(parsed.workspace.preferences).toEqual({ tooltips: false });
  });

  it('does not take an inherited object key for an allowed value', () => {
    const parsed = parseWorkspaceFile({
      ...fileJson(),
      preferences: { locale: 'toString', fontSize: 'constructor', edgeStyle: 'hasOwnProperty' },
    });
    if (!parsed.ok) throw new Error(`parse failed: ${parsed.reason}`);
    expect(parsed.workspace.preferences).toEqual({});
  });

  it('reads a file with no preferences block as no preferences', () => {
    const json = fileJson();
    delete json.preferences;
    const parsed = parseWorkspaceFile(json);
    if (!parsed.ok) throw new Error(`parse failed: ${parsed.reason}`);
    expect(parsed.workspace.preferences).toEqual({});
  });
});

describe('workspaceFileName', () => {
  it('uses the LOCAL date, zero-padded, and the workspace extension', () => {
    // `new Date(y, m, d, ...)` is local time, so this holds in any timezone.
    expect(workspaceFileName(new Date(2026, 8, 18, 23, 59))).toBe(
      'workspace-2026-09-18.cduiworkspace',
    );
    expect(workspaceFileName(new Date(2027, 0, 5, 0, 1))).toBe(
      'workspace-2027-01-05.cduiworkspace',
    );
  });
});
