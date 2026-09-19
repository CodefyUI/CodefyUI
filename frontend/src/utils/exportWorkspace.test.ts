import { describe, it, expect, beforeEach, afterEach, vi, type MockInstance } from 'vitest';
import type { Node } from '@xyflow/react';
import { rememberAppVersion } from './appVersion';
import { collectWorkspaceFile, exportWorkspace } from './exportWorkspace';
import { GRAPH_FORMAT_VERSION } from './formatVersion';
import { importFile, importGraphFile } from './importGraphFile';
import { parseWorkspaceFile } from './workspaceFile';
import { useI18n } from '../i18n';
import { useNodeDefStore } from '../store/nodeDefStore';
import { useTabStore, type TabState } from '../store/tabStore';
import { useToastStore } from '../store/toastStore';
import { useUIStore } from '../store/uiStore';
import type { NodeData, NodeDefinition, SubgraphDefinition } from '../types';

const store = () => useTabStore.getState();
const toasts = () => useToastStore.getState().toasts;

const SECRET_DEF: NodeDefinition = {
  node_name: 'LLMChat',
  category: 'LLM',
  description: '',
  inputs: [],
  outputs: [],
  params: [
    {
      name: 'api_key',
      param_type: 'secret',
      default: '',
      description: '',
      options: [],
      min_value: null,
      max_value: null,
    },
  ],
};

function node(id: string, over: Partial<NodeData> = {}): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position: { x: 0, y: 0 },
    data: { label: id, type: 'Add', params: {}, ...over },
  };
}

/** A tab holding a graph, appended WITHOUT taking the focus. Returns its id. */
function tabWith(title: string, nodes: Node<NodeData>[] = [node('n1')]): string {
  const id = store().createTab({ title, activate: false });
  store().loadGraphDocumentInto(id, { nodes, edges: [], boundFile: null });
  return id;
}

function patchTab(id: string, patch: Partial<TabState>): void {
  useTabStore.setState({
    tabs: store().tabs.map((t) => (t.id === id ? { ...t, ...patch } : t)),
  });
}

/** jsdom's Blob has no .text(); read it the way the import flow does. */
function readBlob(blob: Blob): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsText(blob);
  });
}

let clickSpy: MockInstance<() => void>;

beforeEach(() => {
  localStorage.clear();
  rememberAppVersion(null);
  useI18n.setState({ locale: 'en' });
  useNodeDefStore.setState({ definitions: [], presets: [] });
  useToastStore.setState({ toasts: [] });
  useUIStore.setState({
    fontSize: 'default',
    edgeStyle: 'circuit',
    gridSnapEnabled: false,
    tooltipsEnabled: true,
    beginnerMode: false,
    globalDevice: 'cpu',
  });
  // One EMPTY tab, which is what a fresh editor holds.
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  store().addTab('Tab 1');
  // The same download plumbing stubs Toolbar.test.tsx installs.
  vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:mock');
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
  clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
  localStorage.clear();
});

describe('collectWorkspaceFile', () => {
  it('exports the tabs that hold a graph, in tab-strip order, leaving out empty, transient and read-only ones', () => {
    tabWith('A');
    patchTab(tabWith('Scratch'), { transient: true });
    patchTab(tabWith('Newer format'), { readOnly: true });
    tabWith('D');

    const { file, skippedReadOnly } = collectWorkspaceFile();
    // 'Tab 1' is empty, so it is not there either.
    expect(file!.tabs.map((t) => t.title)).toEqual(['A', 'D']);
    expect(skippedReadOnly).toBe(1);
  });

  it('re-indexes the active tab after skipping, and writes null when it was not exported', () => {
    tabWith('A');
    const readOnly = tabWith('Read-only');
    patchTab(readOnly, { readOnly: true });
    const d = tabWith('D');

    store().setActiveTab(d);
    expect(collectWorkspaceFile().file!.active).toBe(1);

    store().setActiveTab(readOnly);
    expect(collectWorkspaceFile().file!.active).toBeNull();
  });

  it('answers with no file when nothing is exportable', () => {
    expect(collectWorkspaceFile()).toEqual({ file: null, skippedReadOnly: 0 });
    patchTab(tabWith('Read-only'), { readOnly: true });
    expect(collectWorkspaceFile()).toEqual({ file: null, skippedReadOnly: 1 });
  });

  it('writes each graph exactly as Export JSON does, plus format_version', () => {
    tabWith('Plain');
    const onGpu = tabWith('On GPU');
    patchTab(onGpu, { graphDevice: 'cuda:0', description: 'trained on the GPU' });

    const { file } = collectWorkspaceFile();
    expect(Object.keys(file!.tabs[0].graph)).toEqual([
      'name', 'description', 'nodes', 'edges', 'presets', 'segmentGroups', 'subgraphs',
      'format_version',
    ]);
    // `settings` only when the graph assigns a device, as in Export JSON.
    expect(Object.keys(file!.tabs[1].graph)).toEqual([
      'name', 'description', 'nodes', 'edges', 'presets', 'segmentGroups', 'subgraphs',
      'settings', 'format_version',
    ]);
    expect(file!.tabs[1].title).toBe('On GPU');
    expect(file!.tabs[1].graph).toMatchObject({
      name: 'On GPU',
      description: 'trained on the GPU',
      settings: { device: 'cuda:0' },
      format_version: GRAPH_FORMAT_VERSION,
    });
  });

  it('carries each tab\'s run settings and the six preferences', () => {
    useI18n.setState({ locale: 'zh-TW' });
    useUIStore.setState({
      fontSize: 'large',
      edgeStyle: 'curve',
      gridSnapEnabled: true,
      tooltipsEnabled: false,
      beginnerMode: true,
    });
    const tuned = tabWith('Tuned');
    const run = {
      seed: 7,
      deterministic: true,
      recordOutputs: false,
      verboseMode: true,
      weightsPersistent: false,
      backwardMode: true,
      autoBackward: true,
    };
    store().setTabRunSettings(tuned, run);

    const { file } = collectWorkspaceFile();
    expect(file!.tabs[0].run).toEqual(run);
    expect(file!.preferences).toEqual({
      locale: 'zh-TW',
      fontSize: 'large',
      edgeStyle: 'curve',
      gridSnap: true,
      tooltips: false,
      beginnerMode: true,
    });
  });

  it('stamps the cached server version, or null when no health read has answered', () => {
    tabWith('A');
    expect(collectWorkspaceFile().file!.app_version).toBeNull();
    rememberAppVersion('2.8.2');
    expect(collectWorkspaceFile().file!.app_version).toBe('2.8.2');
  });

  it('stamps the export time it was handed', () => {
    tabWith('A');
    const now = new Date('2026-09-18T07:30:00.000Z');
    expect(collectWorkspaceFile(now).file!.exported_at).toBe('2026-09-18T07:30:00.000Z');
  });

  it('never writes secrets, bindings, ids, run state or machine-local settings', () => {
    localStorage.setItem('plugin:graph-copilot:settings', JSON.stringify({ apiKey: 'copilot-SENTINEL' }));
    localStorage.setItem('codefyui-global-device', 'cuda:7');
    localStorage.setItem('codefyui-sidebar-width', '417');
    useUIStore.setState({ globalDevice: 'cuda:7' });
    const id = tabWith('Secretive', [
      node('llm', {
        type: 'LLMChat',
        definition: SECRET_DEF,
        params: { api_key: 'sk-SENTINEL', temperature: 0.5 },
      }),
    ]);
    patchTab(id, {
      lastRunId: 'run-SENTINEL',
      graphId: 'graphid-SENTINEL',
      currentGraphFile: 'file-SENTINEL',
      currentGraphName: 'name-SENTINEL',
      projectOrigin: 'D:/origin-SENTINEL',
      source: { kind: 'agent-variant', pluginId: 'plugin-SENTINEL' },
      revision: 987654,
    });

    // A fixed clock: `exported_at` is the one free-running value in the file,
    // and a millisecond field of 417 would otherwise fail this test at random.
    const { file } = collectWorkspaceFile(new Date('2026-09-18T07:30:00.000Z'));
    const text = JSON.stringify(file);
    expect(file!.tabs).toHaveLength(1);
    // The rest of the node did travel; only the secret was blanked.
    expect(text).toContain('"temperature":0.5');
    expect(text).toContain('"api_key":""');
    // Not the bare prefix `codefyui-`: the format id is `codefyui-workspace`.
    for (const leaked of [
      'SENTINEL', 'cuda:7', '417', '987654',
      'lastRunId', 'graphId', 'currentGraphFile', 'currentGraphName', 'projectOrigin',
      'pluginId', 'revision', 'readOnly', 'transient',
      'globalDevice', 'codefyui-global-device', 'codefyui-sidebar', 'plugin:graph-copilot',
      'sidebar', 'lastLayoutMode',
      'undoStack', 'redoStack', 'logs', 'outputSummaries', 'viewport',
    ]) {
      expect(text, `leaked: ${leaked}`).not.toContain(leaked);
    }
  });

  it('writes graph blocks that Import accepts on their own', async () => {
    tabWith('Solo', [node('only')]);
    const { file } = collectWorkspaceFile();
    const block = new File([JSON.stringify(file!.tabs[0].graph)], 'solo.json', {
      type: 'application/json',
    });
    // The single-graph import replaces the ACTIVE tab, which is still 'Tab 1'.
    expect(await importGraphFile(block)).toBe(true);
    expect(store().getActiveTab().nodes.map((n) => n.id)).toEqual(['only']);
  });
});

describe('exportWorkspace', () => {
  it('downloads workspace-YYYY-MM-DD.cduiworkspace, as a type no browser renames', async () => {
    tabWith('A');
    exportWorkspace();

    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
    const blob = vi.mocked(URL.createObjectURL).mock.calls[0][0] as Blob;
    expect(blob.type).toBe('application/octet-stream');
    const anchor = clickSpy.mock.contexts[0] as HTMLAnchorElement;
    expect(anchor.download).toMatch(/^workspace-\d{4}-\d{2}-\d{2}\.cduiworkspace$/);
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:mock');

    const parsed = parseWorkspaceFile(JSON.parse(await readBlob(blob)));
    expect(parsed.ok).toBe(true);
    // Export JSON raises no success toast either: the browser shows the download.
    expect(toasts()).toHaveLength(0);
  });

  it('warns and downloads nothing when no tab is exportable', () => {
    exportWorkspace();
    expect(URL.createObjectURL).not.toHaveBeenCalled();
    expect(clickSpy).not.toHaveBeenCalled();
    expect(toasts().map((t) => [t.type, t.message])).toEqual([
      ['warning', 'Nothing to export: every tab is empty or read-only.'],
    ]);
  });

  it('still downloads when read-only tabs were skipped, and says how many', () => {
    tabWith('A');
    patchTab(tabWith('Newer format'), { readOnly: true });
    patchTab(tabWith('Another'), { readOnly: true });
    exportWorkspace();

    expect(clickSpy).toHaveBeenCalledTimes(1);
    expect(toasts().map((t) => [t.type, t.message])).toEqual([
      ['info', '2 read-only tab(s) were left out.'],
    ]);
  });
});

/**
 * The two halves against each other, through JSON TEXT and the live Import
 * door. Every other test in this feature covers one half against a
 * hand-written fixture, which is exactly where a field the exporter writes
 * under one name and the importer reads under another would survive.
 */
describe('export -> import, end to end', () => {
  const BLOCK: SubgraphDefinition = {
    id: 'blk',
    name: 'Block',
    description: '',
    nodes: [],
    edges: [],
    interface: { inputs: [], outputs: [], triggerTargets: [] },
  };

  /** A collapsed block's instance node: what keeps its definition exported. */
  function instanceNode(id: string, subgraphId: string): Node<NodeData> {
    return {
      id,
      type: 'subgraphNode',
      position: { x: 0, y: 0 },
      data: { label: 'Block', type: `subgraph:${subgraphId}`, params: {} },
    };
  }

  it('rebuilds every tab from the bytes the exporter wrote', async () => {
    const encoder = store().createTab({ title: 'Encoder', activate: false });
    store().loadGraphDocumentInto(encoder, {
      nodes: [node('a'), node('b'), instanceNode('blk1', 'blk')],
      edges: [
        { id: 'e1', source: 'a', target: 'b', sourceHandle: 'tensor', targetHandle: 'tensor' },
      ],
      boundFile: null,
      description: 'the first half',
      device: 'cuda:0',
      segmentGroups: [{ id: 's1', headNodeId: 'a', tailNodeId: 'b' }],
      subgraphs: [BLOCK],
    });
    const decoder = tabWith('Decoder', [node('c')]);
    store().setTabRunSettings(decoder, { seed: 7, backwardMode: true, weightsPersistent: false });
    store().setActiveTab(decoder);
    useUIStore.setState({ fontSize: 'large', gridSnapEnabled: true });
    const sourceGraphIds = [encoder, decoder].map((id) => store().getTab(id)!.graphId);

    const text = JSON.stringify(collectWorkspaceFile().file);

    // Another browser: one empty tab, and the preferences at their defaults.
    useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
    store().addTab('Tab 1');
    useUIStore.setState({ fontSize: 'default', gridSnapEnabled: false });

    expect(await importFile(new File([text], 'w.cduiworkspace'))).toBe(true);

    // The empty tab is gone, both tabs are back in order, the second is active.
    expect(store().tabs.map((t) => t.name)).toEqual(['Encoder', 'Decoder']);
    expect(store().getActiveTab().name).toBe('Decoder');

    const [first, second] = store().tabs;
    expect(first.nodes.map((n) => [n.id, n.data.type])).toEqual([
      ['a', 'Add'],
      ['b', 'Add'],
      ['blk1', 'subgraph:blk'],
    ]);
    expect(first.edges.map((e) => [e.source, e.target])).toEqual([['a', 'b']]);
    expect(first.description).toBe('the first half');
    expect(first.graphDevice).toBe('cuda:0');
    expect(first.segmentGroups).toEqual([{ id: 's1', headNodeId: 'a', tailNodeId: 'b' }]);
    expect(first.subgraphs.map((d) => d.id)).toEqual(['blk']);

    expect(second.nodes.map((n) => n.id)).toEqual(['c']);
    // All seven, the three that were changed and the four that were not.
    expect(second).toMatchObject({
      seed: 7,
      deterministic: false,
      recordOutputs: true,
      verboseMode: false,
      weightsPersistent: false,
      backwardMode: true,
      autoBackward: false,
    });

    for (const tab of store().tabs) {
      // Bound to nothing, so the first Save asks where the graph should go.
      expect(tab.currentGraphFile).toBeNull();
      // A copied id would make two tabs share trained weights on the server.
      expect(sourceGraphIds).not.toContain(tab.graphId);
    }
    expect(useUIStore.getState()).toMatchObject({ fontSize: 'large', gridSnapEnabled: true });
  });
});
