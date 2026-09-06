import { describe, it, expect } from 'vitest';
import {
  graphDiffKind,
  summarizeGraphDiff,
  MAX_GRAPH_DIFF_LINES,
  MAX_GRAPH_DIFF_VALUE,
} from './graphDiff';

// The documents below are shaped like the real ones: `split_graph`
// (`backend/app/core/project.py:74`) writes a logic node as `{id, type, data}`
// with the params under `data.params`, and copies `edges` verbatim -- so a
// trigger edge really does arrive with NO `targetHandle`, exactly as in
// `examples/Classical/Iris-Sklearn-KNN/graph.json`.

function graphDoc(over: Record<string, unknown> = {}): string {
  return JSON.stringify({
    format_version: 1,
    name: 'Iris',
    description: '',
    nodes: [],
    edges: [],
    presets: [],
    subgraphs: [],
    ...over,
  });
}

function layoutDoc(over: Record<string, unknown> = {}): string {
  return JSON.stringify({
    format_version: 1,
    positions: {},
    notes: {},
    segmentGroups: [],
    subgraphPositions: {},
    ...over,
  });
}

const START = { id: 'start-1', type: 'Start', data: { params: {} } };
const CSV = {
  id: 'csv',
  type: 'CSVReader',
  data: { params: { path: 'data/samples/iris.csv', skip_header: true } },
};
const KNN = {
  id: 'knn',
  type: 'KNN',
  data: { params: { n_neighbors: 5, weights: 'distance' } },
};
// No `targetHandle` -- the shape that makes the naive key end in `undefined`.
const TRIGGER_EDGE = {
  id: 'trig', source: 'start-1', target: 'csv', sourceHandle: 'trigger', type: 'trigger',
};
const DATA_EDGE = {
  id: 'e_csv_knn', source: 'csv', target: 'knn', sourceHandle: 'tensor', targetHandle: 'x_train',
};

// A preset carries a THIRD node shape: `{id, type, params}`, params at the top
// level (`frontend/src/types/index.ts:151`).
const PRESET = {
  preset_name: 'Dense Block',
  category: 'Layers',
  description: '',
  tags: [],
  nodes: [
    { id: 'p_lin', type: 'Linear', params: { out_features: 32 } },
    { id: 'p_act', type: 'ReLU', params: {} },
  ],
  edges: [
    { source: 'p_lin', sourceHandle: 'tensor', target: 'p_act', targetHandle: 'tensor' },
  ],
  exposed_inputs: [],
  exposed_outputs: [],
  exposed_params: [],
};

const SUBGRAPH = {
  id: 'sg-1',
  name: 'Block',
  description: '',
  nodes: [{ id: 'in_a', type: 'Linear', data: { params: { out_features: 8 } } }],
  edges: [],
  interface: { inputs: [], outputs: [], triggerTargets: [] },
};

describe('graphDiffKind', () => {
  it('recognises the two halves of a saved project graph by suffix', () => {
    expect(graphDiffKind('graphs/iris.graph.json')).toBe('graph');
    expect(graphDiffKind('layout/iris.layout.json')).toBe('layout');
  });

  it('has nothing to say about any other file', () => {
    // Non-project mode saves a single `<name>.json`, which carries positions
    // inside the nodes and is deliberately NOT summarised.
    expect(graphDiffKind('backend/data/graphs/cap-probe.json')).toBeNull();
    expect(graphDiffKind('README.md')).toBeNull();
    expect(graphDiffKind('.env')).toBeNull();
    expect(graphDiffKind('')).toBeNull();
    expect(graphDiffKind('graphs/iris.graph.json.bak')).toBeNull();
  });

  it('matches the suffix whatever case the file system handed it back in', () => {
    expect(graphDiffKind('graphs/Iris.GRAPH.JSON')).toBe('graph');
    expect(graphDiffKind('layout/Iris.Layout.Json')).toBe('layout');
  });
});

describe('summarizeGraphDiff, logic files', () => {
  it('counts one node added and one node removed', () => {
    const summary = summarizeGraphDiff(
      graphDoc({ nodes: [START, CSV] }),
      graphDoc({ nodes: [START, KNN] }),
      'graph',
    );
    expect(summary.lines).toEqual([
      { kind: 'nodesAdded', count: 1 },
      { kind: 'nodesRemoved', count: 1 },
    ]);
    expect(summary).toMatchObject({ more: 0, noLogicChange: false, unparseable: false });
  });

  it('names a changed parameter by node, with both values', () => {
    const summary = summarizeGraphDiff(
      graphDoc({ nodes: [KNN] }),
      graphDoc({
        nodes: [{ ...KNN, data: { params: { n_neighbors: 7, weights: 'distance' } } }],
      }),
      'graph',
    );
    expect(summary.lines).toEqual([
      { kind: 'param', node: 'knn', param: 'n_neighbors', from: '5', to: '7' },
    ]);
  });

  it('shows an added or removed parameter with an empty value on the missing side', () => {
    const summary = summarizeGraphDiff(
      graphDoc({ nodes: [{ id: 'n', type: 'T', data: { params: { gone: 1 } } }] }),
      graphDoc({ nodes: [{ id: 'n', type: 'T', data: { params: { fresh: 2 } } }] }),
      'graph',
    );
    expect(summary.lines).toEqual([
      { kind: 'param', node: 'n', param: 'fresh', from: '', to: '2' },
      { kind: 'param', node: 'n', param: 'gone', from: '1', to: '' },
    ]);
  });

  it('compares a parameter whose name is one Object.prototype already answers to', () => {
    // `key in params` is true for `toString` and `constructor` on every object
    // `JSON.parse` builds, so a parameter with such a name used to be compared
    // against an inherited function instead of against nothing: the removal
    // vanished entirely and the addition read `from: null`.
    const dropped = summarizeGraphDiff(
      graphDoc({ nodes: [{ id: 'n', type: 'T', data: { params: { toString: 1 } } }] }),
      graphDoc({ nodes: [{ id: 'n', type: 'T', data: { params: {} } }] }),
      'graph',
    );
    expect(dropped.lines).toEqual([
      { kind: 'param', node: 'n', param: 'toString', from: '1', to: '' },
    ]);

    const gained = summarizeGraphDiff(
      graphDoc({ nodes: [{ id: 'n', type: 'T', data: { params: {} } }] }),
      graphDoc({ nodes: [{ id: 'n', type: 'T', data: { params: { constructor: 2 } } }] }),
      'graph',
    );
    expect(gained.lines).toEqual([
      { kind: 'param', node: 'n', param: 'constructor', from: '', to: '2' },
    ]);
  });

  it('clips a long parameter value to forty characters', () => {
    const long = 'x'.repeat(60);
    const summary = summarizeGraphDiff(
      graphDoc({ nodes: [{ id: 'n', type: 'T', data: { params: { note: '' } } }] }),
      graphDoc({ nodes: [{ id: 'n', type: 'T', data: { params: { note: long } } }] }),
      'graph',
    );
    const line = summary.lines[0];
    expect(line).toMatchObject({ kind: 'param', node: 'n', param: 'note', from: '""' });
    expect(line).toHaveProperty('to');
    const to = (line as { to: string }).to;
    expect(to).toHaveLength(MAX_GRAPH_DIFF_VALUE);
    expect(to).toBe(`"${'x'.repeat(36)}...`);
  });

  it('compares values at full length even though it shows them clipped', () => {
    // Two paths that agree for the first forty characters are still two
    // paths. Comparing what is shown rather than what is stored would hide
    // the edit entirely -- and clipping is exactly what long file paths hit.
    const shared = 'data/samples/a-very-long-directory-name-here/';
    const summary = summarizeGraphDiff(
      graphDoc({ nodes: [{ id: 'n', type: 'T', data: { params: { path: `${shared}one.csv` } } }] }),
      graphDoc({ nodes: [{ id: 'n', type: 'T', data: { params: { path: `${shared}two.csv` } } }] }),
      'graph',
    );
    expect(summary.lines).toHaveLength(1);
    expect(summary.lines[0]).toMatchObject({ kind: 'param', node: 'n', param: 'path' });
    const line = summary.lines[0] as { from: string; to: string };
    expect(line.from).toBe(line.to);
  });

  it('clips by code point so a wide character is never cut in half', () => {
    // U+1F600, which JavaScript stores as two code units. Cutting at a raw
    // index would leave half of one behind, and a lone surrogate renders as
    // a replacement glyph -- data loss, not a value that was too long.
    const wide = '\uD83D\uDE00';
    const summary = summarizeGraphDiff(
      graphDoc({ nodes: [{ id: 'n', type: 'T', data: { params: { note: '' } } }] }),
      graphDoc({ nodes: [{ id: 'n', type: 'T', data: { params: { note: wide.repeat(50) } } }] }),
      'graph',
    );
    const to = (summary.lines[0] as { to: string }).to;
    expect(Array.from(to)).toHaveLength(MAX_GRAPH_DIFF_VALUE);
    expect(to).toBe(`"${wide.repeat(36)}...`);
  });

  it('reports a node whose type changed instead of counting it twice', () => {
    const summary = summarizeGraphDiff(
      graphDoc({ nodes: [{ id: 'clf', type: 'KNN', data: { params: {} } }] }),
      graphDoc({ nodes: [{ id: 'clf', type: 'DecisionTree', data: { params: {} } }] }),
      'graph',
    );
    expect(summary.lines).toEqual([
      { kind: 'typeChanged', node: 'clf', from: 'KNN', to: 'DecisionTree' },
    ]);
  });

  it('says nothing about the parameters of a node whose type changed', () => {
    // Two different node types have two different parameter sets, so every
    // key would read as added or removed -- noise on top of the one line
    // that actually explains the change.
    const summary = summarizeGraphDiff(
      graphDoc({ nodes: [{ id: 'clf', type: 'KNN', data: { params: { n_neighbors: 5 } } }] }),
      graphDoc({ nodes: [{ id: 'clf', type: 'DecisionTree', data: { params: { depth: 3 } } }] }),
      'graph',
    );
    expect(summary.lines).toHaveLength(1);
    expect(summary.lines[0]).toMatchObject({ kind: 'typeChanged' });
  });

  it('names a node by its label when the user has renamed it', () => {
    const summary = summarizeGraphDiff(
      graphDoc({ nodes: [{ id: 'n1', type: 'KNN', data: { label: 'Classifier', params: { k: 1 } } }] }),
      graphDoc({ nodes: [{ id: 'n1', type: 'KNN', data: { label: 'Classifier', params: { k: 2 } } }] }),
      'graph',
    );
    expect(summary.lines[0]).toMatchObject({ node: 'Classifier' });
  });

  it('keeps an id a person could have written as the name', () => {
    // Sixteen characters is the boundary. A hand-written id is what the
    // reader sees on the canvas, so printing the type instead would rename a
    // node they already know.
    const id = 'preprocess-train';
    expect(id).toHaveLength(16);
    const summary = summarizeGraphDiff(
      graphDoc({ nodes: [{ id, type: 'KNN', data: { params: { k: 1 } } }] }),
      graphDoc({ nodes: [{ id, type: 'KNN', data: { params: { k: 2 } } }] }),
      'graph',
    );
    expect(summary.lines[0]).toMatchObject({ node: id });
  });

  it('names a node with a generated id by its type and the head of that id', () => {
    // Every node the palette, an example or a paste inserts carries a UUID
    // and no label, so the summary read
    // "f256484a-51e0-49b4-8134-4aea94b5fd68: default ..." -- thirty-six
    // characters of a narrow column saying nothing about the one node the
    // reader can see.
    const id = 'f256484a-51e0-49b4-8134-4aea94b5fd68';
    const summary = summarizeGraphDiff(
      graphDoc({ nodes: [{ id, type: 'GraphInput', data: { params: { default: 'a' } } }] }),
      graphDoc({ nodes: [{ id, type: 'GraphInput', data: { params: { default: 'b' } } }] }),
      'graph',
    );
    expect(summary.lines[0]).toMatchObject({ node: 'GraphInput f256484a' });
  });

  it('prefers the label over a generated id', () => {
    const id = 'f256484a-51e0-49b4-8134-4aea94b5fd68';
    const summary = summarizeGraphDiff(
      graphDoc({ nodes: [{ id, type: 'GraphInput', data: { label: 'Prompt', params: { default: 'a' } } }] }),
      graphDoc({ nodes: [{ id, type: 'GraphInput', data: { label: 'Prompt', params: { default: 'b' } } }] }),
      'graph',
    );
    expect(summary.lines[0]).toMatchObject({ node: 'Prompt' });
  });

  it('falls back to the head of the id alone when the node has no type', () => {
    // A document is readable without a `type` on every node, and " f256484a"
    // with a leading space is a name nobody wrote.
    const id = 'f256484a-51e0-49b4-8134-4aea94b5fd68';
    const summary = summarizeGraphDiff(
      graphDoc({ nodes: [{ id, data: { params: { k: 1 } } }] }),
      graphDoc({ nodes: [{ id, data: { params: { k: 2 } } }] }),
      'graph',
    );
    expect(summary.lines[0]).toMatchObject({ node: 'f256484a' });
  });

  it('shortens a generated id inside a preset the same way', () => {
    const id = 'f256484a-51e0-49b4-8134-4aea94b5fd68';
    const preset = (out: number) => ({
      ...PRESET, nodes: [{ id, type: 'Linear', params: { out_features: out } }], edges: [],
    });
    const summary = summarizeGraphDiff(
      graphDoc({ presets: [preset(32)] }),
      graphDoc({ presets: [preset(64)] }),
      'graph',
    );
    expect(summary.lines[0]).toMatchObject({ node: 'Dense Block/Linear f256484a' });
  });

  it('counts an edge added even when the trigger edge carries no targetHandle', () => {
    const summary = summarizeGraphDiff(
      graphDoc({ nodes: [START, CSV], edges: [] }),
      graphDoc({ nodes: [START, CSV], edges: [TRIGGER_EDGE] }),
      'graph',
    );
    expect(summary.lines).toEqual([{ kind: 'edgesAdded', count: 1 }]);
  });

  it('treats a missing handle and an empty one as the same edge', () => {
    // Otherwise every trigger edge in the file would read as removed and
    // re-added the moment anything else on the graph changed.
    const summary = summarizeGraphDiff(
      graphDoc({ nodes: [START, CSV], edges: [TRIGGER_EDGE] }),
      graphDoc({ nodes: [START, CSV], edges: [{ ...TRIGGER_EDGE, id: 'other', targetHandle: '' }] }),
      'graph',
    );
    expect(summary.lines).toEqual([]);
    expect(summary.noLogicChange).toBe(true);
  });

  it('ignores the edge id, which copy and paste regenerates', () => {
    const summary = summarizeGraphDiff(
      graphDoc({ nodes: [CSV, KNN], edges: [DATA_EDGE] }),
      graphDoc({ nodes: [CSV, KNN], edges: [{ ...DATA_EDGE, id: 'xyz-999' }] }),
      'graph',
    );
    expect(summary.noLogicChange).toBe(true);
  });

  it('counts a duplicated edge rather than collapsing it into one key', () => {
    const summary = summarizeGraphDiff(
      graphDoc({ nodes: [CSV, KNN], edges: [DATA_EDGE] }),
      graphDoc({
        nodes: [CSV, KNN],
        edges: [DATA_EDGE, { ...DATA_EDGE, id: 'dup' }],
      }),
      'graph',
    );
    expect(summary.lines).toEqual([{ kind: 'edgesAdded', count: 1 }]);
  });

  it('calls key order and whitespace no logic change at all', () => {
    const ordered = JSON.stringify({
      format_version: 1, name: 'Iris', description: '',
      nodes: [CSV], edges: [], presets: [], subgraphs: [],
    });
    const shuffled = JSON.stringify(
      {
        subgraphs: [], presets: [], edges: [],
        nodes: [{ data: { params: { skip_header: true, path: 'data/samples/iris.csv' } }, type: 'CSVReader', id: 'csv' }],
        description: '', name: 'Iris', format_version: 1,
      },
      null,
      2,
    );
    const summary = summarizeGraphDiff(ordered, shuffled, 'graph');
    expect(summary).toEqual({ lines: [], more: 0, noLogicChange: true, unparseable: false });
  });

  it('compares an object-valued parameter by sorted keys, not by written order', () => {
    const summary = summarizeGraphDiff(
      graphDoc({ nodes: [{ id: 'n', type: 'T', data: { params: { opts: { a: 1, b: 2 } } } }] }),
      graphDoc({ nodes: [{ id: 'n', type: 'T', data: { params: { opts: { b: 2, a: 1 } } } }] }),
      'graph',
    );
    expect(summary.noLogicChange).toBe(true);
  });

  it('does not call a change it cannot name a change of nothing', () => {
    // The description is a real edit with no line kind to carry it. Claiming
    // "no logic change" over it would be a lie the text diff then contradicts.
    const summary = summarizeGraphDiff(
      graphDoc({ nodes: [CSV], description: 'before' }),
      graphDoc({ nodes: [CSV], description: 'after' }),
      'graph',
    );
    expect(summary.lines).toEqual([]);
    expect(summary.noLogicChange).toBe(false);
  });

  it('is unmoved by the order the nodes happen to be written in', () => {
    const summary = summarizeGraphDiff(
      graphDoc({ nodes: [START, CSV, KNN] }),
      graphDoc({ nodes: [KNN, START, CSV] }),
      'graph',
    );
    expect(summary.noLogicChange).toBe(true);
  });
});

describe('summarizeGraphDiff, presets and subgraphs', () => {
  it('reads a parameter change inside a preset from `params`, not `data.params`', () => {
    const edited = {
      ...PRESET,
      nodes: [
        { id: 'p_lin', type: 'Linear', params: { out_features: 64 } },
        { id: 'p_act', type: 'ReLU', params: {} },
      ],
    };
    const summary = summarizeGraphDiff(
      graphDoc({ presets: [PRESET] }),
      graphDoc({ presets: [edited] }),
      'graph',
    );
    expect(summary.lines).toEqual([
      { kind: 'param', node: 'Dense Block/p_lin', param: 'out_features', from: '32', to: '64' },
    ]);
  });

  it('gives a preset definition that appeared no line, without calling it unchanged', () => {
    // A definition's insides are not nodes on the canvas, exactly as for a
    // subgraph. `canonicalText` still refuses to call the file unchanged.
    const summary = summarizeGraphDiff(
      graphDoc({ presets: [] }),
      graphDoc({ presets: [PRESET] }),
      'graph',
    );
    expect(summary.lines).toEqual([]);
    expect(summary.noLogicChange).toBe(false);
  });

  it('reports one node added when one preset is dropped on the canvas', () => {
    // The commonest preset edit there is, and the whole reason the Presets
    // tab exists. The first instance of a preset takes its DEFINITION into
    // the file with it (`tabStore.getSerializedGraph`), so counting the
    // definition's insides read one dropped block as "7 node(s) added, 5
    // edge(s) added" over an edit that added one node and no edges.
    const instance = { id: 'dense-1', type: 'Dense Block', data: { params: {} } };
    const summary = summarizeGraphDiff(
      graphDoc({ nodes: [START], presets: [] }),
      graphDoc({ nodes: [START, instance], presets: [PRESET] }),
      'graph',
    );
    expect(summary.lines).toEqual([{ kind: 'nodesAdded', count: 1 }]);
  });

  it('reports one node removed when the last instance of a preset is deleted', () => {
    const instance = { id: 'dense-1', type: 'Dense Block', data: { params: {} } };
    const summary = summarizeGraphDiff(
      graphDoc({ nodes: [START, instance], presets: [PRESET] }),
      graphDoc({ nodes: [START], presets: [] }),
      'graph',
    );
    expect(summary.lines).toEqual([{ kind: 'nodesRemoved', count: 1 }]);
  });

  it('treats a missing presets key as no presets, since the editor may omit it', () => {
    const withoutKey = JSON.stringify({ format_version: 1, name: 'Iris', description: '', nodes: [], edges: [] });
    // Read as anything else, an editor that omits the key would look like an
    // edit against one that writes it empty.
    const same = summarizeGraphDiff(withoutKey, graphDoc({ subgraphs: [] }), 'graph');
    expect(same.unparseable).toBe(false);
    expect(same.noLogicChange).toBe(true);

    const added = summarizeGraphDiff(withoutKey, graphDoc({ presets: [PRESET] }), 'graph');
    expect(added.unparseable).toBe(false);
    expect(added.lines).toEqual([]);
    expect(added.noLogicChange).toBe(false);
  });

  it('gives an added or removed subgraph definition no line, without calling it unchanged', () => {
    // A definition's insides are not nodes on the canvas. Counting them made
    // the commonest subgraph edit -- collapsing a selection, the test below --
    // report edges nobody added, and it contradicted the edit-in-place case,
    // which has no line either. `canonicalText` still refuses to call it
    // unchanged, so nothing is claimed falsely.
    const added = summarizeGraphDiff(graphDoc(), graphDoc({ subgraphs: [SUBGRAPH] }), 'graph');
    expect(added.lines).toEqual([]);
    expect(added.noLogicChange).toBe(false);

    const removed = summarizeGraphDiff(graphDoc({ subgraphs: [SUBGRAPH] }), graphDoc(), 'graph');
    expect(removed.lines).toEqual([]);
    expect(removed.noLogicChange).toBe(false);
  });

  it('reports no added edge when a selection is collapsed into a subgraph', () => {
    // Collapsing moves the selected nodes and the edges between them out of
    // the top level and into the new definition, leaving one instance node
    // behind. Counting the definition's insides read that as "2 edges added"
    // over an edit that added none.
    const a = { id: 'a', type: 'CSVReader', data: { params: {} } };
    const b = { id: 'b', type: 'Linear', data: { params: {} } };
    const c = { id: 'c', type: 'ReLU', data: { params: {} } };
    const inner = [
      { id: 'e1', source: 'a', sourceHandle: 'tensor', target: 'b', targetHandle: 'x' },
      { id: 'e2', source: 'b', sourceHandle: 'tensor', target: 'c', targetHandle: 'x' },
    ];
    const summary = summarizeGraphDiff(
      graphDoc({ nodes: [a, b, c], edges: inner }),
      graphDoc({
        nodes: [{ id: 'sg_inst', type: 'Subgraph', data: { params: {} } }],
        subgraphs: [{ ...SUBGRAPH, nodes: [a, b, c], edges: inner }],
      }),
      'graph',
    );
    expect(summary.lines).toEqual([
      { kind: 'nodesAdded', count: 1 },
      { kind: 'nodesRemoved', count: 3 },
      { kind: 'edgesRemoved', count: 2 },
    ]);
  });

  it('says nothing about a subgraph edited in place, but does not call it unchanged', () => {
    // v1 compares subgraph definitions by id only. Reporting no line is a
    // gap; reporting "no logic change" would be wrong.
    const edited = {
      ...SUBGRAPH,
      nodes: [{ id: 'in_a', type: 'Linear', data: { params: { out_features: 16 } } }],
    };
    const summary = summarizeGraphDiff(
      graphDoc({ subgraphs: [SUBGRAPH] }),
      graphDoc({ subgraphs: [edited] }),
      'graph',
    );
    expect(summary.lines).toEqual([]);
    expect(summary.noLogicChange).toBe(false);
  });
});

describe('summarizeGraphDiff, layout files', () => {
  it('counts the nodes whose position moved', () => {
    const summary = summarizeGraphDiff(
      layoutDoc({ positions: { a: { x: 0, y: 0 }, b: { x: 10, y: 10 }, c: { x: 20, y: 20 } } }),
      layoutDoc({ positions: { a: { x: 40, y: 0 }, b: { x: 10, y: 99 }, c: { x: 20, y: 20 } } }),
      'layout',
    );
    expect(summary.lines).toEqual([{ kind: 'positionsMoved', count: 2 }]);
  });

  it('calls a layout file whose positions are identical unchanged', () => {
    const positions = { a: { x: 0, y: 0 } };
    expect(summarizeGraphDiff(layoutDoc({ positions }), layoutDoc({ positions }), 'layout'))
      .toEqual({ lines: [], more: 0, noLogicChange: true, unparseable: false });
  });

  it('does not call a position that appeared or vanished a move', () => {
    // A node added or deleted is already reported by the `.graph.json` half
    // of the pair; counting it again here as "moved" would double-report it
    // under a word that is not true.
    const summary = summarizeGraphDiff(
      layoutDoc({ positions: { a: { x: 0, y: 0 } } }),
      layoutDoc({ positions: { a: { x: 0, y: 0 }, b: { x: 5, y: 5 } } }),
      'layout',
    );
    expect(summary.lines).toEqual([]);
    expect(summary.noLogicChange).toBe(false);
  });

  it('does not call a position that only just appeared a move, whatever its id is', () => {
    // `id in was` is true for `toString` on every object `JSON.parse` builds,
    // so a node with that id counted as moved the moment it appeared -- and
    // the inherited function is not a point, so it never compared equal.
    const summary = summarizeGraphDiff(
      layoutDoc({ positions: { a: { x: 0, y: 0 } } }),
      layoutDoc({ positions: { a: { x: 0, y: 0 }, toString: { x: 5, y: 5 } } }),
      'layout',
    );
    expect(summary.lines).toEqual([]);
    expect(summary.noLogicChange).toBe(false);
  });

  it('has no line for a layout file that was added or deleted, nor calls it unchanged', () => {
    // The empty-lines-with-noLogicChange-false shape: there is no line kind
    // for "the whole file appeared", so Task 3 renders nothing here rather
    // than an empty strip.
    const layout = layoutDoc({ positions: { a: { x: 0, y: 0 } } });
    expect(summarizeGraphDiff(null, layout, 'layout'))
      .toEqual({ lines: [], more: 0, noLogicChange: false, unparseable: false });
    expect(summarizeGraphDiff(layout, null, 'layout'))
      .toEqual({ lines: [], more: 0, noLogicChange: false, unparseable: false });
  });

  it('leaves segment groups and note geometry to the text diff, without lying about them', () => {
    // Ruling for v1: no locale key exists for either, so neither gets a line
    // -- but a change to one still has to defeat "no logic change".
    const moved = summarizeGraphDiff(
      layoutDoc({ segmentGroups: [] }),
      layoutDoc({ segmentGroups: [{ id: 'g1', headNodeId: 'a', tailNodeId: 'b' }] }),
      'layout',
    );
    expect(moved.lines).toEqual([]);
    expect(moved.noLogicChange).toBe(false);

    const resized = summarizeGraphDiff(
      layoutDoc({ notes: { n1: { noteWidth: 200, noteHeight: 100 } } }),
      layoutDoc({ notes: { n1: { noteWidth: 320, noteHeight: 100 } } }),
      'layout',
    );
    expect(resized.lines).toEqual([]);
    expect(resized.noLogicChange).toBe(false);
  });

  it('refuses a layout document that has no positions object', () => {
    expect(summarizeGraphDiff(layoutDoc(), graphDoc({ nodes: [CSV] }), 'layout').unparseable).toBe(true);
  });
});

describe('summarizeGraphDiff, missing and broken sides', () => {
  it('counts everything on the other side when one side does not exist', () => {
    const doc = graphDoc({ nodes: [START, CSV], edges: [TRIGGER_EDGE] });
    expect(summarizeGraphDiff(null, doc, 'graph').lines).toEqual([
      { kind: 'nodesAdded', count: 2 },
      { kind: 'edgesAdded', count: 1 },
    ]);
    expect(summarizeGraphDiff(doc, null, 'graph').lines).toEqual([
      { kind: 'nodesRemoved', count: 2 },
      { kind: 'edgesRemoved', count: 1 },
    ]);
  });

  it('does not call a file that was added or deleted unchanged', () => {
    const summary = summarizeGraphDiff(null, graphDoc(), 'graph');
    expect(summary.lines).toEqual([]);
    expect(summary.noLogicChange).toBe(false);
  });

  it('has nothing to say when neither side exists', () => {
    expect(summarizeGraphDiff(null, null, 'graph'))
      .toEqual({ lines: [], more: 0, noLogicChange: true, unparseable: false });
  });

  it('reports unparseable rather than guessing at broken JSON', () => {
    const summary = summarizeGraphDiff(graphDoc({ nodes: [CSV] }), '{ "nodes": [', 'graph');
    expect(summary).toEqual({ lines: [], more: 0, noLogicChange: false, unparseable: true });
  });

  it('reports unparseable for JSON that is not a graph document', () => {
    for (const text of ['[]', '"a string"', 'null', '42', '{}', '{"nodes": []}', '{"nodes": {}, "edges": []}']) {
      expect(summarizeGraphDiff(graphDoc(), text, 'graph').unparseable).toBe(true);
    }
  });

  it('refuses a graph document whose presets key is present but not a list', () => {
    // Read as "no presets" it would report every preset on the other side as
    // removed, which is a claim the file does not support.
    expect(summarizeGraphDiff(graphDoc(), graphDoc({ presets: {} }), 'graph').unparseable).toBe(true);
  });

  it('ignores a broken side that is absent rather than malformed', () => {
    expect(summarizeGraphDiff(null, graphDoc({ nodes: [CSV] }), 'graph').unparseable).toBe(false);
  });
});

describe('summarizeGraphDiff, the eight-line budget', () => {
  function manyParams(value: number) {
    return graphDoc({
      nodes: ['n1', 'n2', 'n3'].map((id) => ({
        id, type: 'T', data: { params: { a: value, b: value, c: value, d: value } },
      })),
    });
  }

  it('returns at most eight lines and counts the rest', () => {
    const summary = summarizeGraphDiff(manyParams(1), manyParams(2), 'graph');
    expect(summary.lines).toHaveLength(MAX_GRAPH_DIFF_LINES);
    expect(summary.more).toBe(12 - MAX_GRAPH_DIFF_LINES);
    expect(summary.noLogicChange).toBe(false);
  });

  it('spends the budget on the counts first, so a total is never crowded out', () => {
    // Twelve parameter edits plus one node added: "1 node added" is the most
    // informative line on the screen and must not be the one that is cut.
    const summary = summarizeGraphDiff(
      manyParams(1),
      graphDoc({
        nodes: [
          ...['n1', 'n2', 'n3'].map((id) => ({
            id, type: 'T', data: { params: { a: 2, b: 2, c: 2, d: 2 } },
          })),
          KNN,
        ],
      }),
      'graph',
    );
    expect(summary.lines[0]).toEqual({ kind: 'nodesAdded', count: 1 });
    expect(summary.lines.slice(1).every((l) => l.kind === 'param')).toBe(true);
    expect(summary.more).toBe(12 - (MAX_GRAPH_DIFF_LINES - 1));
  });

  it('orders the counts, then the type changes, then the parameters', () => {
    const summary = summarizeGraphDiff(
      graphDoc({ nodes: [START, { id: 'clf', type: 'KNN', data: { params: { k: 1 } } }], edges: [TRIGGER_EDGE] }),
      graphDoc({
        nodes: [CSV, { id: 'clf', type: 'DecisionTree', data: { params: { k: 1 } } }, { id: 'z', type: 'Print', data: { params: { v: 1 } } }],
        edges: [DATA_EDGE],
      }),
      'graph',
    );
    expect(summary.lines.map((l) => l.kind)).toEqual([
      'nodesAdded', 'nodesRemoved', 'edgesAdded', 'edgesRemoved', 'typeChanged',
    ]);
  });

  it('reports more as zero when everything fits', () => {
    expect(summarizeGraphDiff(graphDoc(), graphDoc({ nodes: [CSV] }), 'graph').more).toBe(0);
  });
});
