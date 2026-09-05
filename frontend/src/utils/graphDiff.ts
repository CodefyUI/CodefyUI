/**
 * What actually changed in a saved graph, read out of the two sides of a diff.
 *
 * A project stores each graph as a PAIR (`backend/app/core/project.py:74`):
 * `graphs/<name>.graph.json` holds the logic and `layout/<name>.layout.json`
 * holds the geometry. Both are JSON, so the text diff of either is a wall of
 * braces in which "I changed k from 5 to 7" is invisible. This module reads
 * the raw text of both sides and answers with the handful of sentences a
 * reviewer wants above that wall.
 *
 * The text is parsed here rather than through `utils/openSavedGraph.ts`: that
 * reader merges unknown presets into the node-definition store and can
 * auto-layout a document, neither of which may happen because the tab is
 * reading an old commit.
 *
 * What is compared, and what is deliberately ignored:
 *
 * - Nodes by `id`. A logic node is `{id, type, data}` with its parameters at
 *   `data.params` -- never at `node.params`, which is the shape a PRESET's
 *   inner nodes use (`frontend/src/types/index.ts:151`). A node is named by
 *   `data.label` when the user has renamed it, otherwise by its id.
 * - Edges by `${source}:${sourceHandle}->${target}:${targetHandle}`, with a
 *   missing handle read as empty. A trigger edge carries NO `targetHandle`
 *   (`examples/Classical/Iris-Sklearn-KNN/graph.json`), so without that
 *   normalization every trigger edge in the file would read as removed and
 *   re-added. The edge `id` is ignored: copy and paste regenerates it, as
 *   `docs/docs/usage/version-control-graphs.md:185` already documents.
 * - Presets by `preset_name`. A preset that changed contributes its own node
 *   and edge differences to the same counts the top level uses, and one that
 *   appeared or vanished contributes everything inside it.
 * - Subgraph definitions by `id`, added or removed only. One edited in place
 *   gets no line -- there is no key for it -- but it does defeat
 *   `noLogicChange`, so the summary never claims nothing happened.
 * - Layout: a node in `positions` whose `{x, y}` changed counts toward
 *   `positionsMoved`. A position that appeared or vanished is NOT a move: the
 *   node was added or deleted, and the `.graph.json` half of the pair reports
 *   that already.
 *
 * RULING for v1: segment-group and note-geometry changes are NOT summarised.
 * Both live in the layout file (`project.py:144`, `:66`) and neither has a
 * locale key, so neither gets a line -- the text diff is what shows them. They
 * still defeat `noLogicChange`, because saying "nothing changed" over a change
 * the reader can see in the diff below is worse than saying nothing at all.
 */

/** Which half of a saved pair a path is, or null for a file with no summary. */
export type GraphDiffKind = 'graph' | 'layout';

/** The line kinds that carry a bare number. */
export type GraphDiffCountKind =
  | 'nodesAdded'
  | 'nodesRemoved'
  | 'edgesAdded'
  | 'edgesRemoved'
  | 'positionsMoved';

/**
 * One sentence of the summary, as data. The component turns each into a
 * `git.gdiff.*` locale key with these fields as its placeholders; nothing in
 * this module knows a word of English.
 */
export type GraphDiffLine =
  | { kind: GraphDiffCountKind; count: number }
  | { kind: 'typeChanged'; node: string; from: string; to: string }
  | { kind: 'param'; node: string; param: string; from: string; to: string };

export interface GraphDiffSummary {
  /** At most `MAX_GRAPH_DIFF_LINES`, counts first. */
  lines: GraphDiffLine[];
  /** How many lines did not fit -- `git.gdiff.more`. */
  more: number;
  /**
   * Both sides say the same thing and the difference is text only: key order,
   * whitespace, an array written in another order, a regenerated edge id.
   */
  noLogicChange: boolean;
  /** A side that exists is not JSON, or is not a document of this kind. */
  unparseable: boolean;
}

/** The summary never runs past eight lines; the rest are counted in `more`. */
export const MAX_GRAPH_DIFF_LINES = 8;
/** Parameter values are clipped to this many code points in `from` / `to`. */
export const MAX_GRAPH_DIFF_VALUE = 40;

/**
 * Which kind of summary a path gets, by suffix, or null for no summary.
 *
 * Non-project mode saves a single `<name>.json` whose nodes still carry their
 * positions; it is deliberately not summarised, which is also why
 * `positionsMoved` can only ever appear on a `*.layout.json` diff.
 */
export function graphDiffKind(path: string): GraphDiffKind | null {
  const lower = path.toLowerCase();
  if (lower.endsWith('.graph.json')) return 'graph';
  if (lower.endsWith('.layout.json')) return 'layout';
  return null;
}

/**
 * Summarise one file's two sides.
 *
 * A `null` side is a file that does not exist at that ref -- an addition or a
 * deletion -- and everything on the other side then counts as added or
 * removed. Two `null` sides is a request about nothing, and answers with
 * nothing.
 */
export function summarizeGraphDiff(
  oldText: string | null,
  newText: string | null,
  kind: GraphDiffKind,
): GraphDiffSummary {
  if (oldText === null && newText === null) {
    return { lines: [], more: 0, noLogicChange: true, unparseable: false };
  }
  const before = oldText === null ? null : readDocument(oldText, kind);
  const after = newText === null ? null : readDocument(newText, kind);
  if ((oldText !== null && before === null) || (newText !== null && after === null)) {
    return { lines: [], more: 0, noLogicChange: false, unparseable: true };
  }

  const empty = kind === 'graph' ? EMPTY_GRAPH : EMPTY_LAYOUT;
  const tally = emptyTally();
  if (kind === 'graph') compareGraphs(before ?? empty, after ?? empty, tally);
  else comparePositions(before ?? empty, after ?? empty, tally);

  const all = toLines(tally);
  return {
    lines: all.slice(0, MAX_GRAPH_DIFF_LINES),
    more: Math.max(0, all.length - MAX_GRAPH_DIFF_LINES),
    // A side that does not exist is not "the same": the file itself is new or
    // gone, which is a change even when it holds nothing.
    noLogicChange:
      all.length === 0
      && before !== null
      && after !== null
      && canonicalText(before, kind) === canonicalText(after, kind),
    unparseable: false,
  };
}

// --- reading -------------------------------------------------------------

type Doc = Record<string, unknown>;

const EMPTY_GRAPH: Doc = { nodes: [], edges: [], presets: [], subgraphs: [] };
const EMPTY_LAYOUT: Doc = { positions: {} };

/**
 * Parse one side, or null when it is not a document of this kind.
 *
 * `nodes` and `edges` are required on a logic file and `positions` on a
 * layout one. Read a missing `edges` as "no edges" and every edge on the
 * other side would be reported as removed -- a claim the file does not
 * support. `presets` and `subgraphs` ARE optional (`GraphSaveData` marks them
 * so), and absent there really does mean none.
 */
function readDocument(text: string, kind: GraphDiffKind): Doc | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isPlainObject(parsed)) return null;
  if (kind === 'layout') return isPlainObject(parsed.positions) ? parsed : null;
  if (!Array.isArray(parsed.nodes) || !Array.isArray(parsed.edges)) return null;
  for (const optional of ['presets', 'subgraphs'] as const) {
    const value = parsed[optional];
    if (value !== undefined && !Array.isArray(value)) return null;
  }
  return parsed;
}

// --- comparing -----------------------------------------------------------

interface Tally {
  nodesAdded: number;
  nodesRemoved: number;
  edgesAdded: number;
  edgesRemoved: number;
  positionsMoved: number;
  typeChanges: { node: string; from: string; to: string }[];
  paramChanges: { node: string; param: string; from: string; to: string }[];
}

function emptyTally(): Tally {
  return {
    nodesAdded: 0,
    nodesRemoved: 0,
    edgesAdded: 0,
    edgesRemoved: 0,
    positionsMoved: 0,
    typeChanges: [],
    paramChanges: [],
  };
}

/** A node reduced to the three things the summary can talk about. */
interface NodeFacts {
  /** What the line calls it: the label if renamed, else the id. */
  name: string;
  type: string;
  params: Doc;
}

function compareGraphs(before: Doc, after: Doc, tally: Tally): void {
  compareNodes(graphNodes(before.nodes), graphNodes(after.nodes), tally);
  compareEdges(edgeCounts(before.edges), edgeCounts(after.edges), tally);

  const oldPresets = byKey(before.presets, 'preset_name');
  const newPresets = byKey(after.presets, 'preset_name');
  for (const [name, preset] of newPresets) {
    const was = oldPresets.get(name);
    if (was === undefined) {
      tally.nodesAdded += asList(preset.nodes).length;
      tally.edgesAdded += asList(preset.edges).length;
      continue;
    }
    compareNodes(presetNodes(was, name), presetNodes(preset, name), tally);
    compareEdges(edgeCounts(was.edges), edgeCounts(preset.edges), tally);
  }
  for (const [name, preset] of oldPresets) {
    if (newPresets.has(name)) continue;
    tally.nodesRemoved += asList(preset.nodes).length;
    tally.edgesRemoved += asList(preset.edges).length;
  }

  // Definitions, added or removed only. One edited in place has no line kind
  // to carry it and is left to `canonicalText`, which will not let the
  // summary call it unchanged.
  const oldSubs = byKey(before.subgraphs, 'id');
  const newSubs = byKey(after.subgraphs, 'id');
  for (const [id, definition] of newSubs) {
    if (oldSubs.has(id)) continue;
    tally.nodesAdded += asList(definition.nodes).length;
    tally.edgesAdded += asList(definition.edges).length;
  }
  for (const [id, definition] of oldSubs) {
    if (newSubs.has(id)) continue;
    tally.nodesRemoved += asList(definition.nodes).length;
    tally.edgesRemoved += asList(definition.edges).length;
  }
}

function compareNodes(before: Map<string, NodeFacts>, after: Map<string, NodeFacts>, tally: Tally): void {
  for (const [id, facts] of after) {
    const was = before.get(id);
    if (was === undefined) {
      tally.nodesAdded += 1;
      continue;
    }
    if (was.type !== facts.type) {
      // No parameter lines under a type change: two node types have two
      // parameter sets, so every key would read as added or removed -- noise
      // on top of the one line that explains the change.
      tally.typeChanges.push({ node: facts.name, from: was.type, to: facts.type });
      continue;
    }
    compareParams(facts.name, was.params, facts.params, tally);
  }
  for (const id of before.keys()) {
    if (!after.has(id)) tally.nodesRemoved += 1;
  }
}

function compareParams(node: string, before: Doc, after: Doc, tally: Tally): void {
  const keys = Object.keys(after);
  for (const key of Object.keys(before)) {
    if (!(key in after)) keys.push(key);
  }
  for (const param of keys) {
    // Compared at full length and clipped only for the line: two values that
    // differ past the fortieth character are still two values.
    const from = valueText(before, param);
    const to = valueText(after, param);
    if (from === to) continue;
    tally.paramChanges.push({ node, param, from: clipValue(from), to: clipValue(to) });
  }
}

function compareEdges(before: Map<string, number>, after: Map<string, number>, tally: Tally): void {
  for (const key of new Set([...before.keys(), ...after.keys()])) {
    // Counted, not set-compared: two edges can legitimately share a key, and
    // a Set would report the second one as neither added nor removed.
    const delta = (after.get(key) ?? 0) - (before.get(key) ?? 0);
    if (delta > 0) tally.edgesAdded += delta;
    else if (delta < 0) tally.edgesRemoved -= delta;
  }
}

function comparePositions(before: Doc, after: Doc, tally: Tally): void {
  const was = isPlainObject(before.positions) ? before.positions : {};
  const now = isPlainObject(after.positions) ? after.positions : {};
  for (const [id, point] of Object.entries(now)) {
    if (!(id in was)) continue;
    if (!samePoint(was[id], point)) tally.positionsMoved += 1;
  }
}

function samePoint(a: unknown, b: unknown): boolean {
  if (!isPlainObject(a) || !isPlainObject(b)) return stableText(a) === stableText(b);
  return a.x === b.x && a.y === b.y;
}

// --- lines ---------------------------------------------------------------

const COUNT_ORDER: GraphDiffCountKind[] = [
  'nodesAdded',
  'nodesRemoved',
  'edgesAdded',
  'edgesRemoved',
  'positionsMoved',
];

/**
 * Flatten the tally, counts first.
 *
 * There are at most five count lines, so ordering them ahead of the per-node
 * detail guarantees a total never loses its place to the eighth of thirty
 * parameter edits -- and a total is the more informative line.
 */
function toLines(tally: Tally): GraphDiffLine[] {
  const lines: GraphDiffLine[] = [];
  for (const kind of COUNT_ORDER) {
    const count = tally[kind];
    if (count > 0) lines.push({ kind, count });
  }
  for (const change of tally.typeChanges) lines.push({ kind: 'typeChanged', ...change });
  for (const change of tally.paramChanges) lines.push({ kind: 'param', ...change });
  return lines;
}

// --- canonical form ------------------------------------------------------

/**
 * The document with the noise the keyed comparison ignores normalised away.
 *
 * Two sides whose canonical text matches differ only in key order, in
 * whitespace, in the order a list happens to be written in, or in an edge id
 * -- none of which is a change to the graph. Anything else surviving into
 * this text is a real difference, which is what stops `noLogicChange` from
 * being claimed over an edit no line kind can describe.
 */
function canonicalText(doc: Doc, kind: GraphDiffKind): string {
  if (kind === 'layout') return stableText(doc);
  return stableText({
    ...doc,
    nodes: sortList(asList(doc.nodes), idOf),
    edges: sortList(asList(doc.edges).map(canonicalEdge), (e) => edgeKey(e) ?? ''),
    presets: sortList(asList(doc.presets).map(canonicalPreset), (p) => keyOf(p, 'preset_name')),
    subgraphs: sortList(asList(doc.subgraphs), idOf),
  });
}

function canonicalPreset(raw: unknown): unknown {
  if (!isPlainObject(raw)) return raw;
  return {
    ...raw,
    nodes: sortList(asList(raw.nodes), idOf),
    edges: sortList(asList(raw.edges).map(canonicalEdge), (e) => edgeKey(e) ?? ''),
  };
}

function canonicalEdge(raw: unknown): unknown {
  if (!isPlainObject(raw)) return raw;
  const copy: Doc = { ...raw };
  // Regenerated by copy and paste, so never a change on its own.
  delete copy.id;
  copy.sourceHandle = asText(raw.sourceHandle);
  copy.targetHandle = asText(raw.targetHandle);
  return copy;
}

function sortList(list: unknown[], keyOfItem: (value: unknown) => string): unknown[] {
  return list
    // The whole item joins the sort key so two entries sharing an id still
    // order deterministically. The separator is written as an escape rather
    // than as the character itself: a NUL byte inside a source file makes
    // grep and ripgrep skip the whole file as binary.
    .map((value) => ({ value, sort: `${keyOfItem(value)}\u0000${stableText(value)}` }))
    .sort((a, b) => (a.sort < b.sort ? -1 : a.sort > b.sort ? 1 : 0))
    .map((entry) => entry.value);
}

// --- shapes --------------------------------------------------------------

function graphNodes(list: unknown): Map<string, NodeFacts> {
  const out = new Map<string, NodeFacts>();
  for (const raw of asList(list)) {
    if (!isPlainObject(raw) || typeof raw.id !== 'string') continue;
    const data = isPlainObject(raw.data) ? raw.data : {};
    const label = typeof data.label === 'string' && data.label !== '' ? data.label : null;
    out.set(raw.id, {
      name: label ?? raw.id,
      type: asText(raw.type),
      params: isPlainObject(data.params) ? data.params : {},
    });
  }
  return out;
}

/**
 * A preset's inner nodes, whose parameters sit at `params` and not under
 * `data`. Named `<preset>/<node>` because a preset's ids live in their own
 * namespace and can collide with the graph's own.
 */
function presetNodes(preset: Doc, presetName: string): Map<string, NodeFacts> {
  const out = new Map<string, NodeFacts>();
  for (const raw of asList(preset.nodes)) {
    if (!isPlainObject(raw) || typeof raw.id !== 'string') continue;
    out.set(raw.id, {
      name: `${presetName}/${raw.id}`,
      type: asText(raw.type),
      params: isPlainObject(raw.params) ? raw.params : {},
    });
  }
  return out;
}

function edgeCounts(list: unknown): Map<string, number> {
  const out = new Map<string, number>();
  for (const raw of asList(list)) {
    const key = edgeKey(raw);
    if (key === null) continue;
    out.set(key, (out.get(key) ?? 0) + 1);
  }
  return out;
}

function edgeKey(raw: unknown): string | null {
  if (!isPlainObject(raw)) return null;
  const source = asText(raw.source);
  const target = asText(raw.target);
  if (source === '' || target === '') return null;
  return `${source}:${asText(raw.sourceHandle)}->${target}:${asText(raw.targetHandle)}`;
}

function byKey(list: unknown, field: 'preset_name' | 'id'): Map<string, Doc> {
  const out = new Map<string, Doc>();
  for (const raw of asList(list)) {
    if (!isPlainObject(raw) || typeof raw[field] !== 'string') continue;
    out.set(raw[field] as string, raw);
  }
  return out;
}

function idOf(value: unknown): string {
  return keyOf(value, 'id');
}

function keyOf(value: unknown, field: string): string {
  return isPlainObject(value) ? asText(value[field]) : '';
}

// --- values --------------------------------------------------------------

/** One parameter as text, or the empty string when the key is not there. */
function valueText(params: Doc, key: string): string {
  return key in params ? stableText(params[key]) : '';
}

function clipValue(text: string): string {
  const points = Array.from(text);
  if (points.length <= MAX_GRAPH_DIFF_VALUE) return text;
  // Sliced by code point: cutting a surrogate pair in half leaves a lone
  // surrogate, which renders as a replacement glyph and reads as data loss.
  return `${points.slice(0, MAX_GRAPH_DIFF_VALUE - 3).join('')}...`;
}

/** JSON with every object key sorted, so written order is never a change. */
function stableText(value: unknown): string {
  return JSON.stringify(sortKeys(value)) ?? 'null';
}

function sortKeys(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortKeys);
  if (!isPlainObject(value)) return value;
  const out: Doc = {};
  for (const key of Object.keys(value).sort()) out[key] = sortKeys(value[key]);
  return out;
}

function isPlainObject(value: unknown): value is Doc {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function asList(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function asText(value: unknown): string {
  return typeof value === 'string' ? value : '';
}
