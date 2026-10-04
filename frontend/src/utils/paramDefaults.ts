import type { Node } from '@xyflow/react';
import type { NodeData, NodeDefinition, SubgraphDefinition } from '../types';
import { buildFlowNode } from '.';

/**
 * The params a document left out of a node, filled in from its definition
 * (#556).
 *
 * A palette drop starts a node with every param its definition declares, each
 * at its default (`buildFlowNode`). A document is written by whoever wrote it
 * -- a hand-edited file, a textbook's starter, an older build -- and can leave
 * any of them out. The panel then draws a missing param at its default
 * (`ParamField` reads `value ?? param.default`) while the node holds nothing:
 * a sibling another field reads is not there (SlidingWindow2D's `kernel_size`,
 * without which its Custom grid stays locked), and the run gets no value at
 * all ("preset=Custom requires `weights`"). So every door that turns a
 * document into tab state completes the params the way a drop would have, and
 * a node behaves the same whichever way it arrived.
 *
 * The rules:
 *  - Only a missing param is filled. A stored value, `null`, `0` and `''`
 *    included, is the document's and stays. `undefined` counts as missing: no
 *    file can hold one.
 *  - A SECRET param is never filled. Its value is a key the user types for
 *    this session, and a missing key and an empty one already read the same
 *    everywhere (the node falls back to its environment variable).
 *  - The definition is the node list's (`/api/nodes`), found by the exact type
 *    name the canvas resolves a node by (`resolveSerializedNodes`). A type the
 *    list does not have -- it has not answered yet, or the plugin is missing --
 *    is left as stored, since its defaults are not known. So are notes, preset
 *    cards and block instances, which no list entry names.
 *
 * Both functions return the SAME array, holding the same objects, when nothing
 * was missing: the persistence record cache compares by identity, so a copy
 * would rewrite every tab's record for nothing.
 */

const ORIGIN = { x: 0, y: 0 };

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function hasEntries(definitions: readonly NodeDefinition[]): boolean {
  return Array.isArray(definitions) && definitions.length > 0;
}

/**
 * What a palette drop of `definition` starts each param at, less what a fill
 * must not write: a SECRET param, and one whose definition states no default.
 *
 * Read off `buildFlowNode` itself, so the drop and the fill cannot disagree. A
 * list entry the palette could not drop either has nothing to give.
 */
function droppedDefaults(definition: NodeDefinition): [string, unknown][] {
  if (!Array.isArray(definition.params)) return [];
  let dropped: Record<string, unknown>;
  try {
    dropped = buildFlowNode(definition, ORIGIN).data.params;
  } catch {
    return [];
  }
  const secret = new Set(
    definition.params.flatMap((p) => (p?.param_type === 'secret' ? [p.name] : [])),
  );
  return Object.entries(dropped).filter(
    ([name, value]) => value !== undefined && !secret.has(name),
  );
}

/** The defaults to fill a node of each type with, worked out once per type. */
function defaultsByType(
  definitions: readonly NodeDefinition[],
): (type: unknown) => [string, unknown][] {
  // Later entries win, as in `resolveSerializedNodes`'s map.
  const byName = new Map<string, NodeDefinition>();
  for (const definition of definitions) {
    if (typeof definition?.node_name === 'string') byName.set(definition.node_name, definition);
  }
  const memo = new Map<string, [string, unknown][]>();
  return (type) => {
    if (typeof type !== 'string') return [];
    let defaults = memo.get(type);
    if (defaults === undefined) {
      const definition = byName.get(type);
      defaults = definition ? droppedDefaults(definition) : [];
      memo.set(type, defaults);
    }
    return defaults;
  };
}

/**
 * `params` with each missing default added, or null when nothing was missing.
 * A params map that is not an object is left alone: that is a malformed file,
 * and the server is the one to refuse it.
 */
function completedParams(
  params: unknown,
  defaults: [string, unknown][],
): Record<string, unknown> | null {
  if (!defaults.length) return null;
  if (params != null && !isPlainRecord(params)) return null;
  const stored = params ?? {};
  let filled: Record<string, unknown> | null = null;
  for (const [name, value] of defaults) {
    if (Object.prototype.hasOwnProperty.call(stored, name) && stored[name] !== undefined) continue;
    if (filled === null) filled = { ...stored };
    filled[name] = value;
  }
  return filled;
}

/** Canvas nodes with the params their definitions have defaults for. */
export function withParamDefaults(
  nodes: Node<NodeData>[],
  definitions: readonly NodeDefinition[],
): Node<NodeData>[] {
  // Read, never trusted: a restored record or a test double can lack the list.
  if (!Array.isArray(nodes) || !nodes.length || !hasEntries(definitions)) return nodes;
  const defaultsOf = defaultsByType(definitions);
  let changed = false;
  const next = nodes.map((node) => {
    const data = node?.data;
    if (!data) return node;
    const params = completedParams(data.params, defaultsOf(data.type));
    if (params === null) return node;
    changed = true;
    return { ...node, data: { ...data, params } };
  });
  return changed ? next : nodes;
}

/**
 * Block definitions whose inner nodes have the params their definitions have
 * defaults for. An inner node is stored in the serialized shape, its type at
 * the top and its params under `data`.
 */
export function withSubgraphParamDefaults(
  subgraphs: SubgraphDefinition[],
  definitions: readonly NodeDefinition[],
): SubgraphDefinition[] {
  if (!Array.isArray(subgraphs) || !subgraphs.length || !hasEntries(definitions)) return subgraphs;
  const defaultsOf = defaultsByType(definitions);
  let listChanged = false;
  const next = subgraphs.map((definition) => {
    if (!Array.isArray(definition?.nodes)) return definition;
    let changed = false;
    const nodes = definition.nodes.map((raw: unknown) => {
      if (!isPlainRecord(raw) || !isPlainRecord(raw.data)) return raw;
      const params = completedParams(raw.data.params, defaultsOf(raw.type));
      if (params === null) return raw;
      changed = true;
      return { ...raw, data: { ...raw.data, params } };
    });
    if (!changed) return definition;
    listChanged = true;
    return { ...definition, nodes };
  });
  return listChanged ? next : subgraphs;
}
