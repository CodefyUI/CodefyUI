/**
 * Run's validation toasts: what the server's check found, in the user's
 * language, naming each node by the title its card shows, with a Show button
 * that selects the node and brings it into view.
 *
 * The server tags each finding with a code, the node it is about and the
 * values its sentence names (`POST /api/graph/validate` -> `issues`). The
 * words come from `graphValidation.<code>`. A finding with no code, or with
 * one this build has no words for, shows the server's English with every
 * node id in it swapped for the node's title. Codes come from the server,
 * never from a pattern over its English: `errorMessages.ts` documents what
 * that costs.
 *
 * The toasts are one set. Each validation replaces the last set instead of
 * stacking another on top, and `useGraphExecution` takes it down at the next
 * Run and on a tab switch: error toasts never time out, so nothing else
 * would.
 */
import type { Node } from '@xyflow/react';
import { validateGraph, type ValidationIssue } from '../api/rest';
import { useI18n, type TranslationKey } from '../i18n';
import { flushSubgraphEditing, useTabStore, type TabState } from '../store/tabStore';
import { useToastStore } from '../store/toastStore';
import { useUIStore } from '../store/uiStore';
import type { NodeData } from '../types';
import { nodesBoundingBox } from './autoLayout';

type Translate = (key: TranslationKey, vars?: Record<string, string | number>) => string;
type CanvasNode = Node<NodeData>;

/** Problems that get a toast each; the rest are counted in one more. */
const SHOWN = 3;

/**
 * The words for each code the server sends. A Map, so a code that happens to
 * be the name of an Object property finds nothing. `no_entry_points` reuses
 * the sentence the client's own check raises for the same fault.
 */
const ISSUE_KEYS = new Map<string, TranslationKey>([
  ['missing_input', 'graphValidation.missing_input'],
  ['param_not_number', 'graphValidation.param_not_number'],
  ['param_below_min', 'graphValidation.param_below_min'],
  ['param_above_max', 'graphValidation.param_above_max'],
  ['unknown_node_type', 'graphValidation.unknown_node_type'],
  ['unknown_preset', 'graphValidation.unknown_preset'],
  ['invalid_output_port', 'graphValidation.invalid_output_port'],
  ['invalid_input_port', 'graphValidation.invalid_input_port'],
  ['type_mismatch', 'graphValidation.type_mismatch'],
  ['cycle', 'graphValidation.cycle'],
  ['preset_input_not_exposed', 'graphValidation.preset_input_not_exposed'],
  ['preset_output_not_exposed', 'graphValidation.preset_output_not_exposed'],
  ['preset_triggered_empty', 'graphValidation.preset_triggered_empty'],
  ['preset_triggered_all_fed', 'graphValidation.preset_triggered_all_fed'],
  ['subgraph_triggered_empty', 'graphValidation.subgraph_triggered_empty'],
  ['trigger_source_missing', 'graphValidation.trigger_source_missing'],
  ['trigger_target_missing', 'graphValidation.trigger_target_missing'],
  ['switch_selector_out_of_range', 'graphValidation.switch_selector_out_of_range'],
  ['switch_selected_unwired', 'graphValidation.switch_selected_unwired'],
  ['switch_input_types_differ', 'graphValidation.switch_input_types_differ'],
  ['no_entry_points', 'execution.error.noEntryPoints'],
]);

/** Params that hold a node id, said as that node's title. */
const NODE_PARAMS = ['source', 'target', 'cause'] as const;

/** Params that hold a parameter's value rather than a name. */
const VALUE_PARAMS = new Set(['value', 'min', 'max']);

/**
 * Ids of the toasts the last validation raised.
 *
 * Module scope, like Toolbar's path warning: the hook that takes them down
 * lives in the toolbar, which unmounts with the last tab.
 */
let raised: string[] = [];

/**
 * The node on the canvas an id from the server stands for, or null.
 *
 * The server checks the graph with blocks inlined, so a node inside a block
 * comes back as `<block>/<inner>` (backend `SUBGRAPH_SEPARATOR`) and one
 * inside a preset card as `<card>__<inner>`; the canvas shows only the block
 * or the card.
 */
export function canvasNodeFor(
  nodeId: string | null | undefined,
  nodes: readonly CanvasNode[],
): CanvasNode | null {
  if (!nodeId) return null;
  const byId = (id: string) => nodes.find((candidate) => candidate.id === id) ?? null;
  const exact = byId(nodeId);
  if (exact) return exact;
  const slash = nodeId.indexOf('/');
  if (slash > 0) {
    const block = byId(nodeId.slice(0, slash));
    if (block) return block;
  }
  const dunder = nodeId.indexOf('__');
  return dunder > 0 ? byId(nodeId.slice(0, dunder)) : null;
}

/** The title the card shows, else its type, else its id. */
export function nodeName(node: CanvasNode): string {
  return node.data?.label || node.data?.type || node.id;
}

/**
 * The canvas on screen, as the server's ids name it: `''` at the top level,
 * else the cards entered, outermost first, each followed by the engine's `/`
 * (the rule `runNodePrefix` in portCaptures follows). A finding from a
 * run from inside a block, or an Export from there, names the open block's
 * nodes behind it.
 */
function levelOf(tab: Pick<TabState, 'subgraphStack'>): string {
  return (tab.subgraphStack ?? []).map((frame) => `${frame.instanceId}/`).join('');
}

/**
 * A node id said as the node's title. An id nothing on the canvas stands for
 * -- a node the graph does not have -- is shortened to its first 8
 * characters, as the run log's node badge shortens one.
 */
function nameFor(id: unknown, nodes: readonly CanvasNode[]): string {
  if (typeof id !== 'string') return valueText(id);
  const found = canvasNodeFor(id, nodes);
  return found ? nodeName(found) : id.slice(0, 8);
}

/**
 * A parameter's value as text: a number as written, anything else as JSON,
 * so a string reads quoted and a cleared box (NaN, which JSON carries as
 * null) reads null.
 */
function valueText(value: unknown): string {
  if (typeof value === 'number') return String(value);
  return JSON.stringify(value) ?? 'null';
}

/** A closed path of names, a name repeated in a row said once. */
function collapseRepeats(names: string[]): string[] {
  return names.filter((name, index) => index === 0 || name !== names[index - 1]);
}

function issueVars(issue: ValidationIssue, nodes: readonly CanvasNode[]): Record<string, string> {
  const params = issue.params ?? {};
  const vars: Record<string, string> = {};
  for (const [key, value] of Object.entries(params)) {
    vars[key] = VALUE_PARAMS.has(key) ? valueText(value) : String(value);
  }
  for (const key of NODE_PARAMS) {
    if (key in params) vars[key] = nameFor(params[key], nodes);
  }
  if (Array.isArray(params.path)) {
    vars.path = collapseRepeats(params.path.map((id) => nameFor(id, nodes))).join(' -> ');
  }
  if (typeof issue.node_id === 'string') vars.node = nameFor(issue.node_id, nodes);
  return vars;
}

function keyFor(issue: ValidationIssue): TranslationKey | null {
  const key = issue.code ? ISSUE_KEYS.get(issue.code) : undefined;
  if (!key) return null;
  // A port a bypass left empty says so, or it reads as a wire the user forgot.
  if (issue.code === 'missing_input' && issue.params?.cause != null) {
    return 'graphValidation.missing_input.bypassed';
  }
  return key;
}

/**
 * Where the server's sentences put a node id: right after the word "node",
 * or in single quotes. An id is a whole run of letters, digits, `_`, `-` and
 * `/`, which covers `<block>/<inner>` and `<card>__<inner>` too.
 */
const ID_POSITION = /(\b[Nn]ode\s+)([\w\-/]+)|'([\w\-/]+)'/g;

/**
 * The server's sentence with each node id in it said as the node's title.
 * Only ids in id positions are swapped: hand-written and example files use
 * word ids (`input`, `start`, `print`), and the same words in the prose
 * around them must stay as they are.
 */
function withNodeNames(message: string, nodes: readonly CanvasNode[]): string {
  return message.replace(
    ID_POSITION,
    (whole: string, prefix?: string, afterNode?: string, quoted?: string) => {
      const found = canvasNodeFor(afterNode ?? quoted, nodes);
      if (!found) return whole;
      return afterNode !== undefined ? `${prefix}${nodeName(found)}` : `'${nodeName(found)}'`;
    },
  );
}

/** One finding as the toast says it. `nodes` are the tab's top level. */
export function issueText(
  issue: ValidationIssue,
  nodes: readonly CanvasNode[],
  t: Translate,
): string {
  const key = keyFor(issue);
  if (key) {
    const vars = issueVars(issue, nodes);
    // Every slot in the words has to be filled, or the toast would show a raw
    // `{port}`: a server whose params do not fit this build's words gets its
    // own sentence instead.
    const slots = t(key).match(/\{[A-Za-z0-9_]+\}/g) ?? [];
    if (slots.every((slot) => slot.slice(1, -1) in vars)) return t(key, vars);
  }
  return withNodeNames(issue.message, nodes);
}

/** What an older server, which sends `errors` alone, found. */
export function issuesFromErrors(errors: readonly string[] | undefined): ValidationIssue[] {
  return (errors ?? []).map((message) => ({ message, code: null, node_id: null, params: {} }));
}

/**
 * The nodes the server's ids name: the tab's top level, also while a block is
 * open. Taken from a flush of the open levels rather than `subgraphStack[0]`,
 * whose cards keep the names they had on entry: a block renamed inside is
 * named as its live definition is (#620).
 */
function topLevelNodes(tab: TabState): CanvasNode[] {
  return flushSubgraphEditing(tab).nodes;
}

/**
 * Select a node on the active tab's canvas and bring it into view. `id` is the
 * server's: on a run from inside a block, or an Export from there, the open
 * block's node behind the cards entered (`levelOf`). Nothing when the node is
 * gone, or once the canvas on screen is another level.
 */
export function focusNode(id: string): void {
  const store = useTabStore.getState();
  const tab = store.tabs.find((candidate) => candidate.id === store.activeTabId);
  // On another level the same canvas id can name another node, so the id is
  // matched with the level on screen in front of it, as a finding from a
  // run from inside a block names it.
  const level = tab ? levelOf(tab) : '';
  const target = tab?.nodes.find((candidate) => level + candidate.id === id);
  if (target === undefined) return;
  store.selectNodeExclusively(target.id);
  // Centred rather than zoomed in hard: FlowCanvas inflates a box this small.
  const bounds = nodesBoundingBox([target]);
  if (bounds) useUIStore.getState().requestLayoutFit(store.activeTabId, bounds);
}

/** Take down the toasts the last validation raised. */
export function dismissValidationToasts(): void {
  if (raised.length === 0) return;
  const { removeToast } = useToastStore.getState();
  for (const id of raised) removeToast(id);
  raised = [];
}

/**
 * Show what the server's check found on tab `tabId`, in place of the last
 * validation's toasts: at most `SHOWN` problems, one toast each, and one more
 * toast counting the rest.
 */
export function showValidationIssues(tabId: string, issues: readonly ValidationIssue[]): void {
  dismissValidationToasts();
  const { tabs, activeTabId } = useTabStore.getState();
  // The check answers after a round trip. A user who has gone to another tab
  // by then would read these as about the tab in front.
  if (activeTabId !== tabId) return;
  const tab = tabs.find((candidate) => candidate.id === tabId);
  if (!tab) return;
  const { t } = useI18n.getState();
  const named = topLevelNodes(tab);
  // Show selects a node of the canvas on screen. On a run from inside a block,
  // or an Export from there, those are the open block's nodes, which the
  // server names behind the cards entered; a finding elsewhere offers no Show.
  // The node is kept under the server's id, so Show can tell the level.
  const level = levelOf(tab);
  const onScreen = (id: string | null | undefined): CanvasNode | null => {
    const found = id?.startsWith(level)
      ? canvasNodeFor(id.slice(level.length), tab.nodes)
      : null;
    return found && { ...found, id: level + found.id };
  };

  const seen = new Set<string>();
  const problems: { text: string; target: CanvasNode | null }[] = [];
  for (const finding of issues) {
    const text = issueText(finding, named, t);
    // On the canvas on screen, the open block's on a run from inside a block.
    const target = onScreen(finding.node_id);
    // Two edges into one bad port are one problem on one node.
    const key = `${target?.id ?? ''}\n${text}`;
    if (seen.has(key)) continue;
    seen.add(key);
    problems.push({ text, target });
  }

  // In the server's order, as every other run of toasts is raised. The
  // container stacks upward from its corner, so the count of the rest,
  // raised last, sits on top.
  const { addToast } = useToastStore.getState();
  for (const { text, target } of problems.slice(0, SHOWN)) {
    raised.push(
      addToast(
        text,
        'error',
        target
          ? { action: { label: t('graphValidation.show'), onClick: () => focusNode(target.id) } }
          : undefined,
      ),
    );
  }
  if (problems.length > SHOWN) {
    raised.push(addToast(t('graphValidation.more', { count: problems.length - SHOWN }), 'error'));
  }
}

/** A refusal Run makes before asking the server, in the same set. */
export function showValidationError(message: string): void {
  dismissValidationToasts();
  raised.push(useToastStore.getState().addToast(message, 'error'));
}

/**
 * Say why the server refused tab `tabId`'s graph outside a Run (Export as
 * Python). Such a refusal is the server's English, node ids and all, so the
 * check Run makes is asked about the same graph and its findings are shown
 * the way Run shows them. `refusal` is shown instead only when that check
 * finds nothing or cannot be reached. Either way the toasts join the set, so
 * the next Run or export and a tab switch take them down; and nothing is
 * shown once another tab is in front.
 */
export async function showGraphRefusal(
  tabId: string,
  graph: { nodes: unknown[]; edges: unknown[]; presets?: unknown[]; subgraphs?: unknown[] },
  refusal: string,
): Promise<void> {
  let validation: Awaited<ReturnType<typeof validateGraph>> | null = null;
  try {
    validation = await validateGraph(graph.nodes, graph.edges, graph.presets, graph.subgraphs);
  } catch {
    // Unreachable: the refusal is all there is to say.
  }
  if (useTabStore.getState().activeTabId !== tabId) return;
  const issues = validation?.valid === false
    ? validation.issues ?? issuesFromErrors(validation.errors)
    : [];
  if (issues.length > 0) showValidationIssues(tabId, issues);
  else showValidationError(refusal);
}
