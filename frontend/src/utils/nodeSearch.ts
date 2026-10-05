import type { NodeDefinition, PresetDefinition } from '../types';
import { pluginNameOf, type PluginIndex } from './provider';

/**
 * What a node search finds and in what order, for both places that search
 * the node library: the sidebar's Nodes tab and the quick search a
 * double-click on the canvas opens.
 *
 * Both used to filter only, so results came out in catalog order: "linear"
 * listed fifteen nodes, most of them because their descriptions mention a
 * linear layer, and the node named Linear came last. Quick search kept the
 * first 20 of such a list, so it could drop the exact match altogether. Both
 * now rank through this module, so they read the same fields and agree on
 * the order.
 *
 * A match is ordered by its tier first:
 *   Exact      the name is the query
 *   Prefix     the name starts with it
 *   Word       it starts at a word inside the name ("split" in TrainTestSplit)
 *   Substring  it is anywhere else in the name
 *   Text       only another text has it (description, details, plugin name)
 * then by the shorter name, then by where the entry stood, so equal matches
 * keep the order they had.
 *
 * A plugin node's name is qualified (`edu:FilterRows`). The part after the
 * last colon counts as a name of its own, so "filterrows" is an exact match.
 */

/** `as const` rather than an enum: the frontend declares none. */
export const MatchTier = { Exact: 0, Prefix: 1, Word: 2, Substring: 3, Text: 4 } as const;
export type MatchTier = (typeof MatchTier)[keyof typeof MatchTier];

/** Where a name splits into words, besides its case changes and digit runs. */
const SEPARATORS = /[:\-_\s]+/;

/**
 * One word: an acronym up to the capital that starts the next word (`LSTM` in
 * `LSTMCell`), a capitalised or lower-case word, or a digit run.
 */
const WORD = /[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+/g;

function wordsOf(name: string): string[] {
  return name
    .split(SEPARATORS)
    .flatMap((part) => part.match(WORD) ?? [])
    .map((word) => word.toLowerCase());
}

/**
 * Whether `query` starts at one of the name's words. The rest of the name is
 * read on from there, so "testsplit" starts at a word of `TrainTestSplit` too.
 */
function startsAtWord(name: string, query: string): boolean {
  const words = wordsOf(name);
  for (let i = 0; i < words.length; i++) {
    if (words.slice(i).join('').startsWith(query)) return true;
  }
  return false;
}

/**
 * How well `query` matches an entry, or null when it does not.
 *
 * `name` is the node or preset name; `texts` are the other strings the search
 * reads (absent and empty ones are skipped). The query is trimmed and matched
 * without regard to case. An empty query is not a search: the callers list
 * everything in their own order without asking, and it gets null here.
 */
export function matchTier(
  name: string,
  texts: readonly (string | null | undefined)[],
  query: string,
): MatchTier | null {
  const q = query.trim().toLowerCase();
  if (!q) return null;
  const full = name.toLowerCase();
  const bare = full.slice(full.lastIndexOf(':') + 1);
  if (full === q || bare === q) return MatchTier.Exact;
  if (full.startsWith(q) || bare.startsWith(q)) return MatchTier.Prefix;
  if (startsAtWord(name, q)) return MatchTier.Word;
  if (full.includes(q)) return MatchTier.Substring;
  if (texts.some((text) => !!text && text.toLowerCase().includes(q))) return MatchTier.Text;
  return null;
}

/** What a match is ordered by. */
export interface MatchRank {
  tier: MatchTier;
  /** The entry's name: of two equal tiers, the shorter name goes first. */
  name: string;
  /** Where the entry stood before ranking: the last tie-break. */
  index: number;
}

/** Sort comparator: better tier, then shorter name, then earlier position. */
export function compareMatches(a: MatchRank, b: MatchRank): number {
  return a.tier - b.tier || a.name.length - b.name.length || a.index - b.index;
}

export interface SearchMatch<T> extends MatchRank {
  item: T;
}

/**
 * The entries `query` matches, best first, each with its tier and its
 * position in `entries`.
 */
export function rankMatches<T>(
  entries: readonly T[],
  nameOf: (entry: T) => string,
  textsOf: (entry: T) => readonly (string | null | undefined)[],
  query: string,
): SearchMatch<T>[] {
  const matches: SearchMatch<T>[] = [];
  entries.forEach((item, index) => {
    const name = nameOf(item);
    const tier = matchTier(name, textsOf(item), query);
    if (tier !== null) matches.push({ item, tier, name, index });
  });
  return matches.sort(compareMatches);
}

/** `useI18n`'s `tn`, narrowed to the two fields a search reads. */
export type TranslateNodeField = (
  nodeName: string,
  field: 'description' | 'details',
  fallback: string,
) => string;

/**
 * The strings a search reads for a node besides its name.
 *
 * `details` as well as the summary: the summary is one line, and the library
 * a node wraps, its caveats and its formula all live in `details`. A search
 * for "sklearn" that stopped matching them would make the node harder to find.
 *
 * The plugin's DISPLAY name: its id already matches through the qualified
 * node name (`edu:FilterRows`), while the name the Plugin Center shows appears
 * in no field of a definition.
 *
 * The UI language's own summary and details, so a Chinese query finds a node
 * by its Chinese description. The English texts stay, so an English query
 * still works in the Chinese UI; node names are never translated. The
 * fallback is empty rather than the English text, which is already listed.
 */
export function nodeSearchTexts(
  definition: NodeDefinition,
  pluginsById: PluginIndex,
  tn: TranslateNodeField,
): (string | null | undefined)[] {
  return [
    definition.description,
    definition.details,
    pluginNameOf(pluginsById, definition.provider),
    tn(definition.node_name, 'description', ''),
    tn(definition.node_name, 'details', ''),
  ];
}

/** The strings a search reads for a preset besides its name. */
export function presetSearchTexts(preset: PresetDefinition): string[] {
  return [preset.description, ...preset.tags];
}
