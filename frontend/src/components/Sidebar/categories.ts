/**
 * Category ordering for the sidebar's Nodes tab (#126): both the node
 * categories it lists and the presets group pinned below them.
 *
 * Presets are filed under the same backend category vocabulary as nodes, so
 * one helper decides for both what order categories appear in and what
 * beginner mode hides. Keeping it in one place is what stops the list and the
 * group under it disagreeing about which categories a beginner sees.
 */

/** Curated teaching order; anything the backend adds that is not listed here
 * follows, sorted alphabetically. */
export const CATEGORY_ORDER = [
  'Control',
  'Data',
  'Classical',
  'IO',
  'CNN',
  'Normalization',
  'RNN',
  'Transformer',
  'LLM',
  'Diffusion',
  'RL',
  'Training',
  'Tensor Operations',
  'Utility',
];

/** The only categories beginner mode leaves visible. */
export const BEGINNER_CATEGORIES = new Set(['Data', 'CNN', 'Training', 'IO']);

/**
 * Put `keys` into display order — curated categories first (in CATEGORY_ORDER),
 * then unknown ones alphabetically — dropping non-beginner categories when
 * `beginnerMode` is on.
 */
export function orderCategories(keys: Iterable<string>, beginnerMode: boolean): string[] {
  const all = new Set(keys);
  return [
    ...CATEGORY_ORDER.filter((c) => all.has(c)),
    ...[...all].filter((c) => !CATEGORY_ORDER.includes(c)).sort(),
  ].filter((c) => !beginnerMode || BEGINNER_CATEGORIES.has(c));
}
