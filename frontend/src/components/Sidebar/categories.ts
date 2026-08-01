/**
 * Category ordering shared by the sidebar's Nodes and Presets tabs (#126).
 *
 * Both tabs draw from the same backend category vocabulary and must agree on
 * the order they appear in and on what beginner mode hides — before #126 the
 * two were one merged list, so keeping this in one place is what stops them
 * drifting apart.
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
