/**
 * Which of a preset's tags is its difficulty, and the words for each (#623).
 *
 * One rule for the Nodes tab's badge and for the node search, so the word on
 * a badge always finds its preset. A map rather than a
 * `palette.preset.difficulty.${tag}` template: `t()` takes a typed key, and a
 * preset's tags are free text.
 */
export const DIFFICULTY_LABEL_KEYS = {
  beginner: 'palette.preset.difficulty.beginner',
  intermediate: 'palette.preset.difficulty.intermediate',
  advanced: 'palette.preset.difficulty.advanced',
} as const;

export type Difficulty = keyof typeof DIFFICULTY_LABEL_KEYS;

/** Own keys only: `in` also answers yes for `constructor`, which every object has. */
export function isDifficulty(tag: unknown): tag is Difficulty {
  return typeof tag === 'string' && Object.prototype.hasOwnProperty.call(DIFFICULTY_LABEL_KEYS, tag);
}
