import type { TranslationKey } from '../i18n';

/**
 * Display labels for the node categories the backend sends (#684).
 *
 * The category stays the identifier everywhere -- the palette groups on it,
 * the colours and the beginner-mode list key on it, and a saved graph carries
 * it -- and only what a reader sees goes through the locale. Acronyms and
 * proper names (CNN, RNN, Transformer, LLM, Diffusion, RL, IO, VLA) read the
 * same in both locales, so they have no key and show as sent, and so does a
 * category a plugin brings that this table does not know.
 */
const NODE_CATEGORY_KEYS: Record<string, TranslationKey> = {
  Classical: 'palette.category.classical',
  Control: 'palette.category.control',
  Data: 'palette.category.data',
  'Data Flow': 'palette.category.dataFlow',
  Normalization: 'palette.category.normalization',
  'Tensor Operations': 'palette.category.tensorOps',
  Training: 'palette.category.training',
  Utility: 'palette.category.utility',
};

/** The label a node category shows in the UI language. */
export function nodeCategoryLabel(category: string, t: (key: TranslationKey) => string): string {
  return Object.prototype.hasOwnProperty.call(NODE_CATEGORY_KEYS, category)
    ? t(NODE_CATEGORY_KEYS[category])
    : category;
}
