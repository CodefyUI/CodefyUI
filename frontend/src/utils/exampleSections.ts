import type { ExampleSummary } from '../api/rest';
import type { TranslationKey } from '../i18n';
import en from '../i18n/locales/en';

/**
 * The one grouping every example surface renders (#141).
 *
 * The empty-canvas overlay, the sidebar's Templates tab and the gallery modal
 * used to group three different ways -- a pinned trio plus a curated category
 * order here, a rank-by-category list there -- so the same example sat in a
 * different place depending on where you opened it from. All three now call
 * `groupExamplesBySection`, and what an example claims travels with the
 * example: `graph.json` carries a `gallery` block and `/api/examples/list`
 * serves it as `main_tag`, `tags`, `family` and `order`. The main tags and
 * their order come from `GET /api/examples/tags` (phase 3: practical first,
 * papers and courses after).
 *
 * Nothing here reads the path except to spot a pack, so adding an example is
 * an edit to its own `graph.json` rather than to a table in the frontend.
 */

/**
 * The main tags this build knows, in the order an older server (one that does
 * not answer `GET /api/examples/tags`) is drawn in. On a current server the
 * order and the list come from that endpoint, which defines them once
 * (`backend/app/core/example_tags.py`); this is only the fallback and the
 * id type. Practical categories first, papers and courses after them.
 */
export const KNOWN_MAIN_TAGS = [
  'quickstart',
  'training',
  'llm',
  'data',
  'evaluation',
  'research',
  'deploy',
  'architectures',
  'concepts',
  'papers',
  'courses',
  'plugin',
  'other',
] as const;

export type MainTagKey = (typeof KNOWN_MAIN_TAGS)[number];

/** @deprecated The section key is the main tag now. */
export type ExampleSectionKey = string;

/** One block inside a section. `label === null` means no sub-header. */
export interface ExampleSubgroup<T> {
  /** Unique across every section, so a surface can key its DOM on it. */
  key: string;
  label: string | null;
  items: T[];
}

export interface ExampleSection<T> {
  /** The main tag id. Display it through `mainTagLabel`. */
  key: string;
  subgroups: ExampleSubgroup<T>[];
}

/** A flattened sub-group, in the shape `CategoryList` takes. */
export interface ExampleGroup<T> {
  /** `CategoryGroup.category`: the sub-group's key, not a category name. */
  category: string;
  label: string;
  sectionKey: string;
  items: T[];
}

/** Main tags the server assigns itself and an author cannot declare. */
const FALLBACK_TAGS = new Set(['plugin', 'other']);

/** Main tags whose examples are grouped under one sub-header per pack. */
const PACK_GROUPED = new Set(['plugin', 'courses']);

/** Sub-header order inside `architectures`; any other family sorts after these,
 * alphabetically. */
const FAMILY_ORDER: string[] = ['CNN', 'RNN', 'Transformer', 'Diffusion', 'RL'];

/** Same prefix `utils/provider` parses a `source` with. */
const PLUGIN_PREFIX = 'plugin:';

function hasKey(key: string): key is TranslationKey {
  return Object.prototype.hasOwnProperty.call(en, key);
}

/**
 * The main tag an example is filed under.
 *
 * A current server always sends `main_tag`. An older one sends only
 * `section` (and nothing for a pack), so that is read the way the server
 * reads it now: a known section is the main tag, a pack example with none is
 * `plugin`, and a built-in with none is `other`.
 */
export function mainTagOf(example: ExampleSummary): string {
  const declared = example.main_tag;
  if (typeof declared === 'string' && declared !== '') return declared;
  const section = example.section;
  if (
    typeof section === 'string' &&
    (KNOWN_MAIN_TAGS as readonly string[]).includes(section) &&
    !FALLBACK_TAGS.has(section)
  ) {
    return section;
  }
  return example.path.startsWith(PLUGIN_PREFIX) ? 'plugin' : 'other';
}

/** A main tag's display name; an id this build has no name for shows as is. */
export function mainTagLabel(id: string, t: (key: TranslationKey) => string): string {
  const key = `examples.mainTag.${id}`;
  return hasKey(key) ? t(key) : id;
}

/** The free tags of an example: every tag except its main tag. */
export function freeTagsOf(example: ExampleSummary): string[] {
  const main = mainTagOf(example);
  const tags = Array.isArray(example.tags) ? example.tags : [];
  return tags.filter((tag, index) => typeof tag === 'string' && tag !== main && tags.indexOf(tag) === index);
}

/** A tag's display name; a tag this build has no name for shows as is. */
export function tagLabel(tag: string, t: (key: TranslationKey) => string): string {
  const key = `examples.tag.${tag}`;
  return hasKey(key) ? t(key) : tag;
}

/** What a card prints: at most `max` free tags, and how many more there are. */
export function cardTags(example: ExampleSummary, max = 2): { shown: string[]; more: number } {
  const tags = freeTagsOf(example);
  return { shown: tags.slice(0, max), more: Math.max(0, tags.length - max) };
}

/**
 * The main tags in the order they render: `apiOrder` (from
 * `GET /api/examples/tags`) when it is known, the fallback order otherwise.
 * A main tag neither lists -- one a newer pack brings -- goes after the known
 * ones and before the two fallbacks.
 */
export function mainTagOrder(present: Iterable<string>, apiOrder?: readonly string[] | null): string[] {
  const base: string[] = apiOrder && apiOrder.length > 0 ? [...apiOrder] : [...KNOWN_MAIN_TAGS];
  const seen = new Set(present);
  const ordered = base.filter((id) => seen.has(id) && !FALLBACK_TAGS.has(id));
  const extra = [...seen].filter((id) => !base.includes(id) && !FALLBACK_TAGS.has(id)).sort();
  // The two fallbacks close the list, plugin then other, whatever the order
  // given says about them.
  const tail = ['plugin', 'other'].filter((id) => seen.has(id));
  return [...ordered, ...extra, ...tail];
}

/** The declared family, or null. An empty string is no family, not a blank
 * sub-header. */
function familyOf(example: ExampleSummary): string | null {
  const family = example.family;
  return typeof family === 'string' && family !== '' ? family : null;
}

/** The declared order, or null for "after everything that declared one".
 * `Number.isFinite` and not just a type test, so a NaN sent by a pack sorts
 * like a missing value instead of poisoning the comparator. */
function orderOf(example: ExampleSummary): number | null {
  const order = example.order;
  return typeof order === 'number' && Number.isFinite(order) ? order : null;
}

/** Ascending, nulls last, ties keeping the server's order -- `sort` is stable,
 * so returning 0 for a tie is what preserves it. */
function byOrder(a: ExampleSummary, b: ExampleSummary): number {
  const left = orderOf(a);
  const right = orderOf(b);
  if (left === right) return 0;
  if (left === null) return 1;
  if (right === null) return -1;
  return left - right;
}

function byFamily(a: string, b: string): number {
  const left = FAMILY_ORDER.indexOf(a);
  const right = FAMILY_ORDER.indexOf(b);
  if (left === -1 && right === -1) return a.localeCompare(b);
  if (left === -1) return 1;
  if (right === -1) return -1;
  return left - right;
}

/** Insertion-ordered buckets, so "server order" survives the grouping. */
function bucket<T>(items: T[], keyOf: (item: T) => string): Map<string, T[]> {
  const buckets = new Map<string, T[]>();
  for (const item of items) {
    const key = keyOf(item);
    const existing = buckets.get(key);
    if (existing) existing.push(item);
    else buckets.set(key, [item]);
  }
  return buckets;
}

function architectureSubgroups<T extends ExampleSummary>(items: T[]): ExampleSubgroup<T>[] {
  const named = items.filter((item) => familyOf(item) !== null);
  const unnamed = items.filter((item) => familyOf(item) === null);
  const subgroups: ExampleSubgroup<T>[] = [
    ...bucket(named, (item) => familyOf(item) as string).entries(),
  ]
    .sort(([left], [right]) => byFamily(left, right))
    .map(([family, group]) => ({
      key: `architectures:${family}`,
      label: family,
      items: [...group].sort(byOrder),
    }));
  if (unnamed.length > 0) {
    subgroups.push({ key: 'architectures', label: null, items: [...unnamed].sort(byOrder) });
  }
  return subgroups;
}

function packSubgroups<T extends ExampleSummary>(
  sectionKey: string,
  items: T[],
  packLabel: (source: string) => string | null,
): ExampleSubgroup<T>[] {
  // One sub-header per pack, packs in the order their first example arrives
  // (the server sorts by main tag and order), examples by order inside.
  return [...bucket(items, (item) => item.source ?? '').entries()].map(([source, group]) => ({
    // The plugin section keeps the bare source as the key it always had (the
    // sidebar remembers collapsed groups by key); another section prefixes
    // its own name so a pack listed under two main tags gets two keys.
    key: source === '' ? sectionKey : sectionKey === 'plugin' ? source : `${sectionKey}:${source}`,
    label: source === '' ? null : packLabel(source),
    items: [...group].sort(byOrder),
  }));
}

/**
 * Split the flat `/api/examples/list` payload into the gallery's sections,
 * one per main tag, in `mainTags` order (`GET /api/examples/tags`), or the
 * fallback order when that is not known.
 *
 * `packLabel` turns a `plugin:<id>` source into the pack's display name for
 * the sub-header -- `utils/provider`'s `pluginNameOf`, bound to the catalog
 * the calling component subscribes to. Null means no sub-header.
 */
export function groupExamplesBySection<T extends ExampleSummary>(
  examples: T[],
  packLabel: (source: string) => string | null,
  mainTags?: readonly string[] | null,
): ExampleSection<T>[] {
  const bySection = bucket(examples, mainTagOf);
  const sections: ExampleSection<T>[] = [];
  for (const key of mainTagOrder(bySection.keys(), mainTags)) {
    const items = bySection.get(key);
    if (items === undefined) continue;
    const subgroups = PACK_GROUPED.has(key)
      ? packSubgroups(key, items, packLabel)
      : key === 'architectures'
        ? architectureSubgroups(items)
        : [{ key, label: null, items: [...items].sort(byOrder) }];
    const filled = subgroups.filter((subgroup) => subgroup.items.length > 0);
    if (filled.length === 0) continue;
    sections.push({ key, subgroups: filled });
  }
  return sections;
}

/**
 * One group per sub-group, for the two surfaces that render a flat list.
 *
 * A named sub-group is labelled by its own name alone rather than
 * "<section> - <name>": the sidebar's category row leaves about 150px for the
 * label, and the pair ellipsises away exactly the half that tells CNN from
 * RNN. The section still reads off the accent the whole run of sub-groups
 * shares.
 */
export function flattenExampleSections<T>(
  sections: ExampleSection<T>[],
  t: (key: TranslationKey) => string,
): ExampleGroup<T>[] {
  return sections.flatMap((section) =>
    section.subgroups.map((subgroup) => ({
      category: subgroup.key,
      label: subgroup.label ?? mainTagLabel(section.key, t),
      sectionKey: section.key,
      items: subgroup.items,
    })),
  );
}

/** Display keys for the example categories whose name is a word to translate
 * (#684). The category is the example's folder and stays the identifier; the
 * acronyms (LLM, RL, RNN, VLA) and names (Diffusion, Transformer) read the same
 * in both locales and have no key. */
const EXAMPLE_CATEGORY_KEYS: Record<string, TranslationKey> = {
  Classical: 'examples.category.Classical',
  Model_Architecture: 'examples.category.Model_Architecture',
  Others: 'examples.category.Others',
  Papers: 'examples.category.Papers',
  Usage_Example: 'examples.category.Usage_Example',
};

/** A category key as a reader sees it, in the UI language. A category this
 * table does not know -- one a plugin's examples bring -- shows its folder
 * name with spaces for underscores. */
export function exampleCategoryLabel(
  category: string,
  t: (key: TranslationKey) => string,
): string {
  return Object.prototype.hasOwnProperty.call(EXAMPLE_CATEGORY_KEYS, category)
    ? t(EXAMPLE_CATEGORY_KEYS[category])
    : category.replace(/_/g, ' ');
}

/** What a card's chip says: the model family when the example names one -- the
 * architectures are all one category, and "CNN" is what tells them apart -- and
 * the category otherwise. */
export function exampleChipLabel(
  example: ExampleSummary,
  t: (key: TranslationKey) => string,
): string {
  return familyOf(example) ?? exampleCategoryLabel(example.category, t);
}
