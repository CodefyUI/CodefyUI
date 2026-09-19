import type { ExampleSummary } from '../api/rest';
import type { TranslationKey } from '../i18n';

/**
 * The one grouping every example surface renders (#141).
 *
 * The empty-canvas overlay, the sidebar's Templates tab and the gallery modal
 * used to group three different ways -- a pinned trio plus a curated category
 * order here, a rank-by-category list there -- so the same example sat in a
 * different place depending on where you opened it from. All three now call
 * `groupExamplesBySection`, and what an example claims travels with the
 * example: `graph.json` carries a `gallery` block and `/api/examples/list`
 * serves it as `section`, `family` and `order`.
 *
 * Nothing here reads the path except to spot a pack, so adding an example is
 * an edit to its own `graph.json` rather than to a table in the frontend.
 */

/** The gallery sections, in the order they render. */
export type ExampleSectionKey =
  | 'quickstart'
  | 'training'
  | 'llm'
  | 'concepts'
  | 'architectures'
  | 'plugin'
  | 'other';

/** One block inside a section. `label === null` means no sub-header. */
export interface ExampleSubgroup<T> {
  /** Unique across every section, so a surface can key its DOM on it. */
  key: string;
  label: string | null;
  items: T[];
}

export interface ExampleSection<T> {
  key: ExampleSectionKey;
  titleKey: TranslationKey;
  subgroups: ExampleSubgroup<T>[];
}

/** A flattened sub-group, in the shape `CategoryList` takes. */
export interface ExampleGroup<T> {
  /** `CategoryGroup.category`: the sub-group's key, not a category name. */
  category: string;
  label: string;
  sectionKey: ExampleSectionKey;
  items: T[];
}

const SECTION_TITLE_KEYS: Record<ExampleSectionKey, TranslationKey> = {
  quickstart: 'examples.section.quickstart',
  training: 'examples.section.training',
  llm: 'examples.section.llm',
  concepts: 'examples.section.concepts',
  architectures: 'examples.section.architectures',
  plugin: 'examples.section.plugin',
  other: 'examples.section.other',
};

const SECTION_ORDER: ExampleSectionKey[] = [
  'quickstart',
  'training',
  'llm',
  'concepts',
  'architectures',
  'plugin',
  'other',
];

/** The sections an example may claim. `plugin` and `other` are ours to assign:
 * a pack's example goes to `plugin` by its path, and `other` is where a
 * built-in that declares nothing (or something we do not know) lands. */
const DECLARABLE_SECTIONS: ExampleSectionKey[] = [
  'quickstart',
  'training',
  'llm',
  'concepts',
  'architectures',
];

/** Sub-header order inside `architectures`; any other family sorts after these,
 * alphabetically. */
const FAMILY_ORDER: string[] = ['CNN', 'RNN', 'Transformer', 'Diffusion', 'RL'];

/** Same prefix `utils/provider` parses a `source` with; an example's PATH
 * carries it too, and the path is what decides the section. */
const PLUGIN_PREFIX = 'plugin:';

function sectionOf(example: ExampleSummary): ExampleSectionKey {
  if (example.path.startsWith(PLUGIN_PREFIX)) return 'plugin';
  const declared = example.section as ExampleSectionKey | null | undefined;
  return declared != null && DECLARABLE_SECTIONS.includes(declared) ? declared : 'other';
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
  items: T[],
  packLabel: (source: string) => string | null,
): ExampleSubgroup<T>[] {
  // Server order throughout: a pack's examples are listed in the order the
  // pack ships them, and a `gallery` block does not change that.
  return [...bucket(items, (item) => item.source ?? '').entries()].map(([source, group]) => ({
    key: source === '' ? 'plugin' : source,
    label: source === '' ? null : packLabel(source),
    items: group,
  }));
}

/**
 * Split the flat `/api/examples/list` payload into the gallery's sections.
 *
 * `packLabel` turns a `plugin:<id>` source into the pack's display name for
 * the sub-header -- `utils/provider`'s `pluginNameOf`, bound to the catalog
 * the calling component subscribes to. Null means no sub-header.
 */
export function groupExamplesBySection<T extends ExampleSummary>(
  examples: T[],
  packLabel: (source: string) => string | null,
): ExampleSection<T>[] {
  const bySection = bucket(examples, sectionOf);
  const sections: ExampleSection<T>[] = [];
  for (const key of SECTION_ORDER) {
    const items = bySection.get(key);
    if (items === undefined) continue;
    const subgroups =
      key === 'plugin'
        ? packSubgroups(items, packLabel)
        : key === 'architectures'
          ? architectureSubgroups(items)
          : [{ key, label: null, items: [...items].sort(byOrder) }];
    const filled = subgroups.filter((subgroup) => subgroup.items.length > 0);
    if (filled.length === 0) continue;
    sections.push({ key, titleKey: SECTION_TITLE_KEYS[key], subgroups: filled });
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
      label: subgroup.label ?? t(section.titleKey),
      sectionKey: section.key,
      items: subgroup.items,
    })),
  );
}

/** A category key as a reader sees it. Categories are the example's folder and
 * stay English in every locale. */
export function exampleCategoryLabel(category: string): string {
  return category.replace(/_/g, ' ');
}

/** What a card's chip says: the model family when the example names one -- the
 * architectures are all one category, and "CNN" is what tells them apart -- and
 * the category otherwise. */
export function exampleChipLabel(example: ExampleSummary): string {
  return familyOf(example) ?? exampleCategoryLabel(example.category);
}
