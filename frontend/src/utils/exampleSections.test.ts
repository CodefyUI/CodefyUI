import { describe, it, expect } from 'vitest';
import {
  exampleCategoryLabel,
  exampleChipLabel,
  flattenExampleSections,
  groupExamplesBySection,
} from './exampleSections';
import type { TranslationKey } from '../i18n';
import type { ExampleSummary } from '../api/rest';

function ex(overrides: Partial<ExampleSummary> & { path: string }): ExampleSummary {
  return {
    name: 'Example',
    description: 'short desc',
    category: 'Usage_Example',
    node_count: 3,
    edge_count: 2,
    source: 'builtin',
    section: null,
    family: null,
    order: null,
    ...overrides,
  };
}

/** No pack catalog: every `plugin:<id>` source resolves to nothing. */
const noPacks = () => null;

/** The pack name a catalog would give, for the sub-header cases. */
const packNames = (source: string) =>
  ({ 'plugin:c1': 'Chapter 1', 'plugin:c2': 'Chapter 2' })[source] ?? null;

/** Identity, so a label assertion names the key the section carries. */
const t = (key: TranslationKey): string => key;

describe('groupExamplesBySection', () => {
  it('renders the sections in the contract order, whatever order the server sent', () => {
    const sections = groupExamplesBySection(
      [
        ex({ path: 'X/unclassified' }),
        ex({ path: 'plugin:c1/Foo', source: 'plugin:c1' }),
        ex({ path: 'Model_Architecture/A', section: 'architectures', family: 'CNN' }),
        ex({ path: 'Classical/A', section: 'concepts' }),
        ex({ path: 'LLM/A', section: 'llm' }),
        ex({ path: 'LLM/Train', section: 'training' }),
        ex({ path: 'Usage_Example/A', section: 'quickstart' }),
      ],
      noPacks,
    );
    expect(sections.map((s) => s.key)).toEqual([
      'quickstart',
      'training',
      'llm',
      'concepts',
      'architectures',
      'plugin',
      'other',
    ]);
    expect(sections.map((s) => s.titleKey)).toEqual([
      'examples.section.quickstart',
      'examples.section.training',
      'examples.section.llm',
      'examples.section.concepts',
      'examples.section.architectures',
      'examples.section.plugin',
      'examples.section.other',
    ]);
  });

  it('drops sections and sub-groups that nothing landed in', () => {
    expect(groupExamplesBySection([], noPacks)).toEqual([]);
    const sections = groupExamplesBySection([ex({ path: 'LLM/A', section: 'llm' })], noPacks);
    expect(sections).toHaveLength(1);
    expect(sections[0].subgroups).toHaveLength(1);
    expect(sections[0].subgroups[0].items.map((e) => e.path)).toEqual(['LLM/A']);
  });

  it('sends a plugin example to the plugin section whatever its gallery block claims', () => {
    // A pack is free to ship a `gallery` block; it never buys a place among
    // the built-in sections, and it does not reorder the pack's own list.
    const sections = groupExamplesBySection(
      [
        ex({ path: 'plugin:c1/Second', source: 'plugin:c1', section: 'quickstart', order: 9 }),
        ex({ path: 'plugin:c1/First', source: 'plugin:c1', section: 'architectures', order: 1 }),
      ],
      noPacks,
    );
    expect(sections.map((s) => s.key)).toEqual(['plugin']);
    expect(sections[0].subgroups[0].items.map((e) => e.path)).toEqual([
      'plugin:c1/Second',
      'plugin:c1/First',
    ]);
  });

  it('sub-groups the plugin section per pack, packs in server order', () => {
    const sections = groupExamplesBySection(
      [
        ex({ path: 'plugin:c2/A', source: 'plugin:c2' }),
        ex({ path: 'plugin:c1/A', source: 'plugin:c1' }),
        ex({ path: 'plugin:c2/B', source: 'plugin:c2' }),
      ],
      packNames,
    );
    expect(sections[0].subgroups.map((g) => [g.key, g.label])).toEqual([
      ['plugin:c2', 'Chapter 2'],
      ['plugin:c1', 'Chapter 1'],
    ]);
    expect(sections[0].subgroups[0].items.map((e) => e.path)).toEqual([
      'plugin:c2/A',
      'plugin:c2/B',
    ]);
  });

  it('leaves a pack nothing names under no sub-header', () => {
    // A backend that predates `source` still lists the pack's examples.
    const sections = groupExamplesBySection([ex({ path: 'plugin:c9/A', source: '' })], noPacks);
    expect(sections[0].subgroups).toEqual([
      { key: 'plugin', label: null, items: [expect.objectContaining({ path: 'plugin:c9/A' })] },
    ]);
  });

  it('puts a built-in with no section, an unknown one, or a section it may not claim, under other', () => {
    const sections = groupExamplesBySection(
      [
        ex({ path: 'A', section: null }),
        ex({ path: 'B', section: 'wat' }),
        // `plugin` and `other` are the frontend's own buckets: a built-in
        // cannot declare its way into the pack list.
        ex({ path: 'C', section: 'plugin' }),
        ex({ path: 'D', section: 'other' }),
      ],
      noPacks,
    );
    expect(sections.map((s) => s.key)).toEqual(['other']);
    expect(sections[0].subgroups[0].items.map((e) => e.path)).toEqual(['A', 'B', 'C', 'D']);
  });

  it('sub-groups architectures by family: the shipped order, then the rest alphabetically, then none', () => {
    // All five shipped families, in an order none of them belongs in, so the
    // assertion pins the sequence rather than the two or three pairs a
    // smaller fixture happens to constrain. Every one of them ships: leaving
    // Diffusion out of the fixture let it be dropped from FAMILY_ORDER
    // entirely -- which sorts DiT-Diffusion-Transformer below the RL
    // examples -- with every test still green.
    const sections = groupExamplesBySection(
      [
        ex({ path: 'm/rl', section: 'architectures', family: 'RL' }),
        ex({ path: 'm/gnn', section: 'architectures', family: 'GNN' }),
        ex({ path: 'm/none', section: 'architectures' }),
        ex({ path: 'm/diff', section: 'architectures', family: 'Diffusion' }),
        ex({ path: 'm/cnn', section: 'architectures', family: 'CNN' }),
        ex({ path: 'm/ann', section: 'architectures', family: 'ANN' }),
        ex({ path: 'm/tf', section: 'architectures', family: 'Transformer' }),
        ex({ path: 'm/rnn', section: 'architectures', family: 'RNN' }),
      ],
      noPacks,
    );
    expect(sections[0].subgroups.map((g) => g.label)).toEqual([
      'CNN',
      'RNN',
      'Transformer',
      'Diffusion',
      'RL',
      'ANN',
      'GNN',
      null,
    ]);
    expect(sections[0].subgroups.map((g) => g.key)).toEqual([
      'architectures:CNN',
      'architectures:RNN',
      'architectures:Transformer',
      'architectures:Diffusion',
      'architectures:RL',
      'architectures:ANN',
      'architectures:GNN',
      'architectures',
    ]);
  });

  it('sorts a sub-group by order, nulls last, ties in server order', () => {
    const sections = groupExamplesBySection(
      [
        ex({ path: 'tie-a', section: 'training', order: 2 }),
        ex({ path: 'last', section: 'training' }),
        ex({ path: 'first', section: 'training', order: 1 }),
        ex({ path: 'tie-b', section: 'training', order: 2 }),
        // A pack that writes junk here must not reorder anything: the backend
        // degrades it to null, and so does this.
        ex({ path: 'junk', section: 'training', order: Number.NaN }),
      ],
      noPacks,
    );
    expect(sections[0].subgroups[0].items.map((e) => e.path)).toEqual([
      'first',
      'tie-a',
      'tie-b',
      'last',
      'junk',
    ]);
  });

  it('sorts inside a family, not across the architectures section', () => {
    const sections = groupExamplesBySection(
      [
        ex({ path: 'm/rnn2', section: 'architectures', family: 'RNN', order: 2 }),
        ex({ path: 'm/cnn2', section: 'architectures', family: 'CNN', order: 2 }),
        ex({ path: 'm/rnn1', section: 'architectures', family: 'RNN', order: 1 }),
        ex({ path: 'm/cnn1', section: 'architectures', family: 'CNN', order: 1 }),
      ],
      noPacks,
    );
    expect(sections[0].subgroups.map((g) => g.items.map((e) => e.path))).toEqual([
      ['m/cnn1', 'm/cnn2'],
      ['m/rnn1', 'm/rnn2'],
    ]);
  });

  it('hands back the type it was given', () => {
    // The surfaces group LocalizedExample, not the bare wire row: whatever the
    // locale resolved has to survive the grouping.
    interface Localized extends ExampleSummary {
      descriptionEn: string;
    }
    const localized: Localized[] = [
      { ...ex({ path: 'LLM/A', section: 'llm' }), descriptionEn: 'the English one' },
    ];
    const sections = groupExamplesBySection(localized, noPacks);
    expect(sections[0].subgroups[0].items[0].descriptionEn).toBe('the English one');
  });
});

describe('flattenExampleSections', () => {
  it('gives every sub-group one group, keyed uniquely', () => {
    const groups = flattenExampleSections(
      groupExamplesBySection(
        [
          ex({ path: 'Usage_Example/A', section: 'quickstart' }),
          ex({ path: 'm/cnn', section: 'architectures', family: 'CNN' }),
          ex({ path: 'm/rnn', section: 'architectures', family: 'RNN' }),
          ex({ path: 'plugin:c1/A', source: 'plugin:c1' }),
        ],
        packNames,
      ),
      t,
    );
    expect(groups.map((g) => g.category)).toEqual([
      'quickstart',
      'architectures:CNN',
      'architectures:RNN',
      'plugin:c1',
    ]);
    expect(new Set(groups.map((g) => g.category)).size).toBe(groups.length);
    expect(groups.map((g) => g.sectionKey)).toEqual([
      'quickstart',
      'architectures',
      'architectures',
      'plugin',
    ]);
  });

  it('labels a sub-group by its own name, and one with no name by its section', () => {
    // The sidebar row is ~150px wide for the name: "Model Architectures - CNN"
    // ellipsises away exactly the half that tells the groups apart.
    const groups = flattenExampleSections(
      groupExamplesBySection(
        [
          ex({ path: 'LLM/A', section: 'llm' }),
          ex({ path: 'm/cnn', section: 'architectures', family: 'CNN' }),
          ex({ path: 'm/none', section: 'architectures' }),
          ex({ path: 'plugin:c1/A', source: 'plugin:c1' }),
        ],
        packNames,
      ),
      t,
    );
    expect(groups.map((g) => g.label)).toEqual([
      'examples.section.llm',
      'CNN',
      'examples.section.architectures',
      'Chapter 1',
    ]);
  });

  it('keeps the items of each sub-group', () => {
    const groups = flattenExampleSections(
      groupExamplesBySection(
        [
          ex({ path: 'first', section: 'training', order: 1 }),
          ex({ path: 'second', section: 'training', order: 2 }),
        ],
        noPacks,
      ),
      t,
    );
    expect(groups).toHaveLength(1);
    expect(groups[0].items.map((e) => e.path)).toEqual(['first', 'second']);
  });
});

describe('example labels', () => {
  it('spells a category out for display', () => {
    expect(exampleCategoryLabel('Model_Architecture')).toBe('Model Architecture');
  });

  it('puts the family on a card that has one, and the category on one that does not', () => {
    expect(exampleChipLabel(ex({ path: 'm/a', category: 'Model_Architecture', family: 'CNN' }))).toBe(
      'CNN',
    );
    expect(exampleChipLabel(ex({ path: 'u/a', category: 'Usage_Example' }))).toBe('Usage Example');
    // An empty family is no family, not an empty chip.
    expect(exampleChipLabel(ex({ path: 'u/b', category: 'Usage_Example', family: '' }))).toBe(
      'Usage Example',
    );
  });
});
