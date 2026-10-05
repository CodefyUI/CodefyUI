import { describe, it, expect, vi } from 'vitest';
import {
  MatchTier,
  compareMatches,
  matchTier,
  nodeSearchTexts,
  presetSearchTexts,
  rankMatches,
  type TranslateNodeField,
} from './nodeSearch';
import type { PluginIndex } from './provider';
import type { NodeDefinition, PresetDefinition } from '../types';

function node(node_name: string, description = '', over: Partial<NodeDefinition> = {}): NodeDefinition {
  return { node_name, category: 'Utility', description, inputs: [], outputs: [], params: [], ...over };
}

// `pluginNameOf` reads nothing of a catalog row but its name.
const edu = { edu: { id: 'edu', name: 'EDU - hands-on teaching nodes' } } as unknown as PluginIndex;

/** What `tn` does in the English UI: every field answers its fallback. */
const english: TranslateNodeField = (_nodeName, _field, fallback) => fallback;

describe('matchTier', () => {
  it('ranks a name equal to the query as Exact, ignoring case and outer spaces', () => {
    expect(matchTier('Linear', [], 'linear')).toBe(MatchTier.Exact);
    expect(matchTier('Linear', [], 'LINEAR')).toBe(MatchTier.Exact);
    expect(matchTier('Linear', [], '  Linear ')).toBe(MatchTier.Exact);
  });

  it('ranks a name that starts with the query as Prefix', () => {
    expect(matchTier('LinearRegression', [], 'linear')).toBe(MatchTier.Prefix);
  });

  it('ranks a query that starts at a word inside the name as Word', () => {
    expect(matchTier('TrainTestSplit', [], 'split')).toBe(MatchTier.Word);
    expect(matchTier('TrainTestSplit', [], 'test')).toBe(MatchTier.Word);
    // The rest of the name counts from that word on, so a query may run on
    // across the next boundary.
    expect(matchTier('TrainTestSplit', [], 'testsplit')).toBe(MatchTier.Word);
    expect(matchTier('TrainTestSplit', [], 'tests')).toBe(MatchTier.Word);
  });

  it('finds words at separators, digit runs and the end of an acronym', () => {
    expect(matchTier('my_custom-node x', [], 'custom')).toBe(MatchTier.Word);
    expect(matchTier('my_custom-node x', [], 'node')).toBe(MatchTier.Word);
    expect(matchTier('my_custom-node x', [], 'x')).toBe(MatchTier.Word);
    expect(matchTier('Conv2dExplicit', [], 'explicit')).toBe(MatchTier.Word);
    expect(matchTier('Conv2dExplicit', [], '2d')).toBe(MatchTier.Word);
    expect(matchTier('LSTMCell', [], 'cell')).toBe(MatchTier.Word);
    expect(matchTier('HFTextGenerate', [], 'textgen')).toBe(MatchTier.Word);
  });

  it('ranks the query anywhere else in the name as Substring', () => {
    expect(matchTier('Bilinear', [], 'linear')).toBe(MatchTier.Substring);
    // Inside a word, not at its start.
    expect(matchTier('TrainTestSplit', [], 'plit')).toBe(MatchTier.Substring);
    expect(matchTier('LSTMCell', [], 'stm')).toBe(MatchTier.Substring);
  });

  it('ranks a query found only in the other texts as Text, ignoring case', () => {
    expect(matchTier('DQN', ['Q-network with a Linear head'], 'linear')).toBe(MatchTier.Text);
    // Absent and empty texts are skipped, not matched.
    expect(matchTier('DQN', [null, undefined, '', 'a linear head'], 'linear')).toBe(MatchTier.Text);
  });

  it('lets the name decide when the name and a text both match', () => {
    expect(matchTier('Linear', ['a linear layer'], 'linear')).toBe(MatchTier.Exact);
    expect(matchTier('Bilinear', ['a linear blend'], 'linear')).toBe(MatchTier.Substring);
  });

  it('answers null when nothing matches', () => {
    expect(matchTier('Conv2d', ['slides a kernel'], 'linear')).toBeNull();
    expect(matchTier('Conv2d', [null, undefined, ''], 'linear')).toBeNull();
  });

  it('answers null for an empty query, which is not a search', () => {
    // Callers list everything in their own order without asking; a caller that
    // forgot would show nothing rather than a silently re-sorted list.
    expect(matchTier('Conv2d', ['slides a kernel'], '')).toBeNull();
    expect(matchTier('Conv2d', ['slides a kernel'], '   ')).toBeNull();
  });

  describe('a qualified plugin name', () => {
    it('matches the part after the prefix as a name of its own', () => {
      expect(matchTier('edu:FilterRows', [], 'filterrows')).toBe(MatchTier.Exact);
      expect(matchTier('edu:FilterRows', [], 'filter')).toBe(MatchTier.Prefix);
      expect(matchTier('edu:FilterRows', [], 'rows')).toBe(MatchTier.Word);
    });

    it('still matches the full name', () => {
      expect(matchTier('edu:FilterRows', [], 'edu:filterrows')).toBe(MatchTier.Exact);
      expect(matchTier('edu:FilterRows', [], 'edu')).toBe(MatchTier.Prefix);
      expect(matchTier('edu:FilterRows', [], 'u:f')).toBe(MatchTier.Substring);
    });

    it('takes the part after the last colon', () => {
      expect(matchTier('pack:edu:FilterRows', [], 'filterrows')).toBe(MatchTier.Exact);
    });
  });
});

describe('compareMatches', () => {
  const key = (tier: MatchTier, name: string, index: number) => ({ tier, name, index });
  const keys = [
    key(MatchTier.Text, 'DQN', 0),
    key(MatchTier.Prefix, 'LinearRegression', 1),
    key(MatchTier.Prefix, 'LinearNet', 2),
    // Same tier and length as LinearNet: the earlier position goes first.
    key(MatchTier.Prefix, 'LinearMLP', 3),
    key(MatchTier.Exact, 'Linear', 4),
    key(MatchTier.Substring, 'Bilinear', 5),
    key(MatchTier.Word, 'SparseLinear', 6),
  ];
  const expected = [
    'Linear',
    'LinearNet',
    'LinearMLP',
    'LinearRegression',
    'SparseLinear',
    'Bilinear',
    'DQN',
  ];

  it('orders by tier, then the shorter name, then the original position', () => {
    expect([...keys].sort(compareMatches).map((k) => k.name)).toEqual(expected);
  });

  it('gives the same order whatever order the entries arrive in', () => {
    expect([...keys].reverse().sort(compareMatches).map((k) => k.name)).toEqual(expected);
  });
});

describe('rankMatches', () => {
  it('keeps the entries that match, best first, each with its tier and position', () => {
    // The UAT case: "linear" finds a dozen nodes through their descriptions,
    // and the node actually named Linear must not be the last of them.
    const names = ['DQN', 'Bilinear', 'TrainTestSplit', 'LinearRegression', 'Conv2d', 'Linear'];
    const descriptions: Record<string, string> = { DQN: 'a linear Q-value head' };
    const ranked = rankMatches(names, (name) => name, (name) => [descriptions[name]], 'linear');
    expect(ranked.map((m) => m.item)).toEqual(['Linear', 'LinearRegression', 'Bilinear', 'DQN']);
    expect(ranked.map((m) => m.tier)).toEqual([
      MatchTier.Exact,
      MatchTier.Prefix,
      MatchTier.Substring,
      MatchTier.Text,
    ]);
    expect(ranked.map((m) => m.index)).toEqual([5, 3, 1, 0]);
  });

  it('answers an empty list when nothing matches', () => {
    expect(rankMatches(['Conv2d'], (name) => name, () => [], 'linear')).toEqual([]);
  });
});

describe('nodeSearchTexts', () => {
  it('reads the summary, the details and the plugin display name', () => {
    const def = node('edu:FilterRows', 'drops rows a predicate rejects', {
      details: 'Wraps pandas.DataFrame.query.',
      provider: 'plugin:edu',
    });
    const texts = nodeSearchTexts(def, edu, english);
    expect(texts).toContain('drops rows a predicate rejects');
    expect(texts).toContain('Wraps pandas.DataFrame.query.');
    expect(texts).toContain('EDU - hands-on teaching nodes');
    expect(matchTier(def.node_name, texts, 'hands-on')).toBe(MatchTier.Text);
  });

  it('adds the translated summary and details, asking for an empty fallback', () => {
    const tn = vi.fn<TranslateNodeField>((nodeName, field, fallback) =>
      nodeName === 'Conv2d' ? `${field}: 在影像上滑動可學習的卷積核` : fallback,
    );
    const texts = nodeSearchTexts(node('Conv2d', 'slides a learned kernel'), {}, tn);
    expect(matchTier('Conv2d', texts, '卷積')).toBe(MatchTier.Text);
    // English still matches in the Chinese UI.
    expect(matchTier('Conv2d', texts, 'kernel')).toBe(MatchTier.Text);
    // An empty fallback, not the English text: an untranslated node would
    // otherwise carry its English summary twice.
    expect(tn).toHaveBeenCalledWith('Conv2d', 'description', '');
    expect(tn).toHaveBeenCalledWith('Conv2d', 'details', '');
  });

  it('finds nothing in a translation the English UI does not have', () => {
    const texts = nodeSearchTexts(node('Conv2d', 'slides a learned kernel'), {}, english);
    expect(matchTier('Conv2d', texts, '卷積')).toBeNull();
  });
});

describe('presetSearchTexts', () => {
  it('reads the description and every tag', () => {
    const preset = {
      preset_name: 'LeNet',
      category: 'CNN',
      description: 'a small convolutional classifier',
      tags: ['beginner', 'vision'],
    } as PresetDefinition;
    expect(presetSearchTexts(preset)).toEqual([
      'a small convolutional classifier',
      'beginner',
      'vision',
    ]);
  });
});
