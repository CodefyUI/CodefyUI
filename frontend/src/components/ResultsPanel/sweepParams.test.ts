import { describe, expect, it } from 'vitest';
import type { NodeDefinition } from '../../types';
import {
  buildRangeDomain,
  buildValuesDomain,
  eligibleSweepParams,
  expandSweepRange,
  previewSweepVariants,
} from './sweepParams';

function definition(node_name: string): NodeDefinition {
  return {
    node_name,
    category: 'Test',
    description: '',
    inputs: [],
    outputs: [],
    params: [
      { name: 'epochs', param_type: 'int', default: 2, description: '', options: [], min_value: 1, max_value: 10 },
      { name: 'rate', param_type: 'float', default: 0.1, description: '', options: [], min_value: 0, max_value: 1 },
      { name: 'enabled', param_type: 'bool', default: true, description: '', options: [], min_value: null, max_value: null },
      { name: 'label', param_type: 'string', default: 'a', description: '', options: [], min_value: null, max_value: null },
      { name: 'mode', param_type: 'select', default: 'fast', description: '', options: ['fast', 'safe'], min_value: null, max_value: null },
      { name: 'api_key', param_type: 'secret', default: '', description: '', options: [], min_value: null, max_value: null },
      { name: 'file', param_type: 'model_file', default: '', description: '', options: [], min_value: null, max_value: null },
    ],
  };
}

function node(id: string, type = 'Train', extra: Record<string, unknown> = {}) {
  return {
    id,
    type: 'customNode',
    data: { label: id, type, params: {}, ...extra },
  };
}

describe('eligibleSweepParams', () => {
  it('excludes registered free-text string parameters', () => {
    const out = eligibleSweepParams([definition('Train')], [node('n')]);
    expect(out.map((entry) => entry.param.name)).not.toContain('label');
  });

  it('returns top-level registered primitive params in canvas order', () => {
    const out = eligibleSweepParams([definition('Train')], [node('b'), node('a')]);
    expect(out.map((entry) => `${entry.nodeId}.${entry.param.name}`)).toEqual([
      'b.epochs', 'b.rate', 'b.enabled', 'b.mode',
      'a.epochs', 'a.rate', 'a.enabled', 'a.mode',
    ]);
    expect(out.some((entry) => entry.param.param_type === 'secret')).toBe(false);
  });

  it('excludes presets, subgraphs, unknown types, and every duplicated id', () => {
    const out = eligibleSweepParams([definition('Train')], [
      node('ok'),
      node('preset', 'preset:Pack', { isPreset: true }),
      node('block', 'subgraph:local'),
      node('missing', 'Unknown'),
      node('dup'), node('dup'),
    ]);
    expect(new Set(out.map((entry) => entry.nodeId))).toEqual(new Set(['ok']));
  });
});

describe('sweep domain builders', () => {
  const eligible = eligibleSweepParams([definition('Train')], [node('n')]);
  const byName = (name: string) => eligible.find((entry) => entry.param.name === name)!;

  it('preserves explicit int, float, bool and select value types', () => {
    expect(buildValuesDomain(byName('epochs'), '2, 4')).toEqual({ node_id: 'n', param: 'epochs', values: [2, 4] });
    expect(buildValuesDomain(byName('rate'), '0.1\n0.25')).toEqual({ node_id: 'n', param: 'rate', values: [0.1, 0.25] });
    expect(buildValuesDomain(byName('enabled'), 'true, false')).toEqual({ node_id: 'n', param: 'enabled', values: [true, false] });
    expect(buildValuesDomain(byName('mode'), 'fast, safe')).toEqual({ node_id: 'n', param: 'mode', values: ['fast', 'safe'] });
  });

  it('rejects mistyped, duplicate, out-of-bounds and unknown select values', () => {
    expect(() => buildValuesDomain(byName('epochs'), '2.5')).toThrow(/whole number/i);
    expect(() => buildValuesDomain(byName('epochs'), '2, 2')).toThrow(/once/i);
    expect(() => buildValuesDomain(byName('epochs'), '0, 2')).toThrow(/at least 1/i);
    expect(() => buildValuesDomain(byName('mode'), 'fast, other')).toThrow(/option/i);
  });

  it('builds valid ranges and expands them exactly for count previews', () => {
    const param = buildRangeDomain(byName('epochs'), {
      min: 1, max: 3, count: 5, scale: 'linear',
    });
    expect(param).toEqual({
      node_id: 'n', param: 'epochs',
      range: { min: 1, max: 3, count: 5, scale: 'linear', type: 'int' },
    });
    expect(expandSweepRange(param.range!)).toEqual([1, 2, 3]);
  });

  it('rejects zero count, descending ranges, and non-positive log ranges', () => {
    expect(() => buildRangeDomain(byName('rate'), { min: 0, max: 1, count: 0, scale: 'linear' })).toThrow(/count/i);
    expect(() => buildRangeDomain(byName('rate'), { min: 2, max: 1, count: 2, scale: 'linear' })).toThrow(/maximum/i);
    expect(() => buildRangeDomain(byName('rate'), { min: 0, max: 1, count: 2, scale: 'log' })).toThrow(/positive/i);
  });
});

describe('previewSweepVariants', () => {
  const domains = [
    { node_id: 'a', param: 'x', values: [1, 2] },
    { node_id: 'b', param: 'y', values: ['a', 'b', 'c'] },
  ];

  it('uses the Cartesian product for grid and samples for random', () => {
    expect(previewSweepVariants('grid', domains, null)).toEqual({ totalCombinations: 6, variantCount: 6, exceedsCap: false });
    expect(previewSweepVariants('random', domains, 4)).toEqual({ totalCombinations: 6, variantCount: 4, exceedsCap: false });
  });

  it('rejects empty domains and impossible samples and marks the cap', () => {
    expect(() => previewSweepVariants('grid', [{ node_id: 'a', param: 'x', values: [] }], null)).toThrow(/at least one/i);
    expect(() => previewSweepVariants('random', domains, 7)).toThrow(/only 6/i);
    expect(previewSweepVariants('grid', [
      { node_id: 'a', param: 'x', values: Array.from({ length: 33 }, (_, i) => i) },
    ], null)).toMatchObject({ variantCount: 33, exceedsCap: true });
  });
});
