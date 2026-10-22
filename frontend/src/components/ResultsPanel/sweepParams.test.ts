import { describe, expect, it } from 'vitest';
import type { NodeDefinition } from '../../types';
import {
  buildRangeDomain,
  buildValuesDomain,
  eligibleSweepParams,
  expandSweepRange,
  previewSweepVariants,
  SweepInputError,
  nodeWithId,
  sweepParamLabel,
  sweepParamTitle,
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

/** The code a helper refused with, so a test reads the reason, not English. */
function refusal(run: () => unknown): { code: string; vars: Record<string, unknown> } | null {
  try {
    run();
  } catch (error) {
    if (error instanceof SweepInputError) return { code: error.code, vars: error.vars };
    throw error;
  }
  return null;
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

  it('refuses bad values with a code and the values the message needs', () => {
    expect(refusal(() => buildValuesDomain(byName('epochs'), '2.5'))).toEqual({ code: 'notWhole', vars: { value: '2.5' } });
    expect(refusal(() => buildValuesDomain(byName('rate'), 'fast'))).toEqual({ code: 'notNumber', vars: { value: 'fast' } });
    expect(refusal(() => buildValuesDomain(byName('epochs'), '2, 2'))).toEqual({ code: 'repeated', vars: { value: '2' } });
    expect(refusal(() => buildValuesDomain(byName('epochs'), '0, 2'))).toEqual({ code: 'belowMin', vars: { name: 'epochs', min: 1 } });
    expect(refusal(() => buildValuesDomain(byName('epochs'), '2, 11'))).toEqual({ code: 'aboveMax', vars: { name: 'epochs', max: 10 } });
    expect(refusal(() => buildValuesDomain(byName('enabled'), 'yes'))).toEqual({ code: 'notBool', vars: { value: 'yes' } });
    expect(refusal(() => buildValuesDomain(byName('mode'), 'fast, other'))).toEqual({ code: 'notOption', vars: { value: 'other' } });
    expect(refusal(() => buildValuesDomain(byName('epochs'), ' , '))).toEqual({ code: 'noValues', vars: {} });
  });

  it('refuses a free-text string param the dialog never offers', () => {
    const label = definition('Train').params.find((param) => param.name === 'label')!;
    expect(refusal(() => buildValuesDomain(
      { nodeId: 'n', nodeLabel: 'n', nodeType: 'Train', param: label },
      'a, b',
    ))).toEqual({ code: 'notSweepable', vars: { type: 'string' } });
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

  it('refuses impossible ranges with a code', () => {
    expect(refusal(() => buildRangeDomain(byName('rate'), { min: 0, max: 1, count: 0, scale: 'linear' }))?.code).toBe('rangeCount');
    expect(refusal(() => buildRangeDomain(byName('rate'), { min: 2, max: 1, count: 2, scale: 'linear' }))?.code).toBe('rangeOrder');
    expect(refusal(() => buildRangeDomain(byName('rate'), { min: 0, max: 1, count: 2, scale: 'log' }))?.code).toBe('rangeLogMin');
    expect(refusal(() => buildRangeDomain(byName('rate'), { min: Number.NaN, max: 1, count: 2, scale: 'linear' }))?.code).toBe('rangeNotFinite');
    expect(refusal(() => buildRangeDomain(byName('epochs'), { min: 1.5, max: 3, count: 2, scale: 'linear' }))?.code).toBe('rangeWholeBounds');
    expect(refusal(() => buildRangeDomain(byName('enabled'), { min: 0, max: 1, count: 2, scale: 'linear' }))?.code).toBe('rangeNumericOnly');
  });
});

describe('previewSweepVariants', () => {
  const domains = [
    { node_id: 'a', param: 'x', values: [1, 2] },
    { node_id: 'b', param: 'y', values: ['a', 'b', 'c'] },
  ];

  it('uses the Cartesian product for grid and samples for random', () => {
    expect(previewSweepVariants('grid', domains, null)).toEqual({ totalCombinations: 6, variantCount: 6, warnings: [] });
    expect(previewSweepVariants('random', domains, 4)).toEqual({ totalCombinations: 6, variantCount: 4, warnings: [] });
  });

  it('refuses empty domains and impossible samples with a code', () => {
    expect(refusal(() => previewSweepVariants('grid', [{ node_id: 'a', param: 'x', values: [] }], null)))
      .toEqual({ code: 'emptyDomain', vars: { address: 'a.x' } });
    expect(refusal(() => previewSweepVariants('random', domains, 7)))
      .toEqual({ code: 'samplesExceedSpace', vars: { samples: 7, total: 6 } });
    expect(refusal(() => previewSweepVariants('random', domains, 0))?.code).toBe('samples');
    expect(refusal(() => previewSweepVariants('grid', [], null))?.code).toBe('noParams');
  });

  it('warns past each default server cap instead of refusing', () => {
    // Every cap is a server setting a deployment may raise, so going past
    // the default is a warning; the server's own refusal stays authoritative.
    expect(previewSweepVariants('grid', [
      { node_id: 'a', param: 'x', values: Array.from({ length: 33 }, (_, i) => i) },
    ], null)).toEqual({
      totalCombinations: 33,
      variantCount: 33,
      warnings: [{ limit: 'values', index: 0, count: 33 }, { limit: 'runs', count: 33 }],
    });
    const five = Array.from({ length: 5 }, (_, i) => ({ node_id: `n${i}`, param: 'x', values: [1] }));
    expect(previewSweepVariants('grid', five, null).warnings).toEqual([{ limit: 'params', count: 5 }]);
  });

  it('never expands a range longer than the default domain cap', () => {
    // 2**32 points is an "Invalid array length" if expanded; a long range is
    // reported from its count alone, and a grid over it has no exact size.
    const huge = { node_id: 'a', param: 'rate', range: { min: 0, max: 1, count: 2 ** 32, scale: 'linear' as const, type: 'float' as const } };
    expect(previewSweepVariants('grid', [huge], null)).toEqual({
      totalCombinations: null,
      variantCount: null,
      warnings: [{ limit: 'values', index: 0, count: 2 ** 32 }],
    });
    // A random sweep still knows its variant count: it is the sample count.
    expect(previewSweepVariants('random', [huge], 3)).toEqual({
      totalCombinations: null,
      variantCount: 3,
      warnings: [{ limit: 'values', index: 0, count: 2 ** 32 }],
    });
  });
});

describe('sweepParamTitle', () => {
  const definitions = [definition('Train')];

  it('names an address by the open node label and the param', () => {
    expect(sweepParamTitle({ node_id: 'n1', param: 'rate' }, definitions, [node('n1', 'Train', { label: 'Trainer' })]))
      .toBe('Trainer · rate');
  });

  it('falls back to the node type when the node has no label', () => {
    expect(sweepParamTitle({ node_id: 'n1', param: 'rate' }, definitions, [node('n1', 'Train', { label: '' })]))
      .toBe('Train · rate');
  });

  it('adds the node id when asked, for two columns that would read the same (#841)', () => {
    expect(sweepParamTitle({ node_id: 'opt-plain', param: 'rate' }, definitions,
      [node('opt-plain', 'Train', { label: 'Optimizer' })], true)).toBe('Optimizer (opt-plain) · rate');
  });

  it('gives no title when the open graph cannot vouch for the address', () => {
    // A sweep may come from another graph: an id that is absent, duplicated,
    // or on a node whose type has no such param says nothing about this one.
    const nodes = [node('n1'), node('dup'), node('dup'), node('other', 'Unknown')];
    expect(sweepParamTitle({ node_id: 'gone', param: 'rate' }, definitions, nodes)).toBeNull();
    expect(sweepParamTitle({ node_id: 'dup', param: 'rate' }, definitions, nodes)).toBeNull();
    expect(sweepParamTitle({ node_id: 'n1', param: 'momentum' }, definitions, nodes)).toBeNull();
    expect(sweepParamTitle({ node_id: 'other', param: 'rate' }, definitions, nodes)).toBeNull();
  });
});

describe('sweep parameter labels (#841)', () => {
  it('names a node by its label and id, or its id alone when they match', () => {
    expect(nodeWithId('Optimizer', 'opt-plain')).toBe('Optimizer (opt-plain)');
    expect(nodeWithId('opt-plain', 'opt-plain')).toBe('opt-plain');
    expect(nodeWithId('', 'opt-plain')).toBe('opt-plain');
  });

  it('tells two branches\' parameters apart', () => {
    const [plain, decay] = eligibleSweepParams([definition('Train')], [
      node('opt-plain', 'Train', { label: 'Optimizer' }),
      node('opt-decay', 'Train', { label: 'Optimizer' }),
    ]).filter((entry) => entry.param.name === 'rate');
    expect(sweepParamLabel(plain)).toBe('Optimizer (opt-plain) · rate');
    expect(sweepParamLabel(decay)).toBe('Optimizer (opt-decay) · rate');
  });
});

describe('sweepParams edge arms', () => {
  it('names an unlabelled node by its id', () => {
    const [first] = eligibleSweepParams([definition('Train')], [node('n7', 'Train', { label: '' })]);
    expect(first.nodeLabel).toBe('n7');
  });

  it('expands a one-point range to its minimum', () => {
    expect(expandSweepRange({ min: 0.3, max: 0.9, count: 1, scale: 'linear', type: 'float' })).toEqual([0.3]);
  });

  it('expands a log range with exact end points', () => {
    const values = expandSweepRange({ min: 0.001, max: 1, count: 4, scale: 'log', type: 'float' });
    expect(values).toHaveLength(4);
    expect(values[0]).toBe(0.001);
    expect(values[1]).toBeCloseTo(0.01);
    expect(values[2]).toBeCloseTo(0.1);
    expect(values[3]).toBe(1);
  });

  it('rounds a negative int range half away from zero', () => {
    // -3, -2.5, -2, -1.5, -1 rounds to -3, -3, -2, -2, -1.
    expect(expandSweepRange({ min: -3, max: -1, count: 5, scale: 'linear', type: 'int' })).toEqual([-3, -2, -1]);
  });

  it('refuses a param with neither values nor a range', () => {
    expect(refusal(() => previewSweepVariants('grid', [{ node_id: 'a', param: 'x' }], null)))
      .toEqual({ code: 'emptyDomain', vars: { address: 'a.x' } });
  });

  it('keeps the total unknown once a long range made it so', () => {
    const long = { node_id: 'a', param: 'rate', range: { min: 0, max: 1, count: 40, scale: 'linear' as const, type: 'float' as const } };
    const short = { node_id: 'c', param: 'rate', range: { min: 0, max: 1, count: 2, scale: 'linear' as const, type: 'float' as const } };
    expect(previewSweepVariants('grid', [long, { node_id: 'b', param: 'x', values: [1, 2] }, short], null)).toEqual({
      totalCombinations: null,
      variantCount: null,
      warnings: [{ limit: 'values', index: 0, count: 40 }],
    });
  });

  it('refuses a random sweep with no sample count', () => {
    expect(refusal(() => previewSweepVariants('random', [{ node_id: 'a', param: 'x', values: [1] }], null))?.code)
      .toBe('samples');
  });
});
