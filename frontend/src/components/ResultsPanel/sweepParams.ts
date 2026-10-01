import type { SweepMethod, SweepParam, SweepRange, SweepValue } from '../../api/rest';
import type { AppNode, NodeDefinition, ParamDefinition } from '../../types';

const SWEEPABLE_TYPES = new Set<ParamDefinition['param_type']>([
  'int', 'float', 'bool', 'string', 'select',
]);

export interface EligibleSweepParam {
  nodeId: string;
  nodeLabel: string;
  nodeType: string;
  param: ParamDefinition;
}

/** Registered, top-level parameters the backend can address as node_id.param. */
export function eligibleSweepParams(
  definitions: readonly NodeDefinition[],
  nodes: readonly Pick<AppNode, 'id' | 'type' | 'data'>[],
): EligibleSweepParam[] {
  const byType = new Map(definitions.map((definition) => [definition.node_name, definition]));
  const idCounts = new Map<string, number>();
  for (const node of nodes) idCounts.set(node.id, (idCounts.get(node.id) ?? 0) + 1);

  const out: EligibleSweepParam[] = [];
  for (const node of nodes) {
    const nodeType = node.data.type;
    if (idCounts.get(node.id) !== 1
        || node.data.isPreset
        || nodeType.startsWith('preset:')
        || nodeType.startsWith('subgraph:')) continue;
    const definition = byType.get(nodeType);
    if (!definition) continue;
    for (const param of definition.params) {
      if (!SWEEPABLE_TYPES.has(param.param_type)) continue;
      out.push({
        nodeId: node.id,
        nodeLabel: node.data.label || node.id,
        nodeType,
        param,
      });
    }
  }
  return out;
}

function tokens(text: string): string[] {
  return text.split(/[\n,]/).map((value) => value.trim()).filter(Boolean);
}

function numberValue(raw: string, kind: 'int' | 'float'): number {
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`“${raw}” is not a finite number`);
  if (kind === 'int' && !Number.isInteger(value)) {
    throw new Error(`“${raw}” must be a whole number`);
  }
  return value;
}

function assertBounds(param: ParamDefinition, value: number): void {
  if (param.min_value !== null && value < param.min_value) {
    throw new Error(`${param.name} must be at least ${param.min_value}`);
  }
  if (param.max_value !== null && value > param.max_value) {
    throw new Error(`${param.name} must be at most ${param.max_value}`);
  }
}

function parseValue(param: ParamDefinition, raw: string): SweepValue {
  switch (param.param_type) {
    case 'int':
    case 'float': {
      const value = numberValue(raw, param.param_type);
      assertBounds(param, value);
      return value;
    }
    case 'bool':
      if (raw === 'true') return true;
      if (raw === 'false') return false;
      throw new Error(`“${raw}” must be true or false`);
    case 'select':
      if (!param.options.includes(raw)) throw new Error(`“${raw}” is not an available option`);
      return raw;
    case 'string':
      return raw;
    default:
      throw new Error(`${param.param_type} parameters cannot be swept`);
  }
}

export function buildValuesDomain(
  eligible: EligibleSweepParam,
  text: string,
): SweepParam {
  const values = tokens(text).map((raw) => parseValue(eligible.param, raw));
  if (values.length === 0) throw new Error('Enter at least one value');
  const seen: SweepValue[] = [];
  for (const value of values) {
    if (seen.includes(value)) throw new Error(`List ${String(value)} only once`);
    seen.push(value);
  }
  return { node_id: eligible.nodeId, param: eligible.param.name, values };
}

export interface SweepRangeInput {
  min: number;
  max: number;
  count: number;
  scale: SweepRange['scale'];
}

export function buildRangeDomain(
  eligible: EligibleSweepParam,
  input: SweepRangeInput,
): SweepParam {
  const type = eligible.param.param_type;
  if (type !== 'int' && type !== 'float') {
    throw new Error('Only numeric parameters support ranges');
  }
  if (![input.min, input.max, input.count].every(Number.isFinite)) {
    throw new Error('Range values must be finite numbers');
  }
  if (!Number.isInteger(input.count) || input.count < 1) {
    throw new Error('Range count must be a positive whole number');
  }
  if (input.max < input.min) throw new Error('Range maximum must be at least its minimum');
  if (input.scale === 'log' && input.min <= 0) {
    throw new Error('A logarithmic range needs a positive minimum');
  }
  if (type === 'int' && (!Number.isInteger(input.min) || !Number.isInteger(input.max))) {
    throw new Error('An integer range needs whole-number bounds');
  }
  assertBounds(eligible.param, input.min);
  assertBounds(eligible.param, input.max);
  return {
    node_id: eligible.nodeId,
    param: eligible.param.name,
    range: { ...input, type },
  };
}

function roundHalfAwayFromZero(value: number): number {
  return value >= 0 ? Math.floor(value + 0.5) : Math.ceil(value - 0.5);
}

/** Mirrors the backend expansion so a count preview includes int deduplication. */
export function expandSweepRange(range: SweepRange): number[] {
  let values: number[];
  if (range.count === 1) {
    values = [range.min];
  } else if (range.scale === 'log') {
    const start = Math.log10(range.min);
    const end = Math.log10(range.max);
    values = Array.from(
      { length: range.count },
      (_, index) => 10 ** (start + (end - start) * index / (range.count - 1)),
    );
    values[0] = range.min;
    values[values.length - 1] = range.max;
  } else {
    values = Array.from(
      { length: range.count },
      (_, index) => range.min + (range.max - range.min) * index / (range.count - 1),
    );
    values[0] = range.min;
    values[values.length - 1] = range.max;
  }
  if (range.type === 'int') values = values.map(roundHalfAwayFromZero);
  return [...new Set(values)];
}

export const DEFAULT_SWEEP_RUN_LIMIT = 32;

export interface SweepPreview {
  totalCombinations: number;
  variantCount: number;
  exceedsCap: boolean;
}

export function previewSweepVariants(
  method: SweepMethod,
  params: readonly SweepParam[],
  samples: number | null,
  maxRuns = DEFAULT_SWEEP_RUN_LIMIT,
): SweepPreview {
  if (params.length === 0) throw new Error('Choose at least one parameter');
  let totalCombinations = 1;
  for (const param of params) {
    const size = param.values?.length ?? (param.range ? expandSweepRange(param.range).length : 0);
    if (size < 1) throw new Error(`${param.node_id}.${param.param} needs at least one value`);
    totalCombinations *= size;
  }
  let variantCount = totalCombinations;
  if (method === 'random') {
    if (!Number.isInteger(samples) || (samples ?? 0) < 1) {
      throw new Error('Random samples must be a positive whole number');
    }
    if ((samples as number) > totalCombinations) {
      throw new Error(`Requested ${samples} samples, but the search space has only ${totalCombinations}`);
    }
    variantCount = samples as number;
  }
  return { totalCombinations, variantCount, exceedsCap: variantCount > maxRuns };
}
