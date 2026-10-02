import type { SweepMethod, SweepParam, SweepRange, SweepValue } from '../../api/rest';
import type { AppNode, NodeDefinition, ParamDefinition } from '../../types';

const SWEEPABLE_TYPES = new Set<ParamDefinition['param_type']>([
  'int', 'float', 'bool', 'select',
]);

/**
 * The defaults of the server's three sweep caps (`backend/app/config.py`).
 *
 * Each is a setting a deployment may raise -- CODEFYUI_MAX_SWEEP_RUNS,
 * CODEFYUI_MAX_SWEEP_PARAMS, CODEFYUI_MAX_SWEEP_DOMAIN -- and no route reports
 * the live value, so going past one is a warning here and the refusal is the
 * server's. Blocking on a default would make a raised cap unusable from the
 * editor.
 */
export const SWEEP_LIMIT_DEFAULTS = { runs: 32, params: 4, values: 32 } as const;

export type SweepInputErrorCode =
  | 'notNumber'
  | 'notWhole'
  | 'belowMin'
  | 'aboveMax'
  | 'notBool'
  | 'notOption'
  | 'notSweepable'
  | 'noValues'
  | 'repeated'
  | 'rangeNumericOnly'
  | 'rangeNotFinite'
  | 'rangeCount'
  | 'rangeOrder'
  | 'rangeLogMin'
  | 'rangeWholeBounds'
  | 'noParams'
  | 'emptyDomain'
  | 'samples'
  | 'samplesExceedSpace'
  | 'seed';

/** A refused input, as a code the dialog translates (`sweeps.error.<code>`). */
export class SweepInputError extends Error {
  constructor(
    readonly code: SweepInputErrorCode,
    readonly vars: Record<string, string | number> = {},
  ) {
    super(code);
    this.name = 'SweepInputError';
  }
}

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

/**
 * A column title for one swept address: the node's label (else its type) and
 * the param. Null unless the open graph has exactly one node with that id and
 * its type has that param -- a sweep may come from another graph, whose ids
 * say nothing about this one.
 */
export function sweepParamTitle(
  address: { node_id: string; param: string },
  definitions: readonly NodeDefinition[],
  nodes: readonly Pick<AppNode, 'id' | 'data'>[],
): string | null {
  const matches = nodes.filter((node) => node.id === address.node_id);
  if (matches.length !== 1) return null;
  const { data } = matches[0];
  const definition = definitions.find((candidate) => candidate.node_name === data.type);
  if (!definition?.params.some((param) => param.name === address.param)) return null;
  return `${data.label || data.type} · ${address.param}`;
}

function tokens(text: string): string[] {
  return text.split(/[\n,]/).map((value) => value.trim()).filter(Boolean);
}

function numberValue(raw: string, kind: 'int' | 'float'): number {
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new SweepInputError('notNumber', { value: raw });
  if (kind === 'int' && !Number.isInteger(value)) {
    throw new SweepInputError('notWhole', { value: raw });
  }
  return value;
}

function assertBounds(param: ParamDefinition, value: number): void {
  if (param.min_value !== null && value < param.min_value) {
    throw new SweepInputError('belowMin', { name: param.name, min: param.min_value });
  }
  if (param.max_value !== null && value > param.max_value) {
    throw new SweepInputError('aboveMax', { name: param.name, max: param.max_value });
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
      throw new SweepInputError('notBool', { value: raw });
    case 'select':
      if (!param.options.includes(raw)) throw new SweepInputError('notOption', { value: raw });
      return raw;
    default:
      throw new SweepInputError('notSweepable', { type: param.param_type });
  }
}

export function buildValuesDomain(
  eligible: EligibleSweepParam,
  text: string,
): SweepParam {
  const values = tokens(text).map((raw) => parseValue(eligible.param, raw));
  if (values.length === 0) throw new SweepInputError('noValues');
  // A Set, not a list scan: a pasted list is parsed on every keystroke.
  const seen = new Set<SweepValue>();
  for (const value of values) {
    if (seen.has(value)) throw new SweepInputError('repeated', { value: String(value) });
    seen.add(value);
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
  if (type !== 'int' && type !== 'float') throw new SweepInputError('rangeNumericOnly');
  if (![input.min, input.max, input.count].every(Number.isFinite)) {
    throw new SweepInputError('rangeNotFinite');
  }
  if (!Number.isInteger(input.count) || input.count < 1) throw new SweepInputError('rangeCount');
  if (input.max < input.min) throw new SweepInputError('rangeOrder');
  if (input.scale === 'log' && input.min <= 0) throw new SweepInputError('rangeLogMin');
  if (type === 'int' && (!Number.isInteger(input.min) || !Number.isInteger(input.max))) {
    throw new SweepInputError('rangeWholeBounds');
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

/**
 * Mirrors the backend expansion so a count preview includes int deduplication.
 * Allocates `count` points: callers bound it (see `previewSweepVariants`).
 */
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

/** A request past one of the default caps; `index` is the param's position. */
export type SweepLimitWarning =
  | { limit: 'runs'; count: number }
  | { limit: 'params'; count: number }
  | { limit: 'values'; index: number; count: number };

export interface SweepPreview {
  /** Null when a range is past the default domain cap and so not expanded. */
  totalCombinations: number | null;
  variantCount: number | null;
  warnings: SweepLimitWarning[];
}

export function previewSweepVariants(
  method: SweepMethod,
  params: readonly SweepParam[],
  samples: number | null,
): SweepPreview {
  if (params.length === 0) throw new SweepInputError('noParams');
  const warnings: SweepLimitWarning[] = [];
  if (params.length > SWEEP_LIMIT_DEFAULTS.params) {
    warnings.push({ limit: 'params', count: params.length });
  }
  let totalCombinations: number | null = 1;
  params.forEach((param, index) => {
    // The server caps what was WRITTEN -- a list's length, a range's count --
    // before it deduplicates, so the warning compares the same number.
    const written = param.values?.length ?? param.range?.count ?? 0;
    if (written < 1) {
      throw new SweepInputError('emptyDomain', { address: `${param.node_id}.${param.param}` });
    }
    if (written > SWEEP_LIMIT_DEFAULTS.values) {
      warnings.push({ limit: 'values', index, count: written });
    }
    if (param.values) {
      if (totalCombinations !== null) totalCombinations *= param.values.length;
    } else if (written > SWEEP_LIMIT_DEFAULTS.values) {
      // Expanding a range allocates every point, on every keystroke; past the
      // default cap its exact size is not worth that.
      totalCombinations = null;
    } else if (totalCombinations !== null) {
      totalCombinations *= expandSweepRange(param.range!).length;
    }
  });
  let variantCount = totalCombinations;
  if (method === 'random') {
    if (!Number.isInteger(samples) || (samples ?? 0) < 1) throw new SweepInputError('samples');
    if (totalCombinations !== null && (samples as number) > totalCombinations) {
      throw new SweepInputError('samplesExceedSpace', { samples: samples as number, total: totalCombinations });
    }
    variantCount = samples as number;
  }
  if (variantCount !== null && variantCount > SWEEP_LIMIT_DEFAULTS.runs) {
    warnings.push({ limit: 'runs', count: variantCount });
  }
  return { totalCombinations, variantCount, warnings };
}
