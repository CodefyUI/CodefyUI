import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { NonTensorView } from './PortGroup';
import { makeHighlight, shapesEqual } from './diff';
import { useI18n } from '../../i18n';
import type { OutputData, TensorOutput } from '../../types';

function tensor(
  partial: Partial<TensorOutput> & Pick<TensorOutput, 'full_shape' | 'values'>,
): TensorOutput {
  return {
    type: 'tensor',
    run_id: 'r',
    node_id: 'n',
    port: 'p',
    dtype: 'float32',
    slice: ':',
    sliced_shape: partial.full_shape,
    truncated: false,
    ...partial,
  };
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
});

describe('NonTensorView', () => {
  it('renders scalar values with an optional label', () => {
    const sc: OutputData = { type: 'scalar', run_id: 'r', node_id: 'n', port: 'p', value: 3.14 };
    render(<NonTensorView value={sc} label="In" />);
    expect(screen.getByText('In')).toBeInTheDocument();
    expect(screen.getByText('scalar')).toBeInTheDocument();
    expect(screen.getByText('3.14')).toBeInTheDocument();
  });

  it('renders string values', () => {
    const s: OutputData = { type: 'string', run_id: 'r', node_id: 'n', port: 'p', value: 'hello' };
    render(<NonTensorView value={s} />);
    expect(screen.getByText('hello')).toBeInTheDocument();
  });

  it('renders model values with params formatting', () => {
    const m: OutputData = {
      type: 'model',
      run_id: 'r',
      node_id: 'n',
      port: 'p',
      class: 'Linear',
      params: 12345,
      trainable: 12345,
      repr: 'Linear(...)',
    };
    render(<NonTensorView value={m} />);
    expect(screen.getByText(/Linear/)).toBeInTheDocument();
    expect(screen.getByText(/12,345/)).toBeInTheDocument();
  });

  it('falls back to Module and ? when model class/params are missing', () => {
    const m = {
      type: 'model',
      run_id: 'r',
      node_id: 'n',
      port: 'p',
      trainable: 0,
      repr: '',
    } as unknown as OutputData;
    render(<NonTensorView value={m} />);
    expect(screen.getByText(/Module · params \?/)).toBeInTheDocument();
  });

  it('renders repr for generic/list types', () => {
    const g: OutputData = {
      type: 'list',
      run_id: 'r',
      node_id: 'n',
      port: 'p',
      length: 0,
      repr: 'list(empty)',
    };
    render(<NonTensorView value={g} />);
    expect(screen.getByText('list(empty)')).toBeInTheDocument();
  });

  it('falls back to the type name when no repr exists', () => {
    const g = { type: 'weird', run_id: 'r', node_id: 'n', port: 'p' } as unknown as OutputData;
    render(<NonTensorView value={g} />);
    expect(screen.getAllByText('weird').length).toBeGreaterThanOrEqual(1);
  });
});

describe('diff helpers', () => {
  it('shapesEqual compares length and every dim', () => {
    expect(shapesEqual([2, 3], [2, 3])).toBe(true);
    expect(shapesEqual([2, 3], [2, 4])).toBe(false);
    expect(shapesEqual([2], [2, 1])).toBe(false);
  });

  it('makeHighlight scores differing cells and zeroes equal ones', () => {
    const inT = tensor({ full_shape: [2, 2], values: [[1, 2], [3, 4]] });
    const outT = tensor({ full_shape: [2, 2], values: [[1, 9], [3, 4]] });
    const fn = makeHighlight(inT, outT);
    expect(fn).toBeDefined();
    expect(fn!(0, 1)).toBeGreaterThan(0);
    expect(fn!(1, 0)).toBe(0);
  });

  it('makeHighlight returns undefined for non-array values', () => {
    const inT = tensor({ full_shape: [], values: 5 });
    const outT = tensor({ full_shape: [], values: 7 });
    expect(makeHighlight(inT, outT)).toBeUndefined();
  });

  it('makeHighlight handles 1D tensors and non-number cells', () => {
    const inT = tensor({ full_shape: [3], values: [1, 'x', 3] });
    const outT = tensor({ full_shape: [3], values: [2, 'y', 3] });
    const fn = makeHighlight(inT, outT);
    expect(fn).toBeDefined();
    expect(fn!(0, 0)).toBeGreaterThan(0);
    expect(fn!(0, 1)).toBe(0);
  });

  it('makeHighlight unwraps 3D+ values to the last two dims', () => {
    const inT = tensor({ full_shape: [1, 2, 2], values: [[[1, 2], [3, 4]]] });
    const outT = tensor({ full_shape: [1, 2, 2], values: [[[1, 8], [3, 4]]] });
    const fn = makeHighlight(inT, outT);
    expect(fn).toBeDefined();
    expect(fn!(0, 1)).toBeGreaterThan(0);
  });
});
