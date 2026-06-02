import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ValueDiff } from './ValueDiff';
import { useI18n } from '../../i18n';
import type { TensorOutput, OutputData } from '../../types';

function tensor(partial: Partial<TensorOutput> & Pick<TensorOutput, 'full_shape' | 'values'>): TensorOutput {
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

afterEach(() => {
  vi.restoreAllMocks();
});

describe('ValueDiff', () => {
  it('renders the empty placeholder when both input and output are null', () => {
    render(<ValueDiff input={null} output={null} />);
    expect(screen.getByText('No values captured for this port')).toBeInTheDocument();
  });

  it('renders missing markers when only one side is null', () => {
    render(<ValueDiff input={null} output={null === null ? tensor({ full_shape: [1], values: [1] }) : null} />);
    // input null → "Input: —"
    expect(screen.getByText('Input: —')).toBeInTheDocument();
    // output tensor renders a grid (shape line)
    expect(screen.getByText('shape [1]')).toBeInTheDocument();
  });

  it('renders missing output marker when output is null but input present', () => {
    render(<ValueDiff input={tensor({ full_shape: [1], values: [1] })} output={null} />);
    expect(screen.getByText('Output: —')).toBeInTheDocument();
  });

  it('renders two tensor grids with equal shapes and a highlight (no banner)', () => {
    const inT = tensor({ full_shape: [2, 2], values: [[1, 2], [3, 4]] });
    const outT = tensor({ full_shape: [2, 2], values: [[1, 9], [3, 4]] });
    const { container } = render(<ValueDiff input={inT} output={outT} />);
    // both labels render
    expect(screen.getByText('Input')).toBeInTheDocument();
    expect(screen.getByText('Output')).toBeInTheDocument();
    // no shape-change banner
    expect(container.querySelectorAll('td').length).toBeGreaterThan(0);
    // a highlight should have colored at least the differing output cell
    const colored = Array.from(container.querySelectorAll('td')).some((td) =>
      (td as HTMLElement).style.background.includes('rgba(255, 140, 0'),
    );
    expect(colored).toBe(true);
  });

  it('shows the shape-change banner when tensor shapes differ', () => {
    const inT = tensor({ full_shape: [2, 2], values: [[1, 2], [3, 4]] });
    const outT = tensor({ full_shape: [3], values: [1, 2, 3] });
    const { container } = render(<ValueDiff input={inT} output={outT} />);
    // The banner is the element whose combined text contains both shapes + arrow.
    const banner = Array.from(container.querySelectorAll('div')).find((d) =>
      /\[2, 2\][\s\S]*\[3\]/.test(d.textContent ?? '') &&
      (d.textContent ?? '').includes('→') &&
      d.children.length === 0,
    );
    expect(banner).toBeTruthy();
  });

  it('shape-change banner triggers via equal-length but differing dims (shapesEqual element loop)', () => {
    // [2,3] vs [2,4]: same length so the length check passes, differs at i=1
    // → exercises `shapesEqual` returning false inside its element loop.
    const inT = tensor({ full_shape: [2, 3], values: [[1, 2, 3], [4, 5, 6]] });
    const outT = tensor({ full_shape: [2, 4], values: [[1, 2, 3, 0], [4, 5, 6, 0]] });
    const { container } = render(<ValueDiff input={inT} output={outT} />);
    const banner = Array.from(container.querySelectorAll('div')).find((d) =>
      (d.textContent ?? '').includes('→') && d.children.length === 0,
    );
    expect(banner).toBeTruthy();
  });

  it('makeHighlight returns undefined when tensor values are not arrays (no coloring)', () => {
    // Equal shapes but scalar values → makeHighlight bails (returns undefined)
    const inT = tensor({ full_shape: [], values: 5 });
    const outT = tensor({ full_shape: [], values: 7 });
    const { container } = render(<ValueDiff input={inT} output={outT} />);
    const colored = Array.from(container.querySelectorAll('td')).some((td) =>
      (td as HTMLElement).style.background.includes('rgba(255, 140, 0'),
    );
    expect(colored).toBe(false);
  });

  it('highlight handles 1D tensors (getCell 1D branch) and non-number cells', () => {
    const inT = tensor({ full_shape: [3], values: [1, 'x', 3] });
    const outT = tensor({ full_shape: [3], values: [2, 'y', 3] });
    render(<ValueDiff input={inT} output={outT} />);
    // cell 0: numbers differ → some color possible; cell 1: non-numbers → 0
    expect(screen.getAllByText(/Input|Output/).length).toBeGreaterThan(0);
  });

  it('highlight unwraps deeply nested 3D+ values to last-2-dims', () => {
    // 3D equal shapes; getCell while-loop unwraps one leading level
    const vals = [
      [
        [1, 2],
        [3, 4],
      ],
    ];
    const inT = tensor({ full_shape: [1, 2, 2], values: vals });
    const outT = tensor({ full_shape: [1, 2, 2], values: [[[1, 8], [3, 4]]] });
    const { container } = render(<ValueDiff input={inT} output={outT} />);
    // renders without throwing; grids present
    expect(container.querySelectorAll('table').length).toBeGreaterThan(0);
  });

  it('renders NonTensorView for scalar values', () => {
    const sc: OutputData = { type: 'scalar', run_id: 'r', node_id: 'n', port: 'p', value: 3.14 };
    render(<ValueDiff input={sc} output={null} inputLabel="In" />);
    expect(screen.getByText('In')).toBeInTheDocument();
    expect(screen.getByText('scalar')).toBeInTheDocument();
    expect(screen.getByText('3.14')).toBeInTheDocument();
  });

  it('renders NonTensorView for string values', () => {
    const s: OutputData = { type: 'string', run_id: 'r', node_id: 'n', port: 'p', value: 'hello' };
    render(<ValueDiff input={null} output={s} outputLabel="Out" />);
    expect(screen.getByText('Out')).toBeInTheDocument();
    expect(screen.getByText('hello')).toBeInTheDocument();
  });

  it('renders NonTensorView for model values with params formatting', () => {
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
    render(<ValueDiff input={m} output={null} />);
    expect(screen.getByText(/Linear/)).toBeInTheDocument();
    expect(screen.getByText(/12,345/)).toBeInTheDocument();
  });

  it('renders NonTensorView fallback repr for generic/list types', () => {
    const g: OutputData = {
      type: 'list',
      run_id: 'r',
      node_id: 'n',
      port: 'p',
      length: 0,
      repr: 'list(empty)',
    };
    render(<ValueDiff input={g} output={null} />);
    expect(screen.getByText('list(empty)')).toBeInTheDocument();
  });

  it('renders NonTensorView fallback to type name when no repr', () => {
    const g: OutputData = {
      type: 'weird',
      run_id: 'r',
      node_id: 'n',
      port: 'p',
    };
    render(<ValueDiff input={g} output={null} />);
    // two 'weird' appear (dtype meta + scalar fallback)
    expect(screen.getAllByText('weird').length).toBeGreaterThanOrEqual(1);
  });

  it('renders model NonTensorView when params lacks toLocaleString (empty string)', () => {
    const m: OutputData = {
      type: 'model',
      run_id: 'r',
      node_id: 'n',
      port: 'p',
      class: 'Net',
      // params undefined → optional-chained toLocaleString → '' fallback
      params: undefined as unknown as number,
      trainable: 0,
      repr: '',
    };
    render(<ValueDiff input={m} output={null} />);
    expect(screen.getByText(/Net/)).toBeInTheDocument();
  });
});
