import { describe, it, expect, afterEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { TensorGridView } from './TensorGridView';
import type { TensorOutput } from '../../types';

function makeTensor(partial: Partial<TensorOutput> & Pick<TensorOutput, 'full_shape' | 'sliced_shape' | 'values'>): TensorOutput {
  return {
    type: 'tensor',
    run_id: 'r1',
    node_id: 'n1',
    port: 'out',
    dtype: 'float32',
    slice: ':',
    truncated: false,
    ...partial,
  };
}

afterEach(() => {
  // no spies, but keep the contract uniform
});

describe('TensorGridView', () => {
  it('renders a 2D grid with cells and stat header', () => {
    const tensor = makeTensor({
      full_shape: [2, 2],
      sliced_shape: [2, 2],
      values: [
        [1, 2.12345],
        [0.0001, 0],
      ],
      min: -1,
      max: 5,
      mean: 1.5,
    });
    render(<TensorGridView tensor={tensor} />);
    expect(screen.getByText('shape [2, 2]')).toBeInTheDocument();
    expect(screen.getByText('float32')).toBeInTheDocument();
    // headerStats joined: min/max/mean present
    expect(screen.getByText(/min -1 · max 5 · mean 1\.5/)).toBeInTheDocument();
    // integer formatting
    expect(screen.getByText('1')).toBeInTheDocument();
    // fixed(4) formatting
    expect(screen.getByText('2.1235')).toBeInTheDocument();
    // exponential formatting for tiny non-zero
    expect(screen.getByText('1.00e-4')).toBeInTheDocument();
  });

  it('renders a 1D row tensor', () => {
    const tensor = makeTensor({
      full_shape: [3],
      sliced_shape: [3],
      values: [10, 20, 30],
    });
    render(<TensorGridView tensor={tensor} />);
    expect(screen.getByText('10')).toBeInTheDocument();
    expect(screen.getByText('20')).toBeInTheDocument();
    expect(screen.getByText('30')).toBeInTheDocument();
    // no stats header when min/max/mean undefined
    expect(screen.queryByText(/min/)).not.toBeInTheDocument();
  });

  it('renders a scalar (non-array drilled value)', () => {
    const tensor = makeTensor({
      full_shape: [],
      sliced_shape: [],
      values: 42,
    });
    render(<TensorGridView tensor={tensor} />);
    expect(screen.getByText('42')).toBeInTheDocument();
  });

  it('renders the optional label', () => {
    const tensor = makeTensor({ full_shape: [1], sliced_shape: [1], values: [1] });
    render(<TensorGridView tensor={tensor} label="MyLabel" />);
    expect(screen.getByText('MyLabel')).toBeInTheDocument();
  });

  it('shows leading-dim selectors for rank > 2 and drills on change', () => {
    // rank 3 → 1 leading dim of size 2; each slice is a distinct 2x2 grid
    const tensor = makeTensor({
      full_shape: [2, 2, 2],
      sliced_shape: [2, 2, 2],
      values: [
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ],
    });
    render(<TensorGridView tensor={tensor} />);
    // first slice shown by default
    expect(screen.getByText('4')).toBeInTheDocument();
    expect(screen.queryByText('8')).not.toBeInTheDocument();
    const select = screen.getByRole('combobox');
    // 2 options for dimSize 2
    expect(screen.getAllByRole('option')).toHaveLength(2);
    fireEvent.change(select, { target: { value: '1' } });
    // now second slice
    expect(screen.getByText('8')).toBeInTheDocument();
  });

  it('falls back to dimSize 1 when sliced_shape entry missing', () => {
    // rank 3 → leadingCount 1, but sliced_shape only has 2 entries so dim 0 lookup is undefined
    const tensor = makeTensor({
      full_shape: [1, 1, 1],
      sliced_shape: [], // forces tensor.sliced_shape[dim] ?? 1
      values: [[[9]]],
    });
    // rank derived from sliced_shape.length === 0 → leadingCount 0, so no selectors.
    // Use a case where rank>2 but sliced_shape[dim] is undefined for the dim index.
    render(<TensorGridView tensor={tensor} />);
    // With sliced_shape [], rank=0 → grid drills nothing; values is [[[9]]] (array of arrays) → is2D
    expect(screen.getByText('9')).toBeInTheDocument();
  });

  it('renders dimSize-1 select when sliced_shape lacks the dim index', () => {
    // rank 3 via sliced_shape length 3, but the dim value at index 0 is undefined
    // (sparse hole) → exercises the `tensor.sliced_shape[dim] ?? 1` fallback.
    const slicedShape: number[] = [];
    slicedShape.length = 3; // [<empty>, <empty>, <empty>] → length 3, index 0 undefined
    const tensor = makeTensor({
      full_shape: [1, 1, 1],
      sliced_shape: slicedShape,
      values: [[[7]]],
    });
    render(<TensorGridView tensor={tensor} />);
    // leadingCount = 1 → one select; dimSize falls back to 1 → one option
    expect(screen.getAllByRole('option')).toHaveLength(1);
    expect(screen.getByText('7')).toBeInTheDocument();
  });

  it('applies heat highlight background to 2D cells', () => {
    const tensor = makeTensor({
      full_shape: [1, 2],
      sliced_shape: [1, 2],
      values: [[1, 2]],
    });
    const { container } = render(
      <TensorGridView tensor={tensor} highlight={(_i, j) => (j === 1 ? 1 : 0)} />,
    );
    const cells = container.querySelectorAll('td');
    // second cell (intensity 1 → alpha min(0.75,1)=0.75) gets a background; first (0) gets undefined
    expect(cells[1].style.background).toContain('rgba(255, 140, 0');
    expect(cells[0].style.background).toBe('');
  });

  it('applies heat highlight to 1D cells', () => {
    const tensor = makeTensor({
      full_shape: [2],
      sliced_shape: [2],
      values: [1, 2],
    });
    const { container } = render(
      <TensorGridView tensor={tensor} highlight={() => 0.5} />,
    );
    const cells = container.querySelectorAll('td');
    expect(cells[0].style.background).toContain('rgba(255, 140, 0');
  });

  it('formats boolean, null and string cell values', () => {
    const tensor = makeTensor({
      full_shape: [4],
      sliced_shape: [4],
      values: [true, false, null, 'hi'],
    });
    render(<TensorGridView tensor={tensor} />);
    expect(screen.getByText('T')).toBeInTheDocument();
    expect(screen.getByText('F')).toBeInTheDocument();
    expect(screen.getByText('·')).toBeInTheDocument();
    expect(screen.getByText('hi')).toBeInTheDocument();
  });

  it('drillTo2D breaks early on non-array while drilling and returns empty for empty slice', () => {
    // rank 3, leading [0]; values is a number at top so drilling breaks immediately → scalar path
    const scalarish = makeTensor({
      full_shape: [2, 2, 2],
      sliced_shape: [2, 2, 2],
      values: 5, // not an array → drilling `cur[i]` guard hits `!Array.isArray(cur) break`
    });
    render(<TensorGridView tensor={scalarish} />);
    expect(screen.getByText('5')).toBeInTheDocument();
  });

  it('renders empty array slice as no rows (length 0 branch)', () => {
    const empty = makeTensor({
      full_shape: [0],
      sliced_shape: [0],
      values: [],
    });
    const { container } = render(<TensorGridView tensor={empty} />);
    // grid === [] → is1D false (length not >0 for 2D check), is1D true? is2D requires length>0.
    // is1D = Array.isArray(grid) && !is2D → true; renders empty <tr> with no <td>
    expect(container.querySelectorAll('td')).toHaveLength(0);
  });
});
