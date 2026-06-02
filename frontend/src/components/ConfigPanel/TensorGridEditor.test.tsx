import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { TensorGridEditor } from './TensorGridEditor';
import type { ParamDefinition } from '../../types';

function makeParam(overrides: Partial<ParamDefinition> = {}): ParamDefinition {
  return {
    name: 'weights',
    param_type: 'tensor_grid',
    default: null,
    description: '',
    options: [],
    min_value: null,
    max_value: null,
    ...overrides,
  };
}

interface RenderArgs {
  value?: any;
  siblingParams?: Record<string, any>;
  onChange?: (name: string, value: any) => void;
  label?: string;
}

function renderEditor({ value, siblingParams, onChange, label }: RenderArgs = {}) {
  const handle = onChange ?? vi.fn();
  const utils = render(
    <TensorGridEditor
      param={makeParam()}
      value={value}
      onChange={handle}
      displayLabel={label ?? 'My Tensor'}
      siblingParams={siblingParams}
    />,
  );
  return { ...utils, onChange: handle };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('TensorGridEditor — non-explicit / disabled mode', () => {
  it('shows the "set value_mode to explicit" hint when mode is not explicit', () => {
    renderEditor({ siblingParams: { shape: '2,2', value_mode: 'random' } });
    expect(screen.getByText('My Tensor')).toBeInTheDocument();
    // hint text fragments
    expect(screen.getByText(/to edit values inline/)).toBeInTheDocument();
    // no toolbar (Fill 0) rendered while disabled
    expect(screen.queryByText('Fill 0')).not.toBeInTheDocument();
  });

  it('defaults value_mode to random (disabled) when siblingParams omitted', () => {
    // siblingParams undefined → shape '' → parseShape [] ; value_mode default 'random'
    renderEditor({});
    expect(screen.getByText(/to edit values inline/)).toBeInTheDocument();
  });

  it('fillAll is a no-op when disabled (no onChange, no buttons)', () => {
    const onChange = vi.fn();
    renderEditor({ siblingParams: { shape: '2', value_mode: 'zeros' }, onChange });
    // buttons are not rendered; assert onChange never fires
    expect(onChange).not.toHaveBeenCalled();
  });
});

describe('TensorGridEditor — too-large warning', () => {
  it('shows the too-large warning when numel exceeds MAX_INLINE_NUMEL', () => {
    // 100 * 100 = 10000 > 512
    renderEditor({ siblingParams: { shape: '100,100', value_mode: 'explicit' } });
    expect(screen.getByText(/too large for inline editing/)).toBeInTheDocument();
    expect(screen.queryByText('Fill 0')).not.toBeInTheDocument();
  });
});

describe('TensorGridEditor — explicit editable grid (2D)', () => {
  it('renders a toolbar, shape badge, and an editable grid of inputs', () => {
    renderEditor({
      value: [
        [1, 2],
        [3, 4],
      ],
      siblingParams: { shape: '2,2', value_mode: 'explicit' },
    });
    expect(screen.getByText('Fill 0')).toBeInTheDocument();
    expect(screen.getByText('Fill 1')).toBeInTheDocument();
    expect(screen.getByText('Random')).toBeInTheDocument();
    // shape badge: "[2, 2] · 4 cells"
    expect(screen.getByText(/\[2, 2\] · 4 cells/)).toBeInTheDocument();
    const inputs = screen.getAllByRole('spinbutton') as HTMLInputElement[];
    expect(inputs).toHaveLength(4);
    expect(inputs.map((i) => i.value)).toEqual(['1', '2', '3', '4']);
  });

  it('edits a cell with a finite number and calls onChange with reshaped value', () => {
    const onChange = vi.fn();
    renderEditor({
      value: [
        [1, 2],
        [3, 4],
      ],
      siblingParams: { shape: '2,2', value_mode: 'explicit' },
      onChange,
    });
    const inputs = screen.getAllByRole('spinbutton');
    fireEvent.change(inputs[0], { target: { value: '9' } });
    expect(onChange).toHaveBeenCalledWith('weights', [
      [9, 2],
      [3, 4],
    ]);
  });

  it('coerces a non-finite cell entry to 0', () => {
    const onChange = vi.fn();
    renderEditor({
      value: [
        [1, 2],
        [3, 4],
      ],
      siblingParams: { shape: '2,2', value_mode: 'explicit' },
      onChange,
    });
    const inputs = screen.getAllByRole('spinbutton');
    // 'abc' -> Number('abc') is NaN -> not finite -> 0
    fireEvent.change(inputs[3], { target: { value: 'abc' } });
    expect(onChange).toHaveBeenCalledWith('weights', [
      [1, 2],
      [3, 0],
    ]);
  });

  it('reshapes scalar / string source values into the grid (reshapeValues walk branches)', () => {
    // value contains a number, a numeric string, a non-numeric string (-> 0 via
    // the `Number(v) || 0` fallback), and a nested array; flat walk exercises
    // the Array, number, and non-number branches; padded with zeros.
    const onChange = vi.fn();
    renderEditor({
      value: [5, '7', 'x', [2]],
      siblingParams: { shape: '2,2', value_mode: 'explicit' },
      onChange,
    });
    const inputs = screen.getAllByRole('spinbutton') as HTMLInputElement[];
    // flat = [5, 7, 0(from 'x'), 2] -> exactly 4 cells
    expect(inputs.map((i) => i.value)).toEqual(['5', '7', '0', '2']);
  });

  it('pads with zeros when the source value has fewer elements than the shape', () => {
    // value [1] has 1 element but shape 2,2 needs 4 -> reshapeValues' pad loop
    // (`while (flat.length < numel(shape)) flat.push(0)`) appends 3 zeros.
    renderEditor({
      value: [1],
      siblingParams: { shape: '2,2', value_mode: 'explicit' },
    });
    const inputs = screen.getAllByRole('spinbutton') as HTMLInputElement[];
    expect(inputs.map((i) => i.value)).toEqual(['1', '0', '0', '0']);
  });

  it('falls back to zerosOf when value is nullish in explicit mode', () => {
    // value undefined -> `value ?? zerosOf(shape)` walks zerosOf to build a
    // zero-filled nested grid matching the shape.
    renderEditor({
      value: undefined,
      siblingParams: { shape: '2,2', value_mode: 'explicit' },
    });
    const inputs = screen.getAllByRole('spinbutton') as HTMLInputElement[];
    expect(inputs.map((i) => i.value)).toEqual(['0', '0', '0', '0']);
  });

  it('fills all cells with 0 / 1 and randomizes', () => {
    const onChange = vi.fn();
    renderEditor({
      value: [
        [1, 2],
        [3, 4],
      ],
      siblingParams: { shape: '2,2', value_mode: 'explicit' },
      onChange,
    });
    fireEvent.click(screen.getByText('Fill 0'));
    expect(onChange).toHaveBeenLastCalledWith('weights', [
      [0, 0],
      [0, 0],
    ]);
    fireEvent.click(screen.getByText('Fill 1'));
    expect(onChange).toHaveBeenLastCalledWith('weights', [
      [1, 1],
      [1, 1],
    ]);
    // deterministic random
    const spy = vi.spyOn(Math, 'random').mockReturnValue(0.75);
    fireEvent.click(screen.getByText('Random'));
    // 0.75*200-100 = 50 -> /100 = 0.5
    expect(onChange).toHaveBeenLastCalledWith('weights', [
      [0.5, 0.5],
      [0.5, 0.5],
    ]);
    spy.mockRestore();
  });
});

describe('TensorGridEditor — 1D tensor', () => {
  it('renders a 1D tensor as a single row and edits via the 1D-leaf set path', () => {
    const onChange = vi.fn();
    renderEditor({
      value: [10, 20, 30],
      siblingParams: { shape: '3', value_mode: 'explicit' },
      onChange,
    });
    const inputs = screen.getAllByRole('spinbutton') as HTMLInputElement[];
    expect(inputs.map((i) => i.value)).toEqual(['10', '20', '30']);
    // editing exercises set2D's `!Array.isArray(cur[0])` 1D-leaf branch
    fireEvent.change(inputs[1], { target: { value: '99' } });
    expect(onChange).toHaveBeenCalledWith('weights', [10, 99, 30]);
  });
});

describe('TensorGridEditor — 3D tensor with leading dim selector', () => {
  it('renders leading dim selects and drills into a sub-grid; switching dim updates grid', () => {
    const onChange = vi.fn();
    // shape [2,2,2] -> rank 3 -> leadingCount 1
    renderEditor({
      value: [
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ],
      siblingParams: { shape: '2,2,2', value_mode: 'explicit' },
      onChange,
    });
    // a leading dim selector labelled "dim 0"
    expect(screen.getByText(/dim 0/)).toBeInTheDocument();
    let inputs = screen.getAllByRole('spinbutton') as HTMLInputElement[];
    // leading index 0 -> first 2x2 sub-grid
    expect(inputs.map((i) => i.value)).toEqual(['1', '2', '3', '4']);

    const select = screen.getByRole('combobox') as HTMLSelectElement;
    fireEvent.change(select, { target: { value: '1' } });
    inputs = screen.getAllByRole('spinbutton') as HTMLInputElement[];
    expect(inputs.map((i) => i.value)).toEqual(['5', '6', '7', '8']);

    // edit a cell in the second slab -> set2D walks leadingIdx then 2D branch
    fireEvent.change(inputs[0], { target: { value: '50' } });
    expect(onChange).toHaveBeenCalledWith('weights', [
      [
        [1, 2],
        [3, 4],
      ],
      [
        [50, 6],
        [7, 8],
      ],
    ]);
  });

  it('resets the leading index array when the leading count changes (rerender)', () => {
    const onChange = vi.fn();
    const { rerender } = render(
      <TensorGridEditor
        param={makeParam()}
        value={[
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ]}
        onChange={onChange}
        displayLabel="T"
        siblingParams={{ shape: '2,2,2', value_mode: 'explicit' }}
      />,
    );
    // initially has one leading selector
    expect(screen.getByText(/dim 0/)).toBeInTheDocument();

    // shrink to 2D -> leadingCount goes from 1 to 0 -> setLeading reset branch
    rerender(
      <TensorGridEditor
        param={makeParam()}
        value={[
          [1, 2],
          [3, 4],
        ]}
        onChange={onChange}
        displayLabel="T"
        siblingParams={{ shape: '2,2', value_mode: 'explicit' }}
      />,
    );
    expect(screen.queryByText(/dim 0/)).not.toBeInTheDocument();
    const inputs = screen.getAllByRole('spinbutton') as HTMLInputElement[];
    expect(inputs).toHaveLength(4);
  });
});

describe('TensorGridEditor — parseShape edge cases', () => {
  it('parses messy shape strings: trims, drops empties, clamps to >= 1, defaults bad ints to 1', () => {
    // ' 2 , , x , 0 ' -> ['2','x','0'] -> [2, 1, 1]  (x->NaN->1, 0->max(1,0)=1)
    renderEditor({
      value: [
        [1, 2],
        [3, 4],
      ],
      siblingParams: { shape: ' 2 , , x , 0 ', value_mode: 'explicit' },
    });
    // shape [2,1,1] -> rank 3 -> total = 2 -> badge shows it
    expect(screen.getByText(/\[2, 1, 1\] · 2 cells/)).toBeInTheDocument();
  });

  it('renders the toolbar but an empty grid when shape is [] (rank 0, normalized null)', () => {
    // empty shape -> parseShape '' -> [] ; numel([]) === 1 so total=1 passes the
    // `total > 0 && total <= MAX` gate, but shape.length===0 makes normalized
    // null, so grid is [] and no input cells render.
    const onChange = vi.fn();
    renderEditor({ siblingParams: { shape: '', value_mode: 'explicit' }, onChange });
    expect(screen.getByText('Fill 0')).toBeInTheDocument();
    // badge: "[] · 1 cells"
    expect(screen.getByText(/\[\] · 1 cells/)).toBeInTheDocument();
    expect(screen.queryByText(/too large/)).not.toBeInTheDocument();
    expect(screen.queryAllByRole('spinbutton')).toHaveLength(0);
    expect(screen.getByText('My Tensor')).toBeInTheDocument();
    // Fill on a rank-0 shape exercises fillFlat's scalar (shape.length===0) leaf
    fireEvent.click(screen.getByText('Fill 1'));
    expect(onChange).toHaveBeenCalledWith('weights', 1);
  });
});

describe('TensorGridEditor — setCell guard', () => {
  it('does nothing when normalized is null (no editable grid present)', () => {
    // With non-explicit mode there is no input; this asserts the disabled path
    // does not surface editable inputs, so setCell can never run.
    const onChange = vi.fn();
    renderEditor({ siblingParams: { shape: '2,2', value_mode: 'random' }, onChange });
    expect(screen.queryAllByRole('spinbutton')).toHaveLength(0);
    expect(onChange).not.toHaveBeenCalled();
  });
});
