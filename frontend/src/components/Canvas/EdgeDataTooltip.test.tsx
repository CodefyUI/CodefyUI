import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { EdgeDataTooltip } from './EdgeDataTooltip';
import type { OutputSummary } from '../../types';

afterEach(() => {
  vi.restoreAllMocks();
});

function renderTooltip(summary: OutputSummary, onClose = vi.fn()) {
  const utils = render(
    <EdgeDataTooltip
      x={100}
      y={50}
      sourceLabel="Src"
      targetLabel="Dst"
      portName="out"
      summary={summary}
      onClose={onClose}
    />,
  );
  return { ...utils, onClose };
}

describe('EdgeDataTooltip', () => {
  it('renders the title with source/target/port and the always-present Type row', () => {
    renderTooltip({ type: 'Tensor' });
    expect(screen.getByText(/Src/)).toBeTruthy();
    expect(screen.getByText(/Dst/)).toBeTruthy();
    expect(screen.getByText(/\(out\)/)).toBeTruthy();
    expect(screen.getByText('Type')).toBeTruthy();
    expect(screen.getByText('Tensor')).toBeTruthy();
  });

  it('positions the tooltip via left/top inline styles', () => {
    const { container } = renderTooltip({ type: 'T' });
    const el = container.firstChild as HTMLElement;
    expect(el.style.left).toBe('100px');
    expect(el.style.top).toBe('50px');
  });

  it('renders every optional row when all fields are present', () => {
    renderTooltip({
      type: 'Tensor',
      shape: [2, 3],
      dtype: 'float32',
      min: 0,
      max: 9,
      mean: 4.5,
      class: 'Linear',
      params: 1000000,
      trainable: 500000,
      value: 42,
      repr: 'tensor(...)',
    });
    expect(screen.getByText('Shape')).toBeTruthy();
    expect(screen.getByText('[2, 3]')).toBeTruthy();
    expect(screen.getByText('Dtype')).toBeTruthy();
    expect(screen.getByText('float32')).toBeTruthy();
    expect(screen.getByText('Min')).toBeTruthy();
    expect(screen.getByText('Max')).toBeTruthy();
    expect(screen.getByText('Mean')).toBeTruthy();
    expect(screen.getByText('Class')).toBeTruthy();
    expect(screen.getByText('Linear')).toBeTruthy();
    expect(screen.getByText('Params')).toBeTruthy();
    // toLocaleString formats with grouping separators.
    expect(screen.getByText((1000000).toLocaleString())).toBeTruthy();
    expect(screen.getByText('Trainable')).toBeTruthy();
    expect(screen.getByText((500000).toLocaleString())).toBeTruthy();
    // Both `value` and `repr` produce a "Value" label.
    expect(screen.getAllByText('Value').length).toBe(2);
    expect(screen.getByText('tensor(...)')).toBeTruthy();
  });

  it('omits optional rows that are undefined / falsy (branch: no extra rows)', () => {
    renderTooltip({ type: 'Scalar' });
    expect(screen.queryByText('Shape')).toBeNull();
    expect(screen.queryByText('Dtype')).toBeNull();
    expect(screen.queryByText('Min')).toBeNull();
    expect(screen.queryByText('Max')).toBeNull();
    expect(screen.queryByText('Mean')).toBeNull();
    expect(screen.queryByText('Class')).toBeNull();
    expect(screen.queryByText('Params')).toBeNull();
    expect(screen.queryByText('Trainable')).toBeNull();
    expect(screen.queryByText('Value')).toBeNull();
  });

  it('treats min/max/mean === 0 as present (undefined check, not falsy)', () => {
    renderTooltip({ type: 'T', min: 0, max: 0, mean: 0, value: 0 });
    expect(screen.getByText('Min')).toBeTruthy();
    expect(screen.getByText('Max')).toBeTruthy();
    expect(screen.getByText('Mean')).toBeTruthy();
    // Value row present for value === 0 (uses !== undefined).
    expect(screen.getByText('Value')).toBeTruthy();
  });

  it('calls onClose when clicking outside the tooltip', () => {
    const { onClose } = renderTooltip({ type: 'T' });
    fireEvent.mouseDown(document.body);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('does NOT call onClose when clicking inside the tooltip', () => {
    const { onClose, container } = renderTooltip({ type: 'T' });
    const inside = container.querySelector('div') as HTMLElement;
    fireEvent.mouseDown(inside);
    expect(onClose).not.toHaveBeenCalled();
  });

  it('calls onClose on Escape key, ignores other keys', () => {
    const { onClose } = renderTooltip({ type: 'T' });
    fireEvent.keyDown(document, { key: 'a' });
    expect(onClose).not.toHaveBeenCalled();
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('removes its listeners on unmount (no onClose after unmount)', () => {
    const { onClose, unmount } = renderTooltip({ type: 'T' });
    unmount();
    fireEvent.keyDown(document, { key: 'Escape' });
    fireEvent.mouseDown(document.body);
    expect(onClose).not.toHaveBeenCalled();
  });
});
