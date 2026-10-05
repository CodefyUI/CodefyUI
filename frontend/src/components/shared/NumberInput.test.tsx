import { describe, it, expect, vi } from 'vitest';
import { useState } from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { NumberInput } from './NumberInput';

// jsdom, like Chrome, reports a number input's value as "" while what is
// typed is not yet a number ("-", "-0.", "1e-"). Setting one of those through
// fireEvent.change therefore hands the component the same "" a student's
// keystroke does.

function renderInput(props: Partial<React.ComponentProps<typeof NumberInput>> = {}) {
  const onCommit = props.onCommit ?? vi.fn();
  const element = (p: Partial<React.ComponentProps<typeof NumberInput>>) => (
    <NumberInput value={5} {...p} onCommit={onCommit} />
  );
  const utils = render(element(props));
  const input = screen.getByRole('spinbutton') as HTMLInputElement;
  return {
    ...utils,
    input,
    onCommit,
    update: (next: Partial<React.ComponentProps<typeof NumberInput>>) =>
      utils.rerender(element(next)),
  };
}

describe('NumberInput — typing', () => {
  it('a half-typed minus commits nothing and stays on screen; -1 then commits', () => {
    const { input, onCommit } = renderInput({ value: 5 });
    fireEvent.change(input, { target: { value: '-' } });
    expect(onCommit).not.toHaveBeenCalled();
    expect(input.value).toBe('');
    fireEvent.change(input, { target: { value: '-1' } });
    expect(onCommit).toHaveBeenCalledTimes(1);
    expect(onCommit).toHaveBeenCalledWith(-1);
  });

  it('decimals and exponents in progress commit nothing until they are numbers', () => {
    const { input, onCommit, update } = renderInput({ value: 5 });
    fireEvent.change(input, { target: { value: '-0.' } });
    fireEvent.change(input, { target: { value: '1e-' } });
    expect(onCommit).not.toHaveBeenCalled();
    fireEvent.change(input, { target: { value: '1e-3' } });
    expect(onCommit).toHaveBeenCalledWith(0.001);
    // The stored value catches up; what is on screen already says it, so it
    // is left as typed.
    update({ value: 0.001 });
    expect(input.value).toBe('1e-3');
  });

  it('reads the whole number, not its leading digits', () => {
    const { input, onCommit } = renderInput({ value: 5 });
    fireEvent.change(input, { target: { value: '2.5e2' } });
    expect(onCommit).toHaveBeenCalledWith(250);
  });

  it('integer: a fraction commits nothing, 1e3 and -2 commit', () => {
    const { input, onCommit } = renderInput({ value: 5, integer: true });
    fireEvent.change(input, { target: { value: '1.5' } });
    expect(onCommit).not.toHaveBeenCalled();
    fireEvent.change(input, { target: { value: '1e3' } });
    expect(onCommit).toHaveBeenLastCalledWith(1000);
    fireEvent.change(input, { target: { value: '-2' } });
    expect(onCommit).toHaveBeenLastCalledWith(-2);
    expect(onCommit).toHaveBeenCalledTimes(2);
  });
});

describe('NumberInput — leaving the field', () => {
  it('a cleared field shows the stored value again and commits nothing', () => {
    const { input, onCommit } = renderInput({ value: 7 });
    fireEvent.change(input, { target: { value: '' } });
    fireEvent.blur(input);
    expect(input.value).toBe('7');
    expect(onCommit).not.toHaveBeenCalled();
  });

  it('an integer field left holding a fraction shows the stored value again', () => {
    const { input, onCommit } = renderInput({ value: 7, integer: true });
    fireEvent.change(input, { target: { value: '7.5' } });
    fireEvent.blur(input);
    expect(input.value).toBe('7');
    expect(onCommit).not.toHaveBeenCalled();
  });

  it('settles a differently written number to the stored one', () => {
    const { input, onCommit, update } = renderInput({ value: 1 });
    fireEvent.change(input, { target: { value: '1.50' } });
    expect(onCommit).toHaveBeenCalledWith(1.5);
    update({ value: 1.5 });
    expect(input.value).toBe('1.50');
    fireEvent.blur(input);
    expect(input.value).toBe('1.5');
  });
});

describe('NumberInput — changes from outside', () => {
  it('a new stored value (undo, load, another panel) replaces what is shown', () => {
    const { input, update } = renderInput({ value: 5 });
    update({ value: 9 });
    expect(input.value).toBe('9');
  });

  it('a new stored value replaces a half-typed minus', () => {
    const { input, update } = renderInput({ value: 5 });
    fireEvent.change(input, { target: { value: '-' } });
    update({ value: 9 });
    expect(input.value).toBe('9');
  });

  it('a re-render with the same stored value keeps a half-typed minus', () => {
    // The rule that fixes the bug: resync when the stored value CHANGES,
    // never because it differs from what is typed.
    const { input, update } = renderInput({ value: 5 });
    fireEvent.change(input, { target: { value: '-' } });
    update({ value: 5 });
    expect(input.value).toBe('');
  });

  it('round-trips through a parent that stores what it is given', () => {
    function Stateful() {
      const [value, setValue] = useState<number>(3);
      return (
        <>
          <NumberInput value={value} onCommit={setValue} />
          <output data-testid="stored">{String(value)}</output>
        </>
      );
    }
    render(<Stateful />);
    const input = screen.getByRole('spinbutton') as HTMLInputElement;
    fireEvent.change(input, { target: { value: '-' } });
    expect(input.value).toBe('');
    expect(screen.getByTestId('stored').textContent).toBe('3');
    fireEvent.change(input, { target: { value: '-1' } });
    expect(input.value).toBe('-1');
    expect(screen.getByTestId('stored').textContent).toBe('-1');
  });
});

describe('NumberInput — what it renders', () => {
  it('is a number input carrying min, max, step, class, style and label', () => {
    renderInput({
      value: 2,
      min: 0,
      max: 10,
      step: 'any',
      className: 'cls',
      style: { width: 40 },
      'aria-label': 'Units',
    });
    const input = screen.getByLabelText('Units') as HTMLInputElement;
    expect(input.type).toBe('number');
    expect(input.min).toBe('0');
    expect(input.max).toBe('10');
    expect(input.step).toBe('any');
    expect(input.className).toBe('cls');
    expect(input.style.width).toBe('40px');
    expect(input.value).toBe('2');
  });

  it('shows nothing for a value that is not set', () => {
    const { input } = renderInput({ value: undefined });
    expect(input.value).toBe('');
  });

  it('accepts a stored numeric string', () => {
    const { input, onCommit } = renderInput({ value: '12' });
    expect(input.value).toBe('12');
    fireEvent.change(input, { target: { value: '13' } });
    expect(onCommit).toHaveBeenCalledWith(13);
  });

  it('renders a stored NaN once rather than re-syncing forever', () => {
    // NaN !== NaN: a plain inequality would see a "new" value on every
    // render and loop.
    const { input, update } = renderInput({ value: NaN });
    update({ value: NaN });
    expect(input.value).toBe('');
  });
});
