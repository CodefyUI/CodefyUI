import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, fireEvent, within } from '@testing-library/react';
import { TokenChip } from './TokenChip';

afterEach(() => {
  vi.restoreAllMocks();
});

describe('TokenChip — display glyphs', () => {
  it('replaces space, newline, and tab with visible glyphs', () => {
    const { container } = render(<TokenChip token={' \n\t'} index={0} />);
    const chip = container.querySelector('span');
    // space → ·, newline → ↵, tab → →
    expect(chip?.textContent).toBe('·↵→');
  });

  it('renders a lone middle-dot when the display text is empty', () => {
    const { container } = render(<TokenChip token="" index={0} />);
    expect(container.querySelector('span')?.textContent).toBe('·');
  });

  it('renders the raw token text when it has no whitespace', () => {
    const { container } = render(<TokenChip token="cat" index={0} />);
    expect(container.querySelector('span')?.textContent).toBe('cat');
  });
});

describe('TokenChip — animation', () => {
  it('adds the animated class and an animation-delay when animated', () => {
    const { container } = render(<TokenChip token="x" index={3} animated />);
    const chip = container.querySelector('span') as HTMLElement;
    expect(chip.className).toMatch(/animated/i);
    // delay = index * 30ms
    expect(chip.style.animationDelay).toBe('90ms');
  });

  it('omits animation-delay when not animated (default)', () => {
    const { container } = render(<TokenChip token="x" index={3} />);
    const chip = container.querySelector('span') as HTMLElement;
    expect(chip.style.animationDelay).toBe('');
  });
});

describe('TokenChip — hover tooltip', () => {
  it('shows the tooltip on mouse enter and hides it on leave', () => {
    const { container } = render(<TokenChip token="hi" index={0} />);
    const chip = container.querySelector('span') as HTMLElement;
    fireEvent.mouseEnter(chip);
    const tip = document.body.querySelector('[class*="tooltip"]');
    expect(tip).toBeTruthy();
    expect(within(tip as HTMLElement).getByText('token')).toBeTruthy();
    // token row shows JSON.stringify(token).
    expect(within(tip as HTMLElement).getByText('"hi"')).toBeTruthy();
    // bytes row is always present.
    expect(within(tip as HTMLElement).getByText('bytes')).toBeTruthy();
    fireEvent.mouseLeave(chip);
    expect(document.body.querySelector('[class*="tooltip"]')).toBeNull();
  });

  it('renders the id row only when id is provided', () => {
    const { rerender, container } = render(<TokenChip token="hi" index={0} id={42} />);
    let chip = container.querySelector('span') as HTMLElement;
    fireEvent.mouseEnter(chip);
    let tip = document.body.querySelector('[class*="tooltip"]') as HTMLElement;
    expect(within(tip).getByText('id')).toBeTruthy();
    expect(within(tip).getByText('42')).toBeTruthy();
    fireEvent.mouseLeave(chip);

    rerender(<TokenChip token="hi" index={0} />);
    chip = container.querySelector('span') as HTMLElement;
    fireEvent.mouseEnter(chip);
    tip = document.body.querySelector('[class*="tooltip"]') as HTMLElement;
    expect(within(tip).queryByText('id')).toBeNull();
  });

  it('renders the offset row only when offset is provided', () => {
    const { container } = render(<TokenChip token="hi" index={0} offset={[2, 5]} />);
    const chip = container.querySelector('span') as HTMLElement;
    fireEvent.mouseEnter(chip);
    const tip = document.body.querySelector('[class*="tooltip"]') as HTMLElement;
    expect(within(tip).getByText('offset')).toBeTruthy();
    expect(within(tip).getByText('[2, 5)')).toBeTruthy();
  });

  it('positions the tooltip from the chip bounding rect', () => {
    const { container } = render(<TokenChip token="hi" index={0} />);
    const chip = container.querySelector('span') as HTMLElement;
    // Provide a non-zero rect so the centering math is exercised.
    vi.spyOn(chip, 'getBoundingClientRect').mockReturnValue({
      left: 100,
      top: 50,
      width: 40,
      height: 20,
      right: 140,
      bottom: 70,
      x: 100,
      y: 50,
      toJSON: () => ({}),
    } as DOMRect);
    fireEvent.mouseEnter(chip);
    const tip = document.body.querySelector('[class*="tooltip"]') as HTMLElement;
    // x = left + width/2 = 120, y = top - 8 = 42
    expect(tip.style.left).toBe('120px');
    expect(tip.style.top).toBe('42px');
  });

  it('does not crash on enter when the ref element is unavailable (defensive)', () => {
    // ref.current is set by React, so this path mainly guards null. We render
    // and enter normally; the tooltip should appear without throwing.
    const { container } = render(<TokenChip token="z" index={1} />);
    const chip = container.querySelector('span') as HTMLElement;
    expect(() => fireEvent.mouseEnter(chip)).not.toThrow();
  });
});

describe('TokenChip — bytes preview', () => {
  it('renders hex bytes for short tokens', () => {
    const { container } = render(<TokenChip token="A" index={0} />);
    fireEvent.mouseEnter(container.querySelector('span') as HTMLElement);
    const tip = document.body.querySelector('[class*="tooltip"]') as HTMLElement;
    // 'A' → 0x41
    expect(within(tip).getByText('41')).toBeTruthy();
  });

  it('truncates and annotates the byte count for long tokens (> 24 bytes)', () => {
    const long = 'x'.repeat(30); // 30 ASCII bytes > max 24
    const { container } = render(<TokenChip token={long} index={0} />);
    fireEvent.mouseEnter(container.querySelector('span') as HTMLElement);
    const tip = document.body.querySelector('[class*="tooltip"]') as HTMLElement;
    expect(within(tip).getByText(/… \(30 bytes\)/)).toBeTruthy();
  });
});
