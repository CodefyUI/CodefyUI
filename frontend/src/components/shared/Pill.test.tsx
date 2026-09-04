import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Pill } from './Pill';

/**
 * The chip both centers wear.
 *
 * What is worth pinning is the contract its callers and their tests read:
 * the tone lands on `data-tone`, the pulse dot is marked and hidden from
 * assistive tech, and the label is the pill's whole accessible text.
 */

describe('Pill', () => {
  it('renders the label and reports its tone as an attribute', () => {
    render(<Pill tone="success">Installed</Pill>);

    const pill = screen.getByText('Installed');
    expect(pill.getAttribute('data-tone')).toBe('success');
    expect(pill.className).toMatch(/pill/);
  });

  it('wears every tone the scale defines', () => {
    const { rerender } = render(<Pill tone="warning">Files missing</Pill>);
    expect(screen.getByText('Files missing').getAttribute('data-tone')).toBe('warning');

    rerender(<Pill tone="info">Installing</Pill>);
    expect(screen.getByText('Installing').getAttribute('data-tone')).toBe('info');

    rerender(<Pill tone="neutral">Not installed</Pill>);
    expect(screen.getByText('Not installed').getAttribute('data-tone')).toBe('neutral');
  });

  it('shows no dot by default', () => {
    const { container } = render(<Pill tone="neutral">Disabled</Pill>);

    expect(container.querySelector('[data-role="pulse"]')).toBeNull();
  });

  it('marks a pulsing pill as live without announcing the dot', () => {
    const { container } = render(<Pill tone="info" pulse>Installing</Pill>);

    const dot = container.querySelector('[data-role="pulse"]');
    expect(dot).not.toBeNull();
    // Decoration: the label beside it already says the state, and a screen
    // reader that stopped on an empty span would be reading punctuation.
    expect(dot?.getAttribute('aria-hidden')).toBe('true');
    // Still the pill's only text, so `getByText('Installing')` in a caller's
    // test keeps matching the pill rather than splitting across two nodes.
    expect(screen.getByText('Installing').textContent).toBe('Installing');
  });
});
