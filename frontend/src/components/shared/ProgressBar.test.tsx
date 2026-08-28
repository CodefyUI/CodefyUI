import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ProgressBar } from './ProgressBar';

/** The fill sits inside the element carrying role="progressbar". */
const fillOf = () => screen.getByRole('progressbar').firstElementChild as HTMLElement;

describe('ProgressBar', () => {
  it('exposes progressbar semantics with a clamped aria-valuenow', () => {
    const { rerender } = render(<ProgressBar value={61} label="Downloading" />);

    const bar = screen.getByRole('progressbar');
    expect(bar.getAttribute('aria-valuemin')).toBe('0');
    expect(bar.getAttribute('aria-valuemax')).toBe('100');
    expect(bar.getAttribute('aria-valuenow')).toBe('61');
    expect(fillOf().style.width).toBe('61%');

    // Over and under the range: the widget never reports outside 0..100, and
    // the fill never overflows its track.
    rerender(<ProgressBar value={150} label="Downloading" />);
    expect(screen.getByRole('progressbar').getAttribute('aria-valuenow')).toBe('100');
    expect(fillOf().style.width).toBe('100%');

    rerender(<ProgressBar value={-5} label="Downloading" />);
    expect(screen.getByRole('progressbar').getAttribute('aria-valuenow')).toBe('0');
    expect(fillOf().style.width).toBe('0%');
  });

  it('omits aria-valuenow and marks the root busy for null', () => {
    const { container, rerender } = render(<ProgressBar value={null} label="Restarting" />);

    const bar = screen.getByRole('progressbar');
    // A progressbar with no aria-valuenow is how ARIA spells "indeterminate";
    // sending 0 instead would announce "0 percent", which is a lie.
    expect(bar.hasAttribute('aria-valuenow')).toBe(false);
    expect(bar.getAttribute('aria-valuemin')).toBe('0');
    expect(bar.getAttribute('aria-valuemax')).toBe('100');
    expect((container.firstElementChild as HTMLElement).getAttribute('aria-busy')).toBe('true');

    const fill = fillOf();
    expect(fill.className).toContain('indeterminate');
    // No inline width: the keyframe owns the fill's position.
    expect(fill.style.width).toBe('');

    // A determinate bar is not busy.
    rerender(<ProgressBar value={40} label="Restarting" />);
    expect((container.firstElementChild as HTMLElement).hasAttribute('aria-busy')).toBe(false);
    expect(fillOf().className).not.toContain('indeterminate');
  });

  it('renders the percentage text when showValue is set', () => {
    const { container, rerender } = render(
      <ProgressBar value={61.4} label="Downloading" showValue />
    );
    // Visual only: the progressbar's aria-valuenow already carries the number.
    expect(screen.getByText('61%').getAttribute('aria-hidden')).toBe('true');

    rerender(<ProgressBar value={61.6} label="Downloading" showValue />);
    expect(screen.getByText('62%')).toBeTruthy();

    // Off by default, and there is no percentage to show for an indeterminate bar.
    rerender(<ProgressBar value={61.6} label="Downloading" />);
    expect(container.textContent).toBe('');
    rerender(<ProgressBar value={null} label="Downloading" showValue />);
    expect(container.textContent).toBe('');
  });

  it('applies the tone and size classes', () => {
    const { container, rerender } = render(
      <ProgressBar value={50} label="Downloading" tone="warning" size="sm" className="extra" />
    );

    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain('sm');
    expect(root.className).toContain('extra');
    expect(fillOf().className).toContain('tone_warning');

    // Defaults: medium track, accent fill.
    rerender(<ProgressBar value={50} label="Downloading" />);
    const defaults = container.firstElementChild as HTMLElement;
    expect(defaults.className).toContain('md');
    expect(defaults.className).not.toContain('extra');
    expect(fillOf().className).toContain('tone_accent');
  });

  it('uses the accessible label', () => {
    render(<ProgressBar value={10} label="Downloading all-MiniLM-L6-v2" />);
    expect(screen.getByRole('progressbar', { name: 'Downloading all-MiniLM-L6-v2' })).toBeTruthy();
  });

  it('treats a NaN value as indeterminate', () => {
    // total_bytes of 0 makes downloaded/total NaN upstream; a NaN width would
    // silently render an empty bar that still claims to know its progress.
    const { container } = render(<ProgressBar value={Number.NaN} label="Downloading" showValue />);

    expect(screen.getByRole('progressbar').hasAttribute('aria-valuenow')).toBe(false);
    expect((container.firstElementChild as HTMLElement).getAttribute('aria-busy')).toBe('true');
    expect(container.textContent).toBe('');
  });
});
