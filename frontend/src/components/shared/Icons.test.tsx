import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import {
  CloseIcon,
  ExpandIcon,
  EyeIcon,
  EyeOffIcon,
  FitIcon,
  ZoomInIcon,
  ZoomOutIcon,
} from './Icons';

const icons = [
  ['EyeIcon', EyeIcon],
  ['EyeOffIcon', EyeOffIcon],
  ['ZoomInIcon', ZoomInIcon],
  ['ZoomOutIcon', ZoomOutIcon],
  ['FitIcon', FitIcon],
  ['ExpandIcon', ExpandIcon],
  ['CloseIcon', CloseIcon],
] as const;

describe('Icons', () => {
  it.each(icons)('%s renders a decorative svg using currentColor', (_name, IconComp) => {
    const { container } = render(<IconComp />);
    const svg = container.querySelector('svg');
    expect(svg).toBeTruthy();
    // Decorative: hidden from a11y tree, with the button supplying the label.
    expect(svg?.getAttribute('aria-hidden')).toBe('true');
    expect(svg?.getAttribute('stroke')).toBe('currentColor');
    // Default size is 14 when no size prop is provided.
    expect(svg?.getAttribute('width')).toBe('14');
  });

  it('honours an explicit size prop', () => {
    const { container } = render(<CloseIcon size={20} />);
    const svg = container.querySelector('svg');
    expect(svg?.getAttribute('width')).toBe('20');
    expect(svg?.getAttribute('height')).toBe('20');
  });
});
