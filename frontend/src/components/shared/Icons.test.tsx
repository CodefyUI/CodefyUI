import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import * as allIcons from './Icons';
import {
  BookIcon,
  ChevronDownIcon,
  CloseIcon,
  CollapseAllIcon,
  CopyIcon,
  DiscardIcon,
  ExpandAllIcon,
  ExpandIcon,
  EyeIcon,
  EyeOffIcon,
  FitIcon,
  GitBranchIcon,
  LayersIcon,
  LibraryIcon,
  MinusIcon,
  MoreHorizontalIcon,
  PackageIcon,
  PanelLeftCloseIcon,
  PanelLeftOpenIcon,
  PlusIcon,
  RefreshIcon,
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
  ['LibraryIcon', LibraryIcon],
  ['LayersIcon', LayersIcon],
  ['BookIcon', BookIcon],
  ['PackageIcon', PackageIcon],
  ['PanelLeftCloseIcon', PanelLeftCloseIcon],
  ['PanelLeftOpenIcon', PanelLeftOpenIcon],
  ['ExpandAllIcon', ExpandAllIcon],
  ['CollapseAllIcon', CollapseAllIcon],
  ['RefreshIcon', RefreshIcon],
  ['CopyIcon', CopyIcon],
  ['GitBranchIcon', GitBranchIcon],
  ['MoreHorizontalIcon', MoreHorizontalIcon],
  ['PlusIcon', PlusIcon],
  ['MinusIcon', MinusIcon],
  ['DiscardIcon', DiscardIcon],
  ['ChevronDownIcon', ChevronDownIcon],
] as const;

describe('Icons', () => {
  // The list above is written by hand, so an icon added to Icons.tsx and
  // forgotten here was simply never checked. Comparing it against the module's
  // own exports is what makes "add it to both lists" enforceable rather than a
  // note somebody has to remember.
  it('covers every exported icon', () => {
    const registered = new Set<string>(icons.map(([name]) => name));
    const exported = Object.keys(allIcons);
    expect(exported.filter((name) => !registered.has(name))).toEqual([]);
    expect([...registered].filter((name) => !exported.includes(name))).toEqual([]);
  });

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
