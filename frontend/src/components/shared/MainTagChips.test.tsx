import { describe, it, expect, beforeEach } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { useState } from 'react';
import { useI18n } from '../../i18n';
import { ALL_MAIN_TAGS, MainTagChips, scrollToShow, wheelToScroll } from './MainTagChips';

// Phase 5 (density): the chips are one row that scrolls sideways. A mouse
// wheel scrolls it, the selected chip is kept in view, and each edge fades
// only while there is more beyond it. jsdom has no layout, so the boxes are
// given their sizes by hand.

const entries = [
  { id: 'quickstart', order: 1, count: 4 },
  { id: 'training', order: 2, count: 12 },
  { id: 'concepts', order: 3, count: 15 },
];

function Harness({ initial = ALL_MAIN_TAGS }: { initial?: string }) {
  const [value, setValue] = useState(initial);
  return <MainTagChips entries={entries} value={value} onChange={setValue} idPrefix="t" panelId="p" />;
}

/** Give the row a 300px window onto 900px of chips, and each chip a box. */
function layOut(row: HTMLElement, scrollLeft = 0) {
  Object.defineProperty(row, 'scrollWidth', { configurable: true, value: 900 });
  Object.defineProperty(row, 'clientWidth', { configurable: true, value: 300 });
  row.scrollLeft = scrollLeft;
  Array.from(row.children).forEach((chip, i) => {
    Object.defineProperty(chip, 'offsetLeft', { configurable: true, value: i * 300 });
    Object.defineProperty(chip, 'offsetWidth', { configurable: true, value: 280 });
  });
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
});

describe('wheelToScroll', () => {
  const box = { scrollLeft: 100, scrollWidth: 900, clientWidth: 300 };

  it('turns a vertical wheel turn into a sideways scroll', () => {
    expect(wheelToScroll({ deltaX: 0, deltaY: 40, deltaMode: 0 }, box)).toBe(40);
    expect(wheelToScroll({ deltaX: 0, deltaY: -40, deltaMode: 0 }, box)).toBe(-40);
  });

  it('scales line and page deltas', () => {
    expect(wheelToScroll({ deltaX: 0, deltaY: 3, deltaMode: 1 }, box)).toBe(48);
    expect(wheelToScroll({ deltaX: 0, deltaY: 1, deltaMode: 2 }, box)).toBe(300);
  });

  it('leaves a sideways gesture and a strip at its end to the browser', () => {
    expect(wheelToScroll({ deltaX: 30, deltaY: 10, deltaMode: 0 }, box)).toBe(0);
    expect(wheelToScroll({ deltaX: 0, deltaY: -40, deltaMode: 0 }, { ...box, scrollLeft: 0 })).toBe(0);
    expect(wheelToScroll({ deltaX: 0, deltaY: 40, deltaMode: 0 }, { ...box, scrollLeft: 600 })).toBe(0);
  });
});

describe('scrollToShow', () => {
  const box = { scrollLeft: 300, clientWidth: 300 };

  it('scrolls back to a chip left of the window, clear of the fade', () => {
    expect(scrollToShow({ offsetLeft: 100, offsetWidth: 80 }, box)).toBe(76);
  });

  it('scrolls on to a chip right of the window, clear of the fade', () => {
    expect(scrollToShow({ offsetLeft: 700, offsetWidth: 80 }, box)).toBe(504);
  });

  it('leaves a chip already in view where it is', () => {
    expect(scrollToShow({ offsetLeft: 400, offsetWidth: 80 }, box)).toBeNull();
  });
});

describe('MainTagChips — one scrolling row', () => {
  it('a wheel turn over the row scrolls it and keeps the page still', () => {
    render(<Harness />);
    const row = screen.getByRole('tablist');
    layOut(row, 0);
    const event = new WheelEvent('wheel', { deltaY: 120, bubbles: true, cancelable: true });
    act(() => {
      row.dispatchEvent(event);
    });
    expect(row.scrollLeft).toBe(120);
    expect(event.defaultPrevented).toBe(true);
  });

  it('brings the chip chosen with the keyboard into view', () => {
    render(<Harness />);
    const row = screen.getByRole('tablist');
    layOut(row, 0);
    const all = screen.getByRole('tab', { name: /All/ });
    fireEvent.keyDown(all, { key: 'End' });
    // Fourth chip (index 3): offsetLeft 900, width 280 -> right edge 1204.
    expect(row.scrollLeft).toBe(1204 - 300);
  });

  it('fades only the edges that have more beyond them', () => {
    render(<Harness />);
    const row = screen.getByRole('tablist');
    layOut(row, 0);
    act(() => {
      row.dispatchEvent(new Event('scroll'));
    });
    expect(row.hasAttribute('data-fade-start')).toBe(false);
    expect(row.hasAttribute('data-fade-end')).toBe(true);
    row.scrollLeft = 300;
    act(() => {
      row.dispatchEvent(new Event('scroll'));
    });
    expect(row.hasAttribute('data-fade-start')).toBe(true);
    expect(row.hasAttribute('data-fade-end')).toBe(true);
    row.scrollLeft = 600;
    act(() => {
      row.dispatchEvent(new Event('scroll'));
    });
    expect(row.hasAttribute('data-fade-start')).toBe(true);
    expect(row.hasAttribute('data-fade-end')).toBe(false);
  });

  it('draws no fade when everything fits', () => {
    render(<Harness />);
    const row = screen.getByRole('tablist');
    Object.defineProperty(row, 'scrollWidth', { configurable: true, value: 300 });
    Object.defineProperty(row, 'clientWidth', { configurable: true, value: 300 });
    act(() => {
      row.dispatchEvent(new Event('scroll'));
    });
    expect(row.hasAttribute('data-fade-start')).toBe(false);
    expect(row.hasAttribute('data-fade-end')).toBe(false);
  });
});
