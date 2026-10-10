import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { GuideTour } from './GuideTour';
import { availableSteps, guideTarget, GUIDE_STEPS } from './guideSteps';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';

const rect = (r: Partial<DOMRect>): DOMRect =>
  ({ x: 0, y: 0, top: 0, left: 0, right: 0, bottom: 0, width: 0, height: 0, toJSON: () => ({}), ...r }) as DOMRect;

/** Boxes by `data-tour` name; the tour card measures 300 x 160. */
let boxes: Record<string, Partial<DOMRect>> = {};

function mountTargets(names: string[]) {
  for (const name of names) {
    const el = document.createElement('div');
    el.setAttribute('data-tour', name);
    document.body.appendChild(el);
  }
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useUIStore.setState({ guideOpen: false });
  Object.defineProperty(window, 'innerWidth', { configurable: true, value: 1200 });
  Object.defineProperty(window, 'innerHeight', { configurable: true, value: 800 });
  boxes = {};
  HTMLElement.prototype.getBoundingClientRect = function () {
    if (this.getAttribute('role') === 'dialog') return rect({ width: 300, height: 160 });
    const name = this.getAttribute('data-tour');
    return rect(name ? boxes[name] ?? { width: 10, height: 10 } : {});
  };
});

afterEach(() => {
  cleanup();
  document.querySelectorAll('[data-tour]').forEach((el) => el.remove());
});

const open = () => act(() => useUIStore.getState().openGuide());
const card = () => screen.getByRole('dialog');
const title = () => card().querySelector('h3')!.textContent;

describe('guideSteps', () => {
  it('keeps the stops whose element is on screen, and the centred ones', () => {
    mountTargets(['run', 'help']);
    boxes.help = { width: 0, height: 0 }; // present but not laid out
    expect(availableSteps().map((s) => s.id)).toEqual(['welcome', 'run']);
    expect(guideTarget(GUIDE_STEPS[0])).toBeNull();
  });
});

describe('GuideTour', () => {
  it('renders nothing until opened', () => {
    render(<GuideTour />);
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('walks the stops on screen with Next and Previous, counting them', () => {
    mountTargets(['palette', 'run']);
    render(<GuideTour />);
    open();
    expect(title()).toBe('Quick tour');
    expect(card()).toHaveTextContent('1 / 3');
    expect(screen.getByRole('button', { name: 'Previous' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Next' })).toHaveFocus();
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    expect(title()).toBe('Nodes');
    expect(card()).toHaveTextContent('2 / 3');
    fireEvent.click(screen.getByRole('button', { name: 'Previous' }));
    expect(title()).toBe('Quick tour');
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    expect(title()).toBe('Run');
    fireEvent.click(screen.getByRole('button', { name: 'Done' }));
    expect(useUIStore.getState().guideOpen).toBe(false);
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('moves with the arrow keys and closes on Escape', () => {
    mountTargets(['run']);
    render(<GuideTour />);
    open();
    fireEvent.keyDown(window, { key: 'ArrowRight' });
    expect(title()).toBe('Run');
    fireEvent.keyDown(window, { key: 'ArrowLeft' });
    expect(title()).toBe('Quick tour');
    fireEvent.keyDown(window, { key: 'ArrowLeft' });
    expect(title()).toBe('Quick tour');
    fireEvent.keyDown(window, { key: 'Tab' });
    expect(title()).toBe('Quick tour');
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(useUIStore.getState().guideOpen).toBe(false);
  });

  it('closes from the skip button', () => {
    render(<GuideTour />);
    open();
    fireEvent.click(screen.getByRole('button', { name: 'Skip the tour' }));
    expect(useUIStore.getState().guideOpen).toBe(false);
  });

  it('centres the first card over a dim screen', () => {
    render(<GuideTour />);
    open();
    expect(card().style.top).toBe('320px');
    expect(card().style.left).toBe('450px');
  });

  // Each target is 100 x 40 before the 6px frame padding.
  it.each([
    ['below', { top: 100, left: 500, width: 100, height: 40 }, { top: '158px', left: '400px' }],
    ['above', { top: 700, left: 500, width: 100, height: 40 }, { top: '522px', left: '400px' }],
    ['to the right', { top: 20, left: 20, width: 100, height: 760 }, { top: '320px', left: '138px' }],
    ['to the left', { top: 20, left: 1000, width: 190, height: 760 }, { top: '320px', left: '682px' }],
  ])('frames the element and puts the card %s it', (_where, box, expected) => {
    mountTargets(['run']);
    boxes.run = box;
    render(<GuideTour />);
    open();
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    const frame = document.body.querySelector('[class*="frame"]') as HTMLElement;
    expect(frame.style.top).toBe(`${box.top - 6}px`);
    expect(frame.style.width).toBe(`${box.width + 12}px`);
    expect(card().style.top).toBe(expected.top);
    expect(card().style.left).toBe(expected.left);
  });

  it('follows the element when the window resizes', () => {
    mountTargets(['run']);
    boxes.run = { top: 100, left: 500, width: 100, height: 40 };
    render(<GuideTour />);
    open();
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    boxes.run = { top: 200, left: 500, width: 100, height: 40 };
    act(() => {
      window.dispatchEvent(new Event('resize'));
    });
    const frame = document.body.querySelector('[class*="frame"]') as HTMLElement;
    expect(frame.style.top).toBe('194px');
  });
});
