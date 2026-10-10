import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { HelpTip } from './HelpTip';
import { useI18n } from '../../i18n';

const rect = (r: Partial<DOMRect>): DOMRect =>
  ({ x: 0, y: 0, top: 0, left: 0, right: 0, bottom: 0, width: 0, height: 0, toJSON: () => ({}), ...r }) as DOMRect;

function placeIcon(icon: HTMLElement, r: Partial<DOMRect>) {
  icon.getBoundingClientRect = () => rect(r);
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  Object.defineProperty(window, 'innerWidth', { configurable: true, value: 800 });
  Object.defineProperty(window, 'innerHeight', { configurable: true, value: 600 });
  // Every card in these tests measures 100 x 40.
  HTMLElement.prototype.getBoundingClientRect = function () {
    return this.getAttribute('role') === 'tooltip' ? rect({ width: 100, height: 40 }) : rect({});
  };
});

afterEach(cleanup);

describe('HelpTip', () => {
  it('names itself after its topic and shows nothing until asked', () => {
    render(<HelpTip text="Does a thing." topic="Seed" />);
    const icon = screen.getByRole('button', { name: 'About Seed' });
    expect(icon).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByRole('tooltip')).toBeNull();
  });

  it('shows on hover and hides when the pointer leaves', () => {
    render(<HelpTip text="Does a thing." topic="Seed" />);
    const icon = screen.getByRole('button', { name: 'About Seed' });
    fireEvent.mouseEnter(icon);
    const tip = screen.getByRole('tooltip');
    expect(tip).toHaveTextContent('Does a thing.');
    expect(icon).toHaveAttribute('aria-describedby', tip.id);
    fireEvent.mouseLeave(icon);
    expect(screen.queryByRole('tooltip')).toBeNull();
    expect(icon).not.toHaveAttribute('aria-describedby');
  });

  it('shows on keyboard focus, and Escape hides it without reaching the dialog around it', () => {
    let outer = 0;
    render(
      <div onKeyDown={() => (outer += 1)}>
        <HelpTip text="Does a thing." topic="Seed" />
      </div>,
    );
    const icon = screen.getByRole('button', { name: 'About Seed' });
    fireEvent.focus(icon);
    expect(screen.getByRole('tooltip')).toBeInTheDocument();
    fireEvent.keyDown(icon, { key: 'Escape' });
    expect(screen.queryByRole('tooltip')).toBeNull();
    expect(outer).toBe(0);
    fireEvent.focus(icon);
    fireEvent.keyDown(icon, { key: 'a' });
    expect(screen.getByRole('tooltip')).toBeInTheDocument();
    fireEvent.blur(icon);
    expect(screen.queryByRole('tooltip')).toBeNull();
  });

  it('toggles on click without clicking the row it sits in', () => {
    let rowClicks = 0;
    render(
      <div onClick={() => (rowClicks += 1)}>
        <HelpTip text="Does a thing." topic="Seed" />
      </div>,
    );
    const icon = screen.getByRole('button', { name: 'About Seed' });
    fireEvent.click(icon);
    expect(screen.getByRole('tooltip')).toBeInTheDocument();
    fireEvent.click(icon);
    expect(screen.queryByRole('tooltip')).toBeNull();
    expect(rowClicks).toBe(0);
  });

  it('puts each line of the text in its own paragraph, or renders rich content', () => {
    const { unmount } = render(<HelpTip text={'First line.\nSecond line.'} topic="A" />);
    fireEvent.mouseEnter(screen.getByRole('button', { name: 'About A' }));
    expect(screen.getByRole('tooltip').querySelectorAll('p')).toHaveLength(2);
    unmount();
    render(
      <HelpTip topic="B">
        <strong>rich</strong>
      </HelpTip>,
    );
    fireEvent.mouseEnter(screen.getByRole('button', { name: 'About B' }));
    expect(screen.getByRole('tooltip').querySelector('strong')).toHaveTextContent('rich');
  });

  it('renders an empty card when given neither text nor content', () => {
    render(<HelpTip topic="C" />);
    fireEvent.mouseEnter(screen.getByRole('button', { name: 'About C' }));
    expect(screen.getByRole('tooltip')).toHaveTextContent('');
  });

  it('goes below the icon, centred and kept inside the window', () => {
    render(<HelpTip text="x" topic="Seed" />);
    const icon = screen.getByRole('button', { name: 'About Seed' });
    placeIcon(icon, { top: 100, bottom: 116, left: 10, width: 16, height: 16 });
    fireEvent.mouseEnter(icon);
    const tip = screen.getByRole('tooltip');
    expect(tip.style.top).toBe('122px');
    // Centred would be -32; it stops at the 8px edge.
    expect(tip.style.left).toBe('8px');
  });

  it('goes above the icon when there is no room below', () => {
    render(<HelpTip text="x" topic="Seed" />);
    const icon = screen.getByRole('button', { name: 'About Seed' });
    placeIcon(icon, { top: 570, bottom: 586, left: 780, width: 16, height: 16 });
    fireEvent.mouseEnter(icon);
    const tip = screen.getByRole('tooltip');
    expect(tip.style.top).toBe('524px');
    expect(tip.style.left).toBe('692px');
  });

  it('closes when the page scrolls or the window resizes', () => {
    render(<HelpTip text="x" topic="Seed" />);
    const icon = screen.getByRole('button', { name: 'About Seed' });
    fireEvent.mouseEnter(icon);
    act(() => {
      window.dispatchEvent(new Event('scroll'));
    });
    expect(screen.queryByRole('tooltip')).toBeNull();
    fireEvent.mouseEnter(icon);
    act(() => {
      window.dispatchEvent(new Event('resize'));
    });
    expect(screen.queryByRole('tooltip')).toBeNull();
  });
});
