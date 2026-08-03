import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { useI18n } from '../../i18n';
import {
  _clearPluginToolbarButtons, registerPluginToolbarButton,
} from '../../plugins/toolbarButtons';
import { PluginToolbarButtons, inlineToolbarCapacity } from './PluginToolbarButtons';

function setWidth(px: number) {
  Object.defineProperty(window, 'innerWidth', {
    value: px, configurable: true, writable: true,
  });
}

/** Register `n` buttons named b0..b(n-1). Returns their click spies. */
function seedButtons(n: number) {
  const clicks = Array.from({ length: n }, () => vi.fn());
  for (let i = 0; i < n; i += 1) {
    registerPluginToolbarButton('p', {
      id: `b${i}`, icon: String(i), tooltip: `Button ${i}`, onClick: clicks[i],
    });
  }
  return clicks;
}

const t = (k: 'plugins.moreActions') => useI18n.getState().t(k);

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  _clearPluginToolbarButtons();
  setWidth(1600);
});

afterEach(() => {
  _clearPluginToolbarButtons();
  vi.restoreAllMocks();
});

describe('inlineToolbarCapacity', () => {
  it('maps width to a capacity, narrowest first', () => {
    expect(inlineToolbarCapacity(320)).toBe(0);
    expect(inlineToolbarCapacity(899)).toBe(0);
    expect(inlineToolbarCapacity(900)).toBe(1);
    expect(inlineToolbarCapacity(1199)).toBe(1);
    expect(inlineToolbarCapacity(1200)).toBe(2);
    expect(inlineToolbarCapacity(1499)).toBe(2);
    expect(inlineToolbarCapacity(1500)).toBe(3);
    expect(inlineToolbarCapacity(4000)).toBe(3);
  });

  it('never goes backwards as the window grows', () => {
    let previous = 0;
    for (let w = 200; w <= 2400; w += 25) {
      const capacity = inlineToolbarCapacity(w);
      expect(capacity).toBeGreaterThanOrEqual(previous);
      previous = capacity;
    }
  });
});

describe('PluginToolbarButtons', () => {
  it('renders nothing when no plugin registered a button', () => {
    const { container } = render(<PluginToolbarButtons />);
    expect(container).toBeEmptyDOMElement();
  });

  it('shows every button inline when they all fit', () => {
    seedButtons(3);
    render(<PluginToolbarButtons />);
    expect(screen.getByTestId('plugin-toolbar-button-p:b0')).toBeInTheDocument();
    expect(screen.getByTestId('plugin-toolbar-button-p:b2')).toBeInTheDocument();
    expect(screen.queryByTestId('plugin-toolbar-overflow')).not.toBeInTheDocument();
  });

  it('labels each button with its tooltip, so an icon is never the only label', () => {
    seedButtons(1);
    render(<PluginToolbarButtons />);
    const button = screen.getByTestId('plugin-toolbar-button-p:b0');
    expect(button).toHaveAttribute('title', 'Button 0');
    expect(button).toHaveAttribute('aria-label', 'Button 0');
  });

  it('moves the overflow into a menu, keeping room for the menu button itself', () => {
    seedButtons(5);
    render(<PluginToolbarButtons />);
    // Capacity 3, six slots wanted: two inline plus the "..." button.
    expect(screen.getByTestId('plugin-toolbar-button-p:b0')).toBeInTheDocument();
    expect(screen.getByTestId('plugin-toolbar-button-p:b1')).toBeInTheDocument();
    expect(screen.queryByTestId('plugin-toolbar-button-p:b2')).not.toBeInTheDocument();
    expect(screen.getByTestId('plugin-toolbar-overflow')).toBeInTheDocument();
  });

  it('puts every button in the menu at the narrowest widths', () => {
    setWidth(800);
    seedButtons(2);
    render(<PluginToolbarButtons />);
    expect(screen.queryByTestId('plugin-toolbar-button-p:b0')).not.toBeInTheDocument();
    expect(screen.getByTestId('plugin-toolbar-overflow')).toBeInTheDocument();
    fireEvent.click(screen.getByTestId('plugin-toolbar-overflow'));
    expect(screen.getByTestId('plugin-toolbar-button-p:b0')).toBeInTheDocument();
    expect(screen.getByTestId('plugin-toolbar-button-p:b1')).toBeInTheDocument();
  });

  it('a single button is still shown inline on a narrow-ish window', () => {
    setWidth(1000);
    seedButtons(1);
    render(<PluginToolbarButtons />);
    expect(screen.getByTestId('plugin-toolbar-button-p:b0')).toBeInTheDocument();
    expect(screen.queryByTestId('plugin-toolbar-overflow')).not.toBeInTheDocument();
  });

  it('re-flows when the window is resized', () => {
    seedButtons(3);
    render(<PluginToolbarButtons />);
    expect(screen.queryByTestId('plugin-toolbar-overflow')).not.toBeInTheDocument();
    act(() => {
      setWidth(1000);
      window.dispatchEvent(new Event('resize'));
    });
    expect(screen.getByTestId('plugin-toolbar-overflow')).toBeInTheDocument();
  });

  it('picks up a button registered after mount', () => {
    render(<PluginToolbarButtons />);
    act(() => { seedButtons(1); });
    expect(screen.getByTestId('plugin-toolbar-button-p:b0')).toBeInTheDocument();
  });

  it('runs the handler on click, inline and from the menu', () => {
    const clicks = seedButtons(5);
    render(<PluginToolbarButtons />);
    fireEvent.click(screen.getByTestId('plugin-toolbar-button-p:b0'));
    expect(clicks[0]).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByTestId('plugin-toolbar-overflow'));
    fireEvent.click(screen.getByTestId('plugin-toolbar-button-p:b4'));
    expect(clicks[4]).toHaveBeenCalledTimes(1);
    // Choosing an item closes the menu.
    expect(screen.queryByTestId('plugin-toolbar-button-p:b3')).not.toBeInTheDocument();
  });

  it('a throwing handler does not take the toolbar down', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    registerPluginToolbarButton('p', {
      id: 'bad', icon: '!', tooltip: 'Bad',
      onClick: () => { throw new Error('plugin exploded'); },
    });
    render(<PluginToolbarButtons />);
    expect(() => fireEvent.click(screen.getByTestId('plugin-toolbar-button-p:bad')))
      .not.toThrow();
    expect(warn).toHaveBeenCalled();
    expect(screen.getByTestId('plugin-toolbar-button-p:bad')).toBeInTheDocument();
  });

  it('closes the menu on outside click and on Escape', () => {
    seedButtons(5);
    render(<PluginToolbarButtons />);
    const trigger = screen.getByTestId('plugin-toolbar-overflow');

    fireEvent.click(trigger);
    expect(trigger).toHaveAttribute('aria-expanded', 'true');
    fireEvent.mouseDown(document.body);
    expect(trigger).toHaveAttribute('aria-expanded', 'false');

    fireEvent.click(trigger);
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(trigger).toHaveAttribute('aria-expanded', 'false');
  });

  it('labels the overflow trigger from the catalog, in the active locale', () => {
    seedButtons(5);
    render(<PluginToolbarButtons />);
    expect(screen.getByTestId('plugin-toolbar-overflow'))
      .toHaveAttribute('aria-label', t('plugins.moreActions'));

    act(() => { useI18n.setState({ locale: 'zh-TW' }); });
    expect(screen.getByTestId('plugin-toolbar-overflow'))
      .toHaveAttribute('aria-label', t('plugins.moreActions'));
    expect(t('plugins.moreActions')).not.toBe('plugins.moreActions');
  });
});
