import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { PaneContextMenu } from './PaneContextMenu';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';

const SCREEN = { x: 120, y: 240 };
const FLOW = { x: 17, y: 42 };

describe('PaneContextMenu', () => {
  beforeEach(() => {
    useI18n.setState({ locale: 'en' });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders both add-note menu items at the screen position', () => {
    render(<PaneContextMenu screen={SCREEN} flow={FLOW} onClose={() => {}} />);
    const addText = screen.getByText('Add Text Note');
    const addImage = screen.getByText('Add Image Note');
    expect(addText).toBeInTheDocument();
    expect(addImage).toBeInTheDocument();

    // The menu container is positioned via the screen coords.
    const menu = addText.closest('div[style*="left"]') as HTMLElement;
    expect(menu.style.left).toBe('120px');
    expect(menu.style.top).toBe('240px');
  });

  it('"Add Text Note" calls addNote("text", flow) then onClose', () => {
    const addNote = vi.fn();
    useTabStore.setState({ addNote });
    const onClose = vi.fn();

    render(<PaneContextMenu screen={SCREEN} flow={FLOW} onClose={onClose} />);
    fireEvent.click(screen.getByText('Add Text Note'));

    expect(addNote).toHaveBeenCalledWith('text', FLOW);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('"Add Image Note" calls addNote("image", flow) then onClose', () => {
    const addNote = vi.fn();
    useTabStore.setState({ addNote });
    const onClose = vi.fn();

    render(<PaneContextMenu screen={SCREEN} flow={FLOW} onClose={onClose} />);
    fireEvent.click(screen.getByText('Add Image Note'));

    expect(addNote).toHaveBeenCalledWith('image', FLOW);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('clicking the backdrop closes the menu', () => {
    const onClose = vi.fn();
    const { container } = render(
      <PaneContextMenu screen={SCREEN} flow={FLOW} onClose={onClose} />,
    );
    // First child is the backdrop div (from NodeContextMenu).
    const backdrop = container.firstElementChild as HTMLElement;
    fireEvent.click(backdrop);
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
