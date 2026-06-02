import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { createRef } from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { FontSizeMenu } from './FontSizeMenu';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';

function makeTriggerRef() {
  // A real button element so `triggerRef.current?.contains(...)` works.
  const ref = createRef<HTMLButtonElement>();
  const btn = document.createElement('button');
  document.body.appendChild(btn);
  (ref as { current: HTMLButtonElement | null }).current = btn;
  return ref;
}

describe('FontSizeMenu', () => {
  beforeEach(() => {
    useI18n.setState({ locale: 'en' });
    useUIStore.setState({ fontSize: 'default' });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    document.body.innerHTML = '';
  });

  it('renders nothing when closed', () => {
    const { container } = render(
      <FontSizeMenu open={false} onClose={vi.fn()} triggerRef={makeTriggerRef()} />,
    );
    expect(container.querySelector('[role="menu"]')).toBeNull();
  });

  it('renders all three size options and marks the active one', () => {
    useUIStore.setState({ fontSize: 'large' });
    render(<FontSizeMenu open onClose={vi.fn()} triggerRef={makeTriggerRef()} />);

    expect(screen.getByText('Small')).toBeInTheDocument();
    expect(screen.getByText('Default')).toBeInTheDocument();
    expect(screen.getByText('Large')).toBeInTheDocument();

    const radios = screen.getAllByRole('menuitemradio');
    expect(radios).toHaveLength(3);
    // 'large' is the third option
    expect(radios[2]).toHaveAttribute('aria-checked', 'true');
    expect(radios[0]).toHaveAttribute('aria-checked', 'false');
  });

  it('selecting a size calls setFontSize and onClose', () => {
    const onClose = vi.fn();
    render(<FontSizeMenu open onClose={onClose} triggerRef={makeTriggerRef()} />);

    fireEvent.click(screen.getByText('Small'));
    expect(useUIStore.getState().fontSize).toBe('small');
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('closes on outside mousedown', () => {
    const onClose = vi.fn();
    render(<FontSizeMenu open onClose={onClose} triggerRef={makeTriggerRef()} />);

    fireEvent.mouseDown(document.body);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('does NOT close when mousedown is inside the panel', () => {
    const onClose = vi.fn();
    render(<FontSizeMenu open onClose={onClose} triggerRef={makeTriggerRef()} />);

    fireEvent.mouseDown(screen.getByRole('menu'));
    expect(onClose).not.toHaveBeenCalled();
  });

  it('does NOT close when mousedown is on the trigger element', () => {
    const onClose = vi.fn();
    const triggerRef = makeTriggerRef();
    render(<FontSizeMenu open onClose={onClose} triggerRef={triggerRef} />);

    fireEvent.mouseDown(triggerRef.current!);
    expect(onClose).not.toHaveBeenCalled();
  });

  it('closes on Escape key', () => {
    const onClose = vi.fn();
    render(<FontSizeMenu open onClose={onClose} triggerRef={makeTriggerRef()} />);

    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('ignores non-Escape keydowns', () => {
    const onClose = vi.fn();
    render(<FontSizeMenu open onClose={onClose} triggerRef={makeTriggerRef()} />);

    fireEvent.keyDown(document, { key: 'Enter' });
    expect(onClose).not.toHaveBeenCalled();
  });

  it('removes document listeners on unmount (no close after unmount)', () => {
    const onClose = vi.fn();
    const { unmount } = render(
      <FontSizeMenu open onClose={onClose} triggerRef={makeTriggerRef()} />,
    );
    unmount();
    fireEvent.mouseDown(document.body);
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onClose).not.toHaveBeenCalled();
  });
});
