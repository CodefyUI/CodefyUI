import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ToastContainer } from './Toast';
import { useToastStore, type ToastType } from '../../store/toastStore';

beforeEach(() => {
  useToastStore.setState({ toasts: [] });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('ToastContainer', () => {
  it('renders nothing when there are no toasts', () => {
    const { container } = render(<ToastContainer />);
    expect(container.firstChild).toBeNull();
  });

  it('renders a toast with its message', () => {
    useToastStore.setState({ toasts: [{ id: '1', message: 'Saved!', type: 'success' }] });
    render(<ToastContainer />);
    expect(screen.getByText('Saved!')).toBeTruthy();
  });

  it('renders the correct icon for every toast type', () => {
    const expected: Record<ToastType, string> = {
      success: '✓',
      error: '✗',
      info: 'ⓘ',
      warning: '⚠',
    };
    useToastStore.setState({
      toasts: (Object.keys(expected) as ToastType[]).map((type, i) => ({
        id: String(i),
        message: type,
        type,
      })),
    });
    const { container } = render(<ToastContainer />);
    const icons = Array.from(container.querySelectorAll('[class*="icon"]')).map((e) => e.textContent);
    expect(icons).toEqual([expected.success, expected.error, expected.info, expected.warning]);
  });

  it('applies the per-type modifier class', () => {
    useToastStore.setState({ toasts: [{ id: '1', message: 'oops', type: 'error' }] });
    const { container } = render(<ToastContainer />);
    // The toast element carries both the base and the type-specific class.
    const toast = container.querySelector('[class*="toast"]') as HTMLElement;
    expect(toast.className).toMatch(/error/);
  });

  it('clicking the close button removes that toast from the store', () => {
    useToastStore.setState({
      toasts: [
        { id: 'a', message: 'one', type: 'info' },
        { id: 'b', message: 'two', type: 'info' },
      ],
    });
    render(<ToastContainer />);
    const closeButtons = screen.getAllByRole('button');
    fireEvent.click(closeButtons[0]);
    expect(useToastStore.getState().toasts.map((t) => t.id)).toEqual(['b']);
  });

  it('renders multiple toasts in order', () => {
    useToastStore.setState({
      toasts: [
        { id: '1', message: 'first', type: 'info' },
        { id: '2', message: 'second', type: 'warning' },
      ],
    });
    const { container } = render(<ToastContainer />);
    const messages = Array.from(container.querySelectorAll('[class*="message"]')).map((e) => e.textContent);
    expect(messages).toEqual(['first', 'second']);
  });
});
