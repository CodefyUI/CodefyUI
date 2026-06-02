import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { useToastStore } from './toastStore';

describe('useToastStore', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    useToastStore.setState({ toasts: [] });
  });

  afterEach(() => {
    vi.runOnlyPendingTimers();
    vi.useRealTimers();
  });

  describe('addToast', () => {
    it('defaults the type to "info" and assigns a string id', () => {
      useToastStore.getState().addToast('hello');
      const toasts = useToastStore.getState().toasts;
      expect(toasts).toHaveLength(1);
      expect(toasts[0].message).toBe('hello');
      expect(toasts[0].type).toBe('info');
      expect(typeof toasts[0].id).toBe('string');
    });

    it('auto-dismisses a non-error toast after 4000ms', () => {
      useToastStore.getState().addToast('temp', 'success');
      expect(useToastStore.getState().toasts).toHaveLength(1);
      vi.advanceTimersByTime(3999);
      expect(useToastStore.getState().toasts).toHaveLength(1);
      vi.advanceTimersByTime(1);
      expect(useToastStore.getState().toasts).toHaveLength(0);
    });

    it('auto-dismisses an info toast', () => {
      useToastStore.getState().addToast('info msg', 'info');
      vi.advanceTimersByTime(4000);
      expect(useToastStore.getState().toasts).toHaveLength(0);
    });

    it('auto-dismisses a warning toast', () => {
      useToastStore.getState().addToast('warn msg', 'warning');
      vi.advanceTimersByTime(4000);
      expect(useToastStore.getState().toasts).toHaveLength(0);
    });

    it('does NOT auto-dismiss an error toast', () => {
      useToastStore.getState().addToast('boom', 'error');
      vi.advanceTimersByTime(10000);
      expect(useToastStore.getState().toasts).toHaveLength(1);
      expect(useToastStore.getState().toasts[0].type).toBe('error');
    });

    it('appends multiple toasts with monotonically increasing ids', () => {
      useToastStore.getState().addToast('a', 'error');
      useToastStore.getState().addToast('b', 'error');
      const toasts = useToastStore.getState().toasts;
      expect(toasts).toHaveLength(2);
      expect(toasts[0].message).toBe('a');
      expect(toasts[1].message).toBe('b');
      expect(Number(toasts[1].id)).toBeGreaterThan(Number(toasts[0].id));
    });

    it('the auto-dismiss timer only removes its own toast', () => {
      // Add a non-error (timed) toast, then an error (untimed) toast.
      useToastStore.getState().addToast('vanishes', 'success');
      const survivorId = useToastStore.getState().toasts[0].id;
      useToastStore.getState().addToast('stays', 'error');
      const errorId = useToastStore.getState().toasts[1].id;
      vi.advanceTimersByTime(4000);
      const remaining = useToastStore.getState().toasts;
      expect(remaining).toHaveLength(1);
      expect(remaining[0].id).toBe(errorId);
      expect(remaining.find((t) => t.id === survivorId)).toBeUndefined();
    });
  });

  describe('removeToast', () => {
    it('removes the toast matching the id', () => {
      useToastStore.getState().addToast('keep', 'error');
      useToastStore.getState().addToast('drop', 'error');
      const dropId = useToastStore.getState().toasts[1].id;
      useToastStore.getState().removeToast(dropId);
      const toasts = useToastStore.getState().toasts;
      expect(toasts).toHaveLength(1);
      expect(toasts[0].message).toBe('keep');
    });

    it('is a no-op for an unknown id', () => {
      useToastStore.getState().addToast('one', 'error');
      useToastStore.getState().removeToast('does-not-exist');
      expect(useToastStore.getState().toasts).toHaveLength(1);
    });
  });
});
