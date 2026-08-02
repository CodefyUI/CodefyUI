import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { createFrameFlusher } from './frameScheduler';

/** Advance past one animation frame with fake timers installed. */
function runFrame() {
  vi.advanceTimersByTime(20);
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('createFrameFlusher', () => {
  it('does not run the flush until a frame fires', () => {
    const flush = vi.fn();
    const flusher = createFrameFlusher(flush);
    flusher.schedule();
    expect(flush).not.toHaveBeenCalled();
    runFrame();
    expect(flush).toHaveBeenCalledTimes(1);
  });

  it('coalesces any number of schedules into one flush per frame', () => {
    const flush = vi.fn();
    const flusher = createFrameFlusher(flush);
    for (let i = 0; i < 50; i += 1) flusher.schedule();
    runFrame();
    expect(flush).toHaveBeenCalledTimes(1);
  });

  it('clears the slot before flushing, so a re-schedule from inside lands on the next frame', () => {
    const flush = vi.fn(() => {
      if (flush.mock.calls.length === 1) flusher.schedule();
    });
    const flusher = createFrameFlusher(flush);
    flusher.schedule();
    runFrame();
    expect(flush).toHaveBeenCalledTimes(1);
    runFrame();
    expect(flush).toHaveBeenCalledTimes(2);
  });

  it('cancel drops a pending flush', () => {
    const flush = vi.fn();
    const flusher = createFrameFlusher(flush);
    flusher.schedule();
    expect(flusher.isScheduled()).toBe(true);
    flusher.cancel();
    expect(flusher.isScheduled()).toBe(false);
    runFrame();
    expect(flush).not.toHaveBeenCalled();
  });

  it('cancel is a no-op when nothing is pending', () => {
    const flusher = createFrameFlusher(vi.fn());
    expect(() => flusher.cancel()).not.toThrow();
    expect(flusher.isScheduled()).toBe(false);
  });

  it('falls back to a macrotask where requestAnimationFrame does not exist', () => {
    vi.stubGlobal('requestAnimationFrame', undefined);
    vi.stubGlobal('cancelAnimationFrame', undefined);
    const flush = vi.fn();
    const flusher = createFrameFlusher(flush);
    flusher.schedule();
    flusher.schedule();
    expect(flush).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(flush).toHaveBeenCalledTimes(1);
  });

  it('cancels a macrotask fallback with clearTimeout, not cancelAnimationFrame', () => {
    vi.stubGlobal('requestAnimationFrame', undefined);
    vi.stubGlobal('cancelAnimationFrame', undefined);
    const flush = vi.fn();
    const flusher = createFrameFlusher(flush);
    flusher.schedule();
    flusher.cancel();
    vi.advanceTimersByTime(50);
    expect(flush).not.toHaveBeenCalled();
  });

  it('survives rAF being stripped between scheduling and cancelling', () => {
    const flush = vi.fn();
    const flusher = createFrameFlusher(flush);
    flusher.schedule();
    vi.stubGlobal('cancelAnimationFrame', undefined);
    expect(() => flusher.cancel()).not.toThrow();
    expect(flusher.isScheduled()).toBe(false);
  });

  it('keeps two flushers independent', () => {
    const a = vi.fn();
    const b = vi.fn();
    const fa = createFrameFlusher(a);
    const fb = createFrameFlusher(b);
    fa.schedule();
    expect(fb.isScheduled()).toBe(false);
    runFrame();
    expect(a).toHaveBeenCalledTimes(1);
    expect(b).not.toHaveBeenCalled();
  });
});
