import { describe, it, expect, beforeEach } from 'vitest';
import {
  rememberViewport,
  recallViewport,
  forgetViewport,
  _resetViewportMemory,
} from './viewportMemory';

beforeEach(() => {
  _resetViewportMemory();
});

describe('viewportMemory', () => {
  it('recalls what was remembered for a tab', () => {
    rememberViewport('t1', { x: 10, y: -20, zoom: 1.5 });
    expect(recallViewport('t1')).toEqual({ x: 10, y: -20, zoom: 1.5 });
  });

  it('returns undefined for a tab that was never viewed', () => {
    expect(recallViewport('never')).toBeUndefined();
  });

  it('keeps tabs independent', () => {
    rememberViewport('t1', { x: 1, y: 1, zoom: 1 });
    rememberViewport('t2', { x: 2, y: 2, zoom: 2 });
    expect(recallViewport('t1')).toEqual({ x: 1, y: 1, zoom: 1 });
    expect(recallViewport('t2')).toEqual({ x: 2, y: 2, zoom: 2 });
  });

  it('overwrites on a second remember', () => {
    rememberViewport('t1', { x: 1, y: 1, zoom: 1 });
    rememberViewport('t1', { x: 9, y: 9, zoom: 3 });
    expect(recallViewport('t1')).toEqual({ x: 9, y: 9, zoom: 3 });
  });

  it('forgets a closed tab', () => {
    rememberViewport('t1', { x: 1, y: 1, zoom: 1 });
    forgetViewport('t1');
    expect(recallViewport('t1')).toBeUndefined();
  });

  it('forgetting an unknown tab is a no-op', () => {
    expect(() => forgetViewport('ghost')).not.toThrow();
  });

  it('a reused tab id starts fresh after being forgotten', () => {
    rememberViewport('t1', { x: 1, y: 1, zoom: 1 });
    forgetViewport('t1');
    rememberViewport('t1', { x: 5, y: 5, zoom: 5 });
    expect(recallViewport('t1')).toEqual({ x: 5, y: 5, zoom: 5 });
  });
});
