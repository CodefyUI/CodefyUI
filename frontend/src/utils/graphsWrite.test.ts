import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

import {
  announceGraphsWrite,
  getGraphsWriteListener,
  setGraphsWriteListener,
} from './graphsWrite';

/**
 * The saved-graph write signal, on its own.
 *
 * The one line in the save path that raises it is covered next door in
 * `saveActiveGraph.test.ts`, where the stores and the REST mock it needs
 * already exist. What is left here is the slot itself, and the reason it is
 * worth pinning at all is that every rule it has is a rule some caller is
 * relying on: the Graphs panel installs its listener from an effect and
 * clears it from that effect's cleanup, so "one slot, last writer wins" and
 * "getting the installed listener back" are what let a remount avoid
 * clearing the listener it has just put back.
 *
 * `listener` is module state, so each test puts the module back to "nobody
 * is listening" on the way out -- the resting state the save path assumes.
 */
beforeEach(() => {
  setGraphsWriteListener(null);
});

afterEach(() => {
  setGraphsWriteListener(null);
});

describe('the saved-graph write signal', () => {
  it('reaches the registered listener, and nobody once it is cleared', () => {
    const heard = vi.fn(() => undefined);
    setGraphsWriteListener(heard);

    announceGraphsWrite();
    expect(heard).toHaveBeenCalledTimes(1);

    setGraphsWriteListener(null);
    announceGraphsWrite();
    expect(heard).toHaveBeenCalledTimes(1);
  });

  it('holds one listener, so registering again replaces rather than adds', () => {
    const first = vi.fn(() => undefined);
    const second = vi.fn(() => undefined);
    setGraphsWriteListener(first);
    setGraphsWriteListener(second);

    announceGraphsWrite();

    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledTimes(1);
  });

  // A save is announced whether or not the sidebar is showing the Graphs
  // panel, so the unlistened call is the common one, not the edge case: it
  // has to be a no-op rather than a crash that takes the save's success
  // toast down with it.
  it('is a no-op when nobody is listening', () => {
    expect(() => announceGraphsWrite()).not.toThrow();
  });

  it('hands back the listener that is installed, so a caller can recognise its own', () => {
    expect(getGraphsWriteListener()).toBeNull();

    const mine = vi.fn(() => undefined);
    setGraphsWriteListener(mine);
    expect(getGraphsWriteListener()).toBe(mine);

    setGraphsWriteListener(null);
    expect(getGraphsWriteListener()).toBeNull();
  });
});
