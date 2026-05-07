import { describe, it, expect, beforeEach } from 'vitest';
import { confirm, prompt } from './dialog';
import { useDialogStore } from '../store/dialogStore';

describe('confirm()', () => {
  beforeEach(() => {
    // Reset store between tests so a leaked dialog from one doesn't pollute another.
    useDialogStore.setState({ active: null, resolve: null });
  });

  it('opens a confirm request and resolves true on close(true)', async () => {
    const p = confirm({ title: 'Delete?' });
    const state = useDialogStore.getState();
    expect(state.active?.kind).toBe('confirm');
    expect(state.active?.title).toBe('Delete?');
    state.close(true);
    await expect(p).resolves.toBe(true);
  });

  it('resolves false on close(false)', async () => {
    const p = confirm({ title: 'X' });
    useDialogStore.getState().close(false);
    await expect(p).resolves.toBe(false);
  });

  it('resolves false on close(null)', async () => {
    const p = confirm({ title: 'X' });
    useDialogStore.getState().close(null);
    await expect(p).resolves.toBe(false);
  });

  it('opening a second dialog cancels the first', async () => {
    const first = confirm({ title: 'first' });
    const second = confirm({ title: 'second' });
    // First should resolve as cancelled — second is the active one now.
    await expect(first).resolves.toBe(false);
    expect(useDialogStore.getState().active?.title).toBe('second');
    useDialogStore.getState().close(true);
    await expect(second).resolves.toBe(true);
  });
});

describe('prompt()', () => {
  beforeEach(() => {
    useDialogStore.setState({ active: null, resolve: null });
  });

  it('resolves with the entered string on close(value)', async () => {
    const p = prompt({ title: 'Name?' });
    useDialogStore.getState().close('alice');
    await expect(p).resolves.toBe('alice');
  });

  it('resolves null on close(null)', async () => {
    const p = prompt({ title: 'Name?' });
    useDialogStore.getState().close(null);
    await expect(p).resolves.toBeNull();
  });

  it('resolves null on close(false) for prompt mode', async () => {
    // Defensive: even though the UI never calls close(false) for prompts,
    // the helper should normalise non-string returns to null.
    const p = prompt({ title: 'Name?' });
    useDialogStore.getState().close(false);
    await expect(p).resolves.toBeNull();
  });

  it('exposes the request shape on the store', () => {
    prompt({ title: 'Save as…', defaultValue: 'untitled', placeholder: 'name' });
    const a = useDialogStore.getState().active;
    expect(a?.kind).toBe('prompt');
    if (a?.kind === 'prompt') {
      expect(a.defaultValue).toBe('untitled');
      expect(a.placeholder).toBe('name');
    }
  });
});
