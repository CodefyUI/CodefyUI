import { describe, it, expect, beforeEach } from 'vitest';
import { useDialogStore } from './dialogStore';
import type { ConfirmRequest, PromptRequest } from './dialogStore';

const confirmReq: ConfirmRequest = { kind: 'confirm', title: 'Sure?' };
const promptReq: PromptRequest = { kind: 'prompt', title: 'Name?' };

// dialog.test.ts already covers the "prior resolver exists" branch of open()
// and the "resolver exists" branch of close(). These tests close the remaining
// branches: opening with NO prior resolver, and closing with NO resolver.
describe('useDialogStore', () => {
  beforeEach(() => {
    useDialogStore.setState({ active: null, resolve: null });
  });

  describe('open', () => {
    it('stores the active request and resolver when none is open (no prior resolver)', () => {
      const resolve = (_v: boolean) => {};
      useDialogStore.getState().open(confirmReq, resolve);
      const state = useDialogStore.getState();
      expect(state.active).toEqual(confirmReq);
      expect(state.resolve).toBe(resolve);
    });

    it('cancels the prior resolver with null when the incoming request is a prompt', () => {
      // The cancel value is keyed off the INCOMING request.kind, not the prior
      // one: open() does `prior(request.kind === 'prompt' ? null : false)`.
      let priorValue: boolean | string | null = 'untouched';
      useDialogStore.getState().open(confirmReq, (v: boolean) => {
        priorValue = v as unknown as boolean;
      });
      useDialogStore.getState().open(promptReq, () => {});
      expect(priorValue).toBeNull();
      expect(useDialogStore.getState().active).toEqual(promptReq);
    });

    it('cancels the prior resolver with false when the incoming request is a confirm', () => {
      let priorValue: boolean | string | null = 'untouched';
      useDialogStore.getState().open(promptReq, (v: string | null) => {
        priorValue = v as unknown as string | null;
      });
      useDialogStore.getState().open(confirmReq, () => {});
      expect(priorValue).toBe(false);
      expect(useDialogStore.getState().active).toEqual(confirmReq);
    });
  });

  describe('close', () => {
    it('clears state and invokes the resolver with the value', () => {
      let received: boolean | string | null = 'untouched';
      useDialogStore.getState().open(confirmReq, (v: boolean) => {
        received = v;
      });
      useDialogStore.getState().close(true);
      expect(received).toBe(true);
      const state = useDialogStore.getState();
      expect(state.active).toBeNull();
      expect(state.resolve).toBeNull();
    });

    it('is a no-op resolver-wise when nothing is open (no resolver branch)', () => {
      // resolve is null here — close must not throw and must leave state cleared.
      expect(() => useDialogStore.getState().close(true)).not.toThrow();
      const state = useDialogStore.getState();
      expect(state.active).toBeNull();
      expect(state.resolve).toBeNull();
    });
  });
});
