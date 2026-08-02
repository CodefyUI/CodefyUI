import { useEffect, useRef, useState } from 'react';
import { validateScript, type ScriptValidation } from '../api/rest';

/**
 * Ask the server whether an in-canvas script satisfies the Tier-0 policy
 * (core#131), debounced, while the user types.
 *
 * The gate is an AST walk, so the verdict can only come from the backend.
 * The point of asking on every edit rather than at run time is that
 * `import requests` should be a red line under the editor two keystrokes
 * later, not a failed run after a five-minute training node upstream.
 *
 * Verdicts are cached by exact source text and shared process-wide: the
 * param column and the Code tab render the same script at the same time, an
 * undo returns to a string that was already checked, and neither should cost
 * a second request.
 */
export type ScriptValidationStatus = 'idle' | 'checking' | 'done' | 'unavailable';

export interface ScriptValidationState {
  status: ScriptValidationStatus;
  result: ScriptValidation | null;
}

const DEBOUNCE_MS = 400;
/** Verdicts kept. Insertion-ordered, so the oldest is the first key. */
const CACHE_LIMIT = 60;

const cache = new Map<string, ScriptValidation>();
const inflight = new Map<string, Promise<ScriptValidation>>();

function remember(code: string, verdict: ScriptValidation): void {
  cache.set(code, verdict);
  while (cache.size > CACHE_LIMIT) {
    const oldest = cache.keys().next();
    if (oldest.done) break;
    cache.delete(oldest.value);
  }
}

function check(code: string): Promise<ScriptValidation> {
  const pending = inflight.get(code);
  if (pending) return pending;
  const request = validateScript(code)
    .then((verdict) => {
      remember(code, verdict);
      return verdict;
    })
    .finally(() => {
      inflight.delete(code);
    });
  inflight.set(code, request);
  return request;
}

/** Test seam: forget every cached verdict. */
export function resetScriptValidationCache(): void {
  cache.clear();
  inflight.clear();
}

export function useScriptValidation(
  code: string,
  enabled = true,
): ScriptValidationState {
  const [state, setState] = useState<ScriptValidationState>(() => {
    const cached = cache.get(code);
    return cached
      ? { status: 'done', result: cached }
      : { status: 'idle', result: null };
  });
  // Guards a response that arrives after the user has typed on: only the
  // request for the CURRENT text may write state.
  const latest = useRef(code);

  useEffect(() => {
    latest.current = code;
    if (!enabled) {
      setState({ status: 'idle', result: null });
      return;
    }

    const cached = cache.get(code);
    if (cached) {
      setState({ status: 'done', result: cached });
      return;
    }

    setState((prev) => ({ status: 'checking', result: prev.result }));
    let cancelled = false;
    const timer = setTimeout(() => {
      check(code)
        .then((verdict) => {
          if (cancelled || latest.current !== code) return;
          setState({ status: 'done', result: verdict });
        })
        .catch(() => {
          if (cancelled || latest.current !== code) return;
          setState({ status: 'unavailable', result: null });
        });
    }, DEBOUNCE_MS);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [code, enabled]);

  return state;
}
