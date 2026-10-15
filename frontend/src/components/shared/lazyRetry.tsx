import { lazy, useEffect, useRef, useState, type ComponentType } from 'react';
import { useI18n } from '../../i18n';
import { useToastStore } from '../../store/toastStore';

/**
 * `React.lazy` for a dialog or panel opened from a button, made safe for a
 * chunk that fails to load: a flaky network, or a 404 once `cdui update`
 * has replaced `dist/` under an open page. Plain `lazy` throws the failure
 * to the nearest error boundary, which for these is the app's crash
 * screen, and caches it for the life of the page.
 *
 * Here a failed import renders nothing, shows an error toast and calls
 * *onFail* with the props (so the caller closes what it was opening), and
 * the next mount asks for the chunk again, as `CodeEditor` and `MathText`
 * do for theirs. Browsers may remember a failed module fetch until a
 * reload; the toast says so.
 */
export function lazyRetry<P extends object>(
  load: () => Promise<ComponentType<P>>,
  onFail: (props: P) => void,
): ComponentType<P> {
  const make = () => lazy(async () => {
    try {
      return { default: await load() };
    } catch (error) {
      return {
        default: (props: P) => (
          <ChunkFailed
            error={error}
            onFail={() => {
              // Only once the failure is on screen: a mount that suspended
              // is rendered again from scratch, and must meet the same
              // settled `lazy` rather than start another import.
              current = make();
              onFail(props);
            }}
          />
        ),
      };
    }
  });
  let current = make();
  function LazyRetry(props: P) {
    // One `lazy` per mount: a retry is a new mount, never a re-render.
    const [Component] = useState(() => current);
    return <Component {...props} />;
  }
  return LazyRetry;
}

function ChunkFailed({ error, onFail }: { error: unknown; onFail: () => void }) {
  const reported = useRef(false);
  useEffect(() => {
    // Once, even when StrictMode runs effects twice.
    if (reported.current) return;
    reported.current = true;
    console.warn('A lazily loaded part of the app failed to load', error);
    useToastStore.getState().addToast(useI18n.getState().t('lazyChunk.failed'), 'error');
    onFail();
  }, [error, onFail]);
  return null;
}
