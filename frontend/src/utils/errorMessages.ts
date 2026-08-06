/**
 * Maps a raw Python exception into a message a beginner can act on.
 *
 * The exception type arrives as its own field (`error_type` on the backend's
 * error payload). It used to be recovered by scanning the message for a
 * `KeyError:` prefix — which `str(exc)` never produces, since
 * `str(KeyError('tensor'))` is just `"'tensor'"`. Every rule here was
 * therefore unreachable in production while its tests passed, because the
 * tests fed it strings carrying a prefix the backend could not emit.
 *
 * `errorType` is optional: run records and server-level errors have no typed
 * field, and in DEBUG the payload carries a traceback whose last line *does*
 * name the class. So a prefix scan stays as a fallback for those sources.
 *
 * Anything unrecognised is returned unchanged, which also makes a second pass
 * over already-friendly text a no-op — several panels call this on messages
 * that were mapped once already at ingestion.
 */
import { useI18n } from '../i18n';

/** Recover the class name from a traceback's last line, for untyped sources. */
function typeFromTraceback(raw: string): string | undefined {
  return raw.match(/\b([A-Z]\w*(?:Error|Exception))\s*:/)?.[1];
}

export function friendlyError(raw: string, errorType?: string): string {
  if (!raw) return raw;
  const t = useI18n.getState().t;
  const kind = errorType || typeFromTraceback(raw);

  // A missing input. The key names the port the node wanted.
  if (kind === 'KeyError') {
    // Typed payloads give a bare "'tensor'"; tracebacks give "KeyError: 'tensor'".
    const key = raw.match(/'([^']+)'/)?.[1];
    if (key) {
      return key === 'tensor'
        ? t('error.missingTensorInput')
        : t('error.missingInput', { key });
    }
  }

  // The most common mistake in a first CNN or MLP: a Linear layer's
  // in_features does not match what the previous layer produced. PyTorch's own
  // wording names two matrices the student never typed, so it reads as an
  // internal fault rather than a wiring mistake.
  const matmul = raw.match(
    /mat1 and mat2 shapes cannot be multiplied \(\d+x(\d+) and (\d+)x\d+\)/,
  );
  if (matmul) {
    return t('error.linearShapeMismatch', { got: matmul[1], expected: matmul[2] });
  }

  // A reshape/view whose target shape does not divide the tensor's size.
  const invalidShape = raw.match(/shape\s+'(\[[^\]]*\])'\s+is invalid for input of size (\d+)/);
  if (invalidShape) {
    return t('error.invalidReshape', { shape: invalidShape[1], size: invalidShape[2] });
  }

  // Channel mismatch between a conv layer and its input.
  const channels = raw.match(
    /expected input\[[^\]]*\] to have (\d+) channels, but got (\d+) channels/,
  );
  if (channels) {
    return t('error.channelMismatch', { expected: channels[1], got: channels[2] });
  }

  // ValueError already carries a message written for a human — surface it
  // without the class name, which adds nothing for a beginner.
  if (kind === 'ValueError') {
    return raw.replace(/^ValueError:\s*/, '').trim() || raw;
  }

  return raw;
}
