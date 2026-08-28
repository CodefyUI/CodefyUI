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
import { usePackStore } from '../store/packStore';
import { packTitle } from './packAvailability';

/** Recover the class name from a traceback's last line, for untyped sources. */
function typeFromTraceback(raw: string): string | undefined {
  return raw.match(/\b([A-Z]\w*(?:Error|Exception))\s*:/)?.[1];
}

/** The id `PackMissingError` always appends: `... (pack=word-vectors)`. */
const PACK_ID = /pack=([a-z0-9-]+)/i;

/**
 * The optional pack a failure is really about, or null.
 *
 * Two ways in, because the same exception reaches the client twice wearing
 * different clothes. On `node_status` it comes typed, so the class name
 * settles it. On the whole-run `execution_error` a fail-fast run re-raises
 * it and only `str(exc)` survives — no type field at all — so the message
 * has to identify itself, which it does by naming the Package Center (the
 * one place that can fix it, and the reason `require_pack` writes it in).
 *
 * BOTH ways still require the `pack=<id>` suffix. Without an id there is no
 * panel to open and nothing to name, so a bare "PackMissingError" is not
 * worth a special sentence; and requiring it is also what keeps this
 * idempotent, since the friendly sentence below names the Package Center
 * too but carries no id.
 */
export function missingPackFromError(raw: string, errorType?: string): string | null {
  if (!raw) return null;
  const packId = raw.match(PACK_ID)?.[1];
  if (!packId) return null;
  if (errorType === 'PackMissingError') return packId;
  return raw.includes('Package Center') ? packId : null;
}

export function friendlyError(raw: string, errorType?: string): string {
  if (!raw) return raw;
  const t = useI18n.getState().t;
  const kind = errorType || typeFromTraceback(raw);

  // An optional pack the install does not have. First, because it is the
  // most specific rule here: its message quotes a model id, which the
  // KeyError branch below would otherwise read as a missing port. `kind`
  // rather than `errorType` so a DEBUG traceback names it too.
  const packId = missingPackFromError(raw, kind);
  if (packId) {
    // The title if the catalog has answered, the id if it has not — "needs
    // the word-vectors pack" is still a usable sentence.
    return t('error.missingPack', {
      pack: packTitle(usePackStore.getState().byId, packId),
    });
  }

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
