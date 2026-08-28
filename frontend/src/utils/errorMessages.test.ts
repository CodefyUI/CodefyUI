import { describe, expect, it, beforeEach } from 'vitest';
import { friendlyError, missingPackFromError } from './errorMessages';
import { useI18n } from '../i18n';
import { usePackStore } from '../store/packStore';
import type { PackSummary } from '../api/rest';

/**
 * These previously fed `friendlyError` strings like "KeyError: 'tensor'".
 * The backend cannot produce that: it sends `str(exc)`, and
 * `str(KeyError('tensor'))` is `"'tensor'"`. So every rule passed its test
 * and never fired in production. The payloads below are what the backend
 * actually emits, now that the type travels as its own field.
 */

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  // The pack rule reads the catalog for a human title; every case that does
  // not seed one is asserting the id fallback.
  usePackStore.setState({ byId: {} });
});

describe('friendlyError — payloads the backend actually sends', () => {
  it('names the missing tensor input from a bare KeyError message', () => {
    // str(KeyError('tensor')) === "'tensor'"
    const out = friendlyError("'tensor'", 'KeyError');
    expect(out).toContain('tensor');
    expect(out).toContain('connected');
  });

  it('names any other missing input', () => {
    expect(friendlyError("'weights'", 'KeyError')).toContain("'weights'");
  });

  it('does not treat a quoted word as a KeyError without the type', () => {
    // No error_type and no traceback -> nothing to key off. It must pass
    // through rather than guess that any quoted token is a missing port.
    expect(friendlyError("'tensor'")).toBe("'tensor'");
  });

  it('turns a Linear size mismatch into the two numbers to change', () => {
    const raw = 'mat1 and mat2 shapes cannot be multiplied (64x784 and 512x10)';
    const out = friendlyError(raw, 'RuntimeError');
    expect(out).toContain('784');
    expect(out).toContain('512');
    expect(out).toContain('in_features');
    expect(out).not.toBe(raw);
  });

  it('explains an invalid reshape with the element count', () => {
    const raw = "shape '[2, 3]' is invalid for input of size 5";
    const out = friendlyError(raw, 'RuntimeError');
    expect(out).toContain('[2, 3]');
    expect(out).toContain('5');
  });

  it('explains a conv channel mismatch', () => {
    const raw =
      'Given groups=1, weight of size [16, 3, 3, 3], expected input[1, 1, 28, 28] to have 3 channels, but got 1 channels instead';
    const out = friendlyError(raw, 'RuntimeError');
    expect(out).toContain('in_channels');
    expect(out).toContain('3');
    expect(out).toContain('1');
  });

  it('surfaces a ValueError message without the class name', () => {
    expect(friendlyError('epochs must be positive', 'ValueError')).toBe(
      'epochs must be positive',
    );
  });

  it('passes an unrecognised message through unchanged', () => {
    expect(friendlyError('something went sideways', 'RuntimeError')).toBe(
      'something went sideways',
    );
  });

  it('returns an empty message unchanged', () => {
    expect(friendlyError('')).toBe('');
  });
});

describe('friendlyError — untyped sources (run records, DEBUG tracebacks)', () => {
  it('recovers the class from a traceback when no type field is given', () => {
    const raw = "Traceback (most recent call last):\n  ...\nKeyError: 'labels'";
    expect(friendlyError(raw)).toContain("'labels'");
  });

  it('still matches shape errors with no type at all', () => {
    const raw = 'mat1 and mat2 shapes cannot be multiplied (8x128 and 64x10)';
    expect(friendlyError(raw)).toContain('in_features');
  });

  it('trims a ValueError prefix when it arrives inside a traceback', () => {
    expect(friendlyError('ValueError:   too many values to unpack  ')).toBe(
      'too many values to unpack',
    );
  });
});

// ── A node that needs an optional pack (PR 2) ────────────────────────────────
// `require_pack` raises PackMissingError with a message that always ends in
// `(pack=<id>)`; this is the exact wording the backend composes.
const PACK_MISSING =
  'Word vectors is not installed. Open Package Center (toolbar > Settings > ' +
  'Optional packs) to install it; graph runs never download (pack=word-vectors)';

describe('missingPackFromError — a node that needs an optional pack', () => {
  it('maps PackMissingError to the friendly sentence and extracts the pack id', () => {
    expect(missingPackFromError(PACK_MISSING, 'PackMissingError')).toBe('word-vectors');
    expect(friendlyError(PACK_MISSING, 'PackMissingError')).toBe(
      'This node needs the word-vectors pack. Install it from the Package Center.',
    );
  });

  it('ignores a plain error mentioning pack=', () => {
    const raw = "invoke() got an unexpected keyword argument 'pack=3'";
    expect(missingPackFromError(raw, 'TypeError')).toBeNull();
    expect(friendlyError(raw, 'TypeError')).toBe(raw);
  });

  it('recognises the untyped whole-run copy of the same message', () => {
    // A fail-fast run re-raises the node's exception, so `execution_error`
    // carries the identical `str(exc)` — with no `error_type` beside it. The
    // message naming the Package Center is what identifies it there.
    expect(missingPackFromError(PACK_MISSING)).toBe('word-vectors');
  });

  it('returns null when the type is right but the id is not in the message', () => {
    expect(missingPackFromError('something went sideways', 'PackMissingError')).toBeNull();
  });

  it('returns null for an empty message', () => {
    expect(missingPackFromError('', 'PackMissingError')).toBeNull();
  });

  it('names the pack by its catalog title once the store has one', () => {
    usePackStore.setState({
      byId: { 'word-vectors': { id: 'word-vectors', title: 'Word vectors' } as PackSummary },
    });
    expect(friendlyError(PACK_MISSING, 'PackMissingError')).toBe(
      'This node needs the Word vectors pack. Install it from the Package Center.',
    );
  });

  it('answers in Traditional Chinese under the zh-TW locale', () => {
    useI18n.setState({ locale: 'zh-TW' });
    const out = friendlyError(PACK_MISSING, 'PackMissingError');
    expect(out).toContain('word-vectors');
    expect(out).toMatch(/[一-鿿]/);
  });

  it('a second pass over the friendly sentence is a no-op', () => {
    // The mapped sentence names the Package Center too, so it would loop if
    // the `pack=` id were not also required.
    const once = friendlyError(PACK_MISSING, 'PackMissingError');
    expect(friendlyError(once)).toBe(once);
  });
});

describe('friendlyError — idempotence', () => {
  it('a second pass over already-friendly text is a no-op', () => {
    // ResultsPanel and RunsPanel both call this on lines that were already
    // mapped once at ingestion.
    const once = friendlyError(
      'mat1 and mat2 shapes cannot be multiplied (64x784 and 512x10)',
    );
    expect(friendlyError(once)).toBe(once);
  });
});

describe('friendlyError — localization', () => {
  it('returns Traditional Chinese under the zh-TW locale', () => {
    useI18n.setState({ locale: 'zh-TW' });
    const out = friendlyError(
      'mat1 and mat2 shapes cannot be multiplied (64x784 and 512x10)',
    );
    expect(out).toContain('784');
    expect(out).toMatch(/[一-鿿]/); // actually Chinese, not English
  });

  it('a raw message with no rule stays raw in either locale', () => {
    useI18n.setState({ locale: 'zh-TW' });
    expect(friendlyError('unmapped failure')).toBe('unmapped failure');
  });
});
