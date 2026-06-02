import { describe, expect, it } from 'vitest';
import { friendlyError } from './errorMessages';

describe('friendlyError', () => {
  it('maps KeyError for "tensor" to a tensor-specific hint', () => {
    const out = friendlyError("KeyError: 'tensor'");
    expect(out).toBe(
      "This node expected a 'tensor' input but did not receive one. Check that all required inputs are connected.",
    );
  });

  it('maps KeyError for any other key to a generic missing-data message', () => {
    expect(friendlyError("KeyError: 'weights'")).toBe(
      "Missing required data: 'weights'. Ensure all inputs are connected.",
    );
  });

  it('detects KeyError embedded in a longer traceback', () => {
    const raw = 'Traceback (most recent call last):\n  ...\nKeyError: \'labels\'';
    expect(friendlyError(raw)).toBe(
      "Missing required data: 'labels'. Ensure all inputs are connected.",
    );
  });

  it('maps RuntimeError shape-invalid to a shape-mismatch message', () => {
    const raw = "RuntimeError: shape '[2, 3]' is invalid for input of size 5";
    expect(friendlyError(raw)).toBe(
      'Tensor shape mismatch. Check the dimensions of connected nodes.',
    );
  });

  it('extracts and trims the message after ValueError:', () => {
    expect(friendlyError('ValueError:   too many values to unpack  ')).toBe(
      'too many values to unpack',
    );
  });

  it('returns the raw message unchanged when nothing matches', () => {
    const raw = 'TypeError: unsupported operand type(s)';
    expect(friendlyError(raw)).toBe(raw);
  });

  it('returns an empty string unchanged', () => {
    expect(friendlyError('')).toBe('');
  });

  it('prefers the KeyError branch over a co-occurring ValueError', () => {
    // KeyError is checked first; a ValueError later in the string is ignored.
    const raw = "KeyError: 'tensor'\nValueError: ignored";
    expect(friendlyError(raw)).toBe(
      "This node expected a 'tensor' input but did not receive one. Check that all required inputs are connected.",
    );
  });
});
