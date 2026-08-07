import { describe, it, expect, afterEach, vi } from 'vitest';
import { generateId } from './ids';

/**
 * `crypto.randomUUID` is a secure-context-only API. Serving the editor over
 * plain HTTP on a LAN address — `cdui start --host <lan-ip>`, the way a
 * classroom is set up — leaves it `undefined`, so these tests pin the
 * behaviour of every degraded shape the browser can hand us.
 */

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

const realCrypto = globalThis.crypto;

function setCrypto(value: unknown) {
  Object.defineProperty(globalThis, 'crypto', { configurable: true, value });
}

afterEach(() => {
  setCrypto(realCrypto);
  vi.restoreAllMocks();
});

describe('generateId', () => {
  it('uses crypto.randomUUID when the page is in a secure context', () => {
    const randomUUID = vi.fn(() => '11111111-2222-4333-8444-555555555555');
    setCrypto({ ...realCrypto, randomUUID });

    expect(generateId()).toBe('11111111-2222-4333-8444-555555555555');
    expect(randomUUID).toHaveBeenCalledOnce();
  });

  describe('without crypto.randomUUID (plain-HTTP LAN serving)', () => {
    it('still returns a UUID-shaped string', () => {
      setCrypto({ getRandomValues: realCrypto.getRandomValues.bind(realCrypto) });
      expect(generateId()).toMatch(UUID_RE);
    });

    it('sets the version and variant nibbles of a v4 UUID', () => {
      setCrypto({ getRandomValues: realCrypto.getRandomValues.bind(realCrypto) });
      const id = generateId();
      expect(id[14]).toBe('4');
      expect(['8', '9', 'a', 'b']).toContain(id[19]);
    });

    it('uses crypto.getRandomValues, which is available in insecure contexts', () => {
      const getRandomValues = vi.fn((a: Uint8Array) => realCrypto.getRandomValues(a));
      setCrypto({ getRandomValues });

      generateId();
      expect(getRandomValues).toHaveBeenCalledOnce();
    });

    it('returns unique values', () => {
      setCrypto({ getRandomValues: realCrypto.getRandomValues.bind(realCrypto) });
      const ids = new Set(Array.from({ length: 1000 }, () => generateId()));
      expect(ids.size).toBe(1000);
    });
  });

  describe('with no crypto object at all', () => {
    it('still returns a UUID-shaped string rather than throwing', () => {
      setCrypto(undefined);
      expect(() => generateId()).not.toThrow();
      expect(generateId()).toMatch(UUID_RE);
    });

    it('returns unique values', () => {
      setCrypto(undefined);
      const ids = new Set(Array.from({ length: 1000 }, () => generateId()));
      expect(ids.size).toBe(1000);
    });
  });

  it('does not throw when crypto exists but exposes neither method', () => {
    setCrypto({});
    expect(() => generateId()).not.toThrow();
    expect(generateId()).toMatch(UUID_RE);
  });
});
