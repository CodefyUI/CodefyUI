/**
 * Client-side id generation.
 *
 * Its own module so leaf utilities (`utils/subgraph.ts`) can use it without
 * importing the barrel `utils/index.ts`, which imports them back. `index.ts`
 * re-exports `generateId`, so every existing caller is unchanged.
 */

/**
 * `crypto.randomUUID` is restricted to secure contexts — HTTPS or localhost.
 * Serving the editor on a plain-HTTP LAN address (`cdui start --host <lan-ip>`,
 * which is how a classroom is set up) leaves it `undefined`, and this module is
 * reached during `tabStore`'s module evaluation. An unguarded call there throws
 * before `createRoot().render()` ever runs, so the whole app renders as a blank
 * page with nothing but a console TypeError to show for it.
 *
 * `crypto.getRandomValues` carries no such restriction, so the fallback keeps
 * both the entropy and the RFC 4122 v4 shape that call sites and
 * `utils/index.test.ts` assert. Only when there is no `crypto` at all does this
 * drop to `Math.random`.
 */
function randomBytes(n: number): Uint8Array {
  const bytes = new Uint8Array(n);
  const c: Crypto | undefined = typeof crypto !== 'undefined' ? crypto : undefined;
  if (typeof c?.getRandomValues === 'function') {
    c.getRandomValues(bytes);
    return bytes;
  }
  for (let i = 0; i < n; i++) bytes[i] = Math.floor(Math.random() * 256);
  return bytes;
}

/** Format 16 random bytes as an RFC 4122 version-4 UUID. */
function uuidV4(): string {
  const b = randomBytes(16);
  b[6] = (b[6] & 0x0f) | 0x40; // version 4
  b[8] = (b[8] & 0x3f) | 0x80; // variant 10xx
  const hex: string[] = [];
  for (let i = 0; i < 16; i++) hex.push(b[i].toString(16).padStart(2, '0'));
  return [
    hex.slice(0, 4).join(''),
    hex.slice(4, 6).join(''),
    hex.slice(6, 8).join(''),
    hex.slice(8, 10).join(''),
    hex.slice(10, 16).join(''),
  ].join('-');
}

export function generateId(): string {
  const c: Crypto | undefined = typeof crypto !== 'undefined' ? crypto : undefined;
  if (typeof c?.randomUUID === 'function') return c.randomUUID();
  return uuidV4();
}
