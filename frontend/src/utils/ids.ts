/**
 * Client-side id generation.
 *
 * Its own module so leaf utilities (`utils/subgraph.ts`) can use it without
 * importing the barrel `utils/index.ts`, which imports them back. `index.ts`
 * re-exports `generateId`, so every existing caller is unchanged.
 */
export function generateId(): string {
  return crypto.randomUUID();
}
