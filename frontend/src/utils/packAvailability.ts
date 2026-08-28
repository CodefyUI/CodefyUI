import { useCallback } from 'react';
import { usePackStore } from '../store/packStore';
import type { PackSummary } from '../api/rest';
import type { NodeDefinition, ParamDefinition } from '../types';

/**
 * The one place that answers "is this option / node usable on this install?".
 *
 * Every consumer -- the param field's greyed-out options, the palette's badge,
 * the node header's PACK chip, the pre-run check -- goes through here rather
 * than reading `usePackStore` itself, for two reasons:
 *
 *  - One rule, one bug. The rule is subtle (a per-item requirement asks a
 *    different question than a whole-pack one) and getting it different in
 *    two places means a node that says PACK over a param that says fine.
 *  - One seam to mock. A component test that only wants to prove a tooltip
 *    mocks THIS module instead of standing up a catalog, a store and the REST
 *    layer behind it.
 *
 * The import graph is deliberately one-way: `packAvailability -> packStore ->
 * rest / toastStore / i18n`. Nothing in the store or in `nodeDefStore` imports
 * this file back.
 *
 * ── Why every unknown resolves to "available" ──────────────────────────
 * The failure modes are not symmetric. Wrongly greying an option out hides a
 * feature that works, with no way for the user to find out why; wrongly
 * leaving it enabled costs one clear backend error naming the pack to
 * install. So an unloaded catalog, a server with no Package Center at all, a
 * pack id this catalog never heard of and an item id it does not list ALL
 * come back available, and only a catalog that positively says "not
 * installed" greys anything out.
 */

/** Packs keyed by id -- `usePackStore`'s `byId` slice. */
export type PackIndex = Record<string, PackSummary>;

/**
 * A parsed `option_packs` / `requires_pack` value.
 *
 * `itemId === null` means the whole pack; a non-null one names a single
 * downloaded model inside it, which is what lets a select grey out only the
 * two embedding models that were not fetched instead of the whole parameter.
 */
export interface PackRequirement {
  packId: string;
  itemId: string | null;
}

/**
 * Split `"<pack>"` or `"<pack>:<item>"`.
 *
 * Anything else -- an empty half, a second colon -- is kept WHOLE as the pack
 * id rather than rejected. That is not laziness: a malformed value is a typo
 * in a node definition or a plugin built against a different convention, and
 * "a pack id nothing in the catalog matches" is exactly the tolerant answer
 * (available) the caller wants for it.
 */
export function parseRequirement(value: string): PackRequirement {
  const trimmed = value.trim();
  const parts = trimmed.split(':').map((part) => part.trim());
  if (parts.length === 2 && parts[0] && parts[1]) {
    return { packId: parts[0], itemId: parts[1] };
  }
  return { packId: trimmed, itemId: null };
}

/**
 * Read one pack out of the index by id, treating an inherited member as absent.
 *
 * `byId` is built from parsed JSON, so a plain `byId[id]` answers with a
 * function for `toString`, `constructor` and the rest of `Object.prototype`.
 * A node declaring `requires_pack: "toString"` would then be measured against
 * `Function.prototype.usable` (undefined) and greyed out for good -- the exact
 * "unknown pack" case this module promises to leave enabled.
 */
function lookupPack(byId: PackIndex, packId: string): PackSummary | undefined {
  return Object.prototype.hasOwnProperty.call(byId, packId) ? byId[packId] : undefined;
}

/**
 * Tolerant on purpose: an unloaded catalog, an unsupported server, an unknown
 * pack or item never greys anything out.
 *
 * A bare pack requirement asks the server's own verdict (`usable`), which
 * folds in the pack's dependencies. A per-item requirement asks the two
 * halves it actually needs -- the Python side installed (`pip_ready`) and
 * THAT file downloaded -- and deliberately ignores `usable`, since a pack
 * whose other three models are missing is still "partial" while the one this
 * option wants is ready to load.
 */
export function isRequirementAvailable(
  byId: PackIndex,
  loaded: boolean,
  unsupported: boolean,
  req: PackRequirement | null,
): boolean {
  if (!req) return true;
  if (!loaded || unsupported) return true;

  const pack = lookupPack(byId, req.packId);
  if (!pack) return true;
  if (req.itemId === null) return pack.usable;

  const item = pack.items.find((candidate) => candidate.id === req.itemId);
  if (!item) return true;
  return pack.pip_ready && item.status === 'present';
}

/**
 * Read a raw requirement string, ignoring the absent and the blank.
 *
 * `typeof` rather than a truthiness check because both maps arrive as parsed
 * JSON: an option literally named `toString` would otherwise inherit a
 * function off `Object.prototype` and be treated as a requirement.
 */
function requirementFrom(value: unknown): PackRequirement | null {
  if (typeof value !== 'string' || !value.trim()) return null;
  return parseRequirement(value);
}

/**
 * The pack a select option needs and does not have, or null when the option
 * works. Null is the answer for every param on a base install, so callers can
 * treat a non-null result as the rare case.
 */
export function missingRequirementForOption(
  param: ParamDefinition,
  option: string,
  byId: PackIndex,
  loaded: boolean,
  unsupported: boolean,
): PackRequirement | null {
  const req = requirementFrom(param.option_packs?.[option]);
  if (!req) return null;
  return isRequirementAvailable(byId, loaded, unsupported, req) ? null : req;
}

/** The same question for a whole node (`requires_pack`). */
export function nodeMissingPack(
  def: NodeDefinition | undefined,
  byId: PackIndex,
  loaded: boolean,
  unsupported: boolean,
): PackRequirement | null {
  const req = requirementFrom(def?.requires_pack);
  if (!req) return null;
  return isRequirementAvailable(byId, loaded, unsupported, req) ? null : req;
}

/**
 * A pack's human name, falling back to its id.
 *
 * The fallback is load-bearing rather than cosmetic: a node can report a pack
 * before the catalog answers, or name one this server does not ship, and
 * "install the word-vectors pack" is still a usable sentence.
 */
export function packTitle(byId: PackIndex, packId: string): string {
  return lookupPack(byId, packId)?.title ?? packId;
}

/**
 * What to call the thing that is missing.
 *
 * Item ids in the catalog are already human-readable (`all-MiniLM-L6-v2`,
 * `glove-6b-50d`) and are what the Package Center lists them under, so the
 * id IS the label; a pack-wide requirement falls back to the pack's title.
 */
export function itemTitle(byId: PackIndex, req: PackRequirement): string {
  return req.itemId ?? packTitle(byId, req.packId);
}

type PackStoreState = ReturnType<typeof usePackStore.getState>;

// Module-scope selectors, so each subscription compares the SAME function's
// output frame to frame. Three narrow selectors rather than one object one:
// an install running elsewhere writes `loading`, `busy`, `job` and the log on
// every long-poll turn, and a component holding this hook must not re-render
// for any of it.
const selectById = (state: PackStoreState): PackIndex => state.byId;
const selectLoaded = (state: PackStoreState): boolean => state.loaded;
const selectUnsupported = (state: PackStoreState): boolean => state.unsupported;

/**
 * The component-facing form: the three slices plus a ready-made predicate
 * over a raw `option_packs` / `requires_pack` value.
 *
 * `isAvailable` is memoised on exactly those three slices, so it is stable
 * between renders and safe in a `useMemo` / `useEffect` dependency list --
 * which matters because the caller is usually mapping it over every option of
 * every select on the canvas.
 */
export function usePackAvailability(): {
  byId: PackIndex;
  loaded: boolean;
  unsupported: boolean;
  isAvailable: (value: string | null | undefined) => boolean;
} {
  const byId = usePackStore(selectById);
  const loaded = usePackStore(selectLoaded);
  const unsupported = usePackStore(selectUnsupported);

  const isAvailable = useCallback(
    (value: string | null | undefined): boolean =>
      isRequirementAvailable(byId, loaded, unsupported, requirementFrom(value)),
    [byId, loaded, unsupported],
  );

  return { byId, loaded, unsupported, isAvailable };
}
