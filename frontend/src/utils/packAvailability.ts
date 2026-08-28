import { useCallback } from 'react';
import { usePackStore } from '../store/packStore';
import type { PackSummary } from '../api/rest';
import en from '../i18n/locales/en';
import type { TranslationKey } from '../i18n';
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
 * The import graph is `packAvailability -> packStore -> rest / toastStore /
 * i18n`, with ONE edge back: the store imports `localizedPackTitle` for the
 * name it puts in its toasts, because a pack has to be called the same thing
 * on a node, on a card and in a toast. The cycle is safe by construction --
 * everything the store reaches for here is a hoisted function declaration
 * that is only CALLED from inside another function, never while either module
 * is being evaluated -- and it is the only one: `nodeDefStore` and the rest
 * of the store layer do not import this file.
 *
 * The only runtime edge to the translation layer is the en message TABLE,
 * read to ask whether this build ships copy for a pack id -- never the i18n
 * STORE: every function here that needs to translate takes its `t` as an
 * argument, so all of them stay callable outside React.
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
 * A pack's RAW server name, falling back to its id.
 *
 * The fallback is load-bearing rather than cosmetic: a node can report a pack
 * before the catalog answers, or name one this server does not ship, and
 * "install the word-vectors pack" is still a usable sentence.
 *
 * Not what a user should be shown: the server's titles are English. Anything
 * with a `t` in hand calls `localizedPackTitle` instead, and this is its last
 * fallback.
 */
export function packTitle(byId: PackIndex, packId: string): string {
  return lookupPack(byId, packId)?.title ?? packId;
}

/** Signature of `useI18n`'s `t`, so this stays callable outside React. */
type Translate = (key: TranslationKey, vars?: Record<string, string | number>) => string;

/**
 * The i18n key for a pack's shipped copy, or null when this build has none.
 *
 * Catalog copy is keyed by PACK ID, and the backend is free to ship a pack
 * this frontend predates. `hasOwnProperty` rather than `in`, because the
 * message table is a plain object: an id like `constructor` would otherwise
 * "exist" and translate to the prototype's own member.
 *
 * Lives HERE, one layer below the Package Center that also imports it, so
 * that the panel and the node side cannot disagree about what a pack is
 * called -- see `localizedPackTitle`.
 */
export function catalogKey(
  packId: string,
  field: 'title' | 'desc',
): TranslationKey | null {
  const key = `packs.catalog.${packId}.${field}`;
  return Object.prototype.hasOwnProperty.call(en, key) ? (key as TranslationKey) : null;
}

/**
 * ONE pack, ONE name -- what every surface that names a pack to a user says.
 *
 * Three sources, in order: the copy this build ships for the pack (the only
 * one that is translated), the server's English title, and the bare id. Every
 * one of them can be the right answer -- a pack from a newer backend has no
 * shipped copy, and a node can name a pack before the catalog has answered at
 * all -- so all three are kept.
 *
 * The rule is centralised because a pack is named at half a dozen points of
 * ONE workflow: the palette badge, the node chip, the select hint, the panel
 * card, and the toast that says it installed. When the node side read the
 * server title while the panel read the catalog copy, a zh-TW reader was told
 * 「glove-50d」需要 Word vectors (GloVe) 套件 on the node, opened a card headed
 * 詞向量（GloVe）, and got 已安裝 Word vectors (GloVe)。 for the same pack.
 */
export function localizedPackTitle(
  t: Translate,
  byId: PackIndex,
  packId: string,
): string {
  const key = catalogKey(packId, 'title');
  return key !== null ? t(key) : packTitle(byId, packId);
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

/**
 * The one sentence that says why a chosen value cannot run.
 *
 * Lives here rather than in either component because BOTH say it: the node
 * card writes it into the `needs pack` marker's tooltip and the config panel
 * writes it under the select. They are two views of one parameter, and a
 * reader who hovers the card and then opens the panel must not be told two
 * different stories about what is missing -- which is exactly what a second
 * copy of these four lines eventually produces.
 */
export function requirementSentence(
  t: Translate,
  byId: PackIndex,
  option: string,
  req: PackRequirement,
): string {
  const pack = localizedPackTitle(t, byId, req.packId);
  return req.itemId === null
    ? t('paramField.packHint', { option, pack })
    : t('paramField.modelHint', { option, pack, item: itemTitle(byId, req) });
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
