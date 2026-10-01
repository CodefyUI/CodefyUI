import type { PresetDefinition } from '../types';

/** Document-owned definitions win; installed definitions only fill gaps. */
export function effectivePresets(
  owned: PresetDefinition[],
  installed: PresetDefinition[],
): PresetDefinition[] {
  const result: PresetDefinition[] = [];
  const names = new Set<string>();
  for (const definition of [...owned, ...installed]) {
    if (names.has(definition.preset_name)) continue;
    names.add(definition.preset_name);
    result.push(definition);
  }
  return result;
}

/** Add definitions to a document without changing an existing name's meaning. */
export function mergeOwnedPresets(
  owned: PresetDefinition[],
  incoming: PresetDefinition[],
): PresetDefinition[] {
  const names = new Set(owned.map((definition) => definition.preset_name));
  let result: PresetDefinition[] | null = null;
  for (const definition of incoming) {
    if (names.has(definition.preset_name)) continue;
    names.add(definition.preset_name);
    if (result === null) result = [...owned];
    result.push(definition);
  }
  return result ?? owned;
}

/**
 * Palette entries a document's reader added, which the server's preset
 * registry does not have (#541). Held by identity, so the server's own list,
 * which the next fetch puts in their place, is never mistaken for one.
 */
const broughtByDocuments = new WeakSet<PresetDefinition>();

/** Imported definitions may extend, but never replace, the installed palette. */
export function mergeUnknownPresetsIntoPalette(
  installed: PresetDefinition[],
  imported: PresetDefinition[],
): PresetDefinition[] {
  const merged = mergeOwnedPresets(installed, imported);
  if (merged !== installed) {
    const alreadyThere = new Set(installed);
    for (const definition of merged) {
      if (!alreadyThere.has(definition)) broughtByDocuments.add(definition);
    }
  }
  return merged;
}

/**
 * Whether a document's reader, not the server, put `definition` in the
 * palette -- so the server never consults it when it blanks the SECRET values
 * of a run it stores.
 */
export function isBroughtByDocument(definition: PresetDefinition): boolean {
  return broughtByDocuments.has(definition);
}
