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

/** Imported definitions may extend, but never replace, the installed palette. */
export function mergeUnknownPresetsIntoPalette(
  installed: PresetDefinition[],
  imported: PresetDefinition[],
): PresetDefinition[] {
  return mergeOwnedPresets(installed, imported);
}
