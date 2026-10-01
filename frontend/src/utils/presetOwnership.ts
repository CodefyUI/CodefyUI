import type { PresetDefinition } from '../types';

/**
 * The name a `presets[]` entry is found by, or null for one nothing can
 * reference. The lists come off files and plugins unvalidated; an entry with
 * no name is dropped by the server too (`build_preset_fallback`), so it is
 * skipped rather than read.
 */
export function presetDefinitionName(definition: unknown): string | null {
  if (definition === null || typeof definition !== 'object') return null;
  const name = (definition as { preset_name?: unknown }).preset_name;
  return typeof name === 'string' ? name : null;
}

/**
 * The fields a `presets[]` entry may leave out, with the defaults the server
 * reads it with (`PresetDefinition` in `backend/app/schemas/models.py`, the
 * installed registry's own defaults).
 */
function optionalPresetDefaults(): Record<string, unknown> {
  return {
    category: 'Preset',
    description: '',
    tags: [],
    exposed_inputs: [],
    exposed_outputs: [],
    exposed_params: [],
  };
}

/**
 * A document's `presets[]` as a tab adopts it (#541). An entry with what the
 * server requires -- a name, and `nodes` and `edges` lists -- gets the server's
 * default for each optional field it leaves out, so it reads everywhere as the
 * definition the server runs. Any other entry is kept exactly as it came: a
 * marker the server refuses by the preset's name.
 *
 * Only a MISSING field is filled. One that is there with the wrong type is
 * the server's to refuse, so it is left as it is. Returns the same array when
 * nothing was missing.
 */
export function withPresetDefaults(list: unknown): PresetDefinition[] {
  if (!Array.isArray(list)) return [];
  let changed = false;
  const adopted = list.map((entry) => {
    if (presetDefinitionName(entry) === null) return entry;
    const source = entry as Record<string, unknown>;
    if (!Array.isArray(source.nodes) || !Array.isArray(source.edges)) return entry;
    let filled: Record<string, unknown> | null = null;
    for (const [field, value] of Object.entries(optionalPresetDefaults())) {
      if (source[field] !== undefined) continue;
      if (filled === null) filled = { ...source };
      filled[field] = value;
    }
    if (filled === null) return entry;
    changed = true;
    return filled;
  });
  return (changed ? adopted : list) as PresetDefinition[];
}

/** Document-owned definitions win; installed definitions only fill gaps. */
export function effectivePresets(
  owned: PresetDefinition[],
  installed: PresetDefinition[],
): PresetDefinition[] {
  const result: PresetDefinition[] = [];
  const names = new Set<string>();
  for (const definition of [...owned, ...installed]) {
    const name = presetDefinitionName(definition);
    if (name === null || names.has(name)) continue;
    names.add(name);
    result.push(definition);
  }
  return result;
}

/** Add definitions to a document without changing an existing name's meaning. */
export function mergeOwnedPresets(
  owned: PresetDefinition[],
  incoming: PresetDefinition[],
): PresetDefinition[] {
  const names = new Set(owned.map(presetDefinitionName));
  let result: PresetDefinition[] | null = null;
  for (const definition of incoming) {
    const name = presetDefinitionName(definition);
    if (name === null || names.has(name)) continue;
    names.add(name);
    if (result === null) result = [...owned];
    result.push(definition);
  }
  return result ?? owned;
}

/**
 * Whether `entry` has every top-level field of a preset definition, each of
 * its type -- what an entry `withPresetDefaults` could complete has, and what
 * the palette's rows read.
 */
export function isCompletePreset(entry: unknown): entry is PresetDefinition {
  if (presetDefinitionName(entry) === null) return false;
  const source = entry as Record<string, unknown>;
  return (
    typeof source.category === 'string'
    && typeof source.description === 'string'
    && Array.isArray(source.tags)
    && source.tags.every((tag) => typeof tag === 'string')
    && Array.isArray(source.nodes)
    && Array.isArray(source.edges)
    && Array.isArray(source.exposed_inputs)
    && Array.isArray(source.exposed_outputs)
    && Array.isArray(source.exposed_params)
  );
}

/**
 * Imported definitions may extend, but never replace, the installed palette.
 * Only complete ones: the palette offers a preset to drop into any tab, and an
 * entry the backend cannot read stays the document's own, as a marker.
 */
export function mergeUnknownPresetsIntoPalette(
  installed: PresetDefinition[],
  imported: PresetDefinition[],
): PresetDefinition[] {
  return mergeOwnedPresets(installed, imported.filter(isCompletePreset));
}
