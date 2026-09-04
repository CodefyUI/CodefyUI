import type { PluginCatalogEntry, PluginStatus } from '../../api/rest';
import type { TranslationKey } from '../../i18n/locales/en';
import type { Translate } from '../PackCenter/packStatus';
import type { PillTone } from '../shared/Pill';

/**
 * The pure half of the Plugin Center: everything the panel needs to decide
 * that does not need a DOM, a store or a clock.
 *
 * The same split as `PackCenter/packStatus.ts`, and for the same reason: each
 * of these is a rule with an edge case worth pinning in a test -- a status
 * this build has no word for, a step id from a newer backend, a plugin with
 * no repository to name -- and none of them are worth mounting a modal to
 * exercise.
 */

/**
 * `Translate` is the pack panel's, not a copy: both panels format the same
 * way, and one signature is what keeps a helper callable from outside React.
 */
export type { Translate };

/**
 * Re-exported from the store because this is where the panel looks for it.
 *
 * The source box needs the CLI's grammar to refuse `not a repo!` without a
 * round trip, and the store needs it to refuse the same string before it
 * inspects. One exported parser, so the two can never disagree about what a
 * source is.
 */
export { parseGitHubSource } from '../../store/pluginStore';

/**
 * A plugin status as a tone.
 *
 * `disabled` and `removed` are neutral on purpose: both are states the user
 * chose, and dressing a deliberate switch-off as a warning would say
 * something went wrong. `missing_files` is the only status that IS wrong --
 * the registry has the plugin and its directory is gone.
 */
export function statusTone(status: PluginStatus): PillTone {
  switch (status) {
    case 'installed':
      return 'success';
    case 'installing':
      return 'info';
    case 'missing_files':
      return 'warning';
    default:
      return 'neutral';
  }
}

/**
 * The label for one plugin status.
 *
 * Three of the six reuse the pack panel's words, because "Installed" is
 * "Installed" and a second translation of it is a second thing to keep in
 * step; `disabled` reuses the sidebar's. Only the two states packs have no
 * equivalent for get keys of their own.
 */
const STATUS_KEY: Record<PluginStatus, TranslationKey> = {
  installed: 'packs.status.installed',
  disabled: 'customNodes.disabled',
  available: 'packs.status.not_installed',
  removed: 'pluginCenter.status.removed',
  installing: 'packs.status.installing',
  missing_files: 'pluginCenter.status.missingFiles',
};

export function statusKey(status: PluginStatus): TranslationKey {
  // A status off a newer server must not render as `undefined`. "Not
  // installed" is the safe reading: it offers an Install button rather than
  // claiming something is in place.
  return STATUS_KEY[status] ?? STATUS_KEY.available;
}

// ── the filter ───────────────────────────────────────────────────────────

/** Which half of the catalog the list is showing. */
export type PluginFilter = 'all' | 'installed' | 'available';

/**
 * Whether a row belongs to the "Available" half.
 *
 * Written as the narrow half and negated for the other, so the two halves are
 * exhaustive by construction: a status this build has never heard of lands
 * under "Installed" rather than under neither, and no filter can make a row
 * disappear from both tabs. `removed` is available -- a tombstone is a plugin
 * that is not here and can be put back -- while `missing_files` and
 * `installing` are not: the lockfile has them.
 */
export function isAvailableStatus(status: PluginStatus): boolean {
  return status === 'available' || status === 'removed';
}

/** Whether *status* passes *filter*. */
export function matchesFilter(filter: PluginFilter, status: PluginStatus): boolean {
  if (filter === 'all') return true;
  return isAvailableStatus(status) === (filter === 'available');
}

/**
 * The eight step ids a plugin job emits, as sentences.
 *
 * `deps` is pip, which the pack panel already has a sentence for -- the same
 * work, said the same way, rather than a second string that has to stay in
 * step with it.
 */
const STEP_KEY: Record<string, TranslationKey | undefined> = {
  resolve: 'pluginCenter.step.resolve',
  download: 'pluginCenter.step.download',
  extract: 'pluginCenter.step.extract',
  verify: 'pluginCenter.step.verify',
  deps: 'packs.activity.step.pip',
  stage: 'pluginCenter.step.stage',
  lock: 'pluginCenter.step.lock',
  reload: 'pluginCenter.step.reload',
};

/**
 * A job step as a sentence, keyed off the step ID rather than its text.
 *
 * The server's `label` is English and written for a log ("Scanning
 * c1-tokenizer for unsafe code"); the step id is a stable vocabulary this can
 * translate. An id from a newer backend falls back to that label, which is at
 * least true, and to the raw id when even the label is empty.
 */
export function stepLabel(t: Translate, step: string, label: string): string {
  // No plugin step carries a `:item` half today, but the job protocol allows
  // one and the pack panel already splits on it: a `download:tarball` from a
  // newer backend still has to find the download sentence.
  const separator = step.indexOf(':');
  const kind = separator === -1 ? step : step.slice(0, separator);
  const key = STEP_KEY[kind];
  return key === undefined ? label || step : t(key);
}

/**
 * Where a plugin came from, as a key -- or null when the row says it better
 * itself.
 *
 * A plain third-party repository gets no chip: the card already prints
 * `owner/repo` beside it, and "GitHub" over a GitHub link is the same fact
 * twice. `local` outranks `official` because a linked folder is what is
 * actually being loaded, whoever published it.
 */
export function originLabel(entry: PluginCatalogEntry): TranslationKey | null {
  if (entry.kind === 'builtin') return 'pluginCenter.origin.builtin';
  if (entry.source_kind === 'local') return 'pluginCenter.origin.local';
  if (entry.official) return 'pluginCenter.origin.official';
  return null;
}

/**
 * The three capabilities a manifest may declare, each as the sentence that
 * says what granting it costs.
 *
 * A fixed map rather than a templated key, because `t()` takes a
 * `TranslationKey` and a capability id is a value off the wire: an id a newer
 * server invents must reach the card as itself, not as a missing key.
 */
const CAPABILITY_KEY: Record<string, TranslationKey | undefined> = {
  network: 'pluginCenter.cap.network',
  filesystem: 'pluginCenter.cap.filesystem',
  'process-env': 'pluginCenter.cap.processEnv',
};

/** The sentence for *id*, or null when the card should print the raw id. */
export function capabilityKey(id: string): TranslationKey | null {
  return CAPABILITY_KEY[id] ?? null;
}

/**
 * The `cdui plugin install ...` line for *entry*, for the terminal fallback
 * the activity pane offers when a job fails.
 *
 * `repo` before `source`, and the ref pinned when there is one: what the user
 * copies should reproduce THIS row rather than whatever the default branch
 * holds by the time they paste it. `ref` is `''` for a default-branch install
 * -- a real answer, not a miss -- so it is appended only when set.
 */
export function cliInstallCommand(entry: PluginCatalogEntry): string {
  const repo = entry.repo ?? '';
  const spec = repo || entry.source || entry.id;
  const ref = repo && entry.ref ? `@${entry.ref}` : '';
  return `cdui plugin install ${spec}${ref}`;
}
