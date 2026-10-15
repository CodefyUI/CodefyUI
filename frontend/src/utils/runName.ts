/**
 * The server's run-name limit (`run_service.MAX_NAME_LENGTH`), in code points.
 * Also a Name field's `maxLength`, which counts UTF-16 units, at least one
 * per code point, so typing stops at the limit or before it. An input method
 * can compose past `maxLength`, so a typed name is cut when sent as well.
 */
export const MAX_RUN_NAME_LENGTH = 64;

/**
 * A run name the server stores rather than refuses (#623). `POST /api/runs`
 * and the sweep route REFUSE a name over the limit instead of cutting it, so
 * a name typed in a dialog is made storable here the way the server makes a
 * canvas run's: trimmed, cut to the limit in code points (`Array.from` counts
 * them, as the server does), and half of a surrogate pair, which the
 * database cannot store, sent as "?".
 */
export function storableRunName(raw: string): string {
  return Array.from(raw.trim(), (char) => {
    const code = char.codePointAt(0)!;
    return code >= 0xd800 && code <= 0xdfff ? '?' : char;
  }).slice(0, MAX_RUN_NAME_LENGTH).join('').trimEnd();
}

/**
 * What a blank Name sends: the tab's name, as a canvas run is named (#623).
 * Checked to be text, since a tab restored from a damaged record may not
 * carry one.
 */
export function tabDefaultName(tab: { name?: unknown } | null): string {
  return tab && typeof tab.name === 'string' ? storableRunName(tab.name) : '';
}
