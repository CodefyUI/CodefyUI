/**
 * The server version the last successful `/api/health` read reported.
 *
 * A module-level cache with one reader: the workspace export stamps
 * `app_version` into the file and must not hold a download up on a network
 * call. `fetchHealth` writes it, and the app calls that once at boot
 * (`App.tsx`), so the value is normally there long before anyone exports.
 * Null means "no health read has answered yet, or the server did not say".
 *
 * Its own module rather than an export of `api/rest.ts`: many test suites
 * replace that module with an explicit mock factory, and reading an export
 * the factory does not define throws.
 */
let _appVersion: string | null = null;

export function rememberAppVersion(version: string | null): void {
  _appVersion = version;
}

export function cachedAppVersion(): string | null {
  return _appVersion;
}
