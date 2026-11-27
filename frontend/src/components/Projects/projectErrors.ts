import { ProjectError } from '../../api/projects';
import type { TranslationKey } from '../../i18n/locales/en';

/**
 * Refusals whose sentence carries nothing but the path the user typed:
 * said in the user's language from the code, with the server's own
 * sentence kept underneath as the detail. A refusal that counts or names
 * things (`runs_active`, `path_refused`, ...) is shown as the server wrote
 * it, because only the server knows what to fill in.
 */
export const TRANSLATED_PROJECT_ERRORS = [
  'not_found', 'not_a_directory', 'not_a_project', 'invalid_manifest',
  'exists', 'invalid_url', 'browse_refused',
] as const;

type Translate = (key: TranslationKey, vars?: Record<string, string | number>) => string;

export function projectErrorText(err: unknown, t: Translate): { message: string; detail: string | null } {
  if (err instanceof ProjectError
    && (TRANSLATED_PROJECT_ERRORS as readonly string[]).includes(err.code)) {
    return { message: t(`projects.error.${err.code}` as TranslationKey), detail: err.message };
  }
  return { message: err instanceof Error ? err.message : String(err), detail: null };
}
