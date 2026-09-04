/**
 * Where the documentation site lives.
 *
 * One constant, in one file, because the alternative is the same host typed
 * into every panel that links out to it -- and a docs site that moves would
 * then move in some of them.
 */
export const DOCS_BASE = 'https://docs.codefyui.com';

/**
 * A link to one documentation page.
 *
 * `docsUrl('/usage/source-control')` and `docsUrl('usage/source-control')`
 * both answer `https://docs.codefyui.com/usage/source-control`: the leading
 * slash is the kind of detail a call site gets wrong once and nobody notices
 * until the link 404s.
 */
export function docsUrl(path: string): string {
  return `${DOCS_BASE}${path.startsWith('/') ? path : `/${path}`}`;
}
