export interface NodeTranslation {
  /** The one-line palette summary. */
  description?: string;
  /** The longer half, shown only in the config panel and the Docs tab. */
  details?: string;
  params?: Record<string, string>;
}

export type NodeTranslations = Record<string, NodeTranslation>;
