import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  listExamples,
  listExampleTags,
  type ExampleSummary,
  type ExampleTagCount,
  type ExampleTagsResponse,
} from '../api/rest';
import { freeTagsOf, mainTagOf, mainTagOrder } from './exampleSections';

export interface MainTagEntry {
  id: string;
  count: number;
}

export interface ExampleCatalog {
  examples: ExampleSummary[];
  /** The main tags that have examples, in gallery order. */
  mainTags: MainTagEntry[];
  /** Every free tag with how many examples carry it, most used first. */
  tags: ExampleTagCount[];
  loading: boolean;
  error: string | null;
  reload: () => void;
}

/** Fetch the tag counts without letting a failure, or an older server that
 * has no such route, take the list down with it. */
async function fetchTags(): Promise<ExampleTagsResponse | null> {
  try {
    const body = await listExampleTags();
    return body && Array.isArray(body.main_tags) && Array.isArray(body.tags) ? body : null;
  } catch {
    return null;
  }
}

/**
 * The example list and its tag counts, as every example surface reads them.
 *
 * Both requests go out together. The main-tag order comes from
 * `GET /api/examples/tags` (the server defines it once); when that answer is
 * missing -- an older server, a failed request -- the order and the counts are
 * derived from the list itself with the fallback order, so the surfaces still
 * render.
 */
export function useExampleCatalog(): ExampleCatalog {
  const [examples, setExamples] = useState<ExampleSummary[]>([]);
  const [tagsBody, setTagsBody] = useState<ExampleTagsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    Promise.all([
      // Wrapped so a mocked or missing function that returns nothing is a
      // rejection, not a TypeError thrown before the promise exists.
      Promise.resolve().then(() => listExamples()),
      fetchTags(),
    ])
      .then(([all, tags]) => {
        if (cancelled) return;
        setExamples(Array.isArray(all) ? all : []);
        setTagsBody(tags);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setExamples([]);
        setTagsBody(null);
        setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => reload(), [reload]);

  const mainTags = useMemo<MainTagEntry[]>(() => {
    const counts = new Map<string, number>();
    for (const example of examples) {
      const id = mainTagOf(example);
      counts.set(id, (counts.get(id) ?? 0) + 1);
    }
    const apiOrder = tagsBody?.main_tags
      .slice()
      .sort((a, b) => a.order - b.order)
      .map((entry) => entry.id);
    return mainTagOrder(counts.keys(), apiOrder).map((id) => ({ id, count: counts.get(id) ?? 0 }));
  }, [examples, tagsBody]);

  const tags = useMemo<ExampleTagCount[]>(() => {
    if (tagsBody) return tagsBody.tags;
    const counts = new Map<string, number>();
    for (const example of examples) {
      for (const tag of freeTagsOf(example)) counts.set(tag, (counts.get(tag) ?? 0) + 1);
    }
    return [...counts.entries()]
      .map(([id, count]) => ({ id, count }))
      .sort((a, b) => b.count - a.count || a.id.localeCompare(b.id));
  }, [examples, tagsBody]);

  return { examples, mainTags, tags, loading, error, reload: () => void reload() };
}
